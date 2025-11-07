"""
main.py
A local Speech-to-Speech (S2S) assistant.

This script captures audio from a microphone, uses Voice Activity Detection (VAD)
to determine when the user finishes speaking, transcribes the audio to text (ASR),
sends the text to a local LLM (Ollama), and streams the response back as
synthesized speech (TTS).

Architecture:
1. VADRecorder: Listens for audio and records when speech is detected.
2. ASR (Whisper): Transcribes the recorded audio file to text.
3. LLM (Ollama): Receives the text and streams a response.
4. TTS (Piper): Receives text sentences, synthesizes them to audio, and
   plays them back in a separate thread.
"""

import os
import sys
import re
import time
import json
import queue           # Used for VAD <-> main thread communication
import threading       # Used to run TTS in the background
from pathlib import Path

# --- Core Dependencies ---
import numpy as np
import sounddevice as sd        # For audio recording and playback
import requests                 # For making HTTP requests to Ollama
import webrtcvad                # For Voice Activity Detection
from faster_whisper import WhisperModel # For Speech-to-Text (ASR)
try:
    from piper.voice import PiperVoice  # For Text-to-Speech (TTS)
except ImportError:
    print("Piper TTS not found. Please install it: pip install piper-tts")
    sys.exit(1)


# --- Configuration ---
# These values are loaded from environment variables if they exist,
# otherwise, they fall back to the provided defaults.

# Ollama Configuration
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct") # Model to use for generation

# ASR (Whisper) Configuration
WHISPER_MODEL= os.getenv("WHISPER_MODEL", "large-v3") # Whisper model size

# VAD & Audio Configuration
SAMPLE_RATE      = 16000 # 16kHz. Required by WebRTC VAD
VAD_FRAME_MS     = 30    # 30ms frame duration. Required by WebRTC VAD.
VAD_AGGRESSIVENESS = 1   # VAD aggressiveness (0=least, 3=most). 1 is a good balance.
SILENCE_LIMIT_MS = 1000  # 1 second of silence triggers end-of-speech.
MAX_REC_SEC      = 30    # Maximum recording duration.

# TTS (Piper) Configuration
VOICE_ONNX = Path("voices/en_US-amy-low.onnx")       # Path to the .onnx model file
VOICE_CFG  = Path("voices/en_US-amy-low.onnx.json")  # Path to the .json config file

# --- Utility Function ---
def log(message: str):
    """Prints a message to the console with flushing enabled."""
    print(message, flush=True)

# --- VAD Recorder Class ---
class VADRecorder:
    """
    Handles microphone recording using Voice Activity Detection (VAD).

    This class continuously listens to the microphone. It only starts
    recording audio frames when it detects speech and stops when it
    detects a specified duration of silence.
    """
    def __init__(self):
        # Initialize the WebRTC VAD component with the set aggressiveness
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        # Calculate the number of audio samples per VAD frame
        self.frame_len = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        
        # A thread-safe queue to pass audio chunks from the audio callback
        # (running in a separate thread) to the main `record` loop.
        self.q = queue.Queue()

    def record(self) -> np.ndarray:
        """
        Listens for speech and records it until silence is detected.

        Returns:
            np.ndarray: A NumPy array of the recorded audio (as float32).
                        Returns an empty array if no speech was detected.
        """
        # Open the microphone input stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.frame_len,  # Read audio in VAD-sized chunks
            callback=self._audio_callback
        )
        
        frames = []           # Buffer to store audio frames when speech is active
        active = False        # Flag to indicate if we are currently recording speech
        silence_ms = 0        # Counter for accumulated silence
        t0 = time.perf_counter() # Start time for max recording timeout

        log("\n[Recording] Speak now...")
        
        with stream:
            while True:
                try:
                    # Get the latest audio chunk from the queue
                    chunk = self.q.get(timeout=1.0)
                except queue.Empty:
                    # If the queue is empty, check for max recording timeout
                    if time.perf_counter() - t0 > MAX_REC_SEC:
                        log(" > Max time reached (timeout).")
                        break
                    continue
                
                # --- VAD Logic ---
                
                # VAD requires 16-bit PCM audio, not 32-bit float.
                # Convert the float chunk to int16 bytes.
                pcm16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
                
                # Ensure the chunk is the exact size VAD expects
                if len(pcm16) != self.frame_len * 2: # 2 bytes per int16 sample
                    continue 

                # Ask VAD if this chunk contains speech
                is_speech = self.vad.is_speech(pcm16, SAMPLE_RATE)

                if is_speech:
                    if not active:
                        # We just detected the start of speech
                        log(" > Voice detected, recording...")
                        active = True
                    # Reset silence counter
                    silence_ms = 0
                elif active:
                    # We are in a silent patch *after* speech has started
                    silence_ms += VAD_FRAME_MS
                
                if active:
                    # Store the chunk (only if speech has started)
                    frames.append(chunk)

                # --- Stop Conditions ---
                
                # 1. Silence limit reached after speech
                if active and silence_ms >= SILENCE_LIMIT_MS:
                    log(f" > Silence detected ({SILENCE_LIMIT_MS}ms), finishing.")
                    break
                
                # 2. Max recording time exceeded
                if time.perf_counter() - t0 > MAX_REC_SEC:
                    log(" > Max time reached (recording limit).")
                    break
        
        # Concatenate all recorded frames into a single NumPy array
        return np.concatenate(frames) if frames else np.array([], dtype=np.float32)

    def _audio_callback(self, indata, frames, time, status):
        """
        This function is called by `sounddevice` in a separate thread
        for each new chunk of audio from the microphone.
        """
        # Put the audio data (1st channel) into the queue
        self.q.put(indata[:, 0].copy())

# --- Audio Processing Helper ---
def normalize_audio(audio: np.ndarray, target_db=-3.0) -> np.ndarray:
    """
    Normalizes the audio peak to a target dB level.
    This helps Whisper transcribe quiet microphone inputs more accurately.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:  # Avoid division by zero for silent audio
        return audio
    
    # Calculate the gain required to reach the target_db
    target_linear = 10 ** (target_db / 20)
    gain = target_linear / peak
    
    return audio * gain

# --- LLM Streaming Function ---
def stream_llm(text: str):
    """
    Streams a reply from the Ollama API, yielding text chunks.

    Args:
        text: The user's transcribed text.

    Yields:
        str: Chunks of the LLM's response.
    """
    # Simple prompt structure
    prompt = f"You are a concise assistant. Reply in under 50 words.\nUser: {text}\nAssistant:"
    
    try:
        # Start a POST request to the Ollama generate endpoint
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            stream=True,  # Enable streaming responses
            timeout=30,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True
            }
        ) as r:
            r.raise_for_status() # Raise an error for bad responses (4xx, 5xx)
            
            # Iterate over the streaming response line by line
            for line in r.iter_lines():
                if not line:
                    continue
                
                try:
                    # Each line is a JSON object
                    j = json.loads(line)
                    
                    # Stop if Ollama signals it's done
                    if j.get("done"):
                        break
                    
                    # Yield the text chunk
                    yield j.get("response", "")
                
                except json.JSONDecodeError:
                    continue # Ignore invalid JSON lines

    except requests.exceptions.RequestException as e:
        log(f"\n[Error] LLM request failed: {e}")
    except Exception as e:
        log(f"\n[Error] LLM stream failed: {e}")


# --- Robust TTS Class ---
class TTS:
    """
    Handles Text-to-Speech in a separate, non-blocking thread.
    
    This class manages a queue of text sentences. A worker thread
    consumes this queue, synthesizes audio with Piper, and plays
    it back without blocking the main application loop.
    """
    def __init__(self):
        # Check for voice files before starting
        if not VOICE_ONNX.exists():
            raise FileNotFoundError(f"TTS model not found: {VOICE_ONNX}")
        if not VOICE_CFG.exists():
            raise FileNotFoundError(f"TTS config not found: {VOICE_CFG}")
            
        self.q = queue.Queue()
        # Start the worker thread. `daemon=True` means it will
        # exit automatically when the main program exits.
        self.thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.thread.start()

    def say(self, text: str):
        """Public method to add a sentence to the TTS queue."""
        self.q.put(text)

    def _tts_loop(self):
        """
        The private worker loop that runs in a separate thread.
        
        Initializes Piper and continuously processes the text queue.
        """
        try:
            # 1. Load the Piper voice model *inside* the thread
            # This avoids potential issues with multi-threading and a loaded model
            voice = PiperVoice.load(str(VOICE_ONNX), str(VOICE_CFG), use_cuda=False)
            
            stream = None       # The audio output stream
            current_sr = None   # The sample rate of the current stream

            # 2. Start the consumer loop
            while True:
                # Wait for text to appear in the queue
                text = self.q.get()
                
                if text is None: # A "None" is a signal to exit the thread
                    break
                
                # 3. Synthesize audio
                # `synthesize` is a generator, yielding audio chunks
                for audio_chunk in voice.synthesize(text):
                    
                    # Get chunk metadata
                    sr = getattr(audio_chunk, "sample_rate", 22050)
                    data = getattr(audio_chunk, "audio_int16_array", None)
                    
                    if data is None or len(data) == 0:
                        continue

                    # Ensure data is a valid NumPy array for sounddevice
                    data_np = np.array(data, dtype=np.int16)

                    # 4. Play the audio chunk
                    
                    # If the audio stream isn't open or the sample rate
                    # has changed, (re)open it.
                    if stream is None or sr != current_sr:
                        if stream:
                            stream.stop()
                            stream.close()
                        
                        stream = sd.RawOutputStream(
                            samplerate=sr,
                            channels=1,
                            dtype='int16'
                        )
                        stream.start()
                        current_sr = sr
                    
                    # Write the audio bytes to the sound device
                    stream.write(data_np.tobytes())
                
                # Add a small, natural pause between sentences
                sd.sleep(150) # 150ms pause

        except Exception as e:
            # If the thread crashes, print the error
            print(f"\n!!! TTS THREAD CRASHED !!!\nError: {e}\n", file=sys.stderr)

# --- Main Application Loop ---
def main():
    """
    The main entry point for the S2S assistant.
    """
    # --- Initialization ---
    log("Loading Whisper ASR model...")
    asr = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    
    log("Initializing Audio systems (VAD & TTS)...")
    try:
        vad = VADRecorder()
        tts = TTS()
    except FileNotFoundError as e:
        log(f"\n[Fatal Error] Could not initialize TTS. Is the 'voices' folder correct?")
        log(f"{e}")
        sys.exit(1)
    except Exception as e:
        log(f"\n[Fatal Error] Could not initialize audio: {e}")
        sys.exit(1)

    log("\n--- READY --- (Ctrl+C to quit)")

    # --- Main Loop ---
    while True:
        try:
            # Wait for user to press Enter to start
            input("[Press Enter] to start recording...")
            
            # 1. Record Audio
            # This blocks until VAD detects silence or timeout
            audio = vad.record()
            
            # If audio is too short, ignore it
            if len(audio) < SAMPLE_RATE * 0.5: # 0.5 seconds
                print("(Ignoring short audio clip)")
                continue

            # 2. Normalize Audio
            # Boost quiet audio to improve transcription accuracy
            audio = normalize_audio(audio)

            # 3. Transcribe (ASR)
            # Run the audio through Whisper
            segs, _ = asr.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,    # Let Whisper do a final VAD pass
                suppress_tokens=[-1] # Suppress special tokens
            )
            # Join all transcribed segments into a single string
            user_text = " ".join(s.text for s in segs).strip()
            
            log(f"\nUser: {user_text}")
            if not user_text:
                continue # Skip if transcription is empty

            # 4. Get Response (LLM + TTS)
            log("Assistant: ")
            
            # This buffer holds incomplete sentences
            sentence_buffer = ""
            
            # Stream the LLM response chunk by chunk
            for chunk in stream_llm(user_text):
                print(chunk, end="", flush=True) # Print chunk to console
                sentence_buffer += chunk
                
                # --- Sentence-based Streaming ---
                # Check if the buffer contains a sentence-ending punctuation
                if re.search(r"[\.\?\!\n]", chunk):
                    
                    # Split the buffer into (sentence, punctuation) pairs
                    # e.g., "Hello. How are you?" -> ["Hello", ".", " How are you", "?"]
                    parts = re.split(r'([\.\?\!\n]+)', sentence_buffer)
                    
                    # Process all *complete* sentences
                    for i in range(0, len(parts) - 1, 2):
                        if i+1 < len(parts):
                            sentence = parts[i] + parts[i+1] # e.g., "Hello" + "."
                            if sentence.strip():
                                tts.say(sentence.strip())
                    
                    # The last part is either an incomplete sentence or empty
                    sentence_buffer = parts[-1] if len(parts) % 2 == 1 else ""
            
            # If there's any text left in the buffer, send it to TTS
            if sentence_buffer.strip():
                tts.say(sentence_buffer.strip())
            
            print("\n") # Add a newline after the assistant's full response

        except KeyboardInterrupt:
            log("\nExiting."); break
        except Exception as e:
            log(f"\n[Error] An error occurred in the main loop: {e}")
            time.sleep(1) # Prevent rapid-fire error loops

if __name__ == "__main__":
    main()
