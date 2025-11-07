"""
main.py
A local Speech-to-Speech (S2S) assistant.

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
import queue            # Used for VAD <-> main thread communication
import threading        # Used to run TTS in the background
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
    """
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_len = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        self.q = queue.Queue()

    def record(self) -> np.ndarray:
        """
        Listens for speech and records it until silence is detected.
        """
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.frame_len,
            callback=self._audio_callback
        )
        
        frames = []
        active = False
        silence_ms = 0
        t0 = time.perf_counter()

        log("\n[Recording] Speak now...")
        
        with stream:
            while True:
                try:
                    chunk = self.q.get(timeout=1.0)
                except queue.Empty:
                    if time.perf_counter() - t0 > MAX_REC_SEC:
                        log(" > Max time reached (timeout).")
                        break
                    continue
                
                # VAD requires 16-bit PCM audio
                pcm16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
                
                if len(pcm16) != self.frame_len * 2:
                    continue 

                is_speech = self.vad.is_speech(pcm16, SAMPLE_RATE)

                if is_speech:
                    if not active:
                        log(" > Voice detected, recording...")
                        active = True
                    silence_ms = 0
                elif active:
                    silence_ms += VAD_FRAME_MS
                
                if active:
                    frames.append(chunk)

                if active and silence_ms >= SILENCE_LIMIT_MS:
                    log(f" > Silence detected ({SILENCE_LIMIT_MS}ms), finishing.")
                    break
                
                if time.perf_counter() - t0 > MAX_REC_SEC:
                    log(" > Max time reached (recording limit).")
                    break
        
        return np.concatenate(frames) if frames else np.array([], dtype=np.float32)

    def _audio_callback(self, indata, frames, time, status):
        self.q.put(indata[:, 0].copy())

# --- Audio Processing Helper ---
def normalize_audio(audio: np.ndarray, target_db=-3.0) -> np.ndarray:
    """Normalizes the audio peak to a target dB level."""
    peak = np.max(np.abs(audio))
    if peak == 0: return audio
    target_linear = 10 ** (target_db / 20)
    gain = target_linear / peak
    return audio * gain

# --- LLM Streaming Function ---
def stream_llm(text: str):
    """Streams a reply from the Ollama API, yielding text chunks."""
    prompt = f"You are a concise assistant. Reply in under 50 words.\nUser: {text}\nAssistant:"
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            stream=True,
            timeout=30,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line: continue
                try:
                    j = json.loads(line)
                    if j.get("done"): break
                    yield j.get("response", "")
                except json.JSONDecodeError: continue
    except Exception as e:
        log(f"\n[Error] LLM stream failed: {e}")

# --- Robust TTS Class ---
class TTS:
    """Handles Text-to-Speech in a separate, non-blocking thread."""
    def __init__(self):
        if not VOICE_ONNX.exists():
            raise FileNotFoundError(f"TTS model not found: {VOICE_ONNX}")
        if not VOICE_CFG.exists():
            raise FileNotFoundError(f"TTS config not found: {VOICE_CFG}")
            
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.thread.start()

    def say(self, text: str):
        self.q.put(text)

    def _tts_loop(self):
        try:
            voice = PiperVoice.load(str(VOICE_ONNX), str(VOICE_CFG), use_cuda=False)
            stream = None
            current_sr = None

            while True:
                text = self.q.get()
                if text is None: break
                
                for audio_chunk in voice.synthesize(text):
                    sr = getattr(audio_chunk, "sample_rate", 22050)
                    data = getattr(audio_chunk, "audio_int16_array", None)
                    if data is None or len(data) == 0: continue

                    data_np = np.array(data, dtype=np.int16)

                    if stream is None or sr != current_sr:
                        if stream:
                            stream.stop(); stream.close()
                        stream = sd.RawOutputStream(samplerate=sr, channels=1, dtype='int16')
                        stream.start()
                        current_sr = sr
                    
                    stream.write(data_np.tobytes())
                
                sd.sleep(150) # Natural pause between sentences

        except Exception as e:
            print(f"\n!!! TTS THREAD CRASHED !!!\nError: {e}\n", file=sys.stderr)

# --- Main Application Loop ---
def main():
    log("Loading Whisper ASR model...")
    asr = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    
    log("Initializing Audio systems (VAD & TTS)...")
    try:
        vad = VADRecorder()
        tts = TTS()
    except Exception as e:
        log(f"\n[Fatal Error] Could not initialize audio: {e}")
        sys.exit(1)

    log("\n--- READY --- (Ctrl+C to quit)")

    while True:
        try:
            input("[Press Enter] to start recording...")
            
            # 1. Record
            audio = vad.record()
            if len(audio) < SAMPLE_RATE * 0.5:
                print("(Ignoring short audio clip)")
                continue

            # 2. Normalize & Transcribe
            audio = normalize_audio(audio)
            segs, _ = asr.transcribe(audio, language="en", beam_size=5, vad_filter=True, suppress_tokens=[-1])
            user_text = " ".join(s.text for s in segs).strip()
            
            log(f"\nUser: {user_text}")
            if not user_text: continue

            # 4. Get Response (LLM + TTS)
            log("Assistant: ")
            sentence_buffer = ""
            
            for chunk in stream_llm(user_text):
                print(chunk, end="", flush=True)
                sentence_buffer += chunk
                
                # Split only on punctuation followed by whitespace
                if re.search(r'(?<=[.?!])\s+', sentence_buffer):
                    parts = re.split(r'(?<=[.?!])\s+', sentence_buffer)
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            tts.say(sentence.strip())
                    sentence_buffer = parts[-1]
            
            if sentence_buffer.strip():
                tts.say(sentence_buffer.strip())
            print("\n")

        except KeyboardInterrupt:
            log("\nExiting."); break
        except Exception as e:
            log(f"\n[Error] Loop error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
