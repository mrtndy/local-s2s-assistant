import os, sys, re, time, json, queue, threading, tempfile, shutil, subprocess
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests, webrtcvad
from faster_whisper import WhisperModel

# --- Config ---
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
WHISPER_MODEL= os.getenv("WHISPER_MODEL", "large-v3")

# VAD Settings (Tuned for accuracy)
SAMPLE_RATE      = 16000
VAD_FRAME_MS     = 30 
VAD_AGGRESSIVENESS = 1     # 0=Most Permissive, 3=Most Aggressive. 1 is a good balance.
SILENCE_LIMIT_MS = 1000    # Wait 1s of silence before assuming user is done
MAX_REC_SEC      = 30

VOICE_ONNX = Path("voices/en_US-amy-low.onnx")
VOICE_CFG  = Path("voices/en_US-amy-low.onnx.json")

def log(x): print(x, flush=True)

# --- VAD Recorder ---
class VADRecorder:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.frame_len = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        self.q = queue.Queue()

    def record(self):
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=self.frame_len, callback=self._cb)
        frames = []; active = False; silence_ms = 0; t0 = time.perf_counter()
        log("\n[Recording] Speak now...")
        with stream:
            while True:
                try: chunk = self.q.get(timeout=1.0)
                except queue.Empty:
                    if time.perf_counter() - t0 > MAX_REC_SEC: break
                    continue
                
                # WebRTC VAD requires 16-bit PCM
                pcm16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
                if len(pcm16) != self.frame_len * 2: continue 

                if self.vad.is_speech(pcm16, SAMPLE_RATE):
                    if not active: log(" > Voice detected")
                    active = True; silence_ms = 0
                elif active:
                    silence_ms += VAD_FRAME_MS
                    
                if active: frames.append(chunk)
                if active and silence_ms >= SILENCE_LIMIT_MS:
                    log(" > Silence detected, finishing."); break
                if time.perf_counter() - t0 > MAX_REC_SEC:
                     log(" > Max time reached."); break

        return np.concatenate(frames) if frames else np.array([], dtype=np.float32)

    def _cb(self, indata, f, t, s): self.q.put(indata[:, 0].copy())

# --- Helpers ---
def normalize_audio(audio, target_db=-3.0):
    """Normalizes audio to a target dB level to help Whisper hear quiet mics."""
    peak = np.max(np.abs(audio))
    if peak == 0: return audio
    target_linear = 10 ** (target_db / 20)
    return audio * (target_linear / peak)

def stream_llm(text):
    """Streams reply from Ollama, yielding chunks of text."""
    prompt = f"You are a concise assistant. Reply in under 50 words.\nUser: {text}\nAssistant:"
    try:
        with requests.post(f"{OLLAMA_URL}/api/generate", stream=True, timeout=30,
                           json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}) as r:
            for line in r.iter_lines():
                if not line: continue
                j = json.loads(line)
                if j.get("done"): break
                yield j.get("response", "")
    except Exception as e: log(f"\nLLM Error: {e}")

# --- Robust TTS ---
class TTS:
    def __init__(self):
        if not VOICE_ONNX.exists(): raise FileNotFoundError(f"Missing {VOICE_ONNX}")
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def say(self, text): self.q.put(text)

    def _loop(self):
        try:
            from piper.voice import PiperVoice
            # Load voice ONCE inside the thread
            voice = PiperVoice.load(str(VOICE_ONNX), str(VOICE_CFG), use_cuda=False)
            
            stream = None
            current_sr = None

            while True:
                text = self.q.get()
                if text is None: break # Poison pill to exit

                # Synthesize and stream audio chunks
                for audio_chunk in voice.synthesize(text):
                    sr = getattr(audio_chunk, "sample_rate", 22050)
                    data = getattr(audio_chunk, "audio_int16_array", None)
                    if data is None or len(data) == 0: continue

                    # Ensure valid numpy array for sounddevice
                    data_np = np.array(data, dtype=np.int16)

                    # Open/Re-open stream if needed
                    if stream is None or sr != current_sr:
                        if stream: stream.stop(); stream.close()
                        stream = sd.RawOutputStream(samplerate=sr, channels=1, dtype='int16')
                        stream.start()
                        current_sr = sr
                    
                    stream.write(data_np.tobytes())
                
                # Small pause between sentences feels more natural
                sd.sleep(150)

        except Exception as e:
            # THIS WILL PRINT THE ACTUAL ERROR IF IT CRASHES AGAIN
            print(f"\n!!! TTS THREAD CRASHED !!!\nError: {e}\n", file=sys.stderr)

# --- Main Loop ---
def main():
    print("Loading Whisper...", end="", flush=True)
    asr = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print(" Done.\nInitializing Audio...", flush=True)
    vad = VADRecorder()
    tts = TTS()
    print("\n--- READY --- (Ctrl+C to quit)")

    while True:
        try:
            input("[Press Enter] to start recording...")
            audio = vad.record()
            
            if len(audio) < SAMPLE_RATE * 0.5:
                print("(Ignoring short audio)")
                continue

            # 1. Normalize audio before sending to Whisper
            audio = normalize_audio(audio)

            # 2. Transcribe
            segs, _ = asr.transcribe(audio, language="en", beam_size=5, 
                                     vad_filter=True, suppress_tokens=[-1])
            user_text = " ".join(s.text for s in segs).strip()
            
            print(f"\nUser: {user_text}")
            if not user_text: continue

            # 3. Stream LLM + TTS
            print("Assistant: ", end="", flush=True)
            buf = ""
            for chunk in stream_llm(user_text):
                print(chunk, end="", flush=True)
                buf += chunk
                # Split by sentence endings to stream TTS
                if re.search(r"[\.\?\!\n]", chunk):
                    # Find complete sentences in buffer
                    parts = re.split(r'([\.\?\!\n]+)', buf)
                    # Process pairs of (sentence, punctuation)
                    for i in range(0, len(parts) - 1, 2):
                        if i+1 < len(parts):
                            sentence = parts[i] + parts[i+1]
                            if sentence.strip(): tts.say(sentence.strip())
                    # Keep the incomplete remaining part in buffer
                    buf = parts[-1] if len(parts) % 2 == 1 else ""
            
            if buf.strip(): tts.say(buf.strip())
            print("\n")

        except KeyboardInterrupt:
            print("\nExiting."); break
        except Exception as e:
             print(f"\nMain Loop Error: {e}")

if __name__ == "__main__":
    main()
