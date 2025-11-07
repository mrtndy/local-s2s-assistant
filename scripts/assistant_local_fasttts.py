import sys
import requests
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# ---------- config ----------
SAMPLE_RATE = 16000
RECORD_SEC  = 2

ASR_MODEL   = "base.en"
ASR_DEVICE  = "cuda"
ASR_COMPUTE = "int8_float16"

OLLAMA_URL  = "http://127.0.0.1:11434"
LLM_MODEL   = "qwen2.5:7b-instruct"

VOICE_ONNX  = r".\voices\en_US-amy-low.onnx"
VOICE_JSON  = r".\voices\en_US-amy-low.onnx.json"
# ----------------------------

def record(sec=RECORD_SEC, rate=SAMPLE_RATE):
    print(f"[REC] Recording {sec}s… Speak now.")
    audio = sd.rec(int(sec * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    return (audio.astype(np.float32).reshape(-1) / 32768.0)

def asr(samples):
    model = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
    segs, info = model.transcribe(samples, vad_filter=True, language="en")
    text = " ".join(s.text for s in segs).strip()
    print(f"[ASR] lang={info.language} text={text}")
    return text

def llm(user_text):
    payload = {
        "model": LLM_MODEL,
        "prompt": f"You are a brief voice assistant.\nUser: {user_text}\nAssistant:",
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.3},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    reply = r.json().get("response", "").strip()
    print(f"[LLM] {reply}")
    return reply

# Keep the Piper model in memory between calls
_VOICE = None
def tts(text):
    global _VOICE
    if _VOICE is None:
        _VOICE = PiperVoice.load(model_path=VOICE_ONNX, config_path=VOICE_JSON, use_cuda=False)
    parts = []
    sr = 22050
    for ch in _VOICE.synthesize(text):
        # Piper 1.3.0 exposes int16 PCM as audio_int16_array and the sample rate per chunk
        arr = np.asarray(ch.audio_int16_array, dtype=np.int16).reshape(-1)
        if arr.size:
            parts.append(arr)
        if hasattr(ch, "sample_rate"):
            sr = int(ch.sample_rate)
    if not parts:
        print("[TTS] no audio produced")
        return
    audio = np.concatenate(parts)
    sd.play(audio, sr)
    sd.wait()

def main():
    samples = record()
    if samples.size == 0:
        print("[ERR] no audio captured"); sys.exit(1)
    text = asr(samples)
    if not text:
        print("[ERR] empty transcript"); sys.exit(1)
    reply = llm(text)
    if reply:
        tts(reply)
        print("[OK] done")

if __name__ == "__main__":
    main()
