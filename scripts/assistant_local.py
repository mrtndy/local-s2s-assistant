import sys
import wave
import subprocess
import requests
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ------------ config ------------
SAMPLE_RATE = 16000
RECORD_SEC  = 2

ASR_MODEL   = "base.en"           # switch to "small.en" or "medium.en" if you want better accuracy
ASR_DEVICE  = "cuda"
ASR_COMPUTE = "int8_float16"

OLLAMA_URL  = "http://127.0.0.1:11434"
LLM_MODEL   = "qwen2.5:7b-instruct"

VOICE_ONNX  = r".\voices\en_US-amy-low.onnx"
VOICE_JSON  = r".\voices\en_US-amy-low.onnx.json"
OUT_WAV     = "reply.wav"
# --------------------------------

def record(sec=RECORD_SEC, rate=SAMPLE_RATE):
    print(f"[REC] Recording {sec}s… Speak now.")
    audio = sd.rec(int(sec * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    return (audio.astype(np.float32).reshape(-1) / 32768.0)

def asr(samples):
    model = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
    segments, info = model.transcribe(samples, vad_filter=True, language="en")
    text = " ".join(s.text for s in segments).strip()
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

def tts_cli(text, out_wav=OUT_WAV):
    # Piper CLI allows prosody controls; adjust if desired
    cmd = [
        "piper",
        "-m", VOICE_ONNX,
        "-c", VOICE_JSON,
        "--length-scale", "0.9",
        "--noise-w-scale", "0.6",
        "--volume", "1.15",
        "-f", out_wav,
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)
    with wave.open(out_wav, "rb") as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        sd.play(data, wf.getframerate())
        sd.wait()

def main():
    samples = record()
    if samples.size == 0:
        print("[ERR] No audio captured.")
        sys.exit(1)
    prompt = asr(samples)
    if not prompt:
        print("[ERR] Empty transcript.")
        sys.exit(1)
    reply = llm(prompt)
    if reply:
        tts_cli(reply)
        print("[OK] done")

if __name__ == "__main__":
    main()
