import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION_SEC = 4

def record(sec=DURATION_SEC, rate=SAMPLE_RATE):
    print(f"Recording {sec} seconds. Speak now.")
    audio = sd.rec(int(sec * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    return (audio.astype(np.float32).reshape(-1) / 32768.0)

def transcribe(samples):
    model = WhisperModel("base.en", device="cuda", compute_type="int8_float16")
    segments, info = model.transcribe(samples, vad_filter=True, language="en")
    text = " ".join(s.text for s in segments).strip()
    print(f"Language: {info.language}")
    print("Transcript:", text)

if __name__ == "__main__":
    samples = record()
    if samples.size == 0:
        print("No audio captured")
    else:
        transcribe(samples)
