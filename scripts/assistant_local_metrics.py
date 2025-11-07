import time
import sys
import requests
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# ---------- config ----------
SAMPLE_RATE = 16000

ASR_MODEL   = "base.en"       # bump to "small.en" or "medium.en" later if desired
ASR_DEVICE  = "cuda"
ASR_COMPUTE = "int8_float16"

OLLAMA_URL  = "http://127.0.0.1:11434"
LLM_MODEL   = "qwen2.5:7b-instruct"
NUM_CTX     = 2048
TEMP        = 0.3

VOICE_ONNX  = r".\voices\en_US-amy-low.onnx"
VOICE_JSON  = r".\voices\en_US-amy-low.onnx.json"
# ----------------------------

# ----- timing helpers -----
T0 = time.perf_counter()

def now_ts():
    t = time.time()
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return time.strftime("%H:%M:%S", lt) + f".{ms:03d}"

def since_start_ms():
    return int(round((time.perf_counter() - T0) * 1000))

def log(msg):
    print(f"[{now_ts()} +{since_start_ms()}ms] {msg}")

def ms(seconds: float) -> int:
    return int(round(seconds * 1000))
# --------------------------

def record_until_silence(rate=SAMPLE_RATE, frame_ms=20, warmup_ms=120, silence_ms=700, max_ms=7000):
    """RMS-based VAD: stop ~silence_ms after speech ends."""
    frame_len = int(rate * frame_ms / 1000)
    warmup_frames = max(1, warmup_ms // frame_ms)
    silence_frames = max(1, silence_ms // frame_ms)
    max_frames = max(1, max_ms // frame_ms)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    buf = []
    noise_window = []
    silence_count = 0
    speech_started = False
    threshold = None

    log("Listening… speak, then pause to end.")
    with sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=frame_len) as stream:
        for _ in range(max_frames):
            frame, _ = stream.read(frame_len)
            f = frame[:, 0]

            if len(noise_window) < warmup_frames:
                noise_window.append(rms(f))
                buf.append(f.copy())
                continue

            if threshold is None:
                noise = np.median(noise_window)
                threshold = max(noise * 4.0, 0.008)  # floor near -42 dBFS

            level = rms(f)
            buf.append(f.copy())

            if level > threshold:
                speech_started = True
                silence_count = 0
            else:
                if speech_started:
                    silence_count += 1
                    if silence_count >= silence_frames:
                        break

    return np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)

# ----- load ASR / TTS once -----
log("[ASR] loading faster-whisper…")
t0 = time.perf_counter()
WHISPER = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
log(f"[METRIC] asr_load_ms={ms(time.perf_counter() - t0)}")

log("[TTS] loading Piper voice…")
t0 = time.perf_counter()
VOICE = PiperVoice.load(model_path=VOICE_ONNX, config_path=VOICE_JSON, use_cuda=False)
log(f"[METRIC] tts_load_ms={ms(time.perf_counter() - t0)}")

def transcribe(samples: np.ndarray) -> str:
    t0 = time.perf_counter()
    segs, info = WHISPER.transcribe(samples, vad_filter=True, language="en")
    text = " ".join(s.text for s in segs).strip()
    t1 = time.perf_counter()
    log(f"[ASR] lang={info.language}  text={text}")
    log(f"[METRIC] asr_ms={ms(t1 - t0)}")
    return text

def chat_ollama(prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": f"You are a brief voice assistant.\nUser: {prompt}\nAssistant:",
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "temperature": TEMP},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    t1 = time.perf_counter()
    r.raise_for_status()
    j = r.json()
    reply = j.get("response", "").strip()
    eval_count = j.get("eval_count")
    eval_dur_ns = j.get("eval_duration")
    log(f"[LLM] {reply}")
    if isinstance(eval_count, int) and isinstance(eval_dur_ns, int) and eval_dur_ns > 0:
        tok_s = eval_count / (eval_dur_ns / 1e9)
        log(f"[METRIC] llm_ms={ms(t1 - t0)}  llm_tokens={eval_count}  llm_tok_s={tok_s:.1f}")
    else:
        log(f"[METRIC] llm_ms={ms(t1 - t0)}")
    return reply

def tts_piper(text: str):
    # Synthesize
    t0 = time.perf_counter()
    parts = []
    sr = 22050
    for ch in VOICE.synthesize(text):
        try:
            arr = np.asarray(ch.audio_int16_array, dtype=np.int16).reshape(-1)
            if arr.size:
                parts.append(arr)
            if hasattr(ch, "sample_rate"):
                sr = int(ch.sample_rate)
        except Exception as e:
            log(f"[TTS] warn bad chunk: {e}")
            continue
    t1 = time.perf_counter()
    synth_ms = ms(t1 - t0)

    if not parts:
        log("[TTS] no audio")
        return

    audio = np.concatenate(parts)
    aud_sec = len(audio) / float(sr)

    # Play
    t2 = time.perf_counter()
    sd.play(audio, sr)
    sd.wait()
    t3 = time.perf_counter()
    play_ms = ms(t3 - t2)

    rtf = (synth_ms / 1000.0) / max(aud_sec, 1e-6)
    log(f"[METRIC] tts_synth_ms={synth_ms}  tts_play_ms={play_ms}  tts_audio_sec={aud_sec:.2f}  tts_rtf={rtf:.2f}")

def main():
    e0 = time.perf_counter()

    # Record
    r0 = time.perf_counter()
    samples = record_until_silence()
    r1 = time.perf_counter()
    dur_sec = samples.size / SAMPLE_RATE
    log(f"[METRIC] record_ms={ms(r1 - r0)}  captured_sec={dur_sec:.2f}")

    if samples.size == 0:
        log("[ASR] empty capture. Try again.")
        sys.exit(0)

    # ASR
    text = transcribe(samples)
    if not text:
        log("[ASR] empty transcript. Try again.")
        sys.exit(0)

    # LLM
    reply = chat_ollama(text)

    # TTS
    tts_piper(reply)

    log(f"[METRIC] e2e_ms={ms(time.perf_counter() - e0)}")
    log("[S2S] done")

if __name__ == "__main__":
    main()
