# scripts/check_versions.py
# Reports installed versions, validates against requirements.txt targets,
# and probes GPU/FFmpeg/Ollama/Piper voice availability.

import os, sys, json, shutil, subprocess, socket
from pathlib import Path

# -------- expected versions (from requirements.txt) --------
EXPECTED = {
    # pip name        : (import_name, specifier)
    "numpy"           : ("numpy", ">=2.3,<2.4"),
    "sounddevice"     : ("sounddevice", ">=0.5.3,<0.6"),
    "PyAudio"         : ("pyaudio", ">=0.2.14,<0.3"),
    "requests"        : ("requests", ">=2.32,<3"),
    "faster-whisper"  : ("faster_whisper", "==1.2.0"),
    "ctranslate2"     : ("ctranslate2", "==4.6.0"),
    "piper-tts"       : ("piper", "==1.3.0"),
    "onnxruntime"     : ("onnxruntime", "==1.23.0"),
}
VOICE_ONNX = Path("voices/en_US-amy-low.onnx")
VOICE_JSON = Path("voices/en_US-amy-low.onnx.json")
OLLAMA_URL = "http://127.0.0.1:11434"

# -------- helpers --------
def ok(s):   return f"[OK]   {s}"
def warn(s): return f"[WARN] {s}"
def fail(s): return f"[FAIL] {s}"

def check_pkg(pip_name, import_name, spec):
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
    except Exception as e:
        print(fail(f"{pip_name}: not importable ({e})"))
        return

    try:
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
        ss = SpecifierSet(spec)
        verdict = ver != "unknown" and Version(ver) in ss
    except Exception:
        verdict = False  # packaging missing or unparsable

    if verdict:
        print(ok(f"{pip_name}=={ver} matches {spec}"))
    else:
        print(warn(f"{pip_name}=={ver} does not match {spec}"))

def check_ctranslate2_capabilities():
    try:
        import ctranslate2 as c
        cpu = c.get_supported_compute_types("cpu")
        cuda = c.get_supported_compute_types("cuda")
        print(ok(f"ctranslate2 compute types cpu={sorted(cpu)} cuda={sorted(cuda)}"))
        needed = "int8_float16"
        if needed not in cuda:
            print(warn(f"cuda missing {needed} (performance may drop)"))
    except Exception as e:
        print(fail(f"ctranslate2 capability probe failed: {e}"))

def check_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        print(warn("ffmpeg not on PATH"))
        return
    try:
        out = subprocess.checkoutput([exe, "-version"], stderr=subprocess.STDOUT, text=True)
    except AttributeError:
        out = subprocess.check_output([exe, "-version"], stderr=subprocess.STDOUT, text=True)
    first = out.splitlines()[0].strip() if out else "ffmpeg present"
    print(ok(first))

def port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def check_ollama():
    if not port_open("127.0.0.1", 11434):
        print(warn("Ollama service not reachable on 127.0.0.1:11434"))
        return
    try:
        import requests
        v = requests.get(f"{OLLAMA_URL}/api/version", timeout=1).json()
        print(ok(f"Ollama API version: {v.get('version','unknown')}"))
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()
        names = [m.get("name","") for m in tags.get("models", [])]
        target = "qwen2.5:7b-instruct"
        if any(n.startswith(target) for n in names):
            print(ok(f"Model present: {target}"))
        else:
            print(warn(f"Model not found: {target} (run: ollama pull {target})"))
    except Exception as e:
        print(warn(f"Ollama API probe failed: {e}"))

def check_piper_voice():
    missing = []
    if not VOICE_ONNX.exists(): missing.append(str(VOICE_ONNX))
    if not VOICE_JSON.exists(): missing.append(str(VOICE_JSON))
    if missing:
        print(warn("Piper voice files missing: " + ", ".join(missing)))
        return
    try:
        sz = VOICE_ONNX.stat().st_size
        print(ok(f"Piper voice ONNX present ({sz/1_000_000:.1f} MB)"))
        print(ok("Piper voice JSON present"))
        from piper.voice import PiperVoice
        _ = PiperVoice.load(model_path=str(VOICE_ONNX), config_path=str(VOICE_JSON), use_cuda=False)
        print(ok("Piper voice load probe succeeded (CPU)"))
    except Exception as e:
        print(warn(f"Piper voice load probe failed: {e}"))

def check_audio_devices():
    try:
        import sounddevice as sd
        default_in, default_out = sd.default.device
        print(ok(f"sounddevice default in={default_in} out={default_out}"))
        devs = sd.query_devices()
        print(ok(f"{len(devs)} audio devices detected"))
    except Exception as e:
        print(warn(f"sounddevice probe failed: {e}"))

def main():
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {os.name} | cwd: {Path.cwd()}")

    print("\n== packages ==")
    for pip_name, (import_name, spec) in EXPECTED.items():
        check_pkg(pip_name, import_name, spec)

    print("\n== ctranslate2 capabilities ==")
    check_ctranslate2_capabilities()

    print("\n== ffmpeg ==")
    check_ffmpeg()

    print("\n== ollama ==")
    check_ollama()

    print("\n== piper voice files ==")
    check_piper_voice()

    print("\n== audio devices ==")
    check_audio_devices()

    print("\nDone.")

if __name__ == "__main__":
    main()
