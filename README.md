# Local Speech to Speech Assistant

ASR: faster-whisper on CUDA  
LLM: Ollama `qwen2.5:7b-instruct`  
TTS: Piper voice (ONNX)  
Extras: silence detection and timing metrics

---

## Table of contents

1. What you will build  
2. System requirements  
3. Quick start  
4. Detailed setup  
   - 4.1 Install FFmpeg  
   - 4.2 Install and start Ollama  
   - 4.3 Pull the LLM  
   - 4.4 Install Python libraries  
   - 4.5 Add a Piper voice  
   - 4.6 Verify the environment  
   - 4.7 ASR smoke test  
   - 4.8 End to end baseline with Piper CLI  
   - 4.9 Low latency TTS in process  
   - 4.10 Silence detection and timing metrics  
5. Usage summary  
6. Tuning and performance  
7. Troubleshooting  
8. Repo layout  
9. Roadmap  
10. License

---

## 1) What you will build

A fully local speech to speech assistant.  
You speak. Whisper transcribes. Ollama replies. Piper speaks back. Recording stops on silence. The script prints per-stage timings.

**Why this setup**  
* Works fully offline  
* Fits in 8 GB VRAM with `int8_float16`  
* Components are swappable

---

## 2) System requirements

* Windows 11  
* NVIDIA RTX 4070 laptop GPU with recent driver  
* Python 3.11 to 3.13  
* PowerShell  
* Internet for first time model downloads

---

## 3) Quick start

```powershell
# Install FFmpeg
winget install --id=Gyan.FFmpeg -e

# Persist FFmpeg to your user PATH, then open a NEW terminal
$ff = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ffmpeg.exe | Select-Object -First 1
if ($ff) {
  $new = "$($ff.DirectoryName);" + [Environment]::GetEnvironmentVariable("Path","User")
  [Environment]::SetEnvironmentVariable("Path", $new, "User")
  Write-Host "Added to PATH:" $ff.DirectoryName
} else {
  Write-Host "ffmpeg.exe not found under WinGet packages."
}

# Verify in a NEW terminal
ffmpeg -version

# Install Ollama and pull model
winget install Ollama.Ollama
Stop-Service Ollama; Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Service Ollama
ollama pull qwen2.5:7b-instruct

# Python libraries (pinned in requirements.txt)
pip install -r requirements.txt

# Voice files
mkdir -Force .\voices
curl.exe -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx" -o ".\voices\en_US-amy-low.onnx"
curl.exe -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json" -o ".\voices\en_US-amy-low.onnx.json"

# Run the metrics variant
python .\scripts\assistant_local_metrics.py
```

---

## 4) Detailed setup

### 4.1 Install FFmpeg

**What**: Install `ffmpeg.exe` and add its folder to your user PATH.  
**Why**: Required by audio tools and Piper. Avoids “not recognized” errors.  
**Check**: `ffmpeg -version` prints version info.

```powershell
winget install --id=Gyan.FFmpeg -e
$ff = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ffmpeg.exe | Select-Object -First 1
if ($ff) {
  $new = "$($ff.DirectoryName);" + [Environment]::GetEnvironmentVariable("Path","User")
  [Environment]::SetEnvironmentVariable("Path", $new, "User")
  Write-Host "Added to PATH:" $ff.DirectoryName
} else {
  Write-Host "ffmpeg.exe not found under WinGet packages."
}
ffmpeg -version
```

---

### 4.2 Install and start Ollama

**What**: Install Ollama and ensure the service is running clean.  
**Why**: Provides a local HTTP endpoint at `127.0.0.1:11434` for LLMs.  
**Check**: `ollama --version` and no port conflicts.

```powershell
winget install Ollama.Ollama
Stop-Service Ollama
Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Service Ollama
ollama --version
```

---

### 4.3 Pull the LLM

**What**: Download `qwen2.5:7b-instruct` into Ollama.  
**Why**: Chat model for responses.  
**Check**: Pull completes without error. `ollama run` returns a reply.

```powershell
ollama pull qwen2.5:7b-instruct
# optional smoke test
ollama run qwen2.5:7b-instruct
```

Notes  
* `503` means transient registry issues. Retry.  
* “file does not exist” means wrong tag.

---

### 4.4 Install Python libraries

**What**: Install pinned packages.  
**Why**: Reproducible ASR and TTS on Windows.  
**Check**: `pip list` shows the expected versions.

`requirements.txt`:

```txt
# Core
numpy>=2.3,<2.4
sounddevice>=0.5.3,<0.6
PyAudio>=0.2.14,<0.3
requests>=2.32,<3

# ASR
faster-whisper==1.2.0
ctranslate2==4.6.0

# TTS
piper-tts==1.3.0

# Runtime for faster-whisper/piper wheels on Windows
onnxruntime==1.23.0
```

Install:

```powershell
pip install -r requirements.txt
```

---

### 4.5 Add a Piper voice

**What**: Download the voice model and config.  
**Why**: Piper needs both files to synthesize.  
**Check**: `.onnx` is about 63 MB. `.json` is a few KB.

```powershell
mkdir -Force .\voices
curl.exe -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx" -o ".\voices\en_US-amy-low.onnx"
curl.exe -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json" -o ".\voices\en_US-amy-low.onnx.json"
```

---

### 4.6 Verify the environment

**What**: Run the checker script.  
**Why**: Validate versions, CUDA types, Ollama reachability, voice load, audio devices.  
**Check**: Mostly `[OK]` lines in output.

```powershell
python .\scripts\check_versions.py
```

Expected  
* CUDA compute types include `float16` and `int8_float16`  
* Ollama reachable and `qwen2.5:7b-instruct` present  
* Piper voice loads

---

### 4.7 ASR smoke test

**What**: Record a short clip and transcribe.  
**Why**: Proves mic and ASR before full loop.  
**Check**: Transcript matches your speech.

```powershell
python .\scripts\record_and_transcribe.py
```

---

### 4.8 End to end baseline with Piper CLI

**What**: Full loop via Piper CLI.  
**Why**: Simple path and prosody flags.  
**Check**: You hear a spoken reply.

```powershell
python .\scripts\assistant_local.py
```

Prosody flags  
* `--length-scale 0.9` faster speech  
* `--noise-w-scale 0.6` smoother prosody  
* `--volume 1.15` louder playback

---

### 4.9 Low latency TTS in process

**What**: Full loop using Piper Python API.  
**Why**: Removes extra process and disk writes.  
**Check**: Lower TTS latency.

```powershell
python .\scripts\assistant_local_fasttts.py
```

---

### 4.10 Silence detection and timing metrics

**What**: VAD based stop and timing printouts.  
**Why**: Measure record, ASR, LLM, TTS, and end to end.  
**Check**: Metrics appear after each turn.

```powershell
python .\scripts\assistant_local_metrics.py
```

Example
```
[METRIC] record_ms=1681  captured_sec=1.44
[METRIC] asr_ms=416
[METRIC] llm_ms=4166  llm_tokens=10  llm_tok_s=56.8
[METRIC] tts_synth_ms=604  tts_play_ms=2651  tts_audio_sec=2.51  tts_rtf=0.24
[METRIC] e2e_ms=9521
```

---

## 5) Usage summary

* `scripts/check_versions.py`  
  Versions, CUDA types, FFmpeg, Ollama, voice files, audio devices

* `scripts/record_and_transcribe.py`  
  Records 4 seconds and prints a transcript

* `scripts/assistant_local.py`  
  End to end using Piper CLI with prosody flags

* `scripts/assistant_local_fasttts.py`  
  End to end using Piper Python API for lower latency

* `scripts/assistant_local_metrics.py`  
  Silence detection and timings for record, ASR, LLM, TTS synth, TTS play, end to end

---

## 6) Tuning and performance

* Recording  
  Increase `silence_ms` if it cuts early. Decrease to end faster

* ASR  
  Start with `base.en`. For higher accuracy try `small.en` or `medium.en`. Keep `int8_float16`

* LLM  
  Keep prompts short. Reduce `num_ctx` if long memory is not needed. Try `qwen2.5:3b-instruct` for speed tests

* TTS  
  Use the CLI variant to tune prosody. The Python API does not accept `length-scale` or `noise-w-scale`

---

## 7) Troubleshooting

* FFmpeg not found  
  Open a new terminal after the PATH update. Rerun the PATH script if required

* Ollama port 11434 is busy
  ```powershell
  Stop-Service Ollama
  Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force
  Start-Service Ollama
  ```

* Ollama pull returns 503  
  Retry the pull

* Voice files show 9 bytes  
  Re-download from the Hugging Face URLs above

* `webrtcvad` fails to build on Python 3.13  
  This repo uses an RMS VAD already

* Whisper accuracy drops  
  Move the mic closer. Reduce background noise. Try `small.en`

* Device selection  
  List devices:
  ```powershell
  python -m sounddevice
  ```
  Then set `sd.default.device = (in_idx, out_idx)` inside scripts if needed

---

## 8) Repo layout

```
scripts/
  check_versions.py
  record_and_transcribe.py
  assistant_local.py
  assistant_local_fasttts.py
  assistant_local_metrics.py
voices/
  .gitkeep
README.md
requirements.txt
```

Voices are user supplied. `voices/*` is ignored. `.gitkeep` keeps the folder.

---

## 9) Roadmap

* API mode using OpenAI or Gemini  
* Optional VAD with WebRTC after installing Build Tools  
* Wake word and push to talk  
* Simple UI

---

## 10) License

MIT
