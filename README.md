# Local Speech to Speech Assistant

ASR: faster-whisper on CUDA  
LLM: Ollama `qwen2.5:7b-instruct`  
TTS: Piper voice (ONNX)  
Extras: silence detection and detailed timing metrics

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
You speak. The system transcribes with Whisper, gets a reply from a local LLM through Ollama, then speaks the answer with Piper. Recording stops on silence. The script prints per stage timings so you can see where latency lives.

**Why this setup**  
- Works fully offline.  
- Fits in 8 GB VRAM using `int8_float16` compute type.  
- Modular scripts let you swap components later.

---

## 2) System requirements

- Windows 11  
- NVIDIA RTX 4070 laptop GPU with recent driver  
- Python 3.11 to 3.13  
- PowerShell  
- Internet for first time model downloads

---

## 3) Quick start

```powershell
# Install FFmpeg
winget install --id=Gyan.FFmpeg -e
$ff = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ffmpeg.exe | Select-Object -First 1
$env:Path = "$($ff.DirectoryName);$env:Path"

# Install Ollama and pull model
winget install Ollama.Ollama
Stop-Service Ollama; Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Service Ollama
ollama pull qwen2.5:7b-instruct

# Python libraries
pip install sounddevice numpy pyaudio faster-whisper piper-tts requests

# Voice files
mkdir -Force .\voices
curl.exe -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx" -o ".\voices\en_US-amy-low.onnx"
curl.exe -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json" -o ".\voices\en_US-amy-low.onnx.json"

# Run a script
python .\scripts\assistant_local_metrics.py
