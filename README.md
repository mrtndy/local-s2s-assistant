Local S2S Assistant

This is a fast, entirely local Speech-to-Speech (S2S) assistant that runs on your local machine. It uses:

VAD: Google's WebRTC VAD for real-time voice activity detection.

ASR: faster-whisper for high-accuracy speech-to-text.

LLM: Ollama (with any model, e.g., qwen2.5:3b-instruct) for generating responses.

TTS: piper-tts for fast, local text-to-speech.

The entire loop runs without cloud services, ensuring privacy and low latency.

Setup

1. Prerequisites

Ollama: You must have Ollama installed and running.

Python: Python 3.10+

NVIDIA GPU (Recommended): Required for faster-whisper to run efficiently on CUDA.

FFmpeg: (Optional, but recommended) A system-wide install of FFmpeg is often needed for audio libraries.

2. Installation

Clone this repository:

git clone [https://github.com/mrtndy/local-s2s-assistant.git](https://github.com/mrtndy/local-s2s-assistant.git)
cd local-s2s-assistant


Create a Python virtual environment (recommended):

python -m venv venv


Activate the environment:

On Windows: .\venv\Scripts\activate

On macOS/Linux: source venv/bin/activate

Install the required Python packages:

pip install -r requirements.txt


Pull the Ollama model (if you haven't already):

ollama pull qwen2.5:3b-instruct


(You can change the model in main.py if you prefer a different one)

Download the TTS voice:

Create a folder named voices.

Download the "amy-low" voice files and place them in that folder:

en_US-amy-low.onnx

en_US-amy-low.onnx.json

Your folder structure should look like this:

local-s2s-assistant/
├── voices/
│   ├── en_US-amy-low.onnx
│   └── en_US-amy-low.onnx.json
├── main.py
└── requirements.txt


Usage

Make sure Ollama is running in a separate terminal. Then, run the main script:

python main.py


The script will load the models and prompt you:

[Press Enter] to start recording...
