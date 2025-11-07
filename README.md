# Local S2S Assistant

This is a fast, entirely local Speech-to-Speech (S2S) assistant that runs on your local machine. It uses:

* **VAD**: Google's WebRTC VAD for real-time voice activity detection.
* **ASR**: `faster-whisper` for high-accuracy speech-to-text.
* **LLM**: `Ollama` (with any model, e.g., `qwen2.5:3b-instruct`) for generating responses.
* **TTS**: `piper-tts` for fast, local text-to-speech.

The entire loop runs without cloud services, ensuring privacy and low latency.

## Setup

### 1. Prerequisites

* **Ollama**: You must have [Ollama](https://ollama.com/) installed and running.
* **Python**: Python 3.10+
* **NVIDIA GPU**: Required for `faster-whisper` to run efficiently on CUDA.
* **FFmpeg**: A system-wide install of FFmpeg is needed for audio libraries.

### 2. Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/mrtndy/local-s2s-assistant.git](https://github.com/mrtndy/local-s2s-assistant.git)
    cd local-s2s-assistant
    ```

2.  Create a Python virtual environment (recommended):
    ```bash
    python -m venv venv
    ```

3.  Activate the environment:
    * On Windows:
        ```powershell
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

5.  Pull the Ollama model (if you haven't already):
    ```bash
    ollama pull qwen2.5:3b-instruct
    ```
    *(You can change the model in `main.py` if you prefer a different one)*

6.  Download the TTS voice:
    * Create a folder named `voices`.
    * Download the "amy-low" voice files and place them in that folder:
        * [en_US-amy-low.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx?download=1)
        * [en_US-amy-low.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json?download=1)

    Your folder structure should look like this:
    ```text
    local-s2s-assistant/
    ├── voices/
    │   ├── en_US-amy-low.onnx
    │   └── en_US-amy-low.onnx.json
    ├── main.py
    └── requirements.txt
    ```

## Usage

1.  Make sure Ollama is running in a separate terminal.
2.  Run the main script:

    ```bash
    python main.py
    ```

3.  The script will load the models and prompt you:
    ```text
    Loading Whisper ASR model... Done.
    Initializing Audio systems (VAD & TTS)...

    --- READY --- (Ctrl+C to quit)

    [Press Enter] to start recording...
    ```
