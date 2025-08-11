# Project3Inference

This repository contains code and audio files for a speech-to-speech chatbot pipeline using:

Link to the video: https://drive.google.com/file/d/1dJcdAoxDNNiODV7VF-67NyIPjQMQCl_K/view?usp=sharing

- **Whisper ASR** — Converts spoken audio into text.
- **LLM** — Generates context-aware responses.
- **Piper TTS** — Converts the generated text back into speech.

---

##  Workflow

1. Record or provide an input `.wav` file.
2. Run it through the `/chat` endpoint — Whisper transcribes the audio.
3. The LLM processes the transcribed text and generates a reply.
4. Piper TTS synthesizes the reply into a `.wav` output.

---

##  Files

### Input Audio
- `sample.wav`, `sample2.wav`, `sample3.wav`, `sample4.wav`, `sample5.wav`: Example input audio files.

### Generated Audio
- `reply.wav`, `reply2.wav`, `reply3.wav`, `reply4.wav`, `reply5.wav`: TTS-generated replies from the system.

### Source Code
- `asr.py`: Handles automatic speech recognition.
- `llm.py`: Manages LLM inference and conversation flow.
- `tts_engine.py`: Uses Piper TTS to convert text to speech.
- `config.py`: Stores configuration for model paths, parameters, and settings.
- `main.py`: FastAPI app serving the `/chat` endpoint.
- `memory.py`: Keeps track of conversation context.
- `test.py`: Simple test script for local runs.
- `requirements.txt`: List of dependencies.

---

##  Usage

### 1. Run the server
```bash
uvicorn main:app --reload
