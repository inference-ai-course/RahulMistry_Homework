# config.py
import os
from pydantic import BaseModel


class Settings(BaseModel):

    whisper_model: str = os.getenv("WHISPER_MODEL", "small")  


    llm_model: str = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "160"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))


    max_turns: int = int(os.getenv("MAX_TURNS", "5"))


    tts_voice: str = os.getenv("TTS_VOICE", "")       
    tts_rate: int = int(os.getenv("TTS_RATE", "185")) 


    piper_model_path: str = os.getenv(
        "PIPER_MODEL_PATH",
        "piper_models/en_US-amy-medium/en_US-amy-medium.onnx",
    )
    piper_model_json: str = os.getenv(
        "PIPER_MODEL_JSON",
        "piper_models/en_US-amy-medium/en_US-amy-medium.onnx.json",
    )

    wav_channels: int = int(os.getenv("WAV_CHANNELS", "1"))       
    wav_rate: int = int(os.getenv("WAV_RATE", "22050"))           
    wav_sample_width: int = int(os.getenv("WAV_SAMPLE_WIDTH", "2"))  


settings = Settings()


WAV_PARAMS = {
    "setnchannels": settings.wav_channels,
    "setframerate": settings.wav_rate,
    "setsamplewidth": settings.wav_sample_width,
}
