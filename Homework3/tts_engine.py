
import os
import wave
import tempfile
import threading
from typing import Optional
from config import settings, WAV_PARAMS

_piper_voice = None
_piper_lock = threading.Lock()

def _load_piper():
    print("[DEBUG] Loading Piper model:", settings.piper_model_path)

    global _piper_voice
    if _piper_voice is None:
        from piper import PiperVoice  
        model_path = settings.piper_model_path
        config_path = settings.piper_model_json 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found at {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Piper config not found at {config_path}")
        _piper_voice = PiperVoice.load(model_path, config_path=config_path)  
    return _piper_voice


def synthesize_speech(text: str, outfile: Optional[str] = None) -> str:
    print("[DEBUG] synthesize_speech called with text:", text[:80])

    voice = _load_piper()

    if outfile is None:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    else:
        wav_path = outfile

    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)

    with _piper_lock:
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(WAV_PARAMS["setnchannels"])
            wav_file.setframerate(WAV_PARAMS["setframerate"])
            wav_file.setsampwidth(WAV_PARAMS["setsamplewidth"])
            voice.synthesize_wav(text, wav_file)

    size = os.path.getsize(wav_path)
    print(f"[TTS] Wrote {size} bytes to {wav_path}")
    print("[DEBUG] Wrote WAV size:", os.path.getsize(wav_path))

    if size < 1000:
        raise RuntimeError(f"Piper TTS wrote a tiny file ({size} bytes).")
    return wav_path
