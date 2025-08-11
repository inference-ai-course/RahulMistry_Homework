import tempfile
import whisper
from config import settings

_asr_model = None

def load_asr():
    global _asr_model
    if _asr_model is None:
        _asr_model = whisper.load_model(settings.whisper_model)
    return _asr_model

def transcribe_audio(audio_bytes: bytes) -> str:
    model = load_asr()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = model.transcribe(tmp.name)
    return result.get("text", "").strip()
