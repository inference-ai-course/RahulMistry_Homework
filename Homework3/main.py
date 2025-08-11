import asyncio
import uuid
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from memory import memory
from asr import transcribe_audio, load_asr
from llm import generate_response, load_llm
from tts_engine import synthesize_speech
from fastapi import FastAPI

app = FastAPI(title="Local Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _warmup():

    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(None, load_asr),
        loop.run_in_executor(None, load_llm),
    )

@app.get("/health")
async def health():
    return {"status": "ok", "whisper": settings.whisper_model, "llm": settings.llm_model}

@app.post("/chat/")
async def chat_endpoint(
    file: UploadFile = File(...),
    session_id: str = Query(default=None, description="Provide a stable session_id to keep memory")
):
    """
    Upload an audio file (wav/m4a/mp3), get an audio/wav reply.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    audio_bytes = await file.read()

    loop = asyncio.get_running_loop()


    user_text = await loop.run_in_executor(None, transcribe_audio, audio_bytes)
    if not user_text:
        return JSONResponse({"error": "Empty transcription", "session_id": session_id}, status_code=400)

    memory.add(session_id, "user", user_text)
    recent = memory.get_recent(session_id)

    bot_text = await loop.run_in_executor(None, generate_response, recent)
    if not bot_text:
        bot_text = "Sorry, I couldn't think of a good answer."

    memory.add(session_id, "assistant", bot_text)

    audio_path = await loop.run_in_executor(None, synthesize_speech, bot_text, None)

    headers = {"X-Session-ID": session_id}
    return FileResponse(audio_path, media_type="audio/wav", headers=headers)

@app.post("/reset")
async def reset_session(session_id: str):
    memory.clear(session_id)
    return {"ok": True, "session_id": session_id}

@app.post("/chat_text/")
async def chat_text(file: UploadFile = File(...), session_id: str | None = None):
    if not session_id:
        import uuid; session_id = str(uuid.uuid4())
    audio_bytes = await file.read()
    loop = asyncio.get_running_loop()
    user_text = await loop.run_in_executor(None, transcribe_audio, audio_bytes)
    memory.add(session_id, "user", user_text)
    recent = memory.get_recent(session_id)
    bot_text = await loop.run_in_executor(None, generate_response, recent)
    memory.add(session_id, "assistant", bot_text)
    return {"session_id": session_id, "user": user_text, "assistant": bot_text[:500]}
