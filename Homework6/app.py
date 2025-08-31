import os
import json
from typing import Dict, Any, Callable
from fastapi import FastAPI
from pydantic import BaseModel
from tools import search_arxiv, calculate
from prompts import SYSTEM_PROMPT

# ====== Configure how to call Llama 3 ======
# MODE options:
#   "mock"   : no model calls; returns deterministic JSON/text responses for testing
#   "ollama" : uses local Ollama server (http://localhost:11434) with model "llama3"
#   "openai" : uses OpenAI-compatible endpoint (e.g., OpenAI or local server). Requires OPENAI_API_KEY.
MODE = os.getenv("LLM_MODE", "mock")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instruct")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ====== Tool registry ======
TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "search_arxiv": search_arxiv,
    "calculate": calculate,
}

# ====== FastAPI app ======
app = FastAPI(title="Week 6 Voice Agent with Function Calling")

class VoiceQuery(BaseModel):
    text: str

def _mock_llm(user_text: str) -> str:
    """
    Deterministic responses so you can test end-to-end without an actual LLM.
    """
    t = user_text.lower().strip()
    if "2+2" in t or "two plus two" in t or "calculate" in t:
        return json.dumps({"function":"calculate","arguments":{"expression":"2+2"}})
    if "arxiv" in t or "quantum" in t or "transformer" in t:
        return json.dumps({"function":"search_arxiv","arguments":{"query":"quantum entanglement"}})
    return "I am a normal response. How can I help you today?"

def llama3_chat_model(user_text: str) -> str:
    """
    Returns either a JSON function call or normal text.
    This is where you integrate your actual Llama 3 call.
    """
    if MODE == "mock":
        return _mock_llm(user_text)

    # Compose messages
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": user_text},
    ]

    if MODE == "ollama":
        # Minimal Ollama chat call
        try:
            import requests
            payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
            r = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Ollama-compatible: data["choices"][0]["message"]["content"]
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Ollama: {e}"

    if MODE == "openai":
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": 0.2,
            }
            r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling OpenAI-compatible endpoint: {e}"

    return "I am a normal response (fallback)."

def route_llm_output(llm_output: str) -> Dict[str, Any]:
    """
    Tries to parse llm_output as a function call; if valid, executes the tool.
    Returns a dict with structured logs for the deliverables.
    """
    log: Dict[str, Any] = {
        "raw_llm_output": llm_output,
        "is_function_call": False,
        "called_function": None,
        "function_args": None,
        "tool_output": None,
        "final_response": None,
        "error": None,
    }
    try:
        obj = json.loads(llm_output)
        func = obj.get("function")
        args = obj.get("arguments", {})
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Not a function call; treat as normal text
        log["final_response"] = llm_output
        return log

    # If we reached here, it parsed as JSON
    log["is_function_call"] = True
    log["called_function"] = func
    log["function_args"] = args
    tool = TOOL_REGISTRY.get(func)
    if not tool:
        log["error"] = f"Unknown function '{func}'"
        log["final_response"] = f"Error: Unknown function '{func}'"
        return log

    try:
        result = tool(**args)
        log["tool_output"] = result
        log["final_response"] = result
    except TypeError as e:
        log["error"] = f"Bad arguments for '{func}': {e}"
        log["final_response"] = f"Error: Bad arguments for '{func}'"
    except Exception as e:
        log["error"] = f"Tool execution failed for '{func}': {e}"
        log["final_response"] = f"Error: Tool '{func}' failed"
    return log

@app.post("/api/voice-query/")
async def voice_query_endpoint(req: VoiceQuery):
    user_text = req.text
    llm_response = llama3_chat_model(user_text)
    log = route_llm_output(llm_response)
    # Here you'd pass log["final_response"] to your TTS step before returning
    return {"logs": log, "response": log["final_response"]}
