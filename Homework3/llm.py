import torch
from transformers import pipeline
from typing import List, Dict
from config import settings

_llm_pipe = None

def load_llm():
    global _llm_pipe
    if _llm_pipe is None:
        _llm_pipe = pipeline(
            "text-generation",
            model=settings.llm_model,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
    return _llm_pipe

def format_chat_prompt(history: List[Dict[str, str]]) -> str:
    """
    Simple chat-style prompt:
    user: ...
    assistant: ...
    """
    lines = []
    for turn in history:
        lines.append(f"{turn['role']}: {turn['text']}")
    lines.append("assistant:") 
    return "\n".join(lines)

def generate_response(recent_turns: List[Dict[str, str]]) -> str:
    llm = load_llm()
    prompt = format_chat_prompt(recent_turns)
    out = llm(
        prompt,
        max_new_tokens=settings.llm_max_new_tokens,
        temperature=settings.llm_temperature,
        do_sample=True,
        pad_token_id=llm.tokenizer.eos_token_id if hasattr(llm, "tokenizer") else None,
    )
    text = out[0]["generated_text"]
    completion = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

    for tag in ("assistant:", "Assistant:", "ASSISTANT:"):
        if completion.startswith(tag):
            completion = completion[len(tag):].strip()
    return completion
