# Week 6 — Voice Agent with Function Calling

## What's inside
- `app.py` — FastAPI server, LLM call helpers, tool registry, and routing logic.
- `prompts.py` — System prompt that instructs Llama 3 to output JSON when a tool should be called.
- `tools.py` — Implements `calculate` (Sympy) and `search_arxiv` (placeholder).
- `tests/demo_script.py` — Runs 3 sample queries and prints the required logs.
- `requirements.txt` — Python dependencies.

## Quickstart
1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Choose how you want to run the LLM:
   - **Mock mode (default):** No real model. Great for end-to-end testing.
     ```bash
     export LLM_MODE=mock
     ```
   - **Ollama mode:** Run an Ollama Llama 3 model locally, then:
     ```bash
     export LLM_MODE=ollama
     export OLLAMA_URL=http://localhost:11434
     export OLLAMA_MODEL=llama3
     ```
   - **OpenAI-compatible mode:** Use an OpenAI-style API endpoint:
     ```bash
     export LLM_MODE=openai
     export OPENAI_API_KEY=sk-...        # required
     export OPENAI_BASE_URL=https://api.openai.com/v1
     export OPENAI_MODEL=llama-3.1-8b-instruct
     ```

4. Run the API:
   ```bash
   uvicorn app:app --reload --port 8000
   ```
   POST text queries to `http://localhost:8000/api/voice-query/` with JSON body:
   ```json
   {"text":"Hey, can you calculate 2+2?"}
   ```

5. Generate test logs (for deliverables):
   ```bash
   python -m tests.demo_script
   ```
   This will print:
   - User’s query text
   - Raw LLM response (JSON or text)
   - Any function call made and its output
   - Final assistant response

## Integrating with Week 3
- After calling `route_llm_output`, send `log["final_response"]` to your TTS component.
- Keep your existing ASR → LLM → TTS flow. Replace your LLM step with:
  1) Call `llama3_chat_model(user_text)`
  2) Pass result to `route_llm_output`
  3) Speak `log["final_response"]`
