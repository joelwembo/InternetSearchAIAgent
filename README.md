# Multi-model FastAPI LLM server with LangChain

This repository includes a single-file FastAPI application (`main.py`) that exposes a `/generate` endpoint and lets clients choose among multiple LLM providers using the **LangChain** framework.

## Key Features

- **LangChain Integration**: All providers now use LangChain for consistent API interaction
- **Gemini as Default**: Google's Gemini is configured as the primary/default LLM provider
- **Automatic Fallback**: If Gemini (commercial provider) fails, the system automatically falls back to **Ollama llama3.2** (local/free provider)
- **Multiple Providers**: Supports OpenAI, Anthropic, Gemini, Grok, and custom providers

## Quick start

1. Install dependencies:

    python -m pip install -r requirements.txt

2. Set API keys as environment variables for the providers you plan to use:

    - `GOOGLE_API_KEY` (for Gemini - default provider)
    - `OPENAI_API_KEY` (optional)
    - `ANTHROPIC_API_KEY` (optional)
    - `GROK_API_KEY` and optionally `GROK_API_URL` (optional)
    - `OLLAMA_BASE_URL` (optional, defaults to http://localhost:11434)

3. **For Ollama fallback**: Install and run Ollama locally with llama3.2 model:
   
   ```bash
   # Install Ollama from https://ollama.ai
   # Then pull the llama3.2 model
   ollama pull llama3.2
   ```

4. Run the server:

    python main.py

or with an ASGI server:

    uvicorn main:app --host 0.0.0.0 --port 8000

## Endpoints

- `GET /health`  
  Returns status, supported providers, and configuration information including LangChain status.

- `POST /generate`  
  JSON body fields:
  - `provider`: one of `"openai"`, `"anthropic"`, `"gemini"`, `"grok"`, `"custom"`
  - `prompt`: the prompt string
  - `model`: optional model name
  - `max_tokens`: optional (default 512)
  - `temperature`: optional (default 0.0)
  - `extra`: optional dict for provider-specific fields

## Example curl commands

1) Gemini (default, with automatic Ollama fallback):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 200
  }'
```

2) OpenAI (assumes `OPENAI_API_KEY` is set):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "prompt": "Summarize the plot of Hamlet in 3 bullets.",
    "model": "gpt-4o-mini",
    "max_tokens": 200
  }'
```

3) Anthropic (assumes `ANTHROPIC_API_KEY` is set):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "prompt": "Explain Newton'\''s second law in one paragraph.",
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 200
  }'
```

4) Custom provider (call any HTTP LLM endpoint):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "custom",
    "prompt": "Write a haiku about autumn.",
    "extra": {
      "url": "https://example-llm-host.local/v1/generate",
      "method": "POST",
      "headers": {"Authorization": "Bearer TOKEN", "Content-Type": "application/json"},
      "body_template": {"input": "{prompt}", "model": "{model}", "max_tokens": "{max_tokens}"}
    }
  }'
```

5) Health check:

```bash
curl http://localhost:8000/health
```

## Gemini + Ollama Fallback Behavior

When you use the `gemini` provider:

1. **Primary**: The system first attempts to use Google's Gemini API (commercial provider)
2. **Fallback**: If Gemini fails (no API key, rate limit, network error, etc.), it automatically falls back to Ollama llama3.2 (local provider)
3. **Error Handling**: If both providers fail, a comprehensive error message is returned

This ensures high availability and allows you to continue using the API even if commercial providers are unavailable.

## Notes

- The server uses LangChain for all provider implementations, providing a consistent interface
- Gemini is the default provider with automatic Ollama llama3.2 fallback for reliability
- Keep API keys secure — do not commit them to the repo
- The fallback mechanism only applies to the Gemini provider
- Other providers (OpenAI, Anthropic) will return errors if their API keys are missing or invalid

LICENSE: add or adjust repository license as desired.
