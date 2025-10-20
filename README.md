# Multi-model FastAPI LLM server

This repository includes a single-file FastAPI application (`main.py`) that exposes a `/generate` endpoint and lets clients choose among multiple LLM providers (openai, anthropic, gemini, grok, custom).

## Quick start

1. Install dependencies:

    python -m pip install -r requirements.txt

2. Set API keys as environment variables for the providers you plan to use:

    - `OPENAI_API_KEY`
    - `ANTHROPIC_API_KEY`
    - `GOOGLE_API_KEY` (for Gemini via google-generativeai)
    - `GROK_API_KEY` and optionally `GROK_API_URL`

3. Run the server:

    python main.py

or with an ASGI server:

    uvicorn main:app --host 0.0.0.0 --port 8000

## Endpoints

- `GET /health`  
  Returns status and supported providers.

- `POST /generate`  
  JSON body fields:
  - `provider`: one of `"openai"`, `"anthropic"`, `"gemini"`, `"grok"`, `"custom"`
  - `prompt`: the prompt string
  - `model`: optional model name
  - `max_tokens`: optional (default 512)
  - `temperature`: optional (default 0.0)
  - `extra`: optional dict for provider-specific fields

## Example curl commands

1) OpenAI (assumes `OPENAI_API_KEY` is set):

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

2) Anthropic (assumes `ANTHROPIC_API_KEY` is set):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "prompt": "Explain Newton'\''s second law in one paragraph.",
    "model": "claude-2.1",
    "max_tokens": 200
  }'
```

3) Custom provider (call any HTTP LLM endpoint):

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

4) Health check:

```bash
curl http://localhost:8000/health
```

## Notes

- The server is intentionally minimal and uses simple REST adapters. Adjust provider implementations to suit official SDKs or to match your provider's exact API payloads and auth schemes.
- Keep API keys secure — do not commit them to the repo.

LICENSE: add or adjust repository license as desired.
