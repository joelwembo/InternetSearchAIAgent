from typing import Optional, Dict, Any
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Multi-Model LLM API")

# Request model
class GenerateRequest(BaseModel):
    provider: str = Field(..., description="one of: openai, anthropic, gemini, grok, custom")
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    extra: Optional[Dict[str, Any]] = None  # provider-specific overrides


# Simple helper for JSON responses
class GenerateResponse(BaseModel):
    provider: str
    model: str
    text: str
    meta: Optional[Dict[str, Any]] = None


# --- Provider implementations (minimal, REST-based where possible) ---

async def generate_openai(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    model = model or "gpt-4o-mini"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra:
        payload.update(extra)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}") from e
        data = r.json()
        # Follow ChatCompletions response shape
        text = ""
        if "choices" in data and len(data["choices"]) > 0:
            ch = data["choices"][0]
            if "message" in ch:
                text = ch["message"].get("content", "")
            else:
                text = ch.get("text", "")
        else:
            text = data.get("error", {}).get("message", "") or str(data)
        return {"text": text, "model": model, "raw": data}


async def generate_anthropic(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    model = model or "claude-2.1"
    url = "https://api.anthropic.com/v1/complete"
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    # Anthropic expects a prompt including human/assistant tokens; keep it simple:
    anthropic_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
    payload = {
        "model": model,
        "prompt": anthropic_prompt,
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
    }
    if extra:
        payload.update(extra)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Anthropic error: {r.text}") from e
        data = r.json()
        text = data.get("completion", "")
        return {"text": text, "model": model, "raw": data}


async def generate_gemini(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    """
    Optional helper using google generative python package. If that package isn't installed,
    explain how to enable it or use the custom provider.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="google.generativeai not installed or not configured. "
                   "Install google-generativeai and set GOOGLE_API_KEY, or use provider='custom'."
        ) from e

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY not set for Gemini")

    genai.configure(api_key=key)
    model = model or "models/text-bison-001"
    # The package may accept different parameters; this is a simple call that works with typical installs.
    resp = genai.generate_text(model=model, prompt=prompt, temperature=temperature, max_output_tokens=max_tokens)
    # resp may have .text or .candidates
    text = getattr(resp, "text", None) or (resp.candidates[0].output if getattr(resp, "candidates", None) else str(resp))
    return {"text": text, "model": model, "raw": resp}


async def generate_grok(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    """
    Grok: generic HTTP call. Configure GROK_API_KEY and optionally GROK_API_URL.
    By default this function posts JSON {"input": prompt, "model": model, ...}.
    Adjust headers/body according to the real Grok API you use.
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GROK_API_KEY not set")

    url = os.getenv("GROK_API_URL", "https://api.grok.ai/v1/generate")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": prompt, "model": model or "grok-1", "temperature": temperature, "max_tokens": max_tokens}
    if extra:
        payload.update(extra)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Grok error: {r.text}") from e
        data = r.json()
        # Try common shapes
        text = data.get("text") or data.get("output") or (data.get("choices", [{}])[0].get("text") if data.get("choices") else str(data))
        return {"text": text, "model": model or "grok-1", "raw": data}


async def generate_custom(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    """
    Call a custom HTTP endpoint provided via extra: {"url": "...", "method": "POST", "headers": {...}, "body_template": {...}}
    If body_template exists, it will be JSON-encoded and rendered with keys prompt/model/temperature/max_tokens.
    """
    if not extra or "url" not in extra:
        raise HTTPException(status_code=400, detail="custom provider requires extra.url")
    url = extra["url"]
    method = (extra.get("method") or "POST").upper()
    headers = extra.get("headers", {"Content-Type": "application/json"})
    # Basic templating
    body_template = extra.get("body_template") or {"prompt": "{prompt}", "model": "{model}", "temperature": "{temperature}", "max_tokens": "{max_tokens}"}
    # Render template
    def render(obj):
        if isinstance(obj, str):
            return obj.format(prompt=prompt, model=model or "", temperature=temperature, max_tokens=max_tokens)
        if isinstance(obj, dict):
            return {k: render(v) for k, v in obj.items()}
        return obj

    body = render(body_template)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(method, url, json=body, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Custom provider error: {r.text}") from e
        # Try to parse JSON
        try:
            data = r.json()
        except Exception:
            return {"text": r.text, "model": model or "custom", "raw": {"status_code": r.status_code, "text": r.text}}
        text = data.get("text") or data.get("output") or (data.get("choices", [{}])[0].get("text") if data.get("choices") else str(data))
        return {"text": text, "model": model or "custom", "raw": data}


# Router
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    provider = req.provider.lower()
    handlers = {
        "openai": generate_openai,
        "anthropic": generate_anthropic,
        "gemini": generate_gemini,
        "grok": generate_grok,
        "custom": generate_custom,
    }
    if provider not in handlers:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider}'. Valid: {list(handlers.keys())}")

    handler = handlers[provider]
    try:
        out = await handler(req.prompt, req.model, req.max_tokens or 512, req.temperature or 0.0, req.extra)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Provider error: {e}") from e

    return GenerateResponse(provider=provider, model=out.get("model", req.model or ""), text=out.get("text", ""), meta={"raw": out.get("raw")})


# Health
@app.get("/health")
def health():
    return {"status": "ok", "providers": ["openai", "anthropic", "gemini (optional)", "grok (requires GROK_API_URL/GROK_API_KEY)", "custom"]}


# Run with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
