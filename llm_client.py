"""Client for llama.cpp llama-server (OpenAI-compatible chat completions)."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

INFERENCE_URL = os.getenv("INFERENCE_URL", "").rstrip("/")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "120"))
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a concise voice assistant. The user spoke aloud; respond briefly "
    "and helpfully to what they said.",
)


def inference_configured() -> bool:
    return bool(INFERENCE_URL)


async def infer_assistant_response(transcript: str) -> str:
    """Send transcript to llama-server and return the assistant reply."""
    if not INFERENCE_URL:
        return "ERROR: INFERENCE_URL is not configured"

    url = f"{INFERENCE_URL}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SEC) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return "ERROR: empty response from inference server"
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return "ERROR: unexpected response format from inference server"
    except httpx.HTTPStatusError as e:
        logger.error("Inference HTTP error: %s %s", e.response.status_code, e.response.text)
        return f"ERROR: inference server returned {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error("Inference request failed: %s", e)
        return f"ERROR: cannot reach inference server ({e})"
    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        return f"ERROR: {e}"
