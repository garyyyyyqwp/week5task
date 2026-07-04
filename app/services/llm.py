import asyncio
import logging

from openai import AsyncOpenAI, RateLimitError

from app.utils.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    VISION_MODEL,
    VISION_API_KEY,
    VISION_BASE_URL,
)

logger = logging.getLogger(__name__)

# Retry settings for rate-limited (429) API calls
LLM_MAX_RETRIES: int = 3
LLM_RETRY_BASE_DELAY: float = 2.0  # seconds, doubles each retry

_text_client: AsyncOpenAI | None = None
_vision_client: AsyncOpenAI | None = None


async def _retry_on_rate_limit(coro, operation: str = "LLM call"):
    """Retry an async LLM call with exponential backoff on 429 errors.

    Args:
        coro: The awaitable to execute (e.g. client.chat.completions.create(...)).
        operation: Human-readable name for logging.

    Returns:
        The result of the coroutine.

    Raises:
        The original exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            return await coro
        except RateLimitError as exc:
            last_exc = exc
            if attempt == LLM_MAX_RETRIES:
                break
            delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "%s rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                operation, attempt + 1, LLM_MAX_RETRIES, delay, exc,
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


def get_client() -> AsyncOpenAI:
    """Get the text LLM client (e.g., glm-4-flash)."""
    global _text_client
    if _text_client is None:
        _text_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _text_client


def get_vision_client() -> AsyncOpenAI:
    """Get the Vision LLM client (e.g., glm-4v)."""
    global _vision_client
    if _vision_client is None:
        _vision_client = AsyncOpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    return _vision_client


def get_model() -> str:
    return OPENAI_MODEL


def get_vision_model() -> str:
    return VISION_MODEL
