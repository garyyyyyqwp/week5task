"""TTS (Text-to-Speech) service.

Supports two providers:
  - openai:   OpenAI TTS API (tts-1 / tts-1-hd) via /v1/audio/speech
  - zhipu:    Zhipu CogTTS via BigModel API

Returns audio bytes that the frontend plays via the HTML5 Audio API.
"""

import io
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.utils.config import (
    TTS_PROVIDER,
    TTS_MODEL,
    TTS_VOICE,
    TTS_API_KEY,
    TTS_BASE_URL,
)

logger = logging.getLogger(__name__)

# TTS input limits
MAX_TTS_CHARS = 4096

# OpenAI voice names the frontend/older clients might send. When the provider
# is Zhipu these are invalid ("音色不存在"), so we map any of them onto a valid
# Zhipu voice rather than passing them through and failing.
_OPENAI_VOICES = {
    "alloy", "echo", "fable", "nova", "onyx", "shimmer", "ash", "coral", "sage",
}
_ZHIPU_DEFAULT_VOICE = "tongtong"

# Zhipu cogtts rejects mp3/opus (error 1214) and returns headerless PCM by
# default — which browsers can't decode. It DOES accept "wav", so we force
# wav for Zhipu and expose the matching MIME type. OpenAI keeps mp3.
TTS_OUTPUT_FORMAT = "wav" if TTS_PROVIDER == "zhipu" else "mp3"
TTS_MEDIA_TYPE = "audio/wav" if TTS_PROVIDER == "zhipu" else "audio/mpeg"


class TTSError(Exception):
    """Raised when TTS synthesis fails."""


def _translate_error(err: Exception) -> str:
    """Turn a raw provider error into a user-facing Chinese message.

    Recognizes common BigModel/OpenAI error codes so the UI can show the real
    cause (e.g. quota exhausted) instead of a generic failure or a misleading
    timeout from a doomed fallback.
    """
    s = str(err)
    if "1113" in s or "余额" in s or "资源包" in s or "insufficient_quota" in s:
        return "语音额度不足，请为账户充值后重试。"
    if "1214" in s:
        return "语音服务不支持当前请求参数。"
    if "1261" in s or "音色" in s:
        return "所选音色不可用，请更换音色。"
    if "429" in s or "rate limit" in s.lower():
        return "语音服务请求过于频繁，请稍后重试。"
    if "timed out" in s.lower() or "timeout" in s.lower():
        return "语音服务响应超时，请稍后重试。"
    return f"语音合成失败: {s}"


def _resolve_voice(voice: str | None) -> str:
    """Resolve the effective voice name for the active provider."""
    selected = voice or TTS_VOICE
    if TTS_PROVIDER == "zhipu" and selected in _OPENAI_VOICES:
        # Client asked for an OpenAI voice but we're on Zhipu — remap.
        return _ZHIPU_DEFAULT_VOICE
    return selected


def _build_tts_kwargs(text: str, voice: str, speed: float) -> dict:
    """Assemble create() kwargs with a provider-appropriate output format.

    Each provider only accepts certain response_format values (Zhipu: wav;
    OpenAI: mp3 etc.), so we always send TTS_OUTPUT_FORMAT rather than an
    arbitrary caller-supplied one that might be rejected (error 1214).
    """
    return {
        "model": TTS_MODEL,
        "input": text,
        "voice": voice,
        "speed": speed,
        "response_format": TTS_OUTPUT_FORMAT,
    }


def _validate_tts_input(text: str) -> None:
    """Validate TTS input text.

    Args:
        text: Text to synthesize.

    Raises:
        ValueError: If text is empty or too long.
    """
    if not text or not text.strip():
        raise ValueError("TTS输入文本不能为空")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(
            f"TTS输入文本过长 ({len(text)}字符)。最大允许: {MAX_TTS_CHARS}字符"
        )


def _get_tts_client() -> AsyncOpenAI:
    """Get the TTS-specific AsyncOpenAI client."""
    return AsyncOpenAI(api_key=TTS_API_KEY, base_url=TTS_BASE_URL)


async def synthesize(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    response_format: str = "mp3",
) -> bytes:
    """Convert text to speech audio bytes.

    Args:
        text: Text to synthesize (max 4096 chars).
        voice: Voice name (defaults to TTS_VOICE from config).
        speed: Playback speed 0.25-4.0.
        response_format: Audio format: mp3, opus, aac, flac, wav, pcm.

    Returns:
        Raw audio bytes.

    Raises:
        ValueError: If text validation fails.
        TTSError: If the TTS API call fails.
    """
    _validate_tts_input(text)

    voice = _resolve_voice(voice)
    client = _get_tts_client()

    try:
        response = await client.audio.speech.create(
            **_build_tts_kwargs(text, voice, speed)
        )
        return response.content
    except Exception as e:
        # No cross-provider fallback: the previous code retried with the same
        # (Zhipu) key against the OpenAI endpoint, which can only time out —
        # slow AND it masked the real error. Fail fast with the real cause.
        logger.error("TTS synthesis failed: %s", e)
        raise TTSError(_translate_error(e)) from e


async def synthesize_streaming(text: str, voice: str | None = None) -> bytes:
    """Synthesize speech and return the full audio bytes (buffered).

    Kept for callers that need the complete blob (e.g. the voice loop, which
    base64-encodes the whole clip). For low-latency playback prefer
    synthesize_stream(), which yields chunks as they arrive.
    """
    _validate_tts_input(text)

    voice = _resolve_voice(voice)
    client = _get_tts_client()

    try:
        async with client.audio.speech.with_streaming_response.create(
            **_build_tts_kwargs(text, voice, 1.0)
        ) as response:
            return await response.read()
    except Exception as e:
        logger.error("TTS streaming synthesis failed: %s", e)
        raise TTSError(_translate_error(e)) from e


async def synthesize_stream(
    text: str,
    voice: str | None = None,
    speed: float = 1.0,
    response_format: str = "mp3",
) -> "AsyncIterator[bytes]":
    """Yield audio bytes as they are produced, for low first-byte latency.

    Streams chunks directly from the provider so the browser can start
    playing before the whole clip is synthesized. Raises TTSError (with a
    translated message) if the request fails — the caller decides how to
    surface it. No silent empty output.

    Args:
        text: Text to synthesize (max 4096 chars).
        voice: Voice name (defaults to TTS_VOICE from config).
        speed: Playback speed 0.25-4.0.
        response_format: Audio format (ignored for Zhipu).

    Yields:
        Audio byte chunks.

    Raises:
        ValueError: If text validation fails.
        TTSError: If synthesis fails.
    """
    _validate_tts_input(text)

    voice = _resolve_voice(voice)
    client = _get_tts_client()

    try:
        async with client.audio.speech.with_streaming_response.create(
            **_build_tts_kwargs(text, voice, speed)
        ) as response:
            async for chunk in response.iter_bytes():
                if chunk:
                    yield chunk
    except Exception as e:
        logger.error("TTS stream failed: %s", e)
        raise TTSError(_translate_error(e)) from e
