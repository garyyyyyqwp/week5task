"""TTS (Text-to-Speech) service.

Supports two providers:
  - openai:   OpenAI TTS API (tts-1 / tts-1-hd) via /v1/audio/speech
  - zhipu:    Zhipu CogTTS via BigModel API

Returns audio bytes that the frontend plays via the HTML5 Audio API.
"""

import io
import logging

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


class TTSError(Exception):
    """Raised when TTS synthesis fails."""


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

    selected_voice = voice or TTS_VOICE
    client = _get_tts_client()

    try:
        if TTS_PROVIDER == "zhipu":
            # Zhipu CogTTS endpoint
            response = await client.audio.speech.create(
                model="cogtts",
                input=text,
                voice=selected_voice,
                speed=speed,
                response_format=response_format,
            )
        else:
            # OpenAI TTS
            response = await client.audio.speech.create(
                model=TTS_MODEL,
                input=text,
                voice=selected_voice,
                speed=speed,
                response_format=response_format,
            )

        return response.content

    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)

        # Fallback: if primary provider fails, try the other
        if TTS_PROVIDER == "zhipu":
            logger.info("Falling back to OpenAI TTS...")
            try:
                fallback_client = AsyncOpenAI(
                    api_key=TTS_API_KEY,
                    base_url="https://api.openai.com/v1/",
                )
                response = await fallback_client.audio.speech.create(
                    model="tts-1",
                    input=text,
                    voice="alloy",
                    speed=speed,
                    response_format="mp3",
                )
                return response.content
            except Exception as fallback_err:
                logger.error("TTS fallback also failed: %s", fallback_err)
                raise TTSError(
                    f"语音合成失败（已尝试主备方案）: {str(fallback_err)}"
                ) from fallback_err

        raise TTSError(f"语音合成失败: {str(e)}") from e


async def synthesize_streaming(text: str, voice: str | None = None) -> bytes:
    """Synthesize speech with streaming response (lower latency).

    Same parameters as synthesize(), but uses streaming for faster first-byte
    time to audio.

    Args:
        text: Text to synthesize.
        voice: Voice name (defaults to TTS_VOICE from config).

    Returns:
        Raw audio bytes.
    """
    _validate_tts_input(text)

    selected_voice = voice or TTS_VOICE
    client = _get_tts_client()

    try:
        async with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL if TTS_PROVIDER != "zhipu" else "cogtts",
            input=text,
            voice=selected_voice,
            speed=1.0,
            response_format="mp3",
        ) as response:
            return await response.read()

    except Exception as e:
        logger.error("TTS streaming synthesis failed: %s", e)
        raise TTSError(f"语音合成失败: {str(e)}") from e
