"""ASR (Automatic Speech Recognition) service.

Supports two providers:
  - openai:   OpenAI Whisper API (whisper-1) via /v1/audio/transcriptions
  - zhipu:    Zhipu GLM-ASR via BigModel API

Auto-detects provider from ASR_PROVIDER env var, with fallback between them.
"""

import io
import logging

from openai import AsyncOpenAI

from app.utils.config import (
    ASR_PROVIDER,
    ASR_MODEL,
    ASR_API_KEY,
    ASR_BASE_URL,
    MAX_AUDIO_SIZE_MB,
)

logger = logging.getLogger(__name__)

# Supported audio MIME types
SUPPORTED_AUDIO_TYPES = {
    "audio/webm",
    "audio/wav",
    "audio/mp3",
    "audio/mp4",
    "audio/mpeg",
    "audio/m4a",
    "audio/ogg",
    "audio/flac",
}


class ASRError(Exception):
    """Raised when ASR processing fails."""


class AudioValidationError(ASRError):
    """Raised when audio fails size/type/duration validation."""


def validate_audio(audio_data: bytes, content_type: str | None = None) -> None:
    """Validate audio data before sending to ASR API.

    Args:
        audio_data: Raw audio bytes.
        content_type: MIME type of the audio.

    Raises:
        AudioValidationError: If audio fails validation.
    """
    # Check content type
    if content_type and content_type not in SUPPORTED_AUDIO_TYPES:
        raise AudioValidationError(
            f"不支持的音频格式: {content_type}。"
            f"支持的格式: {', '.join(sorted(SUPPORTED_AUDIO_TYPES))}"
        )

    # Check size
    size_mb = len(audio_data) / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        raise AudioValidationError(
            f"音频文件过大: {size_mb:.1f}MB。最大允许: {MAX_AUDIO_SIZE_MB}MB"
        )

    # Check not empty
    if len(audio_data) < 100:
        raise AudioValidationError("音频文件过小，可能为空或损坏")


def _get_asr_client() -> AsyncOpenAI:
    """Get the ASR-specific AsyncOpenAI client."""
    return AsyncOpenAI(api_key=ASR_API_KEY, base_url=ASR_BASE_URL)


async def transcribe(
    audio_data: bytes,
    content_type: str = "audio/webm",
    language: str | None = "zh",
    prompt: str | None = None,
) -> str:
    """Transcribe audio to text using the configured ASR provider.

    Args:
        audio_data: Raw audio bytes.
        content_type: MIME type (default: audio/webm for MediaRecorder).
        language: ISO 639-1 language code (None for auto-detect).
        prompt: Optional prompt for guiding transcription style.

    Returns:
        Transcribed text string.

    Raises:
        AudioValidationError: If audio fails validation.
        ASRError: If the ASR API call fails.
    """
    validate_audio(audio_data, content_type)

    # Determine file extension from MIME type
    ext_map = {
        "audio/webm": "webm",
        "audio/wav": "wav",
        "audio/mp3": "mp3",
        "audio/mp4": "mp4",
        "audio/mpeg": "mp3",
        "audio/m4a": "m4a",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
    }
    suffix = f".{ext_map.get(content_type, 'webm')}"

    client = _get_asr_client()

    try:
        if ASR_PROVIDER == "zhipu":
            # Zhipu GLM-ASR endpoint
            response = await client.audio.transcriptions.create(
                model="glm-asr-2512",
                file=(f"audio{suffix}", io.BytesIO(audio_data), content_type),
                language=language,
            )
        else:
            # OpenAI Whisper API
            kwargs: dict = {
                "model": ASR_MODEL,
                "file": (f"audio{suffix}", io.BytesIO(audio_data), content_type),
            }
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            response = await client.audio.transcriptions.create(**kwargs)

        return response.text.strip()

    except Exception as e:
        logger.error("ASR transcription failed: %s", e)

        # Fallback: if primary provider fails, try the other
        if ASR_PROVIDER == "zhipu":
            logger.info("Falling back to OpenAI Whisper...")
            try:
                fallback_client = AsyncOpenAI(
                    api_key=ASR_API_KEY,
                    base_url="https://api.openai.com/v1/",
                )
                response = await fallback_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(f"audio{suffix}", io.BytesIO(audio_data), content_type),
                    language=language,
                )
                return response.text.strip()
            except Exception as fallback_err:
                logger.error("ASR fallback also failed: %s", fallback_err)
                raise ASRError(
                    f"语音识别失败（已尝试主备方案）: {str(fallback_err)}"
                ) from fallback_err

        raise ASRError(f"语音识别失败: {str(e)}") from e
