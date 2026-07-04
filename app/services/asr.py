"""ASR (Automatic Speech Recognition) service.

Supports two providers:
  - openai:   OpenAI Whisper API (whisper-1) via /v1/audio/transcriptions
  - zhipu:    Zhipu GLM-ASR via BigModel API

Auto-detects provider from ASR_PROVIDER env var, with fallback between them.
"""

import io
import logging
import struct

from openai import AsyncOpenAI

from app.utils.config import (
    ASR_PROVIDER,
    ASR_MODEL,
    ASR_API_KEY,
    ASR_BASE_URL,
    MAX_AUDIO_SIZE_MB,
    MAX_AUDIO_DURATION_SECONDS,
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

# Estimated bitrate for duration estimation of non-WAV formats (kbps)
# Conservatively low to err on the side of rejecting potentially long audio.
# 32 kbps is a common voice codec bitrate (Opus @ 32kbps for webm).
ESTIMATED_BITRATE_KBPS = 32


class ASRError(Exception):
    """Raised when ASR processing fails."""


class AudioValidationError(ASRError):
    """Raised when audio fails size/type/duration validation."""


def _translate_error(err: Exception) -> str:
    """Turn a raw provider error into a user-facing Chinese message."""
    s = str(err)
    if "1113" in s or "余额" in s or "资源包" in s or "insufficient_quota" in s:
        return "语音识别额度不足，请为账户充值后重试。"
    if "429" in s or "rate limit" in s.lower():
        return "语音识别请求过于频繁，请稍后重试。"
    if "timed out" in s.lower() or "timeout" in s.lower():
        return "语音识别服务响应超时，请稍后重试。"
    return f"语音识别失败: {s}"


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

    # Check estimated duration
    estimated_sec = _estimate_audio_duration(audio_data, content_type)
    if estimated_sec > MAX_AUDIO_DURATION_SECONDS:
        raise AudioValidationError(
            f"音频时长过长: 约{estimated_sec:.0f}秒。"
            f"最大允许: {MAX_AUDIO_DURATION_SECONDS}秒"
        )


def _estimate_audio_duration(audio_data: bytes, content_type: str | None = None) -> float:
    """Estimate audio duration in seconds from raw bytes.

    For WAV files, reads the header for exact duration.
    For other formats, estimates from file size and a conservative bitrate.

    Args:
        audio_data: Raw audio bytes.
        content_type: MIME type of the audio.

    Returns:
        Estimated duration in seconds (0 if too small to estimate).
    """
    # Try WAV header parsing for exact duration
    if content_type == "audio/wav" and len(audio_data) >= 44:
        try:
            riff_tag = audio_data[:4]
            if riff_tag == b"RIFF":
                # WAV header: bytes 24-27 = sample_rate (little-endian uint32)
                # bytes 28-31 = byte_rate (little-endian uint32)
                # bytes 40-43 = data_size (little-endian uint32)
                sample_rate = struct.unpack_from("<I", audio_data, 24)[0]
                byte_rate = struct.unpack_from("<I", audio_data, 28)[0]
                data_size = struct.unpack_from("<I", audio_data, 40)[0]

                if byte_rate > 0 and sample_rate > 0:
                    return data_size / byte_rate
        except (struct.error, IndexError):
            pass  # Fall through to bitrate estimation

    # For non-WAV formats, estimate from file size and bitrate
    if len(audio_data) < 100:
        return 0.0

    # Duration = file_size_bytes * 8 / (bitrate_kbps * 1000)
    return (len(audio_data) * 8) / (ESTIMATED_BITRATE_KBPS * 1000)


def _detect_audio_format(data: bytes) -> str:
    """Guess audio MIME type from magic bytes.

    Returns a best-guess MIME type, defaulting to audio/webm.
    Only called when the caller doesn't know the MIME type (e.g. raw
    base64 submitted via the chat endpoint).
    """
    if len(data) < 12:
        return "audio/webm"
    if data[:4] == b"RIFF":
        return "audio/wav"
    if data[:3] == b"ID3" or (data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "audio/mpeg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"OggS":
        return "audio/ogg"
    # MP4/M4A: ftyp box at offset 4
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    # WebM: EBML header
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"
    return "audio/webm"


def _get_asr_client() -> AsyncOpenAI:
    """Get the ASR-specific AsyncOpenAI client."""
    return AsyncOpenAI(api_key=ASR_API_KEY, base_url=ASR_BASE_URL)


# Zhipu ASR accepts wav and mp3 (error 1214 for webm/mp4/ogg).
# MediaRecorder on most browsers produces webm or mp4. We relabel these to
# the nearest accepted type — mp4 (AAC audio) sent as "audio/mpeg" with a
# .mp3 extension is accepted by Zhipu's API in practice.
_ZHIPU_ASR_REMAP: dict[str, str] = {
    "audio/webm": "audio/wav",   # webm/Opus → wav (re-labelled; Zhipu reads the header)
    "audio/mp4":  "audio/mpeg",  # mp4/AAC   → mp3 (re-labelled; Zhipu accepts it)
    "audio/m4a":  "audio/mpeg",  # m4a       → mp3
    "audio/ogg":  "audio/wav",   # ogg       → wav
}

_EXT_MAP: dict[str, str] = {
    "audio/webm": "webm",
    "audio/wav":  "wav",
    "audio/mp3":  "mp3",
    "audio/mp4":  "mp4",
    "audio/mpeg": "mp3",
    "audio/m4a":  "m4a",
    "audio/ogg":  "ogg",
    "audio/flac": "flac",
    "audio/x-wav": "wav",
}


def _resolve_asr_content_type(content_type: str) -> str:
    """Remap the content type to one the active ASR provider accepts.

    Zhipu ASR rejects webm/mp4/ogg (error 1214) but accepts wav and mp3.
    We relabel the MIME type so the provider's intake passes, without
    transcoding the audio bytes.
    """
    if ASR_PROVIDER == "zhipu":
        return _ZHIPU_ASR_REMAP.get(content_type, content_type)
    return content_type


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

    # Remap content type to one the provider accepts (e.g. mp4→mp3 for Zhipu)
    api_content_type = _resolve_asr_content_type(content_type)
    suffix = f".{_EXT_MAP.get(api_content_type, 'wav')}"

    client = _get_asr_client()

    try:
        if ASR_PROVIDER == "zhipu":
            response = await client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=(f"audio{suffix}", io.BytesIO(audio_data), api_content_type),
                language=language,
            )
        else:
            kwargs: dict = {
                "model": ASR_MODEL,
                "file": (f"audio{suffix}", io.BytesIO(audio_data), api_content_type),
            }
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            response = await client.audio.transcriptions.create(**kwargs)

        return response.text.strip()

    except Exception as e:
        # No cross-provider fallback: retrying the Zhipu key against the
        # OpenAI endpoint can only time out — it's slow and hides the real
        # error (e.g. quota exhausted). Fail fast with the real cause.
        logger.error("ASR transcription failed: %s", e)
        raise ASRError(_translate_error(e)) from e
