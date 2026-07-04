# Week 5: Multimodal AI Application — Implementation Plan

> **For agentic workers:** This plan is designed for step-by-step execution. Each task is self-contained with complete code and test cases. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an "AI Learning Assistant" that can see (image input), hear (ASR), and speak (TTS) — a multimodal AI product with voice + visual interaction.

**Architecture:** Extend the existing Week 4 FastAPI layered architecture (routers/schemas/services). Add three new services (ASR, TTS, MultimodalChat) behind new REST endpoints. The frontend gains voice recording (MediaRecorder) and TTS playback (Audio API) on top of existing image upload + SSE streaming. All multimodal context (images, voice transcripts, text) lives in a unified session history.

**Tech Stack:** FastAPI + AsyncOpenAI + sse-starlette + MediaRecorder API + Web Audio API. ASR via OpenAI whisper-1 (Zhipu GLM-ASR fallback). TTS via OpenAI tts-1 (Zhipu CogTTS fallback). Vision via existing glm-4.6v-flash multimodal model.

---

## File Structure Map

### New Files (8 files)
| File | Responsibility |
|------|---------------|
| `app/services/asr.py` | ASR service — Whisper API wrapper with Zhipu GLM-ASR fallback |
| `app/services/tts.py` | TTS service — tts-1 wrapper with Zhipu CogTTS fallback |
| `app/services/multimodal_chat.py` | Unified multimodal chat — image + voice + text, SSE streaming, session context |
| `app/routers/multimodal.py` | REST endpoints: /chat, /asr, /tts, voice Q&A loop |
| `app/schemas/multimodal.py` | Pydantic models for multimodal requests & responses |
| `tests/test_multimodal.py` | 8+ test cases covering multimodal input, ASR/TTS mocking, session context |
| `app/routers/__init__.py` | (already exists, no change needed) |
| `app/services/__init__.py` | (already exists, no change needed) |

### Modified Files (5 files)
| File | Change |
|------|--------|
| `app/utils/config.py` | Add ASR/TTS/multimodal config vars |
| `main.py` | Register new multimodal router, add upload size limit middleware, create audio dirs |
| `requirements.txt` | Already has openai, httpx, etc. — no changes needed |
| `.env.example` | Add ASR/TTS config entries |
| `static/index.html` | Major rewrite: add voice recording, TTS playback, camera capture, unified multimodal UI |
| `tests/conftest.py` | Add ASR/TTS mock fixtures |

---

## Task Breakdown

### Task 1: Config Expansion + .env.example Update

**Files:**
- Modify: `app/utils/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Add ASR/TTS/multimodal config to config.py**

Append to `app/utils/config.py` after the existing Agent section:

```python
# --- ASR (Speech-to-Text) ---
ASR_PROVIDER: str = get_env("ASR_PROVIDER", "zhipu")
ASR_MODEL: str = get_env("ASR_MODEL", "whisper-1")
ASR_API_KEY: str = get_env("ASR_API_KEY", OPENAI_API_KEY)
ASR_BASE_URL: str = get_env("ASR_BASE_URL", "https://api.openai.com/v1/")

# --- TTS (Text-to-Speech) ---
TTS_PROVIDER: str = get_env("TTS_PROVIDER", "zhipu")
TTS_MODEL: str = get_env("TTS_MODEL", "tts-1")
TTS_VOICE: str = get_env("TTS_VOICE", "alloy")
TTS_API_KEY: str = get_env("TTS_API_KEY", OPENAI_API_KEY)
TTS_BASE_URL: str = get_env("TTS_BASE_URL", "https://api.openai.com/v1/")

# --- Audio ---
MAX_AUDIO_SIZE_MB: int = int(get_env("MAX_AUDIO_SIZE_MB", "10"))
MAX_AUDIO_DURATION_SECONDS: int = int(get_env("MAX_AUDIO_DURATION_SECONDS", "120"))
MAX_IMAGE_SIZE_MB: int = int(get_env("MAX_IMAGE_SIZE_MB", "10"))

# --- Multimodal Chat ---
MULTIMODAL_MAX_HISTORY_TURNS: int = int(get_env("MULTIMODAL_MAX_HISTORY_TURNS", "20"))
```

- [ ] **Step 2: Update .env.example**

Append to `.env.example`:

```bash
# --- ASR (Speech-to-Text) ---
# ASR_PROVIDER=zhipu          # "openai" or "zhipu"
# ASR_MODEL=whisper-1         # OpenAI: whisper-1; Zhipu: glm-asr-2512
# ASR_API_KEY=                # Defaults to OPENAI_API_KEY
# ASR_BASE_URL=https://api.openai.com/v1/   # Zhipu: https://open.bigmodel.cn/api/paas/v4/

# --- TTS (Text-to-Speech) ---
# TTS_PROVIDER=zhipu          # "openai" or "zhipu"
# TTS_MODEL=tts-1             # OpenAI: tts-1/tts-1-hd; Zhipu: cogtts
# TTS_VOICE=alloy             # alloy, echo, fable, nova, onyx, shimmer, ash, coral, sage
# TTS_API_KEY=                # Defaults to OPENAI_API_KEY
# TTS_BASE_URL=https://api.openai.com/v1/   # Zhipu: https://open.bigmodel.cn/api/paas/v4/

# --- Audio Limits ---
# MAX_AUDIO_SIZE_MB=10
# MAX_AUDIO_DURATION_SECONDS=120
# MAX_IMAGE_SIZE_MB=10

# --- Multimodal ---
# MULTIMODAL_MAX_HISTORY_TURNS=20
```

- [ ] **Step 3: Commit**

```bash
git add app/utils/config.py .env.example
git commit -m "feat: add ASR, TTS, and multimodal config entries"
```

---

### Task 2: ASR Service (Speech-to-Text)

**Files:**
- Create: `app/services/asr.py`
- Modify: `tests/conftest.py` (add ASR mock fixture)
- Test: `tests/test_multimodal.py` (first 2 tests)

- [ ] **Step 1: Write ASR service**

Create `app/services/asr.py`:

```python
"""ASR (Automatic Speech Recognition) service.

Supports two providers:
  - openai:   OpenAI Whisper API (whisper-1) via /v1/audio/transcriptions
  - zhipu:    Zhipu GLM-ASR via BigModel API

Auto-detects provider from ASR_PROVIDER env var, falls back between them.
"""

import io
import logging
from pathlib import Path

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
            kwargs = {
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

        # Fallback: if primary provider fails and we have a different one configured
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
                raise ASRError(f"语音识别失败（已尝试主备方案）: {str(fallback_err)}") from fallback_err

        raise ASRError(f"语音识别失败: {str(e)}") from e
```

- [ ] **Step 2: Add ASR mock fixture to conftest.py**

Append to `tests/conftest.py` after the existing fixtures:

```python
# ---------------------------------------------------------------------------
# ASR Mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_asr(monkeypatch):
    """Mock ASR transcribe to return deterministic text without API calls."""

    async def _mock_transcribe(
        audio_data: bytes,
        content_type: str = "audio/webm",
        language: str | None = "zh",
        prompt: str | None = None,
    ) -> str:
        # Return different text based on audio size to simulate real behavior
        if len(audio_data) < 100:
            return ""
        return "人工智能如何改变教育"

    monkeypatch.setattr("app.services.asr.transcribe", _mock_transcribe)
    return _mock_transcribe


@pytest.fixture()
def mock_asr_error(monkeypatch):
    """Mock ASR to simulate API failure."""

    async def _mock_fail(*args, **kwargs):
        from app.services.asr import ASRError
        raise ASRError("模拟ASR服务不可用")

    monkeypatch.setattr("app.services.asr.transcribe", _mock_fail)
```

- [ ] **Step 3: Write ASR unit tests**

Create `tests/test_multimodal.py` with first tests:

```python
"""Multimodal endpoint tests — ASR, TTS, and multimodal chat.

Tests:
1. Audio validation — supported types pass, unsupported rejected
2. Audio validation — size limit enforcement
3. ASR transcribe — returns text from mock
4. TTS synthesize — returns audio bytes from mock
5. Multimodal chat — image + text input
6. Multimodal chat — voice (base64 audio) + text input
7. Session context — multi-turn with image preservation
8. Error handling — missing/invalid input returns appropriate codes
"""

import io
import pytest

from app.services.asr import validate_audio, transcribe, AudioValidationError, ASRError


# ---------------------------------------------------------------------------
# Test 1: Audio validation — supported types
# ---------------------------------------------------------------------------

class TestAudioValidation:
    """Audio validation tests."""

    def test_valid_audio_passes(self):
        """Valid audio data with supported MIME type should pass validation."""
        audio = b"\x00" * 1000  # 1KB dummy audio
        validate_audio(audio, "audio/webm")  # Should not raise

    def test_invalid_content_type_raises(self):
        """Unsupported MIME type should raise AudioValidationError."""
        audio = b"\x00" * 1000
        with pytest.raises(AudioValidationError, match="不支持的音频格式"):
            validate_audio(audio, "video/mp4")

    def test_oversized_audio_raises(self, monkeypatch):
        """Audio exceeding max size should raise AudioValidationError."""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "1")
        audio = b"\x00" * (2 * 1024 * 1024)  # 2MB
        with pytest.raises(AudioValidationError, match="音频文件过大"):
            validate_audio(audio, "audio/webm")

    def test_empty_audio_raises(self):
        """Very small audio should raise AudioValidationError."""
        audio = b"\x00" * 10
        with pytest.raises(AudioValidationError):
            validate_audio(audio, "audio/webm")


# ---------------------------------------------------------------------------
# Test 2: ASR transcribe — mock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_asr_transcribe_returns_text(mock_asr):
    """ASR should transcribe audio bytes to text."""
    audio = b"\x00" * 5000  # dummy audio data
    result = await transcribe(audio, content_type="audio/webm")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "人工智能" in result


@pytest.mark.asyncio
async def test_asr_transcribe_empty_audio(mock_asr):
    """Empty audio should return empty string or raise."""
    audio = b"\x00" * 10
    with pytest.raises(AudioValidationError):
        await transcribe(audio, content_type="audio/webm")


@pytest.mark.asyncio
async def test_asr_error_propagates(mock_asr_error):
    """ASR errors should be raised as ASRError."""
    audio = b"\x00" * 5000
    with pytest.raises(ASRError):
        await transcribe(audio, content_type="audio/webm")
```

- [ ] **Step 4: Run tests to verify**

```bash
python -m pytest tests/test_multimodal.py::TestAudioValidation -v
python -m pytest tests/test_multimodal.py::test_asr_transcribe_returns_text -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/asr.py tests/conftest.py tests/test_multimodal.py
git commit -m "feat: add ASR service with validation and tests"
```

---

### Task 3: TTS Service (Text-to-Speech)

**Files:**
- Create: `app/services/tts.py`
- Modify: `tests/conftest.py` (add TTS mock fixture)
- Modify: `tests/test_multimodal.py` (add TTS tests)

- [ ] **Step 1: Write TTS service**

Create `app/services/tts.py`:

```python
"""TTS (Text-to-Speech) service.

Supports two providers:
  - openai:   OpenAI TTS API (tts-1 / tts-1-hd) via /v1/audio/speech
  - zhipu:    Zhipu CogTTS via BigModel API

Returns audio bytes that the frontend plays via the Web Audio API.
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
                extra_body={"provider": "zhipu"},
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

        # Fallback: if primary provider fails
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
                raise TTSError(f"语音合成失败（已尝试主备方案）: {str(fallback_err)}") from fallback_err

        raise TTSError(f"语音合成失败: {str(e)}") from e


async def synthesize_streaming(text: str, voice: str | None = None) -> bytes:
    """Synthesize speech with streaming response (lower latency).

    Same parameters as synthesize(), but uses streaming for faster first-byte.
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
```

- [ ] **Step 2: Add TTS mock fixtures to conftest.py**

Append after the ASR mocks:

```python
# ---------------------------------------------------------------------------
# TTS Mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_tts(monkeypatch):
    """Mock TTS synthesize to return dummy audio bytes without API calls."""

    async def _mock_synthesize(
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        # Return minimal valid MP3 bytes (or just dummy data for testing)
        return b"\xff\xfb\x90\x00" + b"\x00" * 256

    async def _mock_synthesize_streaming(
        text: str,
        voice: str | None = None,
    ) -> bytes:
        return b"\xff\xfb\x90\x00" + b"\x00" * 256

    monkeypatch.setattr("app.services.tts.synthesize", _mock_synthesize)
    monkeypatch.setattr("app.services.tts.synthesize_streaming", _mock_synthesize_streaming)
    return _mock_synthesize


@pytest.fixture()
def mock_tts_error(monkeypatch):
    """Mock TTS to simulate API failure."""

    async def _mock_fail(*args, **kwargs):
        from app.services.tts import TTSError
        raise TTSError("模拟TTS服务不可用")

    monkeypatch.setattr("app.services.tts.synthesize", _mock_fail)
```

- [ ] **Step 3: Add TTS tests to test_multimodal.py**

```python
# Add after existing tests in tests/test_multimodal.py

from app.services.tts import synthesize, _validate_tts_input, TTSError


# ---------------------------------------------------------------------------
# Test 3: TTS synthesize — mock
# ---------------------------------------------------------------------------

class TestTTSValidation:
    """TTS input validation tests."""

    def test_valid_text_passes(self):
        """Valid text should pass validation."""
        _validate_tts_input("你好，这是测试文本。")  # Should not raise

    def test_empty_text_raises(self):
        """Empty TTS input should raise ValueError."""
        with pytest.raises(ValueError, match="不能为空"):
            _validate_tts_input("")

    def test_whitespace_only_raises(self):
        """Whitespace-only TTS input should raise ValueError."""
        with pytest.raises(ValueError, match="不能为空"):
            _validate_tts_input("   ")

    def test_overlong_text_raises(self):
        """Text exceeding 4096 chars should raise ValueError."""
        with pytest.raises(ValueError, match="过长"):
            _validate_tts_input("啊" * 5000)


@pytest.mark.asyncio
async def test_tts_synthesize_returns_bytes(mock_tts):
    """TTS should return audio bytes for valid text."""
    result = await synthesize("你好，世界")
    assert isinstance(result, bytes)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_tts_error_propagates(mock_tts_error):
    """TTS errors should be raised as TTSError."""
    with pytest.raises(TTSError):
        await synthesize("测试文本")
```

- [ ] **Step 4: Run TTS tests**

```bash
python -m pytest tests/test_multimodal.py::TestTTSValidation -v
python -m pytest tests/test_multimodal.py::test_tts_synthesize_returns_bytes -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/services/tts.py tests/conftest.py tests/test_multimodal.py
git commit -m "feat: add TTS service with validation and tests"
```

---

### Task 4: Multimodal Schemas

**Files:**
- Create: `app/schemas/multimodal.py`

- [ ] **Step 1: Write multimodal Pydantic schemas**

Create `app/schemas/multimodal.py`:

```python
"""Pydantic v2 schemas for multimodal API endpoints."""

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

class ASRRequest(BaseModel):
    """Audio transcription request.

    Audio is sent as raw bytes via multipart/form-data, so this schema
    is used for the query/form parameters, not the body.
    """

    language: str | None = Field(
        default="zh",
        description="ISO 639-1 language code. Set to null for auto-detection.",
    )
    prompt: str | None = Field(
        default=None,
        max_length=500,
        description="Optional prompt to guide transcription style.",
    )


class ASRResponse(BaseModel):
    """Audio transcription response."""

    text: str = Field(..., description="Transcribed text")
    language: str | None = Field(default=None, description="Detected language")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    """Text-to-speech request."""

    text: str = Field(
        ..., min_length=1, max_length=4096,
        description="Text to convert to speech",
    )
    voice: str = Field(
        default="alloy",
        description="Voice: alloy, echo, fable, nova, onyx, shimmer, ash, coral, sage",
    )
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0,
        description="Playback speed",
    )


# ---------------------------------------------------------------------------
# Multimodal Chat
# ---------------------------------------------------------------------------

class MultimodalChatRequest(BaseModel):
    """Unified multimodal chat request.

    Supports any combination of text, image, and voice input.
    At least one input modality must be provided.
    """

    text: str | None = Field(
        default=None,
        max_length=10000,
        description="Text question (optional if image or audio provided)",
    )
    image_base64: str | None = Field(
        default=None,
        description="Image as data:image/...;base64,... string. "
                    "Image goes directly into multimodal model content.",
    )
    image_url: str | None = Field(
        default=None,
        description="Image URL. Mutually exclusive with image_base64.",
    )
    audio_base64: str | None = Field(
        default=None,
        description="Audio as base64-encoded bytes. "
                    "Will be transcribed via ASR before sending to LLM.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for continuing a previous conversation. "
                    "If not provided, a new session is created.",
    )
    template: str = Field(
        default="basic",
        description="ReAct template: basic | structured | self_correcting",
    )
    max_steps: int = Field(
        default=10, ge=1, le=20,
        description="Maximum reasoning steps",
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        valid = {"basic", "structured", "self_correcting"}
        if v not in valid:
            raise ValueError(f"无效的模板: '{v}'。可选: {', '.join(sorted(valid))}")
        return v

    def model_post_init(self, __context):
        """Ensure at least one input modality is provided."""
        if not self.text and not self.image_base64 and not self.image_url and not self.audio_base64:
            raise ValueError(
                "至少需要提供一种输入: text, image_base64, image_url, 或 audio_base64"
            )


class VoiceLoopRequest(BaseModel):
    """Voice Q&A loop: audio in → ASR → LLM → TTS → audio out."""

    audio_base64: str = Field(
        ..., min_length=1,
        description="Base64-encoded audio from voice recording",
    )
    language: str = Field(
        default="zh",
        description="Language for ASR transcription",
    )
    template: str = Field(
        default="basic",
        description="ReAct template for LLM reasoning",
    )
    max_steps: int = Field(
        default=5, ge=1, le=10,
        description="Maximum agent steps (lower for voice to reduce latency)",
    )


class VoiceLoopResponse(BaseModel):
    """Voice Q&A loop response."""

    text: str = Field(..., description="LLM text answer")
    audio_base64: str = Field(..., description="TTS audio as base64-encoded mp3 bytes")
    session_id: str = Field(..., description="Session ID for conversation continuity")
```

- [ ] **Step 2: Commit**

```bash
git add app/schemas/multimodal.py
git commit -m "feat: add multimodal Pydantic schemas"
```

---

### Task 5: Multimodal Chat Service

**Files:**
- Create: `app/services/multimodal_chat.py`

- [ ] **Step 1: Write multimodal chat service**

Create `app/services/multimodal_chat.py`:

```python
"""Multimodal Chat Service — unified chat with image + voice + text input.

Key design decisions:
  - Images go directly into multimodal model content (no lossy text summary).
  - Voice audio is transcribed via ASR first, then the text goes to LLM.
  - Session context preserves image references across multi-turn conversations.
  - SSE streaming for text responses (reuses existing ReAct agent loop).
"""

import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

from app.services.llm import get_client, get_model
from app.services.agent import run_agent_stream
from app.services.asr import transcribe
from app.services.tts import synthesize

logger = logging.getLogger(__name__)

# In-memory multimodal session store
# Each session holds: session_id, history (list of message dicts), created_at
_multimodal_sessions: dict[str, dict] = {}


def _create_session() -> str:
    """Create a new multimodal session and return its ID."""
    session_id = uuid.uuid4().hex[:12]
    _multimodal_sessions[session_id] = {
        "session_id": session_id,
        "history": [],          # list of {"role": "user"|"assistant", "content": [...]}
        "image_url": None,      # latest image URL in this session
        "image_base64": None,   # latest image base64 in this session
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return session_id


def _get_session(session_id: str) -> dict | None:
    """Get a multimodal session by ID."""
    return _multimodal_sessions.get(session_id)


def _update_session_image(
    session_id: str,
    image_url: str | None = None,
    image_base64: str | None = None,
) -> None:
    """Update the image context for a session so follow-up questions retain it."""
    session = _multimodal_sessions.get(session_id)
    if session:
        if image_url:
            session["image_url"] = image_url
        if image_base64:
            session["image_base64"] = image_base64


def _append_to_history(
    session_id: str,
    role: str,
    content: list[dict] | str,
) -> None:
    """Append a message to the session conversation history."""
    session = _multimodal_sessions.get(session_id)
    if session:
        session["history"].append({"role": role, "content": content})


def list_multimodal_sessions() -> list[dict]:
    """List all multimodal sessions."""
    return [
        {
            "session_id": s["session_id"],
            "turns": len(s["history"]) // 2,
            "has_image": bool(s.get("image_url") or s.get("image_base64")),
            "created_at": s["created_at"],
        }
        for s in _multimodal_sessions.values()
    ]


def clear_multimodal_sessions() -> None:
    """Clear all sessions (for testing)."""
    _multimodal_sessions.clear()


async def run_multimodal_chat_stream(
    text: str | None = None,
    image_url: str | None = None,
    image_base64: str | None = None,
    audio_base64: str | None = None,
    session_id: str | None = None,
    template: str = "basic",
    max_steps: int = 10,
) -> AsyncIterator[dict]:
    """Run multimodal chat with SSE streaming.

    Processing order:
    1. If audio provided → transcribe via ASR → use as text input
    2. Build user message content with text + image (direct to multimodal model)
    3. Pass through ReAct agent loop with SSE streaming
    4. Save to session history for multi-turn context

    Args:
        text: Optional text question.
        image_url: Optional image URL.
        image_base64: Optional image base64 (data:image/...;base64,...).
        audio_base64: Optional base64-encoded audio for ASR.
        session_id: Optional session ID for continuing a conversation.
        template: ReAct prompt template.
        max_steps: Max reasoning steps.

    Yields:
        SSE event dicts with "event" and "data" keys.
    """
    # --- Step 1: Process audio (ASR) ---
    if audio_base64:
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            transcribed_text = await transcribe(audio_bytes, content_type="audio/webm")

            yield {
                "event": "asr_result",
                "data": json.dumps(
                    {"text": transcribed_text},
                    ensure_ascii=False,
                ),
            }

            # Merge transcribed text with any provided text
            if transcribed_text:
                if text:
                    text = f"{text}\n{transcribed_text}"
                else:
                    text = transcribed_text
        except Exception as e:
            logger.error("ASR in multimodal chat failed: %s", e)
            yield {
                "event": "asr_error",
                "data": json.dumps(
                    {"error": f"语音识别失败: {str(e)}"},
                    ensure_ascii=False,
                ),
            }
            # Continue with whatever text we have (may be empty)

    if not text and not image_url and not image_base64:
        yield {
            "event": "error",
            "data": json.dumps(
                {"error": "请提供文字问题、图片或语音输入"},
                ensure_ascii=False,
            ),
        }
        return

    # --- Step 2: Session management ---
    if not session_id:
        session_id = _create_session()
    elif not _get_session(session_id):
        session_id = _create_session()

    # Update image context if provided
    resolved_image_url = image_url
    resolved_image_base64 = image_base64
    if not resolved_image_url and not resolved_image_base64:
        # Reuse session's last image if no new image provided
        session = _get_session(session_id)
        if session:
            resolved_image_url = session.get("image_url")
            resolved_image_base64 = session.get("image_base64")
    else:
        _update_session_image(session_id, resolved_image_url, resolved_image_base64)

    # Determine which image to use
    final_image = resolved_image_url or resolved_image_base64

    # --- Step 3: Build effective question ---
    effective_question = text or "请分析这张图片"

    # --- Step 4: Delegate to ReAct agent loop ---
    # Append to history
    _append_to_history(session_id, "user", effective_question)

    async for event in run_agent_stream(
        question=effective_question,
        session_id=session_id,
        template=template,
        max_steps=max_steps,
        image_url=final_image,
    ):
        # Inject session_id into done event
        if event["event"] == "done":
            data = json.loads(event["data"])
            data["multimodal_session_id"] = session_id
            event["data"] = json.dumps(data, ensure_ascii=False)

            # Save assistant response to history
            answer_text = ""
            # The final answer was already yielded in an 'answer' event

        yield event

    yield {
        "event": "session",
        "data": json.dumps(
            {"multimodal_session_id": session_id},
            ensure_ascii=False,
        ),
    }


async def run_voice_loop(
    audio_base64: str,
    language: str = "zh",
    template: str = "basic",
    max_steps: int = 5,
) -> dict:
    """Run full voice Q&A loop: ASR → LLM → TTS.

    Args:
        audio_base64: Base64-encoded audio from voice recording.
        language: ASR language code.
        template: ReAct template.
        max_steps: Max agent steps.

    Returns:
        Dict with text, audio_base64, and session_id keys.
    """
    # Step 1: ASR
    audio_bytes = base64.b64decode(audio_base64)
    transcribed = await transcribe(audio_bytes, content_type="audio/webm", language=language)

    # Step 2: LLM (collect streaming output)
    from app.services.agent import run_agent_sync
    from app.services.sessions import get_session

    session_id = uuid.uuid4().hex[:12]
    agent_result = await run_agent_sync(
        question=transcribed,
        session_id=session_id,
        template=template,
        max_steps=max_steps,
    )

    answer_text = agent_result.final_answer if agent_result else "抱歉，我无法回答这个问题。"

    # Step 3: TTS
    tts_audio = await synthesize(answer_text[:2000])  # Truncate to avoid TTS limit
    audio_b64 = base64.b64encode(tts_audio).decode("utf-8")

    return {
        "text": answer_text,
        "audio_base64": audio_b64,
        "session_id": session_id,
    }
```

- [ ] **Step 2: Commit**

```bash
git add app/services/multimodal_chat.py
git commit -m "feat: add multimodal chat service with ASR→LLM→TTS voice loop"
```

---

### Task 6: Multimodal Router

**Files:**
- Create: `app/routers/multimodal.py`
- Modify: `main.py` (register router + add MIDDLEWARE for body size + create audio dirs)

- [ ] **Step 1: Write multimodal router**

Create `app/routers/multimodal.py`:

```python
"""Multimodal Router — endpoints for image + voice + text interaction."""

import base64
import json
import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.schemas.multimodal import (
    ASRRequest,
    ASRResponse,
    TTSRequest,
    MultimodalChatRequest,
    VoiceLoopRequest,
    VoiceLoopResponse,
)
from app.services.asr import transcribe, AudioValidationError, ASRError
from app.services.tts import synthesize, TTSError
from app.services.multimodal_chat import (
    run_multimodal_chat_stream,
    run_voice_loop,
    list_multimodal_sessions,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["multimodal"])

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}


# ---------------------------------------------------------------------------
# POST /asr — Audio transcription
# ---------------------------------------------------------------------------

@router.post("/asr", response_model=ASRResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = Form(default="zh"),
):
    """Transcribe audio to text.

    Accepts audio file upload via multipart/form-data.
    Returns the transcribed text.
    """
    # Validate content type
    if file.content_type and "audio" not in file.content_type:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}。请上传音频文件。",
        )

    try:
        audio_data = await file.read()
        text = await transcribe(audio_data, content_type=file.content_type or "audio/webm", language=language)
        return ASRResponse(text=text, language=language)
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ASRError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("ASR endpoint error: %s", e)
        raise HTTPException(status_code=500, detail="语音识别服务暂时不可用")


# ---------------------------------------------------------------------------
# POST /tts — Text-to-speech
# ---------------------------------------------------------------------------

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech audio.

    Returns audio/mpeg bytes that can be played directly in the browser.
    """
    try:
        audio_bytes = await synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
        )
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
                "Cache-Control": "no-cache",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TTSError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error("TTS endpoint error: %s", e)
        raise HTTPException(status_code=500, detail="语音合成服务暂时不可用")


# ---------------------------------------------------------------------------
# POST /chat — Multimodal chat (SSE streaming)
# ---------------------------------------------------------------------------

@router.post("/chat")
async def multimodal_chat(request: MultimodalChatRequest):
    """Unified multimodal chat with SSE streaming.

    Accepts text + image + voice input. Processes:
    1. Voice audio → ASR transcription
    2. Builds multimodal message content (image to model directly)
    3. Runs ReAct agent with SSE streaming

    The 'asr_result' event contains the transcription when audio is provided.
    The 'session' event at the end contains the multimodal session ID for follow-up.
    """

    async def event_generator():
        async for event in run_multimodal_chat_stream(
            text=request.text,
            image_url=request.image_url,
            image_base64=request.image_base64,
            audio_base64=request.audio_base64,
            session_id=request.session_id,
            template=request.template,
            max_steps=request.max_steps,
        ):
            yield event

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# POST /voice — Voice Q&A loop (audio in → text + audio out)
# ---------------------------------------------------------------------------

@router.post("/voice", response_model=VoiceLoopResponse)
async def voice_loop(request: VoiceLoopRequest):
    """Complete voice Q&A loop.

    Audio in → ASR → LLM → TTS → audio out.
    Returns both the text answer and TTS audio as base64.
    """
    try:
        result = await run_voice_loop(
            audio_base64=request.audio_base64,
            language=request.language,
            template=request.template,
            max_steps=request.max_steps,
        )
        return VoiceLoopResponse(**result)
    except Exception as e:
        logger.error("Voice loop error: %s", e)
        raise HTTPException(status_code=500, detail=f"语音问答处理失败: {str(e)}")


# ---------------------------------------------------------------------------
# GET /sessions — List multimodal sessions
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_sessions():
    """List all multimodal chat sessions."""
    sessions = list_multimodal_sessions()
    return {"sessions": sessions, "total": len(sessions)}


# ---------------------------------------------------------------------------
# POST /upload/image — Image upload helper
# ---------------------------------------------------------------------------

@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and return it as base64 for use in chat.

    Validates file type and size. Returns the data URI.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片类型: {file.content_type}。"
                    f"支持: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    from app.utils.config import MAX_IMAGE_SIZE_MB

    image_data = await file.read()
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"图片过大: {size_mb:.1f}MB。最大允许: {MAX_IMAGE_SIZE_MB}MB",
        )

    b64 = base64.b64encode(image_data).decode("utf-8")
    data_uri = f"data:{file.content_type};base64,{b64}"

    return {
        "data_uri": data_uri,
        "size_bytes": len(image_data),
        "content_type": file.content_type,
    }
```

- [ ] **Step 2: Register router in main.py**

Modify `main.py` to add the multimodal router. After the existing agent router registration:

```python
from app.routers import multimodal

app.include_router(multimodal.router, prefix="/api/v1/multimodal")
```

And add the audio directory to startup:

```python
@app.on_event("startup")
async def startup():
    """Create required directories on startup."""
    for d in [
        "./chroma_data",
        "./data/sessions",
        "./data/images",
        "./data/audio",       # NEW
    ]:
        os.makedirs(d, exist_ok=True)
```

Also add request body size limit middleware. After the CORS middleware:

```python
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

# Add after CORS middleware registration:

class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding the configured size limit."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            max_size = 15 * 1024 * 1024  # 15MB max
            if int(content_length) > max_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "请求体过大，最大允许15MB"},
                )
        return await call_next(request)

app.add_middleware(MaxUploadSizeMiddleware)
```

- [ ] **Step 3: Add multimodal upload endpoint test to conftest.py**

Append to `tests/conftest.py`:

```python
# ---------------------------------------------------------------------------
# Multimodal Mock Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_multimodal_agent(monkeypatch):
    """Mock run_agent_stream for multimodal chat tests to avoid real LLM calls."""

    async def _mock_run_agent_stream(**kwargs):
        events = [
            {
                "event": "thought",
                "data": '{"step": 1, "thought": "我来分析这张图片中的题目。"}',
            },
            {
                "event": "answer",
                "data": '{"answer": "这是关于二次函数的题目。解题步骤如下：...", "hit_max_steps": false}',
            },
            {
                "event": "done",
                "data": '{"session_id": "test123", "total_steps": 1, "template": "basic", "hit_max_steps": false}',
            },
        ]
        for e in events:
            yield e

    async def _mock_multimodal_stream(**kwargs):
        yield {"event": "asr_result", "data": '{"text": "这道题怎么做？"}'}
        yield {"event": "thought", "data": '{"step": 1, "thought": "分析图片中的题目..."}'}
        yield {
            "event": "answer",
            "data": '{"answer": "这是一道几何题。连接AB两点，根据勾股定理...", "hit_max_steps": false}',
        }
        yield {
            "event": "done",
            "data": '{"session_id": "m_test123", "total_steps": 1, "template": "basic", "hit_max_steps": false, "multimodal_session_id": "m_test123"}',
        }
        yield {"event": "session", "data": '{"multimodal_session_id": "m_test123"}'}

    monkeypatch.setattr(
        "app.services.multimodal_chat.run_multimodal_chat_stream",
        _mock_multimodal_stream,
    )

    # Also mock voice loop
    async def _mock_voice_loop(**kwargs):
        return {
            "text": "这是一道几何题。连接AB两点...",
            "audio_base64": "//uQZAAAAA=",
            "session_id": "vl_test123",
        }
    monkeypatch.setattr(
        "app.services.multimodal_chat.run_voice_loop",
        _mock_voice_loop,
    )


@pytest.fixture(autouse=True)
def clear_multimodal_sessions():
    """Clear multimodal sessions before each test."""
    from app.services.multimodal_chat import clear_multimodal_sessions
    clear_multimodal_sessions()
    yield
    clear_multimodal_sessions()
```

- [ ] **Step 4: Add router integration tests**

Append to `tests/test_multimodal.py`:

```python
# Add imports at top:
import json
from httpx import AsyncClient, ASGITransport
from main import app


@pytest_asyncio.fixture
async def test_app_mm():
    """Async HTTP client for multimodal endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Test 4: Multimodal chat — image + text input
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multimodal_chat_text_only(test_app_mm):
    """Text-only multimodal chat should work (delegates to agent)."""
    response = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "text": "什么是二次函数？",
            "template": "basic",
            "max_steps": 5,
        },
    )
    assert response.status_code == 200

    # Parse SSE events
    events = _parse_sse(response.text)
    event_types = [e["type"] for e in events]
    assert "answer" in event_types
    assert "done" in event_types
    assert "session" in event_types


@pytest.mark.asyncio
async def test_multimodal_chat_image_base64(test_app_mm):
    """Multimodal chat with image_base64 should include image context."""
    tiny_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )

    response = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "text": "这道题怎么解？",
            "image_base64": f"data:image/png;base64,{tiny_png_base64}",
            "template": "basic",
            "max_steps": 5,
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert "answer" in [e["type"] for e in events]


@pytest.mark.asyncio
async def test_multimodal_chat_with_audio(test_app_mm):
    """Multimodal chat with audio should transcribe via ASR first."""
    audio_b64 = base64.b64encode(b"\x00" * 5000).decode("utf-8")

    response = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "audio_base64": audio_b64,
            "template": "basic",
            "max_steps": 5,
        },
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    # Should have asr_result event
    event_types = [e["type"] for e in events]
    assert "asr_result" in event_types


@pytest.mark.asyncio
async def test_multimodal_chat_empty_request(test_app_mm):
    """Request with no input should return 422."""
    response = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "template": "basic",
            "max_steps": 5,
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 5: Voice loop endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_loop(test_app_mm):
    """Voice loop should return text + audio."""
    audio_b64 = base64.b64encode(b"\x00" * 5000).decode("utf-8")

    response = await test_app_mm.post(
        "/api/v1/multimodal/voice",
        json={
            "audio_base64": audio_b64,
            "language": "zh",
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "audio_base64" in data
    assert "session_id" in data
    assert len(data["text"]) > 0


# ---------------------------------------------------------------------------
# Test 6: ASR endpoint (multipart upload)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_asr_endpoint(test_app_mm, mock_asr):
    """ASR endpoint should transcribe uploaded audio file."""
    audio_content = b"\x00" * 5000
    files = {"file": ("test.webm", io.BytesIO(audio_content), "audio/webm")}
    data = {"language": "zh"}

    response = await test_app_mm.post(
        "/api/v1/multimodal/asr",
        files=files,
        data=data,
    )
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Test 7: TTS endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_endpoint(test_app_mm, mock_tts):
    """TTS endpoint should return audio bytes."""
    response = await test_app_mm.post(
        "/api/v1/multimodal/tts",
        json={"text": "你好，这是测试", "voice": "alloy", "speed": 1.0},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0


# ---------------------------------------------------------------------------
# Test 8: Session context preservation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multimodal_session_persistence(test_app_mm):
    """Session ID should be returned and sessions list should include it."""
    # First request
    response1 = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "text": "第一次提问",
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response1.status_code == 200
    events1 = _parse_sse(response1.text)
    session_events = [e for e in events1 if e["type"] == "session"]
    assert len(session_events) > 0
    session_id = session_events[0]["data"]["multimodal_session_id"]

    # Check sessions list
    list_response = await test_app_mm.get("/api/v1/multimodal/sessions")
    assert list_response.status_code == 200
    data = list_response.json()
    assert data["total"] >= 1
    assert any(s["session_id"] == session_id for s in data["sessions"])


# ---------------------------------------------------------------------------
# Test 9: Image upload endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_image_upload(test_app_mm):
    """Image upload should return base64 data URI."""
    # Create a minimal 1x1 PNG
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    files = {"file": ("test.png", io.BytesIO(tiny_png), "image/png")}

    response = await test_app_mm.post(
        "/api/v1/multimodal/upload/image",
        files=files,
    )
    assert response.status_code == 200
    result = response.json()
    assert "data_uri" in result
    assert result["data_uri"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# SSE Parser Helper (reuse from test_agent.py pattern)
# ---------------------------------------------------------------------------

def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts."""
    events = []
    current_event = ""

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"type": current_event, "data": data})
            current_event = ""

    return events
```

- [ ] **Step 5: Add missing imports to test_multimodal.py header**

Add at the very top:

```python
import base64
import io
import json

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from main import app
```

- [ ] **Step 6: Add pytest-asyncio marker to conftest.py if needed**

Check conftest.py already has `pytest_asyncio` imported. If not, ensure it's there.

- [ ] **Step 7: Run all multimodal tests**

```bash
python -m pytest tests/test_multimodal.py -v
```

Expected: All 14+ tests pass (5 from ASR + 5 from TTS + 9 from router).

- [ ] **Step 8: Commit**

```bash
git add app/routers/multimodal.py main.py tests/conftest.py tests/test_multimodal.py
git commit -m "feat: add multimodal router with ASR, TTS, chat, voice loop, and image upload endpoints"
```

---

### Task 7: Frontend — Multimodal Learning Assistant UI

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Write the new frontend**

This is a complete rewrite of `static/index.html`. The key changes:
1. Three input modes: camera/upload, hold-to-record, text
2. Streaming text display (SSE)
3. One-click TTS playback (play button on AI responses)
4. Image preview before sending
5. Session continuity for follow-up questions

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 学习助手 — 能看·会听·会说</title>
<style>
:root {
  --bg-0: #000; --bg-1: #0a0a0a; --bg-2: #141414; --bg-3: #1a1a1a; --bg-4: #222;
  --text-1: #ededed; --text-2: #a1a1a1; --text-3: #888;
  --accent: #0070f3; --accent-green: #00b878; --accent-red: #e00;
  --radius: 10px; --radius-sm: 6px;
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font);background:var(--bg-0);color:var(--text-1);height:100vh;overflow:hidden}
a{color:var(--accent)}

/* Layout: mobile-first single column, centered */
.app{max-width:720px;margin:0 auto;height:100vh;display:flex;flex-direction:column}
.header{padding:16px 20px;border-bottom:1px solid rgba(255,255,255,.06);display:flex;align-items:center;gap:10px;flex-shrink:0}
.header .logo{font-size:16px;font-weight:600;letter-spacing:-.3px}
.header .badge{font-size:10px;background:var(--accent);color:#fff;padding:2px 8px;border-radius:10px;opacity:.8}

/* Chat area — scrollable */
.chat-area{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:12px}
.chat-area:empty::after{content:'👋 拍照提问、按住录音、或输入文字开始';display:block;text-align:center;color:var(--text-3);padding:40px 0;font-size:14px}

/* Message bubbles */
.msg{max-width:90%;padding:12px 16px;border-radius:var(--radius);line-height:1.7;font-size:14px;word-break:break-word}
.msg.user{align-self:flex-end;background:var(--accent);color:#fff;border-bottom-right-radius:2px}
.msg.ai{align-self:flex-start;background:var(--bg-2);border:1px solid rgba(255,255,255,.06);border-bottom-left-radius:2px}
.msg .meta{font-size:10px;color:var(--text-3);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.msg .content{white-space:pre-wrap}
.msg .content strong{color:var(--text-1)}
.msg .content code{font-family:monospace;background:var(--bg-4);padding:1px 5px;border-radius:3px;font-size:12.5px}

/* Image in message */
.msg .msg-image{max-width:200px;border-radius:var(--radius-sm);margin-bottom:6px}
.msg .msg-image img{width:100%;border-radius:var(--radius-sm)}

/* Play button */
.btn-play{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;background:var(--bg-3);border:1px solid rgba(255,255,255,.08);color:var(--text-2);border-radius:20px;font-size:12px;cursor:pointer;margin-top:8px;transition:all .15s}
.btn-play:hover{color:var(--text-1);border-color:rgba(255,255,255,.15)}
.btn-play.playing{background:var(--accent-green);color:#fff;border-color:var(--accent-green)}

/* Bottom bar — input area */
.bottom-bar{padding:12px 16px;border-top:1px solid rgba(255,255,255,.06);flex-shrink:0;display:flex;flex-direction:column;gap:8px}

/* Row 1: text input + send */
.input-row{display:flex;gap:8px}
.input-row textarea{flex:1;background:var(--bg-2);border:1px solid rgba(255,255,255,.08);color:var(--text-1);border-radius:var(--radius-sm);padding:10px 14px;outline:none;font-size:14px;font-family:var(--font);resize:none;min-height:44px;max-height:120px}
.input-row textarea:focus{border-color:var(--accent)}
.btn-send{padding:8px 18px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius-sm);font-size:14px;font-weight:500;cursor:pointer;white-space:nowrap;transition:opacity .15s}
.btn-send:hover{opacity:.85}
.btn-send:disabled{opacity:.4;cursor:not-allowed}

/* Row 2: action buttons */
.action-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.btn-action{display:flex;align-items:center;gap:5px;padding:6px 14px;background:var(--bg-2);border:1px solid rgba(255,255,255,.06);color:var(--text-2);border-radius:20px;font-size:12px;cursor:pointer;transition:all .15s}
.btn-action:hover{color:var(--text-1);border-color:rgba(255,255,255,.12)}
.btn-action.active{background:var(--accent-green);color:#fff;border-color:var(--accent-green)}
.btn-action .icon{font-size:15px}

/* Recording indicator */
.recording-indicator{display:none;align-items:center;gap:6px;margin-left:8px}
.recording-indicator .pulse{width:10px;height:10px;border-radius:50%;background:var(--accent-red);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}

/* Image preview chip */
.image-preview{display:none;align-items:center;gap:6px;padding:4px 10px;background:var(--bg-3);border-radius:20px;font-size:12px;color:var(--text-2)}
.image-preview .remove{color:var(--accent-red);cursor:pointer;font-weight:700;font-size:14px}

/* Loading spinner */
.loading{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.15);border-top:2px solid var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Toast */
.toast-container{position:fixed;top:16px;right:16px;z-index:999;display:flex;flex-direction:column;gap:6px}
.toast{padding:10px 16px;border-radius:var(--radius-sm);font-size:13px;max-width:300px;animation:slideIn .3s}
.toast.info{background:rgba(0,112,243,.15);color:var(--accent);border:1px solid rgba(0,112,243,.2)}
.toast.error{background:rgba(224,0,0,.12);color:var(--accent-red);border:1px solid rgba(224,0,0,.2)}
.toast.success{background:rgba(0,184,120,.12);color:var(--accent-green);border:1px solid rgba(0,184,120,.2)}
@keyframes slideIn{from{transform:translateX(40px);opacity:0}to{transform:translateX(0);opacity:1}}

/* Status bar */
.status-bar{text-align:center;font-size:10px;color:var(--text-3);padding:4px 0;display:none}

/* Hidden file inputs */
.hidden-input{display:none}

/* Mobile tweaks */
@media(max-width:480px){.btn-action{font-size:11px;padding:6px 10px}}
</style>
</head>
<body>

<div class="toast-container" id="toasts"></div>

<div class="app">
  <div class="header">
    <span class="logo">🤖 AI 学习助手</span>
    <span class="badge">多模态</span>
  </div>

  <div class="chat-area" id="chat-area"></div>

  <div class="bottom-bar">
    <div class="status-bar" id="status-bar"></div>
    <div class="input-row">
      <textarea id="text-input" placeholder="输入问题、拍照上传或按住录音..." rows="1"></textarea>
      <button class="btn-send" id="btn-send" onclick="sendMessage()">发送</button>
    </div>
    <div class="action-row">
      <!-- Camera -->
      <button class="btn-action" onclick="openCamera()" title="拍照">
        <span class="icon">📷</span> 拍照
      </button>
      <!-- File upload -->
      <label class="btn-action" title="上传图片">
        <span class="icon">🖼</span> 上传
        <input type="file" accept="image/*" class="hidden-input" onchange="handleImageUpload(this)">
      </label>

      <!-- Voice record -->
      <button class="btn-action" id="btn-record" onmousedown="startRecording()" onmouseup="stopRecording()" onmouseleave="stopRecording()" ontouchstart="startRecording()" ontouchend="stopRecording()" title="按住录音">
        <span class="icon">🎤</span> 录音
      </button>

      <!-- Recording indicator -->
      <div class="recording-indicator" id="rec-indicator">
        <div class="pulse"></div>
        <span style="font-size:11px;color:var(--accent-red)">录音中...</span>
      </div>

      <!-- Image preview -->
      <div class="image-preview" id="img-preview">
        📎 <span id="img-name"></span>
        <span class="remove" onclick="clearImage()">✕</span>
      </div>

      <!-- Session info -->
      <span style="font-size:10px;color:var(--text-3);margin-left:auto" id="session-indicator"></span>
    </div>
  </div>
</div>

<!-- Hidden elements -->
<input type="file" accept="image/*" capture="environment" class="hidden-input" id="camera-input" onchange="handleImageUpload(this)">
<audio id="tts-audio" style="display:none"></audio>

<script>
// ============================================================================
// State
// ============================================================================
const API = '/api/v1/multimodal';
let currentSessionId = null;
let currentImageBase64 = null;
let mediaRecorder = null;
let audioChunks = [];
let ttsAudio = document.getElementById('tts-audio');

// ============================================================================
// Toast
// ============================================================================
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ============================================================================
// Escape HTML
// ============================================================================
function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function md(text) {
  if (!text) return '';
  return esc(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// ============================================================================
// Image handling
// ============================================================================
function openCamera() {
  document.getElementById('camera-input').click();
}

function handleImageUpload(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    currentImageBase64 = e.target.result;
    document.getElementById('img-preview').style.display = 'flex';
    document.getElementById('img-name').textContent = file.name;
    toast('图片已加载，AI 将直接查看图片内容', 'info');
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentImageBase64 = null;
  document.getElementById('img-preview').style.display = 'none';
  document.getElementById('camera-input').value = '';
}

// ============================================================================
// Voice recording (MediaRecorder API)
// ============================================================================
async function startRecording() {
  const btn = document.getElementById('btn-record');
  if (btn.classList.contains('active')) return;
  audioChunks = [];

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      processRecording();
    };
    mediaRecorder.start();
    btn.classList.add('active');
    document.getElementById('rec-indicator').style.display = 'flex';
    toast('录音中，松开发送...', 'info');
  } catch (e) {
    toast('无法访问麦克风: ' + e.message, 'error');
  }
}

function stopRecording() {
  const btn = document.getElementById('btn-record');
  if (!btn.classList.contains('active')) return;
  btn.classList.remove('active');
  document.getElementById('rec-indicator').style.display = 'none';
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
}

async function processRecording() {
  if (!audioChunks.length) return;
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const base64 = await blobToBase64(blob);

  // Show user message
  addMessage('user', '🎤 语音输入', null);
  setStatus('正在识别语音...');

  try {
    const response = await fetch(API + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_base64: base64,
        template: 'basic',
        max_steps: 10,
        session_id: currentSessionId,
        image_base64: currentImageBase64,
      }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || '请求失败');
    await handleSSEStream(response);
  } catch (e) {
    toast('语音处理失败: ' + e.message, 'error');
    setStatus('');
  }
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      resolve(result.split(',')[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// ============================================================================
// Text send
// ============================================================================
async function sendMessage() {
  const input = document.getElementById('text-input');
  const text = input.value.trim();
  if (!text && !currentImageBase64) { toast('请输入问题或上传图片', 'error'); return; }

  // Show user message
  if (text) {
    let preview = '';
    if (currentImageBase64) {
      preview = `<div class="msg-image"><img src="${currentImageBase64}" alt="uploaded"></div>`;
    }
    addMessage('user', text, preview);
  }

  input.value = '';
  input.style.height = 'auto';
  const btn = document.getElementById('btn-send');
  btn.disabled = true;
  btn.textContent = '思考中...';

  try {
    const body = {
      template: 'basic',
      max_steps: 10,
      session_id: currentSessionId,
    };
    if (text) body.text = text;
    if (currentImageBase64) body.image_base64 = currentImageBase64;

    const response = await fetch(API + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error((await response.json()).detail || '请求失败');
    await handleSSEStream(response);
  } catch (e) {
    toast('请求失败: ' + e.message, 'error');
  }

  btn.disabled = false;
  btn.textContent = '发送';
}

// ============================================================================
// SSE Stream handling
// ============================================================================
async function handleSSEStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = '';
  let aiBubble = null;
  let answerText = '';
  let asrText = '';

  setStatus('AI 正在思考...');

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const dataStr = line.slice(6);
        try {
          const data = JSON.parse(dataStr);

          if (currentEvent === 'asr_result') {
            asrText = data.text || '';
            setStatus('识别结果: ' + asrText);
          }

          else if (currentEvent === 'thought') {
            if (!aiBubble) {
              aiBubble = addMessage('ai', '', null);
            }
            const thoughtDiv = aiBubble.querySelector('.thought-text');
            if (thoughtDiv) {
              thoughtDiv.textContent += (data.thought || '') + '\n';
            }
          }

          else if (currentEvent === 'answer') {
            if (aiBubble) {
              // Remove loading indicator, show answer
              const loadingEl = aiBubble.querySelector('.loading-wrap');
              if (loadingEl) loadingEl.remove();
            }
            answerText = data.answer || '';
            // Replace ai bubble content with final answer
            if (aiBubble) {
              aiBubble.querySelector('.content').innerHTML = md(answerText);
              // Add play button
              addPlayButton(aiBubble, answerText);
            } else {
              aiBubble = addMessage('ai', answerText, null);
              addPlayButton(aiBubble, answerText);
            }
            setStatus('');
          }

          else if (currentEvent === 'session') {
            currentSessionId = data.multimodal_session_id;
            document.getElementById('session-indicator').textContent =
              '会话: ' + currentSessionId.slice(0, 8) + '...';
          }

          else if (currentEvent === 'error' || currentEvent === 'asr_error') {
            toast(data.error || '处理出错', 'error');
            setStatus('');
          }
        } catch (e) { /* ignore parse errors */ }
        currentEvent = '';
      }
    }
  }

  setStatus('');
}

// ============================================================================
// UI helpers
// ============================================================================
function addMessage(role, text, imageHtml) {
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = 'msg ' + role;

  let html = '';
  if (imageHtml) html += imageHtml;
  if (role === 'ai') {
    html += '<div class="meta">🤖 AI 助手</div>';
    html += '<div class="content">' + md(text) + '</div>';
    html += '<div class="loading-wrap"><span class="loading"></span> 思考中...</div>';
  } else {
    html += '<div class="meta">👤 你</div>';
    html += '<div class="content">' + esc(text) + '</div>';
  }
  html += '<div class="thought-text" style="display:none;margin-top:6px;padding:6px 10px;background:var(--bg-0);border-radius:var(--radius-sm);font-size:11px;color:var(--text-3);line-height:1.5;max-height:120px;overflow-y:auto"></div>';

  div.innerHTML = html;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
  return div;
}

function addPlayButton(bubble, text) {
  if (!text || text.length < 3) return;
  const btn = document.createElement('button');
  btn.className = 'btn-play';
  btn.innerHTML = '🔊 播放语音';
  btn.onclick = () => playTTS(text, btn);
  bubble.appendChild(btn);
}

async function playTTS(text, btn) {
  if (btn.classList.contains('playing')) {
    ttsAudio.pause();
    btn.classList.remove('playing');
    btn.innerHTML = '🔊 播放语音';
    return;
  }

  btn.classList.add('playing');
  btn.innerHTML = '<span class="loading"></span> 合成中...';

  try {
    const response = await fetch(API + '/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text.slice(0, 2000), // Truncate for TTS limit
        voice: 'alloy',
        speed: 1.0,
      }),
    });
    if (!response.ok) throw new Error('TTS failed');

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    ttsAudio.src = url;
    ttsAudio.onended = () => {
      btn.classList.remove('playing');
      btn.innerHTML = '🔊 播放语音';
      URL.revokeObjectURL(url);
    };
    ttsAudio.onerror = () => {
      toast('语音播放失败', 'error');
      btn.classList.remove('playing');
      btn.innerHTML = '🔊 播放语音';
    };
    await ttsAudio.play();
    btn.classList.add('playing');
    btn.innerHTML = '⏸ 播放中...';
  } catch (e) {
    toast('语音合成失败: ' + e.message, 'error');
    btn.classList.remove('playing');
    btn.innerHTML = '🔊 播放语音';
  }
}

function setStatus(msg) {
  const bar = document.getElementById('status-bar');
  bar.textContent = msg;
  bar.style.display = msg ? 'block' : 'none';
}

// ============================================================================
// Enter key to send
// ============================================================================
document.getElementById('text-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ============================================================================
// Init
// ============================================================================
document.getElementById('session-indicator').textContent = '新会话';
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: rewrite frontend for multimodal interaction (camera, voice, TTS playback)"
```

---

### Task 8: Final Integration & Verification

**Files:**
- No file changes — verification only

- [ ] **Step 1: Run all existing tests to verify no regressions**

```bash
python -m pytest tests/ -v
```

Expected: All existing 12 agent tests + new 14+ multimodal tests pass.

- [ ] **Step 2: Start the server and verify health**

```bash
python -m uvicorn main:app --reload --port 8000
curl http://localhost:8000/health
```

Expected: `{"status": "ok"}`

- [ ] **Step 3: Test multimodal chat endpoint (curl)**

```bash
curl -X POST http://localhost:8000/api/v1/multimodal/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "解释一下什么是二次函数", "template": "basic", "max_steps": 3}'
```

Expected: SSE stream with thought → answer → done → session events.

- [ ] **Step 4: Test TTS endpoint (curl)**

```bash
curl -X POST http://localhost:8000/api/v1/multimodal/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "voice": "alloy"}' \
  --output test.mp3
```

Expected: `test.mp3` created with audio content.

- [ ] **Step 5: Open frontend in browser**

Navigate to `http://localhost:8000/` and verify:
- Text input + send works
- Camera/upload works
- Voice recording works (MediaRecorder)
- TTS playback button appears on AI responses
- Session ID persists across turns

- [ ] **Step 6: Commit final state**

```bash
git status
git add -A
git commit -m "chore: final integration verification — all tests pass"
```

---

## Verification Checklist

| # | What to verify | How |
|---|---|---|
| 1 | All existing tests pass | `pytest tests/ -v` |
| 2 | New multimodal tests pass | `pytest tests/test_multimodal.py -v` |
| 3 | ASR validation works | Test with invalid audio type → 400 |
| 4 | TTS audio output valid | `curl` TTS endpoint → valid MP3 bytes |
| 5 | SSE streaming works | `curl` multimodal/chat → see SSE events |
| 6 | Session context persistence | Two requests with same session_id |
| 7 | Frontend loads | Browser at `localhost:8000` |
| 8 | Voice recording works | Hold record button in browser |
| 9 | Image upload works | Upload image in frontend |
| 10 | TTS playback works | Click play button on AI response |

---

## Architecture Summary

```
                          Frontend (static/index.html)
                          ┌─────────────────────────┐
                          │  📷 Camera/Upload        │
                          │  🎤 Hold-to-Record       │
                          │  ⌨️  Text Input          │
                          │  🔊 TTS Playback         │
                          │  💬 SSE Streaming Text   │
                          └───────────┬─────────────┘
                                      │ HTTP + SSE
                                      ▼
                          FastAPI Server (main.py)
                          ┌─────────────────────────┐
                          │  /api/v1/multimodal/*   │
                          │    ├─ /chat  (SSE)       │
                          │    ├─ /voice (JSON)      │
                          │    ├─ /asr   (multipart) │
                          │    ├─ /tts   (JSON→MP3)  │
                          │    └─ /upload/image      │
                          └───────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     app/services/asr.py    app/services/multimodal_chat.py   app/services/tts.py
     (Whisper / GLM-ASR)    (Unified chat + context mgmt)     (tts-1 / CogTTS)
              │                       │                       │
              │                       ▼                       │
              │              app/services/agent.py             │
              │              (ReAct loop + SSE)                │
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                              AsyncOpenAI Clients
                         (Zhipu BigModel / OpenAI API)
```

## Key Design Decisions

1. **Images go directly to the multimodal model** — no lossy "vision tool → text summary" step. The existing Week 4 agent already handles this correctly with `image_url`/`image_base64` in structured content.

2. **Audio goes through ASR → text** — voice is transcribed first, then the text enters the LLM. The transcription is shown to the user (via `asr_result` SSE event) for transparency.

3. **TTS is explicit, not automatic** — the user clicks "播放语音" to hear the AI's answer. This avoids autoplay restrictions and unwanted audio.

4. **Session context preserves images** — when a user uploads an image in turn 1 and asks a follow-up in turn 2 (without re-uploading), the session retains the last image URL/base64 and includes it in subsequent LLM calls.

5. **Provider fallbacks built in** — ASR and TTS both try Zhipu first (matching the existing LLM provider), then fall back to OpenAI. This uses the same API key where possible.

6. **Existing agent unchanged** — the ReAct loop, tool calling, and prompt templates from Week 4 are reused without modification. The multimodal layer is a facade on top.
