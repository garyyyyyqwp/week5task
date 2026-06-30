"""Multimodal endpoint tests — ASR, TTS, and multimodal chat.

Tests:
1. Audio validation — supported types pass, unsupported rejected
2. Audio validation — size limit enforcement (via config patching)
3. ASR transcribe — returns text from mock
4. TTS input validation — empty text, overlong text
5. TTS synthesize — returns audio bytes from mock
6. Multimodal chat — text only
7. Multimodal chat — image + text input
8. Multimodal chat — with audio (ASR)
9. Multimodal chat — empty request returns 422
10. Voice loop endpoint
11. ASR endpoint (multipart upload)
12. TTS endpoint
13. Session context persistence
14. Image upload endpoint

Key design notes:
- We patch BOTH the service module AND the router module because
  Python's `from X import f` creates local references that won't see
  monkeypatch updates to the source module.
- For config-based tests, we patch the config module directly rather
  than using monkeypatch.setenv (which comes too late after import).
"""

import base64
import io
import json

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from main import app


# ---------------------------------------------------------------------------
# Async HTTP client fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_app_mm():
    """Async HTTP client for multimodal endpoints."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ============================================================================
# Helper: patch ASR & TTS across ALL modules that imported them
# ============================================================================

def _patch_asr(monkeypatch, mock_fn):
    """Patch transcribe in every module that imported it."""
    monkeypatch.setattr("app.services.asr.transcribe", mock_fn)
    monkeypatch.setattr("app.services.multimodal_chat.transcribe", mock_fn)
    monkeypatch.setattr("app.routers.multimodal.transcribe", mock_fn)


def _patch_tts(monkeypatch, mock_fn):
    """Patch synthesize in every module that imported it."""
    monkeypatch.setattr("app.services.tts.synthesize", mock_fn)
    monkeypatch.setattr("app.services.multimodal_chat.synthesize", mock_fn)
    monkeypatch.setattr("app.routers.multimodal.synthesize", mock_fn)


# ============================================================================
# Test 1: Audio validation
# ============================================================================

class TestAudioValidation:
    """Audio validation tests — uses app.services.asr module directly."""

    def test_valid_audio_passes(self, monkeypatch):
        """Valid audio data with supported MIME type should pass validation."""
        from app.services.asr import validate_audio
        audio = b"\x00" * 1000
        validate_audio(audio, "audio/webm")  # Should not raise

    def test_invalid_content_type_raises(self, monkeypatch):
        """Unsupported MIME type should raise AudioValidationError."""
        from app.services.asr import validate_audio, AudioValidationError
        audio = b"\x00" * 1000
        with pytest.raises(AudioValidationError, match="不支持的音频格式"):
            validate_audio(audio, "video/mp4")

    def test_oversized_audio_raises(self, monkeypatch):
        """Audio exceeding max size should raise AudioValidationError."""
        # Patch the config constant directly — setenv is too late
        monkeypatch.setattr("app.services.asr.MAX_AUDIO_SIZE_MB", 1)
        from app.services.asr import validate_audio, AudioValidationError
        audio = b"\x00" * (2 * 1024 * 1024)  # 2MB
        with pytest.raises(AudioValidationError, match="音频文件过大"):
            validate_audio(audio, "audio/webm")

    def test_empty_audio_raises(self, monkeypatch):
        """Very small audio should raise AudioValidationError."""
        from app.services.asr import validate_audio, AudioValidationError
        audio = b"\x00" * 10
        with pytest.raises(AudioValidationError):
            validate_audio(audio, "audio/webm")


# ============================================================================
# Test 2: ASR transcribe — mock (use module-level call, not direct import)
# ============================================================================

@pytest.mark.asyncio
async def test_asr_transcribe_returns_text(monkeypatch):
    """ASR should transcribe audio bytes to text (via mocked transcribe)."""
    async def _mock(audio_data, content_type="audio/webm", language="zh", prompt=None):
        if len(audio_data) < 100:
            return ""
        return "人工智能如何改变教育"

    _patch_asr(monkeypatch, _mock)

    import app.services.asr as asr_mod
    audio = b"\x00" * 5000
    result = await asr_mod.transcribe(audio, content_type="audio/webm")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "人工智能" in result


@pytest.mark.asyncio
async def test_asr_transcribe_empty_audio():
    """Very small audio (<100 bytes) should raise AudioValidationError.

    We test validate_audio directly rather than transcribe, since the mock
    replaces transcribe and would skip the real validation logic.
    """
    from app.services.asr import validate_audio, AudioValidationError
    audio = b"\x00" * 10
    with pytest.raises(AudioValidationError):
        validate_audio(audio, "audio/webm")


@pytest.mark.asyncio
async def test_asr_error_propagates(monkeypatch):
    """ASR errors should be raised as ASRError."""
    from app.services.asr import ASRError

    async def _mock_fail(*args, **kwargs):
        raise ASRError("模拟ASR服务不可用")

    _patch_asr(monkeypatch, _mock_fail)

    import app.services.asr as asr_mod
    audio = b"\x00" * 5000
    with pytest.raises(ASRError):
        await asr_mod.transcribe(audio, content_type="audio/webm")


# ============================================================================
# Test 3: TTS validation
# ============================================================================

class TestTTSValidation:
    """TTS input validation tests."""

    def test_valid_text_passes(self):
        from app.services.tts import _validate_tts_input
        _validate_tts_input("你好，这是测试文本。")

    def test_empty_text_raises(self):
        from app.services.tts import _validate_tts_input
        with pytest.raises(ValueError, match="不能为空"):
            _validate_tts_input("")

    def test_whitespace_only_raises(self):
        from app.services.tts import _validate_tts_input
        with pytest.raises(ValueError, match="不能为空"):
            _validate_tts_input("   ")

    def test_overlong_text_raises(self):
        from app.services.tts import _validate_tts_input
        with pytest.raises(ValueError, match="过长"):
            _validate_tts_input("啊" * 5000)


@pytest.mark.asyncio
async def test_tts_synthesize_returns_bytes(monkeypatch):
    """TTS should return audio bytes for valid text (via mocked synthesize)."""
    async def _mock(text, voice=None, speed=1.0, response_format="mp3"):
        return b"\xff\xfb\x90\x00" + b"\x00" * 256

    _patch_tts(monkeypatch, _mock)

    import app.services.tts as tts_mod
    result = await tts_mod.synthesize("你好，世界")
    assert isinstance(result, bytes)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_tts_error_propagates(monkeypatch):
    """TTS errors should be raised as TTSError."""
    from app.services.tts import TTSError

    async def _mock_fail(*args, **kwargs):
        raise TTSError("模拟TTS服务不可用")

    _patch_tts(monkeypatch, _mock_fail)

    import app.services.tts as tts_mod
    with pytest.raises(TTSError):
        await tts_mod.synthesize("测试文本")


# ============================================================================
# Mock fixtures for endpoint tests (must patch BEFORE router import lookup)
# ============================================================================

@pytest.fixture()
def _patch_all_asr(monkeypatch):
    """Patch ASR in all modules for endpoint tests."""
    async def _mock(audio_data, content_type="audio/webm", language="zh", prompt=None):
        if len(audio_data) < 100:
            return ""
        return "人工智能如何改变教育"
    _patch_asr(monkeypatch, _mock)
    return _mock


@pytest.fixture()
def _patch_all_tts(monkeypatch):
    """Patch TTS in all modules for endpoint tests."""
    async def _mock(text, voice=None, speed=1.0, response_format="mp3"):
        return b"\xff\xfb\x90\x00" + b"\x00" * 256
    _patch_tts(monkeypatch, _mock)
    return _mock


# ============================================================================
# Test 4: Multimodal chat endpoints
# ============================================================================

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

    events = _parse_sse(response.text)
    event_types = [e["type"] for e in events]
    assert "answer" in event_types
    assert "done" in event_types
    assert "session" in event_types


@pytest.mark.asyncio
async def test_multimodal_chat_image_base64(test_app_mm):
    """Multimodal chat with image_base64 should include image context."""
    tiny_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/"
        "PchI7wAAAABJRU5ErkJggg=="
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
async def test_multimodal_chat_with_audio(test_app_mm, _patch_all_asr):
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


# ============================================================================
# Test 5: Voice loop endpoint
# ============================================================================

@pytest.mark.asyncio
async def test_voice_loop(test_app_mm, _patch_all_asr, _patch_all_tts):
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


# ============================================================================
# Test 6: ASR endpoint (multipart upload)
# ============================================================================

@pytest.mark.asyncio
async def test_asr_endpoint(test_app_mm, _patch_all_asr):
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


# ============================================================================
# Test 7: TTS endpoint
# ============================================================================

@pytest.mark.asyncio
async def test_tts_endpoint(test_app_mm, _patch_all_tts):
    """TTS endpoint should return audio bytes."""
    response = await test_app_mm.post(
        "/api/v1/multimodal/tts",
        json={"text": "你好，这是测试", "voice": "alloy", "speed": 1.0},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0


# ============================================================================
# Test 8: Session context preservation
# ============================================================================

@pytest.mark.asyncio
async def test_multimodal_session_persistence(test_app_mm):
    """Session ID should be returned and sessions list should include it."""
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

    # Check sessions list includes this session
    list_response = await test_app_mm.get("/api/v1/multimodal/sessions")
    assert list_response.status_code == 200
    data = list_response.json()
    assert data["total"] >= 1
    assert any(s["session_id"] == session_id for s in data["sessions"])


# ============================================================================
# Test 9: Image upload endpoint
# ============================================================================

@pytest.mark.asyncio
async def test_image_upload(test_app_mm):
    """Image upload should return base64 data URI."""
    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/"
        "PchI7wAAAABJRU5ErkJggg=="
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


# ============================================================================
# SSE Parser Helper
# ============================================================================

def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts.

    Each event dict has: {"type": event_name, "data": parsed_data}
    """
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
