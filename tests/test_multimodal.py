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
from unittest.mock import AsyncMock, MagicMock

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
    """Patch synthesize (and the streaming variant) in every module that imported it."""
    monkeypatch.setattr("app.services.tts.synthesize", mock_fn)
    monkeypatch.setattr("app.services.multimodal_chat.synthesize", mock_fn)
    monkeypatch.setattr("app.routers.multimodal.synthesize", mock_fn)

    async def _mock_stream(text, voice=None, speed=1.0, response_format="mp3"):
        yield await mock_fn(text, voice=voice, speed=speed)

    monkeypatch.setattr("app.services.tts.synthesize_stream", _mock_stream)
    monkeypatch.setattr("app.routers.multimodal.synthesize_stream", _mock_stream)


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

    def test_duration_too_long_raises(self, monkeypatch):
        """Audio exceeding estimated max duration should raise AudioValidationError."""
        monkeypatch.setattr("app.services.asr.MAX_AUDIO_DURATION_SECONDS", 1)
        from app.services.asr import validate_audio, AudioValidationError
        # Create a WAV with 2 seconds of audio at 8000Hz mono 16-bit
        # = 2 * 8000 * 2 = 32000 bytes + 44 header
        sample_rate = 8000
        duration_sec = 2
        num_samples = sample_rate * duration_sec
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        wav_data = _make_wav_bytes(data_size, sample_rate)
        with pytest.raises(AudioValidationError, match="音频时长"):
            validate_audio(wav_data, "audio/wav")

    def test_duration_within_limit_passes(self, monkeypatch):
        """Audio within duration limit should pass validation."""
        monkeypatch.setattr("app.services.asr.MAX_AUDIO_DURATION_SECONDS", 5)
        from app.services.asr import validate_audio
        # Create a 3-second WAV
        sample_rate = 8000
        data_size = 3 * sample_rate * 2
        wav_data = _make_wav_bytes(data_size, sample_rate)
        validate_audio(wav_data, "audio/wav")  # Should not raise

    def test_duration_estimate_non_wav_uses_bitrate(self, monkeypatch):
        """Non-WAV formats should estimate duration from bitrate."""
        monkeypatch.setattr("app.services.asr.MAX_AUDIO_DURATION_SECONDS", 1)
        monkeypatch.setattr("app.services.asr.ESTIMATED_BITRATE_KBPS", 32)
        from app.services.asr import validate_audio, AudioValidationError
        # 32kbps * 3s = 12KB. Create a webm-sized blob > 12KB
        audio = b"\x00" * 20000  # ~20KB, should be estimated > 1s at 32kbps
        with pytest.raises(AudioValidationError, match="音频时长"):
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


@pytest.mark.asyncio
async def test_voice_loop_with_session_id(test_app_mm, _patch_all_asr, _patch_all_tts):
    """Voice loop with session_id should persist in same session."""
    audio_b64 = base64.b64encode(b"\x00" * 5000).decode("utf-8")

    # First call
    response1 = await test_app_mm.post(
        "/api/v1/multimodal/voice",
        json={
            "audio_base64": audio_b64,
            "language": "zh",
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response1.status_code == 200
    session_id = response1.json()["session_id"]

    # Second call with same session_id
    response2 = await test_app_mm.post(
        "/api/v1/multimodal/voice",
        json={
            "audio_base64": audio_b64,
            "language": "zh",
            "session_id": session_id,
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response2.status_code == 200
    # Should return same session ID when provided
    assert response2.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_voice_loop_with_image_context(test_app_mm, _patch_all_asr, _patch_all_tts):
    """Voice loop with image_base64 should pass image to agent context."""
    audio_b64 = base64.b64encode(b"\x00" * 5000).decode("utf-8")
    tiny_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/"
        "PchI7wAAAABJRU5ErkJggg=="
    )

    response = await test_app_mm.post(
        "/api/v1/multimodal/voice",
        json={
            "audio_base64": audio_b64,
            "image_base64": f"data:image/png;base64,{tiny_png_base64}",
            "language": "zh",
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "audio_base64" in data


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
    # Media type is provider-dependent (wav for zhipu, mpeg for openai)
    assert response.headers["content-type"] in ("audio/mpeg", "audio/wav")
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


def _make_wav_bytes(data_size: int, sample_rate: int = 8000,
                    num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Build a minimal valid WAV file in memory for duration testing."""
    import struct
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    riff_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16, 1, num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + b"\x00" * data_size


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


# ============================================================================
# Test 10: Multi-turn conversation context preservation
# ============================================================================

@pytest.mark.asyncio
async def test_agent_stream_with_history(monkeypatch):
    """run_agent_stream should include history messages in LLM context."""
    from app.services.agent import run_agent_stream

    # Track what messages the LLM receives
    captured_messages = []

    async def mock_create(**kwargs):
        captured_messages.append(kwargs.get("messages", []))
        # Return a simple text answer (no tool calls)
        msg = MagicMock()
        msg.content = "根据上下文，这是第二次回答。"
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

    monkeypatch.setattr("app.services.agent.get_client", lambda: mock_client)
    monkeypatch.setattr("app.services.agent.get_model", lambda: "test-model")

    history = [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
    ]

    events = []
    async for ev in run_agent_stream(
        question="追问问题",
        session_id="hist_test_001",
        template="basic",
        max_steps=3,
        history=history,
    ):
        events.append(ev)

    # Verify history was passed
    assert len(captured_messages) > 0
    msgs = captured_messages[0]
    assert msgs[0]["role"] == "system"
    # Check history messages are present
    user_roles = [m["role"] for m in msgs]
    assert "user" in user_roles
    assert "assistant" in user_roles
    # Check current question is the last user message
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) >= 2  # history user + current user

    # Verify answer event was yielded
    answer_events = [e for e in events if e["event"] == "answer"]
    assert len(answer_events) == 1


@pytest.mark.asyncio
async def test_multimodal_chat_multi_turn_context(test_app_mm):
    """Multi-turn chat should preserve conversation context across turns."""
    # Turn 1: initial question with image
    tiny_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/"
        "PchI7wAAAABJRU5ErkJggg=="
    )

    response1 = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "text": "这道题怎么解？",
            "image_base64": f"data:image/png;base64,{tiny_png_base64}",
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response1.status_code == 200
    events1 = _parse_sse(response1.text)
    session_events = [e for e in events1 if e["type"] == "session"]
    assert len(session_events) > 0
    session_id = session_events[0]["data"]["multimodal_session_id"]

    # Turn 2: follow-up question (no image) — same session
    response2 = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "text": "第二步再详细讲一遍",
            "session_id": session_id,
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response2.status_code == 200
    events2 = _parse_sse(response2.text)
    session_events2 = [e for e in events2 if e["type"] == "session"]
    assert len(session_events2) > 0
    # Should return same session ID
    assert session_events2[0]["data"]["multimodal_session_id"] == session_id

    # Turn 3: another follow-up — same session
    response3 = await test_app_mm.post(
        "/api/v1/multimodal/chat",
        json={
            "text": "还有没有更简单的方法？",
            "session_id": session_id,
            "template": "basic",
            "max_steps": 3,
        },
    )
    assert response3.status_code == 200
    events3 = _parse_sse(response3.text)
    session_events3 = [e for e in events3 if e["type"] == "session"]
    assert session_events3[0]["data"]["multimodal_session_id"] == session_id


# ============================================================================
# Test 11: Image is kept in context for EVERY agent step (regression)
# ============================================================================

@pytest.mark.asyncio
async def test_image_retained_across_all_steps(monkeypatch):
    """Regression: the image must stay in the LLM messages on step 2+.

    Previously the agent stripped image_url after step 1, so the model
    "forgot" the picture once it called a tool. This test forces a tool
    call on step 1, then a final answer on step 2, and asserts the image
    is still present in the step-2 messages.
    """
    from app.services.agent import run_agent_stream

    captured_messages = []
    call_count = {"n": 0}

    def _has_image(messages):
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                if any(item.get("type") == "image_url" for item in content):
                    return True
        return False

    async def mock_create(**kwargs):
        captured_messages.append(kwargs.get("messages", []))
        call_count["n"] += 1

        msg = MagicMock()
        if call_count["n"] == 1:
            # Step 1: request a tool call (calculator)
            msg.content = "先算一下"
            tc = MagicMock()
            tc.id = "call_1"
            tc.function.name = "calculator"
            tc.function.arguments = '{"expression": "1+1"}'
            msg.tool_calls = [tc]
        else:
            # Step 2: final answer, no tool calls
            msg.content = "答案是2"
            msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)

    monkeypatch.setattr("app.services.agent.get_client", lambda: mock_client)
    monkeypatch.setattr("app.services.agent.get_model", lambda: "test-model")

    tiny_png = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )

    async for _ in run_agent_stream(
        question="这道题的答案",
        session_id="img_persist_001",
        template="basic",
        max_steps=5,
        image_url=tiny_png,
    ):
        pass

    # At least two LLM calls happened (tool call + final answer)
    assert call_count["n"] >= 2
    # The image must be present in EVERY step's messages, including step 2+
    assert _has_image(captured_messages[0]), "image missing on step 1"
    assert _has_image(captured_messages[1]), "image was stripped on step 2"


# ============================================================================
# Test 12: History truncation caps conversation growth
# ============================================================================

def test_history_truncation_caps_length(monkeypatch):
    """_truncate_history keeps only the most recent MAX_HISTORY_TURNS turns."""
    import app.services.multimodal_chat as mm

    monkeypatch.setattr(mm, "MULTIMODAL_MAX_HISTORY_TURNS", 3)

    # Build 10 turns = 20 messages
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"问题{i}"})
        history.append({"role": "assistant", "content": f"回答{i}"})

    truncated = mm._truncate_history(history)

    # 3 turns => 6 messages, and they must be the most recent ones
    assert len(truncated) == 6
    assert truncated[0]["content"] == "问题7"
    assert truncated[-1]["content"] == "回答9"


def test_history_truncation_noop_when_short(monkeypatch):
    """Short histories are returned unchanged."""
    import app.services.multimodal_chat as mm

    monkeypatch.setattr(mm, "MULTIMODAL_MAX_HISTORY_TURNS", 20)
    history = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    assert mm._truncate_history(history) == history


# ============================================================================
# Test 13: Multimodal session persistence to disk (regression)
# ============================================================================

def test_multimodal_session_survives_memory_clear():
    """A session written to disk must reload after the in-memory store is wiped.

    Simulates a server restart: create a session with an image + history,
    drop the in-memory cache, then read it back through _get_session (which
    should load from disk).
    """
    import app.services.multimodal_chat as mm

    sid = mm._create_session()
    mm._update_session_image(sid, image_base64="data:image/png;base64,ABC")
    mm._append_to_history(sid, "user", "这道题怎么解？")
    mm._append_to_history(sid, "assistant", "第一步……")

    # Simulate restart: clear only the in-memory dict, keep the files on disk
    mm._multimodal_sessions.clear()
    assert sid not in mm._multimodal_sessions

    reloaded = mm._get_session(sid)
    assert reloaded is not None, "session did not reload from disk"
    assert reloaded["image_base64"] == "data:image/png;base64,ABC"
    assert len(reloaded["history"]) == 2
    assert reloaded["history"][0]["content"] == "这道题怎么解？"

    # Cleanup
    mm.clear_multimodal_sessions()


def test_multimodal_list_includes_disk_only_sessions():
    """list_multimodal_sessions must surface sessions that exist only on disk."""
    import app.services.multimodal_chat as mm

    mm.clear_multimodal_sessions()
    sid = mm._create_session()
    mm._append_to_history(sid, "user", "hi")
    mm._append_to_history(sid, "assistant", "hello")

    # Wipe memory only — file remains
    mm._multimodal_sessions.clear()

    listed = mm.list_multimodal_sessions()
    assert any(s["session_id"] == sid for s in listed)

    mm.clear_multimodal_sessions()
