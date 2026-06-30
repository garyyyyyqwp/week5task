"""Test fixtures — mirrors Week 3 conftest.py pattern with Agent-specific mocks.

Key additions over Week 3:
- Mock LLM that simulates multi-turn function calling (tool_calls responses)
- Mock tool executors that return deterministic results
- Session cleanup between tests
"""

import asyncio
import hashlib
import json
import shutil
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from main import app


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, tmp_path):
    """Set up isolated test environment with temp directories and test API keys."""
    # Temp directories
    chroma_dir = str(tmp_path / "chroma")
    session_dir = str(tmp_path / "sessions")
    image_dir = str(tmp_path / "images")

    monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)
    monkeypatch.setenv("TEXT_COLLECTION_NAME", "test_text_collection")
    monkeypatch.setenv("IMAGE_COLLECTION_NAME", "test_image_collection")
    monkeypatch.setenv("AGENT_SESSION_DIR", session_dir)
    monkeypatch.setenv("IMAGE_SAVE_DIR", image_dir)
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "512")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "50")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_TOP_M", "2")
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    monkeypatch.setenv("SEARCH_WEB_QUERY_MAX_LENGTH", "200")

    # API keys — prevent real calls
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test.example.com/v1/")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("VISION_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("VISION_BASE_URL", "https://api.test.example.com/v1/")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.test.example.com/v1/")

    # Reset VectorStore singleton
    import app.services.vector_store as vs_mod
    vs_mod._store = None

    # Reset LLM clients
    import app.services.llm as llm_mod
    llm_mod._text_client = None
    llm_mod._vision_client = None

    # Clear sessions
    import app.services.sessions as sess_mod
    sess_mod._sessions.clear()
    # Update save dir to temp
    sess_mod._SAVE_DIR = __import__('pathlib').Path(session_dir)
    sess_mod._SAVE_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # Cleanup
    vs_mod._store = None
    llm_mod._text_client = None
    llm_mod._vision_client = None


# ---------------------------------------------------------------------------
# Embedding Mocks
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_embedding(monkeypatch):
    """Mock embedding functions with deterministic hash-based vectors."""
    def _mock_embed(texts, **kwargs):
        results = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [(h[i % len(h)] / 255.0 - 0.5) for i in range(1024)]
            results.append(vec)
        return results

    def _mock_embed_single(text, **kwargs):
        return _mock_embed([text])[0]

    # Patch all embedding entry points
    monkeypatch.setattr("app.services.vector_store.embed_texts", _mock_embed)
    monkeypatch.setattr("app.services.embedding.embed_texts", _mock_embed)
    monkeypatch.setattr("app.services.embedding.embed_single", _mock_embed_single)


@pytest.fixture(autouse=True)
def mock_clip(monkeypatch):
    """Mock CLIP embedding with deterministic 512-d vectors."""
    def _mock_clip_text(text, **kwargs):
        h = hashlib.sha256(text.encode()).digest()
        return [(h[i % len(h)] / 255.0 - 0.5) for i in range(512)]

    def _mock_clip_images(paths, **kwargs):
        results = []
        for p in paths:
            h = hashlib.sha256(str(p).encode()).digest()
            vec = [(h[i % len(h)] / 255.0 - 0.5) for i in range(512)]
            results.append(vec)
        return results

    monkeypatch.setattr("app.services.clip_embedding.clip_encode_text", _mock_clip_text)
    monkeypatch.setattr("app.services.clip_embedding.clip_encode_images", _mock_clip_images)
    # Force not-stub mode
    monkeypatch.setattr("app.services.clip_embedding._is_stub_mode", lambda: False)


# ---------------------------------------------------------------------------
# LLM Mock — Multi-turn Function Calling
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_agent_llm(monkeypatch):
    """Mock LLM that simulates multi-turn function calling for agent tests.

    Returns tool_calls for the first N turns, then a text final answer.
    Also mocks execute_tool so tool executions return deterministic results.
    """
    call_count = [0]  # Mutable counter

    # Define a sequence of responses: each entry is either a tool_call or a text response
    responses = [
        # Turn 1: search_knowledge_base
        _make_tool_call_response(
            thought="我需要先搜索知识库来回答这个问题。",
            tool_name="search_knowledge_base",
            tool_args={"query": "测试查询"},
            tool_call_id="call_001",
        ),
        # Turn 2: get_current_time
        _make_tool_call_response(
            thought="我需要获取当前时间。",
            tool_name="get_current_time",
            tool_args={"timezone_name": "Asia/Shanghai"},
            tool_call_id="call_002",
        ),
        # Turn 3: calculator
        _make_tool_call_response(
            thought="现在我可以计算结果了。",
            tool_name="calculator",
            tool_args={"expression": "2026 - 2017"},
            tool_call_id="call_003",
        ),
        # Turn 4: final answer
        _make_text_response("根据搜索结果，Transformer论文于2017年提出，距今约9年。"),
    ]

    async def _mock_create(**kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    # Mock execute_tool to return deterministic results without real service calls
    async def _mock_execute_tool(tool_name: str, tool_args: dict) -> str:
        mock_results = {
            "search_knowledge_base": "[文本结果1] (来源: test.pdf, 相关度: 0.950)\nTransformer论文于2017年由Vaswani等人提出。",
            "search_web": "[网络搜索结果]\n查询: 测试\n深度学习是机器学习的重要分支。",
            "calculator": "[计算结果]\n表达式: test\n结果: 9",
            "get_current_time": "[当前时间]\n日期: 2026年06月27日\n时间: 14:30:00\n时区: Asia/Shanghai",
        }
        return mock_results.get(tool_name, f"工具 {tool_name} 的模拟结果")

    # Patch the AsyncOpenAI client's create method
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)

    monkeypatch.setattr("app.services.llm._text_client", mock_client)
    monkeypatch.setattr("app.services.llm.get_client", lambda: mock_client)

    # Also patch vision client
    mock_vision = MagicMock()
    monkeypatch.setattr("app.services.llm._vision_client", mock_vision)
    monkeypatch.setattr("app.services.llm.get_vision_client", lambda: mock_vision)

    # Patch execute_tool in the agent module
    monkeypatch.setattr("app.services.agent.execute_tool", _mock_execute_tool)

    return call_count


@pytest.fixture()
def mock_simple_llm(monkeypatch):
    """Mock LLM that returns a direct text answer (no tool calls)."""

    async def _mock_create(**kwargs):
        return _make_text_response("这是一个简单的回答。")

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)

    monkeypatch.setattr("app.services.llm._text_client", mock_client)
    monkeypatch.setattr("app.services.llm.get_client", lambda: mock_client)

    mock_vision = MagicMock()
    monkeypatch.setattr("app.services.llm._vision_client", mock_vision)
    monkeypatch.setattr("app.services.llm.get_vision_client", lambda: mock_vision)


@pytest.fixture()
def mock_max_steps_llm(monkeypatch):
    """Mock LLM that always returns tool_calls, never a final answer.
    Used to test max steps limit. Also mocks execute_tool."""

    call_count = [0]

    async def _mock_create(**kwargs):
        call_count[0] += 1
        return _make_tool_call_response(
            thought=f"继续调用工具 (第{call_count[0]}次)...",
            tool_name="search_knowledge_base",
            tool_args={"query": f"查询{call_count[0]}"},
            tool_call_id=f"call_{call_count[0]:03d}",
        )

    # Mock execute_tool
    async def _mock_execute_tool(tool_name: str, tool_args: dict) -> str:
        return f"工具 {tool_name} 的模拟结果"

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)

    monkeypatch.setattr("app.services.llm._text_client", mock_client)
    monkeypatch.setattr("app.services.llm.get_client", lambda: mock_client)

    mock_vision = MagicMock()
    monkeypatch.setattr("app.services.llm._vision_client", mock_vision)
    monkeypatch.setattr("app.services.llm.get_vision_client", lambda: mock_vision)

    # Patch execute_tool in the agent module
    monkeypatch.setattr("app.services.agent.execute_tool", _mock_execute_tool)

    return call_count


# ---------------------------------------------------------------------------
# Test App Client
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_app():
    """Async HTTP client for testing FastAPI endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers — Build mock LLM responses
# ---------------------------------------------------------------------------

def _make_tool_call_response(thought: str, tool_name: str, tool_args: dict, tool_call_id: str):
    """Build a mock LLM response with tool_calls."""
    # Build function mock with actual string values for .name and .arguments
    fn_mock = MagicMock()
    fn_mock.name = tool_name
    fn_mock.arguments = json.dumps(tool_args, ensure_ascii=False)

    tc_mock = MagicMock()
    tc_mock.id = tool_call_id
    tc_mock.function = fn_mock

    message = MagicMock()
    message.content = thought
    message.tool_calls = [tc_mock]

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_text_response(text: str):
    """Build a mock LLM response with plain text (no tool calls)."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


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
        return b"\xff\xfb\x90\x00" + b"\x00" * 256

    async def _mock_synthesize_streaming(
        text: str,
        voice: str | None = None,
    ) -> bytes:
        return b"\xff\xfb\x90\x00" + b"\x00" * 256

    monkeypatch.setattr("app.services.tts.synthesize", _mock_synthesize)
    monkeypatch.setattr(
        "app.services.tts.synthesize_streaming", _mock_synthesize_streaming
    )
    return _mock_synthesize


@pytest.fixture()
def mock_tts_error(monkeypatch):
    """Mock TTS to simulate API failure."""

    async def _mock_fail(*args, **kwargs):
        from app.services.tts import TTSError
        raise TTSError("模拟TTS服务不可用")

    monkeypatch.setattr("app.services.tts.synthesize", _mock_fail)


# ---------------------------------------------------------------------------
# Multimodal Mock Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_multimodal_agent(monkeypatch):
    """Mock run_agent_stream for multimodal chat tests to avoid real LLM calls.

    Also mocks the voice loop and clears sessions between tests.
    """

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
        yield {
            "event": "thought",
            "data": '{"step": 1, "thought": "分析图片中的题目..."}',
        }
        yield {
            "event": "answer",
            "data": '{"answer": "这是一道几何题。连接AB两点，根据勾股定理...", "hit_max_steps": false}',
        }
        yield {
            "event": "done",
            "data": '{"session_id": "m_test123", "total_steps": 1, "template": "basic", "hit_max_steps": false, "multimodal_session_id": "m_test123"}',
        }
        yield {
            "event": "session",
            "data": '{"multimodal_session_id": "m_test123"}',
        }

    monkeypatch.setattr(
        "app.services.multimodal_chat.run_multimodal_chat_stream",
        _mock_multimodal_stream,
    )

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
    """Clear multimodal sessions before and after each test."""
    from app.services.multimodal_chat import clear_multimodal_sessions

    clear_multimodal_sessions()
    yield
    clear_multimodal_sessions()
