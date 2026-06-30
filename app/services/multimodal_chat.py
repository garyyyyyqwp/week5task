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

from app.services.agent import run_agent_stream, run_agent_sync
from app.services.asr import transcribe
from app.services.tts import synthesize

logger = logging.getLogger(__name__)

# In-memory multimodal session store
_multimodal_sessions: dict[str, dict] = {}


def _create_session() -> str:
    """Create a new multimodal session and return its ID."""
    session_id = uuid.uuid4().hex[:12]
    _multimodal_sessions[session_id] = {
        "session_id": session_id,
        "history": [],
        "image_url": None,
        "image_base64": None,
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
    """List all multimodal sessions with summary info."""
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
            audio_bytes = base64.b64decode(audio_base64)
            transcribed_text = await transcribe(
                audio_bytes, content_type="audio/webm"
            )

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

    # Update image context if new image provided
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

    # Determine which image to use (URL takes precedence over base64)
    final_image = resolved_image_url or resolved_image_base64

    # --- Step 3: Build effective question ---
    effective_question = text or "请分析这张图片"

    # --- Step 4: Delegate to ReAct agent loop ---
    _append_to_history(session_id, "user", effective_question)

    async for event in run_agent_stream(
        question=effective_question,
        session_id=session_id,
        template=template,
        max_steps=max_steps,
        image_url=final_image,
    ):
        # Inject multimodal session_id into done event
        if event["event"] == "done":
            data = json.loads(event["data"])
            data["multimodal_session_id"] = session_id
            event["data"] = json.dumps(data, ensure_ascii=False)

        yield event

    # Always yield session event with multimodal session ID
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
    transcribed = await transcribe(
        audio_bytes, content_type="audio/webm", language=language
    )

    # Step 2: LLM
    from app.services.sessions import get_session as get_agent_session

    session_id = uuid.uuid4().hex[:12]
    agent_result = await run_agent_sync(
        question=transcribed,
        session_id=session_id,
        template=template,
        max_steps=max_steps,
    )

    answer_text = (
        agent_result.final_answer
        if agent_result
        else "抱歉，我无法回答这个问题。"
    )

    # Step 3: TTS (truncate long text to avoid TTS character limit)
    tts_audio = await synthesize(answer_text[:2000])
    audio_b64 = base64.b64encode(tts_audio).decode("utf-8")

    return {
        "text": answer_text,
        "audio_base64": audio_b64,
        "session_id": session_id,
    }
