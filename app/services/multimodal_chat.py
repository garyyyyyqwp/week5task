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
from pathlib import Path
from typing import AsyncIterator

from app.services.agent import run_agent_stream, run_agent_sync
from app.services.asr import transcribe, _detect_audio_format
from app.services.tts import synthesize
from app.utils.config import (
    MULTIMODAL_MAX_HISTORY_TURNS,
    MULTIMODAL_SESSION_DIR,
)

logger = logging.getLogger(__name__)

# In-memory multimodal session store, backed by JSON files on disk so
# conversations (including the image context) survive a server restart —
# matching the durability of agent sessions in app.services.sessions.
_multimodal_sessions: dict[str, dict] = {}
_MM_SAVE_DIR = Path(MULTIMODAL_SESSION_DIR)
_MM_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return _MM_SAVE_DIR / f"{session_id}.json"


def _persist_session(session_id: str) -> None:
    """Write a session to disk. Best-effort — logs on failure."""
    session = _multimodal_sessions.get(session_id)
    if not session:
        return
    try:
        with open(_session_path(session_id), "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to persist multimodal session %s: %s", session_id, e)


def _load_session_from_disk(session_id: str) -> dict | None:
    """Load a session from disk into memory, or return None if absent/corrupt."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)
        _multimodal_sessions[session_id] = session
        return session
    except Exception as e:
        logger.error("Failed to load multimodal session %s: %s", session_id, e)
        return None


def _truncate_history(history: list[dict]) -> list[dict]:
    """Keep only the most recent N turns of conversation history.

    A "turn" is one user message + one assistant reply (2 messages). Capping
    history prevents unbounded context growth (and cost) on long sessions
    while preserving recent context. Older turns are dropped from the front.
    """
    max_messages = MULTIMODAL_MAX_HISTORY_TURNS * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


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
    _persist_session(session_id)
    return session_id


def _get_session(session_id: str) -> dict | None:
    """Get a multimodal session by ID, loading from disk on a memory miss."""
    session = _multimodal_sessions.get(session_id)
    if session is not None:
        return session
    return _load_session_from_disk(session_id)


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
        _persist_session(session_id)


def _append_to_history(
    session_id: str,
    role: str,
    content: list[dict] | str,
) -> None:
    """Append a message to the session conversation history."""
    session = _multimodal_sessions.get(session_id)
    if session:
        session["history"].append({"role": role, "content": content})
        _persist_session(session_id)


def _summarize(s: dict) -> dict:
    return {
        "session_id": s["session_id"],
        "turns": len(s["history"]) // 2,
        "has_image": bool(s.get("image_url") or s.get("image_base64")),
        "created_at": s.get("created_at", ""),
    }


def list_multimodal_sessions() -> list[dict]:
    """List all multimodal sessions (in-memory + on-disk), newest first."""
    summaries: dict[str, dict] = {
        sid: _summarize(s) for sid, s in _multimodal_sessions.items()
    }

    # Merge in any sessions that exist only on disk (e.g. after a restart)
    for path in _MM_SAVE_DIR.glob("*.json"):
        sid = path.stem
        if sid in summaries:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                summaries[sid] = _summarize(json.load(f))
        except Exception:
            continue

    return sorted(
        summaries.values(),
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )


def clear_multimodal_sessions() -> None:
    """Clear all sessions from memory and disk (for testing)."""
    _multimodal_sessions.clear()
    for path in _MM_SAVE_DIR.glob("*.json"):
        try:
            path.unlink()
        except Exception:
            pass


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
            detected_mime = _detect_audio_format(audio_bytes)
            transcribed_text = await transcribe(
                audio_bytes, content_type=detected_mime
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

    # --- Step 3: Build conversation history from session (capped) ---
    session = _get_session(session_id)
    conversation_history = (
        _truncate_history(list(session["history"])) if session else []
    )

    # --- Step 4: Build effective question ---
    effective_question = text or "请分析这张图片"

    # --- Step 5: Delegate to ReAct agent loop ---
    _append_to_history(session_id, "user", effective_question)

    final_answer_text = ""

    async for event in run_agent_stream(
        question=effective_question,
        session_id=session_id,
        template=template,
        max_steps=max_steps,
        image_url=final_image,
        history=conversation_history if conversation_history else None,
    ):
        # Capture the final answer for history
        if event["event"] == "answer":
            data = json.loads(event["data"])
            final_answer_text = data.get("answer", "")

        # Inject multimodal session_id into done event
        if event["event"] == "done":
            data = json.loads(event["data"])
            data["multimodal_session_id"] = session_id
            event["data"] = json.dumps(data, ensure_ascii=False)

        yield event

    # Append assistant response to session history
    if final_answer_text:
        _append_to_history(session_id, "assistant", final_answer_text)

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
    session_id: str | None = None,
    image_base64: str | None = None,
    image_url: str | None = None,
) -> dict:
    """Run full voice Q&A loop: ASR → LLM → TTS.

    Args:
        audio_base64: Base64-encoded audio from voice recording.
        language: ASR language code.
        template: ReAct template.
        max_steps: Max agent steps.
        session_id: Optional multimodal session ID for continuing conversation.
        image_base64: Optional image base64 for visual context.
        image_url: Optional image URL for visual context.

    Returns:
        Dict with text, audio_base64, and session_id keys.
    """
    # Step 1: ASR
    audio_bytes = base64.b64decode(audio_base64)
    detected_mime = _detect_audio_format(audio_bytes)
    transcribed = await transcribe(
        audio_bytes, content_type=detected_mime, language=language
    )

    # Step 2: Resolve session and image context
    resolved_image = image_url or image_base64
    conversation_history = None

    if session_id:
        existing = _get_session(session_id)
        if existing:
            # Reuse session's image if no new image provided
            if not resolved_image:
                resolved_image = existing.get("image_url") or existing.get("image_base64")
            conversation_history = _truncate_history(list(existing["history"]))
            # Append user voice message to history
            _append_to_history(session_id, "user", transcribed)
        else:
            session_id = None  # Invalid session, create new one
    else:
        session_id = _create_session()
        _append_to_history(session_id, "user", transcribed)

    if resolved_image and session_id:
        _update_session_image(session_id, image_url, image_base64)

    # Step 3: LLM
    from app.services.sessions import get_session as get_agent_session

    agent_session_id = uuid.uuid4().hex[:12]
    agent_result = await run_agent_sync(
        question=transcribed,
        session_id=agent_session_id,
        template=template,
        max_steps=max_steps,
        image_url=resolved_image,
        history=conversation_history,
    )

    answer_text = (
        agent_result.final_answer
        if agent_result
        else "抱歉，我无法回答这个问题。"
    )

    # Append assistant response to multimodal session history
    if session_id and answer_text:
        _append_to_history(session_id, "assistant", answer_text)

    # Step 4: TTS (truncate long text to avoid TTS character limit)
    tts_audio = await synthesize(answer_text[:2000])
    audio_b64 = base64.b64encode(tts_audio).decode("utf-8")

    return {
        "text": answer_text,
        "audio_base64": audio_b64,
        "session_id": session_id,
    }
