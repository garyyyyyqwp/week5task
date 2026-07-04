"""Multimodal Router — endpoints for image + voice + text interaction."""

import base64
import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.schemas.multimodal import (
    ASRResponse,
    TTSRequest,
    MultimodalChatRequest,
    VoiceLoopRequest,
    VoiceLoopResponse,
)
from app.services.asr import transcribe, AudioValidationError, ASRError
from app.services.tts import (
    synthesize,
    synthesize_stream,
    TTSError,
    TTS_MEDIA_TYPE,
)
from app.services.multimodal_chat import (
    run_multimodal_chat_stream,
    run_voice_loop,
    list_multimodal_sessions,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["multimodal"])

ALLOWED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif",
}


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
    if file.content_type and "audio" not in (file.content_type or ""):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}。请上传音频文件。",
        )

    try:
        audio_data = await file.read()
        text = await transcribe(
            audio_data,
            content_type=file.content_type or "audio/webm",
            language=language,
        )
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
    """Convert text to speech audio, streamed for low first-byte latency.

    Streams audio/mpeg chunks as they are synthesized so the browser can
    begin playback sooner. Input is validated up front so bad requests
    still return a proper 4xx before the stream starts.
    """
    from app.services.tts import _validate_tts_input

    # Validate before streaming so we can return a 400 (can't change the
    # status code once a StreamingResponse has begun sending bytes).
    try:
        _validate_tts_input(request.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Pull the first chunk BEFORE returning the response. This lets provider
    # errors (quota, bad params, timeout) surface as a proper HTTP 502 with a
    # real message, instead of a 200 with an empty body that the browser can
    # only report as a vague "playback failed".
    stream = synthesize_stream(
        text=request.text,
        voice=request.voice,
        speed=request.speed,
    )
    try:
        first_chunk = await stream.__anext__()
    except StopAsyncIteration:
        raise HTTPException(status_code=502, detail="语音合成未返回音频数据")
    except TTSError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("TTS endpoint error")
        raise HTTPException(status_code=500, detail="语音合成服务暂时不可用")

    async def audio_chunks():
        yield first_chunk
        try:
            async for chunk in stream:
                yield chunk
        except Exception as e:
            # Failure mid-stream after headers are sent — can't change status
            # now, so log and end the body.
            logger.error("TTS mid-stream error: %s", e)

    ext = "wav" if TTS_MEDIA_TYPE == "audio/wav" else "mp3"
    return StreamingResponse(
        audio_chunks(),
        media_type=TTS_MEDIA_TYPE,
        headers={
            "Content-Disposition": f"inline; filename=speech.{ext}",
            "Cache-Control": "no-cache",
        },
    )


# ---------------------------------------------------------------------------
# POST /chat — Multimodal chat (SSE streaming)
# ---------------------------------------------------------------------------

@router.post("/chat")
async def multimodal_chat(request: MultimodalChatRequest):
    """Unified multimodal chat with SSE streaming.

    Accepts text + image + voice input. Processing:
    1. Voice audio → ASR transcription
    2. Build multimodal message content (image goes directly to model)
    3. Run ReAct agent with SSE streaming

    SSE events: asr_result, thought, action, observation, answer, done, session
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
            session_id=request.session_id,
            image_base64=request.image_base64,
            image_url=request.image_url,
        )
        return VoiceLoopResponse(**result)
    except Exception as e:
        logger.error("Voice loop error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"语音问答处理失败: {str(e)}"
        )


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
            detail=(
                f"不支持的图片类型: {file.content_type}。"
                f"支持: {', '.join(ALLOWED_IMAGE_TYPES)}"
            ),
        )

    from app.utils.config import MAX_IMAGE_SIZE_MB

    image_data = await file.read()
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=(
                f"图片过大: {size_mb:.1f}MB。"
                f"最大允许: {MAX_IMAGE_SIZE_MB}MB"
            ),
        )

    b64 = base64.b64encode(image_data).decode("utf-8")
    data_uri = f"data:{file.content_type};base64,{b64}"

    return {
        "data_uri": data_uri,
        "size_bytes": len(image_data),
        "content_type": file.content_type,
    }
