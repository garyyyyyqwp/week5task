"""Pydantic v2 schemas for multimodal API endpoints."""

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

class ASRResponse(BaseModel):
    """Audio transcription response."""

    text: str = Field(..., description="Transcribed text")
    language: str | None = Field(default=None, description="Detected language code")


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
        description="Playback speed 0.25-4.0",
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
            raise ValueError(
                f"无效的模板: '{v}'。可选: {', '.join(sorted(valid))}"
            )
        return v

    @model_validator(mode="after")
    def validate_at_least_one_input(self):
        """Ensure at least one input modality is provided."""
        if (
            not self.text
            and not self.image_base64
            and not self.image_url
            and not self.audio_base64
        ):
            raise ValueError(
                "至少需要提供一种输入: text, image_base64, image_url, 或 audio_base64"
            )
        return self


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
