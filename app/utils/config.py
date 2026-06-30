"""Configuration — environment variable loading with get_env() helper.

Extends Week 3 config with Agent-specific settings for the ReAct loop.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Get environment variable with optional validation.

    Args:
        key: Environment variable name.
        default: Default value if not set.
        required: If True, raises ValueError when not set and no default.

    Returns:
        The environment variable value as a string.

    Raises:
        ValueError: If required and not set with no default.
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(
            f"Environment variable '{key}' is not set. "
            f"Please set it in your .env file or system environment."
        )
    return value


# --- LLM (Multimodal Vision + Text, supports Function Calling) ---
OPENAI_API_KEY: str = get_env("OPENAI_API_KEY", required=True)
OPENAI_BASE_URL: str = get_env("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
OPENAI_MODEL: str = get_env("OPENAI_MODEL", "glm-4.6v-flash")

# --- Vision LLM (deprecated: main model now handles vision directly) ---
VISION_MODEL: str = get_env("VISION_MODEL", "glm-4.6v-flash")
VISION_API_KEY: str = get_env("VISION_API_KEY", OPENAI_API_KEY)
VISION_BASE_URL: str = get_env("VISION_BASE_URL", OPENAI_BASE_URL)

# --- Embedding ---
EMBEDDING_PROVIDER: str = get_env("EMBEDDING_PROVIDER", "zhipu")
EMBEDDING_MODEL: str = get_env("EMBEDDING_MODEL", "embedding-2")
EMBEDDING_API_KEY: str = get_env("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_BASE_URL: str = get_env("EMBEDDING_BASE_URL", OPENAI_BASE_URL)

# --- CLIP ---
CLIP_MODEL_NAME: str = get_env("CLIP_MODEL_NAME", "ViT-B/32")
CLIP_DEVICE: str = get_env("CLIP_DEVICE", "cpu")

# --- ChromaDB ---
CHROMA_PERSIST_DIR: str = get_env("CHROMA_PERSIST_DIR", "./chroma_data")
TEXT_COLLECTION_NAME: str = get_env("TEXT_COLLECTION_NAME", "text_collection")
IMAGE_COLLECTION_NAME: str = get_env("IMAGE_COLLECTION_NAME", "image_collection")

# --- Chunker ---
CHUNK_MAX_TOKENS: int = int(get_env("CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS: int = int(get_env("CHUNK_OVERLAP_TOKENS", "50"))

# --- RAG ---
RAG_TOP_K: int = int(get_env("RAG_TOP_K", "5"))
RAG_TOP_M: int = int(get_env("RAG_TOP_M", "3"))

# --- Image Processing ---
IMAGE_MAX_DIMENSION: int = int(get_env("IMAGE_MAX_DIMENSION", "512"))
IMAGE_SAVE_DIR: str = get_env("IMAGE_SAVE_DIR", "./data/images")

# --- Agent (Week 4) ---
AGENT_MAX_STEPS: int = int(get_env("AGENT_MAX_STEPS", "10"))
AGENT_DEFAULT_STRATEGY: str = get_env("AGENT_DEFAULT_STRATEGY", "basic")
AGENT_SESSION_DIR: str = get_env("AGENT_SESSION_DIR", "./data/sessions")
SEARCH_WEB_QUERY_MAX_LENGTH: int = int(get_env("SEARCH_WEB_QUERY_MAX_LENGTH", "200"))

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

# --- Audio Limits ---
MAX_AUDIO_SIZE_MB: int = int(get_env("MAX_AUDIO_SIZE_MB", "10"))
MAX_AUDIO_DURATION_SECONDS: int = int(get_env("MAX_AUDIO_DURATION_SECONDS", "120"))
MAX_IMAGE_SIZE_MB: int = int(get_env("MAX_IMAGE_SIZE_MB", "10"))

# --- Multimodal Chat ---
MULTIMODAL_MAX_HISTORY_TURNS: int = int(get_env("MULTIMODAL_MAX_HISTORY_TURNS", "20"))
