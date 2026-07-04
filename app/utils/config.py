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
OPENAI_MODEL: str = get_env("OPENAI_MODEL", "glm-4.6v")

# --- Vision LLM (deprecated: main model now handles vision directly) ---
VISION_MODEL: str = get_env("VISION_MODEL", "glm-4.6v")
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

# Provider-specific default endpoints. When a provider is selected but no
# explicit *_BASE_URL / *_MODEL is set, we derive a consistent default so the
# key, model name, and endpoint always belong to the same vendor.
_OPENAI_API_BASE = "https://api.openai.com/v1/"
_ZHIPU_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"

_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": _OPENAI_API_BASE,
        "asr_model": "whisper-1",
        "tts_model": "tts-1",
    },
    "zhipu": {
        "base_url": _ZHIPU_API_BASE,
        "asr_model": "glm-asr-2512",
        "tts_model": "cogtts",
    },
}


def _provider_default(provider: str, key: str) -> str:
    """Look up a provider-specific default, falling back to openai."""
    return _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["openai"])[key]


# --- ASR (Speech-to-Text) ---
ASR_PROVIDER: str = get_env("ASR_PROVIDER", "zhipu")
ASR_MODEL: str = get_env("ASR_MODEL", _provider_default(ASR_PROVIDER, "asr_model"))
ASR_API_KEY: str = get_env("ASR_API_KEY", OPENAI_API_KEY)
ASR_BASE_URL: str = get_env(
    "ASR_BASE_URL", _provider_default(ASR_PROVIDER, "base_url")
)

# --- TTS (Text-to-Speech) ---
TTS_PROVIDER: str = get_env("TTS_PROVIDER", "zhipu")
TTS_MODEL: str = get_env("TTS_MODEL", _provider_default(TTS_PROVIDER, "tts_model"))
# Default voice is provider-specific: OpenAI uses names like "alloy", while
# Zhipu cogtts uses its own set (e.g. "tongtong"). If TTS_VOICE isn't set we
# pick a sensible default for the selected provider.
TTS_VOICE: str = get_env(
    "TTS_VOICE", "tongtong" if TTS_PROVIDER == "zhipu" else "alloy"
)
TTS_API_KEY: str = get_env("TTS_API_KEY", OPENAI_API_KEY)
TTS_BASE_URL: str = get_env(
    "TTS_BASE_URL", _provider_default(TTS_PROVIDER, "base_url")
)

# --- Audio Limits ---
MAX_AUDIO_SIZE_MB: int = int(get_env("MAX_AUDIO_SIZE_MB", "10"))
MAX_AUDIO_DURATION_SECONDS: int = int(get_env("MAX_AUDIO_DURATION_SECONDS", "120"))
MAX_IMAGE_SIZE_MB: int = int(get_env("MAX_IMAGE_SIZE_MB", "10"))

# --- Multimodal Chat ---
MULTIMODAL_MAX_HISTORY_TURNS: int = int(get_env("MULTIMODAL_MAX_HISTORY_TURNS", "20"))
MULTIMODAL_SESSION_DIR: str = get_env("MULTIMODAL_SESSION_DIR", "./data/mm_sessions")
