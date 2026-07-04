"""AI 学习助手 — 多模态 AI 应用 (Week 5)

FastAPI application entry point.
能看、会听、能说的 AI 学习助手，支持拍照提问 + 语音交互 + AI 语音讲解。
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import agent, multimodal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI 研究助手 Agent",
    description="基于 ReAct 模式的 AI 研究助手，支持多工具自主编排",
    version="0.3.0",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS: origins come from the CORS_ALLOW_ORIGINS env var (comma-separated).
# Note the browser spec forbids allow_credentials=True together with the "*"
# wildcard, so we only enable credentials when explicit origins are listed.
_cors_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
if _cors_env == "*":
    _allow_origins = ["*"]
    _allow_credentials = False
else:
    _allow_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: Create directories
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Create required directories on startup."""
    for d in [
        "./chroma_data",
        "./data/sessions",
        "./data/mm_sessions",
        "./data/images",
        "./data/audio",
    ]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(agent.router, prefix="/api/v1/agent")
app.include_router(multimodal.router, prefix="/api/v1/multimodal")


# ---------------------------------------------------------------------------
# Static Files & SPA Fallback
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve the SPA frontend."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with Chinese error messages."""
    # SPA fallback for non-API 404s
    if exc.status_code == 404 and not request.url.path.startswith("/api/"):
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions.

    Logs the full traceback server-side (so failures are debuggable) while
    returning a generic message to the client (so internals aren't leaked).
    """
    logger.exception(
        "Unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "message": "请联系管理员",
        },
    )
