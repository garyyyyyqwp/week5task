"""AI 研究助手 Agent — FastAPI application entry point.

Week 4: ReAct agent with multi-tool orchestration.
Builds on Week 3's multimodal RAG + evaluation platform.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import agent, multimodal


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    """Handle unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "message": "请联系管理员",
        },
    )
