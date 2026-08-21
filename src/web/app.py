"""
FastAPI web server for the Aster & Row support agent.

Endpoints:
  GET  /                     → serve chat UI
  POST /api/chat             → send a message
  POST /api/chat/reset       → reset session
  GET  /api/sessions/{id}    → session info (debug)
  GET  /health               → health check
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import chat, create_session, get_session, list_sessions, reset_session
from src.config import cfg
from src.observability import logger

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aster & Row Support Agent",
    description="RAG-powered AI support agent for Aster & Row.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Pydantic models ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    debug: bool = False


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[str]
    handoff: bool
    confidence: str
    debug: dict | None = None


class ResetRequest(BaseModel):
    session_id: str


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "model": cfg.GROQ_MODEL, "debug": cfg.DEBUG}


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Create a new session if none provided
    session_id = req.session_id or create_session()

    result = chat(
        session_id=session_id,
        user_message=req.message.strip(),
        debug=req.debug or cfg.DEBUG,
    )

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        sources=result.get("sources", []),
        handoff=result.get("handoff", False),
        confidence=result.get("confidence", "medium"),
        debug=result.get("debug"),
    )


@app.post("/api/chat/reset")
async def api_reset(req: ResetRequest):
    reset_session(req.session_id)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/api/sessions")
async def api_sessions():
    if not cfg.DEBUG:
        raise HTTPException(status_code=403, detail="Debug mode required.")
    return {"sessions": list_sessions()}


@app.get("/api/sessions/{session_id}")
async def api_session_detail(session_id: str):
    if not cfg.DEBUG:
        raise HTTPException(status_code=403, detail="Debug mode required.")
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "turn_count": session.get("turn_count", 0),
        "history_length": len(session.get("history", [])),
        "has_summary": bool(session.get("compressed_summary")),
    }
