"""
Structured observability layer.

Every agent interaction produces a structured JSON log entry with:
  - timestamp, session_id, user_message
  - conversation_history snapshot
  - retrieved passages (with metadata + scores)
  - tool calls and sanitized results
  - final response, sources cited, handoff flag
  - errors / fallbacks

Secrets are never logged. The sanitize helpers strip private fields before
anything reaches the log.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import cfg

# ── Bootstrap logging ──────────────────────────────────────────────────────────
import sys
cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)

_log_level = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)
# Use UTF-8 safe stream handler for Windows console compatibility
_stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
    if hasattr(sys.stdout, 'fileno') else sys.stdout
)
_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.basicConfig(level=_log_level, handlers=[_stream_handler])
logger = logging.getLogger("aster_row")

# Debug JSONL log file
_debug_log_path = cfg.LOGS_DIR / "agent_debug.jsonl"


def _write_jsonl(entry: dict) -> None:
    """Append a structured JSON entry to the debug log file."""
    with open(_debug_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")


# ── Public logging helpers ─────────────────────────────────────────────────────


def log_interaction(
    *,
    session_id: str,
    user_message: str,
    conversation_history: list[dict],
    retrieved_passages: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
    final_response: str | None = None,
    sources_cited: list[str] | None = None,
    handoff_recommended: bool = False,
    confidence: str | None = None,
    errors: list[str] | None = None,
    duration_ms: float | None = None,
) -> None:
    """Log a complete agent interaction."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "user_message": user_message,
        "conversation_history_turns": len(conversation_history),
        "retrieved_passages": retrieved_passages or [],
        "tool_calls": tool_calls or [],
        "final_response": final_response,
        "sources_cited": sources_cited or [],
        "handoff_recommended": handoff_recommended,
        "confidence": confidence,
        "errors": errors or [],
        "duration_ms": duration_ms,
    }

    if cfg.DEBUG:
        # Full history in debug mode
        entry["conversation_history"] = conversation_history
        logger.debug("Interaction:\n%s", json.dumps(entry, indent=2, default=str))
    else:
        logger.info(
            "session=%s | turns=%d | handoff=%s | confidence=%s | sources=%s",
            session_id,
            len(conversation_history),
            handoff_recommended,
            confidence,
            sources_cited,
        )

    _write_jsonl(entry)


def log_error(session_id: str, error: Exception, context: str = "") -> None:
    """Log an error without leaking secrets."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "event": "error",
        "context": context,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    logger.error("Error in %s: %s: %s", context, type(error).__name__, str(error))
    _write_jsonl(entry)


def log_tool_call(
    session_id: str,
    tool_name: str,
    arguments: dict,
    result_summary: str,
) -> dict:
    """Build a sanitized tool-call log record."""
    record = {
        "tool_name": tool_name,
        "arguments": arguments,
        "result_summary": result_summary,
    }
    logger.debug(
        "session=%s | tool=%s | args=%s | result=%s",
        session_id,
        tool_name,
        arguments,
        result_summary,
    )
    return record


def log_retrieval(
    session_id: str,
    query: str,
    passages: list[dict],
    conflict_detected: bool,
) -> None:
    logger.debug(
        "session=%s | retrieval query=%r | passages=%d | conflict=%s",
        session_id,
        query,
        len(passages),
        conflict_detected,
    )
