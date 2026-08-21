"""
Agent core — Groq LLaMA 3.3 70B with function calling.

Manages:
  - Per-session conversation memory (windowed)
  - Context compression after N turns
  - RAG tool + Order tool function-calling loop
  - Response post-processing (sources, handoff detection)
  - Observability integration
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from groq import Groq

from src.config import cfg
from src.guardrails import check_input, check_output, check_tool_result_for_injections
from src.observability import log_error, log_interaction, log_retrieval, log_tool_call
from src.order_tool import ORDER_LOOKUP_TOOL, lookup_order
from src.retrieval import retrieve
from src.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger("aster_row.agent")

# ── Groq client ────────────────────────────────────────────────────────────────
_client = Groq(api_key=cfg.GROQ_API_KEY)

# ── KB search tool definition ──────────────────────────────────────────────────
KB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the Aster & Row knowledge base for policy, shipping, warranty, "
            "product care, or other company information. Use this for any company-specific "
            "question that isn't about a specific order status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language query describing what to look up.",
                }
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [KB_SEARCH_TOOL, ORDER_LOOKUP_TOOL]

# ── Session store ──────────────────────────────────────────────────────────────
# In production this would be Redis or a DB. For this mock assignment, in-memory dict.
_sessions: dict[str, dict] = {}


def create_session() -> str:
    session_id = str(uuid.uuid4())[:8]
    _sessions[session_id] = {
        "history": [],
        "turn_count": 0,
        "compressed_summary": None,
    }
    logger.info("New session created: %s", session_id)
    return session_id


def get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "turn_count": 0,
            "compressed_summary": None,
        }
    return _sessions[session_id]


def reset_session(session_id: str) -> None:
    _sessions[session_id] = {
        "history": [],
        "turn_count": 0,
        "compressed_summary": None,
    }
    logger.info("Session reset: %s", session_id)


def list_sessions() -> list[str]:
    return list(_sessions.keys())


# ── Context compression ────────────────────────────────────────────────────────


def _compress_history(session_id: str, history: list[dict]) -> str:
    """
    Use Groq to summarize older conversation turns.
    Returns a summary string.
    """
    logger.info("Compressing conversation history for session %s", session_id)
    compress_prompt = (
        "Summarize the following customer support conversation history concisely. "
        "Keep key facts: what the customer asked, what order IDs were mentioned, "
        "what policies were discussed, and any unresolved issues. "
        "Do not include opinions or filler.\n\n"
        + "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history
            if m["role"] in ("user", "assistant")
        )
    )
    try:
        resp = _client.chat.completions.create(
            model=cfg.GROQ_MODEL,
            messages=[{"role": "user", "content": compress_prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("Context compression failed: %s", e)
        return ""


def _maybe_compress(session_id: str, session: dict) -> list[dict]:
    """
    If history exceeds MAX_HISTORY_MESSAGES, compress the older half and
    replace it with a summary message. Returns the effective history to use.
    """
    history = session["history"]
    if len(history) <= cfg.MAX_HISTORY_MESSAGES:
        return history

    # Compress the older half
    midpoint = len(history) // 2
    old_history = history[:midpoint]
    recent_history = history[midpoint:]

    summary = _compress_history(session_id, old_history)
    if summary:
        session["compressed_summary"] = summary
        session["history"] = recent_history
        logger.info(
            "Compressed %d old turns into summary for session %s",
            midpoint, session_id,
        )
        return recent_history

    # Fallback: just trim to max
    return history[-cfg.MAX_HISTORY_MESSAGES:]


# ── Tool execution ─────────────────────────────────────────────────────────────


def _execute_tool(
    tool_name: str,
    arguments: dict,
    session_id: str,
    retrieval_state: dict,
) -> tuple[str, dict]:
    """
    Execute the named tool and return (result_str, log_record).
    """
    if tool_name == "search_knowledge_base":
        query = arguments.get("query", "")
        result = retrieve(query, session_id=session_id)
        retrieval_state.update(result)  # capture for observability

        passages = result["passages"]
        conflict = result["conflict_detected"]
        conflict_details = result["conflict_details"]

        if not passages:
            result_str = json.dumps({
                "found": False,
                "message": "No relevant information found in the knowledge base.",
            })
        else:
            formatted_passages = []
            for p in passages:
                formatted_passages.append({
                    "filename": p.get("filename", ""),
                    "section": p.get("section_heading", ""),
                    "document_id": p.get("document_id", ""),
                    "status": p.get("status", ""),
                    "policy_authority": p.get("policy_authority", ""),
                    "similarity": p.get("similarity", 0),
                    "text": p.get("text", ""),
                })

            result_dict = {
                "found": True,
                "passages": formatted_passages,
                "conflict_detected": conflict,
            }
            if conflict and conflict_details:
                result_dict["conflict_details"] = conflict_details
                result_dict["conflict_note"] = (
                    "WARNING: Active official documents conflict on this topic. "
                    "Surface both sides and recommend human confirmation."
                )

            result_str = json.dumps(result_dict, ensure_ascii=False)

        log_record = log_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            result_summary=f"passages={len(passages)} conflict={conflict}",
        )
        return result_str, log_record

    elif tool_name == "lookup_order":
        order_id = arguments.get("order_id", "")
        result = lookup_order(order_id)

        # Check for injection-like content in tool result
        result_str = json.dumps(result, ensure_ascii=False)
        injections = check_tool_result_for_injections(result_str)
        if injections:
            logger.warning(
                "session=%s | Injection patterns in order tool result: %s",
                session_id, injections,
            )
            # Sanitize: remove the internal instruction keys before passing to model
            result.pop("internal", None)

        log_record = log_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            arguments={"order_id": order_id},  # safe to log
            result_summary=f"found={result.get('found')} status={result.get('status', 'n/a')}",
        )
        return json.dumps(result, ensure_ascii=False), log_record

    else:
        logger.error("Unknown tool called: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: {tool_name}"}), {}


# ── Response post-processing ───────────────────────────────────────────────────
_HANDOFF_SIGNALS = [
    "support team",
    "human agent",
    "contact support",
    "reach out to",
    "cannot complete",
    "handoff recommended",
    "specialist",
    "🤝",
]

_SOURCE_PATTERN = re.compile(
    r"📄\s*Source:\s*([^\n›]+?)(?:\s*›\s*([^\n]+))?(?:\n|$)"
)


def _extract_sources(text: str) -> list[str]:
    sources = []
    for m in _SOURCE_PATTERN.finditer(text):
        filename = m.group(1).strip()
        section = m.group(2).strip() if m.group(2) else ""
        sources.append(f"{filename} › {section}" if section else filename)
    return sources


def _is_handoff_recommended(text: str) -> bool:
    lower = text.lower()
    return any(signal.lower() in lower for signal in _HANDOFF_SIGNALS)


# ── Confidence from retrieval ──────────────────────────────────────────────────


def _compute_confidence(retrieval_state: dict, tool_calls_made: list) -> str:
    if not retrieval_state and not tool_calls_made:
        return "medium"  # No retrieval needed (e.g., clarifying question)
    if retrieval_state.get("conflict_detected"):
        return "low"
    conf = retrieval_state.get("confidence", "medium")
    return conf


# ── Main chat function ─────────────────────────────────────────────────────────


def chat(
    session_id: str,
    user_message: str,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Process a user message and return the agent response.

    Returns:
    {
        "response": str,
        "sources": [str],
        "handoff": bool,
        "confidence": str,
        "session_id": str,
        "debug": {...} if debug=True,
    }
    """
    start_time = time.time()
    session = get_session(session_id)
    session["turn_count"] += 1

    # ── Input guardrail ────────────────────────────────────────────────────────
    input_check = check_input(user_message)
    if input_check["is_injection"]:
        logger.warning("session=%s | Input injection detected.", session_id)
        # Don't block — the system prompt handles it; just note it

    # ── Add user message to history ────────────────────────────────────────────
    session["history"].append({"role": "user", "content": user_message})

    # ── Build message list ─────────────────────────────────────────────────────
    effective_history = _maybe_compress(session_id, session)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject compressed summary as context if available
    if session.get("compressed_summary"):
        messages.append({
            "role": "system",
            "content": (
                "Earlier conversation summary (for context):\n"
                + session["compressed_summary"]
            ),
        })

    messages.extend(effective_history)

    # ── Agent loop ─────────────────────────────────────────────────────────────
    retrieval_state: dict = {}
    tool_call_logs: list[dict] = []
    max_iterations = 5  # prevent runaway loops

    for iteration in range(max_iterations):
        try:
            response = _client.chat.completions.create(
                model=cfg.GROQ_MODEL,
                messages=messages,
                tools=ALL_TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as e:
            log_error(session_id, e, "groq_api_call")
            return {
                "response": (
                    "I'm experiencing a technical issue. "
                    "Please try again or contact our support team directly."
                ),
                "sources": [],
                "handoff": True,
                "confidence": "none",
                "session_id": session_id,
            }

        choice = response.choices[0]
        msg = choice.message

        # Add assistant message to the running messages list
        messages.append(msg.model_dump(exclude_unset=True))

        # If no tool calls, we have the final response
        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        # Execute all tool calls
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            result_str, log_record = _execute_tool(
                tool_name, arguments, session_id, retrieval_state
            )
            tool_call_logs.append(log_record)

            # Feed tool result back
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })
    else:
        # Exceeded max iterations
        final_text = (
            "I wasn't able to complete processing your request. "
            "Please contact our support team for assistance."
        )

    # ── Output guardrail ───────────────────────────────────────────────────────
    output_check = check_output(final_text)
    if output_check["has_leak"]:
        logger.error(
            "session=%s | OUTPUT GUARDRAIL TRIGGERED: %s",
            session_id, output_check["leaked_types"],
        )
        final_text = output_check["clean_response"]

    # ── Post-process ───────────────────────────────────────────────────────────
    sources = _extract_sources(final_text)
    handoff = _is_handoff_recommended(final_text)
    confidence = _compute_confidence(retrieval_state, tool_call_logs)

    # Add assistant response to session history
    session["history"].append({"role": "assistant", "content": final_text})

    duration_ms = (time.time() - start_time) * 1000

    # ── Log interaction ────────────────────────────────────────────────────────
    log_interaction(
        session_id=session_id,
        user_message=user_message,
        conversation_history=effective_history,
        retrieved_passages=retrieval_state.get("passages"),
        tool_calls=tool_call_logs,
        final_response=final_text,
        sources_cited=sources,
        handoff_recommended=handoff,
        confidence=confidence,
        duration_ms=round(duration_ms, 1),
    )

    result = {
        "response": final_text,
        "sources": sources,
        "handoff": handoff,
        "confidence": confidence,
        "session_id": session_id,
    }

    if debug or cfg.DEBUG:
        result["debug"] = {
            "retrieved_passages": retrieval_state.get("passages", []),
            "conflict_detected": retrieval_state.get("conflict_detected", False),
            "conflict_details": retrieval_state.get("conflict_details", []),
            "tool_calls": tool_call_logs,
            "input_injection_flags": input_check.get("flags", []),
            "output_leak_types": output_check.get("leaked_types", []),
            "duration_ms": round(duration_ms, 1),
            "turn_count": session["turn_count"],
        }

    return result
