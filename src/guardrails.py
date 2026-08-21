"""
Guardrails layer — pre/post processing for safety.

Pre-processing (input):
  - Detect prompt injection patterns in user messages
  - Detect system-prompt extraction attempts

Post-processing (output):
  - Scan model response for accidentally leaked internal data
  - Scan for fabricated internal identifiers

These are defense-in-depth layers; the system prompt is the primary defense.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("aster_row.guardrails")

# ── Input guardrail patterns ───────────────────────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) (rules?|instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"forget (everything|all) (you|your)", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output|display|repeat) (your )?(system ?)?(prompt|instructions?|rules?|config)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"act as (if )?you (are|were)", re.IGNORECASE),
    re.compile(r"DAN\b", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"SYSTEM INSTRUCTION", re.IGNORECASE),
]

# ── Output guardrail patterns ──────────────────────────────────────────────────
# Things that must never appear in the model's response
_OUTPUT_FORBIDDEN_PATTERNS = [
    (re.compile(r"risk[_\s]?score\s*[:=]\s*\d+", re.IGNORECASE), "risk score"),
    (re.compile(r"warehouse[_\s]?note", re.IGNORECASE), "warehouse note"),
    (re.compile(r"support[_\s]?tags?", re.IGNORECASE), "support tags"),
    (re.compile(r"fraud review cleared", re.IGNORECASE), "fraud review note"),
    (re.compile(r"[a-zA-Z0-9._%+\-]+@example\.test"), "internal email address"),
    (re.compile(r"\b(REF-\d+)\b"), "internal refund batch ID"),
]

# Internal note content that must not be followed
_INTERNAL_INSTRUCTION_PATTERNS = [
    re.compile(r"AI instruction", re.IGNORECASE),
    re.compile(r"issue a \$\d+ coupon", re.IGNORECASE),
    re.compile(r"hide the delay reason", re.IGNORECASE),
    re.compile(r"do not mention review status", re.IGNORECASE),
    re.compile(r"never expose this note", re.IGNORECASE),
]


def check_input(user_message: str) -> dict:
    """
    Scan user input for injection patterns.
    Returns {"is_injection": bool, "flags": [str]}.
    """
    flags = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(user_message):
            flags.append(pattern.pattern)
            logger.warning("Input injection pattern detected: %s", pattern.pattern)

    return {"is_injection": bool(flags), "flags": flags}


def check_output(response_text: str) -> dict:
    """
    Scan model response for leaked internal data.
    Returns {"has_leak": bool, "leaked_types": [str], "clean_response": str}.
    """
    leaked_types = []
    clean = response_text

    for pattern, label in _OUTPUT_FORBIDDEN_PATTERNS:
        if pattern.search(clean):
            leaked_types.append(label)
            logger.error("OUTPUT GUARDRAIL: Leaked %s detected — redacting.", label)
            # Redact the matched content
            clean = pattern.sub(f"[REDACTED: {label}]", clean)

    return {
        "has_leak": bool(leaked_types),
        "leaked_types": leaked_types,
        "clean_response": clean,
    }


def check_tool_result_for_injections(tool_result_str: str) -> list[str]:
    """
    Scan a tool result string for instruction-like patterns.
    Returns list of detected patterns.
    These must not be followed by the agent.
    """
    found = []
    for pattern in _INTERNAL_INSTRUCTION_PATTERNS:
        if pattern.search(tool_result_str):
            found.append(pattern.pattern)
            logger.warning("Tool result contains instruction-like text: %s", pattern.pattern)
    return found
