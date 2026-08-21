"""
Order lookup tool.

Provides the lookup_order() function registered as a Groq tool.
Handles:
  - Input normalization (lowercase, whitespace, harmless punctuation)
  - Sanitization (never expose internal fields)
  - Status-aware field suppression (stale ETA on cancelled/returned orders)
  - Exception-order flagging
  - Unknown/malformed ID handling
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.config import cfg

logger = logging.getLogger("aster_row.order_tool")

# ── Load orders data at module import ──────────────────────────────────────────
_orders_data: dict[str, Any] | None = None
_orders_index: dict[str, dict] = {}  # order_id → order record


def _load_orders() -> None:
    global _orders_data, _orders_index
    if _orders_index:
        return
    raw = cfg.ORDERS_FILE.read_text(encoding="utf-8")
    _orders_data = json.loads(raw)
    for order in _orders_data.get("orders", []):
        _orders_index[order["order_id"]] = order
    logger.info("Loaded %d orders.", len(_orders_index))


# Call eagerly
_load_orders()

# Snapshot timestamp used as "current time" for deterministic evaluation
SNAPSHOT_AT: str = _orders_data.get("snapshot_at", "") if _orders_data else ""

# ── Validation ─────────────────────────────────────────────────────────────────
_ORDER_ID_PATTERN = re.compile(r"^ORD-\d+$")


def _normalize_order_id(raw: str) -> str:
    """Strip whitespace, uppercase, remove stray quotes/dots."""
    cleaned = raw.strip().upper()
    cleaned = re.sub(r"[\"'.]+", "", cleaned)
    return cleaned


def _is_valid_format(order_id: str) -> bool:
    return bool(_ORDER_ID_PATTERN.match(order_id))


# ── Sanitization ───────────────────────────────────────────────────────────────

# Fields safe to return to the model (per data dictionary)
_SAFE_ORDER_FIELDS = {
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}

_SAFE_ITEM_FIELDS = {"name", "quantity", "final_sale"}

# Statuses where delivery/carrier fields are stale and must not be reported
_STALE_STATUS = {"cancelled", "returned"}


def _sanitize_order(order: dict) -> dict:
    """
    Return only customer-safe fields from an order record.
    Suppresses stale fields for cancelled/returned orders.
    Suppresses internal fields unconditionally.
    """
    status = order.get("status", "")

    result: dict[str, Any] = {}

    for field in _SAFE_ORDER_FIELDS:
        if field not in order:
            continue
        value = order[field]

        # Suppress stale delivery fields for cancelled/returned orders
        if field in ("carrier", "tracking_number", "estimated_delivery") and status in _STALE_STATUS:
            continue

        # Explicitly mark missing estimates as unavailable
        if field == "estimated_delivery" and value is None:
            result[field] = "unavailable"
            continue

        result[field] = value

    # Items — only safe fields
    safe_items = []
    for item in order.get("items", []):
        safe_item = {k: v for k, v in item.items() if k in _SAFE_ITEM_FIELDS}
        safe_items.append(safe_item)
    result["items"] = safe_items

    # Flag if human review is required
    if status == "exception":
        result["requires_human_review"] = True

    return result


# ── Guardrail: scan result for leaked internal content ─────────────────────────
_INTERNAL_PATTERNS = [
    re.compile(r"risk_score", re.IGNORECASE),
    re.compile(r"warehouse_note", re.IGNORECASE),
    re.compile(r"support_tags", re.IGNORECASE),
    re.compile(r"@example\.test"),  # email addresses
    re.compile(r"\bAI instruction\b", re.IGNORECASE),
]


def _contains_leaked_internal_data(text: str) -> bool:
    return any(pat.search(text) for pat in _INTERNAL_PATTERNS)


# ── Main lookup function ───────────────────────────────────────────────────────


def lookup_order(order_id: str) -> dict[str, Any]:
    """
    Look up an order by ID and return sanitized customer-safe information.

    This function is exposed as a Groq tool. It never exposes internal fields.
    """
    _load_orders()

    normalized = _normalize_order_id(order_id)

    # Validate format
    if not _is_valid_format(normalized):
        logger.warning("Malformed order ID received: %r → %r", order_id, normalized)
        return {
            "found": False,
            "error": "invalid_format",
            "message": (
                f"The order ID '{order_id}' does not match the expected format "
                "(e.g., ORD-1007). Please check and try again."
            ),
        }

    # Look up
    order = _orders_index.get(normalized)
    if order is None:
        logger.info("Order not found: %r", normalized)
        return {
            "found": False,
            "error": "not_found",
            "order_id": normalized,
            "message": (
                f"Order {normalized} was not found in our system. "
                "Please verify the order ID and contact support if the issue persists."
            ),
        }

    safe = _sanitize_order(order)

    # Defensive guardrail: ensure no internal data leaked through sanitization
    safe_str = json.dumps(safe)
    if _contains_leaked_internal_data(safe_str):
        logger.error("GUARDRAIL: Internal data detected in sanitized output for %s. Blocked.", normalized)
        return {
            "found": True,
            "error": "sanitization_error",
            "order_id": normalized,
            "message": "Order data is temporarily unavailable. Please contact support.",
        }

    logger.info("Order lookup: %s → status=%s", normalized, safe.get("status"))
    return {"found": True, **safe}


# ── Groq tool definition ───────────────────────────────────────────────────────

ORDER_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": (
            "Look up the current status of a customer order by order ID. "
            "Only call this when the customer provides or refers to an order ID. "
            "Never call this without an order ID. "
            "Never fabricate order status without calling this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to look up, e.g. ORD-1007.",
                }
            },
            "required": ["order_id"],
        },
    },
}
