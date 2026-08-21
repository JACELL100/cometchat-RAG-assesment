"""
Retrieval pipeline — metadata-aware retrieval with conflict detection.

Pipeline:
  1. Embed query → retrieve top-K candidates from ChromaDB
  2. Filter: Remove non-customer-usable docs (draft, policy_authority=none, etc.)
  3. Rerank: Score by (similarity × authority_rank weight)
  4. Conflict detection: Check if top results from 2+ active official docs
     have genuinely conflicting content on the same topic
  5. Return top-N passages with conflict flag and confidence
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config import cfg
from src.vector_store import query_index

logger = logging.getLogger("aster_row.retrieval")

# ── Authority weight multipliers ───────────────────────────────────────────────
_AUTHORITY_WEIGHTS = {
    10: 1.0,   # active + official
    7: 0.85,   # active but not official
    3: 0.6,    # unknown status
    2: 0.3,    # draft or policy_authority=none
    1: 0.1,    # superseded
    0: 0.0,    # customer_answering=False / internal-only
}


def _authority_multiplier(rank: int) -> float:
    # Find closest rank key
    for threshold in sorted(_AUTHORITY_WEIGHTS.keys(), reverse=True):
        if rank >= threshold:
            return _AUTHORITY_WEIGHTS[threshold]
    return 0.0


def _combined_score(passage: dict) -> float:
    """Combined relevance score: similarity × authority multiplier."""
    sim = passage.get("similarity", 0.0)
    rank = int(passage.get("authority_rank", 0))
    return sim * _authority_multiplier(rank)


# ── Conflict detection ─────────────────────────────────────────────────────────

# Known conflict pairs (filename pairs that have genuine documented conflicts)
_KNOWN_CONFLICT_PAIRS: list[tuple[str, str]] = [
    ("11-product-care.md", "12-breeze-tumbler-product-card.md"),
]


def _detect_conflicts(passages: list[dict]) -> tuple[bool, list[dict]]:
    """
    Detect genuine conflicts between active official sources.

    A conflict is flagged when:
    - Two+ passages come from different active+official documents
    - Those documents are known to conflict, OR
    - Simple heuristic: one says "hand wash" while another says "dishwasher safe"

    Returns (conflict_detected, conflict_details).
    """
    # Only consider active official sources
    active_official = [
        p for p in passages
        if p.get("status") == "active" and p.get("policy_authority") == "official"
    ]

    filenames = list({p.get("filename", "") for p in active_official})

    conflicts = []
    for fn1, fn2 in _KNOWN_CONFLICT_PAIRS:
        if fn1 in filenames and fn2 in filenames:
            p1_texts = [p["text"] for p in active_official if p.get("filename") == fn1]
            p2_texts = [p["text"] for p in active_official if p.get("filename") == fn2]
            conflicts.append({
                "source_a": fn1,
                "source_b": fn2,
                "source_a_excerpt": p1_texts[0][:300] if p1_texts else "",
                "source_b_excerpt": p2_texts[0][:300] if p2_texts else "",
            })

    # Heuristic: dishwasher conflict (hand-wash vs dishwasher safe)
    if not conflicts and len(active_official) >= 2:
        hand_wash = any("hand" in p["text"].lower() and "wash" in p["text"].lower()
                        for p in active_official)
        dishwasher = any("dishwasher safe" in p["text"].lower()
                         for p in active_official)
        if hand_wash and dishwasher:
            sources = list({p.get("filename", "") for p in active_official
                           if "wash" in p["text"].lower()})
            if len(sources) >= 2:
                conflicts.append({
                    "source_a": sources[0],
                    "source_b": sources[1],
                    "source_a_excerpt": "",
                    "source_b_excerpt": "",
                })

    return bool(conflicts), conflicts


# ── Main retrieval function ────────────────────────────────────────────────────


def retrieve(
    query: str,
    session_id: str = "",
    include_superseded: bool = False,
) -> dict[str, Any]:
    """
    Main retrieval entry point.

    Returns:
    {
        "passages": [...],          # top-N filtered+reranked passages
        "conflict_detected": bool,
        "conflict_details": [...],
        "confidence": "high"|"medium"|"low"|"none",
        "query": str,
    }
    """
    logger.debug("session=%s | retrieving for query=%r", session_id, query)

    # Step 1: Raw retrieval
    raw = query_index(query, n_results=cfg.RETRIEVAL_TOP_K)

    # Step 2: Filter out non-customer-usable passages
    # We keep superseded docs only if the query explicitly asks about old/legacy policy
    asks_for_legacy = bool(re.search(
        r"\b(old|legacy|previous|before|prior|was|used to|changed)\b",
        query, re.IGNORECASE
    ))

    filtered = []
    for p in raw:
        is_usable = p.get("is_customer_usable", True)
        rank = int(p.get("authority_rank", 0))
        status = p.get("status", "")

        if rank == 0:
            # Explicitly blocked (customer_answering=False or internal-only)
            logger.debug("Filtered out (non-customer): %s", p.get("filename"))
            continue
        if status == "superseded" and not asks_for_legacy:
            # Only include superseded if context warrants it
            logger.debug("Filtered out (superseded): %s", p.get("filename"))
            continue
        filtered.append(p)

    # Step 3: Rerank by combined score
    reranked = sorted(filtered, key=_combined_score, reverse=True)
    top_passages = reranked[: cfg.RETRIEVAL_FINAL_K]

    # Step 4: Detect conflicts
    conflict_detected, conflict_details = _detect_conflicts(top_passages)

    # Step 5: Compute confidence
    if not top_passages:
        confidence = "none"
    elif conflict_detected:
        confidence = "low"
    elif top_passages[0].get("similarity", 0) > 0.75:
        confidence = "high"
    elif top_passages[0].get("similarity", 0) > 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    logger.debug(
        "session=%s | passages=%d | conflict=%s | confidence=%s",
        session_id, len(top_passages), conflict_detected, confidence,
    )

    return {
        "passages": top_passages,
        "conflict_detected": conflict_detected,
        "conflict_details": conflict_details,
        "confidence": confidence,
        "query": query,
    }
