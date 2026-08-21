"""
Document loader for the Aster & Row knowledge base.

Parses each Markdown file from knowledge-base/:
  - Extracts YAML front-matter metadata
  - Splits the body into section chunks (one chunk per ## heading)
  - Tags each chunk with parent document metadata + section title

Each chunk is a dict ready to be embedded and stored in ChromaDB.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import frontmatter  # python-frontmatter

from src.config import cfg


# ── Fields extracted from front-matter ────────────────────────────────────────
_FRONTMATTER_FIELDS = [
    "document_id",
    "title",
    "status",
    "effective_date",
    "last_reviewed",
    "audience",
    "policy_authority",
    "supersedes",
    "superseded_by",
    "customer_answering",
]


def _authority_rank(meta: dict) -> int:
    """
    Numeric authority rank for sorting/filtering.
    Higher = more authoritative.
    """
    status = meta.get("status", "unknown")
    authority = meta.get("policy_authority", "none")
    customer_answering = meta.get("customer_answering", True)  # True by default

    if customer_answering is False:
        return 0  # explicitly blocked from customer use
    if status == "superseded":
        return 1
    if status == "draft" or authority == "none":
        return 2
    if authority == "official" and status == "active":
        return 10
    if status == "active":
        return 7
    return 3


def _is_usable_for_customers(meta: dict) -> bool:
    """
    True if this document can be used as an authoritative source for customers.
    Superseded docs are indexed for reference but not authoritative.
    """
    ca = meta.get("customer_answering", None)
    if ca is False:
        return False
    if meta.get("audience") == "internal" and meta.get("policy_authority") == "none":
        return False
    return True


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """
    Split a Markdown body by ## headings.
    Returns list of (heading, content) tuples.
    The intro before the first heading is included as ("intro", text).
    """
    sections: list[tuple[str, str]] = []
    # Split on lines that start with ## (but not ###)
    pattern = re.compile(r"^(## .+)$", re.MULTILINE)
    parts = pattern.split(body)

    if parts[0].strip():
        sections.append(("intro", parts[0].strip()))

    # parts alternates: [before, heading1, content1, heading2, content2, ...]
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].lstrip("#").strip()
        content = parts[i + 1].strip()
        if content:
            sections.append((heading, content))
        i += 2

    return sections


def load_document(filepath: Path) -> list[dict[str, Any]]:
    """
    Load one Markdown file and return a list of chunk dicts.
    Each chunk has keys: chunk_id, text, filename, and all metadata fields.
    """
    raw = filepath.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)

    # Extract metadata
    meta: dict[str, Any] = {}
    for field in _FRONTMATTER_FIELDS:
        val = post.metadata.get(field)
        if val is not None:
            # ChromaDB requires string metadata values
            meta[field] = str(val)

    meta["filename"] = filepath.name
    meta["authority_rank"] = _authority_rank(post.metadata)
    meta["is_customer_usable"] = _is_usable_for_customers(post.metadata)

    # Split body into sections
    sections = _split_into_sections(post.content)

    chunks: list[dict[str, Any]] = []
    for idx, (heading, content) in enumerate(sections):
        chunk_id = f"{filepath.stem}__{idx}__{heading.replace(' ', '_')}"
        chunk_text = (
            f"[Document: {meta.get('title', filepath.name)}]\n"
            f"[Section: {heading}]\n\n"
            f"{content}"
        )
        chunk = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "section_heading": heading,
            **meta,
        }
        chunks.append(chunk)

    return chunks


def load_all_documents(kb_dir: Path | None = None) -> list[dict[str, Any]]:
    """
    Load all .md files from the knowledge base directory.
    Returns a flat list of chunk dicts.
    """
    if kb_dir is None:
        kb_dir = cfg.KNOWLEDGE_BASE_DIR

    all_chunks: list[dict[str, Any]] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        chunks = load_document(md_file)
        all_chunks.extend(chunks)

    return all_chunks
