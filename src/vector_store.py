"""
Vector store — ChromaDB + sentence-transformers embeddings.

Handles:
  - Building the index from knowledge-base documents
  - Querying for relevant passages
  - Metadata-based filtering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import cfg
from src.document_loader import load_all_documents

logger = logging.getLogger("aster_row.vector_store")

# ── Singleton embedding model ──────────────────────────────────────────────────
_embedder = None


def _get_embedder():
    """Lazy-load the SentenceTransformer model on first use."""
    global _embedder
    if _embedder is None:
        # Import is deferred here so startup doesn't pay the cost unless the
        # embedder is actually needed (i.e. index is empty or a query arrives).
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading embedding model: %s", cfg.EMBEDDING_MODEL)
        _embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _embedder


# ── ChromaDB client + collection ──────────────────────────────────────────────
_client: chromadb.Client | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        cfg.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(cfg.CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Build index ────────────────────────────────────────────────────────────────


def build_index(force_rebuild: bool = False) -> None:
    """
    Index all knowledge-base documents into ChromaDB.
    If the collection already has documents and force_rebuild=False, skips.
    """
    collection = _get_collection()

    if not force_rebuild and collection.count() > 0:
        logger.info(
            "Index already contains %d chunks. Skipping rebuild (use --force to rebuild).",
            collection.count(),
        )
        return

    if force_rebuild and collection.count() > 0:
        logger.info("Force rebuild — deleting existing index.")
        # Delete and recreate
        global _client, _collection
        _client.delete_collection(cfg.CHROMA_COLLECTION)
        _collection = _client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        collection = _collection

    logger.info("Loading documents from knowledge base...")
    chunks = load_all_documents()
    logger.info("Loaded %d chunks from %d-ish documents.", len(chunks), len(chunks) // 4)

    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    logger.info("Embedding %d chunks...", len(texts))
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # Build metadata dicts (ChromaDB only accepts str/int/float/bool values)
    metadatas = []
    for chunk in chunks:
        meta = {
            k: v
            for k, v in chunk.items()
            if k not in ("chunk_id", "text")
            and isinstance(v, (str, int, float, bool))
        }
        metadatas.append(meta)

    logger.info("Adding chunks to ChromaDB collection...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Index built: %d chunks indexed.", collection.count())


# ── Query ──────────────────────────────────────────────────────────────────────


def query_index(
    query_text: str,
    n_results: int | None = None,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """
    Query the index for relevant chunks.

    Returns a list of passage dicts with:
      - text, filename, section_heading, document_id, title, status,
        policy_authority, authority_rank, is_customer_usable, distance
    """
    collection = _get_collection()
    embedder = _get_embedder()

    if n_results is None:
        n_results = cfg.RETRIEVAL_TOP_K

    query_embedding = embedder.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages: list[dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return passages

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Cosine distance → similarity score (0–1, higher = more similar)
        similarity = 1.0 - dist

        passage = {
            "text": doc,
            "similarity": round(similarity, 4),
            **meta,
        }
        passages.append(passage)

    return passages
