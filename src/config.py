"""
Configuration loader — reads environment variables with sensible defaults.
All components import from here rather than reading env vars directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")


class Config:
    # ── Groq LLM ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    GROQ_EVAL_MODEL: str = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-20b")
    GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "5"))
    GROQ_RETRY_BASE_DELAY: float = float(os.getenv("GROQ_RETRY_BASE_DELAY", "10"))

    # ── Embeddings (local sentence-transformers) ───────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PATH: Path = _root / "chroma_db"
    CHROMA_COLLECTION: str = "aster_row_kb"

    # ── Knowledge base ────────────────────────────────────────────────────────
    KNOWLEDGE_BASE_DIR: Path = _root / "knowledge-base"

    # ── Orders data ───────────────────────────────────────────────────────────
    ORDERS_FILE: Path = _root / "data" / "orders.json"

    # ── Retrieval settings ────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    RETRIEVAL_FINAL_K: int = int(os.getenv("RETRIEVAL_FINAL_K", "5"))

    # ── Conversation memory ───────────────────────────────────────────────────
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
    COMPRESS_AFTER_TURNS: int = int(os.getenv("COMPRESS_AFTER_TURNS", "10"))

    # ── Observability ─────────────────────────────────────────────────────────
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOGS_DIR: Path = _root / "logs"

    # ── Server ────────────────────────────────────────────────────────────────
    PORT: int = int(os.getenv("PORT", "8000"))


cfg = Config()
