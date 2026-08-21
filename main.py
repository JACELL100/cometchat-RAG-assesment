"""
main.py — Entry point for the Aster & Row support agent web server.

Automatically builds the knowledge-base index on first run if not present.

Usage:
  python main.py
  python main.py --port 8080
  python main.py --rebuild-index
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from src.config import cfg
from src.observability import logger
from src.vector_store import _get_collection, build_index


def main():
    parser = argparse.ArgumentParser(description="Aster & Row AI Support Agent")
    parser.add_argument("--port", type=int, default=cfg.PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Force rebuild the vector index before starting."
    )
    args = parser.parse_args()

    # ── Build index if needed ──────────────────────────────────────────────────
    collection = _get_collection()
    if args.rebuild_index or collection.count() == 0:
        if collection.count() == 0:
            logger.info("No index found — building now (first run)...")
        build_index(force_rebuild=args.rebuild_index)

    logger.info(
        "Starting Aster & Row Support Agent on http://%s:%d",
        args.host, args.port,
    )
    logger.info("Model: %s | Debug: %s", cfg.GROQ_MODEL, cfg.DEBUG)

    # ── Start FastAPI server ───────────────────────────────────────────────────
    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level=cfg.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
