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
    parser.add_argument(
        "--no-index-check", action="store_true",
        help="Skip the index presence check at startup (faster boot when index is already built)."
    )
    args = parser.parse_args()

    # ── Build index if needed ──────────────────────────────────────────────────
    if args.rebuild_index:
        # Explicit rebuild requested — always run regardless of --no-index-check.
        build_index(force_rebuild=True)
    elif not args.no_index_check:
        # Default: check if the collection is empty and auto-build on first run.
        collection = _get_collection()
        if collection.count() == 0:
            logger.info("No index found — building now (first run)...")
            build_index(force_rebuild=False)
    else:
        logger.info("Skipping index check (--no-index-check). Assuming index is ready.")

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
