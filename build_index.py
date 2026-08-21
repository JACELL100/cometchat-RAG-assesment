"""
build_index.py — One-time script to embed and index the knowledge base.

Usage:
  python build_index.py           # build if not already built
  python build_index.py --force   # force rebuild
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store import build_index
from src.observability import logger


def main():
    parser = argparse.ArgumentParser(description="Build the ChromaDB knowledge base index.")
    parser.add_argument(
        "--force", action="store_true",
        help="Force a full rebuild even if the index already exists."
    )
    args = parser.parse_args()

    logger.info("Starting knowledge base indexing...")
    build_index(force_rebuild=args.force)
    logger.info("Done.")


if __name__ == "__main__":
    main()
