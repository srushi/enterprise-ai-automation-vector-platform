"""Semantic search CLI over the incident vector index.

Usage:
    python -m vector.search_cli "AS2 certificate failures" [--k 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import config
from vector.index import VectorIndex


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic search over incidents.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=config.VECTOR_INDEX_DIR)
    args = parser.parse_args(argv)

    index = VectorIndex.load(args.index_dir)
    results = index.search(args.query, k=args.k)
    if not results:
        print("No matching documents.")
        return 0
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. [{score:.3f}] ({doc.metadata.get('type', 'doc')}) {doc.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
