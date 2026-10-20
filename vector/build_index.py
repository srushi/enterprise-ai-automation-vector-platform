"""Build the vector index from landing-zone data + transformed products.

Usage:
    python -m vector.build_index [--embedder hashing|tfidf] [--index-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import config
from vector.chunking import build_corpus
from vector.embeddings import HashingEmbedder, TfidfEmbedder
from vector.index import VectorIndex
from warehouse.snowflake_loader import WarehouseLoader


def build_index(
    index_dir: Path,
    embedder_kind: str = "hashing",
    use_warehouse: bool = True,
) -> VectorIndex:
    """Chunk, embed, and persist the incident corpus."""
    config.ensure_dirs()
    warehouse_records = None
    if use_warehouse and config.WAREHOUSE_DB_PATH.exists():
        with WarehouseLoader() as loader:
            loader.ensure_schema()
            warehouse_records = loader.query(
                "SELECT * FROM raw_execution_logs ORDER BY start_time"
            )
    docs = build_corpus(warehouse_records=warehouse_records)
    embedder = (
        TfidfEmbedder() if embedder_kind == "tfidf" else HashingEmbedder()
    )
    index = VectorIndex(embedder=embedder)
    index.add_documents(docs)
    index.build()
    index.save(index_dir)
    return index


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the vector index.")
    parser.add_argument("--embedder", choices=["hashing", "tfidf"], default="hashing")
    parser.add_argument("--index-dir", type=Path, default=config.VECTOR_INDEX_DIR)
    parser.add_argument(
        "--no-warehouse",
        action="store_true",
        help="Build from landing-zone files only (skip the warehouse table).",
    )
    args = parser.parse_args(argv)
    index = build_index(
        args.index_dir,
        embedder_kind=args.embedder,
        use_warehouse=not args.no_warehouse,
    )
    print(
        f"Indexed {len(index)} documents "
        f"(embedder={args.embedder}) -> {args.index_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
