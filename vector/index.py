"""Local vector index: numpy embedding matrix + JSON document store.

A deliberately small, transparent stand-in for a managed vector database
(pgvector / Pinecone / FAISS in production). Vectors are stored as a single
``.npy`` matrix, documents and embedder metadata as JSON — the whole index
is a directory that can be copied, versioned, or mounted into a container.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from vector.chunking import Document
from vector.embeddings import HashingEmbedder, embedder_from_meta


class VectorIndex:
    """In-memory vector index with disk persistence."""

    def __init__(self, embedder=None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    # -- build --------------------------------------------------------------
    def add_documents(self, docs: list[Document]) -> None:
        self.documents.extend(docs)
        self._matrix = None  # invalidate; rebuilt lazily

    def build(self) -> "VectorIndex":
        """(Re)build the embedding matrix from the current documents."""
        texts = [d.text for d in self.documents]
        if hasattr(self.embedder, "fit") and getattr(
            self.embedder, "dim", 0
        ) == 0:
            self.embedder.fit(texts)
        self._matrix = self.embedder.embed_many(texts)
        return self

    def __len__(self) -> int:
        return len(self.documents)

    # -- search ---------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Return the top-k documents by cosine similarity to ``query``."""
        if not self.documents:
            return []
        if self._matrix is None:
            self.build()
        query_vec = self.embedder.embed(query)
        scores = self._matrix @ query_vec  # vectors are L2-normalized
        order = np.argsort(-scores)[:k]
        return [
            (self.documents[int(i)], float(scores[int(i)]))
            for i in order
            if scores[int(i)] > 0
        ]

    # -- persistence ------------------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._matrix is None:
            self.build()
        np.save(directory / "vectors.npy", self._matrix)
        (directory / "documents.json").write_text(
            json.dumps([d.to_dict() for d in self.documents], indent=2),
            encoding="utf-8",
        )
        (directory / "embedder.json").write_text(
            json.dumps(self.embedder.to_meta(), indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        directory = Path(directory)
        meta = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))
        index = cls(embedder=embedder_from_meta(meta))
        docs = json.loads(
            (directory / "documents.json").read_text(encoding="utf-8")
        )
        index.documents = [Document.from_dict(d) for d in docs]
        index._matrix = np.load(directory / "vectors.npy")
        return index
