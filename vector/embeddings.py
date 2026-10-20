"""Dependency-light text embeddings built on numpy.

Two embedders are provided:

* ``HashingEmbedder`` — feature hashing over word n-grams (the "hashing
  trick"). Stateless, deterministic, works for any text including queries
  about terms never seen at index time. This is the default.
* ``TfidfEmbedder`` — a from-scratch TF-IDF vectorizer (no scikit-learn
  dependency). Must be fitted on the corpus; slightly better ranking on
  in-corpus vocabulary.

Both return L2-normalized float32 vectors, so cosine similarity is a plain
dot product — everything runs offline with zero model downloads.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization."""
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Return unigrams plus bigrams (bigrams capture phrases like 'as2 timeout')."""
    grams = list(tokens)
    for i in range(len(tokens) - n + 1):
        grams.append(" ".join(tokens[i : i + n]))
    return grams


class HashingEmbedder:
    """Feature-hashing embedder: signed token counts into a fixed dimension."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.kind = "hashing"

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return idx, sign

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for gram in ngrams(tokenize(text)):
            idx, sign = self._hash(gram)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])

    def to_meta(self) -> dict:
        return {"kind": self.kind, "dim": self.dim}

    @classmethod
    def from_meta(cls, meta: dict) -> "HashingEmbedder":
        return cls(dim=int(meta.get("dim", 384)))


class TfidfEmbedder:
    """From-scratch TF-IDF embedder (fit on corpus, then transform)."""

    kind = "tfidf"

    def __init__(self, max_features: int = 4096) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.dim = 0

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()
        for doc in corpus:
            grams = ngrams(tokenize(doc))
            term_freq.update(grams)
            doc_freq.update(set(grams))
        terms = [
            term
            for term, _ in term_freq.most_common(self.max_features)
        ]
        self.vocab = {term: i for i, term in enumerate(sorted(terms))}
        self.dim = len(self.vocab)
        n_docs = max(len(corpus), 1)
        self.idf = np.array(
            [
                math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
                for term, _ in sorted(self.vocab.items(), key=lambda kv: kv[1])
            ],
            dtype=np.float32,
        )
        return self

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = Counter(ngrams(tokenize(text)))
        for term, count in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = (1.0 + math.log(count)) * float(self.idf[idx])
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])

    def to_meta(self) -> dict:
        return {
            "kind": self.kind,
            "dim": self.dim,
            "vocab": self.vocab,
            "idf": self.idf.tolist(),
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "TfidfEmbedder":
        obj = cls()
        obj.vocab = {k: int(v) for k, v in meta.get("vocab", {}).items()}
        obj.dim = int(meta.get("dim", len(obj.vocab)))
        obj.idf = np.array(meta.get("idf", []), dtype=np.float32)
        return obj


def embedder_from_meta(meta: dict):
    """Rebuild an embedder from its serialized metadata."""
    if meta.get("kind") == "tfidf":
        return TfidfEmbedder.from_meta(meta)
    return HashingEmbedder.from_meta(meta)
