"""RAG: retrieve incident chunks and assemble an answer.

Retrieval is always local (the vector index). Generation is optional:

* If ``OPENAI_API_KEY`` is set and the ``openai`` package is importable, the
  retrieved chunks are sent to the model with the assembled prompt.
* Otherwise an *extractive* answer is returned — a synthesis built purely
  from the retrieved chunks — so the demo works fully offline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from vector.chunking import Document
from vector.index import VectorIndex

PROMPT_TEMPLATE = """You are an integration operations assistant for a semiconductor
supply-chain B2B platform (Dell Boomi, EDI X12, SAP). Answer the question
using ONLY the incident records below. If the records do not contain the
answer, say so. Cite partner names, error codes, and timestamps when relevant.

Incident records:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[tuple[Document, float]]) -> str:
    lines = []
    for i, (doc, score) in enumerate(chunks, start=1):
        lines.append(f"[{i}] (similarity {score:.3f}) {doc.text}")
    return "\n".join(lines) if lines else "(no matching incident records)"


def build_prompt(question: str, chunks: list[tuple[Document, float]]) -> str:
    """Assemble the grounded RAG prompt from retrieved chunks."""
    return PROMPT_TEMPLATE.format(
        context=format_context(chunks), question=question
    )


def extractive_answer(question: str, chunks: list[tuple[Document, float]]) -> str:
    """Build an answer purely from retrieved chunks (offline mode)."""
    if not chunks:
        return (
            f"No incident records matched the question: {question!r}. "
            "Try broadening the query (partner name, error code, or process)."
        )
    partners = sorted(
        {
            d.metadata.get("partner_name")
            for d, _ in chunks
            if d.metadata.get("partner_name")
        }
    )
    error_codes = sorted(
        {
            d.metadata.get("error_code")
            for d, _ in chunks
            if d.metadata.get("error_code")
        }
    )
    lines = [
        f"Based on {len(chunks)} retrieved incident record(s) for: {question}",
    ]
    if partners:
        lines.append(f"Partners involved: {', '.join(partners)}.")
    if error_codes:
        lines.append(f"Error codes seen: {', '.join(error_codes)}.")
    lines.append("Most relevant records:")
    for doc, score in chunks[:3]:
        lines.append(f"- ({score:.3f}) {doc.text}")
    return "\n".join(lines)


def _llm_answer(prompt: str) -> Optional[str]:
    """Call an OpenAI-compatible chat API if configured; else return None."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # optional dependency
    except ImportError:
        return None
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception:
        return None  # fall back to extractive answer on any API failure


class RAGAnswerer:
    """Question answering over the incident vector index."""

    def __init__(self, index: VectorIndex, k: int = 5) -> None:
        self.index = index
        self.k = k

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.index.search(question, k=self.k)

    def answer(self, question: str) -> dict:
        chunks = self.retrieve(question)
        prompt = build_prompt(question, chunks)
        llm = _llm_answer(prompt)
        if llm:
            return {"answer": llm, "mode": "llm", "prompt": prompt,
                    "sources": [d.doc_id for d, _ in chunks]}
        return {
            "answer": extractive_answer(question, chunks),
            "mode": "extractive",
            "prompt": prompt,
            "sources": [d.doc_id for d, _ in chunks],
        }


def main(argv: Optional[list[str]] = None) -> int:
    import config

    parser = argparse.ArgumentParser(description="Ask the incident RAG engine.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=config.VECTOR_INDEX_DIR)
    parser.add_argument(
        "--show-prompt", action="store_true", help="Print the assembled prompt."
    )
    args = parser.parse_args(argv)

    index = VectorIndex.load(args.index_dir)
    result = RAGAnswerer(index, k=args.k).answer(args.question)
    print(f"[mode: {result['mode']}]")
    print(result["answer"])
    if args.show_prompt:
        print("\n--- assembled prompt ---\n")
        print(result["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
