"""Tests for embeddings, the vector index, semantic search, and RAG."""

from __future__ import annotations

import numpy as np
import pytest

from vector.chunking import Document, chunk_execution_records
from vector.embeddings import HashingEmbedder, TfidfEmbedder
from vector.index import VectorIndex
from vector.rag import RAGAnswerer, build_prompt, extractive_answer

RECORDS = [
    {
        "execution_id": "e1",
        "partner_id": "PTNR-1001",
        "partner_name": "GlobalFoundries",
        "process_name": "EDI 856 ASN Outbound",
        "transaction_type": "856",
        "direction": "OUTBOUND",
        "status": "FAILED",
        "start_time": "2026-10-01T08:15:00+00:00",
        "records_processed": 120,
        "records_failed": 40,
        "duration_ms": 4200,
        "error_code": "E_CERT_042",
        "error_message": "SSL certificate expired for AS2 partner channel",
        "source_system": "Dell Boomi",
        "target_system": "SAP S/4HANA",
        "environment": "PROD",
    },
    {
        "execution_id": "e2",
        "partner_id": "PTNR-1002",
        "partner_name": "TSMC Logistics",
        "process_name": "SFTP Shipment Status Feed",
        "transaction_type": "214",
        "direction": "INBOUND",
        "status": "FAILED",
        "start_time": "2026-10-01T09:20:00+00:00",
        "records_processed": 55,
        "records_failed": 55,
        "duration_ms": 900,
        "error_code": "E_SFTP_209",
        "error_message": "SFTP authentication failure: permission denied (publickey)",
        "source_system": "SAP ECC",
        "target_system": "Dell Boomi",
        "environment": "PROD",
    },
    {
        "execution_id": "e3",
        "partner_id": "PTNR-1003",
        "partner_name": "ASE Group",
        "process_name": "EDI 850 Purchase Order Inbound",
        "transaction_type": "850",
        "direction": "INBOUND",
        "status": "SUCCESS",
        "start_time": "2026-10-01T10:00:00+00:00",
        "records_processed": 300,
        "records_failed": 0,
        "duration_ms": 1500,
        "error_code": None,
        "error_message": None,
        "source_system": "SAP ECC",
        "target_system": "Dell Boomi",
        "environment": "PROD",
    },
]


@pytest.fixture()
def index() -> VectorIndex:
    docs = chunk_execution_records(RECORDS)
    idx = VectorIndex(embedder=HashingEmbedder(dim=512))
    idx.add_documents(docs)
    idx.build()
    return idx


def test_chunking_skips_successful_records():
    docs = chunk_execution_records(RECORDS)
    assert len(docs) == 2  # only FAILED / WARNING records become incidents
    assert all(d.metadata["status"] != "SUCCESS" for d in docs)


def test_hashing_embedder_normalized_and_deterministic():
    emb = HashingEmbedder(dim=256)
    a = emb.embed("AS2 certificate expired")
    b = emb.embed("AS2 certificate expired")
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5


def test_tfidf_embedder_ranks_relevant_doc_first():
    docs = chunk_execution_records(RECORDS)
    emb = TfidfEmbedder().fit([d.text for d in docs])
    idx = VectorIndex(embedder=emb)
    idx.add_documents(docs)
    idx.build()
    results = idx.search("SFTP authentication permission denied", k=1)
    assert results
    assert results[0][0].metadata["error_code"] == "E_SFTP_209"


def test_semantic_search_finds_certificate_incident(index):
    results = index.search("SSL certificate expired AS2 channel", k=2)
    assert results, "expected at least one hit"
    top_doc, top_score = results[0]
    assert top_score > 0
    assert top_doc.metadata["error_code"] == "E_CERT_042"


def test_index_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    assert len(loaded) == len(index)
    results = loaded.search("certificate expired", k=1)
    assert results and results[0][0].metadata["error_code"] == "E_CERT_042"


def test_rag_extractive_answer_uses_retrieved_chunks(index, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = RAGAnswerer(index, k=3).answer("Why did GlobalFoundries ASN fail?")
    assert result["mode"] == "extractive"
    assert "GlobalFoundries" in result["answer"]
    assert "E_CERT_042" in result["answer"]
    assert result["sources"]


def test_rag_prompt_contains_context_and_question(index):
    chunks = index.search("certificate", k=2)
    prompt = build_prompt("What failed?", chunks)
    assert "What failed?" in prompt
    assert "certificate" in prompt.lower()


def test_extractive_answer_handles_no_hits():
    empty = VectorIndex(embedder=HashingEmbedder())
    answer = extractive_answer("anything", empty.search("anything"))
    assert "No incident records" in answer


def test_document_serialization_roundtrip():
    doc = Document(doc_id="x", text="hello", metadata={"a": 1})
    assert Document.from_dict(doc.to_dict()) == doc
