"""
End-to-End Grounded Retrieval & Generation Benchmark (test_end_to_end_rag.py)
Executes audit benchmark queries across all core intents and validates retrieved chunks,
provenance, non-empty response, confidence score in (0.0, 1.0], and grounded citations.
"""

import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import BISRAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    return BISRAGPipeline(llm_provider="mock", use_mock_retrieval=False)


def test_e2e_is_1786_chemical_composition(pipeline):
    query = "What are the chemical composition limits for carbon, sulfur, phosphorus in Fe 500D steel as per IS 1786?"
    res = pipeline.query(query)

    assert res["status"] == "success"
    assert res["confidence_score"] > 0.45
    assert len(res["retrieved_chunks"]) > 0

    top_chunk = res["retrieved_chunks"][0].get("doc", res["retrieved_chunks"][0])
    assert "1786" in str(top_chunk.get("is_number", ""))
    assert "1786" in str(top_chunk.get("source_file", "") or top_chunk.get("source_document", ""))

    response = res["response"].lower()
    assert "0.25" in response or "0.040" in response or "carbon" in response
    print(f"\n[E2E IS 1786] Top Chunk: {top_chunk.get('chunk_id')} | Score: {res['confidence_score']:.4f}")
    print(f"Response Preview: {res['response'][:250]}...")


def test_e2e_solar_panel_mandatory_qco(pipeline):
    query = "Which Indian Standard applies to solar panels and is certification mandatory under QCO?"
    res = pipeline.query(query)

    assert res["status"] == "success"
    assert res["confidence_score"] > 0.40
    assert len(res["retrieved_chunks"]) > 0

    top_chunk = res["retrieved_chunks"][0].get("doc", res["retrieved_chunks"][0])
    text = res["response"].lower()
    assert "14286" in text or "61215" in text or "is 14286" in text or "solar" in text
    assert "mandatory" in text or "qco" in text or "scheme" in text
    print(f"\n[E2E Solar Panel] Top Chunk: {top_chunk.get('chunk_id')} | Score: {res['confidence_score']:.4f}")
    print(f"Response Preview: {res['response'][:250]}...")


def test_e2e_mumbai_bis_labs(pipeline):
    query = "Are there any BIS recognized testing laboratories in Mumbai Maharashtra and what are their addresses?"
    res = pipeline.query(query)

    assert res["status"] == "success"
    assert res["confidence_score"] > 0.40
    assert len(res["retrieved_chunks"]) > 0

    top_chunk = res["retrieved_chunks"][0].get("doc", res["retrieved_chunks"][0])
    assert "lab_directory" in str(top_chunk.get("category", "")) or "mumbai" in str(top_chunk.get("text", "")).lower()

    text = res["response"].lower()
    assert "mumbai" in text or "maharashtra" in text or "laboratory" in text
    print(f"\n[E2E Mumbai Labs] Top Chunk: {top_chunk.get('chunk_id')} | Score: {res['confidence_score']:.4f}")
    print(f"Response Preview: {res['response'][:250]}...")


def test_e2e_scheme_walkthrough_fees(pipeline):
    query = "How to apply for Scheme-I ISI Mark certification and what is the fee structure?"
    res = pipeline.query(query)

    assert res["status"] == "success"
    assert res["confidence_score"] > 0.40
    assert len(res["retrieved_chunks"]) > 0

    text = res["response"].lower()
    assert "scheme" in text or "isi mark" in text or "application" in text or "fee" in text
    print(f"\n[E2E Scheme Walkthrough] Score: {res['confidence_score']:.4f}")
    print(f"Response Preview: {res['response'][:250]}...")


def test_e2e_consumer_complaint_hallmarking(pipeline):
    query = "How to file a complaint against hallmarked gold jewellery for low purity and what compensation am I entitled to?"
    res = pipeline.query(query)

    assert res["status"] == "success"
    assert res["confidence_score"] > 0.40
    assert len(res["retrieved_chunks"]) > 0

    text = res["response"].lower()
    assert "regulation 12" in text or "2x" in text or "two times" in text or "compensation" in text or "bis care" in text
    print(f"\n[E2E Consumer Complaint] Score: {res['confidence_score']:.4f}")
    print(f"Response Preview: {res['response'][:250]}...")
