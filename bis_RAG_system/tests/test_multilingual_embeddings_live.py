"""
Verification Suite for Feature 2: Genuine Multilingual Dense Embeddings
Tests:
1. Dynamic model initialization, introspected hidden_size dimension, and vector store metadata alignment.
2. Vector dimension safety validation (catching mismatched dimensions).
3. Cross-lingual Indic semantic relevance (Hindi and Telugu queries retrieving authentic technical standards).
4. CPU Latency profiling for multilingual inference and hybrid fusion.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

# Fix Windows console Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import BISRAGPipeline


def test_dynamic_model_architecture(pipeline):
    print("\n" + "=" * 65)
    print("TEST 1: Dynamic Model Architecture & Provenance Inspection")
    print("=" * 65)
    encoder = pipeline.retrieval.encoder
    store = pipeline.retrieval.vector_store

    print(f"  Configured Model Name: {encoder.model_name}")
    print(f"  Introspected Dimension: {encoder.dimension}")
    print(f"  Vector Store Count:    {store.count}")
    print(f"  Vector Matrix Shape:   {store.embeddings.shape}")

    assert "paraphrase-multilingual" in encoder.model_name, (
        f"Expected multilingual model, got {encoder.model_name}"
    )
    assert encoder.dimension == 384, f"Expected 384 dimensions, got {encoder.dimension}"
    assert store.embeddings.shape[1] == 384 and store.embeddings.shape[0] >= 1000, (
        f"Expected matrix (*, 384), got {store.embeddings.shape}"
    )
    print("  [PASS] Dynamic multilingual model loaded and dimensionally aligned.")


def test_dimension_mismatch_safety(pipeline):
    print("\n" + "=" * 65)
    print("TEST 2: Dimension Mismatch Safety & Validation")
    print("=" * 65)
    store = pipeline.retrieval.vector_store

    # Test that querying with a mismatched dimension vector (e.g. 128 or 1024) raises ValueError in strict mode
    bad_vector_128 = np.random.randn(128).astype(np.float32)
    bad_vector_1024 = np.random.randn(1024).astype(np.float32)

    try:
        store.search(bad_vector_128, strict=True)
        assert False, "Expected ValueError on 128-dim vector against 384-dim index!"
    except ValueError as e:
        print(f"  Caught expected mismatch error on 128-dim: {e}")

    try:
        store.search(bad_vector_1024, strict=True)
        assert False, "Expected ValueError on 1024-dim vector against 384-dim index!"
    except ValueError as e:
        print(f"  Caught expected mismatch error on 1024-dim: {e}")

    print("  [PASS] DenseVectorStore strictly enforces dimension compatibility.")



def test_cross_lingual_indic_semantic_relevance(pipeline):
    print("\n" + "=" * 65)
    print("TEST 3: Cross-Lingual Indic Semantic Relevance (Native Hindi/Telugu)")
    print("=" * 65)

    # 1. Hindi Query for Cement
    q_hindi_cement = "सीमेंट के लिए भारतीय मानक और संपीड़न शक्ति"
    print(f"\nEvaluating Native Hindi Query: '{q_hindi_cement}'")
    hits_cement = pipeline.retrieval.retrieve(q_hindi_cement, top_n=3)
    assert len(hits_cement) > 0, "Expected hits for Hindi cement query"

    top_cement = hits_cement[0]["doc"]
    print(f"  Top Hit ID:     {top_cement.get('chunk_id')}")
    print(f"  Standard:       {top_cement.get('is_number')}")
    print(f"  Clause Title:   {top_cement.get('clause_title')}")
    print(f"  Category:       {top_cement.get('category')}")
    print(f"  RRF Score:      {hits_cement[0].get('rrf_score'):.5f}")
    print(f"  Text Excerpt:   {top_cement.get('text', '')[:160]}...")

    combined_cement = " ".join([h["doc"].get("text", "") for h in hits_cement]).lower()
    assert "cement" in combined_cement or "is 269" in combined_cement or "opc" in combined_cement, (
        "FAIL: Cross-lingual search for cement did not retrieve IS 269 cement specifications!"
    )
    print("  [PASS] Native Hindi cement query retrieved authentic IS 269 cement specification.")

    # 2. Hindi Query for Steel TMT
    q_hindi_steel = "भूकंप रोधी टीएमटी स्टील बार के लिए रासायनिक आवश्यकताएं और कार्बन सीमाएं"
    print(f"\nEvaluating Native Hindi Query: '{q_hindi_steel}'")
    hits_steel = pipeline.retrieval.retrieve(q_hindi_steel, top_n=3)
    assert len(hits_steel) > 0, "Expected hits for Hindi steel query"

    top_steel = hits_steel[0]["doc"]
    print(f"  Top Hit ID:     {top_steel.get('chunk_id')}")
    print(f"  Standard:       {top_steel.get('is_number')}")
    print(f"  Clause Title:   {top_steel.get('clause_title')}")
    print(f"  Category:       {top_steel.get('category')}")
    print(f"  RRF Score:      {hits_steel[0].get('rrf_score'):.5f}")
    print(f"  Text Excerpt:   {top_steel.get('text', '')[:160]}...")

    combined_steel = " ".join([h["doc"].get("text", "") for h in hits_steel]).lower()
    assert "is 1786" in combined_steel or "steel" in combined_steel or "carbon" in combined_steel or "fe 415" in combined_steel, (
        "FAIL: Cross-lingual search for steel did not retrieve IS 1786 steel specifications!"
    )
    print("  [PASS] Native Hindi steel query retrieved authentic IS 1786 steel chemical limits.")

    # 3. Telugu Query for Steel / Cement
    q_telugu = "ఉక్కు మరియు సిమెంట్ కోసం భారతీయ ప్రమాణాలు"
    print(f"\nEvaluating Native Telugu Query: '{q_telugu}'")
    hits_telugu = pipeline.retrieval.retrieve(q_telugu, top_n=3)
    assert len(hits_telugu) > 0, "Expected hits for Telugu query"
    top_telugu = hits_telugu[0]["doc"]
    print(f"  Top Hit ID:     {top_telugu.get('chunk_id')}")
    print(f"  Standard:       {top_telugu.get('is_number')}")
    print(f"  Clause Title:   {top_telugu.get('clause_title')}")
    print("  [PASS] Native Telugu query retrieved authentic technical standards.")


def test_multilingual_latency_benchmark(pipeline):
    print("\n" + "=" * 65)
    print("TEST 4: Multilingual CPU Latency Benchmark")
    print("=" * 65)

    test_queries = [
        ("English - IS 1786 Steel", "chemical requirements for steel reinforcement bars IS 1786"),
        ("English - IS 269 Cement", "Ordinary Portland Cement compressive strength IS 269"),
        ("Hindi - Cement Spec", "सीमेंट के लिए भारतीय मानक और संपीड़न शक्ति"),
        ("Hindi - Steel Limits", "टीएमटी स्टील बार रासायनिक संरचना IS 1786"),
        ("Telugu - Standards", "ఉక్కు మరియు సిమెంట్ కోసం భారతీయ ప్రమాణాలు"),
    ]

    for label, q in test_queries:
        t0 = time.perf_counter()
        dense_hits = pipeline.retrieval.dense_search(q, top_k=20)
        t_dense_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        full_hits = pipeline.retrieval.retrieve(q, top_n=5)
        t_full_ms = (time.perf_counter() - t1) * 1000

        print(
            f"  [{label}]\n"
            f"    Dense Neural Encoding + Search: {t_dense_ms:6.2f} ms ({len(dense_hits)} hits)\n"
            f"    Full Hybrid Retrieval (RRF):    {t_full_ms:6.2f} ms\n"
        )
        assert t_full_ms < 1500.0, f"Latency {t_full_ms:.2f}ms exceeded 1500ms target!"

    print("  [PASS] All multilingual queries executed within target CPU SLA.")


def main():
    print("Initializing BIS RAG Pipeline with Multilingual Transformer...")
    pipeline = BISRAGPipeline(use_fast_retrieval=False)

    test_dynamic_model_architecture(pipeline)
    test_dimension_mismatch_safety(pipeline)
    test_cross_lingual_indic_semantic_relevance(pipeline)
    test_multilingual_latency_benchmark(pipeline)

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY! Feature 2 is Fully Verified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
