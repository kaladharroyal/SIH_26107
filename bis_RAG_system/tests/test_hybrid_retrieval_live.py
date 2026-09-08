"""
Verification and Latency Benchmark Suite for True Hybrid Retrieval (BM25 + Dense Neural Search)
Tests:
1. Index health, authentic counts, and non-empty vector store.
2. Topical semantic relevance (IS 1786 steel chemistry, IS 269 cement strength) vs annual reports.
3. Unicode and Indic language query handling (Hindi/Devanagari).
4. CPU Latency profiling across dense, sparse, and RRF stages.
"""

import os
import sys
import time
from pathlib import Path

# Fix Windows console Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pytest

pytestmark = pytest.mark.live

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import BISRAGPipeline


def test_index_health_and_count(pipeline):
    print("\n" + "=" * 60)
    print("TEST 1: Index Health and Authentic Corpus Counts")
    print("=" * 60)
    store = pipeline.retrieval.vector_store
    bm25 = pipeline.retrieval.bm25_index
    encoder = pipeline.retrieval.encoder

    print(f"  Vector Store Count: {store.count}")
    print(f"  Vector Matrix Shape: {store.embeddings.shape if store.embeddings is not None else None}")
    print(f"  BM25 Documents Count: {bm25.n_docs if bm25 else 0}")
    print(f"  Neural Encoder: {type(encoder).__name__}")
    print(f"  Fast Retrieval Mode: {pipeline.use_fast_retrieval}")

    assert store.count >= 500, f"Expected >= 500 authentic chunks, got {store.count}"
    assert store.embeddings.shape[1] == 384, f"Expected 384 dimensions, got {store.embeddings.shape[1]}"
    assert bm25.n_docs >= 500, f"Expected BM25 to have >= 500 docs, got {bm25.n_docs}"
    print("  [PASS] Index health verified: authentic standards corpus loaded.")


def test_topical_semantic_relevance(pipeline):
    print("\n" + "=" * 60)
    print("TEST 2: Topical Semantic Relevance (Technical Standards vs Administrative Reports)")
    print("=" * 60)

    # 1. Query for IS 1786 (Steel reinforcement / TMT bars)
    q1 = "chemical requirements for steel reinforcement bars IS 1786"
    print(f"\nEvaluating Query 1: '{q1}'")
    hits1 = pipeline.retrieval.retrieve(q1, top_n=3)
    assert len(hits1) > 0, "Expected retrieval hits for IS 1786"

    top1 = hits1[0]["doc"]
    print(f"  Top Hit ID:     {top1.get('chunk_id')}")
    print(f"  Standard:       {top1.get('is_number')}")
    print(f"  Clause Title:   {top1.get('clause_title')}")
    print(f"  Category:       {top1.get('category')}")
    print(f"  RRF Score:      {hits1[0].get('rrf_score'):.5f}")
    print(f"  Text Snippet:   {top1.get('text', '')[:160]}...")

    # Strict topical assertions:
    combined_text1 = " ".join([h["doc"].get("text", "") for h in hits1]).lower()
    assert "is 1786" in combined_text1 or "steel" in combined_text1 or "carbon" in combined_text1, (
        "FAIL: Top hits for IS 1786 do not mention steel or chemical requirements!"
    )
    # Assert absence of annual report accounting
    assert "balance sheet" not in combined_text1 and "auditor's report" not in combined_text1, (
        "FAIL: Retrieved text contains annual report balance sheets instead of technical standards!"
    )
    print("  [PASS] IS 1786 retrieved authentic steel technical requirements without annual report noise.")

    # 2. Query for IS 269 (Cement specifications)
    q2 = "Ordinary Portland Cement compressive strength 43 Grade IS 269"
    print(f"\nEvaluating Query 2: '{q2}'")
    hits2 = pipeline.retrieval.retrieve(q2, top_n=3)
    assert len(hits2) > 0, "Expected retrieval hits for IS 269"

    top2 = hits2[0]["doc"]
    print(f"  Top Hit ID:     {top2.get('chunk_id')}")
    print(f"  Standard:       {top2.get('is_number')}")
    print(f"  Clause Title:   {top2.get('clause_title')}")
    print(f"  Category:       {top2.get('category')}")
    print(f"  RRF Score:      {hits2[0].get('rrf_score'):.5f}")
    print(f"  Text Snippet:   {top2.get('text', '')[:160]}...")

    combined_text2 = " ".join([h["doc"].get("text", "") for h in hits2]).lower()
    assert "cement" in combined_text2 or "is 269" in combined_text2 or "compressive" in combined_text2, (
        "FAIL: Top hits for IS 269 do not mention cement specifications!"
    )
    print("  [PASS] IS 269 retrieved authentic cement specification clauses.")


def test_multilingual_unicode_retrieval(pipeline):
    print("\n" + "=" * 60)
    print("TEST 3: Multilingual & Unicode Indic Query Retrieval")
    print("=" * 60)

    hindi_query = "बीआईएस प्रयोगशाला परीक्षण और मानक"
    print(f"Evaluating Hindi Query: '{hindi_query}'")
    hits = pipeline.retrieval.retrieve(hindi_query, top_n=3)
    print(f"  Retrieved hits: {len(hits)}")
    if hits:
        top = hits[0]["doc"]
        print(f"  Top Hit ID: {top.get('chunk_id')}")
        print(f"  Standard:   {top.get('is_number')}")
        print(f"  Title:      {top.get('clause_title')}")
    assert len(hits) > 0, "FAIL: Dense retrieval returned 0 hits for Hindi Unicode query!"
    print("  [PASS] Unicode Indic query retrieved results successfully.")


def test_cpu_latency_benchmark(pipeline):
    print("\n" + "=" * 60)
    print("TEST 4: CPU Latency Benchmark & SLA Profiling")
    print("=" * 60)

    benchmark_queries = [
        "chemical requirements for steel reinforcement bars IS 1786",
        "Ordinary Portland Cement compressive strength IS 269",
        "electrical safety requirements for LED luminaires IS 10322",
        "water for analytical laboratory use IS 1070",
    ]

    for q in benchmark_queries:
        # 1. Sparse BM25 time
        t0 = time.perf_counter()
        sparse_hits = pipeline.retrieval.sparse_search(q, top_k=20)
        t_sparse_ms = (time.perf_counter() - t0) * 1000

        # 2. Dense Neural Vector Search time (including CPU query encoding)
        t1 = time.perf_counter()
        dense_hits = pipeline.retrieval.dense_search(q, top_k=20)
        t_dense_ms = (time.perf_counter() - t1) * 1000

        # 3. Full Hybrid Pipeline (Sparse + Dense + RRF + Rerank)
        t2 = time.perf_counter()
        full_hits = pipeline.retrieval.retrieve(q, top_n=5)
        t_full_ms = (time.perf_counter() - t2) * 1000

        print(
            f"Query: '{q[:45]}...'\n"
            f"  Sparse BM25:  {t_sparse_ms:6.2f} ms ({len(sparse_hits)} hits)\n"
            f"  Dense Neural: {t_dense_ms:6.2f} ms ({len(dense_hits)} hits)\n"
            f"  Full Hybrid:  {t_full_ms:6.2f} ms (RRF + Rerank)\n"
        )
        # Verify reasonable CPU latency SLA (< 1500ms on CPU)
        assert t_full_ms < 1500.0, f"Full hybrid retrieval latency {t_full_ms:.2f}ms exceeded 1500ms threshold!"

    print("  [PASS] All hybrid retrieval queries completed well within CPU SLA target (<1500ms).")


def main():
    print("Initializing BIS RAG Pipeline with use_fast_retrieval=False...")
    pipeline = BISRAGPipeline(use_fast_retrieval=False)

    test_index_health_and_count(pipeline)
    test_topical_semantic_relevance(pipeline)
    test_multilingual_unicode_retrieval(pipeline)
    test_cpu_latency_benchmark(pipeline)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! Feature 1 is Fully Verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
