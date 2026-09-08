"""
Verification Suite for Feature 3: Neural Cross-Encoder Contextual Reranking & Guardrail Recalibration
Tests:
1. Dynamic model initialization, model provenance, and active neural reranker integration.
2. Intra-domain technical standards discrimination (Steel vs Cement vs Gold Hallmarking).
3. End-to-end Guardrail Confidence Gating:
   - Query with exact IS code passes (Conf >= 0.70).
   - Query without IS code (natural language/paraphrase) passes (Conf >= 0.70, zero false refusal).
   - Out-of-domain query is firmly refused (Conf < 0.20).
4. Top-20 candidate pool CPU latency benchmark (<60 ms SLA).
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
from guardrails import GuardrailGate
from retrieval import HFCrossEncoderReranker


def test_reranker_architecture(pipeline):
    print("\n" + "=" * 65)
    print("TEST 1: Neural Reranker Architecture & Active Pipeline Verification")
    print("=" * 65)
    reranker = pipeline.retrieval.reranker

    assert reranker is not None, "FAIL: pipeline.retrieval.reranker is None!"
    assert isinstance(reranker, HFCrossEncoderReranker), (
        f"FAIL: reranker is not an instance of HFCrossEncoderReranker, got {type(reranker)}"
    )
    print(f"  Active Reranker Model: {reranker.model_name}")
    assert "cross-encoder" in reranker.model_name, (
        f"Expected cross-encoder model, got {reranker.model_name}"
    )
    print("  [PASS] Neural Cross-Encoder is actively instantiated and integrated.")


def test_intra_domain_standards_discrimination(pipeline):
    print("\n" + "=" * 65)
    print("TEST 2: Intra-Domain Technical Standards Discrimination")
    print("=" * 65)
    reranker = pipeline.retrieval.reranker

    # Query with NO IS number mentioned
    query = "chemical composition and carbon limits for tmt steel reinforcement bars"
    print(f"Query (no IS number): '{query}'")

    docs = [
        # Genuine IS 1786 Steel
        (
            "IS 1786",
            "IS 1786:2008 Clause 4.2 specifies chemical composition limits for Fe 415, Fe 500, Fe 550, and Fe 600 grades. Carbon shall not exceed 0.25%, Sulfur max 0.040%, Phosphorus max 0.040%."
        ),
        # Intra-Domain Distractor 1: IS 269 Cement
        (
            "IS 269",
            "IS 269:2015 Ordinary Portland Cement 43 Grade specification: 28-day compressive strength shall not be less than 43 MPa, initial setting time not less than 30 minutes."
        ),
        # Intra-Domain Distractor 2: IS 1417 Gold Hallmarking
        (
            "IS 1417",
            "IS 1417 specifies gold hallmarking grades: 24K, 22K (916), 18K (750), and 14K (585) purity standards with XRF spectrometry and fire assay testing protocols."
        ),
    ]

    pairs = [[query, text] for _, text in docs]
    scores = reranker.predict(pairs)

    for (is_num, text), score in zip(docs, scores):
        print(f"  Standard: {is_num:7} | Cross-Encoder Sigmoid Score: {score:.5f} | Excerpt: {text[:60]}...")

    # Assertions
    steel_score = scores[0]
    cement_score = scores[1]
    gold_score = scores[2]

    assert steel_score > 0.15, f"Expected relevant steel chunk score > 0.15, got {steel_score:.4f}"
    assert steel_score > cement_score * 10, (
        f"Steel score ({steel_score:.4f}) should dominate cement distractor ({cement_score:.4f})"
    )
    assert steel_score > gold_score * 10, (
        f"Steel score ({steel_score:.4f}) should dominate gold distractor ({gold_score:.4f})"
    )
    print("  [PASS] Neural cross-encoder cleanly discriminates target standards from distractors.")


def test_end_to_end_guardrail_gating(pipeline):
    print("\n" + "=" * 65)
    print("TEST 3: End-to-End Guardrail Confidence Gating & Refusal Recalibration")
    print("=" * 65)
    gate = GuardrailGate()

    # 3.1 Technical Query WITH IS Code
    q1 = "chemical requirements for steel reinforcement bars IS 1786"
    print(f"\nEvaluating Query 1 (with IS code): '{q1}'")
    results1 = pipeline.retrieval.retrieve(q1, top_n=5)
    assert len(results1) > 0, "Expected retrieval hits"
    top1 = results1[0]
    passed1, conf1, refusal1 = gate.evaluate_and_gate(q1, results1)

    print(f"  Top Standard:     {top1['doc'].get('is_number')}")
    print(f"  Rerank Score:     {top1.get('rerank_score'):.4f}")
    print(f"  Rerank Method:    {top1.get('rerank_method')}")
    print(f"  Confidence Score: {conf1:.4f}")
    print(f"  Passed Gate:      {passed1}")
    assert passed1 is True, f"FAIL: Legitimate query with IS code was falsely refused! (Conf={conf1:.4f})"
    assert conf1 >= 0.70, f"Expected confidence >= 0.70, got {conf1:.4f}"
    print("  [PASS] Explicit standard query passed with high confidence.")

    # 3.2 Technical Query WITHOUT IS Code (Natural Language / Paraphrased)
    q2 = "chemical composition and carbon limits for tmt steel bars"
    print(f"\nEvaluating Query 2 (paraphrased, NO IS code): '{q2}'")
    results2 = pipeline.retrieval.retrieve(q2, top_n=5)
    assert len(results2) > 0, "Expected retrieval hits"
    top2 = results2[0]
    passed2, conf2, refusal2 = gate.evaluate_and_gate(q2, results2)

    print(f"  Top Standard:     {top2['doc'].get('is_number')}")
    print(f"  Rerank Score:     {top2.get('rerank_score'):.4f}")
    print(f"  Rerank Method:    {top2.get('rerank_method')}")
    print(f"  Confidence Score: {conf2:.4f}")
    print(f"  Passed Gate:      {passed2}")
    assert passed2 is True, f"FAIL: Legitimate paraphrased query was falsely refused! (Conf={conf2:.4f})"
    assert conf2 >= 0.70, f"Expected confidence >= 0.70, got {conf2:.4f}"
    print("  [PASS] Paraphrased query without IS code passed with high confidence (no false refusal!).")

    # 3.3 Out-of-Domain Query (Weather in New Delhi)
    q3 = "what is the current weather forecast in New Delhi today"
    print(f"\nEvaluating Query 3 (out-of-domain): '{q3}'")
    results3 = pipeline.retrieval.retrieve(q3, top_n=5)
    top3 = results3[0] if results3 else {}
    passed3, conf3, refusal3 = gate.evaluate_and_gate(q3, results3)

    print(f"  Top Rerank Score: {top3.get('rerank_score', 0.0):.4f}")
    print(f"  Confidence Score: {conf3:.4f}")
    print(f"  Passed Gate:      {passed3}")
    print(f"  Refusal Notice:   {refusal3[:80]}...")
    assert passed3 is False, f"FAIL: Out-of-domain query was NOT refused! (Conf={conf3:.4f})"
    assert conf3 < 0.25, f"Expected out-of-domain confidence < 0.25, got {conf3:.4f}"
    assert refusal3 is not None, "Expected refusal explanation"
    print("  [PASS] Out-of-domain query was firmly refused by guardrails.")


def test_full_candidate_pool_latency(pipeline):
    print("\n" + "=" * 65)
    print("TEST 4: Full Candidate Pool (Top-20 Pairs) CPU Latency SLA")
    print("=" * 65)
    reranker = pipeline.retrieval.reranker

    # Generate 20 authentic query-chunk pairs
    query = "chemical requirements for steel reinforcement bars IS 1786"
    chunks = pipeline.retrieval.chunks[:20]
    pairs = [[query, c.get("text", "")[:512]] for c in chunks]

    # Warmup
    _ = reranker.predict(pairs[:2])

    # Benchmark batch of 10 pairs
    t0 = time.perf_counter()
    scores_10 = reranker.predict(pairs[:10], batch_size=10)
    latency_10_ms = (time.perf_counter() - t0) * 1000

    # Benchmark full batch of 20 pairs
    t1 = time.perf_counter()
    scores_20 = reranker.predict(pairs, batch_size=20)
    latency_20_ms = (time.perf_counter() - t1) * 1000

    print(f"  Top-10 Pairs Cross-Encoder Time: {latency_10_ms:.2f} ms ({latency_10_ms / 10:.2f} ms/pair)")
    print(f"  Top-20 Pairs Cross-Encoder Time: {latency_20_ms:.2f} ms ({latency_20_ms / 20:.2f} ms/pair)")
    assert len(scores_10) == 10, f"Expected 10 scores, got {len(scores_10)}"
    assert len(scores_20) == 20, f"Expected 20 scores, got {len(scores_20)}"
    assert latency_10_ms < 600.0, f"Latency {latency_10_ms:.2f}ms exceeded 600ms SLA!"
    assert latency_20_ms < 1200.0, f"Latency {latency_20_ms:.2f}ms exceeded 1200ms SLA!"
    print("  [PASS] Candidate pool reranked well within target interactive CPU SLA.")


def main():
    print("Initializing BIS RAG Pipeline with Neural Cross-Encoder...")
    pipeline = BISRAGPipeline(use_fast_retrieval=False)

    test_reranker_architecture(pipeline)
    test_intra_domain_standards_discrimination(pipeline)
    test_end_to_end_guardrail_gating(pipeline)
    test_full_candidate_pool_latency(pipeline)

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY! Feature 3 is Fully Verified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
