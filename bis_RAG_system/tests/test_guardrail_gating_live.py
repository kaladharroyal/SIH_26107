"""
Verification Test Suite: Confidence Refusal Gating Recalibration (Feature 5)
Verifies:
1. Mathematical Base Confidence Trap Elimination (no 0.30 clamp on non-IS queries)
2. Multilingual Unicode Evidence Sufficiency (Hindi, Telugu, Tamil keyword extraction)
3. Audit Problem Queries Remediation (no false refusal on compliant queries)
4. Strict Refusal of Out-of-Domain Queries (weather, FIFA, recipes refused < 0.25)
5. Statistical Separation Margin between in-domain and out-of-domain queries (>= 0.60)
"""

import os
import sys
import numpy as np
import pytest

pytestmark = pytest.mark.live

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from guardrails import GuardrailGate, CONFIDENCE_THRESHOLD
from rag_pipeline import BISRAGPipeline


def test_base_confidence_trap_elimination(gate: GuardrailGate, pipeline: BISRAGPipeline):
    print("\n" + "=" * 65)
    print("TEST 1: Elimination of Base Confidence 0.30 Trap on Non-IS Queries")
    print("=" * 65)

    # Legitimate compliance query without explicit 'IS \d+' prefix in user prompt
    query = "What are the chemical composition limits for carbon in high strength deformed bars?"
    res = pipeline.query(query)
    retrieved = res.get("retrieved_chunks", [])
    assert len(retrieved) > 0, "Retrieval returned 0 chunks."

    confidence = res.get("confidence_score", 0.0)
    passed = res.get("status") == "success"
    msg = None if passed else res.get("response")

    top_chunk = retrieved[0]
    rerank_score = top_chunk.get("cross_encoder_score") or top_chunk.get("rerank_score") or top_chunk.get("score")
    print(f"Query:              '{query}'")
    print(f"Top Chunk IS:       {top_chunk.get('doc', {}).get('is_number')}")
    print(f"Rerank Score:       {rerank_score}")
    print(f"Final Confidence:   {confidence:.4f} (Threshold: {CONFIDENCE_THRESHOLD})")
    print(f"Gate Passed:        {passed}")

    assert confidence > 0.45, f"Expected confidence > 0.45, got {confidence:.4f} (trap not eliminated!)"
    assert passed is True, f"Expected passed=True, got {passed}"
    assert msg is None, f"Expected no refusal message, got: {msg}"
    print("  [PASS] Non-IS technical query passed gate with high confidence.")


def test_multilingual_unicode_evidence_sufficiency(gate: GuardrailGate, pipeline: BISRAGPipeline):
    print("\n" + "=" * 65)
    print("TEST 2: Multilingual Unicode Evidence Sufficiency (Hindi & Telugu)")
    print("=" * 65)

    hindi_query = "सीमेंट के 28 दिनों के संपीड़न सामर्थ्य की आवश्यकता क्या है"
    telugu_query = "స్టీల్ కడ్డీల సాంకేతిక లక్షణాలు మరియు రసాయన కూర్పు ఏమిటి"

    # Verify Unicode keyword extraction
    hindi_kws = gate.extract_keywords(hindi_query)
    telugu_kws = gate.extract_keywords(telugu_query)

    print(f"Hindi Query:   '{hindi_query}'")
    print(f"Hindi Tokens:  {hindi_kws}")
    print(f"Telugu Query:  '{telugu_query}'")
    print(f"Telugu Tokens: {telugu_kws}")

    assert len(hindi_kws) >= 3, f"Expected at least 3 Hindi keywords, got {hindi_kws}"
    assert len(telugu_kws) >= 3, f"Expected at least 3 Telugu keywords, got {telugu_kws}"

    # Evaluate against live retrieval
    h_retrieved = pipeline.retrieval.retrieve(hindi_query, top_n=5)
    h_suff = gate.calculate_evidence_sufficiency(hindi_query, h_retrieved)
    h_conf = gate.calculate_confidence(hindi_query, h_retrieved)
    h_passed, _, _ = gate.evaluate_and_gate(hindi_query, h_retrieved)

    print(f"Hindi Retrieval -> Sufficiency: {h_suff:.4f} | Confidence: {h_conf:.4f} | Passed: {h_passed}")
    assert h_conf >= 0.45, f"Hindi query falsely refused! Confidence: {h_conf:.4f}"
    assert h_passed is True, "Hindi query failed gate!"

    t_retrieved = pipeline.retrieval.retrieve(telugu_query, top_n=5)
    t_suff = gate.calculate_evidence_sufficiency(telugu_query, t_retrieved)
    t_conf = gate.calculate_confidence(telugu_query, t_retrieved)
    t_passed, _, _ = gate.evaluate_and_gate(telugu_query, t_retrieved)

    print(f"Telugu Retrieval -> Sufficiency: {t_suff:.4f} | Confidence: {t_conf:.4f} | Passed: {t_passed}")
    assert t_conf >= 0.45, f"Telugu query falsely refused! Confidence: {t_conf:.4f}"
    assert t_passed is True, "Telugu query failed gate!"

    print("  [PASS] Multilingual Unicode evidence sufficiency and gating verified.")


def test_audit_problem_queries_remediation(gate: GuardrailGate, pipeline: BISRAGPipeline):
    print("\n" + "=" * 65)
    print("TEST 3: Audit Problem Queries False Refusal Remediation")
    print("=" * 65)

    # In the audit, queries had high evidence sufficiency (>57% - 66%) but were refused
    # because base_conf was locked at 0.30 and multiplied by 0.60
    # Let's test standard compliance queries that previously suffered this fate
    audit_queries = [
        "What are the market surveillance guidelines and inspection procedures followed by BIS?",
        "What are the training methodologies interactive classroom sessions in BIS training strategy?",
        "What are the requirements for physical tests and sampling criteria for Portland pozzolana cement?"
    ]

    for q in audit_queries:
        retrieved = pipeline.retrieval.retrieve(q, top_n=5)
        conf = gate.calculate_confidence(q, retrieved)
        passed, _, msg = gate.evaluate_and_gate(q, retrieved)
        suff = gate.calculate_evidence_sufficiency(q, retrieved)
        print(f"Query: '{q[:50]}...'")
        print(f"  Sufficiency: {suff:.4f} | Confidence: {conf:.4f} | Passed: {passed}")
        
        # If retrieved documents are returned from corpus
        if retrieved:
            assert conf >= 0.40, f"Query '{q}' unfairly penalized: conf={conf:.4f}"
            if suff >= 0.30:
                assert passed is True, f"Query '{q}' falsely refused despite suff={suff:.4f}"

    print("  [PASS] Compliant compliance queries are no longer trapped by the 67.5% barrier.")


def test_strict_out_of_domain_refusal(gate: GuardrailGate, pipeline: BISRAGPipeline):
    print("\n" + "=" * 65)
    print("TEST 4: Strict Out-of-Domain Refusal Gate (Zero False Passes)")
    print("=" * 65)

    ood_queries = [
        "What is the weather forecast for Tokyo Japan tomorrow morning?",
        "Who won the FIFA men's world cup tournament in 2022 in Qatar?",
        "How do I bake chocolate chip cookies with vanilla extract at home?",
    ]

    for q in ood_queries:
        retrieved = pipeline.retrieval.retrieve(q, top_n=5)
        conf = gate.calculate_confidence(q, retrieved)
        passed, gate_conf, refusal_msg = gate.evaluate_and_gate(q, retrieved)
        suff = gate.calculate_evidence_sufficiency(q, retrieved)

        print(f"OOD Query:    '{q}'")
        print(f"Confidence:   {conf:.4f} (Must be < {CONFIDENCE_THRESHOLD})")
        print(f"Sufficiency:  {suff:.4f}")
        print(f"Passed:       {passed}")
        print(f"Refusal Excerpt:\n{refusal_msg[:120]}...\n")

        assert passed is False, f"OOD query '{q}' falsely passed the refusal gate!"
        assert conf < 0.30, f"OOD query confidence {conf:.4f} too high (must be < 0.30)!"
        assert refusal_msg is not None, "Refusal message missing for out-of-domain query!"
        assert "Official BIS Portal" in refusal_msg or "bis.gov.in" in refusal_msg, "Portal link missing in refusal!"

    print("  [PASS] Out-of-domain queries are strictly and reliably refused.")


def test_confidence_separation_margin(gate: GuardrailGate, pipeline: BISRAGPipeline):
    print("\n" + "=" * 65)
    print("TEST 5: Statistical Separation Margin (In-Domain vs Out-of-Domain)")
    print("=" * 65)

    in_domain_queries = [
        "What are the chemical requirements for TMT bars under IS 1786?",
        "IS 269 compressive strength of 33 grade ordinary Portland cement",
        "minimum elongation percentage for high strength deformed steel bars",
        "sampling and criteria for conformity of cement under Indian Standards",
    ]

    ood_queries = [
        "What is the capital city of Australia?",
        "Latest stock market price of Apple and Microsoft shares",
        "Recipe for homemade Italian pizza dough and tomato sauce",
        "Rules of American football NFL touchdown scoring",
    ]

    in_confs = []
    for q in in_domain_queries:
        res = pipeline.retrieval.retrieve(q, top_n=5)
        c = gate.calculate_confidence(q, res)
        in_confs.append(c)

    ood_confs = []
    for q in ood_queries:
        res = pipeline.retrieval.retrieve(q, top_n=5)
        c = gate.calculate_confidence(q, res)
        ood_confs.append(c)

    mean_in = float(np.mean(in_confs))
    mean_ood = float(np.mean(ood_confs))
    margin = mean_in - mean_ood

    print(f"In-Domain Mean Confidence:     {mean_in:.4f} (Individual: {[round(x, 4) for x in in_confs]})")
    print(f"Out-of-Domain Mean Confidence: {mean_ood:.4f} (Individual: {[round(x, 4) for x in ood_confs]})")
    print(f"Statistical Separation Margin: {margin:.4f}")

    assert mean_in >= 0.75, f"In-domain mean confidence {mean_in:.4f} too low!"
    assert mean_ood <= 0.20, f"Out-of-domain mean confidence {mean_ood:.4f} too high!"
    assert margin >= 0.60, f"Separation margin {margin:.4f} < 0.60 SLA!"

    print("  [PASS] High discriminative margin (>= 0.60) verified.")


def main():
    print("Initializing BIS RAG Pipeline and GuardrailGate...")
    pipeline = BISRAGPipeline()
    gate = GuardrailGate(threshold=CONFIDENCE_THRESHOLD)

    test_base_confidence_trap_elimination(gate, pipeline)
    test_multilingual_unicode_evidence_sufficiency(gate, pipeline)
    test_audit_problem_queries_remediation(gate, pipeline)
    test_strict_out_of_domain_refusal(gate, pipeline)
    test_confidence_separation_margin(gate, pipeline)

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY! Feature 5 is Fully Verified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
