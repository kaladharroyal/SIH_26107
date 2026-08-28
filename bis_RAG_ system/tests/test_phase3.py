"""
Phase 3, Part 3.4: End-to-End RAG Pipeline Test Suite (test_phase3.py)
Verifies RAG pipeline responses across Product Certification, Hallmarking, Steel Standards,
and Out-of-Corpus Refusal queries.
"""

import logging
from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_phase3")

TEST_CASES = [
    {
        "name": "Product Certification Test",
        "query": "what is the process for product certification under scheme I",
        "expected_status": "success",
    },
    {
        "name": "Hallmarking Regulation Test",
        "query": "what standard applies to gold jewellery hallmarking regulations",
        "expected_status": "success",
    },
    {
        "name": "Steel Standard Test",
        "query": "IS 1786 steel reinforcement requirements",
        "expected_status": "success",
    },
    {
        "name": "Out-of-Corpus Refusal Test",
        "query": "what is the property tax rate in Tokyo Japan",
        "expected_status": "refused",
    },
]


def run_phase3_test_suite():
    print("\n" + "=" * 70)
    print("      PHASE 3: END-TO-END RAG PIPELINE & GROUNDING TEST SUITE")
    print("=" * 70 + "\n")

    pipeline = BISRAGPipeline(confidence_threshold=0.45)
    passed_tests = 0
    total_tests = len(TEST_CASES)

    for test in TEST_CASES:
        name = test["name"]
        q = test["query"]
        expected = test["expected_status"]

        print(f"▶ RUNNING TEST: {name}")
        print(f"  Query: '{q}'")

        res = pipeline.query(q)
        status = res["status"]
        conf = res["confidence_score"]

        print(f"  Status: {status.upper()} (Confidence: {conf:.4f})")
        print(f"  Citations Generated: {len(res.get('citations', []))}")
        print(f"  Response Snippet: {res['response'][:150]}...")

        if status == expected:
            passed_tests += 1
            print("  Result: ✅ PASSED\n")
        else:
            print(f"  Result: ❌ FAILED (Expected status '{expected}', got '{status}')\n")

    pass_rate = (passed_tests / total_tests) * 100
    print("=" * 70)
    print(f"SUMMARY: {passed_tests}/{total_tests} Tests Passed ({pass_rate:.2f}% Pass Rate)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_phase3_test_suite()
