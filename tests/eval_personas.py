"""
Phase 6: Persona Scenario Evaluation Suite (eval_personas.py)
Automated pytest-compatible evaluation of real-world BIS Compliance Personas:
1. MSME Manufacturer (Procedural, fee, and conformity queries)
2. Engineering Student (Technical standards and specification queries)
3. Rural Consumer (Complaint redressal, hallmarking verification, rights)
4. Compliance Officer (International & foreign manufacturer schemes)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import pytest

# Add src and root to python path for clean test execution
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"
for path in [SRC_DIR, TESTS_DIR, BASE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_personas")

PERSONA_SCENARIOS = [
    {
        "persona": "MSME Manufacturer",
        "query": "what is the application fee for ISI mark product certification under Scheme-I",
        "expected_flow": "scheme_walkthrough",
        "expected_standard_or_entity": "1,000",
        "requires_citations": True,
    },
    {
        "persona": "MSME Manufacturer",
        "query": "is certification mandatory for LED bulbs under CRS",
        "expected_flow": "product_recommender",
        "expected_standard_or_entity": "led",
        "requires_citations": True,
    },
    {
        "persona": "Engineering Student",
        "query": "what standard applies to gold jewellery hallmarking regulations",
        "expected_flow": "product_recommender",
        "expected_standard_or_entity": "gold",
        "requires_citations": True,
    },
    {
        "persona": "Engineering Student",
        "query": "IS 1786 steel reinforcement requirements",
        "expected_flow": "general_rag",
        "expected_standard_or_entity": "1786",
        "requires_citations": True,
    },
    {
        "persona": "Rural Consumer",
        "query": "my gold hallmark jewellery is fake how to complain",
        "expected_flow": "consumer_complaint",
        "expected_standard_or_entity": "bis care",
        "requires_citations": True,
    },
    {
        "persona": "Compliance Officer",
        "query": "what is Foreign Manufacturers Certification Scheme FMCS under Scheme-IV",
        "expected_flow": "scheme_walkthrough",
        "expected_standard_or_entity": "foreign",
        "requires_citations": True,
    },
]


@pytest.fixture(scope="module")
def pipeline():
    return BISRAGPipeline(use_fast_retrieval=True)


def _assert_persona_response_contract(res: Dict[str, Any], scenario: Dict[str, Any]):
    """Strict contract validation for persona query responses."""
    assert isinstance(res, dict), "Response must be a structured dictionary"
    assert "status" in res, "Response must declare status"
    assert res["status"] in ["success", "refused"], f"Unexpected status: {res.get('status')}"
    assert "confidence_score" in res, "Confidence score must be exposed"
    assert 0.0 <= res["confidence_score"] <= 1.0, f"Confidence score out of range: {res['confidence_score']}"
    assert "response" in res and len(res["response"]) > 0, "Response text must not be empty"

    # Citations contract
    assert "citations" in res, "Citations field must be present"
    assert isinstance(res["citations"], list), "Citations must be a list"
    if scenario.get("requires_citations") and res["status"] == "success":
        assert len(res["citations"]) > 0, f"Grounded persona query must provide citations: {scenario['query']}"
        for c in res["citations"]:
            assert isinstance(c, dict), "Citation must be a dictionary"
            assert "label" in c or "url" in c or "source_hash" in c, "Citation must contain provenance"

    # Expected standard or domain entity check
    expected = scenario.get("expected_standard_or_entity", "").lower()
    resp_lower = res["response"].lower()
    results_str = str(res.get("results", "")).lower()
    assert (expected in resp_lower or expected in results_str), (
        f"Persona response missing expected standard/entity '{expected}' for query: {scenario['query']}"
    )


@pytest.mark.parametrize("scenario", PERSONA_SCENARIOS)
def test_persona_scenario(pipeline, scenario):
    """Parameterized test verifying every persona scenario against strict contract."""
    res = pipeline.query(scenario["query"])
    _assert_persona_response_contract(res, scenario)


def run_persona_evaluation():
    """CLI runner preserving backwards compatibility."""
    print("\n" + "=" * 70)
    print("      PHASE 6: PERSONA ACCURACY & CITATION VERIFICATION SUITE")
    print("=" * 70 + "\n")

    p = BISRAGPipeline(use_fast_retrieval=True)
    passed = 0
    total = len(PERSONA_SCENARIOS)

    for idx, sc in enumerate(PERSONA_SCENARIOS, 1):
        print(f"▶ TEST #{idx} [{sc['persona']}]")
        print(f"  Query: '{sc['query']}'")
        res = p.query(sc["query"])
        try:
            _assert_persona_response_contract(res, sc)
            passed += 1
            print(f"  Flow: {res.get('flow_used')} | Confidence: {res.get('confidence_score')}")
            print(f"  Result: ✅ PASS\n")
        except AssertionError as e:
            print(f"  Result: ❌ FAIL ({e})\n")

    acc = (passed / total) * 100
    print("=" * 70)
    print(f"PERSONA EVALUATION SUMMARY: {passed}/{total} Passed ({acc:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_persona_evaluation()
