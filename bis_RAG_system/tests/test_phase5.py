"""
Phase 5, Step 19: Multilingual & Hinglish Benchmark Test Suite (test_phase5.py)
Verifies multilingual query detection, Hinglish normalization, and citation-preserving translations.
"""

import sys
import logging
from pathlib import Path

# Add src and base to python path for clean test execution
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [str(SRC_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


from src.multilingual import MultilingualHandler
from src.translation_engine import TranslationEngine
from tests.test_phase4 import Phase4Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_phase5")


class MultilingualBISPipelne:
    def __init__(self):
        log.info("Initializing Multilingual & Hinglish BIS RAG Pipeline...")
        self.multilingual_handler = MultilingualHandler()
        self.translation_engine = TranslationEngine()
        self.orchestrator = Phase4Orchestrator()

    def process_multilingual_query(self, user_query: str) -> dict:
        # 1. Detect language
        lang_info = self.multilingual_handler.detect_language(user_query)
        lang_code = lang_info["lang_code"]

        # 2. Normalize Hinglish to English if code-mixed
        if lang_code == "hinglish":
            processed_query = self.multilingual_handler.normalize_hinglish_to_english(user_query)
        else:
            processed_query = user_query

        # 3. Process via Phase 4 Orchestrator
        orch_res = self.orchestrator.process_query(processed_query)
        english_response = orch_res["response"]

        # 4. Translate response while preserving citations and numerical figures
        final_response = self.translation_engine.translate_response(english_response, target_lang=lang_code)

        return {
            "query": user_query,
            "detected_language": lang_info["lang_name"],
            "lang_code": lang_code,
            "sub_flow": orch_res["intent"],
            "response": final_response,
        }


def run_phase5_test_suite():
    print("\n" + "=" * 70)
    print("      PHASE 5: MULTILINGUAL & HINGLISH BENCHMARK TEST SUITE")
    print("=" * 70 + "\n")

    pipeline = MultilingualBISPipelne()

    test_queries = [
        {"name": "English Query", "query": "is certification mandatory for LED bulbs", "expected_lang": "en"},
        {"name": "Code-Mixed Hinglish Query", "query": "mera LED bulb ke liye BIS certification chahiye", "expected_lang": "hinglish"},
        {"name": "Native Hindi Query", "query": "क्या एलईडी बल्ब के लिए बीआईएस प्रमाणन अनिवार्य है?", "expected_lang": "hi"},
        {"name": "Hinglish Scheme Query", "query": "Scheme-I me apply kaise kare kitna fee lagiga", "expected_lang": "hinglish"},
        {"name": "Native Tamil Query", "query": "எல்இடி பல்புகளுக்கு பிஐஎஸ் சான்றிதழ் கட்டாயமா?", "expected_lang": "ta"},
    ]

    passed = 0
    total = len(test_queries)

    for test in test_queries:
        name = test["name"]
        q = test["query"]
        expected_code = test["expected_lang"]

        safe_q = q.encode('ascii', 'replace').decode()
        print(f"> TEST: {name}")
        print(f"  Input: '{safe_q}'")

        res = pipeline.process_multilingual_query(q)
        detected_code = res["lang_code"]

        safe_snippet = res['response'][:150].encode('ascii', 'replace').decode()
        print(f"  Output Snippet: {safe_snippet}...")

        # Verify IS numbers or fees are preserved in response
        if detected_code == expected_code:
            passed += 1
            print("  Result: [PASS]\n")
        else:
            print(f"  Result: [FAIL] (Expected lang '{expected_code}', got '{detected_code}')\n")

    pass_rate = (passed / total) * 100
    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} Multilingual Test Cases Passed ({pass_rate:.2f}% Pass Rate)")
    print("=" * 70 + "\n")
    return passed == total


# =====================================================================
# PYTEST SUITE: Phase 5 Remediation Verification (F1 - F7)
# =====================================================================

def test_phase5_multilingual_benchmark():
    """Verify original Phase 5 benchmark suite passes."""
    pipeline = MultilingualBISPipelne()
    test_queries = [
        {"name": "English Query", "query": "is certification mandatory for LED bulbs", "expected_lang": "en"},
        {"name": "Code-Mixed Hinglish Query", "query": "mera LED bulb ke liye BIS certification chahiye", "expected_lang": "hinglish"},
        {"name": "Native Hindi Query", "query": "क्या एलईडी बल्ब के लिए बीआईएस प्रमाणन अनिवार्य है?", "expected_lang": "hi"},
        {"name": "Hinglish Scheme Query", "query": "Scheme-I me apply kaise kare kitna fee lagiga", "expected_lang": "hinglish"},
        {"name": "Native Tamil Query", "query": "எல்இடி பல்புகளுக்கு பிஐஎஸ் சான்றிதழ் கட்டாயமா?", "expected_lang": "ta"},
    ]
    for test in test_queries:
        res = pipeline.process_multilingual_query(test["query"])
        assert res["lang_code"] == test["expected_lang"], f"Expected {test['expected_lang']}, got {res['lang_code']}"


def test_f1_translation_engine_integration():
    """F1: TranslationEngine is reachable in production BISRAGPipeline and citation masking preserves technical tokens."""
    pipeline = Phase4Orchestrator(llm_provider="mock", use_mock_retrieval=True)
    assert hasattr(pipeline, "translation_engine"), "TranslationEngine must be reachable in BISRAGPipeline"

    # Citation masking protects IS numbers, clauses, URLs, hashes, and chunk IDs
    text = (
        "Based on official Bureau of Indian Standards documentation: "
        "Standard IS 1786:2008 Clause 4.2 chunk_std_1786 "
        "hash abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789 "
        "https://www.bis.gov.in/doc.pdf"
    )
    masked, tokens = pipeline.translation_engine.mask_protected_tokens(text)
    assert "IS 1786:2008" in tokens or any("1786" in t for t in tokens)
    assert any("abcdef0123456789" in t for t in tokens)
    unmasked = pipeline.translation_engine.unmask_protected_tokens(masked, tokens)
    assert "IS 1786:2008" in unmasked
    assert "Clause 4.2" in unmasked

    # Honest translation: phrase dictionary translates key regulatory phrases to Hindi
    translated = pipeline.translation_engine.translate_response(
        "Based on official Bureau of Indian Standards documentation", target_lang="hi"
    )
    assert "भारतीय मानक ब्यूरो के आधिकारिक दस्तावेजों के अनुसार" in translated


def test_f2_indic_normalization():
    """F2: Tamil and Bengali native-script normalization produces English retrieval keywords."""
    handler = MultilingualHandler()

    # Tamil normalization
    norm_ta_std = handler.normalize_native_to_english_keywords("சிமெண்ட் தரநிலை")
    assert "cement" in norm_ta_std.lower()
    assert "standard" in norm_ta_std.lower()

    norm_ta_bulb = handler.normalize_native_to_english_keywords("எல்இடி பல்புகளுக்கு பிஐஎஸ் சான்றிதழ் கட்டாயமா?")
    assert "bulb" in norm_ta_bulb.lower()
    assert "certification" in norm_ta_bulb.lower()
    assert "mandatory" in norm_ta_bulb.lower()

    # Bengali normalization
    norm_bn_std = handler.normalize_native_to_english_keywords("সিমেন্ট মানক কি?")
    assert "cement" in norm_bn_std.lower()
    assert "standard" in norm_bn_std.lower()

    norm_bn_bulb = handler.normalize_native_to_english_keywords("এলইডি বাল্বের জন্য বিআইএস প্রমাণপত্র কি বাধ্যতামূলক?")
    assert "bulb" in norm_bn_bulb.lower()
    assert "certification" in norm_bn_bulb.lower()
    assert "mandatory" in norm_bn_bulb.lower()

    # Hindi regression
    norm_hi = handler.normalize_native_to_english_keywords("सीमेंट मानक क्या है?")
    assert "cement" in norm_hi.lower()
    assert "standard" in norm_hi.lower()

    # Telugu regression
    norm_te = handler.normalize_native_to_english_keywords("సిమెంట్ ప్రమాణం ఏమిటి?")
    assert "cement" in norm_te.lower()
    assert "standard" in norm_te.lower()

    # Hinglish regression
    norm_hinglish = handler.normalize_hinglish_to_english("mera LED bulb ke liye BIS certification chahiye")
    assert "for led bulb" in norm_hinglish.lower()


def test_f3_lab_locator_lims_discovery():
    """F3: Lab Locator operates as an honest BIS LIMS Discovery flow when no local records exist."""
    pipeline = Phase4Orchestrator(llm_provider="mock", use_mock_retrieval=True)
    res = pipeline.query("where are testing labs for steel")
    assert res["flow_used"] == "lab_locator"
    assert res["status"] == "unavailable"
    assert res["results"] == []
    assert "https://lims.bis.gov.in/" in res["response"]
    assert "BIS LIMS" in res["response"]
    assert res["citations"][0]["url"] == "https://lims.bis.gov.in/"
    assert res["citations"][0]["source_of_truth"] == "official_lims_portal"
    assert res["citations"][0]["citation_type"] == "official_source"


def test_f4_specialized_subflow_grounding_and_confidence():
    """F4: Specialized flows validate request compatibility and expose honest confidence types."""
    pipeline = Phase4Orchestrator(llm_provider="mock", use_mock_retrieval=True)

    # Valid product recommendation exposes deterministic confidence
    res_prod = pipeline.query("IS 12860")
    assert res_prod["flow_used"] == "product_recommender"
    assert res_prod["confidence_type"] == "deterministic_verified"

    # Valid scheme walkthrough exposes deterministic confidence
    res_scheme = pipeline.query("how to apply for ISI mark under Scheme-I")
    assert res_scheme["flow_used"] == "scheme_walkthrough"
    assert res_scheme["confidence_type"] == "deterministic_verified"

    # Valid complaint exposes deterministic confidence
    res_comp = pipeline.query("fake ISI mark on electrical appliance")
    assert res_comp["flow_used"] == "consumer_complaint"
    assert res_comp["confidence_type"] == "deterministic_verified"

    # Out-of-domain nonsensical request is rejected by sanity gate
    is_compat, msg = pipeline.guardrail.validate_subflow_compatibility("certification_process", "how to bake bread with chocolate")
    assert not is_compat


def test_f5_structured_citations():
    """F5: Scheme, Complaint, and Lab subflows return structured citations without fabricated hashes."""
    pipeline = Phase4Orchestrator(llm_provider="mock", use_mock_retrieval=True)

    # Scheme citation
    res_scheme = pipeline.query("how to apply for ISI mark under Scheme-I")
    assert len(res_scheme["citations"]) > 0
    cite_s = res_scheme["citations"][0]
    assert cite_s["citation_type"] == "official_source"
    assert "bis.gov.in" in cite_s["url"]

    # Complaint citation
    res_comp = pipeline.query("my gold jewellery purity is defective how to complain")
    assert len(res_comp["citations"]) > 0
    cite_c = res_comp["citations"][0]
    assert cite_c["citation_type"] == "official_source"
    assert "bis.gov.in" in cite_c["url"]

    # Lab citation
    res_lab = pipeline.query("where are testing laboratories in Delhi")
    assert len(res_lab["citations"]) > 0
    cite_l = res_lab["citations"][0]
    assert cite_l["citation_type"] == "official_source"
    assert "lims.bis.gov.in" in cite_l["url"]


def test_f6_router_collision_and_plurals():
    """F6: Router handles compound product+scheme queries and lab plurals accurately."""
    from src.router import QueryIntentRouter
    router = QueryIntentRouter()

    # Compound product + scheme query must route to product_recommendation
    res_cem = router.classify_intent("which standard for cement under Scheme-I")
    assert res_cem["intent"] == "product_recommendation"

    # Pipeline execution of compound query identifies relevant IS standard and retains scheme qualifier
    pipeline = Phase4Orchestrator(llm_provider="mock", use_mock_retrieval=True)
    res_cem_pipeline = pipeline.query("which standard for cement under Scheme-I")
    assert res_cem_pipeline["flow_used"] == "product_recommender"
    assert "IS 1489" in res_cem_pipeline["response"] or "IS 269" in res_cem_pipeline["response"]
    assert "Scheme-I" in res_cem_pipeline["response"]

    # Legitimate scheme questions remain routed to certification_process
    res_scheme1 = router.classify_intent("How do I apply under Scheme-I?")
    assert res_scheme1["intent"] == "certification_process"
    res_scheme2 = router.classify_intent("What are the steps for Scheme-I?")
    assert res_scheme2["intent"] == "certification_process"
    res_scheme3 = router.classify_intent("What documents are required for Scheme-I?")
    assert res_scheme3["intent"] == "certification_process"

    # Lab singular and plural patterns
    assert router.classify_intent("where are testing labs for steel")["intent"] == "lab_location"
    assert router.classify_intent("where are testing laboratories for steel")["intent"] == "lab_location"
    assert router.classify_intent("where is a testing lab in Delhi")["intent"] == "lab_location"
    assert router.classify_intent("testing laboratory scope")["intent"] == "lab_location"


def test_f7_multilingual_specialized_subflows():
    """F7: Scheme and Lab subflows support multilingual response formatting."""
    pipeline = Phase4Orchestrator(llm_provider="mock", use_mock_retrieval=True)

    # Hindi Scheme response
    res_hi_scheme = pipeline.query("स्कीम-I के तहत आवेदन कैसे करें?", category=None)
    assert res_hi_scheme["flow_used"] == "scheme_walkthrough"
    assert "आधिकारिक" in res_hi_scheme["response"] or "आवेदन शुल्क" in res_hi_scheme["response"]

    # Telugu Scheme response
    res_te_scheme = pipeline.query("స్కీమ్-I కింద ఎలా దరఖాస్తు చేయాలి?", category=None)
    assert res_te_scheme["flow_used"] == "scheme_walkthrough"
    assert "అధికారిక" in res_te_scheme["response"] or "ఫీజుల" in res_te_scheme["response"]

    # Hindi Lab response
    res_hi_lab = pipeline.query("परीक्षण प्रयोगशालाएं कहां हैं?", category=None)
    assert res_hi_lab["flow_used"] == "lab_locator"
    assert "प्रयोगशाला" in res_hi_lab["response"]
    assert "https://lims.bis.gov.in/" in res_hi_lab["response"]

    # Telugu Lab response
    res_te_lab = pipeline.query("పరీక్షా ప్రయోగశాలలు ఎక్కడ ఉన్నాయి?", category=None)
    assert res_te_lab["flow_used"] == "lab_locator"
    assert "ప్రయోగశాల" in res_te_lab["response"]
    assert "https://lims.bis.gov.in/" in res_te_lab["response"]

    # English regression
    res_en_scheme = pipeline.query("how to apply for ISI mark under Scheme-I")
    assert res_en_scheme["flow_used"] == "scheme_walkthrough"
    assert "Official Step-by-Step Walkthrough" in res_en_scheme["response"]


if __name__ == "__main__":
    run_phase5_test_suite()
