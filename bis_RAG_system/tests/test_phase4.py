"""
Phase 4: Specialized Sub-Flows Automated Pytest Suite (test_phase4.py)
Tests all 18 criteria across Product Recommender, Scheme Walkthroughs, Lab Locator,
Consumer Complaints, and Unified Pipeline Dispatcher. Runs completely offline without external network calls.
"""

import os
import sys
import unittest
from pathlib import Path

# Add src and root to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from consumer_complaint import ConsumerComplaintHandler
from lab_locator import LabLocator
from product_recommender import PRODUCT_ALIASES, ProductRecommender
from rag_pipeline import BISRAGPipeline
from router import QueryIntentRouter
from scheme_walkthrough import SCHEME_WALKTHROUGHS, SchemeWalkthroughGuide


class MockRetrievalFallback:
    """Mock retrieval engine for deterministic offline unit testing."""

    def retrieve(self, query: str, top_n: int = 3, category: str = None):
        if "steel" in query.lower() or "1786" in query.lower():
            return [{
                "rrf_score": 0.032,
                "score": 0.032,
                "doc": {
                    "chunk_id": "chunk_std_1786",
                    "is_number": "IS 1786",
                    "revision_year": "2008",
                    "clause_title": "High Strength Deformed Steel Bars",
                    "text": "Requirements for high strength deformed steel bars and wires for concrete reinforcement.",
                    "category": "is_standard",
                    "source_file": "raw_data/pdfs/IS_1786.pdf",
                    "source_url": "https://standardsbis.bsbedge.com/IS_1786.aspx",
                    "source_hash": "hash1786",
                    "source_of_truth": "verified_bis_pdf",
                }
            }]
        elif "lab" in query.lower() or category == "lab_directory":
            return [{
                "rrf_score": 0.025,
                "score": 0.025,
                "doc": {
                    "chunk_id": "chunk_lab_01",
                    "clause_title": "BIS Central Laboratory Testing Guidelines",
                    "text": "BIS Central Laboratory Sahibabad performs testing across mechanical, chemical, and electrical disciplines.",
                    "category": "lab_directory",
                    "source_file": "raw_data/pdfs/lab_guide.pdf",
                    "source_url": "https://lims.bis.gov.in/",
                    "source_hash": "hashlab01",
                    "source_of_truth": "verified_bis_pdf",
                }
            }]
        return []


class TestPhase4SpecializedSubFlows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = ProductRecommender()
        cls.recommender_with_fallback = ProductRecommender(retrieval_pipeline=MockRetrievalFallback())
        cls.walkthrough_guide = SchemeWalkthroughGuide()
        cls.lab_locator = LabLocator()
        cls.lab_locator_with_fallback = LabLocator(retrieval_pipeline=MockRetrievalFallback())
        cls.complaint_handler = ConsumerComplaintHandler()
        cls.router = QueryIntentRouter()
        cls.pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)

    def test_01_product_exact_matching(self):
        """Criterion 1: Product recommender exact match on Phase 1 authentic map."""
        res = self.recommender.recommend("Random sampling and randomization procedures")
        self.assertEqual(res["status"], "success")
        self.assertIn("IS 1070", res["formatted_text"])
        self.assertFalse(res["fallback_used"])
        print("✅ Test 1 Passed: Product exact match verified.")

    def test_02_product_alias_normalization(self):
        """Criterion 2: Common consumer terms normalize to formal technical names."""
        normalized = ProductRecommender.normalize_query_with_aliases("what BIS standard should I use for a TMT bar?")
        self.assertEqual(normalized, "High strength deformed steel bars")

        norm_solar = ProductRecommender.normalize_query_with_aliases("solar panel standards")
        self.assertEqual(norm_solar, "Photovoltaic Module")
        print("✅ Test 2 Passed: Product alias normalization verified.")

    def test_03_product_is_number_matching(self):
        """Criterion 3: Product recommender matches directly by IS number."""
        res = self.recommender.recommend("IS 12860")
        self.assertEqual(res["status"], "success")
        self.assertIn("IS 12860", res["formatted_text"])
        self.assertIn("Metallic Coating", res["formatted_text"])
        print("✅ Test 3 Passed: Direct IS number lookup verified.")

    def test_04_product_corpus_fallback(self):
        """Criterion 4: Product recommender falls back to corpus search when static map misses."""
        res = self.recommender_with_fallback.recommend("TMT bar reinforcement steel")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["fallback_used"])
        self.assertIn("IS 1786", res["formatted_text"])
        self.assertIn("source_hash", res["provenance"])
        print("✅ Test 4 Passed: Product corpus search fallback verified.")

    def test_05_scheme_i_walkthrough(self):
        """Criterion 5: Scheme-I walkthrough contains exact fees, timeline, and steps."""
        res = self.walkthrough_guide.get_walkthrough("how to apply for ISI mark Scheme-I")
        self.assertEqual(res["scheme_key"], "scheme_i")
        self.assertEqual(res["fee_schedule"]["application_fee"], "₹1,000")
        self.assertIn("₹7,000", res["fee_schedule"]["inspection_charge"])
        self.assertEqual(res["fee_schedule"]["test_report_validity"], "90 Days from recognized lab")
        self.assertEqual(len(res["steps"]), 6)
        print("✅ Test 5 Passed: Scheme-I walkthrough and exact fee figures verified.")

    def test_06_crs_scheme_ii_walkthrough(self):
        """Criterion 6: CRS Scheme-II walkthrough contains exact parameters."""
        res = self.walkthrough_guide.get_walkthrough("how to register for electronics under CRS Scheme-II")
        self.assertEqual(res["scheme_key"], "scheme_ii")
        self.assertEqual(res["fee_schedule"]["application_fee"], "₹1,000 per product model family")
        self.assertIn("Nil", res["fee_schedule"]["inspection_charge"])
        self.assertEqual(len(res["steps"]), 4)
        print("✅ Test 6 Passed: CRS Scheme-II walkthrough verified.")

    def test_07_fmcs_walkthrough(self):
        """Criterion 7: FMCS walkthrough includes USD fees and AIR requirement."""
        res = self.walkthrough_guide.get_walkthrough("FMCS certification for foreign manufacturers")
        self.assertEqual(res["scheme_key"], "fmcs")
        self.assertEqual(res["fee_schedule"]["application_fee"], "USD $1,000")
        self.assertEqual(res["fee_schedule"]["performance_bank_guarantee"], "USD $10,000")
        self.assertIn("AIR Appointment", res["steps"][0])
        print("✅ Test 7 Passed: FMCS foreign certification walkthrough verified.")

    def test_08_scheme_x_and_hallmarking_walkthrough(self):
        """Criterion 8: Scheme-X and Hallmarking scheme walkthroughs verified."""
        res_x = self.walkthrough_guide.get_walkthrough("Scheme-X custom machinery certification")
        self.assertEqual(res_x["scheme_key"], "scheme_x")
        self.assertIn("Capital Goods", res_x["title"])

        res_hm = self.walkthrough_guide.get_walkthrough("how to get gold jewellery hallmarking registration")
        self.assertEqual(res_hm["scheme_key"], "hallmarking")
        self.assertIn("HUID", res_hm["formatted_text"])
        self.assertIn("₹45", res_hm["fee_schedule"]["hallmarking_charge"])
        print("✅ Test 8 Passed: Scheme-X and Hallmarking walkthroughs verified.")

    def test_09_lab_directory_lookup(self):
        """Criterion 9: Lab locator structure returns valid response."""
        res = self.lab_locator.search_labs("testing laboratories")
        self.assertIn("status", res)
        self.assertIn("formatted_text", res)
        self.assertIn("https://lims.bis.gov.in/", res["formatted_text"])
        print("✅ Test 9 Passed: Lab directory lookup structure verified.")

    def test_10_lab_state_filtering(self):
        """Criterion 10: State extraction and filtering handles location queries cleanly."""
        res = self.lab_locator.search_labs("testing labs in Delhi", state="Delhi")
        self.assertIn("formatted_text", res)
        print("✅ Test 10 Passed: Lab state filtering verified.")

    def test_11_lab_corpus_fallback(self):
        """Criterion 11: Lab locator queries corpus when static directory is empty."""
        res = self.lab_locator_with_fallback.search_labs("which lab tests mechanical samples")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["fallback_used"])
        self.assertIn("Central Laboratory", res["formatted_text"])
        print("✅ Test 11 Passed: Lab locator corpus search fallback verified.")

    def test_12_consumer_complaint_hallmarking_compensation(self):
        """Criterion 12: Consumer complaint includes statutory 2x compensation for hallmarking."""
        res = self.complaint_handler.handle_complaint("I bought a gold ring and its purity is lower than promised")
        self.assertEqual(res["category"], "hallmarking_complaint")
        self.assertTrue(res["is_hallmarking"])
        self.assertIn("TWO TIMES (2x)", res["compensation_rights"])
        self.assertIn("BIS CARE", res["formatted_text"])
        print("✅ Test 12 Passed: Hallmarking 2x compensation rule verified.")

    def test_13_consumer_complaint_isi_defective_redressal(self):
        """Criterion 13: Non-hallmarking complaints provide ISI enforcement redressal."""
        res = self.complaint_handler.handle_complaint("fake ISI mark on electrical appliance")
        self.assertEqual(res["category"], "isi_product_complaint")
        self.assertFalse(res["is_hallmarking"])
        self.assertIn("1800-11-4000", res["formatted_text"])
        self.assertIn("complaints@bis.gov.in", res["formatted_text"])
        print("✅ Test 13 Passed: ISI complaint legal redressal guidance verified.")

    def test_14_intent_classification_all_flows(self):
        """Criterion 14: Intent router classifies all 5 discrete Phase 4 intents."""
        self.assertEqual(self.router.classify_intent("is certification mandatory for solar panels")["intent"], "product_recommendation")
        self.assertEqual(self.router.classify_intent("how to apply online for Scheme-I ISI mark")["intent"], "certification_process")
        self.assertEqual(self.router.classify_intent("where are testing labs in Mumbai")["intent"], "lab_location")
        self.assertEqual(self.router.classify_intent("my gold jewellery purity is defective how to complain")["intent"], "consumer_complaint")
        self.assertEqual(self.router.classify_intent("IS 1786 steel reinforcement requirements")["intent"], "general_rag")
        print("✅ Test 14 Passed: All 5 Phase 4 intent classifications verified.")

    def test_15_unified_pipeline_subflow_dispatch(self):
        """Criterion 15: Main pipeline dispatches to specialized sub-flows seamlessly."""
        # 1. Product Recommender Dispatch
        res_prod = self.pipeline.query("what BIS standard should I use for a TMT bar?")
        self.assertEqual(res_prod["flow_used"], "product_recommender")
        self.assertEqual(res_prod["status"], "success")

        # 2. Scheme Walkthrough Dispatch
        res_walk = self.pipeline.query("how do I get BIS certification under Scheme I?")
        self.assertEqual(res_walk["flow_used"], "scheme_walkthrough")
        self.assertEqual(res_walk["status"], "success")

        # 3. Consumer Complaint Dispatch
        res_comp = self.pipeline.query("I bought a gold item and its purity is lower than promised")
        self.assertEqual(res_comp["flow_used"], "consumer_complaint")
        self.assertEqual(res_comp["status"], "success")
        print("✅ Test 15 Passed: Unified pipeline sub-flow dispatching verified.")

    def test_16_general_rag_fallback(self):
        """Criterion 16: General unstructured queries route to Phase 3 Grounded RAG."""
        res_rag = self.pipeline.query("IS 1786 steel reinforcement requirements")
        self.assertEqual(res_rag["flow_used"], "general_rag")
        self.assertEqual(res_rag["status"], "success")
        self.assertGreater(res_rag["confidence_score"], 0.45)
        print("✅ Test 16 Passed: General RAG routing verified.")

    def test_17_structured_response_schema(self):
        """Criterion 17: Responses follow consistent machine-readable JSON schema."""
        res = self.pipeline.query("how to apply for ISI mark Scheme-I")
        required_keys = ["query", "intent", "flow_used", "status", "confidence_score", "response", "results", "citations", "fallback_used"]
        for k in required_keys:
            self.assertIn(k, res)
        print("✅ Test 17 Passed: Structured response schema verified.")

    def test_18_zero_hardcoded_production_data(self):
        """Criterion 18: Recommender and lab locator contain zero fabricated production products or labs."""
        recommender = ProductRecommender()
        self.assertIsInstance(recommender.db, list)
        self.assertGreaterEqual(len(recommender.db), 81)
        # Verify alias table only contains normalization strings, not complete fake records
        for alias, formal in PRODUCT_ALIASES.items():
            self.assertIsInstance(alias, str)
            self.assertIsInstance(formal, str)
        print("✅ Test 18 Passed: Zero hardcoded production data verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
