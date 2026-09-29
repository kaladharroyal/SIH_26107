"""
Phase 3: Grounded Generation & Guardrails Automated Pytest Suite (test_phase3.py)
Tests all 17 Phase 3 criteria completely offline without external network dependencies.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from citation_engine import CitationEngine
from generator import (
    GeminiProvider,
    GroundedGenerator,
    MockOfflineProvider,
    OpenAIProvider,
    SYSTEM_GROUNDING_PROMPT,
)
from guardrails import CONFIDENCE_THRESHOLD, GuardrailGate
from rag_pipeline import BISRAGPipeline
from router import INTENT_TO_CORPUS_CATEGORY, QueryIntentRouter


class TestPhase3GenerationAndGuardrails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_chunks = [
            {
                "doc": {
                    "chunk_id": "IS1786_2008_C4.2",
                    "is_number": "IS 1786",
                    "revision_year": "2008",
                    "clause_number": "4.2",
                    "clause_title": "Chemical Composition of High Strength Deformed Steel Bars",
                    "text": "The maximum permissible sulfur and phosphorus content in high strength deformed steel bars shall not exceed 0.055 percent.",
                    "category": "is_standard",
                    "source_file": "raw_data/pdfs/IS_1786.pdf",
                    "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=1786_2008",
                    "source_hash": "a1b2c3d4e5f6001",
                    "source_of_truth": "verified_bis_pdf",
                    "page_start": 4,
                }
            },
            {
                "doc": {
                    "chunk_id": "CERT_FAQ_Q14",
                    "is_number": "BIS-SCHEME-I",
                    "revision_year": "2026",
                    "clause_number": "14",
                    "clause_title": "Product Certification Application Fees",
                    "text": "The application fee for grant of licence under Scheme-I is Rs 1000 along with inspection fee of Rs 7000 per man-day.",
                    "category": "general_policy",
                    "source_file": "raw_data/json/faq_cert.json",
                    "source_url": "https://www.bis.gov.in/product-certification-faq/",
                    "source_hash": "a1b2c3d4e5f6002",
                    "source_of_truth": "verified_bis_api",
                    "page_start": 1,
                }
            },
            {
                "doc": {
                    "chunk_id": "LAB_DIR_001",
                    "is_number": "IS 1070",
                    "revision_year": "2023",
                    "clause_number": "2.1",
                    "clause_title": "Water Testing Laboratory Protocol",
                    "text": "BIS recognized testing laboratories must maintain calibrated conductivity meters with accuracy of 0.1 uS/cm for testing laboratory grade water.",
                    "category": "lab_directory",
                    "source_file": "raw_data/pdfs/IS_1070.pdf",
                    "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=1070_2023",
                    "source_hash": "a1b2c3d4e5f6003",
                    "source_of_truth": "verified_bis_pdf",
                    "page_start": 2,
                }
            },
        ]

    def test_01_grounding_prompt_construction(self):
        """Criterion 1: System prompt enforces 100% context-only grounding and numerical fidelity."""
        self.assertIn("STRICT GROUNDING RULES", SYSTEM_GROUNDING_PROMPT)
        self.assertIn("ONLY using the provided retrieved context chunks", SYSTEM_GROUNDING_PROMPT)
        self.assertIn("Exact numerical figures", SYSTEM_GROUNDING_PROMPT)
        print("✅ Test 1 Passed: Grounding prompt rules verified.")

    def test_02_context_formatting_all_chunk_types(self):
        """Criterion 2: Context block formats standards, general policies, and lab chunks cleanly."""
        formatted = GroundedGenerator.format_context_block(self.sample_chunks)
        self.assertIn("IS Number: IS 1786:2008", formatted)
        self.assertIn("Clause: Clause 4.2", formatted)
        self.assertIn("Chunk ID: CERT_FAQ_Q14", formatted)
        self.assertIn("Source: https://www.bis.gov.in/product-certification-faq/", formatted)
        print("✅ Test 2 Passed: Context block formatting verified across diverse chunk types.")

    def test_03_multi_chunk_offline_synthesis(self):
        """Criterion 3: Offline synthesizer combines multiple retrieved chunks coherently."""
        gen = GroundedGenerator(provider_name="mock")
        res = gen.generate_response("what are the steel limits and certification fees", self.sample_chunks)
        answer = res["response"]
        self.assertIn("0.055", answer)
        self.assertIn("1000", answer)
        self.assertIn("IS 1786", answer)
        print("✅ Test 3 Passed: Multi-chunk offline synthesis combines multiple facts coherently.")

    def test_04_evidence_sufficiency_calculation(self):
        """Criterion 4: Evidence sufficiency measures keyword overlap between query and context."""
        gate = GuardrailGate(threshold=0.45)
        # Relevant query
        score_rel = gate.calculate_evidence_sufficiency("high strength deformed steel bars", self.sample_chunks)
        self.assertGreater(score_rel, 0.50)

        # Irrelevant out-of-corpus query
        score_irr = gate.calculate_evidence_sufficiency("property tax rate in Tokyo Japan", self.sample_chunks)
        self.assertEqual(score_irr, 0.0)
        print(f"✅ Test 4 Passed: Evidence sufficiency: Relevant={score_rel:.2f}, Irrelevant={score_irr:.2f}")

    def test_05_confidence_calibration(self):
        """Criterion 5: Confidence calibration properly scales Phase 2 RRF scores."""
        gate = GuardrailGate(threshold=0.45)
        # High RRF score with keyword match
        conf_high = gate.calculate_confidence(
            "steel reinforcement bars IS 1786",
            [{"rrf_score": 0.032, "doc": self.sample_chunks[0]["doc"]}],
        )
        self.assertGreater(conf_high, 0.60)

        # Out-of-corpus query (0 evidence overlap)
        conf_out = gate.calculate_confidence(
            "corporate tax code in Switzerland",
            [{"rrf_score": 0.016, "doc": self.sample_chunks[0]["doc"]}],
        )
        self.assertLess(conf_out, 0.40)
        print(f"✅ Test 5 Passed: Confidence calibration: In-Corpus={conf_high:.4f}, Out-Corpus={conf_out:.4f}")

    def test_06_valid_retrieval_not_falsely_refused(self):
        """Criterion 6: Valid BIS in-corpus queries pass the refusal gate."""
        gate = GuardrailGate(threshold=0.45)
        passed, conf, msg = gate.evaluate_and_gate(
            "high strength deformed steel bars IS 1786",
            [{"rrf_score": 0.032, "doc": self.sample_chunks[0]["doc"]}],
            category="is_standard",
        )
        self.assertTrue(passed)
        self.assertIsNone(msg)
        print(f"✅ Test 6 Passed: In-corpus query passed refusal gate (Conf: {conf:.4f})")

    def test_07_out_of_corpus_refusal(self):
        """Criterion 7: Out-of-corpus queries are refused and provide official portal guidance."""
        gate = GuardrailGate(threshold=0.45)
        passed, conf, msg = gate.evaluate_and_gate(
            "what is the property tax rate in Tokyo Japan",
            [{"rrf_score": 0.015, "doc": self.sample_chunks[0]["doc"]}],
            category="general",
        )
        self.assertFalse(passed)
        self.assertIsNotNone(msg)
        self.assertIn("https://www.bis.gov.in", msg)
        print(f"✅ Test 7 Passed: Out-of-corpus query refused with official portal redirect.")

    def test_08_dual_citation_generation(self):
        """Criterion 8: Citation engine formats Standard vs FAQ citations distinctly."""
        engine = CitationEngine()
        text = "Steel bars must conform to composition limits [As per IS 1786:2008, Clause 4.2]."
        res = engine.format_citations(text, self.sample_chunks)
        formatted = res["formatted_text"]
        self.assertIn("[Official Standard]", formatted)
        self.assertIn("[FAQ Guideline]", formatted)
        self.assertEqual(len(res["citations_list"]), 3)
        print("✅ Test 8 Passed: Dual citations rendered for Standards and FAQs.")

    def test_09_citation_to_context_validation(self):
        """Criterion 9: Citation validator catches ungrounded / hallucinated citations."""
        engine = CitationEngine()
        # Valid citation
        val_valid = engine.validate_citations_against_context(["IS 1786:2008, Clause 4.2"], self.sample_chunks)
        self.assertTrue(val_valid["all_valid"])
        self.assertEqual(val_valid["valid_count"], 1)

        # Fake/unretrieved citation
        val_fake = engine.validate_citations_against_context(["IS 99999:2099, Clause 99.9"], self.sample_chunks)
        self.assertFalse(val_fake["all_valid"])
        self.assertEqual(val_fake["ungrounded_count"], 1)
        print("✅ Test 9 Passed: Citation-to-context validation caught fake citation correctly.")

    def test_10_provenance_preservation(self):
        """Criterion 10: Formatted citations preserve Phase 1 hash and source_of_truth."""
        engine = CitationEngine()
        res = engine.format_citations("Testing output.", self.sample_chunks)
        for cite in res["citations_list"]:
            self.assertIn("source_hash", cite)
            self.assertIn("source_of_truth", cite)
            self.assertTrue(cite["source_hash"])
        print("✅ Test 10 Passed: 100% provenance preserved in citation list.")

    def test_11_intent_classification(self):
        """Criterion 11: QueryIntentRouter correctly identifies user query intents."""
        router = QueryIntentRouter()
        self.assertEqual(router.classify_intent("where are the testing labs in Delhi")["intent"], "lab_location")
        self.assertEqual(router.classify_intent("how to apply online for ISI mark license")["intent"], "certification_process")
        self.assertEqual(router.classify_intent("my gold jewellery hallmark is fake how to complain")["intent"], "consumer_complaint")
        self.assertEqual(router.classify_intent("is certification mandatory for solar PV modules")["intent"], "product_recommendation")
        print("✅ Test 11 Passed: Intent classification patterns verified.")

    def test_12_intent_to_category_mapping(self):
        """Criterion 12: Router maps user intents to verified Phase 1/2 corpus categories."""
        router = QueryIntentRouter()
        self.assertEqual(router.get_category_for_intent("lab_location"), "lab_directory")
        self.assertEqual(router.get_category_for_intent("product_recommendation"), "product_standard_mapping")
        self.assertEqual(router.get_category_for_intent("general_rag"), None)
        print("✅ Test 12 Passed: Intent-to-category taxonomy mapping verified.")

    def test_13_rag_pipeline_orchestration(self):
        """Criterion 13: Full RAG pipeline executes end-to-end with execution metadata."""
        pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)
        res = pipeline.query("what standard applies to gold jewellery hallmarking regulations")
        self.assertIn("status", res)
        self.assertIn("confidence_score", res)
        self.assertIn("intent", res)
        self.assertIn("citations", res)
        self.assertIn("response", res)
        print(f"✅ Test 13 Passed: Pipeline executed with status='{res['status']}' (Conf: {res['confidence_score']:.4f})")

    def test_14_mock_provider_determinism(self):
        """Criterion 14: Mock provider produces identical deterministic grounded responses."""
        provider = MockOfflineProvider()
        p1 = provider.generate("System", "User Query: steel IS 1786\n\nRetrieved Context:\n--- CONTEXT CHUNK 1 ---\nContent:\nText 123.")
        p2 = provider.generate("System", "User Query: steel IS 1786\n\nRetrieved Context:\n--- CONTEXT CHUNK 1 ---\nContent:\nText 123.")
        self.assertEqual(p1["response"], p2["response"])
        print("✅ Test 14 Passed: Mock provider determinism verified.")

    def test_15_openai_provider_config(self):
        """Criterion 15: OpenAI provider initializes with environment/config values."""
        prov = OpenAIProvider(api_key="sk-test-key", model_name="gpt-4o-mini")
        self.assertEqual(prov.model_name, "gpt-4o-mini")
        self.assertEqual(prov.api_key, "sk-test-key")
        print("✅ Test 15 Passed: OpenAI provider configuration verified.")

    def test_16_gemini_provider_config(self):
        """Criterion 16: Gemini provider initializes with environment/config values."""
        prov = GeminiProvider(api_key="gemini-test-key", model_name="gemini-1.5-flash")
        self.assertEqual(prov.model_name, "gemini-1.5-flash")
        self.assertEqual(prov.api_key, "gemini-test-key")
        print("✅ Test 16 Passed: Gemini provider configuration verified.")

    def test_17_numerical_hallucination_suppression(self):
        """Criterion 17: Grounded generator preserves exact numbers without inventing ungrounded figures."""
        gen = GroundedGenerator(provider_name="mock")
        res = gen.generate_response("application fee under Scheme I", [self.sample_chunks[1]])
        self.assertIn("1000", res["response"])
        self.assertIn("7000", res["response"])
        # Ensure it does not invent random numbers like Rs 50000
        self.assertNotIn("50000", res["response"])
        print("✅ Test 17 Passed: Numerical fidelity strictly preserved.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
