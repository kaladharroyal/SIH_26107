"""
Phase 4: Complete Integrated RAG Pipeline & Sub-Flow Dispatcher (rag_pipeline.py)
Unifies Phase 4 specialized sub-flows (Product Recommender, Scheme Walkthrough, Lab Locator, Consumer Complaints)
with Phase 2 Hybrid Retrieval, Phase 3 Evidence Guardrails, Grounded Generation, and Citation Validation.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from citation_engine import CitationEngine
from consumer_complaint import ConsumerComplaintHandler
from generator import GroundedGenerator
from guardrails import GuardrailGate
from lab_locator import LabLocator
from multilingual import MultilingualHandler
from product_recommender import ProductRecommender
from retrieval import HybridRetrievalPipeline
from router import QueryIntentRouter
from scheme_walkthrough import SchemeWalkthroughGuide

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("rag_pipeline")


class BISRAGPipeline:
    """
    Production BIS AI Compliance Pipeline & Sub-Flow Dispatcher.
    Routes queries to specialized deterministic sub-flows (Phases 4) or Phase 3 Grounded RAG.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        llm_provider: Optional[str] = None,
        use_mock_retrieval: bool = False,
        use_fast_retrieval: bool = True,
    ):
        self.use_fast_retrieval = use_fast_retrieval
        log.info(f"Initializing BIS RAG Pipeline (Fast Retrieval Mode: {self.use_fast_retrieval})...")
        self.router = QueryIntentRouter()
        self.multilingual = MultilingualHandler()
        
        # When fast retrieval is enabled, skip loading heavy neural encoders into memory
        encoder_mock = True if self.use_fast_retrieval else use_mock_retrieval
        self.retrieval = HybridRetrievalPipeline(use_mock_encoder=encoder_mock)
        self.guardrail = GuardrailGate(threshold=confidence_threshold)
        self.generator = GroundedGenerator(provider_name=llm_provider)
        self.citation_engine = CitationEngine()

        # Initialize Phase 4 Specialized Sub-Flow Handlers
        self.product_recommender = ProductRecommender(retrieval_pipeline=self.retrieval)
        self.scheme_walkthrough = SchemeWalkthroughGuide()
        self.lab_locator = LabLocator(retrieval_pipeline=self.retrieval)
        self.consumer_complaint = ConsumerComplaintHandler()

    def process_query(self, user_query: str, **kwargs):
        """Alias for query() to support benchmark test suites."""
        return self.query(user_query, **kwargs)

    def query(
        self,
        user_query: str,
        category: Optional[str] = None,
        top_n: int = 5,
        enable_subflows: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes query routing, specialized sub-flow execution, or grounded RAG retrieval.
        """
        if not user_query or not user_query.strip():
            return {
                "query": user_query,
                "intent": "empty_query",
                "flow_used": "none",
                "status": "refused",
                "confidence_score": 0.0,
                "category_used": None,
                "response": "Please provide a valid query regarding Bureau of Indian Standards compliance or technical standards.",
                "results": None,
                "citations": [],
                "retrieved_chunks": [],
                "fallback_used": False,
            }

        clean_query = user_query.strip()
        t_start = time.time()
        log.info(f"Processing Pipeline Query: '{clean_query[:50]}'")

        # 0. Multilingual Detection & Search Query Normalization
        lang_info = self.multilingual.detect_language(clean_query)
        detected_lang = lang_info.get("lang_name", "English")
        lang_code = lang_info.get("lang_code", "en")

        if lang_code == "hinglish":
            search_query = self.multilingual.normalize_hinglish_to_english(clean_query)
        elif lang_code in ["hi", "te", "ta", "bn"]:
            search_query = self.multilingual.normalize_native_to_english_keywords(clean_query)
        else:
            search_query = clean_query

        # 1. Intent Classification & Sub-Flow Routing
        routing_info = self.router.classify_intent(search_query)
        intent = routing_info["intent"]
        effective_category = category if category is not None else routing_info.get("category")

        # 2. Dispatch to Specialized Sub-Flow if enabled and category is not manually overridden
        if enable_subflows and category is None:
            if intent == "product_recommendation":
                log.info(f"Dispatching to ProductRecommender for query: '{search_query[:40]}' (Lang: {detected_lang})")
                rec_res = self.product_recommender.recommend(search_query, language=detected_lang)
                if rec_res.get("status") == "success":
                    total_ms = round((time.time() - t_start) * 1000, 2)
                    return {
                        "query": clean_query,
                        "intent": intent,
                        "flow_used": "product_recommender",
                        "status": "success",
                        "confidence_score": 0.95,
                        "category_used": "product_standard_mapping",
                        "response": rec_res["formatted_text"],
                        "results": rec_res.get("product_data"),
                        "citations": [rec_res.get("provenance")] if rec_res.get("provenance") else [],
                        "source": rec_res.get("source"),
                        "fallback_used": rec_res.get("fallback_used", False),
                        "response_language": detected_lang,
                        "total_ms": total_ms,
                    }

            elif intent == "certification_process":
                log.info(f"Dispatching to SchemeWalkthroughGuide for query: '{clean_query[:40]}'")
                walk_res = self.scheme_walkthrough.get_walkthrough(clean_query)
                total_ms = round((time.time() - t_start) * 1000, 2)
                return {
                    "query": clean_query,
                    "intent": intent,
                    "flow_used": "scheme_walkthrough",
                    "status": "success",
                    "confidence_score": 0.98,
                    "category_used": "general_policy",
                    "response": walk_res["formatted_text"],
                    "results": {
                        "scheme_key": walk_res.get("scheme_key"),
                        "fee_schedule": walk_res.get("fee_schedule"),
                        "steps": walk_res.get("steps"),
                    },
                    "citations": [],
                    "source": "official_scheme_walkthrough",
                    "fallback_used": False,
                    "response_language": detected_lang,
                    "total_ms": total_ms,
                }

            elif intent == "lab_location":
                log.info(f"Dispatching to LabLocator for query: '{clean_query[:40]}'")
                lab_res = self.lab_locator.search_labs(clean_query)
                total_ms = round((time.time() - t_start) * 1000, 2)
                return {
                    "query": clean_query,
                    "intent": intent,
                    "flow_used": "lab_locator",
                    "status": lab_res.get("status", "success"),
                    "confidence_score": 0.90 if lab_res.get("total_found", 0) > 0 else 0.50,
                    "category_used": "lab_directory",
                    "response": lab_res["formatted_text"],
                    "results": lab_res.get("labs", []),
                    "citations": [],
                    "source": lab_res.get("source"),
                    "fallback_used": lab_res.get("fallback_used", False),
                    "response_language": detected_lang,
                    "total_ms": total_ms,
                }

            elif intent == "consumer_complaint":
                log.info(f"Dispatching to ConsumerComplaintHandler for query: '{clean_query[:40]}'")
                comp_res = self.consumer_complaint.handle_complaint(clean_query)
                total_ms = round((time.time() - t_start) * 1000, 2)
                return {
                    "query": clean_query,
                    "intent": intent,
                    "flow_used": "consumer_complaint",
                    "status": "success",
                    "confidence_score": 0.99,
                    "category_used": "general_policy",
                    "response": comp_res["formatted_text"],
                    "results": {
                        "category": comp_res.get("category"),
                        "is_hallmarking": comp_res.get("is_hallmarking"),
                        "compensation_rights": comp_res.get("compensation_rights"),
                    },
                    "citations": [],
                    "source": "official_bis_care_portal",
                    "fallback_used": False,
                    "response_language": detected_lang,
                    "total_ms": total_ms,
                }

        # 3. Grounded RAG Pipeline Execution (Phases 1-3)
        t_ret_start = time.time()
        if self.use_fast_retrieval:
            log.info(f"Executing Fast Retrieval Pipeline for search query: '{search_query[:40]}' (Category: {effective_category})")
            retrieved_chunks = self.retrieval.retrieve_fast(search_query, top_n=top_n, category=effective_category)
            if not retrieved_chunks and effective_category is not None:
                log.info(f"Scoped category '{effective_category}' yielded 0 hits; falling back to broad fast retrieval.")
                retrieved_chunks = self.retrieval.retrieve_fast(search_query, top_n=top_n, category=None)
                effective_category = "broad_fallback"
        else:
            log.info(f"Executing Full Hybrid Retrieval Pipeline for search query: '{search_query[:40]}' (Category: {effective_category})")
            retrieved_chunks = self.retrieval.retrieve(search_query, top_n=top_n, category=effective_category)
            if not retrieved_chunks and effective_category is not None:
                log.info(f"Scoped category '{effective_category}' yielded 0 hits; falling back to broad retrieval.")
                retrieved_chunks = self.retrieval.retrieve(search_query, top_n=top_n, category=None)
                effective_category = "broad_fallback"
        retrieval_ms = round((time.time() - t_ret_start) * 1000, 2)

        # Evidence sufficiency & uncertainty refusal gate
        passed, confidence, refusal_msg = self.guardrail.evaluate_and_gate(
            query=search_query,
            retrieved_results=retrieved_chunks,
            category=effective_category or "general",
        )

        if not passed:
            total_ms = round((time.time() - t_start) * 1000, 2)
            log.warning(f"Query refused by Guardrail Gate (Confidence: {confidence:.4f})")
            if "hindi" in detected_lang.lower() or detected_lang.lower() == "hi":
                localized_refusal = "उपलब्ध BIS सामग्री इस जानकारी की पुष्टि करने के लिए पर्याप्त नहीं है।"
            elif "telugu" in detected_lang.lower() or detected_lang.lower() == "te":
                localized_refusal = "లభ్యమైన BIS సమాచారం దీనిని నిర్ధారించడానికి సరిపోదు."
            else:
                localized_refusal = refusal_msg

            return {
                "query": clean_query,
                "intent": intent,
                "flow_used": "general_rag",
                "status": "refused",
                "confidence_score": confidence,
                "category_used": effective_category,
                "response": localized_refusal,
                "results": None,
                "retrieved_chunks": retrieved_chunks,
                "citations": [],
                "fallback_used": True,
                "response_language": detected_lang,
                "retrieval_ms": retrieval_ms,
                "generation_ms": 0.0,
                "total_ms": total_ms,
            }

        # Grounded answer generation with strict response language & citation formatting
        t_gen_start = time.time()
        gen_result = self.generator.generate_response(
            query=search_query,
            context_chunks=retrieved_chunks,
            response_language=detected_lang,
            original_query=clean_query,
        )
        generation_ms = round((time.time() - t_gen_start) * 1000, 2)
        total_ms = round((time.time() - t_start) * 1000, 2)

        raw_answer = gen_result.get("response", "")
        citation_result = self.citation_engine.format_citations(raw_answer, retrieved_chunks, language=detected_lang)

        return {
            "query": clean_query,
            "intent": intent,
            "flow_used": "general_rag",
            "status": "success",
            "confidence_score": confidence,
            "category_used": effective_category,
            "response": citation_result["formatted_text"],
            "results": {
                "model_used": gen_result.get("model_used", "unknown"),
                "provider": gen_result.get("provider", "unknown"),
                "chunks_count": len(retrieved_chunks),
            },
            "citations": citation_result["citations_list"],
            "citation_validation": citation_result.get("validation", {}),
            "retrieved_chunks": retrieved_chunks,
            "provider": gen_result.get("provider", "unknown"),
            "model_used": gen_result.get("model_used", "unknown"),
            "fallback_triggered": gen_result.get("fallback_triggered", False),
            "fallback_used": gen_result.get("fallback_triggered", False),
            "response_language": detected_lang,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        }


if __name__ == "__main__":
    pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)
    res = pipeline.query("what BIS standard should I use for a TMT bar?")
    print(f"\n--- SUB-FLOW RESULT (Flow: {res['flow_used']}, Status: {res['status']}) ---")
    print(res["response"])
