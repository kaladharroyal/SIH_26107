"""
Phase 4: Complete Integrated RAG Pipeline & Sub-Flow Dispatcher (rag_pipeline.py)
Unifies Phase 4 specialized sub-flows (Product Recommender, Scheme Walkthrough, Lab Locator, Consumer Complaints)
with Phase 2 Hybrid Retrieval, Phase 3 Evidence Guardrails, Grounded Generation, and Citation Validation.
"""

import logging
from typing import Any, Dict, List, Optional

from citation_engine import CitationEngine
from consumer_complaint import ConsumerComplaintHandler
from generator import GroundedGenerator
from guardrails import GuardrailGate
from lab_locator import LabLocator
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
    ):
        log.info("Initializing BIS RAG Pipeline Orchestrator & Sub-Flow Dispatcher...")
        self.router = QueryIntentRouter()
        self.retrieval = HybridRetrievalPipeline(use_mock_encoder=use_mock_retrieval)
        self.guardrail = GuardrailGate(threshold=confidence_threshold)
        self.generator = GroundedGenerator(provider_name=llm_provider)
        self.citation_engine = CitationEngine()

        # Initialize Phase 4 Specialized Sub-Flow Handlers
        self.product_recommender = ProductRecommender(retrieval_pipeline=self.retrieval)
        self.scheme_walkthrough = SchemeWalkthroughGuide()
        self.lab_locator = LabLocator(retrieval_pipeline=self.retrieval)
        self.consumer_complaint = ConsumerComplaintHandler()

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
        log.info(f"Processing Pipeline Query: '{clean_query[:50]}'")

        # 1. Intent Classification & Sub-Flow Routing
        routing_info = self.router.classify_intent(clean_query)
        intent = routing_info["intent"]
        effective_category = category if category is not None else routing_info.get("category")

        # 2. Dispatch to Specialized Sub-Flow if enabled and category is not manually overridden
        if enable_subflows and category is None:
            if intent == "product_recommendation":
                log.info(f"Dispatching to ProductRecommender for query: '{clean_query[:40]}'")
                rec_res = self.product_recommender.recommend(clean_query)
                if rec_res.get("status") == "success":
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
                    }

            elif intent == "certification_process":
                log.info(f"Dispatching to SchemeWalkthroughGuide for query: '{clean_query[:40]}'")
                walk_res = self.scheme_walkthrough.get_walkthrough(clean_query)
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
                }

            elif intent == "lab_location":
                log.info(f"Dispatching to LabLocator for query: '{clean_query[:40]}'")
                lab_res = self.lab_locator.search_labs(clean_query)
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
                }

            elif intent == "consumer_complaint":
                log.info(f"Dispatching to ConsumerComplaintHandler for query: '{clean_query[:40]}'")
                comp_res = self.consumer_complaint.handle_complaint(clean_query)
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
                }

        # 3. Grounded RAG Pipeline Execution (Phases 1-3)
        log.info(f"Executing Grounded RAG Pipeline for query: '{clean_query[:40]}' (Category: {effective_category})")
        retrieved_chunks = self.retrieval.retrieve(clean_query, top_n=top_n, category=effective_category)

        if not retrieved_chunks and effective_category is not None:
            log.info(f"Scoped category '{effective_category}' yielded 0 hits; falling back to broad retrieval.")
            retrieved_chunks = self.retrieval.retrieve(clean_query, top_n=top_n, category=None)
            effective_category = "broad_fallback"

        # Evidence sufficiency & uncertainty refusal gate
        passed, confidence, refusal_msg = self.guardrail.evaluate_and_gate(
            query=clean_query,
            retrieved_results=retrieved_chunks,
            category=effective_category or "general",
        )

        if not passed:
            log.warning(f"Query refused by Guardrail Gate (Confidence: {confidence:.4f})")
            return {
                "query": clean_query,
                "intent": intent,
                "flow_used": "general_rag",
                "status": "refused",
                "confidence_score": confidence,
                "category_used": effective_category,
                "response": refusal_msg,
                "results": None,
                "retrieved_chunks": retrieved_chunks,
                "citations": [],
                "fallback_used": True,
            }

        # Grounded answer generation & citation formatting
        gen_result = self.generator.generate_response(clean_query, retrieved_chunks)
        raw_answer = gen_result.get("response", "")
        citation_result = self.citation_engine.format_citations(raw_answer, retrieved_chunks)

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
                "chunks_count": len(retrieved_chunks),
            },
            "citations": citation_result["citations_list"],
            "citation_validation": citation_result.get("validation", {}),
            "retrieved_chunks": retrieved_chunks,
            "fallback_used": False,
        }


if __name__ == "__main__":
    pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)
    res = pipeline.query("what BIS standard should I use for a TMT bar?")
    print(f"\n--- SUB-FLOW RESULT (Flow: {res['flow_used']}, Status: {res['status']}) ---")
    print(res["response"])
