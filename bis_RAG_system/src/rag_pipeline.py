"""
Phase 3, Part 3.4: Complete RAG Pipeline Orchestrator (rag_pipeline.py)
Orchestrates the entire grounded generation lifecycle:
  User Query -> Intent Router -> Hybrid Retrieval -> Guardrail Gate -> Grounded Generator -> Citation Engine -> Verified Response
"""

import logging
from typing import Any, Dict, List, Optional

from citation_engine import CitationEngine
from generator import GroundedGenerator
from guardrails import GuardrailGate
from retrieval import HybridRetrievalPipeline
from router import QueryIntentRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("rag_pipeline")


class BISRAGPipeline:
    """
    Production BIS RAG Pipeline Orchestrator.
    Connects Phase 2 Hybrid Retrieval with Phase 3 Grounded Generation, Guardrails,
    Routing, and Citation Validation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        llm_provider: Optional[str] = None,
        use_mock_retrieval: bool = False,
    ):
        log.info("Initializing BIS RAG Pipeline Orchestrator...")
        self.router = QueryIntentRouter()
        self.retrieval = HybridRetrievalPipeline(use_mock_encoder=use_mock_retrieval)
        self.guardrail = GuardrailGate(threshold=confidence_threshold)
        self.generator = GroundedGenerator(provider_name=llm_provider)
        self.citation_engine = CitationEngine()

    def query(
        self,
        user_query: str,
        category: Optional[str] = None,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query answering with strict grounding and guardrails.
        """
        if not user_query or not user_query.strip():
            return {
                "query": user_query,
                "status": "refused",
                "confidence_score": 0.0,
                "intent": "empty_query",
                "category_used": None,
                "response": "Please provide a valid query regarding Bureau of Indian Standards compliance or technical standards.",
                "citations": [],
                "retrieved_chunks": [],
            }

        clean_query = user_query.strip()
        log.info(f"Processing RAG Pipeline Query: '{clean_query[:50]}'")

        # 1. Intent Classification & Category Mapping
        routing_info = self.router.classify_intent(clean_query)
        intent = routing_info["intent"]
        effective_category = category if category is not None else routing_info.get("category")

        # 2. Hybrid Retrieval (Sparse BM25 + Dense BGE-M3 + RRF + Rerank)
        retrieved_chunks = self.retrieval.retrieve(clean_query, top_n=top_n, category=effective_category)

        # Fallback to unconstrained search if strict category filter produced 0 hits
        if not retrieved_chunks and effective_category is not None:
            log.info(f"Scoped category '{effective_category}' yielded 0 hits; falling back to broad retrieval.")
            retrieved_chunks = self.retrieval.retrieve(clean_query, top_n=top_n, category=None)
            effective_category = "broad_fallback"

        # 3. Evidence Sufficiency & Uncertainty Refusal Gate
        passed, confidence, refusal_msg = self.guardrail.evaluate_and_gate(
            query=clean_query,
            retrieved_results=retrieved_chunks,
            category=effective_category or "general",
        )

        if not passed:
            log.warning(f"Query refused by Guardrail Gate (Confidence: {confidence:.4f})")
            return {
                "query": clean_query,
                "status": "refused",
                "confidence_score": confidence,
                "intent": intent,
                "category_used": effective_category,
                "response": refusal_msg,
                "retrieved_chunks": retrieved_chunks,
                "citations": [],
            }

        # 4. Grounded Response Generation
        gen_result = self.generator.generate_response(clean_query, retrieved_chunks)
        raw_answer = gen_result.get("response", "")

        # 5. Dual Citation Formatting & Context Validation
        citation_result = self.citation_engine.format_citations(raw_answer, retrieved_chunks)

        return {
            "query": clean_query,
            "status": "success",
            "confidence_score": confidence,
            "intent": intent,
            "category_used": effective_category,
            "response": citation_result["formatted_text"],
            "citations": citation_result["citations_list"],
            "citation_validation": citation_result.get("validation", {}),
            "model_used": gen_result.get("model_used", "unknown"),
            "retrieved_chunks": retrieved_chunks,
        }


if __name__ == "__main__":
    pipeline = BISRAGPipeline(llm_provider="mock")
    res = pipeline.query("what standard applies to gold jewellery hallmarking regulations")
    print(f"\n--- RAG PIPELINE RESULT (Status: {res['status']}, Conf: {res['confidence_score']:.4f}) ---")
    print(res["response"])
