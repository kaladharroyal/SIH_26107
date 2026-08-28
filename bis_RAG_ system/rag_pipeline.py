"""
Phase 3, Part 3.4: RAG Pipeline Orchestrator (rag_pipeline.py)
Connects Hybrid Retrieval -> Guardrail Gate -> Grounded Generation -> Citation Engine.
"""

import logging
from typing import Dict, Any, Optional

from retrieval import HybridRetrievalPipeline
from guardrails import GuardrailGate
from generator import GroundedGenerator
from citation_engine import CitationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag_pipeline")


class BISRAGPipeline:
    def __init__(self, confidence_threshold: float = 0.45):
        log.info("Initializing Complete Phase 1-3 BIS RAG Pipeline...")
        self.retrieval = HybridRetrievalPipeline()
        self.guardrail = GuardrailGate(threshold=confidence_threshold)
        self.generator = GroundedGenerator()
        self.citation_engine = CitationEngine()

    def query(self, user_query: str, category: Optional[str] = None) -> Dict[str, Any]:
        log.info(f"Processing Pipeline Query: '{user_query}'")

        # 1. Retrieve top context chunks (Hybrid BM25 + Vector + Reranking)
        retrieved_chunks = self.retrieval.retrieve(user_query, top_n=5, category=category)

        # 2. Evaluate Guardrail Refusal Gate
        passed, confidence, refusal_msg = self.guardrail.evaluate_and_gate(
            user_query, retrieved_chunks, category=category or "general"
        )

        if not passed:
            return {
                "query": user_query,
                "status": "refused",
                "confidence_score": confidence,
                "response": refusal_msg,
                "retrieved_chunks": retrieved_chunks,
                "citations": [],
            }

        # 3. Generate Grounded Response
        gen_result = self.generator.generate_response(user_query, retrieved_chunks)
        raw_text = gen_result["response"]

        # 4. Apply Dual Citation Engine
        citation_result = self.citation_engine.format_citations(raw_text, retrieved_chunks)

        return {
            "query": user_query,
            "status": "success",
            "confidence_score": confidence,
            "response": citation_result["formatted_text"],
            "citations": citation_result["citations_list"],
            "model_used": gen_result["model_used"],
            "retrieved_chunks": retrieved_chunks,
        }


if __name__ == "__main__":
    pipeline = BISRAGPipeline()
    res = pipeline.query("what standard applies to gold jewellery hallmarking")
    print(f"\n--- RAG RESPONSE (Confidence: {res['confidence_score']:.4f}) ---")
    print(res["response"])
