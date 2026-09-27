"""
Live Verification Audit of Cloud RAG Remediation.
Tests the 7 mandatory queries against the active production BISRAGPipeline:
1. Define BIS
2. What is the BIS Act?
3. What is TMT steel?
4. What is OPC cement?
5. What is PPC cement?
6. What is the CRS Scheme II?
7. What is UltraConcrete-X?

Records all diagnostic metrics, Qdrant Cloud connection proof, BGE-M3 embeddings,
guardrail confidence scores, Gemini generation, and citation validation.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Set offline mode for local HF cache so no unnecessary HEAD requests are sent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("remediation_audit")

from rag_pipeline import BISRAGPipeline

AUDIT_QUERIES = [
    "Define BIS",
    "What is the BIS Act?",
    "What is TMT steel?",
    "What is OPC cement?",
    "What is PPC cement?",
    "What is the CRS Scheme II?",
    "What is UltraConcrete-X?",
]

def main():
    print("=" * 80)
    print("STARTING LIVE CLOUD RAG REMEDIATION AUDIT (7 MANDATORY QUERIES)")
    print("=" * 80)

    # Instantiate production pipeline exactly as app.py does
    t0_init = time.perf_counter()
    pipeline = BISRAGPipeline()
    init_ms = (time.perf_counter() - t0_init) * 1000
    print(f"\nPipeline initialized in {init_ms:.2f} ms")
    print(f"Retriever active: {type(pipeline.qdrant_retriever).__name__ if pipeline.qdrant_retriever else 'None'}")
    if pipeline.qdrant_retriever:
        print(f"Qdrant Cloud Points: {pipeline.qdrant_retriever.qdrant_points_count}")
        print(f"Vector Dimension:    {pipeline.qdrant_retriever.qdrant_vector_size}")
        print(f"Embedding Model:     {pipeline.qdrant_retriever.model_name}")
        print(f"BM25 Docs Count:     {pipeline.qdrant_retriever.bm25_doc_count}")

    records = []

    for idx, query in enumerate(AUDIT_QUERIES, 1):
        print("\n" + "-" * 75)
        print(f"[{idx}/7] QUERY: '{query}'")
        print("-" * 75)

        t_start = time.perf_counter()
        result = pipeline.query(query)
        turnaround_ms = (time.perf_counter() - t_start) * 1000

        retriever_name = "QdrantHybridRetriever" if pipeline.qdrant_retriever else "HybridRetrievalPipeline"
        qdrant_contacted = pipeline.qdrant_retriever is not None and result.get("flow_used") == "general_rag"
        bgem3_used = "BAAI/bge-m3" if qdrant_contacted else "None"
        retrieved_chunks = result.get("retrieved_chunks", [])
        chunk_count = len(retrieved_chunks)
        guardrail_score = result.get("confidence_score", 0.0)
        provider = result.get("provider", "None")
        model = result.get("model_used", "None")
        status = result.get("status", "unknown")
        fallback_used = result.get("fallback_used", False)
        citations = result.get("citations", [])
        citation_val = result.get("citation_validation", {})
        intent = result.get("intent", "unknown")
        response_text = result.get("response", "")

        rec = {
            "query": query,
            "intent": intent,
            "retriever": retriever_name,
            "qdrant_cloud_contacted": qdrant_contacted,
            "bge_m3_embedding_used": bgem3_used,
            "retrieved_chunks_count": chunk_count,
            "guardrail_score": round(guardrail_score, 4),
            "guardrail_passed": status == "success",
            "provider": provider,
            "model": model,
            "status": status,
            "fallback_used": fallback_used,
            "citations_count": len(citations),
            "citation_validation": citation_val,
            "retrieval_ms": result.get("retrieval_ms"),
            "generation_ms": result.get("generation_ms"),
            "total_ms": round(turnaround_ms, 2),
            "response_preview": response_text[:200] + ("..." if len(response_text) > 200 else ""),
        }
        records.append(rec)

        print(f"Intent:                  {intent}")
        print(f"Retriever:               {retriever_name}")
        print(f"Qdrant Cloud Contacted:  {qdrant_contacted}")
        print(f"BGE-M3 Query Embedding:  {bgem3_used}")
        print(f"Retrieved Chunks:        {chunk_count}")
        print(f"Guardrail Score:         {guardrail_score:.4f} (Status: {status})")
        print(f"LLM Provider:            {provider} (Model: {model})")
        print(f"Fallback Used:           {fallback_used}")
        print(f"Citations:               {len(citations)}")
        print(f"Total Latency:           {turnaround_ms:.2f} ms")
        print(f"Response: {rec['response_preview']}")

        # Pacing between Gemini requests to avoid 429 quota exhaustion
        time.sleep(3)

    out_file = BASE_DIR / "tests" / "results" / "cloud_rag_remediation_audit_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nAll 7 query results saved to: {out_file}")

if __name__ == "__main__":
    main()
