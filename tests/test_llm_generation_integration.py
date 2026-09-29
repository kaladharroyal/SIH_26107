"""
Strict LLM Answer Generation Integration Test Suite.
Verifies the complete pipeline:
  USER REQUEST
       ↓
  BM25 + (BGE-M3 -> QDRANT)
       ↓
      RRF
       ↓
  SELECTED EVIDENCE
       ↓
   REAL LLM (Gemini 2.5 Flash)
       ↓
  NATURAL LANGUAGE ANSWER
       ↓
  CITATION VALIDATION

No retrieval logic in the LLM.
No hardcoded answers or domain mappings.
Zero hallucination. Full provenance traceability.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from citation_engine import CitationEngine
from generator import GroundedGenerator
from qdrant_retrieval import QdrantHybridRetriever

# 11 mandatory user queries
TEST_QUERIES = [
    "Which Indian Standard applies to high strength deformed steel bars for concrete reinforcement?",
    "Which Indian Standard applies to TMT bars?",
    "Which Indian Standard covers Ordinary Portland Cement?",
    "Which IS covers PPC?",
    "What is the Compulsory Registration Scheme?",
    "What is the purpose of CRS?",
    "What is BIS?",
    "What is the BIS Act?",
    "What is hallmarking?",
    "UltraConcrete-X",
    "UnknownProduct",
]

# 1 controlled query where retrieved evidence is intentionally insufficient
CONTROLLED_QUERY = "What is the BIS standard for XYZ imaginary product?"


def file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 70)
    print("PHASE: STRICT LLM ANSWER GENERATION INTEGRATION TEST SUITE")
    print("=" * 70)

    # 1. Baseline Integrity Checks
    chunks_path = BASE_DIR / "processed_chunks.jsonl"
    bm25_path = BASE_DIR / "vector_index" / "bm25_index.pkl"

    print("\n--- BASELINE DATA INTEGRITY ---")
    sha_chunks_before = file_sha256(chunks_path)
    sha_bm25_before = file_sha256(bm25_path)
    print(f"processed_chunks.jsonl SHA-256: {sha_chunks_before}")
    print(f"bm25_index.pkl         SHA-256: {sha_bm25_before}")

    # 2. Initialize Hybrid Retriever
    print("\n--- INITIALIZING HYBRID RETRIEVER (BM25 + BGE-M3 + QDRANT CLOUD) ---")
    t0_init = time.perf_counter()
    retriever = QdrantHybridRetriever(
        bm25_path=bm25_path,
    )
    init_ret_ms = (time.perf_counter() - t0_init) * 1000
    print(f"Retriever initialized in {init_ret_ms:.2f} ms")
    print(f"Qdrant Points Count: {retriever.qdrant_points_count}")
    print(f"BM25 Docs Count:     {retriever.bm25_doc_count}")

    # 3. Initialize Real LLM Generator (Gemini)
    print("\n--- INITIALIZING REAL LLM GENERATOR ---")
    generator = GroundedGenerator(provider_name="gemini")
    provider_type = type(generator.provider).__name__
    model_name = getattr(generator.provider, "model_name", "unknown")
    print(f"Provider: {provider_type} (Model: {model_name})")

    if "mock" in provider_type.lower():
        raise RuntimeError("CRITICAL ERROR: Real LLM provider is unavailable. Mock provider cannot be used!")

    # 4. Initialize Citation Engine
    citation_engine = CitationEngine()

    results = []

    # Run the 11 pipeline tests
    print("\n" + "=" * 70)
    print("RUNNING 11 MANDATORY PIPELINE TESTS")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/11] TEST QUERY: \"{query}\"")
        t_start = time.perf_counter()

        # Step 1: BM25 retrieval
        bm25_hits, t_bm25 = retriever.retrieve_bm25(query, top_k=10)

        # Step 2: BGE-M3 + Qdrant retrieval
        bge_hits, t_embed, t_qdrant = retriever.retrieve_bge_m3(query, top_k=10)

        # Step 3: RRF Fusion
        fused_hits, t_rrf = retriever.fuse_rrf(bm25_hits, bge_hits, rrf_k=60, top_k=10)
        t_retrieval_total = (time.perf_counter() - t_start) * 1000

        # Step 4: LLM Generation
        t0_gen = time.perf_counter()
        gen_res = generator.generate_response(query, fused_hits)
        t_gen = (time.perf_counter() - t0_gen) * 1000

        # Step 5: Citation Validation
        raw_answer = gen_res.get("response", "")
        provider_used = gen_res.get("provider", "unknown")
        model_used = gen_res.get("model_used", "unknown")

        emitted_citations = citation_engine.extract_inline_citations(raw_answer)
        validation_res = citation_engine.validate_citations_against_context(emitted_citations, fused_hits)
        total_time_ms = (time.perf_counter() - t_start) * 1000

        # Check refusal for negative queries
        is_negative_query = query in ["UltraConcrete-X", "UnknownProduct"]
        lower_ans = raw_answer.lower()
        refusal_detected = any(phrase in lower_ans for phrase in ["insufficient", "does not provide", "not found", "cannot establish", "not provide enough", "no information", "no mention"])

        # Trace details
        top_evidence = [
            {
                "chunk_id": h["chunk_id"],
                "standard": h.get("standard"),
                "title": h.get("title"),
                "source_pdf": h.get("source_pdf"),
                "page": h.get("page"),
                "rrf_score": round(h.get("rrf_score", 0.0), 5),
            }
            for h in fused_hits[:3]
        ]

        print(f"  → BM25 Latency:        {t_bm25:.2f} ms ({len(bm25_hits)} hits)")
        print(f"  → BGE-M3 Embed:        {t_embed:.2f} ms")
        print(f"  → Qdrant Latency:      {t_qdrant:.2f} ms ({len(bge_hits)} hits)")
        print(f"  → RRF Latency:         {t_rrf:.2f} ms ({len(fused_hits)} fused)")
        print(f"  → LLM Latency:         {t_gen:.2f} ms (Provider: {provider_used}, Model: {model_used})")
        print(f"  → Total Turnaround:    {total_time_ms:.2f} ms")
        print(f"  → Citations Emitted:   {emitted_citations}")
        print(f"  → Citation Validation: Valid={validation_res['valid_count']}, Ungrounded={validation_res['ungrounded_count']}, AllValid={validation_res['all_valid']}")
        if is_negative_query:
            print(f"  → Negative Query Refusal Verified: {refusal_detected}")
        print(f"  → Generated Answer:\n    {raw_answer.strip()}")

        results.append({
            "test_id": i,
            "type": "negative_query" if is_negative_query else "standard_query",
            "query": query,
            "bm25_latency_ms": round(t_bm25, 2),
            "bge_embed_latency_ms": round(t_embed, 2),
            "qdrant_latency_ms": round(t_qdrant, 2),
            "rrf_latency_ms": round(t_rrf, 2),
            "total_retrieval_ms": round(t_retrieval_total, 2),
            "evidence_count": len(fused_hits),
            "top_evidence": top_evidence,
            "llm_provider": provider_used,
            "llm_model": model_used,
            "llm_latency_ms": round(t_gen, 2),
            "total_latency_ms": round(total_time_ms, 2),
            "emitted_citations": emitted_citations,
            "citation_validation": validation_res,
            "negative_refusal_verified": refusal_detected if is_negative_query else None,
            "generated_answer": raw_answer.strip(),
        })

        # Pacing to adhere smoothly to free-tier rate limits (5 RPM)
        if i < len(TEST_QUERIES):
            print("  (Pacing 3s for API rate limits...)")
            time.sleep(3)

    # Run Controlled Insufficient Evidence Test
    print("\n" + "=" * 70)
    print("RUNNING CONTROLLED TEST (INTENTIONALLY INSUFFICIENT EVIDENCE)")
    print("=" * 70)
    print(f"QUERY: \"{CONTROLLED_QUERY}\"")
    print("  (Pacing 3s before controlled test...)")
    time.sleep(3)

    # Pass unrelated evidence chunks to verify the LLM does NOT use outside knowledge
    unrelated_evidence = [
        {
            "chunk_id": "bis_faq_general_01",
            "source_pdf": "BIS_FAQ_General.pdf",
            "page": "1",
            "standard": "BIS-GUIDELINES",
            "category": "general_policy",
            "title": "General BIS Procedures and Guidelines",
            "text": "The Bureau of Indian Standards operates various conformity assessment schemes under the BIS Act, 2016 for standardization and certification of goods.",
        },
        {
            "chunk_id": "is_10322_part5_sec1",
            "source_pdf": "IS_10322_Part5_Sec1.pdf",
            "page": "3",
            "standard": "IS 10322 (Part 5/Sec 1)",
            "category": "is_standard",
            "title": "Luminaires - Particular Requirements for Fixed General Purpose Luminaires",
            "text": "This standard specifies requirements for fixed general purpose luminaires for use with tungsten filament, tubular fluorescent and other discharge lamps.",
        }
    ]

    t0_ctrl = time.perf_counter()
    ctrl_gen_res = generator.generate_response(CONTROLLED_QUERY, unrelated_evidence)
    t_ctrl_gen = (time.perf_counter() - t0_ctrl) * 1000
    ctrl_answer = ctrl_gen_res.get("response", "")
    ctrl_emitted = citation_engine.extract_inline_citations(ctrl_answer)
    ctrl_val = citation_engine.validate_citations_against_context(ctrl_emitted, unrelated_evidence)

    print(f"  → LLM Latency:         {t_ctrl_gen:.2f} ms")
    print(f"  → Citations Emitted:   {ctrl_emitted}")
    print(f"  → Citation Validation: Valid={ctrl_val['valid_count']}, Ungrounded={ctrl_val['ungrounded_count']}, AllValid={ctrl_val['all_valid']}")
    print(f"  → Generated Answer:\n    {ctrl_answer.strip()}")

    # Check that answer states insufficient evidence
    lower_ans = ctrl_answer.lower()
    is_refusal = any(phrase in lower_ans for phrase in ["insufficient", "does not provide", "not found", "cannot establish", "not provide enough", "no mention", "no information"])
    print(f"  → Grounded Refusal Check Passed: {is_refusal}")

    results.append({
        "test_id": 12,
        "type": "controlled_insufficient_evidence",
        "query": CONTROLLED_QUERY,
        "bm25_latency_ms": 0.0,
        "bge_embed_latency_ms": 0.0,
        "qdrant_latency_ms": 0.0,
        "rrf_latency_ms": 0.0,
        "total_retrieval_ms": 0.0,
        "evidence_count": len(unrelated_evidence),
        "top_evidence": [
            {"chunk_id": c["chunk_id"], "standard": c["standard"], "title": c["title"]}
            for c in unrelated_evidence
        ],
        "llm_provider": ctrl_gen_res.get("provider", "unknown"),
        "llm_model": ctrl_gen_res.get("model_used", "unknown"),
        "llm_latency_ms": round(t_ctrl_gen, 2),
        "total_latency_ms": round(t_ctrl_gen, 2),
        "emitted_citations": ctrl_emitted,
        "citation_validation": ctrl_val,
        "grounded_refusal_confirmed": is_refusal,
        "generated_answer": ctrl_answer.strip(),
    })

    # 5. Post-Test Integrity Checks
    print("\n" + "=" * 70)
    print("POST-TEST DATA INTEGRITY VERIFICATION")
    print("=" * 70)
    sha_chunks_after = file_sha256(chunks_path)
    sha_bm25_after = file_sha256(bm25_path)

    # Re-check Qdrant points count using the existing open client
    col = retriever.client.get_collection("bis_chunks_bge_m3_v1")
    qdrant_count_after = col.points_count

    print(f"processed_chunks.jsonl SHA-256 Before: {sha_chunks_before}")
    print(f"processed_chunks.jsonl SHA-256 After:  {sha_chunks_after}")
    chunks_intact = (sha_chunks_before == sha_chunks_after)
    print(f"  → Chunks File Intact: {chunks_intact}")

    print(f"bm25_index.pkl         SHA-256 Before: {sha_bm25_before}")
    print(f"bm25_index.pkl         SHA-256 After:  {sha_bm25_after}")
    bm25_intact = (sha_bm25_before == sha_bm25_after)
    print(f"  → BM25 Index Intact:  {bm25_intact}")

    print(f"Qdrant Points Count    Before:         {retriever.qdrant_points_count}")
    print(f"Qdrant Points Count    After:          {qdrant_count_after}")
    qdrant_intact = (retriever.qdrant_points_count == qdrant_count_after)
    print(f"  → Qdrant Count Intact: {qdrant_intact}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "llm_provider": generator.provider_name,
        "llm_model": model_name,
        "data_integrity": {
            "processed_chunks_sha256": sha_chunks_after,
            "processed_chunks_intact": chunks_intact,
            "bm25_index_sha256": sha_bm25_after,
            "bm25_index_intact": bm25_intact,
            "qdrant_points_count": qdrant_count_after,
            "qdrant_points_intact": qdrant_intact,
        },
        "tests": results,
    }

    # Save results to tests/results/
    out_dir = BASE_DIR / "tests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "llm_generation_integration_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll results saved to: {out_file}")


if __name__ == "__main__":
    main()
