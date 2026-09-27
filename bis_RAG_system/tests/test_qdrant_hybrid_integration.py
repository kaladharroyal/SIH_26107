"""
Integration Test Suite for Qdrant + BGE-M3 + BM25 Hybrid Retrieval.
Tests all 15 specified queries across BM25-only, BGE-M3/Qdrant-only, and Hybrid RRF.
Measures performance, checks special query behaviors, and verifies data integrity.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Set offline mode for fast startup (weights are already cached)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure src is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from qdrant_retrieval import QdrantHybridRetriever

TEST_QUERIES = [
    # 1. Steel / High Strength
    "Which Indian Standard applies to high strength deformed steel bars for concrete reinforcement?",
    # 2. TMT bars
    "Which Indian Standard applies to TMT bars?",
    # 3. Hindi TMT
    "टीएमटी बार के लिए कौन सा भारतीय मानक है?",
    # 4. Telugu TMT
    "తెలుగులో TMT బార్లకు ఏ భారతీయ ప్రమాణం వర్తిస్తుంది?",
    # 5. OPC
    "Which Indian Standard covers Ordinary Portland Cement?",
    # 6. PPC
    "Which IS covers PPC?",
    # 7. CRS general
    "What is the Compulsory Registration Scheme?",
    # 8. CRS purpose
    "What is the purpose of CRS?",
    # 9. CRS products
    "What products fall under CRS?",
    # 10. BIS general
    "What is BIS?",
    # 11. BIS Act
    "What is the BIS Act?",
    # 12. Hallmarking
    "What is hallmarking?",
    # 13. Negative 1
    "UltraConcrete-X",
    # 14. Negative 2
    "UnknownProduct",
    # 15. FakeSteel
    "FakeSteel",
]


def file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_query_relevance(query_idx: int, query_text: str, hits: List[Dict[str, Any]]) -> int:
    """
    Evaluates how many of the top-10 hits are genuinely relevant to the query topic
    based strictly on retrieved textual content.
    """
    count = 0
    q_lower = query_text.lower()
    for h in hits:
        text = (h.get("text") or "").lower()
        title = (h.get("title") or "").lower()
        std = (h.get("standard") or "").lower()
        combined = f"{std} {title} {text}"

        is_relevant = False
        if query_idx in [0, 1, 2, 3]:  # High strength deformed steel / TMT
            if "1786" in combined or "deformed" in combined or "reinforcement" in combined or "steel bars" in combined or "tmt" in combined:
                is_relevant = True
        elif query_idx == 4:  # Ordinary Portland Cement
            if "269" in combined or "ordinary portland cement" in combined or "portland cement" in combined:
                is_relevant = True
        elif query_idx == 5:  # PPC
            if "1489" in combined or "portland pozzolana" in combined or "ppc" in combined:
                is_relevant = True
        elif query_idx in [6, 7, 8]:  # CRS
            if "compulsory registration" in combined or "crs" in combined or "electronics and information technology" in combined or "scheme-ii" in combined:
                is_relevant = True
        elif query_idx in [9, 10]:  # BIS / BIS Act
            if "bureau of indian standards act" in combined or "functions of the bureau" in combined or "powers of the bureau" in combined or "establishment of the bureau" in combined or "bis act" in combined:
                is_relevant = True
        elif query_idx == 11:  # Hallmarking
            if "hallmark" in combined or "precious metal" in combined or "gold" in combined or "1417" in combined:
                is_relevant = True
        elif query_idx in [12, 13, 14]:  # Negative queries
            # Negative queries have no genuine product in BIS
            is_relevant = False

        if is_relevant:
            count += 1
    return count


def run_benchmark():
    print("=" * 80)
    print("STARTING QDRANT + BGE-M3 + BM25 HYBRID RETRIEVAL INTEGRATION BENCHMARK")
    print("=" * 80)

    chunks_file = BASE_DIR / "processed_chunks.jsonl"
    bm25_file = BASE_DIR / "vector_index" / "bm25_index.pkl"

    # Integrity verification BEFORE test
    print("\n[PRE-CHECK] Computing baseline file hashes and Qdrant state...")
    pre_chunks_hash = file_sha256(chunks_file)
    pre_chunks_size = chunks_file.stat().st_size
    pre_bm25_hash = file_sha256(bm25_file)
    pre_bm25_size = bm25_file.stat().st_size

    retriever = QdrantHybridRetriever(
        bm25_path=bm25_file,
    )

    pre_points_count = retriever.qdrant_points_count
    print(f"  processed_chunks.jsonl SHA-256: {pre_chunks_hash} ({pre_chunks_size} bytes)")
    print(f"  bm25_index.pkl SHA-256:         {pre_bm25_hash} ({pre_bm25_size} bytes)")
    print(f"  Qdrant collection:             {retriever.collection_name}")
    print(f"  Qdrant points count:           {pre_points_count}")
    print(f"  Qdrant vector dimension:       {retriever.qdrant_vector_size}")
    print(f"  Qdrant distance metric:        {retriever.qdrant_distance}")
    print(f"  BM25 document count:           {retriever.bm25_doc_count}")

    # Warmup query
    print("\nWarming up neural models and caches...")
    _ = retriever.retrieve_hybrid("warmup query", top_k=5)

    benchmark_results = []

    print("\nExecuting 15 benchmark queries across BM25, BGE-M3/Qdrant, and Hybrid RRF...\n")

    for q_idx, query in enumerate(TEST_QUERIES):
        print(f"Query {q_idx + 1}/{len(TEST_QUERIES)}: '{query}'")
        res = retriever.retrieve_hybrid(query, top_k=10, rrf_k=60)

        bm25_hits = res["bm25_results"]
        bge_hits = res["bge_results"]
        hybrid_hits = res["hybrid_results"]
        latencies = res["latencies"]

        bm25_rel_count = evaluate_query_relevance(q_idx, query, bm25_hits)
        bge_rel_count = evaluate_query_relevance(q_idx, query, bge_hits)
        hybrid_rel_count = evaluate_query_relevance(q_idx, query, hybrid_hits)

        query_record = {
            "query_number": q_idx + 1,
            "query": query,
            "latencies": latencies,
            "bm25": {
                "top_10_relevant_count": bm25_rel_count,
                "top_3": [
                    {
                        "rank": h["rank"],
                        "chunk_id": h["chunk_id"],
                        "score": round(h["score"], 4),
                        "source_pdf": h["source_pdf"],
                        "page": h["page"],
                        "standard": h["standard"],
                        "category": h["category"],
                        "title": h["title"],
                        "snippet": h["text"][:160].replace("\n", " ").strip(),
                    }
                    for h in bm25_hits[:3]
                ],
            },
            "bge_m3": {
                "top_10_relevant_count": bge_rel_count,
                "top_3": [
                    {
                        "rank": h["rank"],
                        "chunk_id": h["chunk_id"],
                        "score": round(h["score"], 4),
                        "source_pdf": h["source_pdf"],
                        "page": h["page"],
                        "standard": h["standard"],
                        "category": h["category"],
                        "title": h["title"],
                        "snippet": h["text"][:160].replace("\n", " ").strip(),
                    }
                    for h in bge_hits[:3]
                ],
            },
            "hybrid": {
                "top_10_relevant_count": hybrid_rel_count,
                "top_3": [
                    {
                        "rank": h["rank"],
                        "chunk_id": h["chunk_id"],
                        "rrf_score": round(h["rrf_score"], 6),
                        "bm25_rank": h["bm25_rank"],
                        "bge_rank": h["bge_rank"],
                        "bm25_score": round(h["bm25_score"], 4) if h["bm25_score"] is not None else None,
                        "bge_score": round(h["bge_score"], 4) if h["bge_score"] is not None else None,
                        "source_pdf": h["source_pdf"],
                        "page": h["page"],
                        "standard": h["standard"],
                        "category": h["category"],
                        "title": h["title"],
                        "snippet": h["text"][:160].replace("\n", " ").strip(),
                    }
                    for h in hybrid_hits[:3]
                ],
            },
        }

        benchmark_results.append(query_record)

        print(f"  Latencies: BM25={latencies['bm25_ms']}ms | BGE-embed={latencies['bge_embed_ms']}ms | Qdrant={latencies['qdrant_search_ms']}ms | RRF={latencies['rrf_ms']}ms | Total={latencies['total_hybrid_ms']}ms")
        print(f"  Relevant (Top 10): BM25={bm25_rel_count} | BGE={bge_rel_count} | Hybrid={hybrid_rel_count}")
        if hybrid_hits:
            h1 = hybrid_hits[0]
            print(f"  Hybrid #1: [{h1.get('standard') or 'N/A'}] {h1.get('title') or 'N/A'} (RRF={h1.get('rrf_score'):.5f}, BM25_rk={h1.get('bm25_rank')}, BGE_rk={h1.get('bge_rank')})")
        print()

    # Close retriever
    retriever.close()

    # Integrity verification AFTER test
    print("\n[POST-CHECK] Verifying file hashes and Qdrant state integrity...")
    post_chunks_hash = file_sha256(chunks_file)
    post_chunks_size = chunks_file.stat().st_size
    post_bm25_hash = file_sha256(bm25_file)
    post_bm25_size = bm25_file.stat().st_size

    # Reconnect to Qdrant Cloud to verify point count
    from qdrant_client import QdrantClient

    verify_client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    post_col_info = verify_client.get_collection("bis_chunks_bge_m3_v1")
    post_points_count = post_col_info.points_count
    verify_client.close()

    print(f"  processed_chunks.jsonl SHA-256: {post_chunks_hash} ({post_chunks_size} bytes)")
    print(f"  bm25_index.pkl SHA-256:         {post_bm25_hash} ({post_bm25_size} bytes)")
    print(f"  Qdrant points count:           {post_points_count}")

    chunks_intact = pre_chunks_hash == post_chunks_hash and pre_chunks_size == post_chunks_size
    bm25_intact = pre_bm25_hash == post_bm25_hash and pre_bm25_size == post_bm25_size
    qdrant_intact = pre_points_count == post_points_count

    print(f"\nIntegrity Verification Results:")
    print(f"  processed_chunks.jsonl intact: {chunks_intact}")
    print(f"  bm25_index.pkl intact:         {bm25_intact}")
    print(f"  Qdrant point count intact:     {qdrant_intact}")

    if not (chunks_intact and bm25_intact and qdrant_intact):
        raise RuntimeError("DATA INTEGRITY VIOLATION DETECTED!")

    # Save benchmark results
    out_dir = BASE_DIR / "tests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "qdrant_hybrid_retrieval_results.json"

    full_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "qdrant_collection": "bis_chunks_bge_m3_v1",
            "qdrant_points": pre_points_count,
            "qdrant_vector_dim": retriever.qdrant_vector_size,
            "qdrant_distance": retriever.qdrant_distance,
            "bm25_documents": retriever.bm25_doc_count,
            "bge_model": "BAAI/bge-m3",
            "rrf_k": 60,
            "data_integrity": {
                "chunks_hash_match": chunks_intact,
                "bm25_hash_match": bm25_intact,
                "qdrant_points_match": qdrant_intact,
            },
        },
        "query_results": benchmark_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark complete! Full results saved to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
