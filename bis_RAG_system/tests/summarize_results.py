import json
import sys

# Ensure UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("tests/results/qdrant_hybrid_retrieval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total queries: {len(data['query_results'])}\n")

for q in data["query_results"]:
    num = q["query_number"]
    txt = q["query"]
    lats = q["latencies"]
    print(f"============================================================")
    print(f"QUERY {num}: {txt}")
    print(f"Latencies: BM25={lats['bm25_ms']}ms | BGE-embed={lats['bge_embed_ms']}ms | Qdrant={lats['qdrant_search_ms']}ms | RRF={lats['rrf_ms']}ms | Total={lats['total_hybrid_ms']}ms")
    print(f"------------------------------------------------------------")
    print(f"BM25 (Top-10 Relevant Count: {q['bm25']['top_10_relevant_count']}):")
    for h in q["bm25"]["top_3"]:
        print(f"  Rank {h['rank']} | Chunk: {h['chunk_id'][:12]}... | Score: {h['score']} | Standard: {h['standard']} | Title: {h['title'][:50]}")
        print(f"    Source: {h['source_pdf']} | Page: {h['page']} | Category: {h['category']}")
        print(f"    Snippet: {h['snippet'][:120]}...")
    print(f"BGE-M3 / Qdrant (Top-10 Relevant Count: {q['bge_m3']['top_10_relevant_count']}):")
    for h in q["bge_m3"]["top_3"]:
        print(f"  Rank {h['rank']} | Chunk: {h['chunk_id'][:12]}... | Score: {h['score']} | Standard: {h['standard']} | Title: {h['title'][:50]}")
        print(f"    Source: {h['source_pdf']} | Page: {h['page']} | Category: {h['category']}")
        print(f"    Snippet: {h['snippet'][:120]}...")
    print(f"HYBRID / RRF (Top-10 Relevant Count: {q['hybrid']['top_10_relevant_count']}):")
    for h in q["hybrid"]["top_3"]:
        print(f"  Rank {h['rank']} | Chunk: {h['chunk_id'][:12]}... | RRF: {h['rrf_score']} (BM25_rk: {h['bm25_rank']}, BGE_rk: {h['bge_rank']})")
        print(f"    Standard: {h['standard']} | Title: {h['title'][:50]} | Source: {h['source_pdf']}")
        print(f"    Snippet: {h['snippet'][:120]}...")
    print()
