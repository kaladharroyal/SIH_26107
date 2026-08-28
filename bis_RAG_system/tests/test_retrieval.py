import logging
from retrieval import HybridRetrievalPipeline

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Hybrid Retrieval Pipeline...")
    pipeline = HybridRetrievalPipeline()
    print(f"Loaded {len(pipeline.chunks)} chunks into memory.")
    
    query = "gold jewellery hallmarking regulations"
    results = pipeline.retrieve(query, top_n=3)
    
    print("\n--- RETRIEVAL RESULTS ---")
    for idx, r in enumerate(results, 1):
        doc = r["doc"]
        score = r.get("rerank_score", 0.0)
        title = doc.get("clause_title") or doc.get("source_file")
        print(f"{idx}. [{title}] (Score: {score:.4f})")
        print(f"   Text: {doc.get('text', '')[:100]}...\n")

if __name__ == "__main__":
    main()
