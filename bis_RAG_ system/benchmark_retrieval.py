"""
Phase 2, Step 8: Standalone Retrieval Benchmark & Test Suite
Evaluates Top-1 and Top-5 retrieval precision on realistic BIS queries
(Product certification, Hallmarking, Steel standards, Consumer FAQs).
"""

import logging
from retrieval import HybridRetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark_retrieval")

TEST_GROUND_TRUTH = [
    {
        "query": "what standard applies to gold jewellery hallmarking",
        "expected_keyword": "hallmarking",
        "category": "hallmarking",
    },
    {
        "query": "IS 1786 steel reinforcement requirements",
        "expected_keyword": "1786",
        "category": "product_standard_mapping",
    },
    {
        "query": "how to register as a jeweller under BIS scheme",
        "expected_keyword": "jeweller",
        "category": "hallmarking",
    },
    {
        "query": "fees for product certification license",
        "expected_keyword": "fee",
        "category": "licensing_fees",
    },
    {
        "query": "compulsory registration scheme for electronic products CRS",
        "expected_keyword": "compulsory",
        "category": "certification_process",
    },
]


def run_benchmark():
    log.info("Starting Phase 2 Standalone Retrieval Benchmark...")
    pipeline = HybridRetrievalPipeline()
    
    top1_hits = 0
    top5_hits = 0
    total_queries = len(TEST_GROUND_TRUTH)

    for test_case in TEST_GROUND_TRUTH:
        q = test_case["query"]
        expected = test_case["expected_keyword"].lower()
        
        results = pipeline.retrieve(q, top_n=5)
        
        top1_match = False
        top5_match = False
        
        for rank, res in enumerate(results, 1):
            text = res["doc"].get("text", "").lower()
            title = res["doc"].get("clause_title", "").lower()
            
            if expected in text or expected in title:
                top5_match = True
                if rank == 1:
                    top1_match = True
                break

        if top1_match:
            top1_hits += 1
            log.info(f"PASS Top-1: '{q}' -> Top hit matched '{expected}'")
        elif top5_match:
            top5_hits += 1
            log.info(f"PASS Top-5: '{q}' -> Found '{expected}' within top-5")
        else:
            log.warning(f"FAIL: '{q}' -> '{expected}' not found in top-5 results")

    top1_acc = (top1_hits / total_queries) * 100
    top5_acc = ((top1_hits + top5_hits) / total_queries) * 100

    print("\n" + "=" * 60)
    print("      PHASE 2: HYBRID RETRIEVAL BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Test Queries      : {total_queries}")
    print(f"Top-1 Precision Accuracy: {top1_acc:.2f}% ({top1_hits}/{total_queries})")
    print(f"Top-5 Precision Accuracy: {top5_acc:.2f}% ({(top1_hits + top5_hits)}/{total_queries})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
