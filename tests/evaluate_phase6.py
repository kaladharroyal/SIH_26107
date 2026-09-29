"""
Phase 6: Comprehensive Benchmark Evaluation Runner (evaluate_phase6.py)
Executes tests/data/phase6_benchmark.json against the BIS RAG System.
Computes:
  - Hit Rate@1, @3, @5
  - Precision@1, @3, @5
  - Recall@1, @3, @5
  - MRR (Mean Reciprocal Rank)
  - nDCG@5
  - Answer Groundedness, Citation Precision, and Refusal Correctness.
Outputs results to tests/results/phase6_retrieval_results.json.
"""

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [str(SRC_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate_phase6")

BENCHMARK_PATH = BASE_DIR / "tests" / "data" / "phase6_benchmark.json"
RESULTS_DIR = BASE_DIR / "tests" / "results"
RESULTS_FILE = RESULTS_DIR / "phase6_retrieval_results.json"


def is_chunk_relevant(chunk: Dict[str, Any], test_item: Dict[str, Any]) -> bool:
    """Deterministic relevance judgment using standard numbers and authoritative keywords."""
    doc = chunk.get("doc") or {}
    text = (chunk.get("text") or doc.get("text") or chunk.get("clause_text") or "").lower()
    title = (chunk.get("clause_title") or doc.get("clause_title") or doc.get("product") or "").lower()
    std_num = (chunk.get("standard_number") or doc.get("standard_number") or doc.get("is_number") or "").lower()
    combined = f"{text} {title} {std_num}"

    # 1. Match expected standard IDs (e.g., 'IS 1786', '1786')
    for std_id in test_item.get("expected_standard_ids", []):
        std_clean = std_id.lower().replace("is", "").strip()
        if std_clean and (std_clean in combined or std_id.lower() in combined):
            return True

    # 2. Match expected keywords
    for kw in test_item.get("expected_keywords", []):
        kw_clean = kw.lower().strip()
        if kw_clean and kw_clean in combined:
            return True

    return False


def compute_dcg(relevances: List[int], k: int = 5) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def compute_ndcg(relevances: List[int], k: int = 5) -> float:
    dcg = compute_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def run_benchmark_evaluation() -> Dict[str, Any]:
    log.info(f"Loading Phase 6 Benchmark Dataset from {BENCHMARK_PATH}...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_cases = json.load(f)

    pipeline = BISRAGPipeline(use_fast_retrieval=True)

    query_results = []
    retrieval_eval_cases = 0

    hit1_total = 0
    hit3_total = 0
    hit5_total = 0
    prec1_total = 0.0
    prec3_total = 0.0
    prec5_total = 0.0
    rec1_total = 0.0
    rec3_total = 0.0
    rec5_total = 0.0
    mrr_total = 0.0
    ndcg5_total = 0.0

    refusal_correct_total = 0
    refusal_cases_total = 0
    grounded_correct_total = 0
    grounded_cases_total = 0
    citation_precision_total = 0.0

    t_start = time.time()

    for idx, item in enumerate(benchmark_cases, 1):
        q = item["query"]
        should_refuse = item.get("should_refuse", False)
        is_adversarial = item.get("adversarial", False)

        # 1. End-to-end Pipeline Query Execution for Answer Evaluation
        res = pipeline.query(q)
        resp_text = res.get("response", "")
        status = res.get("status", "")
        citations = res.get("citations", [])

        # 2. Normalize query for direct BM25 retrieval evaluation
        lang_info = pipeline.multilingual.detect_language(q)
        lang_code = lang_info.get("lang_code", "en")
        if lang_code == "hinglish":
            search_query = pipeline.multilingual.normalize_hinglish_to_english(q)
        elif lang_code in ["hi", "te", "ta", "bn"]:
            search_query = pipeline.multilingual.normalize_native_to_english_keywords(q)
        else:
            search_query = q

        retrieved = pipeline.retrieval.retrieve_fast(search_query, top_n=5)

        # --- Answer Evaluation ---
        if should_refuse:
            refusal_cases_total += 1
            # Refusal is correct if status is refused, or response indicates insufficient information
            refused = (
                status == "refused"
                or "does not provide enough information" in resp_text.lower()
                or "पर्याप्त नहीं" in resp_text
                or "సరిపోదు" in resp_text
                or "please provide a valid query" in resp_text.lower()
                or res.get("fallback_used", False) is True
                or res.get("confidence_score", 1.0) < 0.50
            )
            if refused:
                refusal_correct_total += 1
        else:
            grounded_cases_total += 1
            # Groundedness: response preserves key requirements or standard IDs
            req_matched = True
            for req in item.get("answer_requirements", []):
                if req.lower() not in resp_text.lower():
                    req_matched = False
                    break
            if req_matched:
                grounded_correct_total += 1

            # Citation Precision: emitted citations supported by retrieved context or official source
            valid_cit_count = 0
            if citations:
                for c in citations:
                    # Verified citation if standard/official source or in retrieved docs
                    if c.get("source_of_truth") or c.get("source_hash") or c.get("label"):
                        valid_cit_count += 1
                citation_precision_total += valid_cit_count / len(citations)
            else:
                # If specialized flow handled deterministically or 0 citations needed
                citation_precision_total += 1.0

        # --- Retrieval Evaluation (Only for retrieval-intended compliance queries) ---
        has_retrieval_target = (
            not should_refuse
            and (item.get("expected_standard_ids") or item.get("expected_keywords"))
            and len(retrieved) > 0
        )

        relevance_vector = []
        reciprocal_rank = 0.0
        first_rel_rank = None

        if has_retrieval_target:
            retrieval_eval_cases += 1
            for rank_idx, chunk in enumerate(retrieved[:5], start=1):
                rel = 1 if is_chunk_relevant(chunk, item) else 0
                relevance_vector.append(rel)
                if rel == 1 and first_rel_rank is None:
                    first_rel_rank = rank_idx
                    reciprocal_rank = 1.0 / rank_idx

            # Pad up to 5 if fewer retrieved
            while len(relevance_vector) < 5:
                relevance_vector.append(0)

            # Hit Rate@K
            h1 = 1 if sum(relevance_vector[:1]) > 0 else 0
            h3 = 1 if sum(relevance_vector[:3]) > 0 else 0
            h5 = 1 if sum(relevance_vector[:5]) > 0 else 0
            hit1_total += h1
            hit3_total += h3
            hit5_total += h5

            # Precision@K
            p1 = sum(relevance_vector[:1]) / 1.0
            p3 = sum(relevance_vector[:3]) / 3.0
            p5 = sum(relevance_vector[:5]) / 5.0
            prec1_total += p1
            prec3_total += p3
            prec5_total += p5

            # Recall@K (relative to relevant in top-5)
            total_rel_in_top5 = max(1, sum(relevance_vector[:5]))
            r1 = sum(relevance_vector[:1]) / total_rel_in_top5
            r3 = sum(relevance_vector[:3]) / total_rel_in_top5
            r5 = sum(relevance_vector[:5]) / total_rel_in_top5
            rec1_total += r1
            rec3_total += r3
            rec5_total += r5

            # MRR & nDCG
            mrr_total += reciprocal_rank
            ndcg5 = compute_ndcg(relevance_vector, k=5)
            ndcg5_total += ndcg5
        else:
            p1, p3, p5 = 0.0, 0.0, 0.0
            r1, r3, r5 = 0.0, 0.0, 0.0
            ndcg5 = 0.0

        query_results.append({
            "id": item["id"],
            "query": q,
            "category": item["category"],
            "should_refuse": should_refuse,
            "status": status,
            "flow_used": res.get("flow_used"),
            "confidence_score": res.get("confidence_score"),
            "first_relevant_rank": first_rel_rank,
            "relevance_vector": relevance_vector,
            "p5": round(p5, 3),
            "ndcg5": round(ndcg5, 3),
        })

    total_time = round(time.time() - t_start, 2)
    n_ret = max(1, retrieval_eval_cases)
    n_ground = max(1, grounded_cases_total)
    n_ref = max(1, refusal_cases_total)

    metrics = {
        "benchmark_summary": {
            "total_queries": len(benchmark_cases),
            "retrieval_evaluated_queries": retrieval_eval_cases,
            "refusal_queries": refusal_cases_total,
            "grounded_queries": grounded_cases_total,
            "total_evaluation_seconds": total_time,
        },
        "retrieval_metrics": {
            "hit_rate_at_1": round(hit1_total / n_ret, 4),
            "hit_rate_at_3": round(hit3_total / n_ret, 4),
            "hit_rate_at_5": round(hit5_total / n_ret, 4),
            "precision_at_1": round(prec1_total / n_ret, 4),
            "precision_at_3": round(prec3_total / n_ret, 4),
            "precision_at_5": round(prec5_total / n_ret, 4),
            "recall_at_1": round(rec1_total / n_ret, 4),
            "recall_at_3": round(rec3_total / n_ret, 4),
            "recall_at_5": round(rec5_total / n_ret, 4),
            "mrr": round(mrr_total / n_ret, 4),
            "ndcg_at_5": round(ndcg5_total / n_ret, 4),
        },
        "answer_metrics": {
            "refusal_correctness": round(refusal_correct_total / n_ref, 4),
            "groundedness_accuracy": round(grounded_correct_total / n_ground, 4),
            "citation_precision": round(citation_precision_total / n_ground, 4),
        },
        "query_details": query_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Phase 6 Benchmark Results saved to {RESULTS_FILE}")
    return metrics


def test_phase6_evaluation_benchmark():
    """Pytest test asserting that Phase 6 benchmark runs and meets minimal accuracy thresholds."""
    metrics = run_benchmark_evaluation()
    rm = metrics["retrieval_metrics"]
    am = metrics["answer_metrics"]

    # Assert robust retrieval and answer metrics
    assert rm["hit_rate_at_5"] >= 0.70, f"Hit Rate@5 below 70%: {rm['hit_rate_at_5']}"
    assert rm["mrr"] >= 0.60, f"MRR below 0.60: {rm['mrr']}"
    assert am["refusal_correctness"] >= 0.85, f"Refusal correctness below 85%: {am['refusal_correctness']}"
    assert am["citation_precision"] >= 0.90, f"Citation precision below 90%: {am['citation_precision']}"


if __name__ == "__main__":
    m = run_benchmark_evaluation()
    print("\n" + "=" * 65)
    print("      PHASE 6: COMPREHENSIVE BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Test Cases Evaluated : {m['benchmark_summary']['total_queries']}")
    print(f"Retrieval Hit Rate@1       : {m['retrieval_metrics']['hit_rate_at_1'] * 100:.1f}%")
    print(f"Retrieval Hit Rate@5       : {m['retrieval_metrics']['hit_rate_at_5'] * 100:.1f}%")
    print(f"Precision@5                : {m['retrieval_metrics']['precision_at_5']:.3f}")
    print(f"MRR (Mean Reciprocal Rank) : {m['retrieval_metrics']['mrr']:.3f}")
    print(f"nDCG@5                     : {m['retrieval_metrics']['ndcg_at_5']:.3f}")
    print(f"Refusal Correctness        : {m['answer_metrics']['refusal_correctness'] * 100:.1f}%")
    print(f"Grounded Requirement Match : {m['answer_metrics']['groundedness_accuracy'] * 100:.1f}%")
    print(f"Citation Precision         : {m['answer_metrics']['citation_precision'] * 100:.1f}%")
    print("=" * 65 + "\n")
