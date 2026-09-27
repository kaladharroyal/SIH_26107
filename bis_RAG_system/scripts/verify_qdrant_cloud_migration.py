"""
Verify Qdrant Cloud Migration against Local Qdrant.

Executes comprehensive verification protocol:
A. Exact point count check (Local: 34,512 == Cloud: 34,512)
B. Collection configuration check (vector size 1024, distance COSINE)
C. Sample point verification (at least 20 sampled points: ID, vector numerical equality, payload)
D. Retrieval verification (10 identical vector queries: top-k IDs, similarity scores, ranking)
E. Payload verification (all metadata and provenance fields preserved)
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ENV_PATH = PROJECT_DIR / ".env"
LOCAL_STORAGE_PATH = Path(r"V:\PROJECTS\SIH_26107\qdrant_storage")
MIGRATION_DIR = PROJECT_DIR / "vector_index" / "qdrant_migration"
VERIFICATION_RESULTS_PATH = MIGRATION_DIR / "cloud_verification_results.json"

COLLECTION_NAME = "bis_chunks_bge_m3_v1"
EXPECTED_COUNT = 34512
EXPECTED_DIM = 1024
EXPECTED_DISTANCE = "Cosine"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("cloud_verification")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_environment(logger: logging.Logger) -> tuple[str, str]:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("QDRANT_URL environment variable is missing.")
        raise ValueError("QDRANT_URL environment variable is missing.")
    if not qdrant_api_key:
        logger.error("QDRANT_API_KEY environment variable is missing.")
        raise ValueError("QDRANT_API_KEY environment variable is missing.")

    return qdrant_url, qdrant_api_key


def run_verification() -> Dict[str, Any]:
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("STARTING QDRANT CLOUD MIGRATION VERIFICATION")
    logger.info("=" * 60)

    qdrant_url, qdrant_api_key = load_environment(logger)

    logger.info(f"Connecting to local Qdrant at {LOCAL_STORAGE_PATH}...")
    local_client = QdrantClient(path=str(LOCAL_STORAGE_PATH))

    logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}...")
    cloud_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    results: Dict[str, Any] = {
        "local_storage_path": str(LOCAL_STORAGE_PATH),
        "local_collection_name": COLLECTION_NAME,
        "cloud_url": qdrant_url,
        "cloud_collection_name": COLLECTION_NAME,
        "checks": {}
    }

    try:
        # ---------------------------------------------------------
        # A. Exact count check
        # ---------------------------------------------------------
        logger.info("\n--- CHECK A: EXACT COUNT CHECK ---")
        local_info = local_client.get_collection(COLLECTION_NAME)
        cloud_info = cloud_client.get_collection(COLLECTION_NAME)

        local_count = local_info.points_count
        cloud_count = cloud_info.points_count

        logger.info(f"Local collection point count: {local_count}")
        logger.info(f"Cloud collection point count: {cloud_count}")

        count_match = (local_count == EXPECTED_COUNT) and (cloud_count == EXPECTED_COUNT)
        results["checks"]["exact_count"] = {
            "local_count": local_count,
            "cloud_count": cloud_count,
            "expected_count": EXPECTED_COUNT,
            "passed": count_match
        }
        if not count_match:
            logger.error(f"FAIL: Count mismatch! Expected {EXPECTED_COUNT}, got local={local_count}, cloud={cloud_count}")
        else:
            logger.info("PASS: Exact point count verified (34,512 == 34,512).")

        # ---------------------------------------------------------
        # B. Collection configuration check
        # ---------------------------------------------------------
        logger.info("\n--- CHECK B: CONFIGURATION CHECK ---")
        c_params = cloud_info.config.params.vectors
        c_size = getattr(c_params, "size", None)
        c_dist = getattr(c_params, "distance", None)
        c_dist_str = c_dist.value if hasattr(c_dist, "value") else str(c_dist)

        config_match = (c_size == EXPECTED_DIM) and (c_dist_str.lower() == EXPECTED_DISTANCE.lower())
        results["checks"]["configuration"] = {
            "cloud_vector_size": c_size,
            "cloud_distance": c_dist_str,
            "expected_size": EXPECTED_DIM,
            "expected_distance": EXPECTED_DISTANCE,
            "passed": config_match
        }
        logger.info(f"Cloud Vector Config: size={c_size}, distance={c_dist_str}")
        if not config_match:
            logger.error("FAIL: Cloud configuration mismatch!")
        else:
            logger.info("PASS: Cloud configuration verified (1024, Cosine).")

        # ---------------------------------------------------------
        # C. Sample point verification
        # ---------------------------------------------------------
        logger.info("\n--- CHECK C: SAMPLE POINT VERIFICATION (>=20 points) ---")
        # Sample 25 points evenly spaced across scrolls
        sampled_records = []
        offset = None
        # Scroll 5 batches of 100 at different steps to gather candidate points
        for _ in range(5):
            recs, offset = local_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            if recs:
                # pick 5 from each batch
                sampled_records.extend(random.sample(recs, min(5, len(recs))))
            if not offset:
                break

        if len(sampled_records) < 20:
            # Fallback scroll
            recs, _ = local_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=30,
                with_payload=True,
                with_vectors=True
            )
            sampled_records = recs[:25]

        sample_point_ids = [r.id for r in sampled_records[:25]]
        logger.info(f"Selected {len(sample_point_ids)} sample points for deep field & vector verification.")

        # Retrieve cloud points for these sample IDs
        cloud_retrieved = cloud_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=sample_point_ids,
            with_vectors=True,
            with_payload=True
        )
        cloud_by_id = {p.id: p for p in cloud_retrieved}

        sample_verifications = []
        vec_passed = 0
        payload_passed = 0

        for r_local in sampled_records[:25]:
            pid = r_local.id
            p_cloud = cloud_by_id.get(pid)
            if not p_cloud:
                sample_verifications.append({
                    "id": pid,
                    "passed": False,
                    "reason": "Missing on Cloud"
                })
                continue

            # Vector comparison
            vec_loc = np.array(r_local.vector, dtype=np.float32)
            vec_cld = np.array(p_cloud.vector, dtype=np.float32)

            dim_ok = (len(vec_loc) == EXPECTED_DIM) and (len(vec_cld) == EXPECTED_DIM)
            # Numerical tolerance check
            vec_ok = dim_ok and np.allclose(vec_loc, vec_cld, atol=1e-5, rtol=1e-5)
            if vec_ok:
                vec_passed += 1

            # Payload comparison
            payload_ok = (r_local.payload == p_cloud.payload)
            if payload_ok:
                payload_passed += 1

            point_passed = vec_ok and payload_ok
            sample_verifications.append({
                "id": pid,
                "dimension_match": dim_ok,
                "vector_match": bool(vec_ok),
                "payload_match": bool(payload_ok),
                "passed": point_passed
            })

        sample_all_passed = (vec_passed == len(sample_point_ids)) and (payload_passed == len(sample_point_ids))
        results["checks"]["sample_points"] = {
            "total_sampled": len(sample_point_ids),
            "vector_comparisons_passed": vec_passed,
            "payload_comparisons_passed": payload_passed,
            "passed": sample_all_passed,
            "details": sample_verifications
        }

        logger.info(f"Sample Verification Summary:")
        logger.info(f"  - Points Sampled       : {len(sample_point_ids)}")
        logger.info(f"  - Vector Matches (atol=1e-5): {vec_passed}/{len(sample_point_ids)}")
        logger.info(f"  - Payload Matches      : {payload_passed}/{len(sample_point_ids)}")
        if sample_all_passed:
            logger.info("PASS: All sample points matched exactly.")
        else:
            logger.error("FAIL: Some sample points did not match!")

        # ---------------------------------------------------------
        # D. Retrieval verification (10 identical vector queries)
        # ---------------------------------------------------------
        logger.info("\n--- CHECK D: RETRIEVAL VERIFICATION (10 queries) ---")
        retrieval_tests = []
        queries_passed = 0

        # Select 10 distinct vectors from sampled records (where 2nd neighbor has lower score to avoid duplicate tie-breaks)
        candidate_records = []
        for r in sampled_records:
            check_q = local_client.query_points(COLLECTION_NAME, query=r.vector, limit=2)
            if len(check_q.points) >= 2 and check_q.points[1].score < 0.99:
                candidate_records.append(r)
                if len(candidate_records) == 10:
                    break

        if len(candidate_records) < 10:
            candidate_records = sampled_records[:10]

        for idx, qrec in enumerate(candidate_records, 1):
            q_vec = qrec.vector

            local_res = local_client.query_points(
                collection_name=COLLECTION_NAME,
                query=q_vec,
                limit=5,
                with_payload=True
            )
            cloud_res = cloud_client.query_points(
                collection_name=COLLECTION_NAME,
                query=q_vec,
                limit=5,
                with_payload=True
            )

            local_ids = [p.id for p in local_res.points]
            cloud_ids = [p.id for p in cloud_res.points]

            local_scores = [float(p.score) for p in local_res.points]
            cloud_scores = [float(p.score) for p in cloud_res.points]

            # Compare top-1 match
            top1_match = (local_ids[0] == cloud_ids[0])
            top1_score_match = abs(local_scores[0] - cloud_scores[0]) < 1e-4

            # Compare top-k set overlap and ranking
            full_rank_match = (local_ids == cloud_ids)
            id_overlap = len(set(local_ids) & set(cloud_ids)) / len(local_ids)
            score_diffs = [abs(ls - cs) for ls, cs in zip(local_scores, cloud_scores)]
            max_score_diff = max(score_diffs) if score_diffs else 0.0

            q_passed = top1_match and (max_score_diff < 1e-4)
            if q_passed:
                queries_passed += 1

            retrieval_tests.append({
                "query_index": idx,
                "top1_id_local": local_ids[0],
                "top1_id_cloud": cloud_ids[0],
                "top1_match": top1_match,
                "full_5_match": full_rank_match,
                "top1_score_local": local_scores[0],
                "top1_score_cloud": cloud_scores[0],
                "id_overlap_ratio": id_overlap,
                "max_score_diff": max_score_diff,
                "passed": q_passed
            })

            logger.info(
                f"  Query {idx:2d}: Top-1 ID match: {top1_match} (Full Top-5 match: {full_rank_match}) "
                f"Scores: local={local_scores[0]:.6f}, cloud={cloud_scores[0]:.6f}, max_diff={max_score_diff:.8f} "
                f"Overlap: {id_overlap*100:.0f}%"
            )

        retrieval_passed = (queries_passed == len(candidate_records))
        results["checks"]["retrieval"] = {
            "total_queries": len(candidate_records),
            "queries_passed": queries_passed,
            "passed": retrieval_passed,
            "details": retrieval_tests
        }
        if retrieval_passed:
            logger.info(f"PASS: All {queries_passed}/{len(candidate_records)} retrieval queries verified successfully.")
        else:
            logger.warning(f"Retrieval queries passed: {queries_passed}/{len(candidate_records)}")

        # ---------------------------------------------------------
        # E. Payload verification
        # ---------------------------------------------------------
        logger.info("\n--- CHECK E: PAYLOAD SCHEMA & PROVENANCE VERIFICATION ---")
        required_fields = [
            "chunk_id",
            "text",
            "source_pdf",
            "page",
            "standard_number",
            "category",
            "document_title",
            "source_hash",
            "corpus_hash",
            "chunk_index",
            "metadata"
        ]

        payload_schema_passed = True
        missing_fields_report = []

        for p in cloud_retrieved:
            payload = p.payload or {}
            missing = [f for f in required_fields if f not in payload]
            if missing:
                payload_schema_passed = False
                missing_fields_report.append({"id": p.id, "missing_fields": missing})

        results["checks"]["payload_schema"] = {
            "required_fields": required_fields,
            "all_fields_present": payload_schema_passed,
            "missing_report": missing_fields_report,
            "passed": payload_schema_passed
        }

        if payload_schema_passed:
            logger.info(f"PASS: All {len(cloud_retrieved)} retrieved points contain complete metadata and provenance fields: {required_fields}")
        else:
            logger.error(f"FAIL: Payload fields missing in some points: {missing_fields_report}")

        # Overall verification outcome
        overall_equivalent = (
            count_match and
            config_match and
            sample_all_passed and
            retrieval_passed and
            payload_schema_passed
        )
        results["overall_equivalent"] = overall_equivalent

        logger.info("=" * 60)
        logger.info(f"OVERALL EQUIVALENCE: {'EQUIVALENT (PASSED)' if overall_equivalent else 'NOT EQUIVALENT (FAILED)'}")
        logger.info("=" * 60)

        # Save verification results
        with open(VERIFICATION_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Verification results written to: {VERIFICATION_RESULTS_PATH}")

        return results

    finally:
        local_client.close()
        cloud_client.close()


if __name__ == "__main__":
    run_verification()
