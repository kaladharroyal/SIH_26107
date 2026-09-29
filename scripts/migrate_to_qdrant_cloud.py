"""
Migrate existing local Qdrant collection to Qdrant Cloud.

Data migration only:
- Copies point IDs, vectors, and payloads directly from local storage to Qdrant Cloud.
- Does NOT recompute or regenerate embeddings.
- Does NOT alter or delete local Qdrant database.
- Does NOT expose API keys in code or logs.
- Fully resumable and idempotent.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Project base paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ENV_PATH = PROJECT_DIR / ".env"
LOCAL_STORAGE_PATH = Path(r"V:\PROJECTS\SIH_26107\qdrant_storage")
MIGRATION_DIR = PROJECT_DIR / "vector_index" / "qdrant_migration"
CHECKPOINT_PATH = MIGRATION_DIR / "cloud_migration_checkpoint.json"
LOG_PATH = MIGRATION_DIR / "cloud_migration.log"

COLLECTION_NAME = "bis_chunks_bge_m3_v1"
EXPECTED_POINT_COUNT = 34512
VECTOR_SIZE = 1024
VECTOR_DISTANCE = models.Distance.COSINE
BATCH_SIZE = 250
MAX_RETRIES = 5


def setup_logger() -> logging.Logger:
    MIGRATION_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cloud_migration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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

    # Never log the API key. Only log the endpoint URL.
    logger.info(f"Target Qdrant Cloud URL: {qdrant_url}")
    logger.info("Qdrant Cloud API key is configured (length: %d chars).", len(qdrant_api_key))

    return qdrant_url, qdrant_api_key


def load_checkpoint() -> Optional[Dict[str, Any]]:
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_checkpoint(data: Dict[str, Any]):
    temp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_path.replace(CHECKPOINT_PATH)


def migrate():
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("STARTING QDRANT CLOUD MIGRATION")
    logger.info("=" * 60)

    qdrant_url, qdrant_api_key = load_environment(logger)

    # 1. Connect and verify local Qdrant
    logger.info(f"Connecting to local Qdrant storage at {LOCAL_STORAGE_PATH}...")
    local_client = QdrantClient(path=str(LOCAL_STORAGE_PATH))

    local_collections = [c.name for c in local_client.get_collections().collections]
    if COLLECTION_NAME not in local_collections:
        logger.error(f"Local collection '{COLLECTION_NAME}' not found in {local_collections}")
        local_client.close()
        raise RuntimeError(f"Local collection '{COLLECTION_NAME}' not found.")

    local_info = local_client.get_collection(COLLECTION_NAME)
    logger.info(f"Local collection verified: '{COLLECTION_NAME}'")
    logger.info(f"  - Points count: {local_info.points_count}")
    logger.info(f"  - Vectors config: {local_info.config.params.vectors}")

    if local_info.points_count != EXPECTED_POINT_COUNT:
        logger.error(f"Local point count mismatch: expected {EXPECTED_POINT_COUNT}, found {local_info.points_count}")
        local_client.close()
        raise RuntimeError(f"Expected {EXPECTED_POINT_COUNT} points locally, found {local_info.points_count}")

    # Verify vector size and distance
    vectors_config = local_info.config.params.vectors
    actual_size = getattr(vectors_config, "size", None)
    actual_distance = getattr(vectors_config, "distance", None)
    if actual_size != VECTOR_SIZE:
        local_client.close()
        raise RuntimeError(f"Expected vector size {VECTOR_SIZE}, found {actual_size}")
    if actual_distance != VECTOR_DISTANCE:
        local_client.close()
        raise RuntimeError(f"Expected distance {VECTOR_DISTANCE}, found {actual_distance}")

    # 2. Connect and verify/create Cloud collection
    logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}...")
    cloud_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    cloud_collections = [c.name for c in cloud_client.get_collections().collections]
    if COLLECTION_NAME not in cloud_collections:
        logger.info(f"Collection '{COLLECTION_NAME}' does not exist on Cloud. Creating...")
        cloud_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=VECTOR_DISTANCE
            )
        )
        logger.info(f"Collection '{COLLECTION_NAME}' successfully created on Cloud with size={VECTOR_SIZE}, distance=COSINE.")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists on Cloud. Verifying compatibility...")
        cloud_info = cloud_client.get_collection(COLLECTION_NAME)
        c_vectors = cloud_info.config.params.vectors
        c_size = getattr(c_vectors, "size", None)
        c_dist = getattr(c_vectors, "distance", None)
        logger.info(f"Cloud collection configuration: size={c_size}, distance={c_dist}, points_count={cloud_info.points_count}")
        if c_size != VECTOR_SIZE or c_dist != VECTOR_DISTANCE:
            local_client.close()
            cloud_client.close()
            raise RuntimeError(f"Cloud collection incompatible: size={c_size}, distance={c_dist}")

    # 3. Check for existing checkpoint
    checkpoint = load_checkpoint()
    last_offset = None
    copied_count = 0
    skipped_count = 0
    total_retries = 0
    total_failures = 0
    batch_idx = 0

    if checkpoint and checkpoint.get("collection_name") == COLLECTION_NAME:
        if checkpoint.get("status") == "completed":
            cloud_info = cloud_client.get_collection(COLLECTION_NAME)
            if cloud_info.points_count == EXPECTED_POINT_COUNT:
                logger.info(f"Migration already recorded as COMPLETED in checkpoint. Cloud points: {cloud_info.points_count}")
                local_client.close()
                cloud_client.close()
                return

        logger.info("Resuming migration from checkpoint...")
        last_offset = checkpoint.get("current_offset")
        copied_count = checkpoint.get("points_copied", 0)
        skipped_count = checkpoint.get("points_skipped_existing", 0)
        total_retries = checkpoint.get("retries_count", 0)
        total_failures = checkpoint.get("failures_count", 0)
        batch_idx = checkpoint.get("batches_completed", 0)
        logger.info(f"Resuming with offset={last_offset}, batches_completed={batch_idx}, points_copied={copied_count}")

    # 4. Scroll local collection and upload to Cloud
    start_time = time.time()
    current_offset = last_offset

    try:
        while True:
            records, next_offset = local_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=BATCH_SIZE,
                offset=current_offset,
                with_payload=True,
                with_vectors=True
            )

            if not records:
                logger.info("No more records returned from local scroll. Migration loop finished.")
                break

            batch_idx += 1
            batch_point_ids = [r.id for r in records]

            # Check existing point IDs in Cloud to avoid duplicates and ensure idempotency
            existing_cloud_ids = set()
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    cloud_retrieved = cloud_client.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=batch_point_ids,
                        with_vectors=False,
                        with_payload=False
                    )
                    existing_cloud_ids = {p.id for p in cloud_retrieved}
                    break
                except Exception as e:
                    total_retries += 1
                    logger.warning(f"Batch {batch_idx}: Cloud retrieve check attempt {attempt} failed: {e}. Retrying in {2 ** attempt}s...")
                    if attempt == MAX_RETRIES:
                        total_failures += 1
                        logger.error(f"Batch {batch_idx}: Failed to check existing points on Cloud after {MAX_RETRIES} attempts.")
                        raise
                    time.sleep(2 ** attempt)

            points_to_upload = [
                models.PointStruct(id=r.id, vector=r.vector, payload=r.payload)
                for r in records if r.id not in existing_cloud_ids
            ]

            skipped_in_batch = len(records) - len(points_to_upload)
            skipped_count += skipped_in_batch

            if points_to_upload:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        cloud_client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=points_to_upload,
                            wait=True
                        )
                        copied_count += len(points_to_upload)
                        break
                    except Exception as e:
                        total_retries += 1
                        logger.warning(f"Batch {batch_idx}: Cloud upsert attempt {attempt} failed: {e}. Retrying in {2 ** attempt}s...")
                        if attempt == MAX_RETRIES:
                            total_failures += 1
                            logger.error(f"Batch {batch_idx}: Failed to upsert points to Cloud after {MAX_RETRIES} attempts.")
                            raise
                        time.sleep(2 ** attempt)

            elapsed = time.time() - start_time
            rate = (copied_count + skipped_count) / elapsed if elapsed > 0 else 0
            if batch_idx % 10 == 0 or not next_offset or len(records) < BATCH_SIZE:
                logger.info(
                    f"Batch {batch_idx:4d} | Processed: {copied_count + skipped_count}/{EXPECTED_POINT_COUNT} "
                    f"(Copied: {copied_count}, Skipped: {skipped_count}) | Speed: {rate:.1f} pts/sec"
                )

            # Update checkpoint after each batch
            current_offset = next_offset
            save_checkpoint({
                "status": "in_progress",
                "collection_name": COLLECTION_NAME,
                "local_source_points": EXPECTED_POINT_COUNT,
                "current_offset": current_offset,
                "batches_completed": batch_idx,
                "points_copied": copied_count,
                "points_skipped_existing": skipped_count,
                "retries_count": total_retries,
                "failures_count": total_failures,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            if next_offset is None:
                logger.info("Reached end of local collection (next_offset is None).")
                break

    finally:
        local_client.close()

    # 5. Final validation and checkpoint update
    cloud_info = cloud_client.get_collection(COLLECTION_NAME)
    final_cloud_count = cloud_info.points_count
    logger.info("=" * 60)
    logger.info(f"MIGRATION COMPLETED")
    logger.info(f"  - Local Point Count : {EXPECTED_POINT_COUNT}")
    logger.info(f"  - Cloud Point Count : {final_cloud_count}")
    logger.info(f"  - Points Copied     : {copied_count}")
    logger.info(f"  - Points Skipped    : {skipped_count}")
    logger.info(f"  - Total Retries     : {total_retries}")
    logger.info(f"  - Total Failures    : {total_failures}")
    logger.info("=" * 60)

    save_checkpoint({
        "status": "completed" if final_cloud_count == EXPECTED_POINT_COUNT else "incomplete",
        "collection_name": COLLECTION_NAME,
        "local_source_points": EXPECTED_POINT_COUNT,
        "cloud_target_points": final_cloud_count,
        "points_copied": copied_count,
        "points_skipped_existing": skipped_count,
        "batches_completed": batch_idx,
        "current_offset": None,
        "retries_count": total_retries,
        "failures_count": total_failures,
        "end_time": datetime.now(timezone.utc).isoformat()
    })

    cloud_client.close()


if __name__ == "__main__":
    migrate()
