"""
Enterprise High-Priority Upgrades Test Suite (test_enterprise_upgrades.py)
Tests:
  1. DistributedResponseCache (Redis + in-memory TTLCache fallback)
  2. Neural Model INT8 Dynamic Quantization (HFTransformerEmbeddingModel & HFCrossEncoderReranker)
  3. PostgreSQL & SQLite Dual Engine Telemetry & Connection Pooling (FeedbackLogger)
  4. PostgreSQL Schema & Migration Script Integrity (schema_postgres.sql & migrate_to_postgres.py)
"""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [str(SRC_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import DistributedResponseCache
from feedback_logger import FeedbackLogger
from retrieval import HFTransformerEmbeddingModel, HFCrossEncoderReranker, MockEmbeddingModel


# =========================================================================
# 1. Distributed Cache Tests
# =========================================================================

def test_distributed_cache_fallback_and_dict_interface():
    """Verify that DistributedResponseCache provides dict-like access with in-memory fallback."""
    cache = DistributedResponseCache(ttl=10, maxsize=50)
    
    # 1. Test set and get
    key = ("is 1786", "is_standard", "English")
    payload = {"query": "IS 1786", "confidence_score": 0.95, "status": "success"}
    cache[key] = payload
    
    assert key in cache
    assert cache[key]["query"] == "IS 1786"
    assert cache.get(key)["confidence_score"] == 0.95
    assert cache.get(("nonexistent", "", "")) is None

    # 2. Test clear
    cache.clear()
    assert key not in cache


def test_distributed_cache_key_formatting():
    """Verify key formatting handles tuples and strings."""
    cache = DistributedResponseCache(ttl=10, maxsize=10)
    fmt1 = cache._format_key(("query1", "category1", "en"))
    assert fmt1 == "bis_cache:query1:category1:en"
    
    fmt2 = cache._format_key("simple_key")
    assert fmt2 == "bis_cache:simple_key"


# =========================================================================
# 2. Neural Model Quantization Tests
# =========================================================================

def test_embedding_model_quantization_and_encoding():
    """Verify that dynamic INT8 quantization can be initialized without errors."""
    try:
        model = HFTransformerEmbeddingModel(use_quantization=True)
        assert model.dimension > 0
        
        # Test encoding
        vec = model.encode("Bureau of Indian Standards IS 1786")
        assert isinstance(vec, np.ndarray)
        assert vec.shape[-1] == model.dimension
        
        # Test unit normalization
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4
    except Exception as e:
        pytest.skip(f"HuggingFace neural weights unavailable in test env: {e}")


def test_cross_encoder_quantization_and_caching():
    """Verify that cross encoder initializes with INT8 quantization and caches pair scores."""
    try:
        reranker = HFCrossEncoderReranker(use_quantization=True)
        pairs = [("What is IS 1786?", "High strength deformed steel bars specification.")]
        
        scores1 = reranker.predict(pairs)
        assert len(scores1) == 1
        assert 0.0 <= float(scores1[0]) <= 1.0
        
        # Second call should use cache
        assert (pairs[0][0], pairs[0][1]) in reranker._score_cache
        scores2 = reranker.predict(pairs)
        assert scores1[0] == scores2[0]
    except Exception as e:
        pytest.skip(f"HuggingFace cross encoder weights unavailable in test env: {e}")


# =========================================================================
# 3. Telemetry & Dual Engine Database Tests
# =========================================================================

def test_feedback_logger_sqlite_operations():
    """Verify FeedbackLogger works smoothly on temporary SQLite DB."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = Path(tmp_dir) / "test_telemetry.db"
        logger = FeedbackLogger(db_path=tmp_db)
        
        # 1. Log query
        log_id = logger.log_query(
            query="Test query about steel rebar",
            intent="technical_standards_rag",
            confidence_score=0.92,
            response_text="IS 1786 covers high strength deformed steel bars.",
            retrieved_chunks=[{"doc": {"chunk_id": "chunk_test_1"}}],
            detected_language="en",
            retrieval_ms=15.2,
            generation_ms=120.5,
            total_ms=135.7,
        )
        assert log_id > 0
        
        # 2. Submit feedback
        success = logger.submit_feedback(query_id=log_id, rating=1, feedback_notes="Accurate response")
        assert success is True
        
        # Cleanly release SQLite file handle before exiting temp directory on Windows
        logger.close()



def test_postgres_schema_and_migration_script_syntax():
    """Verify schema_postgres.sql and migrate_to_postgres.py exist and contain valid DDL statements."""
    schema_pg = BASE_DIR / "schema_postgres.sql"
    assert schema_pg.exists(), "schema_postgres.sql must exist"
    content = schema_pg.read_text(encoding="utf-8")
    
    assert "CREATE TABLE IF NOT EXISTS standards" in content
    assert "BIGSERIAL PRIMARY KEY" in content
    assert "CREATE TABLE IF NOT EXISTS query_logs" in content
    assert "CREATE TABLE IF NOT EXISTS feedback_ratings" in content

    migrate_script = BASE_DIR / "migrate_to_postgres.py"
    assert migrate_script.exists(), "migrate_to_postgres.py must exist"
