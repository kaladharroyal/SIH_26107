"""
Phase 6: Feedback & Interaction Logger Test Suite (test_feedback.py)
Tests feedback acceptance, persistence, malformed input rejection, and safe error handling.
"""

import sys
from pathlib import Path
import tempfile
import sqlite3
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in [str(SRC_DIR), str(BASE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.feedback_logger import FeedbackLogger
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def temp_logger():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = Path(f.name)
    logger = FeedbackLogger(db_path=tmp_path)
    yield logger
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass


def test_feedback_logger_direct(temp_logger):
    """Verifies that direct feedback is successfully persisted in SQLite."""
    success = temp_logger.log_feedback(
        query="IS 1786 steel bars application",
        rating=5,
        notes="Extremely clear and grounded response.",
    )
    assert success is True

    recent = temp_logger.get_recent_feedback(limit=10)
    assert len(recent) == 1
    assert recent[0]["query"] == "IS 1786 steel bars application"
    assert recent[0]["rating"] == 5
    assert recent[0]["feedback_notes"] == "Extremely clear and grounded response."

    stats = temp_logger.get_stats()
    assert stats["total_feedback"] == 1
    assert stats["average_rating"] == 5.0


def test_feedback_logger_update_existing_query(temp_logger):
    """Verifies that user rating updates an existing query log record."""
    log_id = temp_logger.log_query(
        query="packaged drinking water certification",
        intent="product_recommendation",
        confidence_score=0.95,
        response_text="IS 14543 is applicable.",
        retrieved_chunks=[{"chunk_id": "chunk_water_001"}],
    )
    assert log_id is not None

    update_success = temp_logger.log_feedback(
        query="packaged drinking water certification",
        rating=4,
        notes="Helpful standard identification.",
        log_id=str(log_id),
    )
    assert update_success is True

    recent = temp_logger.get_recent_feedback(limit=10)
    assert len(recent) == 1
    assert recent[0]["id"] == log_id
    assert recent[0]["rating"] == 4
    assert recent[0]["feedback_notes"] == "Helpful standard identification."


def test_feedback_logger_safe_error_handling(temp_logger):
    """Verifies that FeedbackLogger handles invalid db paths or errors gracefully without raising."""
    broken_logger = FeedbackLogger(db_path=Path("/invalid_dir_path_xyz_123/feedback.db"))
    # Should not raise exception
    res = broken_logger.log_feedback(query="broken path test", rating=3)
    assert res is False
    assert broken_logger.get_recent_feedback() == []
    assert broken_logger.get_stats()["total_feedback"] == 0


def test_feedback_api_endpoints():
    """Verifies the /api/feedback endpoint using FastAPI TestClient."""
    client = TestClient(app)

    # 1. Valid feedback
    res = client.post("/api/feedback", json={
        "query": "IS 12860 metallic coating",
        "rating": 5,
        "notes": "Verified against BIS preview.",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True

    # 2. Malformed feedback: empty query
    res_empty = client.post("/api/feedback", json={
        "query": "   ",
        "rating": 5,
    })
    assert res_empty.status_code == 400
    assert res_empty.json()["success"] is False

    # 3. Malformed feedback: invalid rating
    res_bad_rating = client.post("/api/feedback", json={
        "query": "valid query",
        "rating": 10,  # outside 1-5
    })
    assert res_bad_rating.status_code == 400
    assert res_bad_rating.json()["success"] is False


def test_health_check_endpoint():
    """Verifies that /health endpoint is operational and returns 200."""
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "bis_ai_assistant"
