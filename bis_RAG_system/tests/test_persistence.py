"""
Relational & Telemetry Persistence Verification Test (test_persistence.py)
Validates that schema.sql is executed, structured product & lab tables are populated,
live queries are logged into query_logs, and feedback ratings are persisted in SQLite.
"""

import sqlite3
import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feedback_logger import FeedbackLogger
from rag_pipeline import BISRAGPipeline


def test_sqlite_schema_and_seeding(tmp_path):
    db_path = tmp_path / "test_telemetry.db"

    logger = FeedbackLogger(db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row

        # 1. Verify structured tables exist
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "standards" in tables
        assert "clauses" in tables
        assert "product_standard_map" in tables
        assert "labs" in tables
        assert "query_logs" in tables
        assert "feedback_ratings" in tables

        # 2. Verify seeded records
        cur = conn.execute("SELECT COUNT(*) FROM product_standard_map")
        psm_count = cur.fetchone()[0]
        assert psm_count > 0, f"Expected seeded product_standard_map, got {psm_count}"

        cur = conn.execute("SELECT COUNT(*) FROM labs")
        labs_count = cur.fetchone()[0]
        assert labs_count > 0, f"Expected seeded labs, got {labs_count}"
    finally:
        conn.close()

    print(f"\n[Persistence Test] SQLite tables verified & seeded: {psm_count} products, {labs_count} laboratories.")



def test_live_query_telemetry_logging():
    pipeline = BISRAGPipeline(use_mock_retrieval=False)
    db_path = pipeline.feedback_logger.db_path

    # Run query
    test_query = "What are the chemical limits in IS 1786 steel?"
    res = pipeline.query(test_query)

    # Verify query was logged in database
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()

        assert row is not None
        assert test_query in row["user_query"]
        assert row["confidence_score"] == pytest.approx(res["confidence_score"], 0.001)
        logged_id = row["id"]

    # Submit feedback
    success = pipeline.feedback_logger.submit_feedback(query_id=logged_id, rating=1, feedback_notes="Accurate chemical composition limits")
    assert success is True

    # Verify feedback was recorded
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        fb_row = cur.execute("SELECT * FROM feedback_ratings WHERE query_id = ?", (logged_id,)).fetchone()
        assert fb_row is not None
        assert fb_row[2] == 1 or fb_row["rating"] == 1
        assert "Accurate" in (fb_row[3] or fb_row["feedback_notes"])

    print(f"\n[Persistence Test] Live query #{logged_id} and user feedback rating successfully verified in SQLite.")

