"""
Phase 6, Step 19: Feedback & Interaction Logger (feedback_logger.py)
Captures live query logs, retrieved chunk IDs, confidence scores, and user ratings in SQLite.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feedback_logger")

DB_PATH = Path(__file__).resolve().parent.parent / "feedback_logs.db"


class FeedbackLogger:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                intent TEXT,
                confidence_score REAL,
                response_text TEXT,
                retrieved_chunks TEXT,
                rating INTEGER,
                feedback_notes TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def log_query(
        self,
        query: str,
        intent: str,
        confidence_score: float,
        response_text: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        chunks_json = json.dumps([c.get("chunk_id") for c in retrieved_chunks if "chunk_id" in c])

        cur.execute(
            """
            INSERT INTO interaction_logs (timestamp, query, intent, confidence_score, response_text, retrieved_chunks)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now, query, intent, confidence_score, response_text, chunks_json),
        )
        log_id = cur.lastrowid
        conn.commit()
        conn.close()
        log.info(f"Logged interaction #{log_id} (intent={intent}, conf={confidence_score:.2f})")
        return log_id

    def submit_feedback(self, log_id: int, rating: int, feedback_notes: Optional[str] = None) -> bool:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE interaction_logs
            SET rating = ?, feedback_notes = ?
            WHERE id = ?
            """,
            (rating, feedback_notes, log_id),
        )
        success = cur.rowcount > 0
        conn.commit()
        conn.close()
        log.info(f"Submitted feedback rating {rating} for log ID #{log_id}")
        return success


if __name__ == "__main__":
    logger = FeedbackLogger()
    lid = logger.log_query(
        query="test query for certification",
        intent="general_search",
        confidence_score=0.90,
        response_text="Test verified standard response text.",
        retrieved_chunks=[{"chunk_id": "test_chunk_001"}],
    )
    logger.submit_feedback(lid, rating=1, feedback_notes="Verification test passed.")
