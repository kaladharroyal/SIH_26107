"""
Phase 6, Step 21: Feedback Loop Logger (feedback_logger.py)
Logs user queries, retrieved chunks, confidence scores, and user feedback (thumbs up/down) into PostgreSQL/SQLite query_logs.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feedback_logger")

DB_PATH = Path("./bis_feedback.db")


class FeedbackLogger:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
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
        """)
        conn.commit()
        conn.close()
        log.info("Initialized Feedback Logger database schema.")

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
        chunks_json = json.dumps(retrieved_chunks)

        cur.execute(
            """
            INSERT INTO query_logs (timestamp, query, intent, confidence_score, response_text, retrieved_chunks)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now, query, intent, confidence_score, response_text, chunks_json),
        )
        log_id = cur.lastrowid
        conn.commit()
        conn.close()
        log.info(f"Logged query #{log_id}: '{query}' (Intent: {intent}, Conf: {confidence_score:.4f})")
        return log_id

    def submit_feedback(self, log_id: int, rating: int, feedback_notes: str = "") -> bool:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE query_logs
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
        query="is certification mandatory for LED bulbs",
        intent="product_recommendation",
        confidence_score=0.95,
        response_text="Yes, LED bulbs require mandatory CRS certification under IS 16102.",
        retrieved_chunks=[{"chunk_id": "IS_16102"}],
    )
    logger.submit_feedback(lid, rating=1, feedback_notes="Great exact answer!")
