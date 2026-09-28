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
try:
    from src.config import FEEDBACK_DB_PATH
    DB_PATH = FEEDBACK_DB_PATH
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DB_PATH = BASE_DIR / "data" / "bis_rag_telemetry.db"
    if not DB_PATH.exists():
        DB_PATH = BASE_DIR / "feedback_logs.db"


class FeedbackLogger:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        try:
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
        except Exception as e:
            log.error(f"Failed to initialize SQLite feedback database at {self.db_path}: {e}")

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

    def log_feedback(
        self,
        query: str,
        rating: int,
        notes: Optional[str] = None,
        log_id: Optional[Any] = None,
    ) -> bool:
        """
        Logs user feedback. If log_id is an integer corresponding to an existing query,
        updates it; otherwise creates a new entry. Never raises exceptions.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            # If log_id provided and is an integer, attempt update
            if log_id is not None:
                try:
                    int_id = int(log_id)
                    cur.execute(
                        "UPDATE interaction_logs SET rating = ?, feedback_notes = ? WHERE id = ?",
                        (rating, notes, int_id),
                    )
                    if cur.rowcount > 0:
                        conn.commit()
                        conn.close()
                        log.info(f"Updated feedback for interaction #{int_id} (rating={rating})")
                        return True
                except (ValueError, TypeError):
                    pass

            # Otherwise insert a direct feedback record
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                """
                INSERT INTO interaction_logs (timestamp, query, intent, confidence_score, response_text, retrieved_chunks, rating, feedback_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now, query, "user_feedback", 1.0, "", "[]", rating, notes),
            )
            conn.commit()
            conn.close()
            log.info(f"Logged direct feedback for query '{query[:30]}' (rating={rating})")
            return True
        except Exception as e:
            log.error(f"Error persisting feedback to SQLite: {e}")
            return False

    def get_recent_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Internal/testing helper to retrieve recent feedback records."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM interaction_logs WHERE rating IS NOT NULL ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = [dict(row) for row in cur.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            log.error(f"Error reading feedback: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Returns aggregate feedback counts and average rating."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), AVG(rating) FROM interaction_logs WHERE rating IS NOT NULL")
            count, avg_rating = cur.fetchone()
            conn.close()
            return {
                "total_feedback": count or 0,
                "average_rating": round(avg_rating, 2) if avg_rating is not None else 0.0,
            }
        except Exception as e:
            log.error(f"Error retrieving feedback stats: {e}")
            return {"total_feedback": 0, "average_rating": 0.0}


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
