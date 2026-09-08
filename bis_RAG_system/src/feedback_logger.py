"""
Feedback Logger & Structured Database Manager (feedback_logger.py)
Initializes relational database from schema.sql / schema_postgres.sql, seeds structured catalog tables,
and records live query telemetry and user feedback ratings into PostgreSQL or SQLite with connection pooling.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("feedback_logger")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "bis_rag_telemetry.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"
SCHEMA_PG_PATH = BASE_DIR / "schema_postgres.sql"

_local_storage = threading.local()


class FeedbackLogger:
    """
    Manages live query telemetry logging, structured database seeding,
    and user feedback recording in PostgreSQL (with ThreadedConnectionPool)
    or SQLite (with WAL mode & thread-local connection pool).
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        schema_path: Optional[Path] = None,
        database_url: Optional[str] = None,
    ):
        self.db_path = db_path or DB_PATH
        self.schema_path = schema_path or SCHEMA_PATH
        self.schema_pg_path = SCHEMA_PG_PATH
        self.database_url = database_url or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        self.is_postgres = bool(self.database_url and ("postgres" in self.database_url.lower()))
        self._pg_pool = None

        if self.is_postgres:
            self._init_pg_pool()
        self._init_db()
        self._seed_initial_data()

    def _init_pg_pool(self):
        """Initializes PostgreSQL thread-safe connection pool if psycopg2 is available."""
        try:
            import psycopg2
            from psycopg2 import pool
            self._pg_pool = pool.ThreadedConnectionPool(
                minconn=int(os.getenv("PG_MIN_CONN", "2")),
                maxconn=int(os.getenv("PG_MAX_CONN", "20")),
                dsn=self.database_url,
            )
            log.info("Initialized PostgreSQL ThreadedConnectionPool successfully.")
        except Exception as e:
            log.warning(f"Failed to initialize PostgreSQL pool ({e}). Falling back to SQLite.")
            self.is_postgres = False
            self._pg_pool = None

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        if not hasattr(_local_storage, "sqlite_connections"):
            _local_storage.sqlite_connections = {}
        db_key = str(Path(self.db_path).resolve())
        conn = _local_storage.sqlite_connections.get(db_key)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable Write-Ahead Logging (WAL) for safe multi-threaded concurrency
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
            except Exception:
                pass
            _local_storage.sqlite_connections[db_key] = conn
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Alias for _get_sqlite_connection for test and direct access compatibility."""
        return self._get_sqlite_connection()

    def _init_db(self):
        """Initializes database tables from schema_postgres.sql or schema.sql."""
        if self.is_postgres and self._pg_pool is not None:
            conn = None
            try:
                conn = self._pg_pool.getconn()
                schema_file = self.schema_pg_path if self.schema_pg_path.exists() else self.schema_path
                with open(schema_file, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
                conn.commit()
                log.info("Initialized PostgreSQL telemetry tables from schema_postgres.sql")
            except Exception as e:
                log.error(f"Error initializing PostgreSQL schema: {e}")
                if conn:
                    conn.rollback()
            finally:
                if conn and self._pg_pool:
                    self._pg_pool.putconn(conn)
        else:
            if self.schema_path.exists():
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                with self._get_sqlite_connection() as conn:
                    conn.executescript(schema_sql)
                    conn.commit()
                log.info(f"Initialized SQLite database at {self.db_path} from schema.sql")

    def _seed_initial_data(self):
        """Populates structured product and lab tables if empty."""
        try:
            psm_file = BASE_DIR / "product_standard_map.json"
            labs_file = BASE_DIR / "labs_directory.json"

            if self.is_postgres and self._pg_pool is not None:
                conn = None
                try:
                    conn = self._pg_pool.getconn()
                    with conn.cursor() as cur:
                        # 1. Seed Product Standard Map
                        cur.execute("SELECT COUNT(*) FROM product_standard_map;")
                        count = cur.fetchone()[0]
                        if count == 0 and psm_file.exists():
                            with open(psm_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            records = data.get("records", []) if isinstance(data, dict) else data
                            insert_rows = [
                                (
                                    r.get("product", ""),
                                    r.get("standard", ""),
                                    r.get("hsn_code", ""),
                                    "general",
                                    1 if r.get("mandatory") else 0,
                                    r.get("scheme", ""),
                                    r.get("source_url", ""),
                                )
                                for r in records
                            ]
                            cur.executemany(
                                """
                                INSERT INTO product_standard_map 
                                (product_name, standard_code, hsn_code, category, mandatory, scheme_name, source_url)
                                VALUES (%s, %s, %s, %s, %s, %s, %s);
                                """,
                                insert_rows,
                            )
                            log.info(f"Seeded {len(insert_rows)} product mappings into PostgreSQL.")

                        # 2. Seed Labs Directory
                        cur.execute("SELECT COUNT(*) FROM labs;")
                        count_labs = cur.fetchone()[0]
                        if count_labs == 0 and labs_file.exists():
                            with open(labs_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            records = data.get("records", []) if isinstance(data, dict) else data
                            insert_labs = [
                                (
                                    l.get("lab_id", ""),
                                    l.get("lab_name", ""),
                                    l.get("city", ""),
                                    l.get("state", ""),
                                    l.get("address", ""),
                                    json.dumps(l.get("testing_scope", [])),
                                    json.dumps(l.get("disciplines", [])),
                                    l.get("contact", ""),
                                    l.get("official_url", ""),
                                    1 if l.get("is_recognized", True) else 0,
                                )
                                for l in records
                            ]
                            cur.executemany(
                                """
                                INSERT INTO labs 
                                (lab_id, name, city, state, address, recognized_scopes, disciplines, contact_info, source_url, is_recognized)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                                """,
                                insert_labs,
                            )
                            log.info(f"Seeded {len(insert_labs)} labs into PostgreSQL.")
                    conn.commit()
                except Exception as e:
                    log.error(f"Error seeding PostgreSQL data: {e}")
                    if conn:
                        conn.rollback()
                finally:
                    if conn and self._pg_pool:
                        self._pg_pool.putconn(conn)
            else:
                with self._get_sqlite_connection() as conn:
                    cur = conn.execute("SELECT COUNT(*) FROM product_standard_map")
                    if cur.fetchone()[0] == 0 and psm_file.exists():
                        with open(psm_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        records = data.get("records", []) if isinstance(data, dict) else data
                        insert_rows = [
                            (
                                r.get("product", ""),
                                r.get("standard", ""),
                                r.get("hsn_code", ""),
                                "general",
                                1 if r.get("mandatory") else 0,
                                r.get("scheme", ""),
                                r.get("source_url", ""),
                            )
                            for r in records
                        ]
                        conn.executemany(
                            "INSERT INTO product_standard_map (product_name, standard_code, hsn_code, category, mandatory, scheme_name, source_url) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            insert_rows,
                        )
                        log.info(f"Seeded {len(insert_rows)} product-standard mappings into SQLite.")

                    cur = conn.execute("SELECT COUNT(*) FROM labs")
                    if cur.fetchone()[0] == 0 and labs_file.exists():
                        with open(labs_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        records = data.get("records", []) if isinstance(data, dict) else data
                        insert_labs = [
                            (
                                l.get("lab_id", ""),
                                l.get("lab_name", ""),
                                l.get("city", ""),
                                l.get("state", ""),
                                l.get("address", ""),
                                json.dumps(l.get("testing_scope", [])),
                                json.dumps(l.get("disciplines", [])),
                                l.get("contact", ""),
                                l.get("official_url", ""),
                                1 if l.get("is_recognized", True) else 0,
                            )
                            for l in records
                        ]
                        conn.executemany(
                            "INSERT INTO labs (lab_id, name, city, state, address, recognized_scopes, disciplines, contact_info, source_url, is_recognized) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            insert_labs,
                        )
                        log.info(f"Seeded {len(insert_labs)} laboratory records into SQLite.")
                    conn.commit()
        except Exception as e:
            log.error(f"Error seeding database: {e}")

    def log_query(
        self,
        query: str,
        intent: str,
        confidence_score: float,
        response_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        detected_language: str = "en",
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> int:
        """Logs live query telemetry into query_logs table (PostgreSQL or SQLite)."""
        chunk_ids = []
        for c in retrieved_chunks:
            doc = c.get("doc", c)
            cid = doc.get("chunk_id") or c.get("chunk_id")
            if cid:
                chunk_ids.append(str(cid))
        chunk_ids_json = json.dumps(chunk_ids)

        if self.is_postgres and self._pg_pool is not None:
            conn = None
            try:
                conn = self._pg_pool.getconn()
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO query_logs (
                            user_query, detected_language, intent, retrieved_chunk_ids,
                            generated_answer, confidence_score, retrieval_ms, generation_ms, total_ms
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            query,
                            detected_language,
                            intent,
                            chunk_ids_json,
                            response_text,
                            confidence_score,
                            retrieval_ms,
                            generation_ms,
                            total_ms,
                        ),
                    )
                    row_id = cur.fetchone()[0]
                conn.commit()
                log.info(f"Logged query #{row_id} in PostgreSQL (Intent: {intent}, Conf: {confidence_score:.4f})")
                return row_id
            except Exception as e:
                log.error(f"Failed to log query telemetry to PostgreSQL: {e}")
                if conn:
                    conn.rollback()
                return -1
            finally:
                if conn and self._pg_pool:
                    self._pg_pool.putconn(conn)
        else:
            try:
                with self._get_sqlite_connection() as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO query_logs (
                            user_query, detected_language, intent, retrieved_chunk_ids,
                            generated_answer, confidence_score, retrieval_ms, generation_ms, total_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            query,
                            detected_language,
                            intent,
                            chunk_ids_json,
                            response_text,
                            confidence_score,
                            retrieval_ms,
                            generation_ms,
                            total_ms,
                        ),
                    )
                    conn.commit()
                    log.info(f"Logged query #{cur.lastrowid} in SQLite (Intent: {intent}, Conf: {confidence_score:.4f})")
                    return cur.lastrowid
            except Exception as e:
                log.error(f"Failed to log query telemetry to SQLite: {e}")
                return -1

    def submit_feedback(self, query_id: Optional[int] = None, rating: int = 1, feedback_notes: Optional[str] = None) -> bool:
        """Records user feedback in feedback_ratings table (PostgreSQL or SQLite)."""
        if self.is_postgres and self._pg_pool is not None:
            conn = None
            try:
                conn = self._pg_pool.getconn()
                with conn.cursor() as cur:
                    target_qid = None
                    if isinstance(query_id, int) and query_id > 0:
                        cur.execute("SELECT id FROM query_logs WHERE id = %s;", (query_id,))
                        row = cur.fetchone()
                        target_qid = row[0] if row else None

                    cur.execute(
                        "INSERT INTO feedback_ratings (query_id, rating, feedback_notes) VALUES (%s, %s, %s);",
                        (target_qid, rating, feedback_notes or ""),
                    )
                conn.commit()
                log.info(f"Recorded user feedback in PostgreSQL for Query #{target_qid} (Rating: {rating})")
                return True
            except Exception as e:
                log.error(f"Failed to record feedback in PostgreSQL: {e}")
                if conn:
                    conn.rollback()
                return False
            finally:
                if conn and self._pg_pool:
                    self._pg_pool.putconn(conn)
        else:
            try:
                with self._get_sqlite_connection() as conn:
                    target_qid = None
                    if isinstance(query_id, int) and query_id > 0:
                        cur = conn.execute("SELECT id FROM query_logs WHERE id = ?", (query_id,))
                        row = cur.fetchone()
                        target_qid = row[0] if row else None

                    conn.execute(
                        "INSERT INTO feedback_ratings (query_id, rating, feedback_notes) VALUES (?, ?, ?)",
                        (target_qid, rating, feedback_notes or ""),
                    )
                    conn.commit()
                    log.info(f"Recorded user feedback in SQLite for Query #{target_qid} (Rating: {rating})")
                    return True
            except Exception as e:
                log.error(f"Failed to record feedback in SQLite: {e}")
                return False

    def close(self):
        """Closes thread-local SQLite connection or PostgreSQL connection pool."""
        if self.is_postgres and self._pg_pool is not None:
            try:
                self._pg_pool.closeall()
                log.info("Closed PostgreSQL connection pool.")
            except Exception as e:
                log.warning(f"Error closing PostgreSQL pool: {e}")
        else:
            if hasattr(_local_storage, "sqlite_connections"):
                db_key = str(Path(self.db_path).resolve())
                conn = _local_storage.sqlite_connections.pop(db_key, None)
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

