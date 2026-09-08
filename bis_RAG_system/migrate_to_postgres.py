"""
SQLite to PostgreSQL Database Migration Utility (migrate_to_postgres.py)
Migrates standards, clauses, product mappings, laboratory directories,
query telemetry logs, and feedback ratings from SQLite to PostgreSQL with connection pooling.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("db_migration")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITE_PATH = BASE_DIR / "bis_rag_telemetry.db"
SCHEMA_PG_PATH = BASE_DIR / "schema_postgres.sql"


def run_migration(sqlite_path: Path, postgres_url: str):
    """Executes schema initialization and batch record migration from SQLite to PostgreSQL."""
    if not sqlite_path.exists():
        log.error(f"Source SQLite database not found at {sqlite_path}")
        sys.exit(1)

    try:
        import psycopg2
        from psycopg2.extras import execute_batch
    except ImportError:
        log.error("psycopg2 is required for PostgreSQL migration. Install with 'pip install psycopg2-binary'.")
        sys.exit(1)

    log.info(f"Connecting to source SQLite DB: {sqlite_path}")
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row

    log.info(f"Connecting to target PostgreSQL: {postgres_url.split('@')[-1] if '@' in postgres_url else 'PostgreSQL'}")
    pg_conn = psycopg2.connect(postgres_url)
    pg_conn.autocommit = False

    try:
        # 1. Initialize PostgreSQL Schema
        log.info("Applying schema_postgres.sql to target database...")
        with open(SCHEMA_PG_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with pg_conn.cursor() as cur:
            cur.execute(schema_sql)
        pg_conn.commit()
        log.info("✓ PostgreSQL schema initialized.")

        # 2. Migrate Standards
        cur_sqlite = sqlite_conn.cursor()
        cur_sqlite.execute("SELECT is_number, part, revision_year, title, is_current, source_file, source_url, content_hash, ingested_at FROM standards")
        standards = [tuple(row) for row in cur_sqlite.fetchall()]
        if standards:
            with pg_conn.cursor() as cur_pg:
                execute_batch(
                    cur_pg,
                    """
                    INSERT INTO standards (is_number, part, revision_year, title, is_current, source_file, source_url, content_hash, ingested_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (is_number, part, revision_year) DO NOTHING;
                    """,
                    standards,
                    page_size=500,
                )
            pg_conn.commit()
            log.info(f"✓ Migrated {len(standards)} standards.")

        # 3. Migrate Product Standard Map
        cur_sqlite.execute("SELECT product_name, standard_code, hsn_code, category, mandatory, scheme_name, source_url, updated_at FROM product_standard_map")
        psm_rows = [tuple(row) for row in cur_sqlite.fetchall()]
        if psm_rows:
            with pg_conn.cursor() as cur_pg:
                execute_batch(
                    cur_pg,
                    """
                    INSERT INTO product_standard_map (product_name, standard_code, hsn_code, category, mandatory, scheme_name, source_url, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    psm_rows,
                    page_size=500,
                )
            pg_conn.commit()
            log.info(f"✓ Migrated {len(psm_rows)} product-standard mapping records.")

        # 4. Migrate Labs Directory
        cur_sqlite.execute("SELECT lab_id, name, city, state, address, recognized_scopes, disciplines, contact_info, source_url, is_recognized, updated_at FROM labs")
        lab_rows = [tuple(row) for row in cur_sqlite.fetchall()]
        if lab_rows:
            with pg_conn.cursor() as cur_pg:
                execute_batch(
                    cur_pg,
                    """
                    INSERT INTO labs (lab_id, name, city, state, address, recognized_scopes, disciplines, contact_info, source_url, is_recognized, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (lab_id) DO NOTHING;
                    """,
                    lab_rows,
                    page_size=500,
                )
            pg_conn.commit()
            log.info(f"✓ Migrated {len(lab_rows)} lab directory records.")

        # 5. Migrate Query Logs & Feedback Ratings
        cur_sqlite.execute("SELECT id, user_query, detected_language, intent, retrieved_chunk_ids, generated_answer, confidence_score, retrieval_ms, generation_ms, total_ms, created_at FROM query_logs ORDER BY id ASC")
        qlog_rows = [tuple(row) for row in cur_sqlite.fetchall()]
        if qlog_rows:
            with pg_conn.cursor() as cur_pg:
                execute_batch(
                    cur_pg,
                    """
                    INSERT INTO query_logs (id, user_query, detected_language, intent, retrieved_chunk_ids, generated_answer, confidence_score, retrieval_ms, generation_ms, total_ms, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    qlog_rows,
                    page_size=500,
                )
                # Reset sequence for bigserial
                cur_pg.execute("SELECT setval('query_logs_id_seq', (SELECT COALESCE(MAX(id), 1) FROM query_logs));")
            pg_conn.commit()
            log.info(f"✓ Migrated {len(qlog_rows)} query interaction logs.")

        cur_sqlite.execute("SELECT id, query_id, rating, feedback_notes, submitted_at FROM feedback_ratings ORDER BY id ASC")
        feedback_rows = [tuple(row) for row in cur_sqlite.fetchall()]
        if feedback_rows:
            with pg_conn.cursor() as cur_pg:
                execute_batch(
                    cur_pg,
                    """
                    INSERT INTO feedback_ratings (id, query_id, rating, feedback_notes, submitted_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    feedback_rows,
                    page_size=500,
                )
                cur_pg.execute("SELECT setval('feedback_ratings_id_seq', (SELECT COALESCE(MAX(id), 1) FROM feedback_ratings));")
            pg_conn.commit()
            log.info(f"✓ Migrated {len(feedback_rows)} feedback ratings.")

        log.info("=======================================================")
        log.info("🎉 SQLite to PostgreSQL migration completed successfully!")
        log.info("=======================================================")

    except Exception as e:
        pg_conn.rollback()
        log.error(f"Migration failed: {e}")
        raise e
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate BIS RAG Telemetry from SQLite to PostgreSQL")
    parser.add_argument("--sqlite", type=str, default=str(DEFAULT_SQLITE_PATH), help="Path to SQLite database file")
    parser.add_argument("--postgres-url", type=str, default=os.getenv("DATABASE_URL"), help="PostgreSQL connection URI (e.g. postgresql://user:pass@localhost:5432/dbname)")
    args = parser.parse_args()

    pg_url = args.postgres_url
    if not pg_url:
        log.error("PostgreSQL URL is required. Provide via --postgres-url or DATABASE_URL env var.")
        sys.exit(1)

    run_migration(Path(args.sqlite), pg_url)
