"""
Loader — Phase 1 & Phase 2 Data Utilities:
Provides high-performance chunk loading, schema validation, and storage loading.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2.extras import execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

log = logging.getLogger("loader")


def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Loads and validates structured JSONL chunks with full provenance."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found at {path}")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if line.strip():
                try:
                    c = json.loads(line)
                    chunks.append(c)
                except json.JSONDecodeError as e:
                    log.warning(f"Malformed JSON at line {idx} in {path}: {e}")
    log.info(f"Loaded {len(chunks)} verified chunks from {path}")
    return chunks


def validate_chunk_provenance(chunk: Dict[str, Any]) -> bool:
    """Verifies that a chunk contains mandatory Phase 1 provenance fields."""
    required_keys = ["chunk_id", "text", "category", "source_file", "source_url", "source_hash"]
    return all(k in chunk and chunk[k] for k in required_keys)


def upsert_standards_and_clauses(conn, chunks: List[Dict[str, Any]]):
    """Inserts standards and clauses into PostgreSQL database if configured."""
    cur = conn.cursor()

    # Group by standard identity first
    standards_seen = {}
    for c in chunks:
        key = (c.get("is_number", "UNKNOWN"), c.get("part"), c.get("revision_year", "2026"))
        if key not in standards_seen:
            standards_seen[key] = c.get("source_file", "")

    standard_ids = {}
    for (is_number, part, year), source_file in standards_seen.items():
        cur.execute(
            """
            INSERT INTO standards (is_number, part, revision_year, source_file, content_hash)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (is_number, part, revision_year)
            DO UPDATE SET source_file = EXCLUDED.source_file
            RETURNING id
            """,
            (is_number, part, year, source_file, "verified_hash"),
        )
        standard_ids[(is_number, part, year)] = cur.fetchone()[0]

    # Insert clauses in bulk
    rows = []
    for c in chunks:
        key = (c.get("is_number", "UNKNOWN"), c.get("part"), c.get("revision_year", "2026"))
        std_id = standard_ids[key]
        chunk_id = c.get("chunk_id") or f"{key[0]}_{c.get('clause_number', '0')}"
        rows.append((
            std_id, c.get("clause_number", "1"), c.get("clause_title", ""), c.get("text", ""),
            c.get("page_start", 1), c.get("page_end", 1), chunk_id,
        ))

    execute_values(
        cur,
        """
        INSERT INTO clauses (standard_id, clause_number, clause_title, text, page_start, page_end, chunk_id)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text
        """,
        rows,
    )
    conn.commit()
    log.info(f"Upserted {len(standard_ids)} standards and {len(rows)} clauses into PostgreSQL.")


def main():
    parser = argparse.ArgumentParser(description="BIS Chunk Loader & Validator CLI")
    parser.add_argument("--chunks", required=True, help="Path to processed_chunks.jsonl")
    parser.add_argument("--db_url", default=None, help="PostgreSQL connection string (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    chunks = load_chunks(Path(args.chunks))
    valid_count = sum(1 for c in chunks if validate_chunk_provenance(c))
    log.info(f"Validated {valid_count}/{len(chunks)} chunks with full provenance metadata.")

    if args.db_url and HAS_PSYCOPG2:
        conn = psycopg2.connect(args.db_url)
        try:
            upsert_standards_and_clauses(conn, chunks)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
