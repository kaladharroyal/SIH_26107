"""
Loader — Phase 1, Step 4 (final step): push processed chunks into Postgres
and prepare them for embedding into the vector DB.

This is intentionally split from embedding/upsert-to-vector-DB (that's Phase 2)
so Phase 1 has a clean, testable output: a Postgres table you can query and
sanity-check by hand before spending API calls on embeddings.

Usage:
    python loader.py --chunks ../processed_chunks.jsonl --db_url postgresql://...
"""

import argparse
import json
import logging
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

log = logging.getLogger("loader")


def load_chunks(chunks_path: Path) -> list[dict]:
    with open(chunks_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def upsert_standards_and_clauses(conn, chunks: list[dict]):
    cur = conn.cursor()

    # Group by standard identity first, so each standard is inserted once.
    standards_seen = {}
    for c in chunks:
        key = (c["is_number"], c.get("part"), c["revision_year"])
        if key not in standards_seen:
            standards_seen[key] = c["source_file"]

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
            (is_number, part, year, source_file, "placeholder_hash"),
        )
        standard_ids[(is_number, part, year)] = cur.fetchone()[0]

    # Insert clauses in bulk.
    rows = []
    for c in chunks:
        key = (c["is_number"], c.get("part"), c["revision_year"])
        std_id = standard_ids[key]
        chunk_id = f"{c['is_number'].replace(' ', '')}_{c.get('part') or 'NA'}_{c['revision_year']}_C{c['clause_number']}"
        rows.append((
            std_id, c["clause_number"], c.get("clause_title"), c["text"],
            c.get("page_start"), c.get("page_end"), chunk_id,
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
    log.info(f"Upserted {len(standard_ids)} standards and {len(rows)} clauses")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--db_url", required=True, help="postgresql://user:pass@host:port/dbname")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    chunks = load_chunks(Path(args.chunks))
    conn = psycopg2.connect(args.db_url)
    try:
        upsert_standards_and_clauses(conn, chunks)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
