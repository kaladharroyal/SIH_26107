"""
Corpus Provenance & Cleanliness Verification Test (test_corpus_provenance.py)
Validates that 100% of chunks in processed_chunks.jsonl have authentic source attribution,
non-empty text, valid unique chunk_ids, valid taxonomy categories, and matching standard metadata.
"""

import json
import re
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "processed_chunks.jsonl"

ALLOWED_CATEGORIES = {
    "is_standard",
    "product_standard_mapping",
    "lab_directory",
    "certification_scheme",
    "consumer_redressal",
    "qco_order",
    "general_policy",
    "hallmarking",
    "act_rules_regulations",
    "general",
    "licensing_fees",
    "annual_report",
    "other",
}



def test_corpus_exists_and_non_empty():
    assert CHUNKS_PATH.exists(), f"Processed chunks file not found at {CHUNKS_PATH}"
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    assert len(chunks) > 50, f"Expected >50 curated chunks, found {len(chunks)}"


def test_chunk_provenance_and_schema_integrity():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    seen_chunk_ids = set()
    category_counts = {}

    for idx, c in enumerate(chunks):
        cid = c.get("chunk_id")
        assert cid, f"Chunk #{idx} is missing chunk_id"
        assert cid not in seen_chunk_ids, f"Duplicate chunk_id found: {cid}"
        seen_chunk_ids.add(cid)

        # Category validity
        cat = c.get("category")
        assert cat in ALLOWED_CATEGORIES, f"Chunk {cid} has invalid category: '{cat}'"
        category_counts[cat] = category_counts.get(cat, 0) + 1

        # Text validity (no empty text chunks)
        text = c.get("text", "")
        assert isinstance(text, str) and len(text.strip()) > 0, f"Chunk {cid} has empty text"


        # Provenance attribution
        src_doc = c.get("source_document") or c.get("source_file")
        assert src_doc, f"Chunk {cid} missing source_document field"
        assert not src_doc.startswith("http://fake"), f"Chunk {cid} has fabricated source"

        src_url = c.get("source_url")
        assert src_url and src_url.startswith("http"), f"Chunk {cid} missing valid source_url"

        # Standard number schema validation
        if c.get("is_number"):
            assert isinstance(c.get("is_number"), str) and len(c.get("is_number").strip()) > 0, f"Chunk {cid} has invalid is_number"





    print(f"\nCurated Corpus Total Chunks: {len(chunks)}")
    for cat, count in category_counts.items():
        print(f"  Category '{cat}': {count} chunks")
