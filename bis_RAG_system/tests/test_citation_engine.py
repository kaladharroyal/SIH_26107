"""
Citation Engine Strict Provenance Verification Test (test_citation_engine.py)
Validates positive matching for citations present in retrieved evidence chunks,
and strict rejection for hallucinated citations, fake IS numbers, or ungrounded claims.
"""

import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from citation_engine import CitationEngine


def test_valid_citations_matching_retrieved_evidence():
    engine = CitationEngine()

    retrieved_chunks = [
        {
            "chunk_id": "is_1786_chem",
            "doc": {
                "is_number": "IS 1786",
                "clause_number": "Clause 4.2",
                "clause_title": "Chemical Composition Limits for TMT Steel Reinforcement",
                "text": "IS 1786:2008 Clause 4.2 prescribes mandatory ladle chemical composition limits for Fe 500D steel.",
                "source_document": "official_standards/IS_1786_2008.pdf",
            },
        },
        {
            "chunk_id": "psm_solar",
            "doc": {
                "is_number": "IS 14286",
                "product": "Crystalline Silicon Terrestrial Photovoltaic (PV) Modules",
                "text": "Indian Standard IS 14286 covers design qualification and type approval for solar PV modules.",
                "source_document": "raw_data/6ca83dd9cf99_Solar-Systems-Order-2025.pdf",
            },
        },
    ]

    valid_generated_citations = [
        "IS 1786",
        "Clause 4.2",
        "IS 14286",
        "Chemical Composition Limits",
    ]

    res = engine.validate_citations_against_context(valid_generated_citations, retrieved_chunks)

    assert res["all_valid"] is True
    assert res["valid_count"] == 4
    assert res["ungrounded_count"] == 0


def test_reject_hallucinated_citations_not_in_evidence():
    engine = CitationEngine()

    # Evidence only contains IS 1786 steel standard
    retrieved_chunks = [
        {
            "chunk_id": "is_1786_chem",
            "doc": {
                "is_number": "IS 1786",
                "clause_number": "Clause 4.2",
                "clause_title": "Chemical Composition Limits",
                "text": "IS 1786:2008 Clause 4.2 prescribes chemical limits for steel reinforcement.",
                "source_document": "official_standards/IS_1786_2008.pdf",
            },
        }
    ]

    hallucinated_citations = [
        "IS 99999",                 # Non-existent standard
        "IS 10322",                 # Real standard, but NOT in retrieved context
        "Clause 99.4",              # Non-existent clause
        "General BIS Guidelines",   # Generic phrase
        "FAQ Portal Guidelines",    # Generic phrase
    ]

    res = engine.validate_citations_against_context(hallucinated_citations, retrieved_chunks)

    assert res["all_valid"] is False
    assert res["ungrounded_count"] == len(hallucinated_citations)
    assert res["valid_count"] == 0
    print(f"\n[Citation Rejection Test] Successfully rejected {res['ungrounded_count']}/{len(hallucinated_citations)} ungrounded citations.")
