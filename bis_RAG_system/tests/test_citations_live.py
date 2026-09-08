"""
Verification Test Suite: Clickable Citations & Verifiable Provenance Engine (Feature 10)
Verifies:
1. Zero Synthetic Hash 404 URLs (ensures no local hash prefixes like '5e2367ca2545_' in web URLs)
2. Canonical URL Resolution (standards -> bsbedge, labs -> lims, complaints -> bis_care, schemes -> conformity)
3. Multi-Source Citation Retention & Verified Badge Taxonomy ([Official Standard], [Conformity Scheme], etc.)
4. Multilingual Provenance Block Localization (English, Hindi, Telugu)
5. Pipeline & API Payload Schema Compliance (structured citations returned with metadata)
"""

import os
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src is on Python path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from citation_engine import (
    OFFICIAL_BIS_PORTAL,
    OFFICIAL_COMPLAINTS_PORTAL,
    OFFICIAL_CRS_PORTAL,
    OFFICIAL_LIMS_PORTAL,
    OFFICIAL_STANDARDS_PORTAL,
    CitationEngine,
    resolve_canonical_bis_url,
)
from rag_pipeline import BISRAGPipeline


def test_zero_synthetic_hash_urls():
    print("\n" + "=" * 65)
    print("TEST 1: Zero Synthetic Hash 404 URLs Verification")
    print("=" * 65)

    test_docs = [
        {"source_file": "raw_data/pdfs/5e2367ca2545_Training-Strategy_14-March-2023.pdf", "is_number": None},
        {"source_file": "raw_data/pdfs/1f38c9758aa6_Mandatory-Hallmarking-Order-15.01.2020.pdf", "category": "scheme"},
        {"source_file": "raw_data/pdfs/e2fb91aa4f7d_MarketSurveillance-Guidelines-25Feb2026.pdf", "category": "general"},
        {"source_file": "raw_data/pdfs/d42464bf3dde_IS_1786.pdf", "is_number": "IS 1786"},
    ]

    for doc in test_docs:
        url = resolve_canonical_bis_url(doc)
        print(f"File: {doc['source_file']}")
        print(f"  -> Resolved Canonical URL: {url}")

        # Assert no 12-char hex hash is leaked into the web URL
        assert "5e2367ca2545" not in url, "Leaked local scraper hash in URL!"
        assert "1f38c9758aa6" not in url, "Leaked local scraper hash in URL!"
        assert "e2fb91aa4f7d" not in url, "Leaked local scraper hash in URL!"
        assert "d42464bf3dde" not in url, "Leaked local scraper hash in URL!"
        assert url.startswith("http://") or url.startswith("https://"), "URL must be valid http/https"

    print("\n  [PASS] Zero synthetic hash URLs verified: all resolved to authentic endpoints.")


def test_canonical_url_resolution():
    print("\n" + "=" * 65)
    print("TEST 2: Canonical BIS Endpoint Resolution by Category & Standard")
    print("=" * 65)

    cases = [
        {"doc": {"is_number": "IS 1786"}, "expected_domain": "standardsbis.bsbedge.com", "expected_path": "IS_1786.aspx"},
        {"doc": {"is_number": "IS 16102"}, "expected_domain": "standardsbis.bsbedge.com", "expected_path": "IS_16102.aspx"},
        {"doc": {"category": "lab_directory"}, "expected_domain": "lims.bis.gov.in", "expected_path": ""},
        {"doc": {"category": "consumer_protection"}, "expected_domain": "www.bis.gov.in", "expected_path": "online-complaint-registration"},
        {"doc": {"category": "scheme_ii"}, "expected_domain": "www.crsbis.in", "expected_path": ""},
    ]

    for c in cases:
        url = resolve_canonical_bis_url(c["doc"])
        print(f"Category/Standard: {c['doc']}")
        print(f"  -> URL: {url}")
        assert c["expected_domain"] in url, f"Expected domain {c['expected_domain']} in {url}"
        if c["expected_path"]:
            assert c["expected_path"] in url, f"Expected path {c['expected_path']} in {url}"

    print("\n  [PASS] Canonical domain routing verified across standards, labs, complaints, and schemes.")


def test_multisource_retention_and_badges(engine: Optional[CitationEngine] = None):
    print("\n" + "=" * 65)
    print("TEST 3: Multi-Source Citation Retention & Badge Taxonomy")
    print("=" * 65)

    if engine is None:
        engine = CitationEngine()

    context_chunks = [
        {
            "doc": {
                "chunk_id": "chunk_std_01",
                "is_number": "IS 1786",
                "revision_year": "2008",
                "clause_number": "4.2",
                "clause_title": "Chemical Composition Limits",
                "category": "is_standard",
                "page_start": 6,
                "source_file": "raw_data/pdfs/IS_1786.pdf",
            }
        },
        {
            "doc": {
                "chunk_id": "chunk_scheme_01",
                "clause_title": "Product Certification Scheme (Scheme-I)",
                "category": "scheme_i",
                "source_file": "raw_data/pdfs/certification-process.pdf",
            }
        },
        {
            "doc": {
                "chunk_id": "chunk_lab_01",
                "clause_title": "Central Laboratory Testing Facilities",
                "category": "lab_directory",
                "source_file": "labs_directory.json",
            }
        },
    ]

    sample_response = "TMT bars must conform to IS 1786 [As per IS 1786:2008, Clause 4.2] under Scheme-I certification."
    res = engine.format_citations(sample_response, context_chunks)

    citations = res["citations_list"]
    print(f"Total Sources Retained: {len(citations)}")
    assert len(citations) == 3, f"Expected 3 retained sources, got {len(citations)}"

    badges = [c["badge"] for c in citations]
    print(f"Assigned Badges: {badges}")

    assert "[Official Standard]" in badges, "Missing [Official Standard] badge"
    assert "[Conformity Scheme]" in badges, "Missing [Conformity Scheme] badge"
    assert "[LIMS Lab Directory]" in badges, "Missing [LIMS Lab Directory] badge"

    # Verify page parameter deep-linking
    std_cite = next(c for c in citations if c["badge"] == "[Official Standard]")
    print(f"Standard Citation Details: {std_cite}")
    assert std_cite["page"] == 6

    print("\n  [PASS] Multi-source citation retention and verified badge taxonomy verified.")


def test_multilingual_citation_formatting(engine: Optional[CitationEngine] = None):
    print("\n" + "=" * 65)
    print("TEST 4: Multilingual Citation Block Localization (English, Hindi, Telugu)")
    print("=" * 65)

    if engine is None:
        engine = CitationEngine()

    dummy_chunk = [{
        "doc": {
            "chunk_id": "chunk_01",
            "is_number": "IS 16102",
            "revision_year": "2012",
            "category": "is_standard",
            "source_file": "IS_16102.pdf",
        }
    }]

    # English
    res_en = engine.format_citations("LED bulb requirements.", dummy_chunk, language="English")
    assert "### Source" in res_en["formatted_text"]
    assert "Open official BIS" in res_en["formatted_text"]
    print("  [✓] English Header & Action Text Verified.")

    # Hindi
    res_hi = engine.format_citations("एलईडी बल्ब मानक आवश्यकताएं।", dummy_chunk, language="Hindi")
    assert "### स्रोत" in res_hi["formatted_text"]
    assert "आधिकारिक BIS" in res_hi["formatted_text"]
    print("  [✓] Hindi Header & Action Text Verified.")

    # Telugu
    res_te = engine.format_citations("ఎల్ఈడీ బల్బ్ అవసరాలు.", dummy_chunk, language="Telugu")
    assert "### మూలం" in res_te["formatted_text"]
    assert "అధికారిక BIS" in res_te["formatted_text"]
    print("  [✓] Telugu Header & Action Text Verified.")

    print("\n  [PASS] Multilingual provenance formatting localized across languages.")


def test_pipeline_citation_payload(pipeline: BISRAGPipeline):
    print("\n" + "=" * 65)
    print("TEST 5: Live Pipeline Citations Payload Integrity")
    print("=" * 65)

    query = "what is the Indian standard for TMT steel bar"
    res = pipeline.query(query)

    print(f"Query: '{query}'")
    print(f"  Flow Used:   {res.get('flow_used')}")
    print(f"  Status:      {res.get('status')}")
    citations = res.get("citations", [])
    print(f"  Citations Count: {len(citations)}")
    print(f"  Citations List:  {citations}")

    assert len(citations) > 0, "Pipeline returned empty citations!"
    for c in citations:
        url = c if isinstance(c, str) else c.get("url", "")
        assert url.startswith("http"), f"Citation URL must be valid web URL, got {url}"

    print("\n  [PASS] Pipeline returns structured, verifiable citations payload.")


def run_all_tests():
    print("=" * 65)
    print("RUNNING FEATURE 10 VERIFICATION SUITE: CLICKABLE CITATIONS ENGINE")
    print("=" * 65)

    engine = CitationEngine()
    pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)

    test_zero_synthetic_hash_urls()
    test_canonical_url_resolution()
    test_multisource_retention_and_badges(engine)
    test_multilingual_citation_formatting(engine)
    test_pipeline_citation_payload(pipeline)

    print("\n" + "=" * 65)
    print("ALL 5 CLICKABLE CITATIONS ENGINE TESTS PASSED 100%!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
