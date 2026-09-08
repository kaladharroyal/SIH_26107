"""
Zero Hardcoded Confidence & Behavioral Audit (test_zero_hardcoded_scores.py)
1. Static check: Scans request-handling Python files in src/ to ensure zero hardcoded confidence literals exist.
2. Behavioral check: Runs varied queries through the live pipeline and verifies dynamic score variance without clustering.
"""

import os
import re
import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_static_zero_hardcoded_confidence_literals():
    """
    Regex scan across all src/ pipeline and subflow modules to verify
    no hardcoded confidence scores (e.g. 0.95, 0.98, 0.99, 0.40) are assigned in payload dictionaries.
    """
    target_files = [
        SRC_DIR / "rag_pipeline.py",
        SRC_DIR / "product_recommender.py",
        SRC_DIR / "scheme_walkthrough.py",
        SRC_DIR / "lab_locator.py",
        SRC_DIR / "consumer_complaint.py",
    ]

    # Pattern detecting hardcoded assignment to confidence_score
    hardcode_patterns = [
        r'["\']confidence_score["\']\s*:\s*(?:0\.95|0\.98|0\.99|0\.40|0\.50)',
        r'confidence_score\s*=\s*(?:0\.95|0\.98|0\.99|0\.40|0\.50)',
        r'0\.95\s+if\s+.*?\s+else\s+0\.40',
        r'0\.98\s+if\s+.*?\s+else\s+0\.40',
        r'0\.99\s+if\s+.*?\s+else\s+0\.40',
    ]

    violations = []

    for tf in target_files:
        if not tf.exists():
            continue
        with open(tf, "r", encoding="utf-8") as f:
            content = f.read()

        for pattern in hardcode_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"{tf.name}: Matched hardcoded pattern '{pattern}' -> {matches}")

    assert len(violations) == 0, f"Found hardcoded confidence violations:\n" + "\n".join(violations)


def test_behavioral_confidence_score_variance():
    """
    Behavioral check: Runs varied queries through BISRAGPipeline and verifies
    that confidence scores vary dynamically and are not clamped to a constant value.
    """
    from rag_pipeline import BISRAGPipeline

    pipeline = BISRAGPipeline(use_mock_retrieval=False)

    test_queries = [
        "What are the chemical composition limits for Fe 500D steel in IS 1786?",
        "Is solar panel certification mandatory under QCO?",
        "Where are BIS recognized testing labs in Mumbai?",
        "How to apply for ISI Mark Scheme-I and what are the fees?",
        "What is the statutory compensation for low purity hallmarked gold?",
        "What is the capital of Mars in solar system?", # Out-of-domain
    ]

    scores = []
    for q in test_queries:
        res = pipeline.query(q)
        score = res.get("confidence_score", 0.0)
        scores.append((q, score, res.get("status")))

    print("\n--- Behavioral Confidence Scores ---")
    for q, s, status in scores:
        print(f"  [{status}] Score: {s:.4f} -> Query: '{q[:45]}...'")

    # Verify scores are not all identical
    unique_scores = set(round(s, 3) for _, s, _ in scores)
    assert len(unique_scores) >= 3, f"Expected distinct confidence scores across varied queries, got: {unique_scores}"

    # Verify out-of-domain query gets significantly lower score than in-domain technical query
    in_domain_score = scores[0][1] # IS 1786 query
    out_domain_score = scores[-1][1] # Mars query
    assert in_domain_score > out_domain_score, f"Expected in-domain ({in_domain_score}) > out-of-domain ({out_domain_score})"
