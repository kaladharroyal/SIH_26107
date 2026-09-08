"""
Confidence Threshold Calibration & Score Separation Benchmark (test_confidence_calibration.py)
Evaluates confidence separation between labeled in-domain BIS queries and out-of-domain / out-of-scope queries.
Computes score separation margin to validate calibrated threshold tau = 0.45.
"""

import numpy as np
import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import BISRAGPipeline


@pytest.fixture(scope="module")
def pipeline():
    return BISRAGPipeline(use_mock_retrieval=False)


def test_confidence_distribution_separation(pipeline):
    in_domain_queries = [
        "What are the chemical composition limits for carbon and sulfur in IS 1786?",
        "What is the yield strength and tensile ratio for Fe 500D steel?",
        "Is solar panel mandatory under quality control order in India?",
        "Where can I find BIS recognized testing laboratories in Mumbai?",
        "How to apply for Scheme-I ISI Mark certification and what is the fee schedule?",
        "What is the statutory compensation for low purity hallmarked gold under Regulation 12?",
        "What are the microbiological limits for packaged drinking water in IS 14543?",
        "What is the testing requirement for secondary lithium cells in IS 16046?",
    ]

    out_of_domain_queries = [
        "What is the average flight duration from London to Tokyo?",
        "How to bake a chocolate cake without sugar?",
        "Who won the FIFA World Cup in 1998?",
        "What is the income tax slab for freelance writers in Canada?",
        "What is quantum entanglement in theoretical physics?",
        "How to repair a broken bicycle chain in rainy weather?",
        "What is the stock price history of Apple in 2012?",
        "How to cultivate organic tomatoes in rooftop garden?",
    ]

    in_domain_scores = []
    for q in in_domain_queries:
        res = pipeline.query(q)
        in_domain_scores.append(res.get("confidence_score", 0.0))

    out_domain_scores = []
    for q in out_of_domain_queries:
        res = pipeline.query(q)
        out_domain_scores.append(res.get("confidence_score", 0.0))

    in_mean = float(np.mean(in_domain_scores))
    in_min = float(np.min(in_domain_scores))
    out_mean = float(np.mean(out_domain_scores))
    out_max = float(np.max(out_domain_scores))
    separation_margin = in_mean - out_mean

    print("\n--- Confidence Calibration Distribution ---")
    print(f"In-Domain Queries (N={len(in_domain_queries)}): Mean={in_mean:.4f}, Min={in_min:.4f}, Max={max(in_domain_scores):.4f}")
    print(f"Out-of-Domain Queries (N={len(out_of_domain_queries)}): Mean={out_mean:.4f}, Min={min(out_domain_scores):.4f}, Max={out_max:.4f}")
    print(f"Separation Margin (In-Mean - Out-Mean): {separation_margin:.4f}")
    print(f"Calibrated Threshold Tau: {pipeline.guardrail.threshold:.2f}")

    assert in_mean > out_mean, "In-domain confidence mean must exceed out-of-domain confidence mean"
    assert separation_margin >= 0.20, f"Expected separation margin >= 0.20, got {separation_margin:.4f}"
    assert out_max < pipeline.guardrail.threshold or out_mean < 0.30, "Out-of-domain queries must be safely refused"
