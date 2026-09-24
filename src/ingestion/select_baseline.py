"""
Deterministic 1,000-Source Baseline Selector - Phase 1 Data Foundation (select_baseline.py)
Creates a deterministic, representative baseline manifest of exactly at most 1,000 logical sources.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "raw_data"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"
BASELINE_MANIFEST_1000_PATH = RAW_DIR / "phase1_baseline_manifest_1000.jsonl"

TARGET_BASELINE_SIZE = 1000


def build_1000_baseline_manifest() -> List[Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_records = [json.loads(line) for line in f if line.strip()]

    # Group by category
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for m in manifest_records:
        cat = m.get("category", "other")
        by_category.setdefault(cat, []).append(m)

    # Sort each category deterministically by content_hash and source_url
    for cat in by_category:
        by_category[cat].sort(key=lambda x: (x.get("content_hash", ""), x.get("source_url", "")))

    selected_records: List[Dict[str, Any]] = []

    # Priority categories to include 100%
    priority_categories = [
        "certification_faq",
        "certification_process",
        "hallmarking_faq",
        "hallmarking",
        "consumer_faq",
        "consumer",
        "lab_faq",
        "lab_directory",
        "licensing_fees",
        "licensing_procedure",
        "training",
        "training_faq",
        "lims_recognized_labs",
        "lims_empaneled_labs",
        "know_your_standards",
    ]

    for cat in priority_categories:
        if cat in by_category:
            selected_records.extend(by_category[cat])

    # Add representative product standard mappings (up to 450)
    if "product_standard_mapping" in by_category:
        psm_records = by_category["product_standard_mapping"]
        selected_records.extend(psm_records[:450])

    # Fill remaining capacity up to TARGET_BASELINE_SIZE with bis_act_rules_regulations
    remaining_quota = TARGET_BASELINE_SIZE - len(selected_records)
    if remaining_quota > 0 and "bis_act_rules_regulations" in by_category:
        act_records = by_category["bis_act_rules_regulations"]
        selected_records.extend(act_records[:remaining_quota])

    # Safety assertion: Must not exceed TARGET_BASELINE_SIZE
    if len(selected_records) > TARGET_BASELINE_SIZE:
        raise ValueError(
            f"Selected baseline exceeds {TARGET_BASELINE_SIZE} records ({len(selected_records)}). "
            f"Silent truncation is prohibited."
        )

    # Sort final selected records deterministically by content_hash
    selected_records.sort(key=lambda x: (x.get("content_hash", ""), x.get("source_url", "")))

    with open(BASELINE_MANIFEST_1000_PATH, "w", encoding="utf-8") as f:
        for r in selected_records:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"Generated deterministic 1,000-source baseline manifest at {BASELINE_MANIFEST_1000_PATH}")
    print(f"Total Selected Baseline Sources: {len(selected_records)}")
    return selected_records


if __name__ == "__main__":
    build_1000_baseline_manifest()
