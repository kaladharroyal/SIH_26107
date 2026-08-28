"""
Classified Hard-Coding & Corpus Contamination Audit - Phase 1 Data Foundation (audit_phase1_hardcoding.py)
Inspects codebase and production corpus for unauthorized hard-coded BIS knowledge and test-fixture contamination.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

# Root directory of bis_RAG_ system
ROOT_DIR = Path(__file__).resolve().parent.parent

# Allowed configuration / parser definitions (not violations)
EXCLUDED_PATTERNS = [
    r"ALLOWED_CATEGORIES",
    r"ALLOWED_PRODUCTION_SOURCE_OF_TRUTH",
    r"ALLOWED_IDENTITY_STATUS",
    r"CLAUSE_HEADER_RE",
    r"STANDARD_ID_RE",
    r"REFUSAL_THRESHOLD",
    r"BM25_TOP_K",
    r"DENSE_TOP_K",
    r"RERANK_TOP_K",
    r"BASE_DIR",
    r"RAW_DIR",
]


class HardCodingAuditor:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.findings: Dict[str, List[Dict[str, Any]]] = {
            "CONFIGURATION": [],
            "PARSER_LOGIC": [],
            "TEST_FIXTURE": [],
            "DOCUMENTATION": [],
            "PRODUCTION_DATA": [],
        }
        self.corpus_contamination_violations: List[str] = []

    def audit_codebase(self):
        """Scans .py files in src/ for hardcoded compliance knowledge."""
        src_dir = self.root_dir / "src"
        if not src_dir.exists():
            return

        for py_file in src_dir.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines, 1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#") or clean_line.startswith('"""'):
                    continue

                # Check for parser logic / regex patterns
                if any(re.search(pat, clean_line) for pat in EXCLUDED_PATTERNS) or "re.compile" in clean_line:
                    self.findings["PARSER_LOGIC"].append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "line": line_idx,
                        "content": clean_line[:100],
                        "classification": "PARSER_LOGIC",
                    })
                    continue

                # Check for hardcoded product/lab dictionaries
                if re.search(r"PRODUCT_DATABASE\s*=\s*\[", clean_line) or re.search(r"LAB_DIRECTORY\s*=\s*\[", clean_line) or re.search(r"FALLBACK_LABS\s*=\s*\[", clean_line):
                    self.findings["PRODUCTION_DATA"].append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "line": line_idx,
                        "content": clean_line[:100],
                        "classification": "PRODUCTION_DATA",
                    })

    def audit_corpus_contamination(self):
        """Verifies that processed_chunks.jsonl and structured tables contain 0 test fixtures or mock placeholders."""
        chunks_file = self.root_dir / "processed_chunks.jsonl"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        sot = record.get("source_of_truth")
                        sfile = record.get("source_file", "")
                        surl = record.get("source_url", "")

                        if sot == "test_fixture":
                            self.corpus_contamination_violations.append(
                                f"Line {line_idx}: chunk {record.get('chunk_id')} has source_of_truth == 'test_fixture'"
                            )
                        if "mock" in sfile.lower() or "test_fixture" in sfile.lower():
                            self.corpus_contamination_violations.append(
                                f"Line {line_idx}: chunk {record.get('chunk_id')} originates from test file {sfile}"
                            )
                        if "example.com" in surl or "mock" in surl.lower():
                            self.corpus_contamination_violations.append(
                                f"Line {line_idx}: chunk {record.get('chunk_id')} has mock URL {surl}"
                            )
                    except Exception as e:
                        self.corpus_contamination_violations.append(f"Line {line_idx}: invalid JSON - {e}")

        # Check structured tables
        psm_file = self.root_dir / "product_standard_map.json"
        if psm_file.exists():
            try:
                data = json.load(open(psm_file, encoding="utf-8"))
                for r in data.get("records", []):
                    prod = str(r.get("product", "")).lower()
                    if prod in ["test product", "mock product", "dummy product", "sample product"]:
                        self.corpus_contamination_violations.append(f"Product map contains mock product: {r.get('product')}")
                    if r.get("source_of_truth") == "test_fixture":
                        self.corpus_contamination_violations.append(f"Product record carries test_fixture: {r}")
            except Exception as e:
                self.corpus_contamination_violations.append(f"product_standard_map.json error: {e}")

        # Check labs
        labs_file = self.root_dir / "labs_directory.json"
        if labs_file.exists():
            try:
                data = json.load(open(labs_file, encoding="utf-8"))
                for r in data.get("records", []):
                    lab_name = str(r.get("lab_name", "")).lower()
                    if "mock" in lab_name or "dummy" in lab_name or "fake" in lab_name:
                        self.corpus_contamination_violations.append(f"Labs directory contains mock lab: {r.get('lab_name')}")
            except Exception as e:
                self.corpus_contamination_violations.append(f"labs_directory.json error: {e}")

    def run(self) -> bool:
        print("=" * 70)
        print("🔍 RUNNING PHASE 1 CLASSIFIED HARD-CODING & CONTAMINATION AUDIT")
        print("=" * 70)

        self.audit_codebase()
        self.audit_corpus_contamination()

        prod_violations = len(self.findings["PRODUCTION_DATA"])
        contamination_violations = len(self.corpus_contamination_violations)

        print(f"  • CONFIGURATION items:    {len(self.findings['CONFIGURATION'])}")
        print(f"  • PARSER_LOGIC items:     {len(self.findings['PARSER_LOGIC'])}")
        print(f"  • TEST_FIXTURE items:     {len(self.findings['TEST_FIXTURE'])}")
        print(f"  • PRODUCTION_DATA items:  {prod_violations} (Must be 0)")
        print(f"  • CORPUS CONTAMINATIONS:  {contamination_violations} (Must be 0)")

        if self.findings["PRODUCTION_DATA"]:
            print("\n❌ PRODUCTION DATA VIOLATIONS FOUND:")
            for v in self.findings["PRODUCTION_DATA"][:10]:
                print(f"   - {v['file']}:{v['line']} -> {v['content']}")

        if self.corpus_contamination_violations:
            print("\n❌ CORPUS CONTAMINATION VIOLATIONS FOUND:")
            for v in self.corpus_contamination_violations[:10]:
                print(f"   - {v}")

        if prod_violations == 0 and contamination_violations == 0:
            print("\n✅ HARD-CODING & CORPUS CONTAMINATION AUDIT PASSED (0 VIOLATIONS)")
            return True
        else:
            print(f"\n❌ AUDIT FAILED WITH {prod_violations + contamination_violations} TOTAL VIOLATIONS")
            return False


if __name__ == "__main__":
    auditor = HardCodingAuditor(ROOT_DIR)
    passed = auditor.run()
    sys.exit(0 if passed else 1)
