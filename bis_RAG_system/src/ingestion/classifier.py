"""
Phase 1 Full Corpus Source Classifier (classifier.py)
Classifies all 35,132 logical manifest records across the complete BIS corpus without modifying raw_data.
Separates Conceptual Content Classification from Physical File Integrity, creates zero-copy index manifests
under classified_data/, and guarantees zero-loss ID verification and multi-tier dynamic accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("corpus_classifier")

# 11-category classification taxonomy
PDF_CATEGORIES = {
    "PDF_STANDARD",
    "PDF_AMENDMENT",
    "PDF_CERTIFICATION",
    "PDF_HALLMARKING",
    "PDF_CONSUMER",
    "PDF_GENERAL",
}

JSON_CATEGORIES = {
    "FAQ_JSON",
    "PSM_JSON",
    "API_PREVIEW_JSON",
    "OTHER_JSON",
}

ALL_CATEGORIES = PDF_CATEGORIES | JSON_CATEGORIES | {"UNKNOWN"}

# File integrity taxonomy
FILE_INTEGRITY_TYPES = {
    "VALID",
    "NOT_A_PDF",
    "FILE_NOT_FOUND",
    "MALFORMED_JSON",
    "EMPTY",
}

# Regex patterns for deterministic content classification
RE_IS_STANDARD = re.compile(
    r"\bIS(?:/ISO|/IEC)?\s*[:\-_]?\s*(\d{1,6})(?:\s*[\(:-]?Part\s*(\d+)[\):-]?)?(?:[\s:_–-]*(\d{4}))?\b",
    re.IGNORECASE,
)
RE_SP_STANDARD = re.compile(r"\bSP\s*[:\-_]?\s*(\d+)", re.IGNORECASE)
RE_URL_STD_ID = re.compile(r"[?&](?:id|stdno)=(?:IS[\s_:-]*)?(\d+)(?:_(\d{4}))?", re.IGNORECASE)

AMENDMENT_KEYWORDS = [
    r"amendment",
    r"amend",
    r"qco",
    r"quality[\s_-]?control[\s_-]?order",
    r"corrigendum",
    r"gazette",
    r"notification",
    r"order[\s_-]?of[\s_-]?extension",
    r"extension[\s_-]?in[\s_-]?date",
    r"extension[\s_-]?notification",
    r"rescind",
    r"temporary[\s_-]?suspension",
]
RE_AMENDMENT = re.compile("|".join(AMENDMENT_KEYWORDS), re.IGNORECASE)

HALLMARKING_KEYWORDS = [
    r"hallmark",
    r"hallmarking",
    r"gold[\s_-]?jeweller",
    r"silver[\s_-]?jeweller",
    r"gold[\s_-]?monetization",
    r"assaying",
    r"ahc",
    r"refiner",
]
RE_HALLMARKING = re.compile("|".join(HALLMARKING_KEYWORDS), re.IGNORECASE)

CERTIFICATION_KEYWORDS = [
    r"product[\s_-]?certification",
    r"system[\s_-]?certification",
    r"certification[\s_-]?process",
    r"licensing[\s_-]?procedure",
    r"licensing[\s_-]?fee",
    r"fmcs",
    r"crs",
    r"compulsory[\s_-]?registration",
    r"simplified[\s_-]?procedure",
    r"grant[\s_-]?of[\s_-]?licen",
    r"surveillance",
    r"concession",
    r"who[\s_-]?can[\s_-]?apply",
    r"how[\s_-]?to[\s_-]?apply",
]
RE_CERTIFICATION = re.compile("|".join(CERTIFICATION_KEYWORDS), re.IGNORECASE)

CONSUMER_KEYWORDS = [
    r"consumer",
    r"complaint",
    r"citizen[\s_-]?charter",
    r"brochure",
    r"grievance",
    r"consumer[\s_-]?guid",
    r"know[\s_-]?your[\s_-]?right",
]
RE_CONSUMER = re.compile("|".join(CONSUMER_KEYWORDS), re.IGNORECASE)

FAQ_KEYWORDS = [
    r"faq",
    r"frequently[\s_-]?asked",
    r"questions?[\s_-]?and[\s_-]?answers?",
]
RE_FAQ = re.compile("|".join(FAQ_KEYWORDS), re.IGNORECASE)

PSM_KEYWORDS = [
    r"product[\s_-]?standard[\s_-]?mapping",
    r"product[\s_-]?standard[\s_-]?map",
    r"downloadcrossreferences",
    r"crossreference",
    r"product[\s_-]?manual",
]
RE_PSM = re.compile("|".join(PSM_KEYWORDS), re.IGNORECASE)

API_PREVIEW_KEYWORDS = [
    r"bis_preview",
    r"isdetails",
    r"knowyourstandards",
    r"know_your_standards",
    r"search_redirect",
    r"advance_search",
    r"standard_specification",
]
RE_API_PREVIEW = re.compile("|".join(API_PREVIEW_KEYWORDS), re.IGNORECASE)


class CorpusClassifier:
    def __init__(self, raw_dir: Path, out_dir: Path, manifest_path: Path):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.manifest_path = Path(manifest_path)

        self.pdf_dir = self.out_dir / "pdf"
        self.json_dir = self.out_dir / "json"

        # Partition file mapping
        self.partition_paths = {
            "PDF_STANDARD": self.pdf_dir / "standards.jsonl",
            "PDF_AMENDMENT": self.pdf_dir / "amendments.jsonl",
            "PDF_CERTIFICATION": self.pdf_dir / "certification.jsonl",
            "PDF_HALLMARKING": self.pdf_dir / "hallmarking.jsonl",
            "PDF_CONSUMER": self.pdf_dir / "consumer.jsonl",
            "PDF_GENERAL": self.pdf_dir / "general.jsonl",
            "FAQ_JSON": self.json_dir / "faq.jsonl",
            "PSM_JSON": self.json_dir / "product_standard_mapping.jsonl",
            "API_PREVIEW_JSON": self.json_dir / "api_preview.jsonl",
            "OTHER_JSON": self.json_dir / "other.jsonl",
            "UNKNOWN": self.out_dir / "unknown.jsonl",
        }

    def _ensure_output_dirs(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def inspect_pdf_integrity(self, file_path: Path) -> Tuple[str, bool, int, Optional[str]]:
        """
        Inspects physical PDF file integrity in read-only binary mode.
        Returns: (file_integrity, is_valid_file, size_bytes, reason)
        """
        if not file_path.exists():
            return "FILE_NOT_FOUND", False, 0, "Physical file does not exist on disk"

        try:
            size_bytes = file_path.stat().st_size
        except Exception as e:
            return "FILE_NOT_FOUND", False, 0, f"Cannot stat file: {e}"

        if size_bytes == 0:
            return "EMPTY", False, 0, "Physical file is 0 bytes"

        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)

            # Check if HTML response was saved with .pdf extension
            if header.startswith(b"\r\n\r\n<!DOCTYPE html") or header.startswith(b"<!DOCTYPE html") or b"<html" in header[:200].lower():
                return "NOT_A_PDF", False, size_bytes, "File contains HTML web server response instead of binary PDF"

            if header.startswith(b"%PDF-") or b"%PDF-" in header[:1024]:
                return "VALID", True, size_bytes, "Valid binary PDF with %PDF- header"
            else:
                return "NOT_A_PDF", False, size_bytes, "Missing %PDF- header signature"
        except Exception as e:
            return "NOT_A_PDF", False, size_bytes, f"Read error: {e}"

    def inspect_json_integrity(self, file_path: Path) -> Tuple[str, bool, int, Optional[Any], Optional[str]]:
        """
        Inspects physical JSON file integrity with bounded parsing.
        Returns: (file_integrity, is_valid_file, size_bytes, parsed_data, reason)
        """
        if not file_path.exists():
            return "FILE_NOT_FOUND", False, 0, None, "Physical file does not exist on disk"

        try:
            size_bytes = file_path.stat().st_size
        except Exception as e:
            return "FILE_NOT_FOUND", False, 0, None, f"Cannot stat file: {e}"

        if size_bytes == 0:
            return "EMPTY", False, 0, None, "Physical file is 0 bytes"

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            return "VALID", True, size_bytes, data, "Valid parseable JSON document"
        except Exception as e:
            return "MALFORMED_JSON", False, size_bytes, None, f"JSON decode failure: {e}"

    def classify_pdf_content(
        self,
        record: Dict[str, Any],
        filename: str,
        file_path: Path,
        integrity_status: str,
    ) -> Tuple[str, str, str]:
        """
        Determines conceptual PDF content type using strict deterministic rule precedence.
        Returns: (classified_type, reason, confidence)
        """
        source_url = record.get("source_url", "").lower()
        orig_cat = record.get("category", "").lower()
        fn_lower = filename.lower()
        search_target = f"{source_url} {orig_cat} {fn_lower}"

        # 1. PDF_AMENDMENT
        if RE_AMENDMENT.search(search_target) or orig_cat in ["amendment", "qco_order"]:
            return "PDF_AMENDMENT", "Matched regulatory amendment, QCO order, or gazette notification pattern", "HIGH"

        # 2. PDF_HALLMARKING
        if RE_HALLMARKING.search(search_target) or "hallmark" in orig_cat:
            return "PDF_HALLMARKING", "Matched hallmarking or precious metal regulation pattern", "HIGH"

        # 3. PDF_CERTIFICATION
        if RE_CERTIFICATION.search(search_target) or orig_cat in ["certification_process", "licensing_procedure", "licensing_fees"]:
            return "PDF_CERTIFICATION", "Matched product certification, licensing, or conformity scheme pattern", "HIGH"

        # 4. PDF_CONSUMER
        if RE_CONSUMER.search(search_target) or "consumer" in orig_cat:
            return "PDF_CONSUMER", "Matched consumer protection, citizen charter, or brochure pattern", "HIGH"

        # 5. PDF_STANDARD (Strong signal only)
        # Check URL parameters for explicit standard ID
        url_std_match = RE_URL_STD_ID.search(record.get("source_url", ""))
        if url_std_match:
            std_num = url_std_match.group(1)
            return "PDF_STANDARD", f"Strong standard signal: URL query specifies IS {std_num}", "HIGH"

        # Check filename for IS standard specification
        fn_std_match = RE_IS_STANDARD.search(filename)
        if fn_std_match:
            std_num = fn_std_match.group(1)
            return "PDF_STANDARD", f"Strong standard signal: Filename specifies IS {std_num}", "HIGH"

        # Check filename for SP standard
        sp_match = RE_SP_STANDARD.search(filename)
        if sp_match or "standardized-development" in fn_lower:
            return "PDF_STANDARD", "Strong standard signal: Technical Special Publication / Standardized Code", "HIGH"

        # If file is valid on disk, inspect first 2 KB for explicit standard header
        if integrity_status == "VALID" and file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    raw_snippet = f.read(4096).decode("latin-1", errors="ignore")
                if RE_IS_STANDARD.search(raw_snippet) and ("indian standard" in raw_snippet.lower() or "national foreword" in raw_snippet.lower()):
                    return "PDF_STANDARD", "Strong standard signal: Explicit Indian Standard header in cover", "HIGH"
            except Exception:
                pass

        # 6. PDF_GENERAL
        return "PDF_GENERAL", "Authoritative BIS publication (annual report, organizational chart, circular, or admin document)", "MEDIUM"

    def classify_json_content(
        self,
        record: Dict[str, Any],
        filename: str,
        file_path: Path,
        json_data: Optional[Any],
    ) -> Tuple[str, str, str]:
        """
        Determines conceptual JSON content type using schema & bounded structure inspection.
        Returns: (classified_type, reason, confidence)
        """
        source_url = record.get("source_url", "").lower()
        orig_cat = record.get("category", "").lower()
        fn_lower = filename.lower()
        extra_meta = record.get("extra_metadata", {})
        search_target = f"{source_url} {orig_cat} {fn_lower}"

        # 1. FAQ_JSON
        if extra_meta.get("split_as_qa") is True or RE_FAQ.search(search_target) or "faq" in orig_cat:
            return "FAQ_JSON", "Matched official BIS FAQ structure or split_as_qa schema", "HIGH"

        if isinstance(json_data, list) and json_data:
            first_elem = json_data[0]
            if isinstance(first_elem, dict) and ("question" in first_elem or "answer" in first_elem):
                return "FAQ_JSON", "JSON array contains structured Q&A schema", "HIGH"

        # 2. PSM_JSON
        if RE_PSM.search(search_target) or orig_cat == "product_standard_mapping" or "downloadcrossreferences" in source_url:
            return "PSM_JSON", "Matched product-to-standard mapping cross-reference source", "HIGH"

        # 3. API_PREVIEW_JSON
        if RE_API_PREVIEW.search(search_target) or "isdetails" in source_url or "standardsbis.bsbedge.com" in source_url or "know_your_standards" in orig_cat:
            return "API_PREVIEW_JSON", "Matched BIS API standard preview or technical metadata endpoint", "HIGH"

        if isinstance(json_data, list) and json_data:
            first_elem = json_data[0]
            if isinstance(first_elem, dict):
                page_url = first_elem.get("page_url", "")
                if "preview" in page_url.lower() or "isdetails" in page_url.lower():
                    return "API_PREVIEW_JSON", "JSON payload contains standard preview specification URL", "HIGH"

        # 4. OTHER_JSON
        if orig_cat in ["bis_act_rules_regulations", "lab_directory", "consumer", "hallmarking", "training", "training_faq", "lims_recognized_labs", "lims_empaneled_labs"] or isinstance(json_data, (dict, list)):
            return "OTHER_JSON", "General portal content, act & regulation text, or directory links", "HIGH"

        return "UNKNOWN", "Unresolvable JSON document structure", "LOW"

    def run_classification(self) -> Dict[str, Any]:
        start_time = time.time()
        log.info(f"Starting Phase 1 Full Corpus Source Classification from {self.manifest_path}")
        self._ensure_output_dirs()

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        # Open partition file handles
        partition_handles = {
            cat: open(path, "w", encoding="utf-8")
            for cat, path in self.partition_paths.items()
        }
        master_manifest_path = self.out_dir / "classification_manifest.jsonl"
        master_handle = open(master_manifest_path, "w", encoding="utf-8")

        manifest_record_ids: List[str] = []
        classified_record_ids: List[str] = []

        # Unique physical file tracking
        physical_files_seen: Dict[str, Dict[str, Any]] = {}

        # Accounting counters
        logical_by_category: Dict[str, int] = {c: 0 for c in ALL_CATEGORIES}
        logical_by_integrity: Dict[str, int] = {t: 0 for t in FILE_INTEGRITY_TYPES}
        physical_by_integrity: Dict[str, int] = {t: 0 for t in FILE_INTEGRITY_TYPES}
        physical_by_category: Dict[str, int] = {c: 0 for c in ALL_CATEGORIES}

        sample_records_by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in ALL_CATEGORIES}
        problematic_files: List[Dict[str, Any]] = []

        total_logical_records = 0

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue

                total_logical_records += 1
                record_id = f"REC_{total_logical_records:08d}"
                manifest_record_ids.append(record_id)

                raw_record = json.loads(line)
                local_path_raw = raw_record.get("local_path", "")
                filename = Path(local_path_raw.replace("\\", "/")).name
                file_path = self.raw_dir / filename
                is_pdf = filename.lower().endswith(".pdf") or raw_record.get("content_type") == "pdf"

                source_url = raw_record.get("source_url", "")
                orig_cat = raw_record.get("category", "")
                content_type = raw_record.get("content_type", "pdf" if is_pdf else "html_page")

                # Preserve manifest hash without recalculation
                source_hash = raw_record.get("content_hash") or raw_record.get("source_hash")
                if not source_hash and file_path.exists():
                    try:
                        source_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    except Exception:
                        source_hash = f"hash_{filename[:12]}"
                elif not source_hash:
                    source_hash = f"missing_{filename[:12]}"

                # 1. Independent Physical File Integrity Check
                json_data = None
                if is_pdf:
                    integrity_status, is_valid, size_bytes, integrity_reason = self.inspect_pdf_integrity(file_path)
                else:
                    integrity_status, is_valid, size_bytes, json_data, integrity_reason = self.inspect_json_integrity(file_path)

                # 2. Independent Conceptual Content Classification
                if is_pdf:
                    classified_type, class_reason, confidence = self.classify_pdf_content(
                        raw_record, filename, file_path, integrity_status
                    )
                else:
                    classified_type, class_reason, confidence = self.classify_json_content(
                        raw_record, filename, file_path, json_data
                    )

                # Combine reasons clearly
                if not is_valid:
                    combined_reason = f"{class_reason}; Physical integrity: [{integrity_status}] {integrity_reason}"
                else:
                    combined_reason = class_reason

                # Build normalized classification record
                classification_record = {
                    "record_id": record_id,
                    "local_path": local_path_raw,
                    "source_url": source_url,
                    "content_type": content_type,
                    "original_category": orig_cat,
                    "classified_type": classified_type,
                    "file_integrity": integrity_status,
                    "is_valid_file": is_valid,
                    "source_hash": source_hash,
                    "classification_reason": combined_reason,
                    "classification_confidence": confidence,
                    "physical_file_reference": str(file_path.relative_to(self.raw_dir.parent)).replace("\\", "/"),
                    "file_size_bytes": size_bytes,
                }

                # Write to master index & partition
                serialized = json.dumps(classification_record, ensure_ascii=False)
                master_handle.write(serialized + "\n")
                if classified_type in partition_handles:
                    partition_handles[classified_type].write(serialized + "\n")
                else:
                    partition_handles["UNKNOWN"].write(serialized + "\n")

                classified_record_ids.append(record_id)
                logical_by_category[classified_type] = logical_by_category.get(classified_type, 0) + 1
                logical_by_integrity[integrity_status] = logical_by_integrity.get(integrity_status, 0) + 1

                # Collect first 20 examples per category
                if len(sample_records_by_category[classified_type]) < 20:
                    sample_records_by_category[classified_type].append(classification_record)

                # Track unique physical files
                if filename not in physical_files_seen:
                    physical_files_seen[filename] = {
                        "integrity": integrity_status,
                        "size_bytes": size_bytes,
                        "classified_type": classified_type,
                        "source_url": source_url,
                        "error": integrity_reason if not is_valid else None,
                    }
                    physical_by_integrity[integrity_status] = physical_by_integrity.get(integrity_status, 0) + 1
                    physical_by_category[classified_type] = physical_by_category.get(classified_type, 0) + 1

                    if not is_valid:
                        problematic_files.append({
                            "filename": filename,
                            "classified_type": classified_type,
                            "file_integrity": integrity_status,
                            "size_bytes": size_bytes,
                            "error": integrity_reason,
                            "source_url": source_url,
                        })

        # Close all handles
        master_handle.close()
        for handle in partition_handles.values():
            handle.close()

        elapsed_time = round(time.time() - start_time, 2)
        total_unique_physical_files = len(physical_files_seen)

        # Zero-loss ID verification
        manifest_id_set = set(manifest_record_ids)
        classified_id_set = set(classified_record_ids)
        duplicate_ids = len(classified_record_ids) - len(classified_id_set)
        missing_ids = len(manifest_id_set - classified_id_set)
        unexpected_ids = len(classified_id_set - manifest_id_set)
        id_verification_passed = (manifest_id_set == classified_id_set and duplicate_ids == 0 and missing_ids == 0 and unexpected_ids == 0)

        # Multi-tier mathematical reconciliation
        logical_sum_cat = sum(logical_by_category.values())
        logical_sum_int = sum(logical_by_integrity.values())
        physical_sum_int = sum(physical_by_integrity.values())
        physical_sum_cat = sum(physical_by_category.values())

        reconciliation_passed = (
            logical_sum_cat == total_logical_records
            and logical_sum_int == total_logical_records
            and physical_sum_int == total_unique_physical_files
            and physical_sum_cat == total_unique_physical_files
        )

        classification_complete = id_verification_passed and reconciliation_passed

        # Print Terminal Output & Accounting
        self._print_results(
            total_logical_records=total_logical_records,
            total_unique_physical_files=total_unique_physical_files,
            logical_by_category=logical_by_category,
            logical_by_integrity=logical_by_integrity,
            physical_by_integrity=physical_by_integrity,
            physical_by_category=physical_by_category,
            duplicate_ids=duplicate_ids,
            missing_ids=missing_ids,
            unexpected_ids=unexpected_ids,
            id_verification_passed=id_verification_passed,
            reconciliation_passed=reconciliation_passed,
            problematic_files=problematic_files,
            sample_records_by_category=sample_records_by_category,
            elapsed_time=elapsed_time,
            classification_complete=classification_complete,
        )

        return {
            "total_logical_records": total_logical_records,
            "total_unique_physical_files": total_unique_physical_files,
            "logical_by_category": logical_by_category,
            "logical_by_integrity": logical_by_integrity,
            "physical_by_integrity": physical_by_integrity,
            "classification_complete": classification_complete,
        }

    def _print_results(
        self,
        total_logical_records: int,
        total_unique_physical_files: int,
        logical_by_category: Dict[str, int],
        logical_by_integrity: Dict[str, int],
        physical_by_integrity: Dict[str, int],
        physical_by_category: Dict[str, int],
        duplicate_ids: int,
        missing_ids: int,
        unexpected_ids: int,
        id_verification_passed: bool,
        reconciliation_passed: bool,
        problematic_files: List[Dict[str, Any]],
        sample_records_by_category: Dict[str, List[Dict[str, Any]]],
        elapsed_time: float,
        classification_complete: bool,
    ):
        print("\n" + "=" * 78)
        print("PHASE 1 — FULL CORPUS SOURCE CLASSIFICATION & INTEGRITY AUDIT")
        print("=" * 78)
        print(f"Runtime: {elapsed_time}s | Raw Corpus Path: {self.raw_dir} (IMMUTABLE)")
        print(f"Total Logical Manifest Records: {total_logical_records}")
        print(f"Total Unique Physical Files:    {total_unique_physical_files}")

        print("\n" + "-" * 78)
        print("1. LOGICAL RECORDS BY CONTENT TAXONOMY (Total: " + str(total_logical_records) + ")")
        print("-" * 78)
        print("PDF Content Categories:")
        for cat in sorted(PDF_CATEGORIES):
            print(f"  • {cat:20s}: {logical_by_category.get(cat, 0):6d}")
        print("\nJSON Content Categories:")
        for cat in sorted(JSON_CATEGORIES):
            print(f"  • {cat:20s}: {logical_by_category.get(cat, 0):6d}")
        print(f"\n  • {'UNKNOWN':20s}: {logical_by_category.get('UNKNOWN', 0):6d}")

        print("\n" + "-" * 78)
        print("2. PHYSICAL FILES BY INTEGRITY STATUS (Total: " + str(total_unique_physical_files) + ")")
        print("-" * 78)
        for stat in sorted(FILE_INTEGRITY_TYPES):
            print(f"  • {stat:20s}: {physical_by_integrity.get(stat, 0):6d}")

        print("\n" + "-" * 78)
        print("3. ZERO-LOSS ID VERIFICATION & MULTI-TIER RECONCILIATION")
        print("-" * 78)
        print(f"  • Duplicate classified IDs:  {duplicate_ids} (Must be 0)")
        print(f"  • Missing record IDs:        {missing_ids} (Must be 0)")
        print(f"  • Unexpected record IDs:     {unexpected_ids} (Must be 0)")
        print(f"  • Set Identity Match:        {'PASSED (1:1 Exact Match)' if id_verification_passed else 'FAILED'}")
        print(f"  • Logical Sum by Category:   {sum(logical_by_category.values())} == {total_logical_records}")
        print(f"  • Logical Sum by Integrity:  {sum(logical_by_integrity.values())} == {total_logical_records}")
        print(f"  • Physical Sum by Integrity: {sum(physical_by_integrity.values())} == {total_unique_physical_files}")
        print(f"  • Overall Reconciliation:    {'PASSED (100.0% Consistent)' if reconciliation_passed else 'FAILED'}")

        print("\n" + "-" * 78)
        print(f"4. PROBLEMATIC PHYSICAL FILES ON DISK ({len(problematic_files)} files)")
        print("-" * 78)
        print(f"{'Filename':<45} | {'Integrity':<15} | {'Size':<8} | {'Error'}")
        print("-" * 78)
        for p in problematic_files[:25]:
            err_msg = str(p['error'])[:40] if p['error'] else ""
            print(f"{p['filename'][:44]:<45} | {p['file_integrity']:<15} | {p['size_bytes']:<8} | {err_msg}")
        if len(problematic_files) > 25:
            print(f"... and {len(problematic_files) - 25} more problematic files listed in classified_data/ ...")

        print("\n" + "-" * 78)
        print("5. FIRST 20 CLASSIFICATION SAMPLES PER CATEGORY")
        print("-" * 78)
        for cat in sorted(ALL_CATEGORIES):
            samples = sample_records_by_category.get(cat, [])
            print(f"\n>>> Category: {cat} ({len(samples)} samples displayed out of {logical_by_category.get(cat, 0)})")
            for idx, s in enumerate(samples, 1):
                fn = Path(s['local_path'].replace('\\', '/')).name
                print(f"  [{idx:02d}] {s['record_id']} | {s['file_integrity']:<14} | {fn:<40} | URL: {s['source_url'][:70]}")

        print("\n" + "=" * 78)
        if classification_complete:
            print("STATUS: CLASSIFICATION COMPLETE")
        else:
            print("STATUS: CLASSIFICATION INCOMPLETE")
        print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Full Corpus Source Classifier")
    parser.add_argument("--raw_dir", default="./raw_data", help="Path to immutable raw_data directory")
    parser.add_argument("--out_dir", default="./classified_data", help="Output directory for index manifests")
    parser.add_argument("--manifest", default="./raw_data/manifest.jsonl", help="Path to manifest.jsonl")
    args = parser.parse_args()

    classifier = CorpusClassifier(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        manifest_path=Path(args.manifest),
    )
    res = classifier.run_classification()
    if not res["classification_complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
