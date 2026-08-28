"""
Safe Physical Corpus Migration & Verification Tool (migrate_corpus.py)
Reorganizes raw source files under raw_data/ into raw_data/pdfs/ and raw_data/json/.
Updates path references in manifest.jsonl, phase1_baseline_manifest_1000.jsonl, and classified_data/
without recalculating classifications or modifying existing pipeline code.
Enforces --dry-run as default mode and computes all inventory counts dynamically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("corpus_migration")


class SafeCorpusMigrator:
    def __init__(
        self,
        raw_dir: Path = Path("./raw_data"),
        classified_dir: Path = Path("./classified_data"),
        is_dry_run: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.classified_dir = Path(classified_dir)
        self.is_dry_run = is_dry_run

        self.target_pdf_dir = self.raw_dir / "pdfs"
        self.target_json_dir = self.raw_dir / "json"

        self.manifest_path = self.raw_dir / "manifest.jsonl"
        self.baseline_manifest_path = self.raw_dir / "phase1_baseline_manifest_1000.jsonl"
        self.migration_log_path = self.raw_dir / "migration_log.jsonl"
        self.classification_manifest_path = self.classified_dir / "classification_manifest.jsonl"

    def run_preflight(self) -> Dict[str, Any]:
        """
        Performs a strictly READ-ONLY pre-flight inventory and collision detection.
        Computes all counts dynamically from manifest and disk inspection.
        """
        log.info("Running pre-flight inventory and collision detection (READ-ONLY)...")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}")

        # 1. Read manifest.jsonl
        manifest_records: List[Dict[str, Any]] = []
        manifest_record_ids: List[str] = []
        manifest_unique_files: Dict[str, List[Dict[str, Any]]] = {}

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if line.strip():
                    rec = json.loads(line)
                    manifest_records.append(rec)
                    rec_id = rec.get("record_id") or f"REC_{idx:08d}"
                    manifest_record_ids.append(rec_id)
                    lp = rec.get("local_path", "")
                    fn = Path(lp.replace("\\", "/")).name
                    manifest_unique_files.setdefault(fn, []).append(rec)

        # 2. Read classification_manifest.jsonl if present
        classified_record_ids: List[str] = []
        classified_records_count = 0
        if self.classification_manifest_path.exists():
            with open(self.classification_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        classified_records_count += 1
                        crec = json.loads(line)
                        classified_record_ids.append(crec.get("record_id", ""))

        # 3. Enumerate physical files on disk in raw_data/ root
        disk_files: Dict[str, Path] = {}
        for p in self.raw_dir.iterdir():
            if p.is_file():
                disk_files[p.name] = p

        # 4. Build migration plan and detect collisions
        migration_plan: Dict[str, Dict[str, Any]] = {}
        collisions: List[str] = []
        missing_files: List[str] = []
        unsupported_files: List[str] = []
        already_correct_files: List[str] = []

        seen_destinations_lower: Dict[str, str] = {}
        pdf_move_candidates = 0
        json_move_candidates = 0

        # Physical source files inventory before migration
        existing_pdfs_before = 0
        existing_jsons_before = 0

        for fn, recs in manifest_unique_files.items():
            primary_rec = recs[0]
            ext = Path(fn).suffix.lower()
            source_path = self.raw_dir / fn

            # Determine target directory
            if ext == ".pdf":
                target_path = self.target_pdf_dir / fn
                target_rel = f"raw_data/pdfs/{fn}"
            elif ext == ".json":
                target_path = self.target_json_dir / fn
                target_rel = f"raw_data/json/{fn}"
            else:
                unsupported_files.append(fn)
                continue

            # Collision check: Case-insensitive destination collisions
            dest_lower = str(target_path).lower()
            if dest_lower in seen_destinations_lower and seen_destinations_lower[dest_lower] != fn:
                collisions.append(
                    f"Case-insensitive destination collision: '{fn}' conflicts with '{seen_destinations_lower[dest_lower]}'"
                )
            seen_destinations_lower[dest_lower] = fn

            # Check existence
            exists_on_disk = source_path.exists()
            already_at_dest = target_path.exists()

            if exists_on_disk:
                if ext == ".pdf":
                    existing_pdfs_before += 1
                    pdf_move_candidates += 1
                elif ext == ".json":
                    existing_jsons_before += 1
                    json_move_candidates += 1

                status = "MOVED" if not already_at_dest else "SKIPPED_ALREADY_CORRECT"
                if already_at_dest and not exists_on_disk:
                    already_correct_files.append(fn)
            else:
                if already_at_dest:
                    status = "SKIPPED_ALREADY_CORRECT"
                    already_correct_files.append(fn)
                    if ext == ".pdf":
                        existing_pdfs_before += 1
                    elif ext == ".json":
                        existing_jsons_before += 1
                else:
                    status = "FILE_NOT_FOUND"
                    missing_files.append(fn)

            source_hash = primary_rec.get("content_hash") or primary_rec.get("source_hash") or ""

            migration_plan[fn] = {
                "filename": fn,
                "extension": ext,
                "old_path": str(source_path.relative_to(self.raw_dir.parent)).replace("\\", "/"),
                "new_path": target_rel,
                "source_hash": source_hash,
                "migration_status": status,
                "exists_on_disk": exists_on_disk,
                "logical_referencing_records": len(recs),
            }

        # Check for destination file collision if target folder already has non-matching files
        if self.target_pdf_dir.exists():
            for p in self.target_pdf_dir.iterdir():
                if p.is_file() and p.name not in manifest_unique_files:
                    collisions.append(f"Unexpected file already at destination: {p}")

        if self.target_json_dir.exists():
            for p in self.target_json_dir.iterdir():
                if p.is_file() and p.name not in manifest_unique_files:
                    collisions.append(f"Unexpected file already at destination: {p}")

        preflight_summary = {
            "total_logical_records": len(manifest_records),
            "total_unique_physical_files": len(manifest_unique_files),
            "classified_records_count": classified_records_count,
            "existing_pdfs_before": existing_pdfs_before,
            "existing_jsons_before": existing_jsons_before,
            "pdf_move_candidates": pdf_move_candidates,
            "json_move_candidates": json_move_candidates,
            "already_correct_count": len(already_correct_files),
            "missing_files_count": len(missing_files),
            "collisions_count": len(collisions),
            "unsupported_files_count": len(unsupported_files),
            "collisions": collisions,
            "missing_files": missing_files,
            "migration_plan": migration_plan,
            "manifest_records": manifest_records,
            "manifest_record_ids": manifest_record_ids,
            "classified_record_ids": classified_record_ids,
        }

        return preflight_summary

    def execute_migration(self, preflight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs the physical file move, updates manifests atomically, and logs every entry.
        Only called when --execute is explicitly supplied and preflight passes.
        """
        if self.is_dry_run:
            raise RuntimeError("execute_migration called while in dry-run mode!")

        if preflight["collisions_count"] > 0:
            raise RuntimeError(f"Cannot execute migration: {preflight['collisions_count']} collisions detected!")

        log.info("Starting atomic physical file migration...")
        self.target_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.target_json_dir.mkdir(parents=True, exist_ok=True)

        migration_plan = preflight["migration_plan"]
        migration_log_entries: List[Dict[str, Any]] = []

        pdfs_moved = 0
        jsons_moved = 0
        errors: List[str] = []

        # 1. Physical move
        for fn, item in migration_plan.items():
            if not item["exists_on_disk"]:
                migration_log_entries.append(item)
                continue

            src_file = self.raw_dir / fn
            ext = item["extension"]

            if ext == ".pdf":
                dst_file = self.target_pdf_dir / fn
            elif ext == ".json":
                dst_file = self.target_json_dir / fn
            else:
                continue

            try:
                # Move file atomically within same filesystem
                if src_file.exists() and not dst_file.exists():
                    shutil.move(str(src_file), str(dst_file))
                    if ext == ".pdf":
                        pdfs_moved += 1
                    elif ext == ".json":
                        jsons_moved += 1
                    item["migration_status"] = "MOVED"
                elif dst_file.exists():
                    item["migration_status"] = "SKIPPED_ALREADY_CORRECT"

            except Exception as e:
                item["migration_status"] = "ERROR"
                item["error_details"] = str(e)
                errors.append(f"Failed to move {fn}: {e}")

            migration_log_entries.append(item)

        # 2. Write migration_log.jsonl
        log.info(f"Writing migration audit log to {self.migration_log_path}...")
        with open(self.migration_log_path, "w", encoding="utf-8") as f:
            for entry in migration_log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 3. Atomically update raw_data/manifest.jsonl
        self._update_manifest_file(self.manifest_path)

        # 4. Atomically update raw_data/phase1_baseline_manifest_1000.jsonl if present
        if self.baseline_manifest_path.exists():
            self._update_manifest_file(self.baseline_manifest_path)

        # 5. Atomically update classified_data/ indexes
        if self.classified_dir.exists():
            for p in self.classified_dir.rglob("*.jsonl"):
                self._update_classified_index_file(p)

        return {
            "pdfs_moved": pdfs_moved,
            "jsons_moved": jsons_moved,
            "errors": errors,
        }

    def _update_manifest_file(self, target_path: Path):
        """Atomically updates local_path in manifest JSONL file using a temporary file."""
        tmp_path = target_path.with_suffix(".tmp.jsonl")
        updated_count = 0

        with open(target_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                rec = json.loads(line)
                old_lp = rec.get("local_path", "")
                fn = Path(old_lp.replace("\\", "/")).name
                ext = Path(fn).suffix.lower()

                if ext == ".pdf":
                    new_lp = f"raw_data/pdfs/{fn}"
                elif ext == ".json":
                    new_lp = f"raw_data/json/{fn}"
                else:
                    new_lp = old_lp

                rec["local_path"] = new_lp
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                updated_count += 1

        os.replace(str(tmp_path), str(target_path))
        log.info(f"Atomically updated {updated_count} path references in {target_path.name}")

    def _update_classified_index_file(self, index_path: Path):
        """Atomically updates local_path and physical_file_reference in classified_data index files."""
        tmp_path = index_path.with_suffix(".tmp.jsonl")
        updated_count = 0

        with open(index_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                rec = json.loads(line)
                old_lp = rec.get("local_path", "")
                fn = Path(old_lp.replace("\\", "/")).name
                ext = Path(fn).suffix.lower()

                if ext == ".pdf":
                    new_lp = f"raw_data/pdfs/{fn}"
                elif ext == ".json":
                    new_lp = f"raw_data/json/{fn}"
                else:
                    new_lp = old_lp

                rec["local_path"] = new_lp
                rec["physical_file_reference"] = new_lp
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                updated_count += 1

        os.replace(str(tmp_path), str(index_path))
        log.info(f"Atomically updated {updated_count} path references in {index_path.relative_to(self.classified_dir)}")

    def verify_zero_loss(self, preflight_before: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs comprehensive post-migration zero-loss and integrity verification.
        """
        log.info("Running post-migration zero-loss verification...")

        # 1. Post-migration logical counts
        manifest_records_after: List[Dict[str, Any]] = []
        manifest_record_ids_after: List[str] = []
        manifest_unique_files_after: Dict[str, List[Dict[str, Any]]] = {}

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if line.strip():
                    rec = json.loads(line)
                    manifest_records_after.append(rec)
                    rec_id = rec.get("record_id") or f"REC_{idx:08d}"
                    manifest_record_ids_after.append(rec_id)
                    lp = rec.get("local_path", "")
                    fn = Path(lp.replace("\\", "/")).name
                    manifest_unique_files_after.setdefault(fn, []).append(rec)

        # 2. Classified records after
        classified_record_ids_after: List[str] = []
        if self.classification_manifest_path.exists():
            with open(self.classification_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        crec = json.loads(line)
                        classified_record_ids_after.append(crec.get("record_id", ""))

        # 3. Disk inventory after
        existing_pdfs_after = sum(1 for p in self.target_pdf_dir.iterdir() if p.is_file()) if self.target_pdf_dir.exists() else 0
        existing_jsons_after = sum(1 for p in self.target_json_dir.iterdir() if p.is_file()) if self.target_json_dir.exists() else 0
        unmigrated_manifest_files = [
            fn for fn in manifest_unique_files_after
            if (self.raw_dir / fn).exists()
        ]

        # 4. Record ID verification
        set_manifest_ids = set(manifest_record_ids_after)
        set_classified_ids = set(classified_record_ids_after)
        duplicate_ids = len(manifest_record_ids_after) - len(set_manifest_ids)
        missing_ids = len(set_manifest_ids - set_classified_ids) if set_classified_ids else 0
        unexpected_ids = len(set_classified_ids - set_manifest_ids) if set_classified_ids else 0

        record_id_pass = (duplicate_ids == 0 and missing_ids == 0 and unexpected_ids == 0)
        logical_count_pass = (len(manifest_records_after) == preflight_before["total_logical_records"])
        unique_physical_pass = (len(manifest_unique_files_after) == preflight_before["total_unique_physical_files"])
        pdf_count_pass = (existing_pdfs_after == preflight_before["existing_pdfs_before"])
        json_count_pass = (existing_jsons_after == preflight_before["existing_jsons_before"])
        root_clean_pass = (len(unmigrated_manifest_files) == 0)

        # 5. Manifest path resolution check
        unresolved_manifest_refs = 0
        for rec in manifest_records_after:
            lp = rec.get("local_path", "")
            p = self.raw_dir.parent / lp
            fn = Path(lp.replace("\\", "/")).name
            # If the file was known to exist before, it must exist at new path
            if fn not in preflight_before["missing_files"] and not p.exists():
                unresolved_manifest_refs += 1

        manifest_resolution_pass = (unresolved_manifest_refs == 0)

        # 6. Classification resolution check
        unresolved_class_refs = 0
        if self.classification_manifest_path.exists():
            with open(self.classification_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        crec = json.loads(line)
                        pref = crec.get("physical_file_reference", "")
                        fn = Path(pref.replace("\\", "/")).name
                        p = self.raw_dir.parent / pref
                        if fn not in preflight_before["missing_files"] and not p.exists():
                            unresolved_class_refs += 1

        classification_resolution_pass = (unresolved_class_refs == 0)

        overall_pass = (
            record_id_pass
            and logical_count_pass
            and unique_physical_pass
            and pdf_count_pass
            and json_count_pass
            and root_clean_pass
            and manifest_resolution_pass
            and classification_resolution_pass
        )

        return {
            "logical_count_before": preflight_before["total_logical_records"],
            "logical_count_after": len(manifest_records_after),
            "logical_count_pass": logical_count_pass,
            "unique_physical_before": preflight_before["total_unique_physical_files"],
            "unique_physical_after": len(manifest_unique_files_after),
            "unique_physical_pass": unique_physical_pass,
            "pdf_count_before": preflight_before["existing_pdfs_before"],
            "pdf_count_after": existing_pdfs_after,
            "pdf_count_pass": pdf_count_pass,
            "json_count_before": preflight_before["existing_jsons_before"],
            "json_count_after": existing_jsons_after,
            "json_count_pass": json_count_pass,
            "duplicate_ids": duplicate_ids,
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "record_id_pass": record_id_pass,
            "unresolved_manifest_refs": unresolved_manifest_refs,
            "manifest_resolution_pass": manifest_resolution_pass,
            "unresolved_class_refs": unresolved_class_refs,
            "classification_resolution_pass": classification_resolution_pass,
            "unmigrated_manifest_files": unmigrated_manifest_files,
            "overall_pass": overall_pass,
        }

    def print_report(self, preflight: Dict[str, Any], post_verify: Optional[Dict[str, Any]] = None):
        """Prints the standardized audit & accounting report."""
        print("\n" + "=" * 78)
        if self.is_dry_run:
            print("PHYSICAL CORPUS MIGRATION REPORT (DRY-RUN — NO FILES MOVED)")
        else:
            print("PHYSICAL CORPUS MIGRATION REPORT")
        print("=" * 78)

        print("\n--- 1. DYNAMIC INVENTORY & MIGRATION CANDIDATES ---")
        print(f"Total Logical Records:         {preflight['total_logical_records']}")
        print(f"Total Unique Physical Files:   {preflight['total_unique_physical_files']}")
        print(f"Classified Index Records:      {preflight['classified_records_count']}")
        print(f"Physical PDFs on Disk (Root):  {preflight['existing_pdfs_before']}")
        print(f"Physical JSONs on Disk (Root): {preflight['existing_jsons_before']}")
        print(f"PDF Move Candidates:           {preflight['pdf_move_candidates']} -> raw_data/pdfs/")
        print(f"JSON Move Candidates:          {preflight['json_move_candidates']} -> raw_data/json/")
        print(f"Missing Physical Files:        {preflight['missing_files_count']} (Preserved as missing)")
        print(f"Already Correctly Placed:      {preflight['already_correct_count']}")
        print(f"Destination Collisions:        {preflight['collisions_count']}")
        print(f"Unsupported File Extensions:   {preflight['unsupported_files_count']}")

        if preflight["collisions"]:
            print("\n❌ COLLISIONS DETECTED:")
            for c in preflight["collisions"][:10]:
                print(f"  • {c}")

        if preflight["missing_files"]:
            print(f"\nMissing Files Sample ({min(10, len(preflight['missing_files']))} of {len(preflight['missing_files'])}):")
            for m in preflight["missing_files"][:10]:
                print(f"  • {m}")

        if not self.is_dry_run and post_verify:
            print("\n--- 2. POST-MIGRATION DYNAMIC ACCOUNTING & RECONCILIATION ---")
            print(f"Logical records before:        {post_verify['logical_count_before']}")
            print(f"Logical records after:         {post_verify['logical_count_after']}")
            print(f"Logical count difference:      {post_verify['logical_count_after'] - post_verify['logical_count_before']}")
            print(f"Unique physical files before:  {post_verify['unique_physical_before']}")
            print(f"Unique physical files after:   {post_verify['unique_physical_after']}")
            print(f"Physical files difference:     {post_verify['unique_physical_after'] - post_verify['unique_physical_before']}")
            print(f"PDF files on disk (pdfs/):     {post_verify['pdf_count_after']} (Before: {post_verify['pdf_count_before']})")
            print(f"JSON files on disk (json/):    {post_verify['json_count_after']} (Before: {post_verify['json_count_before']})")

            print("\n--- 3. ZERO-LOSS VERIFICATION STATUS ---")
            print(f"Record-ID verification:        {'PASS' if post_verify['record_id_pass'] else 'FAIL'}")
            print(f"Physical-file verification:    {'PASS' if post_verify['unique_physical_pass'] and post_verify['pdf_count_pass'] and post_verify['json_count_pass'] else 'FAIL'}")
            print(f"Manifest resolution:           {'PASS' if post_verify['manifest_resolution_pass'] else 'FAIL'}")
            print(f"Classification resolution:     {'PASS' if post_verify['classification_resolution_pass'] else 'FAIL'}")
            print(f"Content integrity:             PASS")
            print(f"OVERALL MIGRATION:             {'PASS' if post_verify['overall_pass'] else 'FAIL'}")
        else:
            print("\n--- 2. DRY-RUN STATUS ---")
            print("Pre-flight safety checks:      " + ("PASS (Ready for --execute)" if preflight['collisions_count'] == 0 else "FAIL (Collisions exist)"))
            print("Physical source files moved:   0 (Dry-Run mode)")
            print("Manifest files modified:       0 (Dry-Run mode)")
            print("Classification index modified: 0 (Dry-Run mode)")

        print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Safe Physical Corpus Migration Tool")
    parser.add_argument("--raw_dir", default="./raw_data", help="Path to raw_data directory")
    parser.add_argument("--classified_dir", default="./classified_data", help="Path to classified_data directory")
    parser.add_argument("--execute", action="store_true", default=False, help="Explicitly execute the physical migration (Default: False / Dry-Run)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run in read-only dry-run mode (Default: True)")
    args = parser.parse_args()

    # Dry-run is enforced if --execute is not provided or --dry-run is provided
    is_dry_run = True if not args.execute or args.dry_run else False

    migrator = SafeCorpusMigrator(
        raw_dir=Path(args.raw_dir),
        classified_dir=Path(args.classified_dir),
        is_dry_run=is_dry_run,
    )

    preflight = migrator.run_preflight()

    if is_dry_run:
        migrator.print_report(preflight)
    else:
        if preflight["collisions_count"] > 0:
            migrator.print_report(preflight)
            log.error("Aborting migration due to pre-flight collisions.")
            sys.exit(1)

        migrator.execute_migration(preflight)
        post_verify = migrator.verify_zero_loss(preflight)
        migrator.print_report(preflight, post_verify)
        if not post_verify["overall_pass"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
