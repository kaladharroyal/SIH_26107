"""
BIS PDF Corpus Parser & Ingestion Pipeline - Phase 1 Data Foundation (pdf_parser.py)
Orchestrates physical PDF corpus discovery from raw_data/pdfs/, layout-aware clause extraction,
real-time periodic checkpointing, change detection, atomic output writing, strict validation,
and deterministic chunk generation for the full BIS PDF corpus.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set

from pypdf import PdfReader

# Ensure ingestion package is in path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(src_dir / "ingestion") not in sys.path:
    sys.path.insert(0, str(src_dir / "ingestion"))

from metadata import (
    ChunkRecord,
    ALLOWED_CATEGORIES,
    ALLOWED_PRODUCTION_SOURCE_OF_TRUTH,
    ALLOWED_IDENTITY_STATUS,
    generate_canonical_chunk_id,
)
from validator import QuarantineLogger, IngestionValidator
from deduplicator import Deduplicator
from pdf_ingestor import PDFIngestor
from faq_ingestor import FAQIngestor
from preview_ingestor import PreviewIngestor
from structured_ingestor import StructuredIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("bis_pdf_parser")


@dataclass
class Clause:
    """Backward-compatible Clause dataclass supporting existing callers while providing new Phase 1 fields."""
    is_number: Optional[str]
    part: Optional[str]
    revision_year: Optional[str]
    clause_number: str
    clause_title: Optional[str]
    text: str
    page_start: int
    page_end: int
    source_file: str
    category: str = "general_policy"
    source_url: str = ""
    source_of_truth: str = "verified_bis_pdf"
    identity_status: str = "verified"
    identity_reason: str = ""
    chunk_id: str = ""


class IngestionPipeline:
    def __init__(
        self,
        raw_dir: Path = Path("./raw_data"),
        pdf_dir: Optional[Path] = None,
        out_path: Path = Path("./processed_chunks.jsonl"),
        manifest_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        mode: str = "full",
        dataset_name: Optional[str] = None,
        max_limit: int = 1000,
        max_workers: int = 8,
    ):
        self.raw_dir = Path(raw_dir)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else (self.raw_dir / "pdfs" if (self.raw_dir / "pdfs").exists() else self.raw_dir)
        self.out_path = Path(out_path)
        self.mode = mode.lower()
        self.max_workers = max_workers
        self.max_limit = max_limit

        self.quarantine_path = self.raw_dir / "quarantine_log.jsonl"
        self.quarantine_logger = QuarantineLogger(self.quarantine_path)
        self.validator = IngestionValidator(self.quarantine_logger)
        self.deduplicator = Deduplicator()
        self.pdf_ingestor = PDFIngestor(self.validator)

        if self.mode == "full":
            self.dataset_name = dataset_name or "PHASE1_FULL_PDF_CORPUS"
            self.manifest_path = manifest_path or (self.raw_dir / "manifest.jsonl")
            self.checkpoint_path = checkpoint_path or (self.raw_dir / "ingestion_checkpoint_full_pdf.json")
            self.faq_ingestor = None
            self.preview_ingestor = None
            self.structured_ingestor = None
        else:
            self.dataset_name = dataset_name or "PHASE1_BASELINE_CORPUS_1000"
            self.manifest_path = manifest_path or (self.raw_dir / "phase1_baseline_manifest_1000.jsonl")
            if not self.manifest_path.exists():
                self.manifest_path = self.raw_dir / "manifest.jsonl"
            self.checkpoint_path = checkpoint_path or (self.raw_dir / "ingestion_checkpoint_1000.json")
            self.faq_ingestor = FAQIngestor(self.validator)
            self.preview_ingestor = PreviewIngestor(self.validator)
            self.structured_ingestor = StructuredIngestor(self.raw_dir, self.out_path.parent)

        self.report_path = self.out_path.parent / "phase1_data_quality_report.json"
        self.checkpoint_cache = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict[str, Any]:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Error reading checkpoint: {e}")
        return {}

    def _save_checkpoint_atomic(self, cache: Dict[str, Any]):
        """Atomically writes checkpoint to disk to survive interruptions without corruption."""
        tmp_checkpoint = self.checkpoint_path.with_suffix(".tmp.json")
        try:
            with open(tmp_checkpoint, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            os.replace(str(tmp_checkpoint), str(self.checkpoint_path))
        except Exception as e:
            log.warning(f"Error writing checkpoint: {e}")

    def load_manifest_metadata_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads manifest records into an in-memory dictionary keyed by filename for provenance matching.
        Does NOT restrict the population to manifest records when running in full PDF mode.
        """
        manifest_by_file: Dict[str, List[Dict[str, Any]]] = {}
        if not self.manifest_path.exists():
            log.info(f"Manifest not found at {self.manifest_path}; provenance will use file:// fallback")
            return manifest_by_file

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        lp = entry.get("local_path", "")
                        fn = Path(lp.replace("\\", "/")).name
                        manifest_by_file.setdefault(fn, []).append(entry)
                    except Exception as e:
                        log.warning(f"Error reading manifest line: {e}")

        log.info(f"Loaded provenance metadata for {len(manifest_by_file)} unique files from {self.manifest_path.name}")
        return manifest_by_file

    def validate_pdf_file(self, pdf_path: Path) -> Tuple[str, bool, str]:
        """
        Performs read-only validation of physical PDF integrity before parsing.
        Returns: (integrity_status, is_valid, reason)
        """
        if not pdf_path.exists():
            return "FILE_NOT_FOUND", False, "File does not exist on disk"

        try:
            sz = pdf_path.stat().st_size
        except Exception as e:
            return "FILE_NOT_FOUND", False, f"Cannot stat file: {e}"

        if sz == 0:
            return "EMPTY", False, "File size is 0 bytes"

        try:
            with open(pdf_path, "rb") as f:
                header = f.read(1024)

            if header.startswith(b"\r\n\r\n<!DOCTYPE html") or header.startswith(b"<!DOCTYPE html") or b"<html" in header[:200].lower():
                return "NOT_A_PDF", False, "File contains HTML web server response instead of binary PDF"

            if not (header.startswith(b"%PDF-") or b"%PDF-" in header[:1024]):
                return "NOT_A_PDF", False, "Missing %PDF- header signature"

            reader = PdfReader(str(pdf_path))
            if len(reader.pages) == 0:
                return "EMPTY", False, "PDF contains 0 pages"

            return "VALID", True, f"Valid PDF with {len(reader.pages)} pages"
        except Exception as e:
            return "CORRUPT", False, f"PDF reader error: {e}"

    def run_full_pdf_corpus(self) -> List[ChunkRecord]:
        """
        Executes Full PDF Corpus Parsing directly from raw_data/pdfs/.
        Bypasses baseline limits and processes zero JSON files.
        """
        start_time = time.time()
        log.info(f"Starting full PDF corpus ingestion from {self.pdf_dir}")

        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory does not exist: {self.pdf_dir}")

        # 1. Discover all physical PDF files dynamically
        pdf_paths = sorted(list(self.pdf_dir.rglob("*.pdf")))
        total_discovered_pdfs = len(pdf_paths)
        log.info(f"Discovered {total_discovered_pdfs} physical PDF files in {self.pdf_dir}")

        if total_discovered_pdfs == 0:
            log.warning(f"No PDF files found in {self.pdf_dir}")
            return []

        # 2. Load provenance metadata mapping from full manifest
        manifest_by_file = self.load_manifest_metadata_map()

        all_chunks: List[ChunkRecord] = []
        new_checkpoint: Dict[str, Any] = dict(self.checkpoint_cache)

        processed_pdf_files_count = 0
        quarantined_pdf_files_count = 0
        skipped_pdf_files_count = 0
        total_raw_chunks_count = 0

        # 3. Threaded PDF parsing worker function
        def process_single_physical_pdf(pdf_path: Path) -> Tuple[str, List[ChunkRecord], str, str, bool]:
            fn = pdf_path.name
            entries = manifest_by_file.get(fn, [{}])
            primary_entry = entries[0]

            # Content hash computation
            file_hash = primary_entry.get("content_hash") or primary_entry.get("source_hash")
            if not file_hash and pdf_path.exists():
                try:
                    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
                except Exception:
                    file_hash = f"hash_{fn[:12]}"
            elif not file_hash:
                file_hash = f"missing_{fn[:12]}"

            # Validation gate
            status, is_valid, reason = self.validate_pdf_file(pdf_path)
            if not is_valid:
                src_url = primary_entry.get("source_url", f"file:///{fn}")
                self.quarantine_logger.log_quarantine(str(pdf_path), src_url, status, reason)
                return fn, [], file_hash, status, False

            # Checkpoint cache check
            cached = self.checkpoint_cache.get(fn)
            if cached and cached.get("hash") == file_hash and "chunks" in cached:
                cached_chunks = [ChunkRecord(**c) for c in cached["chunks"]]
                return fn, cached_chunks, file_hash, "CACHED", True

            # Ingest chunks for each logical reference (or single file entry)
            file_chunks: List[ChunkRecord] = []
            for ent in entries:
                extracted = self.pdf_ingestor.ingest_pdf(pdf_path, ent)
                file_chunks.extend(extracted)

            return fn, file_chunks, file_hash, "PARSED", True

        # 4. Multi-threaded execution with periodic checkpointing
        log.info(f"Ingesting {total_discovered_pdfs} PDF files using {self.max_workers} worker threads...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_physical_pdf, p): p
                for p in pdf_paths
            }

            for completed_idx, future in enumerate(as_completed(future_to_file), 1):
                p = future_to_file[future]
                fn = p.name
                try:
                    filename, chunk_list, file_hash, status, is_valid = future.result()

                    if not is_valid:
                        quarantined_pdf_files_count += 1
                    elif chunk_list:
                        processed_pdf_files_count += 1
                        total_raw_chunks_count += len(chunk_list)
                        for ch in chunk_list:
                            if not self.deduplicator.is_duplicate(ch.chunk_id):
                                all_chunks.append(ch)
                    else:
                        skipped_pdf_files_count += 1

                    # Update checkpoint state
                    new_checkpoint[fn] = {
                        "hash": file_hash,
                        "chunks": [c.to_dict() for c in chunk_list]
                    }

                    # Periodic atomic checkpoint save & progress logging
                    if completed_idx % 100 == 0 or completed_idx == total_discovered_pdfs:
                        self._save_checkpoint_atomic(new_checkpoint)
                        pct = (completed_idx / total_discovered_pdfs) * 100
                        log.info(
                            f"Processed {completed_idx}/{total_discovered_pdfs} PDF files ({pct:.1f}%) | "
                            f"Chunks: {len(all_chunks)} unique ({self.deduplicator.duplicate_count} duplicates)"
                        )

                except Exception as e:
                    log.error(f"Error processing future for {fn}: {e}")
                    quarantined_pdf_files_count += 1

        # Final checkpoint flush
        self._save_checkpoint_atomic(new_checkpoint)

        # 5. Atomic Output Write
        tmp_out = self.out_path.with_suffix(".tmp.jsonl")
        log.info(f"Writing {len(all_chunks)} finalized chunks to temporary file {tmp_out}")
        sorted_chunks = sorted(all_chunks, key=lambda c: c.chunk_id)
        with open(tmp_out, "w", encoding="utf-8") as f:
            for ch in sorted_chunks:
                f.write(json.dumps(ch.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

        # Promote temporary file atomically
        if tmp_out.exists():
            if self.out_path.exists():
                backup_path = self.raw_dir / "processed_chunks_legacy_backup.jsonl"
                if not backup_path.exists():
                    os.replace(str(self.out_path), str(backup_path))
            os.replace(str(tmp_out), str(self.out_path))
            log.info(f"Promoted validated corpus to {self.out_path} ({len(all_chunks)} unique chunks)")

        elapsed_time = round(time.time() - start_time, 2)

        # 6. Reconciled Dynamic Population Reporting
        duplicates_eliminated = self.deduplicator.duplicate_count
        unique_chunks = len(all_chunks)

        cat_dist = {}
        for c in all_chunks:
            cat_dist[c.category] = cat_dist.get(c.category, 0) + 1

        verified_is_count = sum(1 for c in all_chunks if c.identity_status == "verified" and c.is_number)
        non_standard_count = sum(1 for c in all_chunks if c.identity_status == "non_standard")
        unknown_count = sum(1 for c in all_chunks if c.identity_status == "unknown")

        quality_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": self.dataset_name,
            "dataset_scope": "Full Physical PDF Corpus (raw_data/pdfs/)",
            "runtime_seconds": elapsed_time,
            "raw_records": {
                "discovered": total_discovered_pdfs,
                "processed": processed_pdf_files_count,
                "quarantined": quarantined_pdf_files_count,
                "explicitly_skipped": skipped_pdf_files_count,
                "reconciliation_equation": f"{total_discovered_pdfs} == {processed_pdf_files_count} + {quarantined_pdf_files_count} + {skipped_pdf_files_count}",
                "reconciliation_valid": (total_discovered_pdfs == processed_pdf_files_count + quarantined_pdf_files_count + skipped_pdf_files_count),
            },
            "logical_source_records": {
                "discovered": total_discovered_pdfs,
                "processed": processed_pdf_files_count,
                "quarantined": quarantined_pdf_files_count,
                "explicitly_skipped": skipped_pdf_files_count,
                "reconciliation_equation": f"{total_discovered_pdfs} == {processed_pdf_files_count} + {quarantined_pdf_files_count} + {skipped_pdf_files_count}",
                "reconciliation_valid": (total_discovered_pdfs == processed_pdf_files_count + quarantined_pdf_files_count + skipped_pdf_files_count),
            },
            "pdf_records": {
                "discovered": total_discovered_pdfs,
                "processed": processed_pdf_files_count,
                "quarantined": quarantined_pdf_files_count,
                "explicitly_skipped": skipped_pdf_files_count,
                "reconciliation_equation": f"{total_discovered_pdfs} == {processed_pdf_files_count} + {quarantined_pdf_files_count} + {skipped_pdf_files_count}",
                "reconciliation_valid": (total_discovered_pdfs == processed_pdf_files_count + quarantined_pdf_files_count + skipped_pdf_files_count),
            },
            "pdf_file_accounting": {
                "discovered_pdf_files": total_discovered_pdfs,
                "processed_pdf_files": processed_pdf_files_count,
                "quarantined_pdf_files": quarantined_pdf_files_count,
                "skipped_pdf_files": skipped_pdf_files_count,
                "reconciliation_equation": f"{total_discovered_pdfs} == {processed_pdf_files_count} + {quarantined_pdf_files_count} + {skipped_pdf_files_count}",
                "reconciliation_valid": (total_discovered_pdfs == processed_pdf_files_count + quarantined_pdf_files_count + skipped_pdf_files_count),
            },
            "chunk_accounting": {
                "total_raw_chunks": total_raw_chunks_count,
                "unique_chunks": unique_chunks,
                "duplicates_eliminated": duplicates_eliminated,
                "reconciliation_equation": f"{total_raw_chunks_count} == {unique_chunks} + {duplicates_eliminated}",
                "reconciliation_valid": (total_raw_chunks_count == unique_chunks + duplicates_eliminated),
            },
            "metadata_integrity": {
                "verified_is_standards": verified_is_count,
                "non_standard_policy_chunks": non_standard_count,
                "unknown_identity_chunks": unknown_count,
                "category_distribution": cat_dist,
            },
            "corpus_hash": Deduplicator.compute_canonical_corpus_hash([c.to_dict() for c in all_chunks]),
        }

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        log.info(f"Generated Phase 1 Data Quality Report at {self.report_path} (Runtime: {elapsed_time}s)")

        self._print_summary(
            total_discovered=total_discovered_pdfs,
            processed=processed_pdf_files_count,
            quarantined=quarantined_pdf_files_count,
            skipped=skipped_pdf_files_count,
            total_raw_chunks=total_raw_chunks_count,
            unique_chunks=unique_chunks,
            duplicates=duplicates_eliminated,
            elapsed_time=elapsed_time,
        )

        return all_chunks

    def _print_summary(
        self,
        total_discovered: int,
        processed: int,
        quarantined: int,
        skipped: int,
        total_raw_chunks: int,
        unique_chunks: int,
        duplicates: int,
        elapsed_time: float,
    ):
        print("\n" + "=" * 78)
        print("PHASE 1 — FULL PDF CORPUS INGESTION SUMMARY")
        print("=" * 78)
        print(f"Runtime: {elapsed_time}s | Source Directory: {self.pdf_dir}")
        print("\n1. PDF FILE POPULATION ACCOUNTING:")
        print(f"  • Discovered PDF files:     {total_discovered}")
        print(f"  • Processed PDF files:      {processed}")
        print(f"  • Quarantined PDF files:    {quarantined}")
        print(f"  • Skipped PDF files:        {skipped}")
        print(f"  • Reconciliation Formula:   {total_discovered} == {processed} + {quarantined} + {skipped}")
        print(f"  • Accounting Match:         {'PASSED' if total_discovered == processed + quarantined + skipped else 'FAILED'}")

        print("\n2. CHUNK POPULATION ACCOUNTING:")
        print(f"  • Total raw chunks:         {total_raw_chunks}")
        print(f"  • Unique chunks:            {unique_chunks}")
        print(f"  • Duplicates eliminated:    {duplicates}")
        print(f"  • Reconciliation Formula:   {total_raw_chunks} == {unique_chunks} + {duplicates}")
        print(f"  • Chunk Match:              {'PASSED' if total_raw_chunks == unique_chunks + duplicates else 'FAILED'}")

        print("\n3. OUTPUT & ARTIFACT STATUS:")
        print(f"  • Output Chunk File:        {self.out_path} ({unique_chunks} verified chunks)")
        print(f"  • Data Quality Report:      {self.report_path}")
        print(f"  • Checkpoint File:          {self.checkpoint_path}")
        print(f"  • Quarantine Log:           {self.quarantine_path}")
        print("=" * 78 + "\n")

    def run_baseline_mode(self) -> List[ChunkRecord]:
        """Runs the legacy 1,000-source baseline ingestion mode for backward compatibility."""
        start_time = time.time()
        log.info(f"Starting Phase 1 Baseline Ingestion from {self.raw_dir} using {self.manifest_path.name}")

        manifest_records: List[Dict[str, Any]] = []
        manifest_by_file: Dict[str, List[Dict[str, Any]]] = {}

        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        manifest_records.append(entry)
                        lp = entry.get("local_path", "")
                        fn = Path(lp.replace("\\", "/")).name
                        manifest_by_file.setdefault(fn, []).append(entry)

        if len(manifest_records) > self.max_limit:
            raise ValueError(f"Baseline manifest exceeds limit of {self.max_limit} records")

        all_chunks: List[ChunkRecord] = []
        new_checkpoint: Dict[str, Any] = dict(self.checkpoint_cache)

        pdf_filenames = [fn for fn in manifest_by_file if fn.endswith(".pdf")]
        faq_filenames = [fn for fn in manifest_by_file if fn.endswith(".json") and "faq" in fn.lower()]
        psm_filenames = [fn for fn in manifest_by_file if fn.endswith(".json") and "product_standard_mapping" in fn.lower()]
        other_json_filenames = [fn for fn in manifest_by_file if fn.endswith(".json") and fn not in faq_filenames and fn not in psm_filenames]

        processed_physical_files_count = 0

        # Ingest baseline PDFs
        for fn in pdf_filenames:
            pdf_path = self.pdf_dir / fn if (self.pdf_dir / fn).exists() else self.raw_dir / fn
            entries = manifest_by_file.get(fn, [{}])
            primary_entry = entries[0]
            file_hash = primary_entry.get("content_hash") or (hashlib.sha256(pdf_path.read_bytes()).hexdigest() if pdf_path.exists() else f"missing_{fn[:12]}")

            if not pdf_path.exists():
                for ent in entries:
                    src_url = ent.get("source_url", f"file:///{fn}")
                    self.quarantine_logger.log_quarantine(fn, src_url, "FILE_NOT_FOUND", f"Physical file {fn} does not exist")
                    self.validator.record_skipped(f"FILE_NOT_FOUND: {fn}")
                continue

            cached = self.checkpoint_cache.get(fn)
            if cached and cached.get("hash") == file_hash and "chunks" in cached:
                chunks = [ChunkRecord(**c) for c in cached["chunks"]]
            else:
                chunks = []
                for ent in entries:
                    extracted = self.pdf_ingestor.ingest_pdf(pdf_path, ent)
                    chunks.extend(extracted)
                new_checkpoint[fn] = {
                    "hash": file_hash,
                    "chunks": [c.to_dict() for c in chunks]
                }

            if chunks:
                processed_physical_files_count += 1
                for ch in chunks:
                    if not self.deduplicator.is_duplicate(ch.chunk_id):
                        all_chunks.append(ch)

        # Ingest baseline JSONs
        if self.faq_ingestor:
            for fn in faq_filenames:
                json_path = (self.raw_dir / "json" / fn) if (self.raw_dir / "json" / fn).exists() else self.raw_dir / fn
                entries = manifest_by_file.get(fn, [{}])
                primary_entry = entries[0]
                file_hash = primary_entry.get("content_hash") or (hashlib.sha256(json_path.read_bytes()).hexdigest() if json_path.exists() else f"missing_{fn[:12]}")

                if not json_path.exists():
                    for ent in entries:
                        src_url = ent.get("source_url", f"file:///{fn}")
                        self.quarantine_logger.log_quarantine(fn, src_url, "FILE_NOT_FOUND", f"Physical file {fn} does not exist")
                        self.validator.record_skipped(f"FILE_NOT_FOUND: {fn}")
                    continue

                cached = self.checkpoint_cache.get(fn)
                if cached and cached.get("hash") == file_hash and "chunks" in cached:
                    chunks = [ChunkRecord(**c) for c in cached["chunks"]]
                else:
                    chunks = []
                    for ent in entries:
                        extracted = self.faq_ingestor.ingest_faq_json(json_path, ent)
                        chunks.extend(extracted)
                    new_checkpoint[fn] = {
                        "hash": file_hash,
                        "chunks": [c.to_dict() for c in chunks]
                    }

                if chunks:
                    processed_physical_files_count += 1
                    for ch in chunks:
                        if not self.deduplicator.is_duplicate(ch.chunk_id):
                            all_chunks.append(ch)

        # Extract structured datasets in baseline mode
        if self.structured_ingestor:
            self.structured_ingestor.extract_product_standard_map(manifest_records)
            self.structured_ingestor.extract_labs_directory(manifest_records)

        # Atomic write
        tmp_out = self.out_path.with_suffix(".tmp.jsonl")
        sorted_chunks = sorted(all_chunks, key=lambda c: c.chunk_id)
        with open(tmp_out, "w", encoding="utf-8") as f:
            for ch in sorted_chunks:
                f.write(json.dumps(ch.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

        if tmp_out.exists():
            os.replace(str(tmp_out), str(self.out_path))

        self._save_checkpoint_atomic(new_checkpoint)
        return all_chunks

    def run(self) -> List[ChunkRecord]:
        """Main entry point: delegates to run_full_pdf_corpus (default) or run_baseline_mode."""
        if self.mode == "baseline":
            return self.run_baseline_mode()
        else:
            return self.run_full_pdf_corpus()


def main():
    parser = argparse.ArgumentParser(description="BIS PDF Corpus Parser & Ingestion Pipeline")
    parser.add_argument("--mode", default="full", choices=["full", "baseline"], help="Ingestion mode: 'full' (default) or 'baseline'")
    parser.add_argument("--raw_dir", default="./raw_data", help="Path to raw_data directory")
    parser.add_argument("--pdf_dir", default=None, help="Path to PDF source directory (default: raw_data/pdfs)")
    parser.add_argument("--out", default="./processed_chunks.jsonl", help="Output path for processed_chunks.jsonl")
    parser.add_argument("--manifest", default=None, help="Path to manifest JSONL file")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file")
    parser.add_argument("--dataset_name", default=None, help="Dataset identifier name")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads")
    parser.add_argument("--max_limit", type=int, default=1000, help="Maximum allowed records in baseline mode")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    pdf_dir_path = Path(args.pdf_dir) if args.pdf_dir else None

    pipeline = IngestionPipeline(
        raw_dir=Path(args.raw_dir),
        pdf_dir=pdf_dir_path,
        out_path=Path(args.out),
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        mode=args.mode,
        dataset_name=args.dataset_name,
        max_limit=args.max_limit,
        max_workers=args.workers,
    )
    chunks = pipeline.run()
    print(f"PDF Ingestion completed successfully. Generated {len(chunks)} verified chunks.")


if __name__ == "__main__":
    main()
