"""
Canonical BIS Corpus Recovery Engine - Phase 1 Remediation Implementation (recover_corpus.py)

Deterministic, idempotent, provenance-preserving physical PDF recovery engine.
Processes recovery_manifest.jsonl, streams binaries to disk in 64KB chunks,
strictly validates PDF magic bytes and structure via pypdf, verifies SHA-256 hashes,
places valid documents into domain partitions under raw_data/pdfs/, quarantines failures,
maintains append-only recovery_log.jsonl, and incrementally writes corpus_registry.jsonl
and quarantine_log.jsonl with thread-safe flushing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import requests
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    import pypdf
except ImportError:
    pypdf = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("corpus_recovery")


class CorpusRecoveryEngine:
    def __init__(
        self,
        manifest_path: Path,
        raw_dir: Path,
        log_path: Path,
        registry_path: Path,
        quarantine_log_path: Path,
        workers: int = 2,
        timeout: Optional[Any] = None,
        connect_timeout: int = 20,
        read_timeout: int = 180,
        max_retries: int = 10,
    ):
        self.manifest_path = Path(manifest_path)
        self.raw_dir = Path(raw_dir)
        self.pdf_dir = self.raw_dir / "pdfs"
        self.quarantine_dir = self.raw_dir / "quarantine"
        self.temp_dir = self.raw_dir / ".tmp_recovery"
        self.log_path = Path(log_path)
        self.registry_path = Path(registry_path)
        self.quarantine_log_path = Path(quarantine_log_path)

        self.workers = workers
        if timeout is not None:
            if isinstance(timeout, (int, float)):
                self.timeout = (timeout, timeout)
            else:
                self.timeout = timeout
        else:
            self.timeout = (connect_timeout, read_timeout)
        self.max_retries = max_retries

        # Ensure directory structure
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        for dom in ["standards", "amendments", "certification", "consumer", "general", "hallmarking"]:
            (self.pdf_dir / dom).mkdir(parents=True, exist_ok=True)

        self._thread_local = threading.local()
        self._log_lock = threading.Lock()
        self._registry_lock = threading.Lock()
        self._quarantine_lock = threading.Lock()

        # Load existing registry to ensure idempotent appends
        self._registered_doc_ids: Set[str] = set()
        if self.registry_path.exists() and self.registry_path.stat().st_size > 0:
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            if "document_id" in entry:
                                self._registered_doc_ids.add(entry["document_id"])
                log.info(f"Loaded {len(self._registered_doc_ids)} existing registered documents from {self.registry_path}")
            except Exception as e:
                log.warning(f"Error loading existing registry: {e}")

        # Load existing quarantined record_ids to prevent duplicate quarantine lines
        self._quarantined_record_ids: Set[str] = set()
        if self.quarantine_log_path.exists() and self.quarantine_log_path.stat().st_size > 0:
            try:
                with open(self.quarantine_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            if "record_id" in entry:
                                self._quarantined_record_ids.add(entry["record_id"])
                log.info(f"Loaded {len(self._quarantined_record_ids)} existing quarantined records from {self.quarantine_log_path}")
            except Exception as e:
                log.warning(f"Error loading existing quarantine log: {e}")

    def _get_session(self) -> requests.Session:
        """Returns a thread-local session to ensure thread safety on Windows."""
        if not hasattr(self._thread_local, "session"):
            session = requests.Session()
            retries = Retry(
                total=self.max_retries,
                backoff_factor=2.0,
                status_forcelist=[500, 502, 503, 504],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retries)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.bis.gov.in/",
            })
            self._thread_local.session = session
        return self._thread_local.session

    def _validate_pdf_file(self, file_path: Path) -> Tuple[bool, str, int]:
        """Validates PDF signature, non-zero size, structure, and page count on disk."""
        if not file_path.exists():
            return False, "FILE_NOT_FOUND", 0
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "EMPTY_FILE", 0

        with open(file_path, "rb") as f:
            header = f.read(512)
            if not header.startswith(b"%PDF-"):
                if b"<html" in header.lower() or b"<!doctype html" in header.lower():
                    return False, "NOT_A_PDF (HTML response)", 0
                return False, "NOT_A_PDF (Missing %PDF- header)", 0

        # Structural inspection using pypdf
        if pypdf is not None:
            try:
                reader = pypdf.PdfReader(str(file_path), strict=False)
                num_pages = len(reader.pages)
                if num_pages <= 0:
                    return False, "ZERO_PAGES", 0
                return True, "VALID", num_pages
            except Exception as e:
                return False, f"CORRUPT_PDF: {str(e)[:100]}", 0
        else:
            with open(file_path, "rb") as f:
                f.seek(max(0, file_size - 1024))
                tail = f.read()
                if b"%%EOF" not in tail:
                    return False, "TRUNCATED_PDF (Missing %%EOF)", 0
            return True, "VALID", 1

    def _append_registry(self, record_meta: Dict[str, Any], file_path: Path, sha256_hash: str, page_count: int) -> None:
        """Thread-safe incremental append to corpus_registry.jsonl."""
        doc_id = record_meta["document_id"]
        with self._registry_lock:
            if doc_id in self._registered_doc_ids:
                return
            entry = {
                "document_id": doc_id,
                "record_id": record_meta["record_id"],
                "sha256": sha256_hash,
                "filename": record_meta["filename"],
                "local_path": str(file_path.as_posix()),
                "source_url": record_meta["source_url"],
                "domain": record_meta["domain"],
                "classification": record_meta.get("classification", "STANDARDS"),
                "title": record_meta.get("title", ""),
                "standard_number": record_meta.get("standard_number", ""),
                "file_size": file_path.stat().st_size,
                "page_count": page_count,
                "validation_status": "VALID"
            }
            with open(self.registry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
            self._registered_doc_ids.add(doc_id)

    def _append_quarantine(self, failure_info: Dict[str, Any]) -> None:
        """Thread-safe incremental append to quarantine_log.jsonl."""
        rec_id = failure_info["record_id"]
        with self._quarantine_lock:
            entry = {
                "record_id": rec_id,
                "document_id": failure_info.get("document_id"),
                "source_url": failure_info.get("source_url"),
                "expected_sha256": failure_info.get("expected_sha256"),
                "actual_sha256": failure_info.get("actual_sha256"),
                "failure_reason": failure_info.get("failure_reason"),
                "http_status": failure_info.get("http_status"),
                "attempt_timestamp": failure_info.get("attempt_timestamp"),
                "quarantine_timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "QUARANTINED"
            }
            with open(self.quarantine_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
            self._quarantined_record_ids.add(rec_id)

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single canonical record with bounded retries, disk streaming,
        PDF validation, SHA-256 verification, atomic file placement, and immediate
        registry/quarantine emission.
        """
        record_id = record["record_id"]
        doc_id = record["document_id"]
        source_url = record["source_url"]
        expected_sha = record["expected_sha256"]
        filename = record["filename"]
        domain = record["domain"].lower()

        target_dir = self.pdf_dir / domain
        target_path = target_dir / filename
        temp_file = self.temp_dir / f"{record_id}_{filename}.tmp"

        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Resumability & Idempotency Check on existing target file
        if target_path.exists():
            try:
                hasher = hashlib.sha256()
                with open(target_path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                existing_sha = hasher.hexdigest()

                if existing_sha == expected_sha:
                    is_valid, v_reason, pages = self._validate_pdf_file(target_path)
                    if is_valid:
                        # Ensure immediately registered if not already
                        self._append_registry(record, target_path, existing_sha, pages)
                        return {
                            "record_id": record_id,
                            "document_id": doc_id,
                            "source_url": source_url,
                            "attempt_timestamp": now_iso,
                            "http_status": 200,
                            "response_content_type": "application/pdf",
                            "downloaded_size": target_path.stat().st_size,
                            "expected_sha256": expected_sha,
                            "actual_sha256": existing_sha,
                            "validation_status": "VALID",
                            "page_count": pages,
                            "failure_reason": None,
                            "final_local_path": str(target_path.as_posix()),
                            "recovery_status": "RECOVERED",
                            "cached": True,
                        }
                else:
                    # Mismatch on disk - quarantine existing corrupted file
                    quarantine_path = self.quarantine_dir / f"{record_id}_CORRUPT_EXISTING_{filename}"
                    shutil.move(str(target_path), str(quarantine_path))
                    log.warning(f"[{record_id}] Existing file {filename} had hash mismatch. Moved to quarantine.")
            except Exception as e:
                log.warning(f"[{record_id}] Error checking existing file {target_path}: {e}")

        # 2. Download with bounded exponential backoff retries and chunked streaming
        last_failure = None
        http_status = None
        content_type = ""
        downloaded_size = 0
        actual_sha = None

        backoff_delays = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]

        for attempt in range(1, self.max_retries + 1):
            current_size = temp_file.stat().st_size if temp_file.exists() else 0
            req_headers = {}
            if current_size > 0:
                req_headers["Range"] = f"bytes={current_size}-"

            try:
                session = self._get_session()
                resp = session.get(
                    source_url,
                    headers=req_headers,
                    stream=True,
                    timeout=self.timeout,
                    verify=False,
                    allow_redirects=True
                )
                http_status = resp.status_code
                content_type = resp.headers.get("Content-Type", "")

                if http_status == 416:
                    # Range not satisfiable -> file is already complete on disk
                    last_failure = None
                    break
                elif http_status in (404, 403):
                    last_failure = f"HTTP_{http_status}"
                    break
                elif http_status == 206:
                    mode = "ab"
                elif http_status == 200:
                    mode = "wb"
                else:
                    last_failure = f"HTTP_{http_status}"
                    delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                    time.sleep(delay)
                    continue

                with open(temp_file, mode) as tf:
                    # 64KB chunk size for smooth socket streaming
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            tf.write(chunk)

                # Stream completed cleanly
                last_failure = None
                break

            except (urllib3.exceptions.IncompleteRead, requests.exceptions.ChunkedEncodingError) as ire:
                last_failure = f"INCOMPLETE_READ: {str(ire)[:80]}"
                delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                log.info(f"[{record_id}] IncompleteRead on attempt {attempt}/{self.max_retries} (got {temp_file.stat().st_size if temp_file.exists() else 0} bytes). Resuming with range in {delay}s...")
                time.sleep(delay)
            except requests.exceptions.Timeout:
                last_failure = "TIMEOUT"
                delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                time.sleep(delay)
            except requests.exceptions.ConnectionError as ce:
                last_failure = f"CONNECTION_FAILED: {str(ce)[:80]}"
                delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                time.sleep(delay)
            except Exception as e:
                last_failure = f"DOWNLOAD_FAILED: {str(e)[:80]}"
                delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
                time.sleep(delay)

        # Check if downloaded file is complete and compute SHA-256
        if temp_file.exists() and temp_file.stat().st_size > 0:
            downloaded_size = temp_file.stat().st_size
            hasher = hashlib.sha256()
            with open(temp_file, "rb") as tf:
                while chunk := tf.read(65536):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest()
            if actual_sha == expected_sha:
                last_failure = None

        # 3. Handle Download Failures
        if last_failure or not temp_file.exists():
            temp_file.unlink(missing_ok=True)
            log.warning(f"[{record_id}] Final download failure from {source_url}: {last_failure}")
            failure_dict = {
                "record_id": record_id,
                "document_id": doc_id,
                "source_url": source_url,
                "attempt_timestamp": now_iso,
                "http_status": http_status,
                "response_content_type": content_type,
                "downloaded_size": downloaded_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "validation_status": "FAILED",
                "page_count": 0,
                "failure_reason": last_failure or "DOWNLOAD_FAILED",
                "final_local_path": None,
                "recovery_status": "QUARANTINED",
                "cached": False,
            }
            self._append_quarantine(failure_dict)
            return failure_dict

        # 4. PDF Structure & Header Validation on Disk
        is_pdf_valid, val_reason, page_count = self._validate_pdf_file(temp_file)
        if not is_pdf_valid:
            quarantine_path = self.quarantine_dir / f"{record_id}_INVALID_{filename}"
            shutil.move(str(temp_file), str(quarantine_path))
            log.error(f"[{record_id}] PDF validation failed for {filename}: {val_reason}")
            failure_dict = {
                "record_id": record_id,
                "document_id": doc_id,
                "source_url": source_url,
                "attempt_timestamp": now_iso,
                "http_status": http_status,
                "response_content_type": content_type,
                "downloaded_size": downloaded_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "validation_status": "FAILED",
                "page_count": 0,
                "failure_reason": val_reason,
                "final_local_path": str(quarantine_path.as_posix()),
                "recovery_status": "QUARANTINED",
                "cached": False,
            }
            self._append_quarantine(failure_dict)
            return failure_dict

        # 5. SHA-256 Hash Verification
        if actual_sha != expected_sha:
            quarantine_path = self.quarantine_dir / f"{record_id}_MISMATCH_{filename}"
            shutil.move(str(temp_file), str(quarantine_path))
            log.error(f"[{record_id}] SHA-256 MISMATCH for {filename}: Expected {expected_sha}, got {actual_sha}")
            failure_dict = {
                "record_id": record_id,
                "document_id": doc_id,
                "source_url": source_url,
                "attempt_timestamp": now_iso,
                "http_status": http_status,
                "response_content_type": content_type,
                "downloaded_size": downloaded_size,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "validation_status": "FAILED",
                "page_count": page_count,
                "failure_reason": "HASH_MISMATCH",
                "final_local_path": str(quarantine_path.as_posix()),
                "recovery_status": "QUARANTINED",
                "cached": False,
            }
            self._append_quarantine(failure_dict)
            return failure_dict

        # 6. ACCEPT: Atomic move to canonical physical location and immediate registry emission
        shutil.move(str(temp_file), str(target_path))
        self._append_registry(record, target_path, actual_sha, page_count)
        log.info(f"[{record_id}] Successfully recovered and verified {filename} ({downloaded_size:,} bytes, {page_count} pages)")
        return {
            "record_id": record_id,
            "document_id": doc_id,
            "source_url": source_url,
            "attempt_timestamp": now_iso,
            "http_status": http_status,
            "response_content_type": content_type,
            "downloaded_size": downloaded_size,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "validation_status": "VALID",
            "page_count": page_count,
            "failure_reason": None,
            "final_local_path": str(target_path.as_posix()),
            "recovery_status": "RECOVERED",
            "cached": False,
        }

    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Executes recovery across the manifest with controlled multithreading and immediate logging."""
        # 1. Clean any orphaned temporary artifacts from prior aborted runs
        for tf in self.temp_dir.glob("*.tmp"):
            log.info(f"Removing orphaned temporary artifact: {tf.name}")
            tf.unlink(missing_ok=True)

        log.info(f"Loading recovery manifest from {self.manifest_path}...")
        records: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if limit:
            records = records[:limit]
            log.info(f"Test mode: limited to {limit} records.")

        log.info(f"Starting recovery for {len(records)} records with {self.workers} workers...")
        results: List[Dict[str, Any]] = []
        start_time = time.time()

        log_f = open(self.log_path, "a", encoding="utf-8")

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_rec = {executor.submit(self.process_record, r): r for r in records}
            completed_count = 0

            for future in as_completed(future_to_rec):
                res = future.result()
                results.append(res)
                with self._log_lock:
                    log_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    log_f.flush()

                completed_count += 1
                if completed_count % 25 == 0 or completed_count == len(records):
                    elapsed = time.time() - start_time
                    log.info(f"Progress: {completed_count}/{len(records)} completed ({completed_count/len(records)*100:.1f}%) in {elapsed:.1f}s")

        log_f.close()

        # Cleanup temp dir on clean finish
        for tf in self.temp_dir.glob("*.tmp"):
            tf.unlink(missing_ok=True)

        recovered_results = [r for r in results if r["recovery_status"] == "RECOVERED"]
        quarantined_results = [r for r in results if r["recovery_status"] == "QUARANTINED"]

        summary = {
            "target_records": len(records),
            "recovered": len(recovered_results),
            "quarantined": len(quarantined_results),
            "unresolved": len(records) - len(recovered_results) - len(quarantined_results),
            "total_bytes_recovered": sum(r["downloaded_size"] for r in recovered_results),
            "total_pages_recovered": sum(r["page_count"] for r in recovered_results),
            "runtime_seconds": round(time.time() - start_time, 2)
        }
        log.info(f"Recovery Run Complete: {summary}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Canonical BIS Corpus Recovery Engine - Phase 1 Remediation")
    parser.add_argument("--manifest", type=str, default="recovery_manifest.jsonl", help="Path to recovery manifest")
    parser.add_argument("--raw-dir", type=str, default="raw_data", help="Target raw_data directory")
    parser.add_argument("--log-file", type=str, default="recovery_log.jsonl", help="Append-only recovery log")
    parser.add_argument("--registry-file", type=str, default="corpus_registry.jsonl", help="Corpus registry file")
    parser.add_argument("--quarantine-log", type=str, default="quarantine_log.jsonl", help="Quarantine log file")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent worker threads (default 2)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records (for testing)")

    args = parser.parse_args()

    engine = CorpusRecoveryEngine(
        manifest_path=Path(args.manifest),
        raw_dir=Path(args.raw_dir),
        log_path=Path(args.log_file),
        registry_path=Path(args.registry_file),
        quarantine_log_path=Path(args.quarantine_log),
        workers=args.workers,
    )

    engine.run(limit=args.limit)


if __name__ == "__main__":
    main()
