"""
Validation Gate & Quarantine Manager - Phase 1 Data Foundation (validator.py)
Enforces schema validity, taxonomy compliance, URL cross-validation, and quarantine logging.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

from metadata import (
    ChunkRecord,
    ALLOWED_CATEGORIES,
    ALLOWED_PRODUCTION_SOURCE_OF_TRUTH,
    ALLOWED_IDENTITY_STATUS,
)

log = logging.getLogger("ingestion_validator")


class QuarantineLogger:
    def __init__(self, quarantine_file: Path):
        self.quarantine_file = Path(quarantine_file)
        self.quarantined_count = 0

    def log_quarantine(self, source_file: str, source_url: str, reason_code: str, details: str):
        self.quarantined_count += 1
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source_file),
            "source_url": source_url,
            "reason_code": reason_code,
            "details": details,
        }
        with open(self.quarantine_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.warning(f"Quarantined {source_file}: [{reason_code}] {details}")


class IngestionValidator:
    def __init__(self, quarantine_logger: QuarantineLogger):
        self.quarantine_logger = quarantine_logger
        self.processed_records_count = 0
        self.quarantined_records_count = 0
        self.skipped_records_count = 0
        self.skipped_reasons: Dict[str, int] = {}

    def record_skipped(self, reason: str):
        self.skipped_records_count += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def cross_validate_url_identity(
        self, url_is_num: Optional[str], url_year: Optional[str], text: str, title: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], str, str]:
        """
        Cross-validates an IS number extracted from URL query parameters against the document text/title.
        Returns: (is_number, revision_year, identity_status, identity_reason)
        """
        if not url_is_num:
            return None, None, "unknown", "No standard identifier extracted from source URL"

        search_corpus = f"{title or ''} {text[:1500]}".lower()
        # Look for the numeric standard reference in text
        is_pattern = rf"\b(is|standard|indian standard)?[\s_:\-]*{re.escape(url_is_num)}\b"

        if re.search(is_pattern, search_corpus, re.IGNORECASE) or (title and url_is_num in title):
            return f"IS {url_is_num}", url_year, "verified", f"Standard IS {url_is_num} verified against document text/title"
        else:
            return None, None, "unknown", f"URL-derived IS {url_is_num} could not be confirmed in document body or title"

    def validate_chunk(self, chunk: ChunkRecord) -> bool:
        """
        Validates a ChunkRecord against Phase 1 criteria:
        1. Category must be in ALLOWED_CATEGORIES.
        2. Source of truth must be in ALLOWED_PRODUCTION_SOURCE_OF_TRUTH (no test fixtures).
        3. Identity status must be in ALLOWED_IDENTITY_STATUS.
        4. If identity_status == 'non_standard', is_number must be None.
        5. Text must be non-empty (>= 15 chars).
        6. Source URL and source hash must be non-empty.
        """
        if not chunk.text or len(chunk.text.strip()) < 15:
            self.quarantine_logger.log_quarantine(
                chunk.source_file, chunk.source_url, "EMPTY_OR_INSUFFICIENT_TEXT", f"Text length is {len(chunk.text.strip())} characters"
            )
            self.quarantined_records_count += 1
            return False

        if chunk.category not in ALLOWED_CATEGORIES:
            self.quarantine_logger.log_quarantine(
                chunk.source_file, chunk.source_url, "INVALID_CATEGORY", f"Category '{chunk.category}' not in ALLOWED_CATEGORIES"
            )
            self.quarantined_records_count += 1
            return False

        if chunk.source_of_truth not in ALLOWED_PRODUCTION_SOURCE_OF_TRUTH:
            self.quarantine_logger.log_quarantine(
                chunk.source_file, chunk.source_url, "INVALID_SOURCE_OF_TRUTH", f"Source of truth '{chunk.source_of_truth}' is prohibited in production corpus"
            )
            self.quarantined_records_count += 1
            return False

        if chunk.identity_status not in ALLOWED_IDENTITY_STATUS:
            self.quarantine_logger.log_quarantine(
                chunk.source_file, chunk.source_url, "INVALID_IDENTITY_STATUS", f"Identity status '{chunk.identity_status}' not in ALLOWED_IDENTITY_STATUS"
            )
            self.quarantined_records_count += 1
            return False

        if chunk.identity_status == "non_standard" and chunk.is_number is not None:
            self.quarantine_logger.log_quarantine(
                chunk.source_file, chunk.source_url, "NON_STANDARD_IS_NUMBER_MISMATCH", f"is_number must be None for non_standard documents (got '{chunk.is_number}')"
            )
            self.quarantined_records_count += 1
            return False

        if not chunk.source_url or not chunk.source_hash:
            self.quarantine_logger.log_quarantine(
                chunk.source_file, chunk.source_url, "MISSING_URL_OR_HASH", "source_url or source_hash is missing"
            )
            self.quarantined_records_count += 1
            return False

        self.processed_records_count += 1
        return True
