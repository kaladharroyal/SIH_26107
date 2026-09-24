"""
FAQ Ingestor - Phase 1 Data Foundation (faq_ingestor.py)
Ingests and splits official BIS FAQ Q&A entries with preserved question numbers and authoritative URLs.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from metadata import ChunkRecord, generate_canonical_chunk_id
from validator import IngestionValidator

log = logging.getLogger("faq_ingestor")

QA_SPLIT_PATTERN = re.compile(r"(?:^|\n)\s*(?:Q\s*\.?\s*(\d+)|(\d+)\.)\s*[:.-]?\s*", re.IGNORECASE)


class FAQIngestor:
    def __init__(self, validator: IngestionValidator):
        self.validator = validator

    def ingest_faq_json(
        self, json_path: Path, manifest_entry: Optional[Dict[str, Any]] = None
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        source_url = manifest_entry.get("source_url") if manifest_entry else "https://www.bis.gov.in/faqs/"
        source_hash = manifest_entry.get("content_hash") if manifest_entry else f"hash_{json_path.stem[:12]}"
        manifest_cat = manifest_entry.get("category") if manifest_entry else "faq"

        # Map category to controlled taxonomy
        if "hallmark" in json_path.name.lower() or manifest_cat == "hallmarking_faq":
            category = "hallmarking"
        elif "consumer" in json_path.name.lower() or manifest_cat == "consumer_faq":
            category = "consumer"
        elif "lab" in json_path.name.lower() or manifest_cat == "lab_faq":
            category = "lab_directory"
        elif "cert" in json_path.name.lower() or manifest_cat == "certification_faq":
            category = "certification"
        else:
            category = "faq"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for idx, item in enumerate(data, 1):
                    raw_text = item.get("qa_text") or item.get("page_text") or ""
                    if not raw_text.strip() or len(raw_text.strip()) < 20:
                        continue

                    # Extract Q number if present
                    q_match = re.search(r"Q\s*\.?\s*(\d+)", raw_text, re.IGNORECASE)
                    q_num = q_match.group(1) if q_match else str(idx)

                    # Extract first sentence as question title
                    first_line = raw_text.strip().split("\n")[0][:120]
                    title = f"FAQ Q.{q_num}: {first_line}"

                    clause_id = f"FAQ_Q{q_num}_{idx}"
                    chunk_id = generate_canonical_chunk_id(source_hash, clause_id, raw_text)

                    record = ChunkRecord(
                        chunk_id=chunk_id,
                        is_number=None,
                        part=None,
                        revision_year=None,
                        identity_status="non_standard",
                        identity_reason="Official BIS FAQ Guideline",
                        clause_number=f"Q.{q_num}",
                        clause_title=title,
                        text=raw_text.strip(),
                        category=category,
                        content_type="html_page",
                        source_url=source_url,
                        source_file=f"raw_data/{json_path.name}",
                        page_range="Web FAQ",
                        source_hash=source_hash,
                        source_of_truth="verified_bis_html",
                    )
                    if self.validator.validate_chunk(record):
                        chunks.append(record)

        except Exception as e:
            self.validator.quarantine_logger.log_quarantine(
                str(json_path), source_url, "FAQ_PARSE_EXCEPTION", str(e)
            )

        return chunks
