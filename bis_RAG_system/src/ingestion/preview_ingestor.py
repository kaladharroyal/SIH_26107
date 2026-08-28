"""
Standard Preview & Amendment Ingestor - Phase 1 Data Foundation (preview_ingestor.py)
Ingests standard preview specifications and amendment records with URL identity cross-validation.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from metadata import ChunkRecord, generate_canonical_chunk_id
from validator import IngestionValidator

log = logging.getLogger("preview_ingestor")

URL_ID_PATTERN = re.compile(r"[?&]id=(\d+)(?:_(\d{4}))?", re.IGNORECASE)
URL_STDNO_PATTERN = re.compile(r"[?&]stdno=IS[\s_:-]*(\d+)", re.IGNORECASE)
SECTION_SPLIT_RE = re.compile(r"(?:^|\n)\s*(\d+(?:\.\d+)?)\.?\s+([A-Z\s]{3,40})\s*", re.UNICODE)


class PreviewIngestor:
    def __init__(self, validator: IngestionValidator):
        self.validator = validator

    def ingest_preview_json(
        self, json_path: Path, manifest_entry: Optional[Dict[str, Any]] = None
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        source_url = manifest_entry.get("source_url") if manifest_entry else "https://standardsbis.bsbedge.com/"
        source_hash = manifest_entry.get("content_hash") if manifest_entry else f"hash_{json_path.stem[:12]}"
        manifest_cat = manifest_entry.get("category") if manifest_entry else "product_standard_mapping"

        # Determine controlled category
        if "amendment" in json_path.name.lower() or "amend" in source_url.lower():
            category = "amendment"
        elif "qco" in json_path.name.lower():
            category = "qco_order"
        elif manifest_cat == "product_standard_mapping":
            category = "product_standard_mapping"
        else:
            category = "is_standard"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for idx, item in enumerate(data, 1):
                    item_url = item.get("page_url") or source_url
                    raw_text = item.get("page_text") or ""
                    if not raw_text.strip() or len(raw_text.strip()) < 30:
                        continue

                    # 1. Extract potential IS number from URL
                    url_is_num = None
                    url_year = None
                    id_m = URL_ID_PATTERN.search(item_url)
                    if id_m:
                        url_is_num = id_m.group(1)
                        url_year = id_m.group(2)
                    else:
                        stdno_m = URL_STDNO_PATTERN.search(item_url)
                        if stdno_m:
                            url_is_num = stdno_m.group(1)

                    # 2. Cross-validate through the URL Identity Validation Gate
                    is_no, year, ident_status, ident_reason = self.validator.cross_validate_url_identity(
                        url_is_num=url_is_num,
                        url_year=url_year,
                        text=raw_text,
                        title=json_path.stem,
                    )

                    # 3. Clause/Section Extraction
                    sections = list(SECTION_SPLIT_RE.finditer(raw_text))
                    if len(sections) >= 2:
                        for s_idx, match in enumerate(sections):
                            start = match.start()
                            end = sections[s_idx + 1].start() if s_idx + 1 < len(sections) else len(raw_text)
                            sec_text = raw_text[start:end].strip()
                            sec_num = match.group(1)
                            sec_title = match.group(2).strip()

                            clause_id = f"{is_no or 'PREV'}_C{sec_num}_{s_idx}"
                            chunk_id = generate_canonical_chunk_id(source_hash, clause_id, sec_text)

                            record = ChunkRecord(
                                chunk_id=chunk_id,
                                is_number=is_no,
                                part=None,
                                revision_year=year,
                                identity_status=ident_status,
                                identity_reason=ident_reason,
                                clause_number=sec_num,
                                clause_title=sec_title,
                                text=sec_text,
                                category=category,
                                content_type="json_api",
                                source_url=item_url,
                                source_file=f"raw_data/{json_path.name}",
                                page_range=f"Section {sec_num}",
                                source_hash=source_hash,
                                source_of_truth="verified_bis_api",
                            )
                            if self.validator.validate_chunk(record):
                                chunks.append(record)
                    else:
                        clause_id = f"{is_no or 'PREV'}_Overview_{idx}"
                        chunk_id = generate_canonical_chunk_id(source_hash, clause_id, raw_text)
                        record = ChunkRecord(
                            chunk_id=chunk_id,
                            is_number=is_no,
                            part=None,
                            revision_year=year,
                            identity_status=ident_status,
                            identity_reason=ident_reason,
                            clause_number="1",
                            clause_title="Scope & Technical Specifications",
                            text=raw_text.strip(),
                            category=category,
                            content_type="json_api",
                            source_url=item_url,
                            source_file=f"raw_data/{json_path.name}",
                            page_range="Standard Preview",
                            source_hash=source_hash,
                            source_of_truth="verified_bis_api",
                        )
                        if self.validator.validate_chunk(record):
                            chunks.append(record)

        except Exception as e:
            self.validator.quarantine_logger.log_quarantine(
                str(json_path), source_url, "PREVIEW_PARSE_EXCEPTION", str(e)
            )

        return chunks
