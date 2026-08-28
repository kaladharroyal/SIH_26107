"""
PDF Document Ingestor - Phase 1 Data Foundation (pdf_ingestor.py)
Extracts clause-level chunks from standard PDFs, QCO orders, and administrative publications using pypdf.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from pypdf import PdfReader

from metadata import ChunkRecord, generate_canonical_chunk_id
from validator import IngestionValidator

log = logging.getLogger("pdf_ingestor")

CLAUSE_HEADER_RE = re.compile(
    r"^\s*([\d\u0966-\u096f]{1,2}(?:\.[\d\u0966-\u096f]{1,2}){0,3})\s+([^\n]{2,100})?\s*$",
    re.UNICODE,
)

STANDARD_ID_RE = re.compile(
    r"IS[\s_:-]?(\d+)(?:[\s_(]*Part[\s_:-]?(\d+)\)?)?(?:[\s:_–-]*(\d{4}))?",
    re.IGNORECASE,
)

ADMIN_PATTERNS = [
    r"annual[\s_-]?report",
    r"review[\s_-]?statement",
    r"delay[\s_-]?statement",
    r"organisation[\s_-]?chart",
    r"internship[\s_-]?scheme",
    r"handbook[\s_-]?for[\s_-]?tc",
    r"training[\s_-]?strategy",
    r"revised[\s_-]?sfm",
    r"eoi",
]


def extract_standard_identity(pdf_path: Path, first_page_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], str, str]:
    """
    Identifies whether the PDF is a technical IS standard or an administrative non-standard publication.
    Returns: (is_number, part, revision_year, identity_status, identity_reason)
    """
    name_lower = pdf_path.name.lower()
    
    # Check for administrative publications first
    for admin_pat in ADMIN_PATTERNS:
        if re.search(admin_pat, name_lower):
            return None, None, None, "non_standard", "BIS Annual Report / Administrative Notice - not an Indian Standard"

    # Search for standard identifier in filename and first page
    for text_source in (pdf_path.stem, first_page_text[:1000]):
        match = STANDARD_ID_RE.search(text_source)
        if match:
            num, part, year = match.groups()
            is_no = f"IS {num}"
            return is_no, part, year, "verified", f"Standard {is_no} identified from PDF header/filename"

    return None, None, None, "unknown", "No technical Indian Standard number found in document cover"


class PDFIngestor:
    def __init__(self, validator: IngestionValidator):
        self.validator = validator

    def ingest_pdf(
        self, pdf_path: Path, manifest_entry: Optional[Dict[str, Any]] = None
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        source_url = manifest_entry.get("source_url") if manifest_entry else f"file:///{pdf_path.name}"
        source_hash = manifest_entry.get("content_hash") if manifest_entry else f"hash_{pdf_path.stem[:12]}"
        manifest_cat = manifest_entry.get("category") if manifest_entry else "other"

        try:
            reader = PdfReader(str(pdf_path))
            if not reader.pages:
                self.validator.quarantine_logger.log_quarantine(
                    str(pdf_path), source_url, "EMPTY_PDF", "PDF contains 0 pages"
                )
                return []

            first_page_text = reader.pages[0].extract_text() or ""
            is_no, part, year, ident_status, ident_reason = extract_standard_identity(pdf_path, first_page_text)

            # Determine controlled category
            if ident_status == "non_standard":
                category = "annual_report" if "annual" in pdf_path.name.lower() else "general_policy"
            elif "qco" in pdf_path.name.lower() or manifest_cat == "crs_qco":
                category = "qco_order"
            elif manifest_cat in ["hallmarking", "hallmarking_faq"]:
                category = "hallmarking"
            elif manifest_cat in ["lab_directory", "lab_faq"]:
                category = "lab_directory"
            elif is_no:
                category = "is_standard"
            else:
                category = manifest_cat if manifest_cat in ["is_standard", "qco_order", "certification", "hallmarking", "general_policy"] else "general_policy"

            current_lines: List[str] = []
            current_header: Tuple[str, Optional[str]] = ("0", "Preamble")
            current_page_start = 1

            def flush_clause(end_page: int):
                if current_lines:
                    full_text = "\n".join(current_lines).strip()
                    if len(full_text) >= 15:
                        clause_id = f"{is_no or 'DOC'}_C{current_header[0]}_P{current_page_start}"
                        chunk_id = generate_canonical_chunk_id(source_hash, clause_id, full_text)
                        src_file_rel = f"raw_data/pdfs/{pdf_path.name}" if pdf_path.parent.name == "pdfs" else f"raw_data/{pdf_path.name}"
                        record = ChunkRecord(
                            chunk_id=chunk_id,
                            is_number=is_no,
                            part=part,
                            revision_year=year,
                            identity_status=ident_status,
                            identity_reason=ident_reason,
                            clause_number=str(current_header[0]),
                            clause_title=current_header[1] or "Overview",
                            text=full_text,
                            category=category,
                            content_type="pdf",
                            source_url=source_url,
                            source_file=src_file_rel,
                            page_range=f"{current_page_start}-{end_page}",
                            source_hash=source_hash,
                            source_of_truth="verified_bis_pdf",
                        )
                        if self.validator.validate_chunk(record):
                            chunks.append(record)

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""

                if not page_text.strip():
                    continue

                for line in page_text.split("\n"):
                    match = CLAUSE_HEADER_RE.match(line)
                    if match:
                        flush_clause(page_num)
                        current_lines = []
                        current_header = (match.group(1), match.group(2))
                        current_page_start = page_num
                    else:
                        current_lines.append(line)

            flush_clause(len(reader.pages))

        except Exception as e:
            self.validator.quarantine_logger.log_quarantine(
                str(pdf_path), source_url, "PDF_PARSE_EXCEPTION", str(e)
            )

        return chunks
