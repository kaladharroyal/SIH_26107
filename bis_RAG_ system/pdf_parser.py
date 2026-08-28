"""
Clause-aware PDF Parser — Phase 1, Steps 3-4

Most IS standard PDFs follow a numbered clause structure:
    1. Scope
    2. Normative References
    3. Terminology
    4. Requirements
        4.1 General
        4.2 Material
            4.2.1 ...

Naive fixed-size chunking (e.g. "every 500 tokens") destroys this structure —
a chunk might start mid-clause, and you lose the ability to cite "IS 302,
Clause 4.2" accurately. This parser instead:

  1. Extracts text page-by-page with pdfplumber (keeps layout better than
     PyPDF2 for multi-column standard documents).
  2. Detects clause headers via regex (numbered headings like "4.2" or
     "4.2.1", optionally followed by a title).
  3. Groups text under its nearest preceding clause header.
  4. Emits one chunk per clause (splitting further only if a clause is very
     long), each tagged with: IS number, clause number, clause title, page
     range, and the standard's revision year.

The IS number and revision year are pulled from the document's cover page /
filename convention (e.g. "IS_302_Part_1_2008.pdf" -> IS 302 Part 1 : 2008).
Adjust `parse_standard_identity` if your corpus's filenames differ.
"""

import re
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import pdfplumber

log = logging.getLogger("pdf_parser")

# Matches clause headers like "4", "4.2", "4.2.1" or Devanagari "४.२" at the start of a line,
# followed by any English, Hindi, or bilingual heading text.
CLAUSE_HEADER_RE = re.compile(
    r"^\s*([\d\u0966-\u096f]{1,2}(?:\.[\d\u0966-\u096f]{1,2}){0,3})\s+([^\n]{2,100})?\s*$",
    re.UNICODE
)

# Matches "IS 302", "IS 302 Part 1", "IS 302 (Part 1)" etc. plus a year.
STANDARD_ID_RE = re.compile(r"IS[\s_]?(\d+)(?:[\s_(]*Part[\s_]?(\d+)\)?)?.*?(\d{4})", re.IGNORECASE)


@dataclass
class Clause:
    is_number: str          # e.g. "IS 302"
    part: str | None        # e.g. "1" or None
    revision_year: str      # e.g. "2008"
    clause_number: str      # e.g. "4.2"
    clause_title: str | None
    text: str
    page_start: int
    page_end: int
    source_file: str

    def chunk_id(self) -> str:
        part_str = f"_Part{self.part}" if self.part else ""
        return f"{self.is_number.replace(' ', '')}{part_str}_{self.revision_year}_C{self.clause_number}"


# Lazy loaded EasyOCR instance to avoid model re-initialization overhead per page
_EASYOCR_READER = None

def get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(['en', 'hi'], gpu=False)
    return _EASYOCR_READER


def parse_standard_identity(pdf_path: Path, first_page_text: str) -> tuple[str, str | None, str]:
    """
    Try filename first (more reliable than OCR/layout-dependent cover-page text),
    fall back to first-page text.
    """
    for source in (pdf_path.stem, first_page_text[:500]):
        match = STANDARD_ID_RE.search(source)
        if match:
            number, part, year = match.groups()
            return f"IS {number}", part, year
    log.warning(f"Could not parse standard identity for {pdf_path.name} - flagging for manual review")
    return "UNKNOWN", None, "UNKNOWN"


def extract_page_text_with_ocr(page, page_num: int, pdf_path: Path) -> str:
    """
    Extract text from pdfplumber Page object.
    If plain text extraction is empty or too short (scanned page / image PDF),
    attempt OCR fallback using pytesseract or easyocr if installed.
    """
    try:
        text = page.extract_text() or ""
    except Exception as pe:
        log.warning(f"Extract text failed on page {page_num} of {pdf_path.name}: {pe}")
        text = ""

    if len(text.strip()) >= 30:
        return text

    # Fallback OCR 1: Pytesseract (English + Hindi)
    try:
        import pytesseract
        pil_img = page.to_image(resolution=200).original
        ocr_text = pytesseract.image_to_string(pil_img, lang="eng+hin")
        if len(ocr_text.strip()) > 10:
            log.info(f"OCR (pytesseract) extracted text on page {page_num} of {pdf_path.name}")
            return ocr_text
    except Exception:
        pass

    # Fallback OCR 2: EasyOCR (Lazy-loaded singleton)
    try:
        import numpy as np
        reader = get_easyocr_reader()
        pil_img = page.to_image(resolution=150).original
        results = reader.readtext(np.array(pil_img), detail=0)
        easy_text = "\n".join(results)
        if len(easy_text.strip()) > 10:
            log.info(f"OCR (easyocr) extracted text on page {page_num} of {pdf_path.name}")
            return easy_text
    except Exception:
        pass

    return text


def extract_clauses(pdf_path: Path) -> list[Clause]:
    clauses: list[Clause] = []
    current_lines: list[str] = []
    current_header: tuple[str, str | None] = ("0", "Preamble")
    current_page_start = 1

    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            log.warning(f"{pdf_path.name} has no pages")
            return []
        try:
            first_page_text = pdf.pages[0].extract_text() or ""
        except Exception:
            first_page_text = ""
        is_number, part, year = parse_standard_identity(pdf_path, first_page_text)

        def flush(end_page: int):
            if current_lines:
                clauses.append(Clause(
                    is_number=is_number,
                    part=part,
                    revision_year=year,
                    clause_number=current_header[0],
                    clause_title=current_header[1],
                    text="\n".join(current_lines).strip(),
                    page_start=current_page_start,
                    page_end=end_page,
                    source_file=pdf_path.name,
                ))

        for page_num, page in enumerate(pdf.pages, start=1):
            text = extract_page_text_with_ocr(page, page_num, pdf_path)
            for line in text.split("\n"):
                header_match = CLAUSE_HEADER_RE.match(line)
                if header_match:
                    # New clause starts — flush the previous one first.
                    flush(page_num)
                    current_lines = []
                    current_header = (header_match.group(1), header_match.group(2))
                    current_page_start = page_num
                else:
                    current_lines.append(line)

        flush(len(pdf.pages))

    return [c for c in clauses if c.text]  # drop empty trailing artifacts


def split_long_clause(clause: Clause, max_chars: int = 1500) -> list[Clause]:
    """
    A clause like "4. Requirements" can span pages. Split on paragraph
    boundaries if it exceeds max_chars, keeping the same clause_number so
    citations still resolve correctly.
    """
    if len(clause.text) <= max_chars:
        return [clause]

    chunks = []
    paras = clause.text.split("\n\n")
    current_para_group: list[str] = []
    current_len = 0

    for p in paras:
        if current_len + len(p) > max_chars and current_para_group:
            chunks.append(Clause(
                is_number=clause.is_number,
                part=clause.part,
                revision_year=clause.revision_year,
                clause_number=clause.clause_number,
                clause_title=clause.clause_title,
                text="\n\n".join(current_para_group),
                page_start=clause.page_start,
                page_end=clause.page_end,
                source_file=clause.source_file,
            ))
            current_para_group = []
            current_len = 0
        current_para_group.append(p)
        current_len += len(p)

    if current_para_group:
        chunks.append(Clause(
            is_number=clause.is_number,
            part=clause.part,
            revision_year=clause.revision_year,
            clause_number=clause.clause_number,
            clause_title=clause.clause_title,
            text="\n\n".join(current_para_group),
            page_start=clause.page_start,
            page_end=clause.page_end,
            source_file=clause.source_file,
        ))

    return chunks


def record_parser_warning(filename: str, reason: str):
    try:
        with open("warning_downloads.txt", "a", encoding="utf-8") as f:
            f.write(f"- {filename} (Parser Note: {reason})\n")
    except Exception:
        pass


def process_pdf(pdf_path: Path) -> list[dict]:
    """Returns a list of chunk dicts ready for embedding + Postgres insert."""
    clauses = extract_clauses(pdf_path)
    chunks = []
    for clause in clauses:
        for sub in split_long_clause(clause):
            chunks.append(asdict(sub))
    log.info(f"{pdf_path.name}: extracted {len(chunks)} chunks across {len(clauses)} clauses")
    if not chunks:
        record_parser_warning(pdf_path.name, "0 chunks extracted - likely scanned/image PDF without OCR text")
    return chunks


def process_directory(raw_dir: Path, out_path: Path):
    all_chunks = []
    for pdf_path in raw_dir.glob("*.pdf"):
        try:
            all_chunks.extend(process_pdf(pdf_path))
        except Exception as e:
            log.error(f"Failed on {pdf_path.name}: {e}")
            record_parser_warning(pdf_path.name, f"Failed parse: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(all_chunks)} total chunks to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./raw_data")
    parser.add_argument("--out", default="./processed_chunks.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    process_directory(Path(args.raw_dir), Path(args.out))
