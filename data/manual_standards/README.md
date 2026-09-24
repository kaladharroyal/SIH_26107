# Manually curated Indian Standards

Drop full-text IS standard PDFs here that you've downloaded yourself via a
BIS e-Sale account (standardsbis.bsbedge.com). Don't try to automate this —
downloads are login-gated and watermarked with your account email; a bulk
scraper can't and shouldn't replicate that flow.

Naming convention pdf_parser.py expects (via parse_standard_identity):
    IS_<number>_Part_<part>_<year>.pdf   e.g. IS_302_Part_1_2008.pdf
    IS_<number>_<year>.pdf               e.g. IS_1786_2008.pdf   (no parts)

Suggested starter set for a demo (pick ~20-30 covering common categories):
  - IS 302 (Part 1) — household electrical appliance safety
  - IS 1786 — reinforcement bars (construction, very commonly asked about)
  - IS 15820 — hallmarking centre requirements (referenced constantly in
    hallmarking FAQs, worth having the actual clauses for)
  - IS 1417 — gold jewellery fineness grades
  - A couple of CRS-mandatory electronics standards (check the CRS product
    list scraped in Phase 1 for current examples — the list changes)

Run pdf_parser.py against this folder the same way as raw_data/:
    python ingestion/pdf_parser.py --raw_dir manual_standards --out manual_standards_chunks.jsonl
