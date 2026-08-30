# BIS Assistant — Phase 1: Data Foundation

Implements Steps 1–4 of Phase 1 from the implementation plan: source identification,
scraping, clause-aware PDF parsing, and structured storage.

## Pipeline order

```
sources.yaml → scraper.py → raw_data/*.pdf + manifest.jsonl
                                  ↓
                            pdf_parser.py → processed_chunks.jsonl
                                  ↓
                            loader.py → Postgres (standards, clauses tables)
```

## Setup

```bash
pip install -r requirements.txt
createdb bis_assistant
psql bis_assistant < schema.sql
```

## Data source map

`ingestion/sources.example.yaml` now contains **27 verified, live URLs** (checked
2026-08-27) organized by problem-statement capability rather than BIS's own site
structure:

| Capability                          | Category tags                              | Count |
|--------------------------------------|---------------------------------------------|-------|
| Product → standard recommender      | `product_standard_mapping`                  | 2     |
| Certification scheme guidance       | `certification_process`, `certification_faq`| 9     |
| Licensing procedure & fees          | `licensing_fees`, `licensing_procedure`     | 3     |
| Testing labs                        | `lab_directory`, `lab_faq`                  | 2     |
| Hallmarking                         | `hallmarking`, `hallmarking_faq`            | 6     |
| Consumer queries / complaints       | `consumer`, `consumer_faq`                  | 3     |
| Training / capacity building        | `training`, `training_faq`                  | 2     |

Copy it to `sources.yaml` to use as-is, or trim it down for your demo scope.

**Two things flagged inline in the config that matter before you run it:**
1. Several BIS lookups (lab search, "Know Your Standards", jeweller/AHC lists) are
   JS-rendered single-page apps — plain `requests` won't see their data. They're
   commented out with a note on what to do instead (find the underlying API via
   browser devtools).
2. `manakonline.in`, `crsbis.in`, and `huid.manakonline.in` are separate portals
   BIS runs alongside `bis.gov.in` — the scraper needs to hit all of them, not
   just the main domain.

## Step 1-2: Identify sources & scrape

Fill in real BIS URLs in `ingestion/sources.example.yaml` (copy it to
`sources.yaml` first) — you'll need to manually browse bis.gov.in and the
BIS Care portal to find the actual listing pages, since URL structure isn't
documented anywhere. Look for:
- The IS standards catalog/search page (for PDF listings)
- CRS/QCO product lists (usually a table, sometimes a downloadable PDF/XLS)
- The BIS-recognized labs directory
- Hallmarking and other scheme documentation pages

```bash
cd ingestion
python scraper.py --config sources.yaml --out ../raw_data
```

This produces `raw_data/*.pdf` (or `.json` for table sources) plus a
`manifest.jsonl` recording every fetch with a content hash — re-run
periodically and diff hashes to detect when BIS revises a standard.

## Step 3: Parse into clause-level chunks

```bash
python pdf_parser.py --raw_dir ../raw_data --out ../processed_chunks.jsonl
```

**This has been tested against a mock IS-standard-formatted PDF and correctly
splits numbered clauses (1, 4, 4.1, 4.2, 4.2.1) into individually tagged
chunks with clause number, title, and page range preserved.**

Real BIS PDFs will need calibration — some are scanned images (need OCR
first, see the pdf-reading approach for scanned docs), some use different
heading conventions (Roman numerals, "Clause X" prefixes). Spot-check the
output on ~10 real standards before trusting it at scale, and adjust
`CLAUSE_HEADER_RE` / `STANDARD_ID_RE` in `pdf_parser.py` accordingly.

## Step 4: Load into Postgres

```bash
python loader.py --chunks ../processed_chunks.jsonl --db_url postgresql://user:pass@localhost/bis_assistant
```

Sanity-check before moving to Phase 2 (embeddings):

```sql
SELECT is_number, part, revision_year, count(*) 
FROM clauses c JOIN standards s ON c.standard_id = s.id 
GROUP BY 1,2,3;

SELECT clause_number, clause_title, left(text, 100) 
FROM clauses LIMIT 20;
```

If clause counts look wrong (e.g. one giant "clause 0" per document) for a
given PDF, that document's headings probably don't match the regex — flag
it for manual chunking rather than silently ingesting garbage.

## Full standard texts: manual_standards/ convention

Automated bulk scraping of full IS standard PDFs isn't feasible (see the
e-Sale login/watermarking constraint noted earlier). Instead:

1. Create an account at standardsbis.bsbedge.com yourself.
2. Download a curated set of standards relevant to your demo — see
   `manual_standards/README.md` for a suggested starter list and the
   filename convention `pdf_parser.py` expects.
3. Run the parser against that folder directly:
   ```bash
   python ingestion/pdf_parser.py --raw_dir ../manual_standards --out ../manual_standards_chunks.jsonl
   ```
4. Load into Postgres the same way as the scraped chunks.

This keeps the demo honest about scope (a curated 20-30 standards, not a claim
of full 19,000+ standard coverage) while still exercising the full clause-aware
pipeline end-to-end.

## Known gaps to handle before Phase 2

- **Scanned/image-only PDFs**: `pdf_parser.py` assumes extractable text.
  Older IS standards are often scanned — route those through OCR first.
- **JS-rendered lookups**: lab search (`lims.bis.gov.in`), "Know Your
  Standards" (`standards.bis.gov.in`), and jeweller/AHC lists
  (`manakonline.in`) are single-page apps — the scraper needs their
  underlying API endpoints, not their HTML, and those aren't identified yet.
- **Revision tracking**: `superseded_by` in the schema is there but nothing
  populates it yet — needs a manual or semi-automated cross-reference step
  when you find "supersedes IS XXXX:YYYY" language in a standard's foreword.
- **FAQ Q&A splitting**: `scrape_html_page`'s regex was verified against the
  real product-certification FAQ page (handles `Q 1`, `Q.2`, `Q 12:`, `Q17.`,
  `Q. 25` variants) — but spot-check it against 2-3 other FAQ pages in the
  config, since BIS doesn't use fully consistent numbering site-wide.
