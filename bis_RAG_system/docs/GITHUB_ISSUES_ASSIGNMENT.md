# 🚀 SIH 2026 — BIS RAG Assistant: 6-Person Team Work Division & GitHub Issues

This document outlines the complete work breakdown for a **6-developer team** across all 7 phases of the BIS RAG Assistant implementation plan.

---

## 👥 Team Work Assignment Overview

| Developer | Role & Focus Area | Assigned Phases | Core Responsibilities |
|---|---|---|---|
| **Developer 1** | Data Ingestion & Scraper Lead | Phase 1 | Site crawling, JS-API scraping, PDF downloads, warning logs, manifest tracking |
| **Developer 2** | DB, Embeddings & Hybrid Search | Phase 2 | Clause-aware parsing, PostgreSQL + PGVector, BGE-M3 embeddings, BM25 + RRF Reranking |
| **Developer 3** | RAG Generation & Grounding | Phase 3 | Citation engine, prompt engineering, hallucination guardrails, confidence threshold gate |
| **Developer 4** | Intent Router & Sub-Flows | Phase 4 | Intent classification, product recommender lookup, lab locator, consumer complaint routing |
| **Developer 5** | Multilingual & Translation | Phase 5 | Hinglish / Hindi handling, translation pipeline, citation preservation during translation |
| **Developer 6** | Frontend UI & Demo Prep | Phase 6 & 7 | Next.js/Vite Web UI, expandable citations, UI routing, feedback loop, test suite & demo script |

---

## 📋 Detailed Issue Cards & Acceptance Criteria

### 👤 Developer 1: Data Ingestion & Scraper Lead
**Issue 1.1: Web Scraping & Multi-Domain Pipeline (`scraper.py` & `full_site_scraper.py`)**
- **Tasks**:
  - Maintain and optimize `full_site_scraper.py` across `bis.gov.in`, `crsbis.in`, `standards.bis.gov.in`, `bsbedge.com`, `irportal.bis.gov.in`.
  - Implement chunked streaming (`1MB` buffers) for heavy PDFs (>50MB–150MB).
  - Automatically log all HTTP 403, 404, or incomplete read errors to `warning_downloads.txt`.
- **Acceptance Criteria**: All 27 core seed URLs in `sources.yaml` + discovery links scraped; `manifest.jsonl` tracks `source_url`, `content_hash`, `category`, and `local_path`.

**Issue 1.2: JS-Rendered Portal & API Scrapers (LIMS, ManakOnline, Know Your Standards)**
- **Tasks**:
  - Reverse-engineer backend API endpoints for JS-rendered portals:
    - LIMS Recognized Labs (`https://lims.bis.gov.in/home/labs/`)
    - LIMS Empanelled Labs (`https://lims.bis.gov.in/home/empaneled_labs/`)
    - Search Labs BY IS Number (`https://lims.bis.gov.in/home/search_is_number/`)
    - Know Your Standards (`https://standards.bis.gov.in/website/know-your-standards`)
  - Save extracted tabular JSON data directly into `./raw_data`.
- **Acceptance Criteria**: Filterable JSON files for 431+ labs and IS-to-lab mappings stored in `./raw_data`.

---

### 👤 Developer 2: Database, Embeddings & Hybrid Search Specialist
**Issue 2.1: Clause-Aware Multilingual PDF Parser (`pdf_parser.py`)**
- **Tasks**:
  - Enhance `pdf_parser.py` regex `CLAUSE_HEADER_RE` to detect Devanagari numerals (`०-९`) and non-ASCII titles.
  - Implement OCR fallback using `pytesseract` (`lang="eng+hin"`) and `easyocr` for scanned pages without text layers.
  - Attach `page_start`, `page_end`, and `source_file` metadata to every chunk.
- **Acceptance Criteria**: `processed_chunks.jsonl` generated cleanly without `cp1252` encoding crashes.

**Issue 2.2: Vector Storage, Hybrid Search & Reranking (`loader.py` & `retrieval.py`)**
- **Tasks**:
  - Setup PostgreSQL schema (`schema.sql`) with `pgvector` for `standards`, `clauses`, `product_standard_map`, and `query_logs`.
  - Embed chunks using `BGE-M3` (dense vectors).
  - Implement Hybrid Search: Dense vector similarity + Sparse BM25 keyword search combined via Reciprocal Rank Fusion (RRF).
  - Re-rank top candidates using `BGE-Reranker-v2-m3`.
- **Acceptance Criteria**: Retrieval test query returns top-5 relevant chunks in <300ms.

---

### 👤 Developer 3: RAG Generation & Grounding Specialist
**Issue 3.1: Strict Citation Engine & Grounded Prompt Engineering**
- **Tasks**:
  - Design RAG prompt template instructing LLM to answer strictly using retrieved context chunks.
  - Implement citation formatter enforcing standardized formats:
    - Clause-based: `As per IS 1786:2008, Clause 4.2`
    - FAQ-based: `As per Product Certification FAQ Q3`
  - Differentiate visually/textually between official standard regulations vs. simplified FAQ guidelines.
- **Acceptance Criteria**: Every generated answer contains exact, clickable citation tags matching retrieved chunk IDs.

**Issue 3.2: Confidence Scoring & Uncertainty Refusal Gate**
- **Tasks**:
  - Calculate confidence scores based on reranker top-1 score and score distribution margin.
  - Implement refusal gate: if top chunk score < threshold (e.g. 0.45), return structured refusal: *"I cannot find a specific BIS clause or regulation covering this query. Please check official portals."*
- **Acceptance Criteria**: Out-of-corpus queries trigger clean refusal without hallucinating fake IS numbers.

---

### 👤 Developer 4: Intent Router & Deterministic Sub-Flow Specialist
**Issue 4.1: Query Intent Classifier & Sub-Flow Dispatcher (`router.py`)**
- **Tasks**:
  - Build Intent Classifier routing queries into 5 categories:
    1. `product_recommendation` ("which standard applies to my product")
    2. `certification_process` ("how do I apply for ISI mark")
    3. `lab_location` ("where can I test IS 1786 steel in Guntur")
    4. `consumer_complaint` ("how to complain about bad hallmark")
    5. `general_rag` (general technical or regulatory queries)
- **Acceptance Criteria**: Intent router classifies 50 sample user queries with >95% accuracy.

**Issue 4.2: Product Recommender & Lab Locator Lookup Engines**
- **Tasks**:
  - Implement Product Recommender: Direct lookup in `product_standard_map` table (bypassing prose RAG generation to guarantee 0% hallucination).
  - Implement Lab Locator: Filter lab directory by region, district, IS number, and scope.
  - Build Consumer Complaint Router: Provide direct links and steps for BIS CARE App and ManakOnline portal.
- **Acceptance Criteria**: Recommender returns exact IS numbers, scheme names (ISI/CRS/FMCS), and mandatory status badges.

---

### 👤 Developer 5: Multilingual & Translation Specialist
**Issue 5.1: Multilingual Retrieval Strategy (BGE-M3 vs Translate-then-Retrieve)**
- **Tasks**:
  - Benchmark BGE-M3 direct embedding of Hindi queries against English chunk index vs. Translate-then-Retrieve (Option A vs. Option B).
  - Optimize hybrid search weights for non-English queries (reduce BM25 weight, increase dense vector weight).
  - Handle Code-Mixed "Hinglish" queries (*"mera LED bulb ke liye BIS certification chahiye"*).
- **Acceptance Criteria**: Hindi and Hinglish queries retrieve the same top-3 English clauses as equivalent English queries.

**Issue 5.2: Citation Preservation & Response Translation Engine**
- **Tasks**:
  - Implement citation-masking regex pass during translation (`[IS 1786:2008, Clause 4.2]` -> `[CITE_REF_1]`).
  - Translate response prose into target language (Hindi/Regional).
  - Unmask `[CITE_REF_1]` back into the translated text to guarantee IS numbers and clause citations remain character-for-character untouched.
- **Acceptance Criteria**: Response translated into Hindi while citations like `IS 1786:2008, Clause 4.2` remain 100% exact.

---

### 👤 Developer 6: Full-Stack UI, Feedback Loop & Demo Lead
**Issue 6.1: Modern Web UI & Interactive Citation Viewer**
- **Tasks**:
  - Build responsive Javascript/HTML or React Web Application.
  - Implement chat interface with expandable/clickable inline citations.
  - Render citation links directly to PDF page numbers using `page_start`/`page_end` metadata.
  - Build Fallback Filtered Search Panel (for direct lab/product search without chat).
- **Acceptance Criteria**: UI displays grounded answers, clickable PDF citations, and fallback filter tabs cleanly.

**Issue 6.2: Feedback Logging & End-to-End Persona Test Suite (Phase 7)**
- **Tasks**:
  - Wire thumbs up/down UI buttons to log entries into `query_logs` table (`query`, `response`, `retrieved_chunk_ids`, `user_feedback`).
  - Create test suite covering MSME, Student, and Consumer personas.
  - Prepare 4-step Demo Script:
    1. Grounded answer with clickable citation
    2. Correct refusal on out-of-corpus query
    3. Structured UI rendering for Product Recommender
    4. Multilingual Hindi query demonstration
- **Acceptance Criteria**: Live demo script verified working end-to-end.

---

## 🛠️ GitHub CLI (`gh`) Commands to Create All Issues

You can create these issues directly on your GitHub repository (`kaladharroyal/SIH_26107`) by running these commands:

```bash
# Developer 1 Issues
gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 1: Web Scraping & Multi-Domain Pipeline" --body "Implement and maintain scraper.py & full_site_scraper.py across bis.gov.in, crsbis.in, standards.bis.gov.in, bsbedge.com." --label "Phase-1,Scraper"

gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 1: JS-Rendered Portal & API Scrapers (LIMS, ManakOnline)" --body "Reverse-engineer and scrape LIMS labs, Know Your Standards, and ManakOnline API endpoints into raw_data/." --label "Phase-1,Scraper"

# Developer 2 Issues
gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 2: Clause-Aware Multilingual PDF Parser" --body "Enhance pdf_parser.py with Devanagari regex and OCR fallback (pytesseract/easyocr) for scanned PDFs." --label "Phase-1,Parser"

gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 2: PostgreSQL PGVector, Hybrid Search & BGE Reranker" --body "Setup schema.sql with pgvector, BGE-M3 dense embeddings, BM25 sparse search, and RRF + BGE-Reranker-v2-m3." --label "Phase-2,Search"

# Developer 3 Issues
gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 3: Strict Citation Engine & Grounded Prompts" --body "Design grounded prompt templates and citation engine for 'As per IS XXXX:YYYY, Clause Z'." --label "Phase-3,RAG"

gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 3: Confidence Scoring & Uncertainty Refusal Gate" --body "Implement confidence score estimator and refusal gate when top reranker score < threshold." --label "Phase-3,Guardrails"

# Developer 4 Issues
gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 4: Query Intent Classifier & Sub-Flow Dispatcher" --body "Build intent router classifying queries into product_recommendation, certification, lab_location, consumer, RAG." --label "Phase-4,Router"

gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 4: Deterministic Product Recommender & Lab Locator" --body "Build product_standard_map lookup engine (0% hallucination) and filterable lab locator." --label "Phase-4,SubFlows"

# Developer 5 Issues
gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 5: Multilingual Strategy & Hinglish Query Benchmark" --body "Evaluate BGE-M3 direct retrieval vs Translate-then-Retrieve on Hindi and Hinglish queries." --label "Phase-5,Multilingual"

gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 5: Citation Masking & Response Translation Engine" --body "Build citation-masking pass to ensure IS numbers and clause references are never corrupted during translation." --label "Phase-5,Translation"

# Developer 6 Issues
gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 6: Modern Web UI & Interactive Citation Viewer" --body "Build chat UI, clickable PDF page citations, intent-based UI rendering, and fallback search panel." --label "Phase-6,Frontend"

gh issue create --repo kaladharroyal/SIH_26107 --title "Dev 6: Feedback Loop Logging & Persona Demo Preparation" --body "Wire thumbs up/down feedback to query_logs table and execute 4-step persona demo script (Phase 7)." --label "Phase-7,Demo"
```
