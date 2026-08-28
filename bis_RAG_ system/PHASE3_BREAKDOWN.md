# 📑 Phase 3: Generation & Guardrails — 4-Part Work Division

This document divides **Phase 3 (Generation Layer & Guardrails)** into 4 distinct, actionable parts for implementation and GitHub assignment.

---

## 📌 Part 3.1: System Prompt & Grounded-Answer Prompting (`generator.py`)
- **Focus**: Context-Constrained LLM Generation & Strict Grounding
- **Key Tasks**:
  1. Build System Prompt enforcing that the LLM may ONLY state facts present in the retrieved context chunks.
  2. Pass context chunks with explicit metadata visible: `[Chunk ID | IS Number | Clause Number | Page Range | Source URL]`.
  3. Enforce strict accuracy for numerical compliance figures (fees, timelines, inspection charges, validity periods) copied directly from retrieved FAQ/Clause text.
- **Deliverable**: `generator.py` (Prompt template & LLM generation handler).

---

## 📌 Part 3.2: Confidence Scoring & Uncertainty Refusal Gate (`guardrails.py`)
- **Focus**: Out-of-Corpus Query Refusal & Hallucination Prevention
- **Key Tasks**:
  1. Calculate confidence score based on top-1 reranker score and top-K distribution margin.
  2. Implement Refusal Gate: if score < threshold (e.g. `0.45`), block raw LLM output.
  3. Return structured fallback message: *"I cannot find a specific BIS clause or regulation covering this query. Please verify directly with official portals."* with direct links to official BIS inquiry portals.
- **Deliverable**: `guardrails.py` (Confidence estimator & fallback gateway).

---

## 📌 Part 3.3: Dual Citation Formatter Engine (`citation_engine.py`)
- **Focus**: Clickable PDF/Web Citation Formatting
- **Key Tasks**:
  1. Parse LLM response and format citations into standardized markdown links:
     - **Clause Regulation Citation**: `[As per IS 1786:2008, Clause 4.2](file:///...#page=4)`
     - **FAQ Guideline Citation**: `[Per BIS Product Certification FAQ (Q.14)](https://www.bis.gov.in/...)`
  2. Differentiate visually between official legal standard text vs simplified FAQ guidelines.
  3. Attach `page_start`/`page_end` anchors to PDF document links.
- **Deliverable**: `citation_engine.py` (Citation parser & link renderer).

---

## 📌 Part 3.4: RAG Pipeline Orchestrator & End-to-End Test Suite (`rag_pipeline.py` & `test_phase3.py`)
- **Focus**: Full Pipeline Integration & Grounding Test Suite
- **Key Tasks**:
  1. Connect Phase 2 Hybrid Retrieval -> Guardrail Gate -> Prompt Construction -> LLM Generation -> Citation Engine.
  2. Build test suite running real queries across Product Certification, Hallmarking, Steel Standards, and Out-of-Corpus queries.
  3. Verify 0% hallucination rate on numerical compliance figures.
- **Deliverable**: `rag_pipeline.py` & `test_phase3.py` (Full RAG Pipeline & Test Runner).

---

## 🛠️ GitHub CLI (`gh`) Commands to Create Phase 3 Issues

```bash
gh issue create --repo kaladharroyal/SIH_26107 --title "Part 3.1: System Prompt & Grounded-Answer Prompting (generator.py)" --body "Build system prompt enforcing strict context-only grounding and numerical figure accuracy." --label "Phase-3,Generation"

gh issue create --repo kaladharroyal/SIH_26107 --title "Part 3.2: Confidence Scoring & Uncertainty Refusal Gate (guardrails.py)" --body "Calculate reranker confidence scores and implement fallback gate for out-of-corpus queries." --label "Phase-3,Guardrails"

gh issue create --repo kaladharroyal/SIH_26107 --title "Part 3.3: Dual Citation Formatter Engine (citation_engine.py)" --body "Build citation parser formatting Clause citations vs FAQ citations with PDF page anchors." --label "Phase-3,Citations"

gh issue create --repo kaladharroyal/SIH_26107 --title "Part 3.4: RAG Pipeline Orchestrator & End-to-End Test Suite (rag_pipeline.py)" --body "Connect retrieval to generator and build test suite verifying grounding and refusal rates." --label "Phase-3,Integration"
```
