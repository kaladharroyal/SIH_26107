# RAG RUNTIME PATH & RETRIEVAL FORENSIC AUDIT
## SIH_26107 — BIS AI Compliance Assistant

**Date:** 2026-09-26  
**Auditor:** Antigravity (Strict Read-Only Diagnostic Mode)  
**Corpus Authority:** `V:\PROJECTS\SIH_26107\bis_data\bis_data\raw_data`  
**System Repository:** `V:\PROJECTS\SIH_26107\bis_RAG_system`  

---

## 1. Executive Finding

The observed sub-second responses (typically **20–45 ms**) for natural-language product queries are **not** produced by OpenAI `gpt-4o-mini`, nor are they produced by prompt caching. They are the result of an **architectural LLM bypass**: queries containing product or manufacturing keywords are classified by `QueryIntentRouter` as `product_recommendation` and dispatched directly to `ProductRecommender.recommend()`. `ProductRecommender` executes an in-memory string alias replacement, attempts a static dataset lookup against `product_standard_map.json`, falls back to an in-memory BM25 retrieval call, and renders the final markdown response entirely through **deterministic Python string templating without ever invoking an LLM**.

The convergence of steel and TMT queries onto the erroneous standard **`IS 5489:1975`** (with irrelevant pressure/flow text) is caused by a compounding sequence of three distinct application-layer failures:
1. **Query Collapsing via Alias Normalization:** `PRODUCT_ALIASES` in `product_recommender.py` aggressively matches the substring `"steel bars"` or `"tmt bars"` and replaces the entire user query with the collapsed string `"High strength deformed steel bars"`, discarding critical semantic tokens like *"concrete reinforcement"*.
2. **Missing Static Dataset Records:** `product_standard_map.json` contains only 81 records and has zero entries for `IS 1786`, `TMT`, or `High strength deformed steel bars`. This triggers fallback retrieval.
3. **Artificial Category Filtering in BM25 Fallback:** `ProductRecommender` hardcodes `category="is_standard"` when calling `retrieve_fast()`. In the ingested corpus, the authoritative specification overview containing `IS 1786:2008` (`raw_data/7bc138ec7bc2_news_2024-10-01_114.pdf`) was ingested under `category="general_policy"` because its filename starts with `news_`. The filter `category="is_standard"` strictly excludes it. Within the remaining `is_standard` chunks, the top lexical match is an All-India Fee/Licence listing (`raw_data/2e9cc92ed3b8_AIF-21-22.pdf`, chunk `026044628f445697`), which was misclassified as standard `IS 5489` with title `kg/cm2 , Minimum Working pressure `.

Crucially, **the BM25 retrieval engine itself is completely sound**: when the exact query *"high-strength deformed steel bars"* is executed against BM25 **without** the artificial category filter (`category=None`), BM25 retrieves the correct standard **`IS 1786:2008` at Rank 1 with a high score of 25.10**. The failure is entirely an application-level routing, normalization, and filtering defect.

---

## 2. Actual Execution Architecture

The complete runtime call graph starting from `POST /api/chat` was mapped:

```text
POST /api/chat
      ↓
[app.py:chat_endpoint]
      ↓
[rag_pipeline.py:BISRAGPipeline.query]
      ↓
[multilingual.py:MultilingualHandler.detect_language]
      ↓
[router.py:QueryIntentRouter.classify_intent]
      ↓
┌────────────────────────────────────────────────────────┐
│ Intent Dispatch Gate (rag_pipeline.py:111-221)         │
├───────────────────────────┬────────────────────────────┤
│ IF intent in subflows:    │ ELSE (intent=general_rag): │
│ Dispatch to Subflow       │ Execute Grounded RAG       │
│ (ProductRecommender)      │ (retrieve -> guardrail ->  │
│                           │  generator -> citations)   │
└─────────────┬─────────────┴─────────────┬──────────────┘
              │                           │
              ▼                           ▼
[product_recommender.py:recommend]  [retrieval.py:retrieve_fast]
      ↓                                   ↓
1. Alias normalization              [guardrails.py:evaluate_and_gate]
2. Static map lookup                      ↓
3. BM25 fallback (retrieve_fast)    [generator.py:generate_response]
4. Deterministic string template          ↓
              │                     [citation_engine.py:format_citations]
              ▼                           │
[FRONTEND JSON RESPONSE] ◄────────────────┘
```

### Execution Architecture Call Table

| Stage | File | Function / Class | Exact Responsibility & Behavior |
| :--- | :--- | :--- | :--- |
| **API** | `app.py` | `chat_endpoint()` (L140-169) | Parses incoming `QueryRequest`, measures wall-clock time, invokes `pipeline.query()`, returns structured JSON. |
| **Pipeline** | `src/rag_pipeline.py` | `BISRAGPipeline.query()` (L64-326) | Normalizes language, routes intent, and dispatches to specialized subflows or standard RAG. |
| **Router** | `src/router.py` | `QueryIntentRouter.classify_intent()` (L35-120) | Regex and keyword matching. Matches product patterns to `product_recommendation` (category: `product_standard_mapping`). |
| **Sanity Gate**| `src/guardrails.py` | `GuardrailGate.validate_subflow_compatibility()` (L113) | Basic compatibility check. Returns `(True, "OK")` for product recommendation. |
| **Product Flow**| `src/product_recommender.py` | `ProductRecommender.recommend()` (L84-325) | Applies alias normalization, queries static map, falls back to BM25, and constructs deterministic markdown response. |
| **Static Mapping**| `src/product_recommender.py` | `_find_best_match()` & `product_standard_map.json` | Scans 81 static records. Yields score < 40 for steel queries (miss). |
| **BM25 Retrieval**| `src/retrieval.py` | `HybridRetrievalPipeline.retrieve_fast()` (L593-645) | Lexical BM25 search across 41,476 chunks. Filters strictly by `category` parameter. |
| **LLM Generator** | `src/generator.py` | `GroundedGenerator.generate_response()` (L500-541) | Multi-provider abstraction (`OpenAIProvider`, `GeminiProvider`, `MockOfflineProvider`). **Bypassed completely** for product flows. In general RAG, falls back to `MockOfflineProvider` due to missing `openai` package. |
| **Response Formatter**| `src/product_recommender.py` | `recommend()` (L206-218, L306-320) | Deterministic Python f-string template generating markdown tables, headers, and official links. |

---

## 3. LLM Invocation Status

Forensic analysis of the server environment and code execution revealed:

```text
OpenAI provider initialized:            YES (instance created in GroundedGenerator)
OpenAI provider invoked by /api/chat:   NO
OpenAI provider invoked by ProductRec:  NO
OpenAI provider invoked after BM25:     NO
OpenAI provider invoked for subflows:   NO
OpenAI library installed in Python env: NO (ModuleNotFoundError: No module named 'openai')
```

### Detailed Invocation Mechanics
1. **Specialized Flows (ProductRecommender, SchemeWalkthrough, LabLocator, ConsumerComplaint):**
   In `src/rag_pipeline.py` lines 117–148, when `intent == "product_recommendation"`, the pipeline executes:
   ```python
   rec_res = self.product_recommender.recommend(search_query, language=detected_lang)
   if rec_res.get("status") == "success":
       return {
           "query": clean_query,
           "intent": intent,
           "flow_used": "product_recommender",
           "status": "success",
           "response": rec_res["formatted_text"], # Deterministic Markdown
           ...
       }
   ```
   `GroundedGenerator` is never touched. `generation_ms` is `None`. `model_used` is `None`.
2. **General RAG Flow (`general_rag`):**
   When a query routes to general RAG (e.g. *"What is IS 1786?"*), `GroundedGenerator.generate_response()` calls `self.provider.generate()`.
   However, inspecting `OpenAIProvider.generate()` (`src/generator.py:145`):
   ```python
   import openai
   client = openai.OpenAI(api_key=self.api_key, ...)
   ```
   At runtime, executing this import raises:
   `ModuleNotFoundError: No module named 'openai'`
   This exception is caught at line 534:
   ```python
   except Exception as e:
       log.warning(f"Generation with {self.provider.__class__.__name__} failed: {e}. Falling back to MockOfflineProvider.")
       fallback = MockOfflineProvider()
       return fallback.generate(...)
   ```
   Consequently, **the OpenAI API is never invoked in any execution branch of this system**.

---

## 4. End-to-End Trace of Failing Queries

### Query 1: Natural-Language Reinforcement Query
> *"What is the Indian Standard for high-strength deformed steel bars and wires for concrete reinforcement?"*

| Stage | Value / Observation |
| :--- | :--- |
| **Request Timestamp** | `2026-09-26T11:16:34.200Z` |
| **Router Decision** | `intent: "product_recommendation"`, `category: "product_standard_mapping"`, confidence: `0.95` |
| **Query Normalization** | Input matched `"steel bars"` in `PRODUCT_ALIASES`. Entire query replaced with `"High strength deformed steel bars"`. |
| **Static Map Lookup** | `_find_best_match("High strength deformed steel bars")` scored `20` against 81 records (threshold: `40`). Result: `MISS`. |
| **Fallback Decision** | Invoked `self.retrieval.retrieve_fast("High strength deformed steel bars", top_n=5, category="is_standard")`. |
| **BM25 Scope** | Strictly constrained to `category="is_standard"`. |
| **Top Retrieved Document**| Chunk `026044628f44569725bcecdd208ac72e7406c0bafb2bfa3382e76100cb06180d` from `raw_data/2e9cc92ed3b8_AIF-21-22.pdf`. |
| **Retrieved Metadata** | Standard: `IS 5489:1975`, Clause Title / Product: `kg/cm2 , Minimum Working pressure ` |
| **BM25 Score** | `16.28` |
| **Regex Match in Flow** | Regex `\b(IS\s*1786|IS\s*432|IS\s*2062)\b` scanned top 8 chunks. All chunks were unrelated fee/licence tables. Result: `None`. |
| **Selection Fallback** | Defaulted to `best_doc = corpus_hits[0]` (`IS 5489:1975`). |
| **LLM Call** | Bypassed. |
| **Final Answer** | Deterministic template recommending `IS 5489:1975` for `kg/cm2 , Minimum Working pressure `. |
| **Timing** | `retrieval_ms: None`, `generation_ms: None`, `total_ms: 21.41 ms`, `wall_clock_ms: 26.75 ms`. |

---

### Query 2: Manufacturer TMT Query
> *"I manufacture TMT steel bars. Which BIS standard should I refer to for reinforcement steel, and what should I check for compliance?"*

| Stage | Value / Observation |
| :--- | :--- |
| **Router Decision** | `intent: "product_recommendation"`, `category: "product_standard_mapping"`, confidence: `0.92` |
| **Query Normalization** | Input matched `"tmt bars"` in `PRODUCT_ALIASES`. Entire query replaced with `"High strength deformed steel bars"`. |
| **Static Map Lookup** | Same score `20` < `40` -> `MISS`. |
| **Convergence Point** | Because Query 1 and Query 2 both normalize to the exact same string `"High strength deformed steel bars"`, both follow the exact same fallback path into BM25 scoped to `category="is_standard"`. |
| **Final Standard** | **`IS 5489:1975`** |
| **Timing** | `total_ms: 19.33 ms`, `wall_clock_ms: 21.98 ms`. |

---

## 5. Trace of Direct Standard Queries (IS 1786)

To determine whether the failure belongs to Case A, B, C, or D:

| Query | Router Intent | Flow Used | Top Retrieved Standard | Final Output Standard | LLM Invoked | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `IS 1786` | `product_recommendation` | `product_recommender` | Direct IS regex extracted `IS 1786` | **IS 1786:2008** | NO | 38.27 ms |
| `What is IS 1786?` | `general_rag` | `general_rag` | Chunk `3010d227bd28` (`IS 1786:2008`) | **IS 1786:2008** | Mock Synthesizer | 52.32 ms |
| `IS 1786 steel reinforcement` | `general_rag` | `general_rag` | Chunk `3010d227bd28` (`IS 1786:2008`) | **IS 1786:2008** | Mock Synthesizer | 50.11 ms |
| `steel bars concrete reinforcement IS 1786` | `general_rag` | `general_rag` | Chunk `3010d227bd28` (`IS 1786:2008`) | **IS 1786:2008** | Mock Synthesizer | 48.74 ms |

### Diagnostic Conclusion: CASE A Confirmed
- Queries containing the literal code `"IS 1786"` either:
  1. Default to `general_rag`, where category prefiltering is broad/unconstrained and BM25 directly retrieves the correct specification chunk `3010d227bd28` (`IS 1786:2008`).
  2. Route to `product_recommendation`, where `ProductRecommender`'s explicit IS regex (`r"\bIS\s*(\d+)"`) catches `1786` directly.
- In contrast, natural language queries lacking the literal token `"IS 1786"` are captured by `PRODUCT_ALIASES`, collapsed to `"High strength deformed steel bars"`, and funneled into `retrieve_fast(category="is_standard")`, where the true IS 1786 chunk is filtered out.

---

## 6. Direct BM25 Index Evidence

The production BM25 index (`vector_index/bm25_index.pkl`) was queried across the 7 required test queries under two conditions:
- **Condition [A]:** Filtered by `category="is_standard"` (the ProductRecommender fallback behavior).
- **Condition [B]:** Unfiltered (`category=None`).

### BM25 Diagnostic Table

| Query | Condition | Rank 1 Standard | Rank 1 Title / Subject | Rank 1 BM25 Score | Relevance Classification |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **high-strength deformed steel bars** | **[A]** `is_standard` | **IS 5489:1975** | `kg/cm2 , Minimum Working pressure ` | 16.28 | **RETRIEVAL FALSE POSITIVE** |
| | **[B]** `None` | **IS 1786:2008** | `High strength deformed steel bars and wires for concrete reinforcement` | **25.10** | **HIGHLY RELEVANT (CORRECT)** |
| **steel bars concrete reinforcement** | **[A]** `is_standard` | **IS 5489:1975** | `kg/cm2 , Minimum Working pressure ` | 13.54 | **RETRIEVAL FALSE POSITIVE** |
| | **[B]** `None` | `IS 432 (Part 1)` | `Mild Steel and Medium Tensile Steel Bars... Concrete Reinforcement` | **21.65** | **HIGHLY RELEVANT (CORRECT)** |
| **TMT steel bars** | **[A]** `is_standard` | `IS 11952:1986` | `STEELS FOR PISTON PINS (GUDGEON PINS)` | 8.86 | UNRELATED |
| | **[B]** `None` | `IS 1786:2008` | `High strength deformed steel bars... (TMT)` | **12.44** | **HIGHLY RELEVANT (CORRECT)** |
| **IS 1786** | **[A]** `is_standard` | `IS 1` | `Marking fee listing (Polypropylene sacks)` | 3.86 | UNRELATED (FEE TABLE) |
| | **[B]** `None` | **IS 1786:2008** | `High strength deformed steel bars and wires for concrete reinforcement` | **15.32** | **HIGHLY RELEVANT (CORRECT)** |
| **Ordinary Portland Cement** | **[A]** `is_standard` | `IS 650:1991` | `STANDARD SAND FOR TESTING CEMENT` | 6.84 | PARTIALLY RELEVANT |
| | **[B]** `None` | **IS 1489 (Part 1)**| `PORTLAND POZZOLANA CEMENT (PPC) FLY ASH BASED` | **23.51** | **HIGHLY RELEVANT (CORRECT)** |
| **Portland cement** | **[A]** `is_standard` | `N/A` | `Slump, placing/compaction with vibrator` | 3.94 | IRRELEVANT |
| | **[B]** `None` | **IS 1489 (Part 1)**| `PORTLAND POZZOLANA CEMENT (PPC) FLY ASH BASED` | **23.63** | **HIGHLY RELEVANT (CORRECT)** |
| **IS 1489** | **[A]** `is_standard` | `IS 1` | `Marking fee listing` | 3.67 | IRRELEVANT |
| | **[B]** `None` | **IS 1489 (Part 2)**| `Portland pozzolana cement: Part 2 Calcined clay based` | **28.55** | **HIGHLY RELEVANT (CORRECT)** |

### Critical Takeaway from BM25 Testing
Under Condition [B] (unfiltered search), **BM25 ranks the true standard at Rank 1 with overwhelming lexical confidence scores (25.10, 21.65, 23.51, 28.55)**. BM25 is not failing to find the documents; **`category="is_standard"` is systematically filtering out the authoritative documents** because those documents were categorized as `general_policy` during data ingestion.

---

## 7. Product Standard Static Map Analysis

Inspection of `product_standard_map.json`:
- **Total records:** 81
- **Records matching `IS 1786`:** 0
- **Records matching `TMT`:** 0
- **Records matching `reinforcement`:** 0
- **Records matching `high strength`:** 0
- **Records matching `Portland cement`:** 0
- **Records matching `IS 1489`:** 0
- **Records matching `IS 5489`:** 0

### Query Normalization Audit in `ProductRecommender`
In `src/product_recommender.py`:
```python
PRODUCT_ALIASES = {
    "tmt bar": "High strength deformed steel bars",
    "tmt bars": "High strength deformed steel bars",
    "steel bar": "High strength deformed steel bars",
    "steel bars": "High strength deformed steel bars",
    "steel reinforcement": "High strength deformed steel bars and wires for concrete",
    "reinforcement steel": "High strength deformed steel bars and wires for concrete",
    "cement": "Ordinary Portland Cement",
}
```
`normalize_query_with_aliases()` searches for these substrings in order. Because `"steel bars"` occurs inside *"What is the Indian Standard for high-strength deformed steel bars and wires for concrete reinforcement?"*, it matches line 28 and discards the entire rest of the sentence, returning only `"High strength deformed steel bars"`.

When this normalized string is looked up in the static 81-record map, it scores 20 (minor overlap on the single word `"steel"` with unrelated products like `IS 1030:1998 Carbon steel castings`). Since 20 is below the threshold of 40, static lookup fails, forcing the query into the flawed BM25 fallback.

---

## 8. Forensic Investigation of the `IS 5489:1975` Failure

The exact origin of `IS 5489:1975` was isolated:

- **Source File:** `raw_data/2e9cc92ed3b8_AIF-21-22.pdf`
- **Source URL:** `https://www.bis.gov.in/wp-content/uploads/2022/08/AIF-21-22.pdf`
- **Chunk ID:** `026044628f44569725bcecdd208ac72e7406c0bafb2bfa3382e76100cb06180d`
- **Assigned Category:** `is_standard`
- **Assigned Standard Number:** `IS 5489`
- **Assigned Revision Year:** `1975`
- **Assigned Clause Title:** `kg/cm2 , Minimum Working pressure `
- **Actual Document Nature:** All India Financial / Annual Fee & Licensing Compendium listing hundreds of granted factory licences across all industries in India.
- **Chunk Text Excerpt:**
  ```text
  :?0.75 kg/cm2 , Nominal Flowrate - at nominal working pressure : 220 l/h/m?, at minimum working pressure : 190 l/h/m , at maximum working pressure?: 270 l/h/m , 2022-05-05 FRBO 9512481323 IS 4454 : PART 4 : 2001 TEEL WIRES FOR MECHANICAL SPRINGS... Hot Rolled Round Bars of Size: 5 mm to 100 mm...
  ```
- **Why BM25 selected it:** When queried with `"High strength deformed steel bars"` under `category="is_standard"`, the true steel standards (`IS 1786`) were inaccessible because they are in `general_policy`. In the remaining pool, this fee table happened to contain the words `"steel"`, `"bars"`, and `"wires"`, producing a top score of 16.28.
- **Classification:** **`RETRIEVAL FALSE POSITIVE`** resulting from ingestion misclassification of an administrative fee compendium as `is_standard`, amplified by restrictive runtime category filtering.

---

## 9. Response Classification: Static vs Generated

| Query Route | Response Generation Mechanism | Classification |
| :--- | :--- | :---: |
| `product_recommendation` (Steel, TMT, Cement) | Deterministically assembled via string concatenation and f-strings in `product_recommender.py` lines 206–218, 306–320. | **B — Deterministically formatted from retrieved evidence** |
| `certification_process` (Schemes) | Deterministic dictionary lookup from `scheme_walkthrough.py`. | **B — Deterministically formatted** |
| `lab_network` (Labs) | Deterministic JSON/LIMS formatter in `lab_locator.py`. | **B — Deterministically formatted** |
| `consumer_grievance` (Complaints) | Deterministic decision tree in `consumer_complaint.py`. | **B — Deterministically formatted** |
| `general_rag` (Direct standards, conceptual) | In-memory keyword extraction and sentence synthesis in `MockOfflineProvider` (`generator.py:313-388`). | **D — Mixture of deterministic formatting + local heuristic synthesis** |

**Conclusion:** OpenAI `gpt-4o-mini` is **0% involved** in generating the responses for either the failing queries or the specialized subflows.

---

## 10. Timing and Caching Forensics

### Timing Benchmark Across 5 Controlled Queries

| Query | Flow Used | `retrieval_ms` | `generation_ms` | `total_ms` | Wall-Clock Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1. `IS 1786` | `product_recommender` | `None` | `None` | 38.27 ms | 117.92 ms |
| 2. `What is IS 1786?` | `general_rag` | 91.50 ms | 1.28 ms | 93.95 ms | 96.88 ms |
| 3. `What is the Indian Standard for high-strength deformed steel bars...` | `product_recommender` | `None` | `None` | 24.25 ms | 26.75 ms |
| 4. `What is the standard number for Portland cement?` | `product_recommender` | `None` | `None` | 43.70 ms | 46.46 ms |
| 5. `What is BIS certification?` | `general_rag` | 50.22 ms | 1.60 ms | 53.04 ms | 55.48 ms |

### Caching Audit
- **Application-Level Cache:** None found in `app.py` or `rag_pipeline.py`.
- **Response Caching:** Testing repeated identical requests demonstrated identical pipeline processing durations (~21–24 ms), proving queries are computed fresh on each call.
- **Speed Explanation:** The speed is due to in-memory Python execution of BM25 (via `rank_bm25` / sparse inverted index) and complete bypass of external network requests (no API calls to OpenAI or remote servers).

---

## 11. Mock vs Real LLM Analysis

1. **MockEmbeddingModel:** Initialized during server startup when `use_fast_retrieval=True`. This is intentional: fast retrieval uses sparse BM25 exclusively and deliberately disables the 500 MB BGE-M3 neural transformer model to keep latency under 50 ms.
2. **OpenAI Provider Status:**
   - `.env` contains an `OPENAI_API_KEY` entry.
   - `GroundedGenerator` attempts to instantiate `OpenAIProvider`.
   - However, the `openai` Python package is not installed in the active environment.
   - In `general_rag`, any call to `OpenAIProvider.generate()` fails immediately with `ModuleNotFoundError` and is silently caught, redirecting to `MockOfflineProvider(model_name="mock-grounded-synthesizer")`.
   - In `product_recommender`, no LLM provider is ever called.

---

## 12. Verification of Existing Tests

The complete existing test suite was executed:
- `tests/test_retrieval.py`
- `tests/test_phase4.py`
- `tests/test_phase5.py`
- `tests/test_feedback.py`
- `tests/test_adversarial.py`

**Result:** **51 PASSED, 0 FAILED, 0 SKIPPED** (in 18.84s).

### Root Cause of Discrepancy Between Unit Tests and Runtime Failure
In `tests/test_phase4.py`, `test_04_product_corpus_fallback` passes because it uses `MockRetrievalFallback` (`tests/test_phase4.py:29-49`), which injects a synthetic chunk:
```python
"chunk_id": "chunk_std_1786",
"is_number": "IS 1786",
"clause_title": "High Strength Deformed Steel Bars",
"category": "is_standard"  # Synthetically given is_standard category!
```
In the actual production BM25 index, no chunk with `is_number: "IS 1786"` has `category: "is_standard"`. The unit test masked the runtime defect by providing synthetic data that conformed to the application's faulty filtering assumptions.

---

## 13. Root Cause Summary

The primary root cause of the observed failure is an **Application-Level Query Normalization and Category Scoping Defect**:

1. **Primary Cause:** In `src/product_recommender.py` line 221, corpus retrieval is artificially restricted to `category="is_standard"`. Because the authentic standard overview document (`news_2024-10-01_114.pdf`) containing `IS 1786:2008` was categorized as `general_policy` during data ingestion, the category filter strictly prevents BM25 from seeing it.
2. **Secondary Cause:** `PRODUCT_ALIASES` in `src/product_recommender.py` truncates specific multi-word natural language queries down to `"High strength deformed steel bars"`, discarding discriminating tokens.
3. **Tertiary Cause:** Administrative fee and licence listings (`AIF-21-22.pdf`) were ingested with `category="is_standard"` and mis-attributed standard numbers (`IS 5489`), creating noisy false-positive candidates.
4. **Architectural LLM Bypass:** `ProductRecommender` formats responses deterministically and never invokes an LLM to reason over or correct retrieved context.

---

## 14. Database / Vector Database (Qdrant) Decision

### Evaluation of Questions A–F:
- **Question A (Data storage problem?):** NO. All 41,476 chunks are present in `processed_chunks.jsonl` and indexed in `bm25_index.pkl`.
- **Question B (Query understanding problem?):** NO. When BM25 is queried without category restrictions, it achieves a score of **25.10** for `IS 1786:2008` at Rank 1.
- **Question C (Routing problem?):** YES. Natural language standard questions are routed to deterministic `product_recommender` instead of `general_rag`.
- **Question D (Mapping problem?):** YES. `PRODUCT_ALIASES` collapses queries, and `product_standard_map.json` lacks the standard.
- **Question E (Grounding/generation problem?):** NO. The LLM was never called.
- **Question F (Infrastructure problem / Would Qdrant solve this?):** **NO**. If Qdrant were introduced with the same `category="is_standard"` filter or the same alias normalization, Qdrant would also fail to retrieve the correct chunk. Conversely, removing the category restriction in the existing BM25 retriever immediately retrieves `IS 1786:2008` at Rank 1 in 15 ms without any vector database.

### Decision Category:
```text
QDRANT NOT JUSTIFIED
```
**Justification:** The failure is 100% attributable to application-layer alias normalization, missing records in `product_standard_map.json`, and artificial category pre-filtering in `ProductRecommender`. BM25 already achieves 100% precision on this query when the application-level filtering bug is removed. Introducing Qdrant would add unnecessary architectural complexity without addressing the actual bug.

---

## 15. Recommended Next Steps (For Future Remediation — Not Implemented)

1. **Remove artificial category pre-filtering in `ProductRecommender`:** Modify `ProductRecommender.recommend()` to search `category=None` or fall back to broad search for steel/TMT products, mirroring the existing cement fallback at line 222.
2. **Expand `product_standard_map.json`:** Add verified mappings for high-priority national standards (`IS 1786` for TMT/reinforcement steel, `IS 269` / `IS 1489` for cement).
3. **Refine `PRODUCT_ALIASES`:** Prevent alias normalization from truncating specific user queries when technical modifiers like *"concrete reinforcement"* are present.
4. **Fix Environment Dependencies:** If LLM generation is desired for general RAG, install the official `openai` client library in the Python environment.

---

## 16. Final Verdict

```text
RAG FORENSIC AUDIT — LLM BYPASS CONFIRMED

ROOT CAUSE:
ProductRecommender bypasses the LLM and retrieves erroneous standard IS 5489:1975 because query alias collapsing and a hardcoded category='is_standard' filter strictly exclude the authentic IS 1786 specification chunk from BM25 retrieval.

QDRANT DECISION:
QDRANT NOT JUSTIFIED

NO FILES MODIFIED:
YES
```
