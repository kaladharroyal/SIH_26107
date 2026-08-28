# 🎯 Phase 4: Specialized Sub-Flows — Detailed Technical Implementation Plan

This document outlines the step-by-step implementation plan for **Phase 4 (Specialized Sub-Flows)**.

---

## 🏗️ Phase 4 System Architecture & Sub-Flow Routing

```
                          ┌──────────────────────────┐
                          │   Incoming User Query    │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                        ┌──────────────────────────────┐
                        │ Intent Router (`router.py`)  │
                        └───────────────┬──────────────┘
                                        │
       ┌────────────────────┬───────────┼───────────┬────────────────────┐
       │                    │           │           │                    │
       ▼                    ▼           ▼           ▼                    ▼
┌──────────────┐    ┌──────────────┐ ┌──────┐ ┌─────────────┐    ┌──────────────┐
│ Product      │    │ Certification│ │ Lab  │ │ Consumer    │    │ General RAG  │
│ Recommender  │    │ Walkthrough  │ │Locat.│ │ Complaints  │    │ Pipeline     │
│ (Step 12)    │    │ (Step 13)    │ │(14)  │ │ (Step 15)   │    │ (Phases 1-3) │
└──────┬───────┘    └──────┬───────┘ └──┬───┘ └──────┬──────┘    └──────┬───────┘
       │                   │            │            │                  │
       └───────────────────┴─────┬──────┴────────────┴──────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Structured Response / │
                     │ Grounded Answer Output│
                     └───────────────────────┘
```

---

## 🛠️ Detailed Component Specifications & Module Breakdown

### 1. **Intent Router & Sub-Flow Dispatcher (`router.py`) — Step 16**
- **Objective**: Classify incoming query into 5 discrete intents before execution:
  - `product_recommendation`: Queries asking if a product requires certification (e.g. *"is certification mandatory for LED bulbs"*).
  - `certification_process`: Queries asking how to apply or steps for a scheme (e.g. *"how to apply for ISI mark under Scheme-I"*).
  - `lab_location`: Queries asking where to test a product or find labs (e.g. *"labs in Delhi that test IS 1786 steel"*).
  - `consumer_complaint`: Queries about defective products/hallmarking (e.g. *"my gold hallmark is fake how to complain"*).
  - `general_rag`: Unstructured technical/regulatory queries routed to Phase 3 RAG.
- **File**: `router.py`

### 2. **Product → Standard Recommender (`product_recommender.py`) — Step 12**
- **Objective**: Deterministic tabular lookup against `product_standard_mapping.json` (0% hallucination).
- **Matching Logic**:
  - Exact & Fuzzy string matching + Keyword aliases (e.g. `"solar panel"` -> `"Crystalline Silicon Terrestrial Photovoltaic Module"`).
- **Output Data**:
  - `is_number` (e.g. `IS 14286`), `mandatory_status` (`MANDATORY_QCO` / `MANDATORY_CRS` / `VOLUNTARY`), `scheme_name` (`Scheme-I` / `CRS` / `Scheme-X`), and `official_notification_ref`.
- **File**: `product_recommender.py`

### 3. **Certification Scheme Walkthrough Guide (`scheme_walkthrough.py`) — Step 13**
- **Objective**: Pre-built structured step-by-step application walkthroughs for Scheme-I (ISI), Scheme-II (CRS), FMCS, and Scheme-X.
- **Embedded Compliance Figures**:
  - Application Fee: **₹1,000**
  - Inspection Fee: **₹7,000 per man-day**
  - Test Report Validity: **90 Days**
  - Renewal Cycle: **2 Years**
- **File**: `scheme_walkthrough.py`

### 4. **Lab Locator Engine (`lab_locator.py`) — Step 14**
- **Objective**: Filterable laboratory search querying `lims_recognized_labs.json` and `lims_empaneled_labs.json`.
- **Filtering Parameters**:
  - `state` / `city` (e.g. `Delhi`, `Maharashtra`, `Guntur`).
  - `is_number` (e.g. `IS 1786`, `IS 302`).
  - `lab_type` (`RECOGNIZED` / `EMPANELED` / `CENTRAL`).
- **File**: `lab_locator.py`

### 5. **Consumer Complaint & Rights Router (`consumer_complaint.py`) — Step 15**
- **Objective**: Direct consumer complaint queries to official BIS channels with exact legal compensation rights.
- **Compensation Formula**:
  - For underweight / low-purity hallmarked gold: **2x the purity shortfall value + testing fees**.
- **Official Channel Routing**:
  - BIS CARE Mobile App & `manakonline.in` Standard Promotion Portal.
- **File**: `consumer_complaint.py`

### 6. **Phase 4 Integrated Pipeline & Test Suite (`test_phase4.py`)**
- **Objective**: Connect Intent Router -> Sub-Flow Execution -> Integrated Response.
- **File**: `test_phase4.py`

---

## 📋 Task Allocation & Target Files

| Module File | Target Feature | Primary Task |
|---|---|---|
| [`router.py`](file:///d:/kaladharroyal/bis_RAG_%20system/router.py) | Issue 4.1 | Query Intent Classification & Sub-Flow Routing |
| [`product_recommender.py`](file:///d:/kaladharroyal/bis_RAG_%20system/product_recommender.py) | Issue 4.2 | Deterministic Product-to-Standard Matching |
| [`scheme_walkthrough.py`](file:///d:/kaladharroyal/bis_RAG_%20system/scheme_walkthrough.py) | Issue 4.2 | Structured Step-by-Step Scheme Walkthroughs |
| [`lab_locator.py`](file:///d:/kaladharroyal/bis_RAG_%20system/lab_locator.py) | Issue 4.2 | Filterable LIMS Laboratory Locator |
| [`consumer_complaint.py`](file:///d:/kaladharroyal/bis_RAG_%20system/consumer_complaint.py) | Issue 4.2 | Consumer Rights & BIS CARE App Router |
| [`test_phase4.py`](file:///d:/kaladharroyal/bis_RAG_%20system/test_phase4.py) | Phase 4 Validation | End-to-End Sub-Flow Validation Test Suite |
