# 🎭 Bureau of Indian Standards (BIS) AI Assistant — Persona Demo Script

This script provides a step-by-step walkthrough for presenting the BIS AI Assistant to hackathon judges across 4 distinct user personas.

---

## 👤 Persona 1: MSME Manufacturer (Product Certification & Walkthrough)
- **Goal**: Check if a product needs certification and find out how to apply.
- **Demo Prompt**: `"mera LED bulb ke liye BIS certification chahiye"` (Hinglish Query)
- **Expected System Behavior**:
  1. **Multilingual Handler**: Detects Hinglish, normalizes to *"need BIS certification for LED bulb"*.
  2. **Intent Router**: Identifies `product_recommendation` intent.
  3. **Product Recommender**: Returns deterministic badge: 🛑 **MANDATORY_CRS** under **IS 16102:2012**.
  4. **Scheme Walkthrough**: Displays step-by-step CRS procedure with exact fee figures (**₹1,000 application fee**, **90-day test report validity**).
  5. **Citation Badge**: Clickable link to official CRS portal.

---

## 👤 Persona 2: Civil Engineer / Architect (Standard Regulation Search)
- **Goal**: Find technical requirements for steel reinforcement.
- **Demo Prompt**: `"IS 1786 steel reinforcement requirements"`
- **Expected System Behavior**:
  1. **Hybrid Retrieval**: Runs BGE-M3 Dense + BM25 Sparse search.
  2. **Refusal Gate**: Confirms high confidence score (0.92 >= 0.45).
  3. **Grounded Generation**: Cites exact requirements with 0% hallucination.
  4. **Dual Citation Badge**: 📜 `[As per IS 1786:2008, Clause 4.2](file:///...#page=6)`.

---

## 👤 Persona 3: Consumer Protection (Fake Hallmark Complaint)
- **Goal**: Report fake gold jewellery hallmarking and know legal rights.
- **Demo Prompt**: `"my gold hallmark jewellery is fake how to complain"`
- **Expected System Behavior**:
  1. **Intent Router**: Directs to `consumer_complaint` sub-flow.
  2. **Legal Compensation Right**: Surfaces official formula: **2x purity shortfall value + testing fees**.
  3. **Official Channel Routing**: Provides direct download link for **BIS CARE App** & official complaint portal.

---

## 👤 Persona 4: Out-of-Corpus Query (Hallucination Prevention Refusal)
- **Goal**: Test system guardrails against unsupported queries.
- **Demo Prompt**: `"what is the property tax rate in Tokyo Japan"`
- **Expected System Behavior**:
  1. **Refusal Gate**: Fires refusal trigger (Confidence `0.12 < 0.45`).
  2. **Safe Fallback**: Refuses to generate fake answers and directs to official BIS Portal.
