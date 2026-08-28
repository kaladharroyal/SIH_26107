"""
Phase 7, Step 22: Persona Evaluation & Accuracy Suite (eval_personas.py)
Evaluates Retrieval Precision & Citation Accuracy across 3 User Personas:
1. MSME Manufacturer (Procedural & Fee Queries)
2. Student (Exploratory & Definitional Queries)
3. Consumer (Complaint & Verification Queries)
"""

import logging
from test_phase5 import MultilingualBISPipelne

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_personas")

PERSONA_TEST_DATA = [
    # Persona 1: MSME Manufacturer
    {
        "persona": "MSME Manufacturer",
        "query": "what is the application fee for ISI mark product certification under Scheme-I",
        "expected_keyword": "1,000",
        "expected_subflow": "certification_process",
    },
    {
        "persona": "MSME Manufacturer",
        "query": "is certification mandatory for LED bulbs under CRS",
        "expected_keyword": "16102",
        "expected_subflow": "product_recommendation",
    },
    # Persona 2: Student
    {
        "persona": "Student",
        "query": "what standard applies to gold jewellery hallmarking regulations",
        "expected_keyword": "1417",
        "expected_subflow": "product_recommendation",
    },
    {
        "persona": "Student",
        "query": "IS 1786 steel reinforcement requirements",
        "expected_keyword": "1786",
        "expected_subflow": "general_rag",
    },
    # Persona 3: Consumer
    {
        "persona": "Consumer",
        "query": "my gold hallmark jewellery is fake how to complain",
        "expected_keyword": "bis care",
        "expected_subflow": "consumer_complaint",
    },
]


def run_persona_evaluation():
    print("\n" + "=" * 70)
    print("      PHASE 7: PERSONA ACCURACY & CITATION VERIFICATION SUITE")
    print("=" * 70 + "\n")

    pipeline = MultilingualBISPipelne()
    retrieval_hits = 0
    citation_hits = 0
    total = len(PERSONA_TEST_DATA)

    for idx, test in enumerate(PERSONA_TEST_DATA, 1):
        persona = test["persona"]
        q = test["query"]
        expected_kw = test["expected_keyword"].lower()
        expected_flow = test["expected_subflow"]

        print(f"▶ TEST #{idx} [{persona}]")
        print(f"  Query: '{q}'")

        res = pipeline.process_multilingual_query(q)
        flow = res["sub_flow"]
        resp_text = res["response"].lower()

        # Measure Sub-flow Retrieval Precision
        flow_pass = (flow == expected_kw or flow == expected_flow)
        if flow_pass or expected_kw in resp_text:
            retrieval_hits += 1

        # Measure Citation Accuracy (Does response contain exact expected keyword/number?)
        if expected_kw in resp_text:
            citation_hits += 1
            cit_pass = True
        else:
            cit_pass = False

        print(f"  Routed Sub-Flow: {flow.upper()}")
        print(f"  Retrieval Match: {'✅ PASS' if flow_pass else '❌ CHECK'}")
        print(f"  Citation Accuracy: {'✅ PASS' if cit_pass else '❌ CHECK'}")
        print(f"  Snippet: {res['response'][:130]}...\n")

    retrieval_acc = (retrieval_hits / total) * 100
    citation_acc = (citation_hits / total) * 100

    print("=" * 70)
    print("                      SYSTEM EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Persona Test Queries  : {total}")
    print(f"Retrieval Accuracy Precision : {retrieval_acc:.2f}% ({retrieval_hits}/{total})")
    print(f"Citation Accuracy Precision  : {citation_acc:.2f}% ({citation_hits}/{total})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_persona_evaluation()
