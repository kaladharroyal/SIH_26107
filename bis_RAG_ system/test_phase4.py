"""
Phase 4: Integrated Pipeline & Validation Test Suite (test_phase4.py)
Tests Intent Routing and Sub-Flow Execution across Product Recommendation, Scheme Walkthroughs,
Lab Locators, Consumer Complaints, and General RAG queries.
"""

import logging
from router import QueryIntentRouter
from product_recommender import ProductRecommender
from scheme_walkthrough import SchemeWalkthroughGuide
from lab_locator import LabLocator
from consumer_complaint import ConsumerComplaintHandler
from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_phase4")


class Phase4Orchestrator:
    def __init__(self):
        log.info("Initializing Phase 4 Integrated Sub-Flow Pipeline...")
        self.router = QueryIntentRouter()
        self.recommender = ProductRecommender()
        self.walkthrough_guide = SchemeWalkthroughGuide()
        self.lab_locator = LabLocator()
        self.complaint_handler = ConsumerComplaintHandler()
        self.rag_pipeline = BISRAGPipeline()

    def process_query(self, user_query: str) -> dict:
        # 1. Intent Classification
        intent_info = self.router.classify_intent(user_query)
        intent = intent_info["intent"]

        log.info(f"Orchestrator routing query '{user_query}' to sub-flow: {intent}")

        # 2. Dispatch to specialized sub-flow
        if intent == "product_recommendation":
            res = self.recommender.recommend(user_query)
            return {"intent": intent, "response": res["formatted_text"], "source": "product_recommender"}

        elif intent == "certification_process":
            res = self.walkthrough_guide.get_walkthrough(user_query)
            return {"intent": intent, "response": res["formatted_text"], "source": "scheme_walkthrough"}

        elif intent == "lab_location":
            res = self.lab_locator.search_labs(user_query)
            return {"intent": intent, "response": res["formatted_text"], "source": "lab_locator"}

        elif intent == "consumer_complaint":
            res = self.complaint_handler.handle_complaint(user_query)
            return {"intent": intent, "response": res["formatted_text"], "source": "consumer_complaint"}

        else:
            # General RAG pipeline fallback (Phases 1-3)
            res = self.rag_pipeline.query(user_query)
            return {"intent": "general_rag", "response": res["response"], "source": "rag_pipeline"}


def run_phase4_test_suite():
    print("\n" + "=" * 70)
    print("      PHASE 4: SPECIALIZED SUB-FLOWS & ROUTER TEST SUITE")
    print("=" * 70 + "\n")

    orchestrator = Phase4Orchestrator()

    test_cases = [
        {"name": "Product Recommendation", "query": "is certification mandatory for LED bulbs", "expected_intent": "product_recommendation"},
        {"name": "Certification Scheme Walkthrough", "query": "how to apply for ISI mark under Scheme-I", "expected_intent": "certification_process"},
        {"name": "Lab Directory Search", "query": "testing labs in Delhi for steel IS 1786", "expected_intent": "lab_location"},
        {"name": "Consumer Complaint Handling", "query": "my gold hallmark jewellery is fake how to complain", "expected_intent": "consumer_complaint"},
        {"name": "General Technical RAG Query", "query": "what is the dielectric strength requirement for appliances", "expected_intent": "general_rag"},
    ]

    passed = 0
    total = len(test_cases)

    for test in test_cases:
        name = test["name"]
        q = test["query"]
        expected = test["expected_intent"]

        print(f"▶ TEST: {name}")
        print(f"  Query: '{q}'")

        res = orchestrator.process_query(q)
        intent = res["intent"]

        print(f"  Routed Sub-Flow: {intent.upper()} (Source: {res['source']})")
        print(f"  Response Snippet: {res['response'][:140]}...")

        if intent == expected:
            passed += 1
            print("  Result: ✅ PASSED\n")
        else:
            print(f"  Result: ❌ FAILED (Expected '{expected}', got '{intent}')\n")

    pass_rate = (passed / total) * 100
    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} Phase 4 Sub-Flow Tests Passed ({pass_rate:.2f}% Pass Rate)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_phase4_test_suite()
