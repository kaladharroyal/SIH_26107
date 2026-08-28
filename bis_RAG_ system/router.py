"""
Phase 4, Step 16: Intent Router & Sub-Flow Dispatcher (router.py)
Classifies user queries into 5 discrete intent sub-flows:
1. product_recommendation ("is certification mandatory for LED bulbs")
2. certification_process ("how to apply for ISI mark under Scheme-I")
3. lab_location ("labs in Delhi for IS 1786 steel testing")
4. consumer_complaint ("my gold hallmark is fake how to complain")
5. general_rag (general technical/regulatory RAG queries)
"""

import re
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("intent_router")


class QueryIntentRouter:
    def __init__(self):
        # Rule-based regex patterns for fast, deterministic intent routing
        self.patterns = {
            "product_recommendation": [
                r"\b(product|standard|mandatory|compulsory|apply|applies|require|required|need|is\s*\d+)\b.*\b(certification|licence|license|qco|crs|scheme)\b",
                r"\b(do i need|which standard|is it mandatory|compulsory certification|scheme-x)\b",
                r"\b(bulb|steel|toy|helmet|battery|pv module|solar|cable|cement|valve|water)\b.*\b(standard|mandatory|certify)\b",
            ],
            "certification_process": [
                r"\b(how to apply|apply online|application process|licensing procedure|grant of licence|scheme-i|scheme-ii|fmcs|walkthrough|steps to get)\b",
                r"\b(fee|cost|application fee|inspection fee|man-day|validity|renewal|documents required)\b.*\b(process|apply)\b",
            ],
            "lab_location": [
                r"\b(lab|laboratory|testing facility|where to test|test scope|testing centre|ahc|assaying)\b",
                r"\b(labs in|testing in|recognized lab|empaneled lab)\b",
            ],
            "consumer_complaint": [
                r"\b(complaint|complain|fake|defective|fraud|shortfall|compensation|underweight|purity|bis care|rights|consumer protection)\b",
            ],
        }

    def classify_intent(self, query: str) -> Dict[str, Any]:
        q_clean = query.strip().lower()

        for intent, regex_list in self.patterns.items():
            for pattern in regex_list:
                if re.search(pattern, q_clean, re.IGNORECASE):
                    log.info(f"Intent Classifier matched '{intent}' for query: '{query}'")
                    return {"intent": intent, "pattern_matched": pattern, "confidence": 0.95}

        log.info(f"Intent Classifier defaulted to 'general_rag' for query: '{query}'")
        return {"intent": "general_rag", "pattern_matched": None, "confidence": 0.80}


if __name__ == "__main__":
    router = QueryIntentRouter()
    queries = [
        "is certification mandatory for LED bulbs",
        "how to apply for ISI mark under Scheme-I",
        "where are the recognized testing labs in Delhi",
        "my gold hallmark jewellery is fake how to complain",
        "what is the dielectric strength requirement for appliances",
    ]
    for q in queries:
        res = router.classify_intent(q)
        print(f"Query: '{q}' -> Intent: {res['intent']} ({res['confidence']})")
