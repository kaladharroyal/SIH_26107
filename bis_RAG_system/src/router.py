"""
Phase 3, Part 3.5: Query Intent Router & Taxonomy Mapping (router.py)
Classifies user queries into discrete intent sub-flows and maps them
directly to verified Phase 1/2 corpus category partitions.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("intent_router")

# Mapping of detected user intents to verified Phase 1/2 corpus taxonomy
INTENT_TO_CORPUS_CATEGORY: Dict[str, Optional[str]] = {
    "product_recommendation": "product_standard_mapping",
    "lab_location": "lab_directory",
    "certification_process": "general_policy",
    "consumer_complaint": "general_policy",
    "general_rag": None,  # None means unconstrained multi-category search
}


class QueryIntentRouter:
    """
    Classifies user queries using fast, deterministic regex patterns
    and maps intents to corpus category partitions.
    """

    def __init__(self, category_mapping: Optional[Dict[str, Optional[str]]] = None):
        self.mapping = category_mapping or INTENT_TO_CORPUS_CATEGORY
        self.patterns: Dict[str, List[str]] = {
            "certification_process": [
                r"\b(how to apply|apply online|application process|licensing procedure|grant of licence|scheme-i|scheme-ii|fmcs|walkthrough|steps to get)\b",
                r"\b(fee|cost|application fee|inspection fee|man-day|validity|renewal|documents required)\b.*\b(process|apply|licence|certification)\b",
                r"\b(process for product certification|grant of license|renewal of license)\b",
            ],
            "lab_location": [
                r"\b(lab|laboratory|testing facility|where to test|test scope|testing centre|ahc|assaying|recognized lab|empaneled lab)\b",
                r"\b(labs in|testing in|testing laboratories)\b",
            ],
            "consumer_complaint": [
                r"\b(complaint|complain|fake|defective|fraud|shortfall|compensation|underweight|purity|bis care|rights|consumer protection)\b",
            ],
            "product_recommendation": [
                r"\b(product|standard|mandatory|compulsory|applies|require|required|need|is\s*\d+)\b.*\b(certification|licence|license|qco|crs|scheme)\b",
                r"\b(certification|licence|license|qco|crs|scheme)\b.*\b(product|standard|mandatory|compulsory|applies|require|required|need|is\s*\d+)\b",
                r"\b(do i need|which standard|is it mandatory|compulsory certification|scheme-x|isi mark for)\b",
                r"\b(bulb|steel|toy|helmet|battery|pv module|solar|cable|cement|valve|water|gold|jewellery|reinforcement)\b.*\b(standard|mandatory|certify|applicable|module|product)\b",
                r"\b(standard|mandatory|certify|applicable)\b.*\b(bulb|steel|toy|helmet|battery|pv module|solar|cable|cement|valve|water|gold|jewellery|reinforcement)\b",
            ],
        }

    def get_category_for_intent(self, intent: str) -> Optional[str]:
        """Returns the corresponding corpus partition category for a given intent."""
        return self.mapping.get(intent, None)

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classifies query intent and provides corresponding corpus category filter.
        """
        if not query or not query.strip():
            return {
                "intent": "general_rag",
                "category": None,
                "confidence": 0.50,
                "pattern_matched": None,
            }

        q_clean = query.strip().lower()

        for intent, regex_list in self.patterns.items():
            for pattern in regex_list:
                if re.search(pattern, q_clean, re.IGNORECASE):
                    category = self.get_category_for_intent(intent)
                    log.info(f"Intent Router matched '{intent}' (category: '{category}') for query: '{query[:40]}'")
                    return {
                        "intent": intent,
                        "category": category,
                        "confidence": 0.92,
                        "pattern_matched": pattern,
                    }

        log.info(f"Intent Router defaulted to 'general_rag' (category: None) for query: '{query[:40]}'")
        return {
            "intent": "general_rag",
            "category": None,
            "confidence": 0.75,
            "pattern_matched": None,
        }


if __name__ == "__main__":
    router = QueryIntentRouter()
    test_queries = [
        "is certification mandatory for LED bulbs",
        "how to apply for ISI mark under Scheme-I",
        "where are testing laboratories in Delhi",
        "my hallmark gold is fake how to complain",
        "IS 1786 steel reinforcement requirements",
    ]
    for q in test_queries:
        res = router.classify_intent(q)
        print(f"Query: '{q}'\n -> Intent: {res['intent']} | Category: {res['category']} (Conf: {res['confidence']})\n")
