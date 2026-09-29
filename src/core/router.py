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
                r"\b(how to get|how do i get|how to apply|apply online|application process|licensing procedure|grant of licence|grant of license|scheme\s*[-–]?\s*[i|ii|x|1|2]|fmcs|walkthrough|steps to get|get bis certification|स्कीम|योजना|స్కీమ్|పథకం|திட்டம்|স্কিম)\b",
                r"\b(fee|cost|application fee|inspection fee|man-day|validity|renewal|documents required)\b.*\b(process|apply|licence|certification)\b",
                r"\b(process for product certification|grant of license|renewal of license)\b",
            ],
            "lab_location": [
                r"\b(testing\s+laboratories|testing\s+laboratory|testing\s+labs|testing\s+lab|laboratories|laboratory|labs|lab|testing\s+facilit(?:y|ies)|where\s+to\s+test|test\s+scope|testing\s+centres?|ahc|assaying|recognized\s+labs?|empaneled\s+labs?|प्रयोगशाला|परीक्षण|ప్రయోగశాల|పరీక్ష|ஆய்வகம்|பரிசோதனை|পরীক্ষাগার|ল্যাব)\b",
                r"\b(labs\s+in|testing\s+in|laboratories\s+in)\b",
            ],
            "consumer_complaint": [
                r"\b(complaint|complain|fake|defective|fraud|shortfall|compensation|underweight|purity|bis care|rights|consumer protection)\b",
            ],
            "product_recommendation": [
                r"^\s*is\s*\d+(?::\d+)?\s*$",
                r"\b(product|standard|is\s*\d+)\b.*\b(mandatory\s+certification|licence|license|qco|crs\s+standard|scheme\s*[-–]?\s*[i|ii|x|1|2])\b",
                r"\b(mandatory\s+certification|licence|license|qco|crs\s+standard|scheme\s*[-–]?\s*[i|ii|x|1|2])\b.*\b(product|standard|is\s*\d+)\b",
                r"\b(do i need|which standard|what standard|what bis standard|applicable standard|standard for|standards for|standard should i use|compulsory certification|scheme-x|isi mark for)\b",
                r"\b(bulb|steel|toy|helmet|battery|pv module|solar|cable|cement|valve|water|gold|jewellery|reinforcement|tmt|tmt bar|স্টিল|ইস্পাত|রিইনফোর্সমেন্ট)\b.*\b(standard|mandatory|certify|applicable|module|product|use)\b",
                r"\b(standard|mandatory|certify|applicable|use)\b.*\b(bulb|steel|toy|helmet|battery|pv module|solar|cable|cement|valve|water|gold|jewellery|reinforcement|tmt|tmt bar|স্টিল|ইস্পাত|রিইনফোর্সমেন্ট)\b",
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

        # F6: Prioritize explicit product-standard inquiries over generic scheme tokens (e.g. "which standard for cement under Scheme-I", "IS 12860")
        explicit_standard_pattern = (
            r"(?:^\s*is\s*\d+(?::\d+)?\s*$|\b("
            r"(?:which|what)\s+(?:is\s+)?(?:the\s+)?(?:bis\s+)?standard|"
            r"standard\s+applies|"
            r"which\s+standard\s+applies|"
            r"what\s+standard\s+applies|"
            r"applicable\s+standard|"
            r"standards?\s+for|"
            r"standards?\s+should\s+i\s+use|"
            r"is\s+standard\s+for"
            r")\b)"
        )
        if re.search(explicit_standard_pattern, q_clean, re.IGNORECASE):
            category = self.get_category_for_intent("product_recommendation")
            log.info(f"Intent Router matched 'product_recommendation' (explicit standard question) for query: '{query[:40]}'")
            return {
                "intent": "product_recommendation",
                "category": category,
                "confidence": 0.95,
                "pattern_matched": explicit_standard_pattern,
            }

        # R1.2: Broad scheme / policy questions (CRS coverage, scheme explanation, list of covered products)
        # must route to scheme/general flow, NOT product_recommendation.
        scheme_overview_pattern = (
            r"\b("
            r"what\s+(?:products|items|goods)?\s*(?:fall|are\s+covered|come)\s+under|"
            r"which\s+products\s+(?:fall|are\s+covered|come)\s+under|"
            r"list\s+(?:of\s+)?products\s+under|"
            r"products\s+covered\s+under|"
            r"what\s+is\s+(?:the\s+)?(?:crs|compulsory\s+registration\s+scheme)|"
            r"explain\s+(?:the\s+)?(?:crs|compulsory\s+registration\s+scheme)|"
            r"tell\s+me\s+about\s+(?:the\s+)?(?:crs|compulsory\s+registration\s+scheme)|"
            r"about\s+(?:the\s+)?(?:crs|compulsory\s+registration\s+scheme)"
            r")\b"
        )
        if re.search(scheme_overview_pattern, q_clean, re.IGNORECASE):
            log.info(f"Intent Router matched 'general_rag' (scheme overview/coverage inquiry) for query: '{query[:40]}'")
            return {
                "intent": "general_rag",
                "category": None,
                "confidence": 0.90,
                "pattern_matched": scheme_overview_pattern,
            }

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
