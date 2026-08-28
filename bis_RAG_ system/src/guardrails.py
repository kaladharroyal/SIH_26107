"""
Phase 3, Part 3.2: Confidence Estimator & Uncertainty Refusal Gate (guardrails.py)
Prevents hallucinations by calculating retrieval confidence scores and refusing out-of-corpus queries.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("guardrails")

CONFIDENCE_THRESHOLD = 0.45

OFFICIAL_FALLBACK_PORTALS = {
    "product_certification": "https://www.bis.gov.in/product-certification/product-certification-overview/?lang=en",
    "hallmarking": "https://www.bis.gov.in/hallmarking-overview/?lang=en",
    "compulsory_registration": "https://www.crsbis.in/BIS/",
    "consumer_complaint": "https://www.bis.gov.in/consumer-overview/online-complaint-registration/?lang=en",
    "general": "https://www.bis.gov.in/?lang=en",
}


class GuardrailGate:
    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def calculate_confidence(self, retrieved_results: List[Dict[str, Any]]) -> float:
        if not retrieved_results:
            return 0.0

        top_result = retrieved_results[0]
        # Check rerank_score or rrf_score
        raw_score = top_result.get("rerank_score") or top_result.get("rrf_score") or top_result.get("score", 0.0)

        # Normalize score between 0.0 and 1.0
        if raw_score > 1.0:
            confidence = min(1.0, raw_score / 10.0 + 0.5)
        else:
            confidence = max(0.0, min(1.0, float(raw_score)))

        log.info(f"Calculated Retrieval Confidence Score: {confidence:.4f} (Raw Score: {raw_score})")
        return confidence

    def evaluate_and_gate(
        self, query: str, retrieved_results: List[Dict[str, Any]], category: str = "general"
    ) -> Tuple[bool, float, Optional[str]]:
        confidence = self.calculate_confidence(retrieved_results)

        if confidence < self.threshold:
            log.warning(f"Refusal Gate FIRED for query: '{query}' (Confidence {confidence:.4f} < Threshold {self.threshold})")
            portal_url = OFFICIAL_FALLBACK_PORTALS.get(category, OFFICIAL_FALLBACK_PORTALS["general"])
            
            refusal_message = (
                f"I cannot find a specific BIS clause, regulation, or FAQ entry directly covering '{query}' "
                f"in the official database corpus.\n\n"
                f"To ensure complete compliance accuracy, please verify directly with the official BIS portal:\n"
                f"🔗 [Official BIS Portal]({portal_url}) or download the BIS CARE App."
            )
            return False, confidence, refusal_message

        log.info(f"Refusal Gate PASSED for query: '{query}' (Confidence {confidence:.4f} >= Threshold {self.threshold})")
        return True, confidence, None


if __name__ == "__main__":
    gate = GuardrailGate(threshold=0.45)
    
    # Test High Confidence
    passed, conf, msg = gate.evaluate_and_gate("IS 1786 steel requirements", [{"rerank_score": 0.85}])
    print(f"High Conf Test -> Passed: {passed}, Conf: {conf}")

    # Test Low Confidence (Refusal)
    passed, conf, msg = gate.evaluate_and_gate("what is the tax law in France", [{"rerank_score": 0.12}])
    print(f"Low Conf Test -> Passed: {passed}, Conf: {conf}\nRefusal Msg:\n{msg}")
