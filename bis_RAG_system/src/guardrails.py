"""
Phase 3, Part 3.2: Confidence Estimator & Uncertainty Refusal Gate (guardrails.py)
Calculates calibrated retrieval confidence, verifies evidence sufficiency,
and enforces refusal/redirection for out-of-corpus or ungrounded queries.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("guardrails")

CONFIDENCE_THRESHOLD = 0.45

OFFICIAL_FALLBACK_PORTALS = {
    "product_certification": "https://www.bis.gov.in/product-certification/product-certification-overview/?lang=en",
    "hallmarking": "https://www.bis.gov.in/hallmarking-overview/?lang=en",
    "compulsory_registration": "https://www.crsbis.in/BIS/",
    "consumer_complaint": "https://www.bis.gov.in/consumer-overview/online-complaint-registration/?lang=en",
    "lab_directory": "https://www.bis.gov.in/laboratory-overview/?lang=en",
    "product_standard_mapping": "https://standardsbis.bsbedge.com/",
    "general": "https://www.bis.gov.in/?lang=en",
}

STOP_WORDS: Set[str] = {
    "what", "is", "the", "process", "for", "under", "in", "to", "of", "and", "a", "an",
    "how", "do", "i", "can", "you", "tell", "me", "about", "which", "are", "by", "from",
    "with", "on", "as", "per", "give", "details", "information", "requirements", "requirement",
    "regulation", "regulations", "standard", "standards", "rule", "rules", "scheme", "schemes"
}


class GuardrailGate:
    """
    Evaluates retrieval confidence, evidence sufficiency, and decides
    whether to pass query to generator or trigger safe refusal.
    """

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        """Extracts meaningful query keywords excluding common stop words."""
        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        return {t for t in tokens if t not in STOP_WORDS}

    def calculate_evidence_sufficiency(self, query: str, retrieved_results: List[Dict[str, Any]]) -> float:
        """
        Calculates lexical term overlap between user query and retrieved context.
        Returns a score in [0.0, 1.0].
        """
        if not query or not retrieved_results:
            return 0.0

        q_keywords = self.extract_keywords(query)
        if not q_keywords:
            return 0.5  # Neutral fallback if query only had generic terms

        # Combine text from top candidate documents
        context_text = ""
        for item in retrieved_results[:3]:
            doc = item.get("doc", item)
            text = doc.get("text", "")
            title = doc.get("clause_title") or doc.get("product") or ""
            is_num = doc.get("is_number", "")
            context_text += f" {is_num} {title} {text}".lower()

        matched_tokens = 0
        for kw in q_keywords:
            # Check for exact word boundary or substring for IS numbers
            if re.search(r"\b" + re.escape(kw) + r"\b", context_text):
                matched_tokens += 1

        overlap_ratio = matched_tokens / len(q_keywords)
        return float(overlap_ratio)

    def calculate_confidence(self, query: str, retrieved_results: List[Dict[str, Any]]) -> float:
        """
        Calculates a calibrated confidence score in [0.0, 1.0] from Phase 2 retrieval
        signals (RRF score, rerank score, BM25/dense score) and evidence sufficiency.
        """
        if not retrieved_results:
            return 0.0

        top_result = retrieved_results[0]
        evidence_sufficiency = self.calculate_evidence_sufficiency(query, retrieved_results)

        # 1. Base Score from Retrieval Rerank / RRF
        rerank_score = top_result.get("rerank_score")
        rrf_score = top_result.get("rrf_score")
        raw_score = top_result.get("score")

        if rrf_score is None and raw_score is not None and 0.0 < raw_score < 0.10:
            rrf_score = raw_score

        base_conf = 0.0

        if rerank_score is not None and rerank_score > 0.0:
            # Rerank score with IS boost can be > 0.5
            if rerank_score >= 0.5:
                base_conf = min(1.0, 0.70 + (rerank_score - 0.5) * 0.4)
            else:
                base_conf = min(0.70, max(0.30, rerank_score * 1.4))
        elif rrf_score is not None:
            # Calibrated Phase 2 RRF mapping:
            # Rank 1 in both dense & sparse: RRF ~= 0.0328 -> Conf ~ 0.85
            # Rank 1 in one method: RRF ~= 0.0164 -> Conf ~ 0.70
            # Rank 5 in one method: RRF ~= 0.0151 -> Conf ~ 0.65
            # Low rank: RRF < 0.008 -> Conf < 0.40
            if rrf_score >= 0.030:
                base_conf = min(0.95, 0.75 + (rrf_score - 0.030) * 10.0)
            elif rrf_score >= 0.015:
                base_conf = min(0.75, 0.60 + (rrf_score - 0.015) * 10.0)
            elif rrf_score >= 0.008:
                base_conf = min(0.60, 0.40 + (rrf_score - 0.008) * 25.0)
            else:
                base_conf = max(0.05, rrf_score * 40.0)
        elif raw_score is not None:
            if raw_score > 1.0:
                base_conf = min(0.95, raw_score / 15.0 + 0.4)
            else:
                base_conf = max(0.0, min(1.0, float(raw_score)))
        else:
            base_conf = 0.50

        # 2. Evidence Sufficiency Gating
        # If very low keyword overlap, severely penalize confidence to guarantee refusal
        if evidence_sufficiency < 0.25:
            final_conf = base_conf * 0.15
        elif evidence_sufficiency < 0.50:
            final_conf = base_conf * 0.55
        else:
            final_conf = (base_conf * 0.60) + (evidence_sufficiency * 0.40)

        calibrated = max(0.0, min(1.0, float(final_conf)))
        log.info(
            f"Confidence Calculation -> Query: '{query[:40]}' | Base: {base_conf:.4f} | "
            f"Evidence Sufficiency: {evidence_sufficiency:.4f} | Calibrated: {calibrated:.4f}"
        )
        return calibrated

    def evaluate_and_gate(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        category: str = "general",
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Evaluates retrieval sufficiency. If confidence < threshold, returns refusal message.
        """
        confidence = self.calculate_confidence(query, retrieved_results)

        if confidence < self.threshold or not retrieved_results:
            log.warning(
                f"Refusal Gate FIRED for query: '{query}' "
                f"(Confidence {confidence:.4f} < Threshold {self.threshold})"
            )
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
    
    # 1. Valid Query Test
    dummy_results = [{
        "rrf_score": 0.032,
        "doc": {
            "is_number": "IS 1786",
            "text": "High strength deformed steel bars for concrete reinforcement IS 1786 requirements.",
            "clause_title": "Steel Specification"
        }
    }]
    passed, conf, msg = gate.evaluate_and_gate("IS 1786 steel reinforcement requirements", dummy_results)
    print(f"Valid Query -> Passed: {passed}, Conf: {conf:.4f}")

    # 2. Out-of-Corpus Query Test (Zero Keyword Overlap)
    out_results = [{
        "rrf_score": 0.016,
        "doc": {
            "is_number": "IS 1070",
            "text": "Water for analytical laboratory use specification.",
            "clause_title": "Water Protocol"
        }
    }]
    passed2, conf2, msg2 = gate.evaluate_and_gate("what is the property tax rate in Tokyo Japan", out_results)
    print(f"Out-of-Corpus Query -> Passed: {passed2}, Conf: {conf2:.4f}")
