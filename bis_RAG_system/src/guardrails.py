"""
Confidence Estimator & Uncertainty Refusal Gate (guardrails.py)
Calculates calibrated retrieval confidence, verifies multilingual evidence sufficiency,
and enforces refusal/redirection for out-of-corpus or ungrounded queries.
"""

import logging
import math
import os
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

# Multilingual Stop Words (English, Hindi, Telugu)
STOP_WORDS: Set[str] = {
    # English
    "what", "is", "the", "process", "for", "under", "in", "to", "of", "and", "a", "an",
    "how", "do", "i", "can", "you", "tell", "me", "about", "which", "are", "by", "from",
    "with", "on", "as", "per", "give", "details", "information", "requirements", "requirement",
    "regulation", "regulations", "standard", "standards", "rule", "rules", "scheme", "schemes",
    # Hindi
    "क्या", "है", "के", "लिए", "और", "में", "से", "का", "की", "को", "पर", "द्वारा", "बताएं", "दीजिए",
    # Telugu
    "ఏమిటి", "మరియు", "కొరకు", "యొక్క", "లో", "నుండి", "ద్వారా", "వివరాలు", "తెలపండి"
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
        """
        Extracts meaningful query keywords using Unicode-aware tokenization
        supporting all Indian languages (Devanagari, Telugu, Tamil, etc.).
        Preserves unified standard tokens (e.g. 'is_1786', 'is_7400') and bare numbers.
        """
        norm_text = re.sub(r"\bis[\s:\-_]*(\d{2,6})\b", r"is_\1", text, flags=re.IGNORECASE)
        # Unicode-aware word tokenization
        tokens = re.findall(r"[\w\-]{2,}", norm_text.lower(), flags=re.UNICODE)
        res = set()
        for t in tokens:
            if t not in STOP_WORDS or t.isdigit() or t.startswith("is_"):
                res.add(t)
                if t.startswith("is_") and t[3:].isdigit():
                    res.add(t[3:])
                elif t.isdigit() and len(t) >= 3:
                    res.add(f"is_{t}")
                if t.endswith("s") and len(t) > 3:
                    res.add(t[:-1])
        return res

    def calculate_evidence_sufficiency(self, query: str, retrieved_results: List[Dict[str, Any]]) -> float:
        """
        Calculates lexical term overlap between user query and retrieved context
        with sub-linear length damping. Returns a score in [0.0, 1.0].
        """
        if not query or not retrieved_results:
            return 0.0

        q_keywords = self.extract_keywords(query)
        if not q_keywords:
            return 0.50  # Neutral fallback if query only had generic terms

        # Combine text from top candidate documents
        context_text = ""
        for item in retrieved_results[:3]:
            doc = item.get("doc", item)
            text = str(doc.get("text", ""))
            title = str(doc.get("clause_title") or doc.get("product") or doc.get("title") or "")
            is_num = str(doc.get("is_number") or doc.get("standard") or "")
            context_text += f" {is_num} {title} {text}".lower()

        # Tokenize combined context text with extract_keywords
        context_keywords = self.extract_keywords(context_text)
        matched_tokens = 0
        for kw in q_keywords:
            if kw in context_keywords:
                matched_tokens += 1

        overlap_ratio = matched_tokens / len(q_keywords)
        # Sub-linear length damping to avoid penalizing longer conversational queries
        damped_sufficiency = min(1.0, overlap_ratio * (1.0 + 0.05 * min(4, len(q_keywords))))
        return float(damped_sufficiency)

    def calculate_confidence(self, query: str, retrieved_results: List[Dict[str, Any]]) -> float:
        """
        Calculates a calibrated confidence score in [0.0, 1.0] from neural cross-encoder,
        dense cosine similarity, RRF score, and evidence sufficiency.
        Eliminates artificial 0.30 clamping, 0.96 ceiling, and discontinuous step-function penalties.
        """
        if not retrieved_results:
            return 0.0

        top_result = retrieved_results[0]
        evidence_sufficiency = self.calculate_evidence_sufficiency(query, retrieved_results)

        # 1. Base Score from Retrieval Signals
        rerank_score = top_result.get("rerank_score")
        cross_encoder_score = top_result.get("cross_encoder_score")
        rerank_method = top_result.get("rerank_method", "heuristic")
        dense_score = top_result.get("dense_score")
        rrf_score = top_result.get("rrf_score")
        raw_score = top_result.get("score")

        if rrf_score is None and raw_score is not None and 0.0 < raw_score < 0.10:
            rrf_score = raw_score

        # Detect non-Latin Indic scripts (Devanagari, Telugu, Tamil, etc.)
        is_indic_query = any(ord(c) > 127 and not c.isnumeric() for c in query)

        # Cross-encoder base confidence (Continuous without artificial ceiling)
        ce_base_conf = 0.0
        if rerank_method == "neural" and cross_encoder_score is not None:
            s = float(cross_encoder_score)
            ce_base_conf = max(0.01, min(0.9999, s))

        # Dense embedding base confidence (Continuous sigmoid mapping)
        dense_base_conf = 0.0
        if dense_score is not None and isinstance(dense_score, (int, float)):
            ds = float(dense_score)
            dense_base_conf = max(0.01, min(0.9999, float(1.0 / (1.0 + math.exp(-8.0 * (ds - 0.28))))))

        # RRF fused base confidence: only trust RRF if dense semantic match or BM25 match exists
        rrf_base_conf = 0.0
        if rrf_score is not None and rrf_score > 0.0:
            if dense_score is not None and float(dense_score) < 0.22 and evidence_sufficiency < 0.10:
                rrf_base_conf = max(0.01, float(rrf_score) * 4.0)
            else:
                rrf_base_conf = min(0.95, float(rrf_score) * 28.0)

        # Heuristic / fallback base confidence
        heur_base_conf = 0.0
        if rerank_score is not None and rerank_score > 0.0:
            rs = float(rerank_score)
            heur_base_conf = max(0.01, min(0.99, rs))
        elif raw_score is not None and raw_score > 0.0:
            rs = float(raw_score)
            heur_base_conf = max(0.01, min(0.99, rs))

        # Principled Multi-Signal Arbitration:
        if is_indic_query:
            base_conf = max(dense_base_conf, rrf_base_conf, ce_base_conf)
            if dense_base_conf >= 0.30 or rrf_base_conf >= 0.30:
                evidence_sufficiency = max(evidence_sufficiency, base_conf)
        else:
            if rerank_method == "neural" and cross_encoder_score is not None:
                if dense_base_conf > 0.0:
                    base_conf = 0.85 * ce_base_conf + 0.15 * dense_base_conf
                else:
                    base_conf = ce_base_conf
            else:
                base_conf = max(dense_base_conf, rrf_base_conf, heur_base_conf)

            # If strong dense semantic match exists, boost evidence sufficiency for natural language paraphrases
            if dense_base_conf >= 0.45:
                evidence_sufficiency = max(evidence_sufficiency, base_conf * 0.70)
            elif base_conf < 0.30:
                # Discard raw word count evidence if semantic relevance is negligible
                evidence_sufficiency = evidence_sufficiency * (base_conf / 0.30)

        # 2. Continuous Dual-Signal Convex Combination
        alpha = 0.50 + 0.35 / (1.0 + math.exp(-10.0 * (base_conf - 0.40)))
        raw_combined = (base_conf * alpha) + (evidence_sufficiency * (1.0 - alpha))

        # 3. Continuous Out-of-Domain Smooth Attenuation
        joint_signal = (base_conf * 0.65) + (evidence_sufficiency * 0.35)
        attenuation = 1.0 / (1.0 + math.exp(-25.0 * (joint_signal - 0.25)))
        final_conf = raw_combined * attenuation

        calibrated = max(0.0, min(1.0, float(final_conf)))
        log.info(
            f"Confidence Calculation -> Query: '{query[:40]}' | Base: {base_conf:.4f} | "
            f"Evidence Sufficiency: {evidence_sufficiency:.4f} | Calibrated: {calibrated:.4f}"
        )
        return calibrated

    @staticmethod
    def detect_prompt_injection(query: str) -> bool:
        """
        Detects adversarial system overrides, roleplay escapes, and prompt injection patterns
        across English, Romanized Indic (Hinglish/Tenglish/Tanglish), and native Indic scripts (Hindi, Telugu, Tamil).
        """
        if not query or not query.strip():
            return False

        patterns = [
            # 1. English Jailbreak & System Override Patterns
            r"ignore\s+(?:all\s+)?(?:previous|prior|system|grounding)\s+instructions?",
            r"you\s+are\s+now\s+(?:dan|jailbreak|unrestricted|an\s+ai\s+without\s+rules|in\s+developer\s+mode)",
            r"disregard\s+(?:all\s+)?(?:rules|guidelines|context|grounding|limitations|safety)",
            r"(?:system\s+prompt\s+override|developer\s+mode\s+enabled|unrestricted\s+mode)",
            r"reveal\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions|system\s+message)",
            r"pretend\s+you\s+are\s+not\s+(?:bis|an\s+ai|bound\s+by\s+rules)",
            r"bypass\s+(?:all\s+)?(?:filters|rules|safeguards|grounding|restrictions)",
            r"(?:dan\s+mode|jailbreak\s+active|jailbreak\s+prompt)",
            r"do\s+anything\s+now",

            # 2. Romanized Indic (Hinglish / Tenglish / Tanglish) Jailbreak Patterns
            r"(?:pichle|pichhla|purane|saare|sab)\s+(?:nirdesh|instructions?|rules?)\s+(?:bhul|bhol|chhod|ignore)\s+(?:jao|karo|do|kijiye)",
            r"(?:nirdesh|rules?)\s+(?:ko\s+)?(?:ignore|override|bypass)\s+karo",
            r"system\s+prompt\s+(?:batao|dikhao|reveal\s+karo|share\s+karo|kya\s+hai)",
            r"tum\s+ab\s+(?:dan|unrestricted|azad|free)\s+ho",
            r"(?:rules?|niyam)\s+(?:tod\s+do|hatao|khatam\s+karo|ignore\s+karo)",
            r"apne\s+(?:rules|nirdesh)\s+mat\s+mano",
            r"(?:gata|mundhati)\s+sochanalanu\s+(?:vismarinchandi|marichipo)",
            r"system\s+prompt\s+(?:cheppandi|cheppu|chupinchu)",

            # 3. Native Indic Scripts (Devanagari Hindi, Telugu, Tamil)
            # Hindi Devanagari
            r"पिछल[ीे]\s+(?:सभी\s+|सारे\s+)?(?:निर्देशों?|हिदायतों?|नियमों?)\s+को\s+(?:भूल\s+जाओ|अनदेखा\s+करो|नज़रअंदाज़\s+करो)",
            r"सिस्टम\s+(?:प्रॉम्प्ट|निर्देश|नियम)\s+(?:दिखाओ|बताओ|प्रकट\s+करो)",
            r"(?:सभी\s+)?(?:प्रतिबंध|नियम|सीमाएं)\s+(?:हटाओ|तोड़\s+दो|समाप्त\s+करो)",
            r"तुम\s+अब\s+(?:डैन|स्वतंत्र|अप्रतिबंधित)\s+हो",
            # Telugu
            r"గత\s+(?:అన్ని\s+)?సూచనలను\s+(?:విస్మరించండి|మరచిపోండి|వదిలేయండి)",
            r"సిస్టమ్\s+ప్రాంప్ట్\s+(?:చూపించు|చెప్పు|బహిర్గతం\s+చేయి)",
            r"(?:నిబంధనలను|పరిమితులను)\s+(?:రద్దు\s+చేయండి|విస్మరించండి)",
            # Tamil
            r"முந்தைய\s+(?:அனைத்து\s+)?வழிமுறைகளைப்\s+புறக்கணிக்கவும்",
            r"கணினி\s+கட்டளைகளைக்\s+காட்டு",
        ]

        q_clean = query.strip()
        q_low = q_clean.lower()
        for p in patterns:
            if re.search(p, q_low, flags=re.IGNORECASE | re.UNICODE):
                log.warning(f"Adversarial Prompt Injection Detected matching pattern: '{p}'")
                return True
        return False


    def evaluate_and_gate(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        category: str = "general",
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Evaluates retrieval sufficiency and security. If confidence < threshold or injection detected, returns refusal.
        """
        # 0. Security Guardrail: Check for prompt injection
        if self.detect_prompt_injection(query):
            log.warning(f"Security Guardrail FIRED: Prompt injection detected in query '{query[:50]}'")
            security_refusal = (
                "Security Guardrail Notice: The submitted query contains disallowed system override instructions. "
                "The BIS AI Assistant only provides grounded answers regarding official Indian Standards and compliance."
            )
            return False, 0.0, security_refusal

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
        "rerank_score": 0.032,
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
        "rerank_score": 0.016,
        "doc": {
            "is_number": "IS 1070",
            "text": "Water for analytical laboratory use specification.",
            "clause_title": "Water Protocol"
        }
    }]
    passed2, conf2, msg2 = gate.evaluate_and_gate("what is the property tax rate in Tokyo Japan", out_results)
    print(f"Out-of-Corpus Query -> Passed: {passed2}, Conf: {conf2:.4f}")
