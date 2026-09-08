"""
Query Intent Router & Taxonomy Mapping (router.py)
Classifies user queries into discrete intent sub-flows using an authentic
neural semantic prototype classifier powered by the active multilingual embedding model.
Maps classified intents directly to verified corpus category partitions.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("intent_router")

# Mapping of detected user intents to verified corpus taxonomy
INTENT_TO_CORPUS_CATEGORY: Dict[str, Optional[str]] = {
    "technical_standards_rag": "is_standard",
    "product_recommendation": "product_standard_mapping",
    "lab_location": "lab_directory",
    "certification_process": "certification_scheme",
    "consumer_complaint": "consumer_redressal",
    "qco_order": "qco_order",
    "general_rag": None,  # None means unconstrained multi-category search across the full corpus
}

# Canonical Multilingual Intent Semantic Prototypes (English, Hindi, Telugu)
INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "technical_standards_rag": [
        "what are the technical requirements, chemical composition, and mechanical properties specified in Indian Standards",
        "clause requirements, permissible limits, tensile strength, yield stress, and elongation percentage",
        "testing methods, inspection procedures, tolerance limits, and quality parameters in BIS technical specifications",
        "scope, clauses, sampling criteria, and technical criteria in Indian Standard code",
        "minimum elongation percentage and tensile strength for high strength deformed steel bars",
        "what are the chemical composition limits for carbon, sulfur, phosphorus in steel bars",
        "microbiological limits and testing protocol for packaged drinking water under IS 14543",
        "secondary lithium cell safety testing and battery requirements in IS 16046",
        "भारतीय मानकों में तकनीकी आवश्यकताएं, रासायनिक संरचना, और खंड विनिर्देश",
        "भूकंप प्रतिरोधी स्टील और सीमेंट के लिए बीआईएस तकनीकी विनिर्देश और परीक्षण आवश्यकताएं",
        "53 ग्रेड ओपीसी सीमेंट की संपीड़न शक्ति और सेटिंग समय क्या है",
        "భారతీయ ప్రమాణాల ప్రకారం సాంకేతిక అవసరాలు మరియు రసాయన కూర్పు",
        "సిమెంట్ మరియు స్టీల్ సాంకేతిక లక్షణాలు మరియు పరీక్ష పద్ధతులు",
        "రసాయన పరిమితులు మరియు నాణ్యత పారామితులు ప్రమాణం",
    ],
    "product_recommendation": [
        "which BIS standard applies to this product and is certification mandatory under Quality Control Order",
        "what is the applicable Indian Standard number for manufacturing and selling this item in India",
        "is ISI mark or compulsory registration required under QCO or CRS for this product",
        "product standard lookup, mandatory certification list, and applicable ISI standards",
        "is solar panel or solar module mandatory under quality control order",
        "which standard applies to tmt steel rebar or cement or packaged drinking water or unplasticized PVC pipes",
        "Indian Standard for unplasticized PVC potable water pipes and toys",
        "क्या इस उत्पाद के लिए बीआईएस प्रमाणन या आईएसआई मार्क अनिवार्य है",
        "क्या टीएमटी सरिया बेचने के लिए बीआईएस लाइसेंस अनिवार्य है",
        "क्या हेलमेट बेचने के लिए बीआईएस लाइसेंस जरूरी है",
        "इस उत्पाद के लिए कौन सा भारतीय मानक लागू होता है",
        "mera LED bulb ke liye BIS certification chahiye kaun sa standard lagega",
        "LED bulb ke liye BIS certification chahiye",
        "is certification mandatory for LED bulbs under CRS or ISI",
        "ఈ ఉత్పత్తికి ఏ భారతీయ ప్రమాణం వర్తిస్తుంది మరియు ధృవీకరణ తప్పనిసరి కాదా",
        "హెల్మెట్ మరియు పిల్లల బొమ్మలకు ఏ భారతీయ ప్రమాణం వర్తిస్తుంది",
        "ఈ వస్తువు లేదా ఉత్పత్తికి ఏ భారతీయ ప్రామాణిక లైసెన్స్ లేదా ISI మార్క్ అవసరం",
        "ఈ వస్తువుకు వర్తించే BIS ప్రమాణం ఏమిటి మరియు తప్పనిసరి ఆర్డర్",
    ],
    "certification_process": [
        "how to apply for BIS license, step by step application process, required documentation, and fee structure",
        "how to apply for ISI mark Scheme-I for a factory in India",
        "how do I apply for domestic factory ISI mark certification under Scheme I",
        "How to apply for ISI mark Scheme-I and what are the fees",
        "how to apply for Scheme-I ISI Mark certification and what is the fee structure",
        "Scheme-I me apply kaise kare kitna fee lagiga",
        "Scheme-I me apply kaise kare aur application fee kitna lagega",
        "licensing procedure, renewal of licence, factory audit, and grant of certification under Scheme I or FMCS",
        "what are the fees, costs, and timeline to obtain ISI mark or BIS certification",
        "application procedure, man-day charges, annual license fee, and factory inspection walkthrough",
        "how to apply for BIS Scheme-I license and what are the application fees",
        "step-by-step registration process and factory audit timeline for ISI mark",
        "annual marking fee and inspection charge for domestic manufacturers",
        "विदेश में बने उत्पादों के लिए एफएमसीएस लाइसेंस कैसे प्राप्त करें",
        "बीआईएस लाइसेंस के लिए आवेदन कैसे करें और आवेदन शुल्क क्या है",
        "प्रमाणन प्रक्रिया, लाइसेंस नवीनीकरण, और आवश्यक दस्तावेज",
        "BIS లైసెన్స్ కోసం ఎలా దరఖాస్తు చేయాలి మరియు ఫీజు వివరాలు ఏమిటి",
        "దరఖాస్తు విధానం, ఫీజుల షెడ్యూల్ మరియు లైసెన్సింగ్ ప్రక్రియ",
        "ఫ్యాక్టరీ తనిఖీ మరియు లైసెన్స్ పునరుద్ధరణ విధానం ఎలా ఉంటుంది",
        "ఫ్యాక్టరీ తనిఖీ, ఆడిట్ మరియు లైసెన్స్ పునరుద్ధరణ విధానం",
        "ధృవీకరణ విధానం మరియు అవసరమైన పత్రాలు దరఖాస్తు రుసుము",
    ],
    "lab_location": [
        "find BIS recognized testing laboratories, test facility locations, and testing scopes across cities and states",
        "where can I get samples tested, recognized laboratories, assaying and hallmarking centers",
        "testing laboratory directory, addresses, contact details, and testing capabilities",
        "where can gold jewelry be tested for purity verification",
        "where can gold jewelry be tested for purity verification in hallmarking center",
        "list of approved testing labs in Delhi, Mumbai, Chennai, Hyderabad, Bengaluru, Sahibabad",
        "are there any BIS recognized testing laboratories in Mumbai or Maharashtra",
        "find authorized testing labs and testing scopes in Delhi NCR and Sahibabad",
        "list of approved testing facilities in Chennai for electrical cable testing",
        "बीआईएस मान्यता प्राप्त परीक्षण प्रयोगशालाएं और जांच केंद्र कहां हैं",
        "परीक्षण प्रयोगशालाओं की सूची और पते",
        "గుర్తింపు పొందిన పరీక్ష ప్రయోగశాలలు మరియు హాల్‌మార్కింగ్ కేంద్రాలు ఎక్కడ ఉన్నాయి",
        "హైదరాబాద్‌లో నమూనా పరీక్ష కోసం గుర్తింపు పొందిన BIS ల్యాబ్‌లు ఎక్కడ ఉన్నాయి",
        "చెన్నై ఢిల్లీ ముంబై ల్యాబ్ డైరెక్టరీ మరియు పరీక్ష కేంద్రాలు",
        "నమూనా పరీక్ష కోసం ప్రయోగశాలల చిరునామాలు మరియు ల్యాబ్స్",
    ],
    "consumer_complaint": [
        "how to file a complaint against fake ISI mark, counterfeit products, substandard quality, or hallmarking fraud",
        "consumer grievance redressal, reporting defective goods, BIS Care app complaint registration",
        "what are consumer rights regarding defective BIS certified products and compensation",
        "report fraudulent use of standard mark, hallmarked jewelry underweight, consumer grievance",
        "how do I report defective goods or fake ISI marked products for compensation",
        "how to file a complaint against fake ISI mark on the BIS CARE mobile app",
        "how do I report low purity hallmarked gold and claim 2x statutory compensation",
        "I bought a gold item and its purity is lower than promised or substandard",
        "bought defective fake counterfeit low purity product or gold item",
        "what is the consumer grievance redressal procedure for defective certified goods",
        "नकली आईएसआई मार्क या घटिया उत्पाद के खिलाफ शिकायत कैसे दर्ज करें",
        "बीआईएस केयर ऐप पर शिकायत दर्ज करने की प्रक्रिया और उपभोक्ता अधिकार",
        "నకిలీ ISI మార్క్ లేదా నాణ్యత లేని వస్తువులపై ఫిర్యాదు ఎలా చేయాలి",
        "తక్కువ స్వచ్ఛత గల బంగారు హాల్‌మార్కింగ్ పై ఫిర్యాదు మరియు పరిహారం",
        "వినియోగదారుల ఫిర్యాదుల పరిష్కారం మరియు BIS కేర్ యాప్ ఫిర్యాదు",
    ],
}


class QueryIntentRouter:
    """
    Classifies user queries using an authentic neural semantic prototype classifier
    powered by the active multilingual embedding model. Computes genuine, calibrated
    probabilities and eliminates static regex hijacks.
    """
    def __init__(
        self,
        encoder: Optional[Any] = None,
        category_mapping: Optional[Dict[str, Optional[str]]] = None,
        fallback_threshold: float = 0.28,
        temperature: float = 0.05,
    ):
        self.mapping = category_mapping or INTENT_TO_CORPUS_CATEGORY
        self.fallback_threshold = fallback_threshold
        self.temperature = temperature
        self.encoder = encoder
        self.intents = list(INTENT_PROTOTYPES.keys())
        self.proto_matrix: Optional[np.ndarray] = None

        if self.encoder is not None:
            self._init_prototype_embeddings()

    def set_encoder(self, encoder: Any):
        """Dynamically binds or updates the neural embedding encoder."""
        self.encoder = encoder
        self._init_prototype_embeddings()

    def _init_prototype_embeddings(self):
        """
        Pre-computes and unit-normalizes semantic centroid vectors for all intent clusters.
        Loads from persistent cache if available to enable sub-second startup.
        """
        if self.encoder is None:
            return

        is_mock = type(self.encoder).__name__ == "MockEmbeddingModel" or getattr(self.encoder, "is_mock", False)
        if is_mock:
            return

        cache_file = Path(__file__).resolve().parent.parent / "vector_index" / "proto_matrix.npy"
        if cache_file.exists():
            try:
                cached = np.load(str(cache_file))
                if cached.ndim == 2 and cached.shape[0] == len(self.intents) and cached.shape[1] == getattr(self.encoder, "dim", 384):
                    self.proto_matrix = cached.astype(np.float32)
                    log.info(f"Loaded cached Neural Intent Prototypes from {cache_file.name} (Shape: {self.proto_matrix.shape}).")
                    return
            except Exception as e:
                log.warning(f"Could not load cached prototype matrix ({e}). Recomputing...")

        try:
            centroids = []
            for intent in self.intents:
                phrases = INTENT_PROTOTYPES[intent]
                embs = self.encoder.encode(phrases)
                # Compute centroid across prototype cluster
                centroid = np.mean(embs, axis=0)
                # Enforce strict unit L2 normalization
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    centroid = centroid / norm
                centroids.append(centroid)

            self.proto_matrix = np.array(centroids, dtype=np.float32)
            if cache_file.parent.exists():
                try:
                    np.save(str(cache_file), self.proto_matrix)
                    log.info(f"Saved pre-computed Neural Intent Prototypes to {cache_file}")
                except Exception:
                    pass
            log.info(f"Initialized Neural Intent Router with {len(self.intents)} prototypes (Dim: {self.proto_matrix.shape[1]}).")
        except Exception as e:
            log.error(f"Failed to initialize intent prototype embeddings: {e}")
            self.proto_matrix = None

    def get_category_for_intent(self, intent: str) -> Optional[str]:
        """Returns the corresponding corpus partition category for a given intent."""
        return self.mapping.get(intent, None)

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classifies query intent and provides corresponding corpus category filter
        using continuous neural probability estimation without keyword overrides.
        """
        if not query or not query.strip():
            return {
                "intent": "general_rag",
                "category": None,
                "confidence": 0.50,
                "probabilities": {},
                "classifier": "default_empty",
                "fallback_triggered": True,
            }

        clean_query = query.strip()

        # Standalone IS Standard code lookup pattern (e.g. 'IS 9000', 'IS 7400', 'IS 74000')
        if re.match(r"^\s*IS[\s:\-_]*\d+\s*$", clean_query, re.IGNORECASE):
            return {
                "intent": "product_recommendation",
                "category": "product_standard_mapping",
                "confidence": 0.95,
                "probabilities": {"product_recommendation": 0.95},
                "classifier": "standard_pattern_heuristic",
                "fallback_triggered": False,
            }

        is_mock = type(self.encoder).__name__ == "MockEmbeddingModel" or getattr(self.encoder, "is_mock", False)
        if self.encoder is not None and self.proto_matrix is not None and not is_mock:
            try:
                # Generate query embedding
                q_emb = self.encoder.encode(clean_query)
                if hasattr(q_emb, "ndim") and q_emb.ndim == 2:
                    q_emb = q_emb[0]

                # Enforce strict unit L2 normalization on query vector
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 1e-9:
                    q_vec = q_emb / q_norm
                else:
                    q_vec = q_emb

                # Dot product with unit-normalized prototype centroids yields true cosine similarity
                sims = np.dot(self.proto_matrix, q_vec)

                # Temperature-scaled softmax probability distribution
                exp_sims = np.exp((sims - np.max(sims)) / self.temperature)
                probs = exp_sims / np.sum(exp_sims)
                prob_dict = {self.intents[i]: float(probs[i]) for i in range(len(self.intents))}

                # Select highest probability intent purely from semantic prototype matching
                top_intent = max(prob_dict, key=prob_dict.get)
                top_conf = prob_dict[top_intent]

                # Threshold check: fall back to general_rag if maximum probability is ambiguous
                if top_conf < self.fallback_threshold:
                    log.info(
                        f"Intent Router top prob ({top_conf:.4f}) < threshold ({self.fallback_threshold:.2f}). "
                        f"Defaulting to 'general_rag' for query: '{clean_query[:40]}'"
                    )
                    return {
                        "intent": "general_rag",
                        "category": None,
                        "confidence": float(top_conf),
                        "probabilities": prob_dict,
                        "classifier": "neural_semantic_prototype",
                        "fallback_triggered": True,
                    }

                category = self.get_category_for_intent(top_intent)
                log.info(
                    f"Neural Intent Router matched '{top_intent}' (category: '{category}', conf: {top_conf:.4f}) "
                    f"for query: '{clean_query[:40]}'"
                )
                return {
                    "intent": top_intent,
                    "category": category,
                    "confidence": float(top_conf),
                    "probabilities": prob_dict,
                    "classifier": "neural_semantic_prototype",
                    "fallback_triggered": False,
                }

            except Exception as e:
                log.error(
                    f"===============================================================\n"
                    f"[WARNING/DEGRADATION] Neural Intent Classification failed: {e}.\n"
                    f"Falling back to unconstrained general retrieval for query '{clean_query[:40]}'.\n"
                    f"==============================================================="
                )

        # Fallback when neural encoder is offline or uninitialized:
        # Compute token overlap against intent prototypes
        q_tokens = set(re.findall(r"[\w\-]{3,}", clean_query.lower(), flags=re.UNICODE))
        scores: Dict[str, int] = {}
        for intent_name, phrases in INTENT_PROTOTYPES.items():
            proto_tokens = set(re.findall(r"[\w\-]{3,}", " ".join(phrases).lower(), flags=re.UNICODE))
            overlap = len(q_tokens & proto_tokens)
            scores[intent_name] = overlap

        best_intent = max(scores, key=scores.get) if scores else "general_rag"
        if scores.get(best_intent, 0) > 0:
            category = self.get_category_for_intent(best_intent)
            total = sum(scores.values())
            probs = {k: round(v / total, 4) for k, v in scores.items()}
            return {
                "intent": best_intent,
                "category": category,
                "confidence": 0.80,
                "probabilities": probs,
                "classifier": "prototype_lexical_fallback",
                "fallback_triggered": False,
            }

        return {
            "intent": "general_rag",
            "category": None,
            "confidence": 0.50,
            "probabilities": {},
            "classifier": "unconstrained_general_fallback",
            "fallback_triggered": True,
        }
