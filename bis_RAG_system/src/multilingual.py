"""
Phase 5, Step 17: Language Detection, Hinglish Normalizer & Multilingual Strategy (multilingual.py)
Detects input language (English, Hindi, Hinglish, Tamil, Telugu, etc.) and normalizes code-mixed Hinglish.
"""

import re
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multilingual")

# Common Hinglish (Romanized Hindi) keywords
HINGLISH_KEYWORDS = [
    "mera", "meri", "mere", "chahiye", "kaise", "kab", "kitna", "kitni", "kya", "hoga", "hogi", "hai", "hain",
    "karna", "kare", "karen", "karein", "parega", "padega", "paise", "paisa", "rupaye", "fee", "roopaye", "shubh",
    "batao", "bataiye", "batayein", "jaankari", "shuru", "dono", "sabse", "badhiya", "ke", "liye", "kaunsa",
    "kaun", "kaunsi", "lagta", "lagiga", "lagoge", "mujhe", "aap", "hum", "nahi", "nahin", "hota", "hoti", "standard"
]

# Devanagari Unicode Range Regex
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")


class MultilingualHandler:
    def detect_language(self, text: str) -> Dict[str, Any]:
        raw_text = text.strip()
        
        # 1. Check Native Scripts
        if DEVANAGARI_RE.search(raw_text):
            return {"lang_code": "hi", "lang_name": "Hindi", "is_code_mixed": False}
        if TELUGU_RE.search(raw_text):
            return {"lang_code": "te", "lang_name": "Telugu", "is_code_mixed": False}
        if TAMIL_RE.search(raw_text):
            return {"lang_code": "ta", "lang_name": "Tamil", "is_code_mixed": False}
        if BENGALI_RE.search(raw_text):
            return {"lang_code": "bn", "lang_name": "Bengali", "is_code_mixed": False}

        # 2. Check Romanized Hinglish vs Pure English
        words = re.findall(r"\w+", raw_text.lower())
        hinglish_matches = sum(1 for w in words if w in HINGLISH_KEYWORDS)
        phrase_matches = re.search(r"\b(ke liye|kaunsa|kaun sa|kya hai|kaise kare|kaise apply|kitna fee|batao|bataiye|batayein|chahiye)\b", raw_text, re.IGNORECASE)

        if hinglish_matches >= 2 or phrase_matches:
            log.info(f"Detected Code-Mixed Hinglish input for: '{text}'")
            return {"lang_code": "hinglish", "lang_name": "Hinglish", "is_code_mixed": True}

        return {"lang_code": "en", "lang_name": "English", "is_code_mixed": False}

    def normalize_hinglish_to_english(self, text: str) -> str:
        """Translates Hinglish key intents to clean English keywords for optimal RAG retrieval."""
        normalized = text.lower()
        replacements = [
            (r"\bmera\b|\bmeri\b", "my"),
            (r"\bled bulb ke liye\b", "for LED bulb"),
            (r"\btmt bar ke liye\b", "for TMT bar steel reinforcement"),
            (r"\bkaunsa bis standard applicable hai\b|\bkaun sa bis standard\b", "what BIS standard applies"),
            (r"\bis certification chahiye\b", "need BIS certification"),
            (r"\bkaise apply kare\b|\bkaise apply karen\b", "how to apply"),
            (r"\bkitna fee lagiga\b|\bkitna fee lagega\b", "what is the application fee"),
            (r"\bgold ring fake hai\b", "fake gold ring complaint"),
            (r"\bchahiye\b", "require"),
            (r"\bkitna\b", "how much"),
            (r"\bkaise\b", "how"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)

        log.info(f"Normalized Hinglish '{text}' -> '{normalized}'")
        return normalized

    def normalize_native_to_english_keywords(self, text: str) -> str:
        """
        Translates common Hindi/Telugu/Tamil domain keywords to English
        so that BM25 / retrieval index can find relevant BIS documents.
        """
        t = text
        # Hindi keywords
        t = re.sub(r"सीमेंट", "cement Ordinary Portland Cement", t)
        t = re.sub(r"स्टील|इस्पात", "steel reinforcement", t)
        t = re.sub(r"मानक", "standard", t)
        t = re.sub(r"लागू", "applies", t)
        t = re.sub(r"प्रमाणन|प्रमाणपत्र", "certification", t)
        t = re.sub(r"शुल्क", "fee", t)
        t = re.sub(r"प्रयोगशाला|लैब", "laboratory", t)
        t = re.sub(r"सोना|स्वर्ण", "gold hallmarking", t)
        t = re.sub(r"हॉलमार्क", "hallmark", t)
        t = re.sub(r"शिकायत", "complaint", t)
        
        # Telugu keywords
        t = re.sub(r"సిమెంట్(?:కు)?", "cement Ordinary Portland Cement", t)
        t = re.sub(r"ఉక్కు", "steel reinforcement", t)
        t = re.sub(r"ప్రమాణం|ప్రమాణాలు", "standard", t)
        t = re.sub(r"వర్తిస్తుంది", "applies", t)
        t = re.sub(r"ధృవీకరణ", "certification", t)
        t = re.sub(r"రుసుము", "fee", t)
        t = re.sub(r"ప్రయోగశాల", "laboratory", t)
        t = re.sub(r"బంగారం", "gold hallmarking", t)
        t = re.sub(r"ఫిర్యాదు", "complaint", t)

        return t


if __name__ == "__main__":
    handler = MultilingualHandler()
    samples = [
        "is certification mandatory for LED bulbs",
        "क्या एलईडी बल्ब के लिए बीआईएस प्रमाणन अनिवार्य है?",
        "mera LED bulb ke liye BIS certification chahiye",
        "எல்இடி பல்புகளுக்கு பிஐஎஸ் சான்றிதழ் கட்டாயமா?",
    ]
    for s in samples:
        det = handler.detect_language(s)
        norm = handler.normalize_hinglish_to_english(s) if det["lang_code"] == "hinglish" else s
        print(f"Text: '{s}' -> Lang: {det['lang_name']} | Normalized: '{norm}'")
