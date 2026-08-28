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
    "mera", "meri", "chahiye", "kaise", "kab", "kitna", "kya", "hoga", "hai",
    "karna", "parega", "paise", "paisa", "rupaye", "fee", "roopaye", "shubh",
    "batao", "bataiye", "jaankari", "shuru", "dono", "sabse", "badhiya"
]

# Devanagari Unicode Range Regex
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")


class MultilingualHandler:
    def detect_language(self, text: str) -> Dict[str, Any]:
        raw_text = text.strip()
        
        # Check Native Scripts
        if DEVANAGARI_RE.search(raw_text):
            return {"lang_code": "hi", "lang_name": "Hindi (Devanagari)", "is_code_mixed": False}
        if TAMIL_RE.search(raw_text):
            return {"lang_code": "ta", "lang_name": "Tamil", "is_code_mixed": False}
        if TELUGU_RE.search(raw_text):
            return {"lang_code": "te", "lang_name": "Telugu", "is_code_mixed": False}
        if BENGALI_RE.search(raw_text):
            return {"lang_code": "bn", "lang_name": "Bengali", "is_code_mixed": False}

        # Check Romanized Hinglish vs Pure English
        words = re.findall(r"\w+", raw_text.lower())
        hinglish_matches = sum(1 for w in words if w in HINGLISH_KEYWORDS)

        if hinglish_matches >= 1:
            log.info(f"Detected Code-Mixed Hinglish input for: '{text}'")
            return {"lang_code": "hinglish", "lang_name": "Hinglish (Code-Mixed)", "is_code_mixed": True}

        return {"lang_code": "en", "lang_name": "English", "is_code_mixed": False}

    def normalize_hinglish_to_english(self, text: str) -> str:
        """Translates Hinglish key intents to clean English keywords for optimal RAG retrieval."""
        normalized = text.lower()
        replacements = [
            (r"\bmera\b|\bmeri\b", "my"),
            (r"\bled bulb ke liye\b", "for LED bulb"),
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
