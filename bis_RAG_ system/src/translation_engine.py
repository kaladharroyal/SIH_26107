"""
Phase 5, Step 18: Citation-Preserving Response Translation Engine (translation_engine.py)
Masks citations, URLs, and fee figures before translation and restores them byte-for-byte post-translation.
"""

import re
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("translation_engine")

# Dictionary of offline pre-verified translations for key regulatory phrases
HINDI_DICTIONARY = {
    "Based on official Bureau of Indian Standards documentation": "भारतीय मानक ब्यूरो के आधिकारिक दस्तावेजों के अनुसार",
    "Product Category": "उत्पाद श्रेणी",
    "Applicable Indian Standard": "लागू भारतीय मानक",
    "Certification Status": "प्रमाणन स्थिति",
    "BIS Scheme": "बीआईएस योजना",
    "Quality Control Order (QCO)": "गुणवत्ता नियंत्रण आदेश (क्यूसीओ)",
    "Official Fee & Duration Schedule": "आधिकारिक शुल्क और अवधि तालिका",
    "Application Fee": "आवेदन शुल्क",
    "Factory Inspection Charge": "कारखाना निरीक्षण शुल्क",
    "Lab Test Report Validity": "प्रयोगशाला परीक्षण रिपोर्ट की वैधता",
    "Licence Duration": "लाइसेंस अवधि",
    "Step-by-Step Licensing Procedure": "चरण-दर-चरण लाइसेंसिंग प्रक्रिया",
    "Official BIS Portal Link": "आधिकारिक बीआईएस पोर्टल लिंक",
    "Verification Sources & Citations": "सत्यापन स्रोत और उद्धरण",
    "Official Regulation": "आधिकारिक नियम",
    "FAQ Guideline": "एफएक्यू दिशा-निर्देश",
}


class TranslationEngine:
    def __init__(self):
        # Regex patterns to preserve as untranslatable tokens
        self.citation_pattern = re.compile(
            r"(\[As per IS [^\]]+\]|\[Per BIS FAQ [^\]]+\]|\(file:///[^\)]+\)|\(https?://[^\)]+\)|₹\d+[\d,]*|\bIS\s*\d+(?:\s*\(Part\s*\d+\))?:\d{4}\b|\bClause\s*[\d\.]+\b)",
            re.IGNORECASE,
        )

    def mask_protected_tokens(self, text: str) -> (str, List[str]):
        tokens = []

        def replacer(match):
            tokens.append(match.group(0))
            return f" __PROTECTED_TOKEN_{len(tokens) - 1}__ "

        masked_text = self.citation_pattern.sub(replacer, text)
        log.info(f"Masked {len(tokens)} protected citation & numeric tokens.")
        return masked_text, tokens

    def unmask_protected_tokens(self, masked_text: str, tokens: List[str]) -> str:
        unmasked = masked_text
        for idx, token in enumerate(tokens):
            placeholder = f"__PROTECTED_TOKEN_{idx}__"
            # Remove potential space padding added during masking
            unmasked = re.sub(rf"\s*{re.escape(placeholder)}\s*", f" {token} ", unmasked)

        return unmasked.strip()

    def translate_response(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text

        log.info(f"Translating response to target language: '{target_lang}' with citation protection...")
        
        # 1. Mask citations and numbers
        masked_text, tokens = self.mask_protected_tokens(text)

        # 2. Perform phrase-level translation
        translated_masked = masked_text
        if target_lang in ["hi", "hinglish"]:
            for eng, hindi in HINDI_DICTIONARY.items():
                translated_masked = translated_masked.replace(eng, hindi)

        # 3. Unmask protected citation tokens byte-for-byte
        final_translated = self.unmask_protected_tokens(translated_masked, tokens)
        return final_translated


if __name__ == "__main__":
    translator = TranslationEngine()
    raw_response = (
        "Based on official Bureau of Indian Standards documentation: "
        "Application Fee: ₹1,000 for LED Bulbs. "
        "As per IS 16102:2012, Clause 4.2 (file:///raw_data/IS_16102.pdf#page=4)."
    )
    hi_res = translator.translate_response(raw_response, target_lang="hi")
    print("Original:\n", raw_response)
    print("\nHindi Translated (Citations Preserved):\n", hi_res)
