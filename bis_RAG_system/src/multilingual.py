"""
Phase 4 (integrated): Language Detection, Hinglish Normalizer & Multilingual Strategy (multilingual.py)
Detects input language from the COMPLETE user sentence (English, Hindi, Telugu, Tamil, Bengali, Hinglish)
using script distribution and full-sentence phrase analysis without reliance on initial tokens.
"""

import logging
import re
import unicodedata
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multilingual")

# Unicode ranges for Indic scripts
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_RE = re.compile(r"[a-zA-Z]")

# Technical identifiers / neutral terms to exclude from script dominance calculation
TECHNICAL_TOKENS_RE = re.compile(
    r"\b(BIS|IS\s*\d+|IS|CRS|FMCS|QCO|ISI|OPC|PPC|TMT|LED|PVC|AHC|HUID|URL|HTTP|HTTPS|WWW|standard|standards|certification|product|products)\b",
    re.IGNORECASE,
)

# Common Romanized Hinglish vocabulary (functional words, verbs, pronouns, prepositions)
HINGLISH_KEYWORDS = {
    "mera", "meri", "mere", "chahiye", "kaise", "kab", "kitna", "kitni", "kya", "hoga", "hogi", "hai", "hain",
    "karna", "kare", "karen", "karein", "karun", "karega", "parega", "padega", "paise", "paisa", "rupaye", "fee",
    "roopaye", "batao", "bataiye", "batayein", "jaankari", "shuru", "dono", "sabse", "badhiya", "ke", "liye",
    "kaunsa", "kaun", "kaunsi", "kaunse", "lagta", "lagiga", "lagoge", "mujhe", "aap", "hum", "nahi", "nahin",
    "hota", "hoti", "hote", "shikayat", "darj", "utpad", "khilaf", "kar", "sakte", "shikayat", "pramanit",
    "pata", "karein", "kijiye", "kaha", "kahan"
}

# Multi-word Hinglish phrase patterns across complete sentence
HINGLISH_PHRASE_PATTERNS = [
    r"\bke\s+liye\b",
    r"\bkaun\s*sa\b|\bkaun\s*si\b|\bkaun\s*se\b",
    r"\bkya\s+hai\b|\bkya\s+hoga\b|\bkya\s+hogi\b",
    r"\bkaise\s+(?:kare|karen|karein|karun|apply|file|darj|hota|hoti|hoga)\b",
    r"\bkitna\s+(?:fee|paisa|cost|kharcha|lagega|lagiga|lagta)\b",
    r"\b(?:ka|ki|ke)\s+process\s+kya\s+hai\b",
    r"\bshikayat\s+kaise\b",
    r"\bcomplaint\s+kaise\b",
    r"\bkarna\s+(?:hoga|padega|chahiye)\b",
    r"\bjaankari\s+chahiye\b",
    r"\bbatao\b|\bbataiye\b|\bbatayein\b",
]


class MultilingualHandler:
    """
    Robust script and sentence-level language detector.
    Analyzes the COMPLETE query string to determine dominant language/script,
    ensuring initial Latin tokens (e.g. 'BIS', 'IS 1786') do not skew detection.
    """

    def detect_language(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"lang_code": "en", "lang_name": "English", "is_code_mixed": False}

        raw_text = unicodedata.normalize("NFKC", text.strip())

        # Strip technical acronyms, numbers, punctuation, URLs, and symbols for fair script counting
        stripped_for_script = re.sub(r"https?://\S+", "", raw_text)
        stripped_for_script = TECHNICAL_TOKENS_RE.sub("", stripped_for_script)
        
        # Count characters by script across the entire sentence
        devanagari_count = len(DEVANAGARI_RE.findall(stripped_for_script))
        telugu_count = len(TELUGU_RE.findall(stripped_for_script))
        tamil_count = len(TAMIL_RE.findall(stripped_for_script))
        bengali_count = len(BENGALI_RE.findall(stripped_for_script))
        latin_count = len(LATIN_RE.findall(stripped_for_script))

        # Check raw text directly for any Indic characters
        raw_devanagari = len(DEVANAGARI_RE.findall(raw_text))
        raw_telugu = len(TELUGU_RE.findall(raw_text))
        raw_tamil = len(TAMIL_RE.findall(raw_text))
        raw_bengali = len(BENGALI_RE.findall(raw_text))

        # 1. Native Indic Script Dominance (Complete Sentence Analysis)
        if devanagari_count >= 3 or raw_devanagari >= 3:
            log.info(f"Complete sentence analysis detected Devanagari ({devanagari_count} chars) -> Hindi for: '{raw_text[:50]}'")
            return {"lang_code": "hi", "lang_name": "Hindi", "is_code_mixed": (latin_count > 0)}

        if telugu_count >= 3 or raw_telugu >= 3:
            log.info(f"Complete sentence analysis detected Telugu ({telugu_count} chars) -> Telugu for: '{raw_text[:50]}'")
            return {"lang_code": "te", "lang_name": "Telugu", "is_code_mixed": (latin_count > 0)}

        if tamil_count >= 3 or raw_tamil >= 3:
            log.info(f"Complete sentence analysis detected Tamil ({tamil_count} chars) -> Tamil for: '{raw_text[:50]}'")
            return {"lang_code": "ta", "lang_name": "Tamil", "is_code_mixed": (latin_count > 0)}

        if bengali_count >= 3 or raw_bengali >= 3:
            log.info(f"Complete sentence analysis detected Bengali ({bengali_count} chars) -> Bengali for: '{raw_text[:50]}'")
            return {"lang_code": "bn", "lang_name": "Bengali", "is_code_mixed": (latin_count > 0)}

        if raw_devanagari > 0 and raw_devanagari >= raw_telugu:
            return {"lang_code": "hi", "lang_name": "Hindi", "is_code_mixed": True}
        if raw_telugu > 0:
            return {"lang_code": "te", "lang_name": "Telugu", "is_code_mixed": True}

        # 2. Latin-Script Analysis: Hinglish vs Pure English across the COMPLETE sentence
        q_lower = raw_text.lower()
        
        # Check full-sentence phrase matches first
        for pat in HINGLISH_PHRASE_PATTERNS:
            if re.search(pat, q_lower, re.IGNORECASE):
                log.info(f"Complete sentence matched Hinglish phrase pattern '{pat}' -> Hinglish for: '{raw_text[:50]}'")
                return {"lang_code": "hinglish", "lang_name": "Hinglish", "is_code_mixed": True}

        # Tokenize words and count Hinglish vocabulary
        words = re.findall(r"\b[a-zA-Z]+\b", q_lower)
        non_tech_words = [w for w in words if w not in {"bis", "is", "crs", "fmcs", "qco", "isi", "opc", "ppc", "tmt", "led", "pvc"}]
        hinglish_matches = sum(1 for w in non_tech_words if w in HINGLISH_KEYWORDS)

        if hinglish_matches >= 1:
            log.info(f"Complete sentence matched Hinglish vocabulary ({hinglish_matches} words) -> Hinglish for: '{raw_text[:50]}'")
            return {"lang_code": "hinglish", "lang_name": "Hinglish", "is_code_mixed": True}

        # 3. Default to English
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
            (r"\bkaise apply kare\b|\bkaise apply karen\b|\bkaise apply\b", "how to apply"),
            (r"\bkitna fee lagiga\b|\bkitna fee lagega\b|\bkitna fee\b", "what is the application fee"),
            (r"\bgold ring fake hai\b|\bshikayat kaise\b|\bcomplaint kaise\b", "complaint grievance"),
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
        t = re.sub(r"स्कीम(?:[-–\s]*[i|ii|x|1|2])?|योजना", "scheme", t)
        t = re.sub(r"सीमेंट", "cement Ordinary Portland Cement", t)
        t = re.sub(r"स्टील|इस्पात", "steel reinforcement", t)
        t = re.sub(r"मानक", "standard", t)
        t = re.sub(r"लागू", "applies", t)
        t = re.sub(r"प्रमाणित|प्रमाणन|प्रमाणपत्र", "certification certified", t)
        t = re.sub(r"उत्पाद", "product", t)
        t = re.sub(r"शिकायत", "complaint grievance", t)
        t = re.sub(r"दर्ज|पंजीकरण", "register file lodge", t)
        t = re.sub(r"शुल्क|फीस", "fee cost", t)
        t = re.sub(r"परीक्षण", "testing", t)
        t = re.sub(r"प्रयोगशाला(?:एं|ओं|शाला)?|लैब", "laboratory testing lab", t)
        t = re.sub(r"सोना|स्वर्ण|आभूषण", "gold hallmarking jewellery", t)
        t = re.sub(r"हॉलमार्क", "hallmark", t)
        t = re.sub(r"प्रक्रिया|नियम", "process procedure", t)
        t = re.sub(r"आवेदन|लाइसेंस", "apply license", t)
        t = re.sub(r"कहाँ|कहां", "where", t)
        t = re.sub(r"कैसे\s+करें", "how to", t)
        
        # Telugu keywords
        t = re.sub(r"స్కీమ్(?:[-–\s]*[i|ii|x|1|2])?|పథకం", "scheme", t)
        t = re.sub(r"సిమెంట్(?:కు)?", "cement Ordinary Portland Cement", t)
        t = re.sub(r"ఉక్కు", "steel reinforcement", t)
        t = re.sub(r"ప్రమాణం|ప్రమాణాలు", "standard", t)
        t = re.sub(r"వర్తిస్తుంది", "applies", t)
        t = re.sub(r"ధృవీకరించిన|ధృవీకరణ", "certification certified", t)
        t = re.sub(r"ఉత్పత్తి|ఉత్పత్తులు|ఉత్పత్తిపై", "product", t)
        t = re.sub(r"ఫిర్యాదు", "complaint grievance", t)
        t = re.sub(r"నమోదు", "register file lodge", t)
        t = re.sub(r"రుసుము", "fee cost", t)
        t = re.sub(r"పరీక్ష(?:లు|ణ|ా)?", "testing", t)
        t = re.sub(r"ప్రయోగశాల(?:లు)?", "laboratory testing lab", t)
        t = re.sub(r"బంగారం", "gold hallmarking", t)
        t = re.sub(r"విధానం|దరఖాస్తు", "process apply", t)
        t = re.sub(r"ఎక్కడ", "where", t)
        t = re.sub(r"ఎలా", "how to", t)

        # Tamil keywords
        t = re.sub(r"பிஐஎஸ்|பி\.ஐ\.எஸ்", "BIS", t)
        t = re.sub(r"எல்இடி|எல்\.இ\.டி", "LED", t)
        t = re.sub(r"சான்றிதழ்(?:கள்)?|சான்றளிப்பு", "certification certified", t)
        t = re.sub(r"தரநிலை(?:கள்)?", "standard", t)
        t = re.sub(r"பல்புகளுக்கு|பல்பு(?:கள்)?(?:க்கு)?", "bulb LED", t)
        t = re.sub(r"சிமெண்ட்|சிமென்ட்", "cement Ordinary Portland Cement", t)
        t = re.sub(r"எஃகு|இரும்பு|கம்பிகள்", "steel reinforcement TMT", t)
        t = re.sub(r"கட்டாயம்|கட்டாயமா(?:க)?", "mandatory compulsory", t)
        t = re.sub(r"பொரு(?:ட்கள்|ள்)", "product", t)
        t = re.sub(r"புகார்", "complaint grievance", t)
        t = re.sub(r"பதிவு", "register file lodge", t)
        t = re.sub(r"கட்டணம்", "fee cost", t)
        t = re.sub(r"ஆய்வகம்|பரிசோதனை", "laboratory testing lab", t)
        t = re.sub(r"தங்கம்|நகை", "gold hallmarking jewellery", t)
        t = re.sub(r"விண்ணப்பம்|நடைமுறை", "apply process", t)
        t = re.sub(r"திட்டம்", "scheme", t)
        t = re.sub(r"எங்கே", "where", t)
        t = re.sub(r"எப்படி", "how to", t)

        # Bengali keywords
        t = re.sub(r"বিআইএস", "BIS", t)
        t = re.sub(r"এলইডি", "LED", t)
        t = re.sub(r"মানক|মান|মানদণ্ড", "standard", t)
        t = re.sub(r"সিমেন্ট", "cement Ordinary Portland Cement", t)
        t = re.sub(r"প্রমাণপত্র|সার্টিফিকেশন|শংসাপত্র", "certification certified", t)
        t = re.sub(r"বাল্বের|বাল্ব", "bulb LED", t)
        t = re.sub(r"ইস্পাত|লোহা", "steel reinforcement TMT", t)
        t = re.sub(r"বাধ্যতামূলক", "mandatory compulsory", t)
        t = re.sub(r"পণ্য", "product", t)
        t = re.sub(r"অভিযোগ", "complaint grievance", t)
        t = re.sub(r"দায়ের|নিবন্ধন", "register file lodge", t)
        t = re.sub(r"ফি|খরচ", "fee cost", t)
        t = re.sub(r"গবেষণাগার|পরীক্ষাগার|ল্যাব", "laboratory testing lab", t)
        t = re.sub(r"সোনা|স্বর্ণ|গয়না|হলমার্ক", "gold hallmarking jewellery hallmark", t)
        t = re.sub(r"আবেদন|প্রক্রিয়া", "apply process", t)
        t = re.sub(r"স্কিম|যোজনা", "scheme", t)
        t = re.sub(r"পরীক্ষা", "testing", t)
        t = re.sub(r"কোথায়", "where", t)
        t = re.sub(r"কীভাবে", "how to", t)
        return t


if __name__ == "__main__":
    handler = MultilingualHandler()
    samples = [
        "BIS प्रमाणित उत्पाद के खिलाफ शिकायत कैसे दर्ज कर सकते हैं?",
        "सीमेंट के लिए कौन सा BIS मानक लागू होता है?",
        "BIS प्रमाणित उत्पाद के लिए शिकायत कैसे दर्ज करें?",
        "BIS ధృవీకరించిన ఉత్పత్తిపై ఫిర్యాదు ఎలా నమోదు చేయాలి?",
        "సిమెంట్కు ఏ BIS ప్రమాణం వర్తిస్తుంది?",
        "BIS certification ka process kya hai?",
        "TMT bar ke liye kaunsa BIS standard applicable hai?",
        "What BIS standard applies to cement?",
        "How can a manufacturer apply for BIS certification?",
        "Can I apply for BIS प्रमाणन online?",
    ]
    for s in samples:
        det = handler.detect_language(s)
        print(f"'{s}' -> Lang: {det['lang_name']} ({det['lang_code']})")
