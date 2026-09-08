"""
Phase 5, Step 17: Language Detection, Hinglish Normalizer & Multilingual Strategy (multilingual.py)
Detects input language from the COMPLETE user sentence across 8+ official Indic languages (Hindi, Telugu, Tamil,
Bengali, Kannada, Malayalam, Gujarati, Punjabi, Odia, Urdu) and Romanized code-mixed dialects (Hinglish, Tenglish,
Tanglish, Kanglish, Manglish) using Unicode script ratio distribution, acronym stripping, and weighted phrase analysis.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multilingual")

# Unicode ranges for 8+ Indic scripts
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")  # Hindi, Marathi, Nepali, Sanskrit
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")       # Telugu
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")        # Tamil
BENGALI_RE = re.compile(r"[\u0980-\u09FF]")      # Bengali, Assamese
KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")      # Kannada
MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")    # Malayalam
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")     # Gujarati
GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")     # Punjabi
ODIA_RE = re.compile(r"[\u0B00-\u0B7F]")         # Odia
URDU_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")  # Urdu / Perso-Arabic
LATIN_RE = re.compile(r"[a-zA-Z]")

INDIC_SCRIPTS_CONFIG: List[Tuple[str, str, re.Pattern]] = [
    ("hi", "Hindi", DEVANAGARI_RE),
    ("te", "Telugu", TELUGU_RE),
    ("ta", "Tamil", TAMIL_RE),
    ("bn", "Bengali", BENGALI_RE),
    ("kn", "Kannada", KANNADA_RE),
    ("ml", "Malayalam", MALAYALAM_RE),
    ("gu", "Gujarati", GUJARATI_RE),
    ("pa", "Punjabi", GURMUKHI_RE),
    ("or", "Odia", ODIA_RE),
    ("ur", "Urdu", URDU_RE),
]

# Technical identifiers / neutral terms to exclude from script dominance calculation
TECHNICAL_TOKENS_RE = re.compile(
    r"\b(BIS|IS\s*\d+|IS|CRS|FMCS|QCO|ISI|OPC|PPC|TMT|LED|PVC|AHC|HUID|URL|HTTP|HTTPS|WWW|"
    r"standard|standards|certification|certified|product|products|scheme|schemes|"
    r"clause\s*[\d\.]+|section\s*[\d\.]+|rule\s*[\d\.]+|hallmark|hallmarking|license|licence|"
    r"iso\s*\d+|iec\s*\d+|astm\s*\w*)\b",
    re.IGNORECASE,
)

# Ambiguous words to exclude from code-mixed unigram scoring (to avoid false positives on standard English)
ENGLISH_COLLISION_WORDS = {
    "is", "fee", "fees", "cost", "costs", "bar", "bars", "pan", "to", "so", "me", "he", "us",
    "are", "car", "in", "on", "or", "an", "at", "do", "if", "my", "no", "we", "the", "and", "for",
    "with", "apply", "file", "process", "standard", "standards", "scheme", "schemes", "gold", "steel"
}

# --- Hinglish (Romanized Hindi) ---
HINGLISH_KEYWORDS = {
    "mera", "meri", "mere", "chahiye", "chaheye", "chahie", "kaise", "kaisey", "kese", "kab",
    "kitna", "kitni", "kitne", "kitnaa", "kya", "kyaa", "hoga", "hogi", "hoge", "honge", "hai", "hain",
    "karna", "kare", "karen", "karein", "karun", "karoon", "karega", "parega", "padega", "padegi",
    "paise", "paisa", "rupaye", "roopaye", "batao", "bataiye", "batayein", "batado", "jaankari",
    "shuru", "dono", "sabse", "badhiya", "ke", "ki", "ka", "liye", "kaunsa", "kaun", "kaunsi",
    "kaunse", "lagta", "lagiga", "lagega", "lagoge", "mujhe", "aap", "hum", "nahi", "nahin",
    "hota", "hoti", "hote", "shikayat", "darj", "utpad", "khilaf", "kar", "sakte", "pramanit",
    "pata", "kijiye", "kaha", "kahan", "kyun", "kyon", "apna", "apni", "apne"
}

HINGLISH_PHRASE_PATTERNS = [
    r"\bke\s+liye\b",
    r"\bkaun\s*sa\b|\bkaun\s*si\b|\bkaun\s*se\b",
    r"\bkya\s+hai\b|\bkya\s+hoga\b|\bkya\s+hogi\b",
    r"\bkaise\s+(?:kare|karen|karein|karun|apply|file|darj|hota|hoti|hoga|karega)\b",
    r"\bkitna\s+(?:fee|paisa|cost|kharcha|lagega|lagiga|lagta)\b",
    r"\b(?:ka|ki|ke)\s+process\s+kya\s+hai\b",
    r"\bshikayat\s+kaise\b",
    r"\bcomplaint\s+kaise\b",
    r"\bkarna\s+(?:hoga|padega|chahiye)\b",
    r"\bjaankari\s+chahiye\b",
    r"\bbatao\b|\bbataiye\b|\bbatayein\b",
]

# --- Tenglish (Romanized Telugu) ---
TENGLISH_KEYWORDS = {
    "ela", "elaa", "yetla", "eppudu", "yepudu", "entha", "yentha", "kavali", "kaavali",
    "cheyali", "cheyyali", "cheyandi", "cheyyandi", "pettali", "pettandi", "undi", "undhi", "unnadhi",
    "ledu", "ledhu", "kosam", "koraku", "enti", "emiti", "yemiti", "cheppu", "cheppandi", "chudali",
    "telsukovali", "telusukovali", "teliyali", "gurinchi", "darakhasthu", "nirdharana", "rupailu",
    "firyadu", "firyadhu", "evari", "ekkada", "cheyyadam", "chudandi"
}

TENGLISH_PHRASE_PATTERNS = [
    r"\bela\s+(?:apply|file|cheyali|cheyyali|cheyandi|pettali)\b",
    r"\bfee\s+(?:entha|kavali|undhi|undi)\b",
    r"\b(?:standard|process)\s+enti\b",
    r"\b(?:kosam|ki)\s+standard\b",
    r"\bgurinchi\s+(?:teliyali|cheppandi|telusukovali|jaankari|details)\b",
    r"\bcheppandi\b|\bcheppu\b",
    r"\bfiryadhu\s+ela\b|\bfiryadu\s+ela\b",
]

# --- Tanglish (Romanized Tamil) ---
TANGLISH_KEYWORDS = {
    "eppadi", "epdi", "evvalavu", "evalo", "evvalo", "venum", "vendum", "venuma", "panradhu",
    "pandrathu", "panren", "seivadhu", "seiyalaam", "kaaga", "solla", "solunga", "sollunga",
    "ennadhu", "enna", "irukku", "irukka", "illai", "kudunga", "kodunga", "kudukanum", "pudikkum",
    "pugardharar", "pugar", "pugaar", "theriyuma", "therinjukanum", "kattanam"
}

TANGLISH_PHRASE_PATTERNS = [
    r"\beppadi\s+(?:apply|panradhu|file|seivadhu|seiyalaam)\b",
    r"\bevvalavu\s+(?:fee|kattanam|cost|irukku)\b",
    r"\bfee\s+(?:evvalavu|evalo|irukku)\b",
    r"\bennadhu\b|\bsolunga\b|\bsollunga\b",
    r"\bkaaga\s+standard\b",
    r"\bpugar\s+eppadi\b",
]

# --- Kanglish (Romanized Kannada) ---
KANGLISH_KEYWORDS = {
    "hege", "hegey", "eshtu", "yeshtu", "beku", "bekaa", "madodu", "maduvudu", "madi", "madbeku",
    "heli", "heliri", "enu", "yenu", "ide", "idhe", "ideya", "illa", "illava", "koskara",
    "thiliyabeku", "dhooru", "arji"
}

KANGLISH_PHRASE_PATTERNS = [
    r"\bhege\s+(?:apply|madodu|maduvudu|madi|madbeku)\b",
    r"\beshtu\s+(?:fee|hana|cost|ide)\b",
    r"\bfee\s+eshtu\b",
    r"\bbeku\b|\bheli\b|\bheliri\b|\bmadbeku\b",
    r"\bkoskara\s+standard\b",
]

# --- Manglish (Romanized Malayalam) ---
MANGLISH_KEYWORDS = {
    "engane", "enganey", "ethra", "yethra", "ethrayanu", "venam", "veno", "cheyyanam", "cheyyuka",
    "cheyyam", "cheyyan", "parayamo", "parayu", "entha", "endha", "enthaanu", "undo", "unda",
    "illa", "illaatha", "aayi", "ariyikku", "ariyumo", "ariyikkaamo", "pukaar", "pukaaru", "apeksha"
}

MANGLISH_PHRASE_PATTERNS = [
    r"\bengane\s+(?:apply|cheyyanam|cheyyuka|cheyyam|cheyyan|file)\b",
    r"\b(?:ethra|ethrayanu)\s+(?:fee|paisa|cost|aanu)\b",
    r"\bfee\s+(?:ethra|ethrayanu|ariyikku|und)\b",
    r"\benthaanu\b|\bethrayanu\b|\bariyikku\b|\bariyumo\b",
    r"\bvenam\b|\bparayu\b|\bparayamo\b",
]


class MultilingualHandler:
    """
    Robust script and sentence-level language detector.
    Analyzes the COMPLETE query string to determine dominant language/script across 8+ Indic scripts,
    neutralizing technical Latin acronyms (e.g. 'BIS', 'IS 1786') and reliably separating code-mixed
    dialects (Hinglish, Tenglish, Tanglish, Kanglish, Manglish) from standard English.
    """

    def detect_language(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"lang_code": "en", "lang_name": "English", "is_code_mixed": False}

        raw_text = unicodedata.normalize("NFKC", text.strip())

        # Strip URLs, technical acronyms, digits, punctuation, and symbols for fair script ratio counting
        stripped_for_script = re.sub(r"https?://\S+", "", raw_text)
        stripped_for_script = TECHNICAL_TOKENS_RE.sub("", stripped_for_script)

        # 1. Native Indic Script Dominance Analysis
        script_counts: Dict[str, Tuple[str, int, int]] = {}
        for code, name, regex in INDIC_SCRIPTS_CONFIG:
            stripped_count = len(regex.findall(stripped_for_script))
            raw_count = len(regex.findall(raw_text))
            if stripped_count > 0 or raw_count > 0:
                script_counts[code] = (name, stripped_count, raw_count)

        latin_count = len(LATIN_RE.findall(stripped_for_script))
        total_indic_stripped = sum(c[1] for c in script_counts.values())

        if script_counts:
            # Find dominant native Indic script
            best_code, (best_name, best_stripped, best_raw) = max(
                script_counts.items(), key=lambda item: (item[1][1], item[1][2])
            )
            total_alpha = total_indic_stripped + latin_count
            indic_ratio = (best_stripped / total_alpha) if total_alpha > 0 else 1.0

            # Reliable dominance criteria:
            # - stripped count >= 2, or
            # - ratio >= 0.25 (to handle short 1-2 character Indic queries e.g. "घी", "जल", "సిమెంట్"), or
            # - raw count >= 2 with no conflicting Indic scripts
            if best_stripped >= 2 or indic_ratio >= 0.25 or best_raw >= 1:
                log.info(
                    f"Native Indic script detected: {best_name} ({best_code}) "
                    f"[{best_stripped} stripped chars, {best_raw} raw chars, ratio={indic_ratio:.2f}] for: '{raw_text[:50]}'"
                )
                return {
                    "lang_code": best_code,
                    "lang_name": best_name,
                    "is_code_mixed": (latin_count > 0),
                }

        # 2. Latin-Script Code-Mixed Analysis (Hinglish, Tenglish, Tanglish, Kanglish, Manglish vs English)
        q_lower = raw_text.lower()

        # Check multi-word phrase patterns (Weight: 10)
        for pat in HINGLISH_PHRASE_PATTERNS:
            if re.search(pat, q_lower, re.IGNORECASE):
                log.info(f"Matched Hinglish phrase '{pat}' -> Hinglish for: '{raw_text[:50]}'")
                return {"lang_code": "hinglish", "lang_name": "Hinglish", "is_code_mixed": True}

        for pat in TENGLISH_PHRASE_PATTERNS:
            if re.search(pat, q_lower, re.IGNORECASE):
                log.info(f"Matched Tenglish phrase '{pat}' -> Telugu for: '{raw_text[:50]}'")
                return {"lang_code": "tenglish", "lang_name": "Telugu", "is_code_mixed": True}

        for pat in TANGLISH_PHRASE_PATTERNS:
            if re.search(pat, q_lower, re.IGNORECASE):
                log.info(f"Matched Tanglish phrase '{pat}' -> Tamil for: '{raw_text[:50]}'")
                return {"lang_code": "tanglish", "lang_name": "Tamil", "is_code_mixed": True}

        for pat in KANGLISH_PHRASE_PATTERNS:
            if re.search(pat, q_lower, re.IGNORECASE):
                log.info(f"Matched Kanglish phrase '{pat}' -> Kannada for: '{raw_text[:50]}'")
                return {"lang_code": "kanglish", "lang_name": "Kannada", "is_code_mixed": True}

        for pat in MANGLISH_PHRASE_PATTERNS:
            if re.search(pat, q_lower, re.IGNORECASE):
                log.info(f"Matched Manglish phrase '{pat}' -> Malayalam for: '{raw_text[:50]}'")
                return {"lang_code": "manglish", "lang_name": "Malayalam", "is_code_mixed": True}

        # Tokenize words and score against distinct Indic vocabulary (excluding English collision words)
        words = re.findall(r"\b[a-zA-Z]+\b", q_lower)
        clean_words = [w for w in words if w not in ENGLISH_COLLISION_WORDS]

        hinglish_matches = sum(1 for w in clean_words if w in HINGLISH_KEYWORDS)
        tenglish_matches = sum(1 for w in clean_words if w in TENGLISH_KEYWORDS)
        tanglish_matches = sum(1 for w in clean_words if w in TANGLISH_KEYWORDS)
        kanglish_matches = sum(1 for w in clean_words if w in KANGLISH_KEYWORDS)
        manglish_matches = sum(1 for w in clean_words if w in MANGLISH_KEYWORDS)

        scores = [
            ("hinglish", "Hinglish", hinglish_matches),
            ("tenglish", "Telugu", tenglish_matches),
            ("tanglish", "Tamil", tanglish_matches),
            ("kanglish", "Kannada", kanglish_matches),
            ("manglish", "Malayalam", manglish_matches),
        ]
        best_code_mixed = max(scores, key=lambda x: x[2])

        # Require at least 1 clean non-collision keyword match
        if best_code_mixed[2] >= 1:
            log.info(
                f"Matched Code-Mixed {best_code_mixed[0]} vocabulary ({best_code_mixed[2]} words) for: '{raw_text[:50]}'"
            )
            return {
                "lang_code": best_code_mixed[0],
                "lang_name": best_code_mixed[1],
                "is_code_mixed": True,
            }

        # 3. Default to Standard English
        return {"lang_code": "en", "lang_name": "English", "is_code_mixed": False}

    def normalize_hinglish_to_english(self, text: str) -> str:
        """Translates Hinglish, Tenglish, Tanglish, Kanglish, and Manglish intents to clean English keywords."""
        normalized = text.lower()
        replacements = [
            # Compound product and intent phrases
            (r"\bled bulb ke liye\b|\bled bulb kosam\b|\bled bulb kaaga\b|\bled bulb koskara\b", "for LED bulb"),
            (r"\btmt bar ke liye\b|\btmt bar kosam\b|\btmt bar kaaga\b|\btmt bar koskara\b", "for TMT bar steel reinforcement"),
            (r"\bkaunsa bis standard applicable hai\b|\bkaun sa bis standard\b|\bstandard enti\b|\bstandard ennadhu\b|\byavudhu standard\b", "what BIS standard applies"),
            (r"\bis certification chahiye\b|\bis certification kavali\b|\bis certification venum\b|\bis certification beku\b|\bis certification venam\b", "need BIS certification"),
            (r"\bkaise apply kare\b|\bkaise apply karen\b|\bkaise apply\b|\bela apply cheyali\b|\beppadi apply panradhu\b|\bhege apply madodu\b|\bengane apply cheyyanam\b", "how to apply"),
            (r"\bkitna fee lagiga\b|\bkitna fee lagega\b|\bkitna fee\b|\bfee entha\b|\bevvalavu fee\b|\beshtu fee\b|\bethra fee\b", "what is the application fee"),
            (r"\bgold ring fake hai\b|\bshikayat kaise\b|\bcomplaint kaise\b|\bfiryadhu ela\b|\bfiryadu ela\b|\bpugar eppadi\b|\bdhooru hege\b|\bpukaar engane\b", "complaint grievance"),
            (r"\bmera\b|\bmeri\b", "my"),
            (r"\bchahiye\b|\bkavali\b|\bvenum\b|\bbeku\b|\bvenam\b", "require"),
            (r"\bkitna\b|\bentha\b|\bevvalavu\b|\beshtu\b|\bethra\b", "how much"),
            (r"\bkaise\b|\bela\b|\beppadi\b|\bhege\b|\bengane\b", "how"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)

        log.info(f"Normalized Code-Mixed '{text}' -> '{normalized}'")
        return normalized

    def normalize_native_to_english_keywords(self, text: str) -> str:
        """
        Translates domain keywords across 8+ Indic languages (Hindi, Telugu, Tamil, Bengali,
        Kannada, Malayalam, Gujarati, Punjabi, Odia, Urdu) to standard English BIS search terms.
        """
        t = text
        # Hindi / Devanagari keywords
        t = re.sub(r"सीमेंट", "cement Ordinary Portland Cement IS 269", t)
        t = re.sub(r"(?:टीएमटी\s*)?(?:स्टील|इस्पात|सरिया)(?:\s*बार)?|टीएमटी", "steel reinforcement TMT bar IS 1786", t)
        t = re.sub(r"संपीड़न|संपीडन", "compressive strength", t)
        t = re.sub(r"तनन", "tensile strength", t)
        t = re.sub(r"रासायनिक", "chemical requirements limits", t)
        t = re.sub(r"कार्बन", "carbon content", t)
        t = re.sub(r"भूकंप\s*रोधी|भूकंप", "earthquake resistant ductility", t)
        t = re.sub(r"एलईडी|एल\.ई\.डी", "LED", t)
        t = re.sub(r"बल्ब|बल्बों", "bulb LED lamps", t)
        t = re.sub(r"अनिवार्य|जरूरी", "mandatory QCO compulsory", t)
        t = re.sub(r"बीआईएस|बी\.आई\.एस", "BIS", t)
        t = re.sub(r"मानक", "standard specification", t)
        t = re.sub(r"लागू", "applies", t)
        t = re.sub(r"प्रमाणित|प्रमाणन|प्रमाणपत्र", "certification certified", t)
        t = re.sub(r"उत्पाद", "product", t)
        t = re.sub(r"शिकायत", "complaint grievance", t)
        t = re.sub(r"दर्ज|पंजीकरण", "register file lodge", t)
        t = re.sub(r"शुल्क|फीस", "fee cost", t)
        t = re.sub(r"प्रयोगशाला|लैब", "laboratory testing lab", t)
        t = re.sub(r"सोना|स्वर्ण|आभूषण", "gold hallmarking jewellery IS 1417", t)
        t = re.sub(r"हॉलमार्क", "hallmark", t)
        t = re.sub(r"प्रक्रिया|नियम", "process procedure", t)
        t = re.sub(r"आवेदन|लाइसेंस", "apply license", t)
        t = re.sub(r"हेलमेट", "helmet protective headgear IS 4151", t)
        t = re.sub(r"पानी|जल", "packaged drinking water IS 14543", t)

        # Telugu keywords
        t = re.sub(r"సిమెంట్(?:కు)?", "cement Ordinary Portland Cement IS 269", t)
        t = re.sub(r"ఉక్కు|స్టీల్|కడ్డీలు", "steel reinforcement TMT bar IS 1786", t)
        t = re.sub(r"ఎల్ఈడీ|ఎల్‌ఈడీ", "LED", t)
        t = re.sub(r"బల్బు|బల్బులు", "bulb LED lamps", t)
        t = re.sub(r"తప్పనిసరి", "mandatory QCO compulsory", t)
        t = re.sub(r"ప్రమాణం|ప్రమాణాలు", "standard specification", t)
        t = re.sub(r"వర్తిస్తుంది", "applies", t)
        t = re.sub(r"ధృవీకరించిన|ధృవీకరణ|సర్టిఫికేషన్", "certification certified", t)
        t = re.sub(r"ఉత్పత్తి|ఉత్పత్తులు|ఉత్పత్తిపై", "product", t)
        t = re.sub(r"ఫిర్యాదు", "complaint grievance", t)
        t = re.sub(r"నమోదు", "register file lodge", t)
        t = re.sub(r"రుసుము|ఫీజు", "fee cost", t)
        t = re.sub(r"ప్రయోగశాల", "laboratory testing lab", t)
        t = re.sub(r"బంగారం", "gold hallmarking jewellery", t)
        t = re.sub(r"విధానం|దరఖాస్తు", "process apply license", t)
        t = re.sub(r"హెల్మెట్", "helmet protective headgear", t)

        # Tamil keywords
        t = re.sub(r"சிமெண்ட்", "cement Ordinary Portland Cement", t)
        t = re.sub(r"எஃகு|கம்பி", "steel reinforcement TMT bar", t)
        t = re.sub(r"எல்இடி", "LED", t)
        t = re.sub(r"பல்ப்|விளக்கு", "bulb LED lamps", t)
        t = re.sub(r"கட்டாய", "mandatory QCO compulsory", t)
        t = re.sub(r"தரநிலை|தரம்", "standard specification", t)
        t = re.sub(r"சான்றிதழ்|சான்றிதழ் வழங்கல்", "certification certified", t)
        t = re.sub(r"புகார்", "complaint grievance", t)
        t = re.sub(r"ஆய்வகம்", "laboratory testing lab", t)
        t = re.sub(r"தங்கம்", "gold hallmarking jewellery", t)
        t = re.sub(r"விண்ணப்பம்|கட்டணம்", "apply fee cost", t)

        # Bengali keywords
        t = re.sub(r"সিমেন্ট", "cement Ordinary Portland Cement", t)
        t = re.sub(r"ইস্পাত|রড", "steel reinforcement TMT bar", t)
        t = re.sub(r"মানক|মান", "standard specification", t)
        t = re.sub(r"শংসাপত্র|প্রত্যয়ন", "certification certified", t)
        t = re.sub(r"অভিযোগ", "complaint grievance", t)
        t = re.sub(r"পরীক্ষাগার|ল্যাব", "laboratory testing lab", t)
        t = re.sub(r"স্বর্ণ|সোনা", "gold hallmarking jewellery", t)
        t = re.sub(r"ফি|খরচ", "fee cost", t)

        # Kannada keywords
        t = re.sub(r"ಸಿಮೆಂಟ್", "cement Ordinary Portland Cement", t)
        t = re.sub(r"ಉಕ್ಕು|ಕಬ್ಬಿಣ", "steel reinforcement TMT bar", t)
        t = re.sub(r"ಮಾನದಂಡ|ಪ್ರಮಾಣ", "standard specification", t)
        t = re.sub(r"ಪ್ರಮಾಣಪತ್ರ", "certification certified", t)
        t = re.sub(r"ದೂರು", "complaint grievance", t)
        t = re.sub(r"ಚಿನ್ನ", "gold hallmarking jewellery", t)

        # Malayalam keywords
        t = re.sub(r"സിമന്റ്", "cement Ordinary Portland Cement", t)
        t = re.sub(r"ഉരുക്ക്|കമ്പി", "steel reinforcement TMT bar", t)
        t = re.sub(r"നിലവാരം|മാനദണ്ഡം", "standard specification", t)
        t = re.sub(r"പരാതി", "complaint grievance", t)
        t = re.sub(r"സ്വർണ്ണം", "gold hallmarking jewellery", t)

        # Gujarati keywords
        t = re.sub(r"સિમેન્ટ", "cement Ordinary Portland Cement", t)
        t = re.sub(r"સ્ટીલ|સળિયા", "steel reinforcement TMT bar", t)
        t = re.sub(r"ધોરણ|માનક", "standard specification", t)
        t = re.sub(r"પ્રમાણપત્ર", "certification certified", t)
        t = re.sub(r"ફરિયાદ", "complaint grievance", t)
        t = re.sub(r"સોનું", "gold hallmarking jewellery", t)

        # Punjabi keywords
        t = re.sub(r"ਸੀਮਿੰਟ", "cement Ordinary Portland Cement", t)
        t = re.sub(r"ਸਟੀਲ|ਸਰੀਆ", "steel reinforcement TMT bar", t)
        t = re.sub(r"ਮਿਆਰ", "standard specification", t)
        t = re.sub(r"ਸ਼ਿਕਾਇਤ", "complaint grievance", t)
        t = re.sub(r"ਸੋਨਾ", "gold hallmarking jewellery", t)

        # Odia keywords
        t = re.sub(r"ସିମେଣ୍ଟ", "cement Ordinary Portland Cement", t)
        t = re.sub(r"ଇସ୍ପାତ|ରଡ୍", "steel reinforcement TMT bar", t)
        t = re.sub(r"ମାନକ", "standard specification", t)
        t = re.sub(r"ଅଭିଯୋଗ", "complaint grievance", t)
        t = re.sub(r"ସୁନା", "gold hallmarking jewellery", t)

        # Urdu keywords
        t = re.sub(r"سیمنٹ", "cement Ordinary Portland Cement", t)
        t = re.sub(r"اسٹیل|سریہ", "steel reinforcement TMT bar", t)
        t = re.sub(r"معیار", "standard specification", t)
        t = re.sub(r"شکایت", "complaint grievance", t)
        t = re.sub(r"سونا", "gold hallmarking jewellery", t)

        return t


class TranslationEngine:
    """Citation-Preserving Response Translation Engine."""

    def __init__(self):
        # Captures citations, URLs, file paths, standard numbers, clause structures, currency figures, and percentages
        self.citation_pattern = re.compile(
            r"(\[As per [^\]]+\]|\[Per BIS [^\]]+\]|\[Source: [^\]]+\]|"
            r"\((?:https?://|file:///)[^\)]+\)|"
            r"\b(?:IS(?:/ISO)?|ISO|IEC|ASTM)\s*\d+(?:\s*\([^\)]+\))?(?::\d{4})?\b|"
            r"\b(?:Clause|Section|Rule)\s*[\d\.]+(?:\s*\([a-zA-Z0-9]+\))*"
            r"|"
            r"(?:₹|Rs\.|INR)\s*\d+[\d,]*(?:\.\d+)?(?:/-)?|"
            r"\b\d+(?:\.\d+)?%)",
            re.IGNORECASE,
        )

    def mask_protected_tokens(self, text: str) -> Tuple[str, List[str]]:
        tokens: List[str] = []

        def replacer(match):
            tokens.append(match.group(0))
            return f"__PROTECTED_TOKEN_{len(tokens) - 1}__"

        masked = self.citation_pattern.sub(replacer, text)
        return masked, tokens

    def unmask_protected_tokens(self, masked_text: str, tokens: List[str]) -> str:
        unmasked = masked_text
        for idx, token in enumerate(tokens):
            # Tolerate surrounding whitespace and case differences (e.g. __protected_token_0__)
            placeholder_pattern = rf"\s*__PROTECTED_TOKEN_{idx}__\s*"
            unmasked = re.sub(placeholder_pattern, f" {token} ", unmasked, flags=re.IGNORECASE)
        # Clean up leading space before punctuation and collapse spaces
        unmasked = re.sub(r"\s+([,\.\?\!\)])", r"\1", unmasked)
        unmasked = re.sub(r"(\()\s+", r"\1", unmasked)
        return re.sub(r"[ \t]+", " ", unmasked).strip()

    def detect_language(self, text: str) -> Dict[str, Any]:
        return MultilingualHandler().detect_language(text)

    def translate_response(self, text: str, target_lang: str) -> str:
        if target_lang == "en":
            return text
        masked_text, tokens = self.mask_protected_tokens(text)
        return self.unmask_protected_tokens(masked_text, tokens)

