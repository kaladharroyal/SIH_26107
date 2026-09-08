"""
Verification Test Suite: Multilingual Detection & Normalization Engine (Feature 13)
Verifies:
1. Native Indic Script Dominance across 10 official languages (Hindi, Telugu, Tamil, Bengali, Kannada, Malayalam, Gujarati, Punjabi, Odia, Urdu)
2. Short 1-2 character Indic queries without false rejections
3. Technical Acronym Neutralization (ensuring initial Latin tokens don't skew script detection)
4. Code-Mixed Detection for Hinglish, Tenglish, Tanglish, Kanglish, and Manglish
5. Prevention of False-Positive Code-Mixed Detection on standard English queries containing collision words
6. Domain Keyword Normalization for BM25 sparse search across native and code-mixed queries
7. Citation & Technical Token Masking/Unmasking Preservation
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src and root are on Python path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
SRC_DIR = BASE_DIR / "src"
for path in [SRC_DIR, BASE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multilingual import MultilingualHandler, TranslationEngine


def test_indic_script_dominance():
    print("\n" + "=" * 65)
    print("TEST 1: Native Indic Script Dominance (10 Official Languages)")
    print("=" * 65)

    handler = MultilingualHandler()

    test_cases = [
        ("Hindi", "सीमेंट के लिए कौन सा BIS मानक लागू होता है?", "hi", "Hindi"),
        ("Telugu", "సిమెంట్కు ఏ BIS ప్రమాణం వర్తిస్తుంది?", "te", "Telugu"),
        ("Tamil", "சிமெண்டிற்கு எந்த BIS தரநிலை பொருந்தும்?", "ta", "Tamil"),
        ("Bengali", "সিমেন্টের জন্য কোন BIS মানক প্রযোজ্য?", "bn", "Bengali"),
        ("Kannada", "ಸಿಮೆಂಟ್ ಗೆ ಯಾವ BIS ಮಾನದಂಡ ಅನ್ವಯಿಸುತ್ತದೆ?", "kn", "Kannada"),
        ("Malayalam", "സിമന്റിന് ഏത് BIS നിലവാരമാണ് ബാധകം?", "ml", "Malayalam"),
        ("Gujarati", "સિમેન્ટ માટે કયો BIS ધોરણ લાગુ પડે છે?", "gu", "Gujarati"),
        ("Punjabi", "ਸੀਮਿੰਟ ਲਈ ਕਿਹੜਾ BIS ਮਿਆਰ ਲਾਗੂ ਹੁੰਦਾ ਹੈ?", "pa", "Punjabi"),
        ("Odia", "ସିମେଣ୍ଟ ପାଇଁ କେଉଁ BIS ମାନକ ପ୍ରଯୁଜ୍ୟ?", "or", "Odia"),
        ("Urdu", "سیمنٹ کے لیے کون سا بی آئی ایس معیار لاگو ہوتا ہے؟", "ur", "Urdu"),
        ("English", "What BIS standard applies to cement and concrete?", "en", "English"),
    ]

    for label, query, exp_code, exp_name in test_cases:
        res = handler.detect_language(query)
        print(f"[{label:10}] Query: '{query[:45]}...' -> Detected: {res['lang_name']} ({res['lang_code']})")
        assert res["lang_code"] == exp_code, f"Expected {exp_code}, got {res['lang_code']}"
        assert res["lang_name"] == exp_name, f"Expected {exp_name}, got {res['lang_name']}"

    print("\n  [PASS] All 10 official Indic scripts accurately identified across complete sentences.")


def test_short_indic_queries():
    print("\n" + "=" * 65)
    print("TEST 2: Short & Single-Word Indic Queries (1-2 Characters)")
    print("=" * 65)

    handler = MultilingualHandler()

    short_queries = [
        ("Hindi Short 1", "जल", "hi", "Hindi"),
        ("Hindi Short 2", "घी", "hi", "Hindi"),
        ("Hindi Short 3", "दूध", "hi", "Hindi"),
        ("Hindi Product", "सरिया", "hi", "Hindi"),
        ("Telugu Short", "నార", "te", "Telugu"),
        ("Telugu Product", "సిమెంట్", "te", "Telugu"),
        ("Tamil Product", "எஃகு", "ta", "Tamil"),
        ("Kannada Product", "ಉಕ್ಕು", "kn", "Kannada"),
        ("Bengali Product", "সোনা", "bn", "Bengali"),
    ]

    for label, query, exp_code, exp_name in short_queries:
        res = handler.detect_language(query)
        print(f"[{label:16}] Query: '{query}' -> Detected: {res['lang_name']} ({res['lang_code']})")
        assert res["lang_code"] == exp_code, f"Expected {exp_code}, got {res['lang_code']}"

    print("\n  [PASS] Short 1-2 character Indic words correctly recognized without false rejections.")


def test_acronym_neutralization():
    print("\n" + "=" * 65)
    print("TEST 3: Technical Acronym Neutralization (Zero Initial-Token Bias)")
    print("=" * 65)

    handler = MultilingualHandler()

    acronym_queries = [
        ("BIS IS 1786:2008 स्टील प्रमाणन कैसे प्राप्त करें?", "hi", "Hindi"),
        ("BIS IS 16102 LED బల్బ్ సర్టిఫికేషన్ ఎలా పొందాలి?", "te", "Telugu"),
        ("BIS QCO CRS சான்றிதழ் பெறுவது எப்படி?", "ta", "Tamil"),
        ("BIS IS 1070 ল্যাবরেটরি নির্দেশিকা কি?", "bn", "Bengali"),
        ("ISO 9001 এবং BIS স্কিম নির্দেশিকা", "bn", "Bengali"),
        ("BIS FMCS Scheme-I ಸಿಮೆಂಟ್ ಪರವಾನಗಿ ಪ್ರಕ್ರಿಯೆ", "kn", "Kannada"),
    ]

    for query, exp_code, exp_name in acronym_queries:
        res = handler.detect_language(query)
        print(f"Acronym-heavy query: '{query}' -> Detected: {res['lang_name']} ({res['lang_code']})")
        assert res["lang_code"] == exp_code, f"Expected {exp_code}, got {res['lang_code']}"

    print("\n  [PASS] Technical Latin tokens neutralized: 100% accurate native script detection.")


def test_code_mixed_detection():
    print("\n" + "=" * 65)
    print("TEST 4: Code-Mixed Detection (Hinglish, Tenglish, Tanglish, Kanglish, Manglish)")
    print("=" * 65)

    handler = MultilingualHandler()

    code_mixed_queries = [
        ("Hinglish", "TMT bar ke liye kaunsa BIS standard applicable hai", "hinglish", "Hinglish"),
        ("Hinglish", "Scheme-I me apply kaise kare kitna fee lagiga", "hinglish", "Hinglish"),
        ("Tenglish", "TMT bar kosam standard enti ela apply cheyali", "tenglish", "Telugu"),
        ("Tenglish", "BIS certification fee entha undi", "tenglish", "Telugu"),
        ("Tanglish", "LED bulb kaaga standard ennadhu eppadi apply panradhu", "tanglish", "Tamil"),
        ("Tanglish", "BIS certification evvalavu fee aagum", "tanglish", "Tamil"),
        ("Kanglish", "Cement ge yavudhu standard beku hege apply madodu", "kanglish", "Kannada"),
        ("Manglish", "LED bulb certification engane apply cheyyanam ethra fee", "manglish", "Malayalam"),
    ]

    for label, query, exp_code, exp_name in code_mixed_queries:
        res = handler.detect_language(query)
        print(f"[{label:10}] Query: '{query}' -> Detected: {res['lang_name']} ({res['lang_code']})")
        assert res["lang_code"] == exp_code, f"Expected code {exp_code}, got {res['lang_code']}"
        assert res["is_code_mixed"] is True, "Must be flagged as code-mixed"

    print("\n  [PASS] Code-mixed Hinglish, Tenglish, Tanglish, Kanglish, and Manglish successfully classified.")


def test_english_collision_prevention():
    print("\n" + "=" * 65)
    print("TEST 5: Prevention of False-Positive Code-Mixed Detection on Standard English")
    print("=" * 65)

    handler = MultilingualHandler()

    english_queries = [
        "What is the fee for steel testing standard?",
        "Tell me about the application process and cost for cement certification",
        "Is there a scheme for packaged drinking water?",
        "Where can I file a consumer complaint for fake gold jewelry?",
        "How much is the license renewal fee for small scale industry?",
        "What is the standard for two wheeler helmets?",
    ]

    for query in english_queries:
        res = handler.detect_language(query)
        print(f"English Query: '{query}' -> Detected: {res['lang_name']} ({res['lang_code']})")
        assert res["lang_code"] == "en", f"Expected 'en', got {res['lang_code']}"
        assert res["is_code_mixed"] is False, "Standard English query should not be marked code-mixed"

    print("\n  [PASS] Zero false positives: standard English queries with collision words correctly stay English.")


def test_query_normalization_for_retrieval():
    print("\n" + "=" * 65)
    print("TEST 6: Query Normalization for BM25 / Hybrid Retrieval")
    print("=" * 65)

    handler = MultilingualHandler()

    # Code-mixed normalization
    norm_hi = handler.normalize_hinglish_to_english("tmt bar ke liye kaunsa bis standard applicable hai kaise apply kare")
    print(f"Hinglish Normalized: '{norm_hi}'")
    assert "for tmt bar steel reinforcement" in norm_hi.lower()
    assert "what bis standard applies" in norm_hi.lower()
    assert "how to apply" in norm_hi.lower()

    norm_te = handler.normalize_hinglish_to_english("tmt bar kosam standard enti ela apply cheyali fee entha")
    print(f"Tenglish Normalized: '{norm_te}'")
    assert "for tmt bar steel reinforcement" in norm_te.lower()
    assert "what bis standard applies" in norm_te.lower()
    assert "how to apply" in norm_te.lower()

    # Native script keyword expansion
    expanded_hi = handler.normalize_native_to_english_keywords("सीमेंट और स्टील के लिए मानक और शिकायत")
    print(f"Hindi Expanded: '{expanded_hi}'")
    assert "cement Ordinary Portland Cement" in expanded_hi
    assert "steel reinforcement TMT bar" in expanded_hi
    assert "complaint grievance" in expanded_hi

    expanded_te = handler.normalize_native_to_english_keywords("సిమెంట్ మరియు ఉక్కు ప్రమాణాలు మరియు ఫిర్యాదు")
    print(f"Telugu Expanded: '{expanded_te}'")
    assert "cement Ordinary Portland Cement" in expanded_te
    assert "steel reinforcement TMT bar" in expanded_te
    assert "complaint grievance" in expanded_te

    expanded_kn = handler.normalize_native_to_english_keywords("ಸಿಮೆಂಟ್ ಮತ್ತು ಉಕ್ಕು ಮಾನದಂಡ ಮತ್ತು ದೂರು")
    print(f"Kannada Expanded: '{expanded_kn}'")
    assert "cement Ordinary Portland Cement" in expanded_kn
    assert "steel reinforcement TMT bar" in expanded_kn
    assert "complaint grievance" in expanded_kn

    print("\n  [PASS] Code-mixed and native query normalization maps cleanly to regulatory search terms.")


def test_protected_token_preservation():
    print("\n" + "=" * 65)
    print("TEST 7: Protected Technical Token Preservation")
    print("=" * 65)

    engine = TranslationEngine()
    sample_text = (
        "As per IS 1786:2008, Clause 4.2.1(a), the certification fee is ₹50,000 (or Rs. 50,000) "
        "with 15% MSME concession under [Per BIS Guidelines (Clause 4.2)] (https://www.bis.gov.in/fee-structure)."
    )

    masked, tokens = engine.mask_protected_tokens(sample_text)
    print(f"Masked Text: {masked}")
    print(f"Protected Tokens: {tokens}")

    assert len(tokens) >= 4, f"Expected at least 4 protected tokens, found {len(tokens)}"
    assert any("IS 1786:2008" in t for t in tokens)
    assert any("50,000" in t for t in tokens)
    assert any("https://www.bis.gov.in" in t for t in tokens)

    unmasked = engine.unmask_protected_tokens(masked, tokens)
    print(f"Unmasked Text: {unmasked}")
    assert "IS 1786:2008" in unmasked
    assert "Clause 4.2.1(a)" in unmasked
    assert "50,000" in unmasked
    assert "https://www.bis.gov.in/fee-structure" in unmasked

    print("\n  [PASS] Technical tokens, citations, and currency symbols preserved with 100% integrity.")


def run_all_tests():
    print("=" * 65)
    print("RUNNING FEATURE 13 VERIFICATION SUITE: MULTILINGUAL DETECTION ENGINE")
    print("=" * 65)

    test_indic_script_dominance()
    test_short_indic_queries()
    test_acronym_neutralization()
    test_code_mixed_detection()
    test_english_collision_prevention()
    test_query_normalization_for_retrieval()
    test_protected_token_preservation()

    print("\n" + "=" * 65)
    print("ALL 7 MULTILINGUAL DETECTION & NORMALIZATION TESTS PASSED 100%!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
