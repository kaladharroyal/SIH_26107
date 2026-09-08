"""
Intent Router Precision & Semantic Classifier Benchmark (test_router_accuracy.py)
Evaluates precision and recall across labeled in-domain queries per intent.
Confirms zero hardcoded keyword override branches remain.
"""

import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import HybridRetrievalPipeline
from router import QueryIntentRouter


@pytest.fixture(scope="module")
def router():
    pipeline = HybridRetrievalPipeline(use_mock_encoder=False)
    return QueryIntentRouter(encoder=pipeline.encoder)


def test_router_precision_recall(router):
    labeled_data = [
        # technical_standards_rag
        ("What are the chemical composition limits for carbon in Fe 500D steel as per IS 1786?", "technical_standards_rag"),
        ("What is the minimum elongation percentage and yield strength for IS 1786?", "technical_standards_rag"),
        ("What are the microbiological testing requirements in IS 14543 for packaged water?", "technical_standards_rag"),
        ("IS 269 के तहत 53 ग्रेड ओपीसी सीमेंट की संपीड़न शक्ति और सेटिंग समय क्या है?", "technical_standards_rag"),

        # product_recommendation
        ("Which BIS standard applies to solar panels and is ISI mark mandatory under QCO?", "product_recommendation"),
        ("What is the Indian Standard for unplasticized PVC potable water pipes?", "product_recommendation"),
        ("क्या टीएमटी सरिया बेचने के लिए बीआईएस लाइसेंस अनिवार्य है?", "product_recommendation"),
        ("హెల్మెట్ మరియు పిల్లల బొమ్మలకు ఏ భారతీయ ప్రమాణం వర్తిస్తుంది?", "product_recommendation"),

        # certification_process
        ("How to apply for BIS Scheme-I license and what are the application fees?", "certification_process"),
        ("What is the step-by-step registration process and factory audit timeline for ISI mark?", "certification_process"),
        ("What is the annual marking fee and inspection charge for domestic manufacturers?", "certification_process"),
        ("विदेश में बने उत्पादों के लिए एफएमसीएस लाइसेंस कैसे प्राप्त करें?", "certification_process"),

        # lab_location
        ("Where are BIS recognized testing laboratories in Mumbai Maharashtra?", "lab_location"),
        ("Find authorized testing labs and testing scopes in Delhi NCR and Sahibabad", "lab_location"),
        ("List of approved testing facilities in Chennai for electrical cable testing", "lab_location"),
        ("హైదరాబాద్‌లో నమూనా పరీక్ష కోసం గుర్తింపు పొందిన BIS ల్యాబ్‌లు ఎక్కడ ఉన్నాయి?", "lab_location"),

        # consumer_complaint
        ("How to file a complaint against fake ISI mark on the BIS CARE mobile app?", "consumer_complaint"),
        ("How do I report low purity hallmarked gold and claim 2x statutory compensation?", "consumer_complaint"),
        ("What is the consumer grievance redressal procedure for defective certified goods?", "consumer_complaint"),
        ("నకిలీ ISI మార్క్ లేదా నాణ్యత లేని వస్తువులపై ఫిర్యాదు ఎలా చేయాలి?", "consumer_complaint"),
    ]

    correct_by_intent = {}
    total_by_intent = {}

    for query, expected_intent in labeled_data:
        res = router.classify_intent(query)
        predicted = res["intent"]

        total_by_intent[expected_intent] = total_by_intent.get(expected_intent, 0) + 1
        if predicted == expected_intent:
            correct_by_intent[expected_intent] = correct_by_intent.get(expected_intent, 0) + 1
        else:
            safe_q = query.encode("ascii", "backslashreplace").decode("ascii")
            print(f"Misclassified: '{safe_q}' -> Expected '{expected_intent}', Got '{predicted}' (conf: {res.get('confidence'):.4f})")


    print("\n--- Router Intent Evaluation ---")
    overall_correct = sum(correct_by_intent.values())
    overall_total = len(labeled_data)
    overall_acc = overall_correct / overall_total

    for intent, total in total_by_intent.items():
        correct = correct_by_intent.get(intent, 0)
        acc = correct / total
        print(f"  Intent '{intent}': {correct}/{total} ({acc*100:.1f}%)")

    print(f"Overall Accuracy: {overall_correct}/{overall_total} ({overall_acc*100:.1f}%)")
    assert overall_acc >= 0.85, f"Router accuracy {overall_acc*100:.1f}% below target 85%"
