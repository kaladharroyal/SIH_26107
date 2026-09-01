"""
Phase 4, Step 15: Consumer Query & Complaint Handling Router (consumer_complaint.py)
Directs consumer quality grievances, fake ISI mark reports, and hallmarking purity disputes
to official BIS CARE systems and enforces statutory consumer compensation rights.
"""

import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("consumer_complaint")

OFFICIAL_COMPLAINT_URL = "https://www.bis.gov.in/consumer-overview/online-complaint-registration/?lang=en"
BIS_CARE_APP_URL = "https://play.google.com/store/apps/details?id=com.bis.bis_care"
BIS_HELPLINE = "1800-11-4000"
BIS_COMPLAINT_EMAIL = "complaints@bis.gov.in"


class ConsumerComplaintHandler:
    """
    Handles consumer grievances, explains legal redressal rights, and routes complaints
    directly to the official BIS System of Record in the user's requested language.
    """

    def handle_complaint(self, query: str, language: str = "English") -> Dict[str, Any]:
        q_clean = query.strip() if query else ""
        q_lower = q_clean.lower()
        log.info(f"Processing Consumer Complaint Query: '{q_clean}' (Lang: '{language}')")

        is_hallmarking = any(k in q_lower for k in ["gold", "jewel", "hallmark", "silver", "purity", "carat", "karat", "huid", "सोना", "स्वर्ण", "आभूषण", "బంగారం"])

        if is_hallmarking:
            category = "hallmarking_complaint"
            compensation_text = (
                "Under the BIS (Hallmarking) Regulations, 2018:\n"
                "- If hallmarked gold/silver jewellery fails the prescribed purity test when verified at a BIS Recognized "
                "Assaying and Hallmarking Centre (AHC), the consumer is legally entitled to **compensation equal to TWO TIMES (2x) "
                "the value of the purity shortfall**, along with full reimbursement of testing charges incurred."
            )
        else:
            category = "isi_product_complaint"
            compensation_text = (
                "Under the BIS Act, 2016 and Consumer Protection Act, 2019:\n"
                "- Selling non-certified goods under mandatory QCO/CRS or using a counterfeit/fake ISI mark is a punishable offense.\n"
                "- The consumer is entitled to product replacement, full refund with interest, and the manufacturer/seller "
                "faces immediate product recall, license cancellation, and statutory penalties."
            )

        how_to_file = [
            "1. **BIS CARE Mobile App**: Open the BIS CARE App, select 'Complaints', and file an instant geo-tagged grievance.",
            f"2. **Online Complaint Portal**: Register online on the [Official BIS Complaint Registration Portal]({OFFICIAL_COMPLAINT_URL}).",
            f"3. **Toll-Free National Helpline**: Call `{BIS_HELPLINE}` (Mon-Fri, 9:00 AM - 5:30 PM).",
            f"4. **Email Support**: Send formal grievance with purchase receipt to `{BIS_COMPLAINT_EMAIL}`.",
        ]

        lang_lower = (language or "english").lower()

        if "hindi" in lang_lower or lang_lower == "hi":
            if is_hallmarking:
                comp_body = (
                    "बीआईएस (हॉलमार्किंग) विनियम, 2018 के तहत:\n"
                    "- यदि हॉलमार्क वाले सोने/चांदी के आभूषण बीआईएस मान्यता प्राप्त एसेइंग और हॉलमार्किंग केंद्र (AHC) पर जांच के दौरान निर्धारित शुद्धता में विफल होते हैं, तो उपभोक्ता कानूनी रूप से **शुद्धता की कमी के मूल्य के दो गुना (2x) के बराबर मुआवजे** और जांच शुल्क की पूर्ण प्रतिपूर्ति का हकदार है।"
                )
            else:
                comp_body = (
                    "बीआईएस अधिनियम, 2016 और उपभोक्ता संरक्षण अधिनियम, 2019 के तहत:\n"
                    "- अनिवार्य QCO/CRS के तहत गैर-प्रमाणित सामान बेचना या नकली ISI मार्क का उपयोग करना दंडनीय अपराध है।\n"
                    "- उपभोक्ता उत्पाद प्रतिस्थापन, ब्याज सहित पूर्ण धनवापसी का हकदार है, और निर्माता/विक्रेता को उत्पाद वापसी, लाइसेंस रद्दीकरण और वैधानिक दंड का सामना करना पड़ता है।"
                )

            formatted = (
                "### 🛡️ आधिकारिक बीआईएस उपभोक्ता संरक्षण एवं शिकायत मार्गदर्शन\n\n"
                "यदि आपको घटिया गुणवत्ता वाले उत्पाद, नकली ISI मार्क, या कम शुद्धता वाले आभूषण प्राप्त हुए हैं, "
                "तो आप बीआईएस के आधिकारिक पोर्टल पर शिकायत दर्ज कर सकते हैं।\n\n"
                "#### ⚖️ वैधानिक उपभोक्ता मुआवजा और कानूनी अधिकार:\n"
                f"{comp_body}\n\n"
                "#### 📲 आधिकारिक शिकायत कैसे दर्ज करें:\n"
                "1. **BIS CARE मोबाइल ऐप**: BIS CARE ऐप खोलें, 'शिकायत' चुनें और त्वरित शिकायत दर्ज करें।\n"
                f"2. **ऑनलाइन पोर्टल**: [आधिकारिक बीआईएस शिकायत पंजीकरण पोर्टल]({OFFICIAL_COMPLAINT_URL}) पर ऑनलाइन पंजीकरण करें।\n"
                f"3. **टोल-फ्री राष्ट्रीय हेल्पलाइन**: कॉल करें `{BIS_HELPLINE}` (सोम-शुक्र, सुबह 9:00 - शाम 5:30)।\n"
                f"4. **ईमेल सहायता**: खरीद रसीद के साथ औपचारिक शिकायत `{BIS_COMPLAINT_EMAIL}` पर भेजें।\n\n"
                "### स्रोत\n\n"
                f"📄 **BIS CARE उपभोक्ता शिकायत पोर्टल**\n"
                f"[आधिकारिक BIS CARE पोर्टल खोलें ↗]({OFFICIAL_COMPLAINT_URL})\n"
            )
        elif "telugu" in lang_lower or lang_lower == "te":
            formatted = (
                "### 🛡️ అధికారిక BIS వినియోగదారు రక్షణ మరియు ఫిర్యాదు మార్గదర్శకత్వం\n\n"
                "మీరు నాణ్యత లేని ఉత్పత్తులు, నకిలీ ISI మార్కులు లేదా తక్కువ స్వచ్ఛత గల బంగారు ఆభరణాలను స్వీకరించినట్లయితే, "
                "మీరు నేరుగా BIS రికార్డ్ సిస్టమ్‌లో అధికారిక ఫిర్యాదును నమోదు చేయవచ్చు.\n\n"
                "#### ⚖️ చట్టబద్ధమైన వినియోగదారు పరిహారం మరియు హక్కులు:\n"
                f"{compensation_text}\n\n"
                "#### 📲 అధికారిక ఫిర్యాదును ఎలా నమోదు చేయాలి:\n"
                "1. **BIS CARE మొబైల్ యాప్**: BIS CARE యాప్‌ను తెరిచి, 'ఫిర్యాదులు' ఎంచుకోండి మరియు ఫిర్యాదు చేయండి.\n"
                f"2. **ఆన్‌లైన్ ఫిర్యాదు పోర్టల్**: [అధికారిక BIS ఫిర్యాదు నమోదు పోర్టల్]({OFFICIAL_COMPLAINT_URL}) లో ఆన్‌లైన్‌లో నమోదు చేసుకోండి.\n"
                f"3. **టోల్-ఫ్రీ జాతీయ హెల్ప్‌లైన్**: కాల్ చేయండి `{BIS_HELPLINE}` (సోమ-శుక్ర, 9:00 AM - 5:30 PM).\n"
                f"4. **ఇమెయిల్ మద్దతు**: కొనుగోలు రసీదుతో పాటు `{BIS_COMPLAINT_EMAIL}` కు ఫిర్యాదును పంపండి.\n\n"
                "### మూలం\n\n"
                f"📄 **BIS CARE వినియోగదారు పోర్టల్**\n"
                f"[అధికారిక BIS పోర్టల్‌ను తెరవండి ↗]({OFFICIAL_COMPLAINT_URL})\n"
            )
        else:
            formatted = (
                "### 🛡️ Official BIS Consumer Protection & Grievance Guidance\n\n"
                f"If you have received substandard products, counterfeit ISI marks, or low-purity gold jewellery, "
                f"you can lodge an official grievance directly into the BIS System of Record.\n\n"
                f"#### ⚖️ Statutory Consumer Compensation & Legal Rights:\n"
                f"{compensation_text}\n\n"
                f"#### 📲 How to File an Official Complaint:\n"
            )
            for step in how_to_file:
                formatted += f"{step}\n"

            formatted += (
                f"\n### Source\n\n"
                f"📄 **BIS CARE Consumer Grievance Portal**\n"
                f"[Open official BIS CARE portal ↗]({OFFICIAL_COMPLAINT_URL})\n"
            )

        return {
            "intent": "consumer_complaint",
            "flow": "consumer_complaint",
            "status": "success",
            "category": category,
            "is_hallmarking": is_hallmarking,
            "compensation_rights": compensation_text,
            "how_to_file": how_to_file,
            "formatted_text": formatted,
            "source": "official_bis_care_portal",
            "fallback_used": False,
        }


if __name__ == "__main__":
    handler = ConsumerComplaintHandler()
    print(handler.handle_complaint("BIS प्रमाणित उत्पाद के खिलाफ शिकायत कैसे दर्ज कर सकते हैं?", language="Hindi")["formatted_text"])
