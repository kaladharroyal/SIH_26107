"""
Phase 4, Step 13: Certification Scheme Walkthrough Guide (scheme_walkthrough.py)
Provides pre-verified step-by-step application walkthroughs with exact fee schedules and inspection charges.
"""

import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scheme_walkthrough")

SCHEME_WALKTHROUGHS = {
    "scheme_i": {
        "title": "Product Certification Scheme (Scheme-I / ISI Mark)",
        "target_audience": "Domestic Manufacturers in India",
        "fee_schedule": {
            "application_fee": "₹1,000",
            "inspection_charge": "₹7,000 per man-day + actual travel expenses",
            "test_report_validity": "90 Days from recognized lab",
            "license_duration": "2 Years (Renewable)",
        },
        "steps": [
            "1. **Online Application**: Register on `manakonline.in` portal and submit Form-V along with product details and manufacturing process flow.",
            "2. **Document Submission**: Upload Factory Registration Certificate, Machinery List, Testing Equipment calibration certificates, and Raw Material test certificates.",
            "3. **Application Fee Payment**: Pay the non-refundable application fee of **₹1,000** online.",
            "4. **Factory Inspection**: BIS Inspecting Officer conducts on-site factory verification (Fee: **₹7,000/man-day**), checks quality control, and draws counter-sealed samples.",
            "5. **Sample Testing**: Factory sample is dispatched to a BIS Recognized Laboratory for complete verification against applicable Indian Standard.",
            "6. **Grant of Licence**: Upon successful test report verification, BIS issues the Performance Licence granting permission to use the ISI Mark.",
        ],
        "official_url": "https://www.bis.gov.in/product-certification/product-certification-overview/?lang=en",
    },
    "scheme_ii": {
        "title": "Compulsory Registration Scheme (Scheme-II / CRS)",
        "target_audience": "Electronics & IT Goods Manufacturers (Domestic & International)",
        "fee_schedule": {
            "application_fee": "₹1,000 per product model family",
            "inspection_charge": "Nil (No pre-license factory inspection required)",
            "test_report_validity": "90 Days from BIS Recognized Lab",
            "license_duration": "2 Years (Renewable)",
        },
        "steps": [
            "1. **Sample Testing**: Send product sample to a BIS Recognized LIMS Laboratory in India to test compliance against relevant IS standard.",
            "2. **Online Registration**: Register on `crsbis.in` portal after obtaining the valid test report.",
            "3. **Application Submission**: Submit self-declaration of conformity along with the 90-day valid lab test report.",
            "4. **Registration Grant**: BIS verifies documents online and grants Registration Number within 15 working days without physical factory inspection.",
        ],
        "official_url": "https://www.crsbis.in/BIS/",
    },
    "fmcs": {
        "title": "Foreign Manufacturers Certification Scheme (FMCS)",
        "target_audience": "Manufacturers located outside India exporting goods to India",
        "fee_schedule": {
            "application_fee": "USD $1,000",
            "inspection_charge": "Actual travel, lodging, and per-diem costs for BIS Officers",
            "performance_bank_guarantee": "USD $10,000",
            "license_duration": "1 Year (Renewable)",
        },
        "steps": [
            "1. **AIR Appointment**: Appoint an Authorized Indian Representative (AIR) residing in India.",
            "2. **Application Submission**: Submit physical & online application with factory layout and quality manuals.",
            "3. **Overseas Factory Inspection**: BIS officers travel abroad to inspect manufacturing facility and draw samples.",
            "4. **Sample Testing & Grant**: Samples tested in India. Upon passing and submitting PBG ($10,000), ISI mark license is granted.",
        ],
        "official_url": "https://www.bis.gov.in/fmcs/fmcs-overview/?lang=en",
    },
}


class SchemeWalkthroughGuide:
    def get_walkthrough(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()

        if "crs" in q_lower or "scheme-ii" in q_lower or "electronic" in q_lower:
            scheme_key = "scheme_ii"
        elif "fmcs" in q_lower or "foreign" in q_lower or "export" in q_lower:
            scheme_key = "fmcs"
        else:
            scheme_key = "scheme_i"

        data = SCHEME_WALKTHROUGHS[scheme_key]

        formatted = (
            f"### 📋 Step-by-Step Walkthrough: {data['title']}\n"
            f"**Target Audience**: {data['target_audience']}\n\n"
            f"#### 💰 Official Fee & Duration Schedule:\n"
            f"- **Application Fee**: {data['fee_schedule']['application_fee']}\n"
            f"- **Factory Inspection Charge**: {data['fee_schedule']['inspection_charge']}\n"
            f"- **Lab Test Report Validity**: {data['fee_schedule']['test_report_validity']}\n"
            f"- **Licence Duration**: {data['fee_schedule']['license_duration']}\n\n"
            f"#### 🚀 Step-by-Step Licensing Procedure:\n"
        )
        for step in data["steps"]:
            formatted += f"{step}\n"

        formatted += f"\n🔗 [Official Portal Guide]({data['official_url']})\n"

        return {
            "scheme_key": scheme_key,
            "title": data["title"],
            "formatted_text": formatted,
        }


if __name__ == "__main__":
    guide = SchemeWalkthroughGuide()
    print(guide.get_walkthrough("how to apply for ISI mark Scheme-I")["formatted_text"])
