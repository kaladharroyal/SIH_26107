"""
Phase 4, Step 13: Certification Scheme Walkthrough Guide (scheme_walkthrough.py)
Provides pre-verified, deterministic step-by-step application walkthroughs with exact fee schedules
for Scheme-I (ISI), Scheme-II (CRS), FMCS, Scheme-X, and the Hallmarking Scheme.
"""

import logging
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("scheme_walkthrough")

SCHEME_WALKTHROUGHS: Dict[str, Dict[str, Any]] = {
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
            "1. **Online Application**: Register on the `manakonline.in` portal and submit Form-V along with manufacturing process details.",
            "2. **Document Upload**: Upload Factory Registration Certificate, Machinery List, Testing Equipment calibration records, and Quality Manual.",
            "3. **Application Fee Payment**: Pay the non-refundable application fee of **₹1,000** online.",
            "4. **Factory Inspection**: BIS Inspecting Officer conducts on-site factory verification (Fee: **₹7,000/man-day**), inspects testing facilities, and draws sealed samples.",
            "5. **Sample Testing**: Factory samples are tested at a BIS Recognized Laboratory for full standard compliance.",
            "6. **Grant of Licence**: Upon successful verification, BIS issues the Performance Licence granting permission to apply the ISI Mark.",
        ],
        "official_url": "https://www.bis.gov.in/product-certification/product-certification-overview/?lang=en",
    },
    "scheme_ii": {
        "title": "Compulsory Registration Scheme (Scheme-II / CRS)",
        "target_audience": "Electronics & IT Goods Manufacturers (Domestic & International)",
        "fee_schedule": {
            "application_fee": "₹1,000 per product model family",
            "inspection_charge": "Nil (No pre-license factory inspection required)",
            "test_report_validity": "90 Days from BIS Recognized LIMS Lab",
            "license_duration": "2 Years (Renewable)",
        },
        "steps": [
            "1. **Sample Testing**: Send product samples to a BIS Recognized Laboratory in India to test compliance against applicable IS standard.",
            "2. **Online Registration**: Register on the `crsbis.in` portal after obtaining the valid 90-day lab test report.",
            "3. **Application Submission**: Submit Self-Declaration of Conformity along with technical test reports and manufacturer declarations.",
            "4. **Registration Grant**: BIS verifies documents online and issues the Registration Number (R-Number) without physical factory inspection.",
        ],
        "official_url": "https://www.crsbis.in/BIS/",
    },
    "fmcs": {
        "title": "Foreign Manufacturers Certification Scheme (FMCS)",
        "target_audience": "Manufacturers located outside India exporting goods to India",
        "fee_schedule": {
            "application_fee": "USD $1,000",
            "inspection_charge": "Actual international travel, lodging, and per-diem costs for BIS Inspecting Officers",
            "performance_bank_guarantee": "USD $10,000",
            "license_duration": "1 Year (Renewable)",
        },
        "steps": [
            "1. **AIR Appointment**: Appoint an Authorized Indian Representative (AIR) legally resident in India.",
            "2. **Application Submission**: Submit physical and online application with factory layout and quality manuals.",
            "3. **Overseas Factory Inspection**: BIS officers travel to the overseas facility to inspect manufacturing and draw samples.",
            "4. **Sample Testing & Grant**: Samples are tested in India. Upon passing and submitting the $10,000 PBG, the ISI mark licence is granted.",
        ],
        "official_url": "https://www.bis.gov.in/fmcs/fmcs-overview/?lang=en",
    },
    "scheme_x": {
        "title": "Conformity Assessment for Capital Goods & Machinery (Scheme-X)",
        "target_audience": "Manufacturers of Custom-Built Machinery, Heavy Industrial Equipment & Sub-Assemblies",
        "fee_schedule": {
            "application_fee": "₹1,000",
            "inspection_charge": "₹7,000 per man-day for design & audit verification",
            "testing_basis": "Design calculation appraisal & in-situ prototype testing",
            "license_duration": "2 Years (Renewable)",
        },
        "steps": [
            "1. **Design Documentation**: Submit detailed engineering drawings, stress calculations, and component bill of materials.",
            "2. **Technical Audit**: BIS technical experts audit the design methodology and manufacturing quality system.",
            "3. **Factory & Field Testing**: On-site prototype testing and performance validation under operating conditions.",
            "4. **Certificate of Conformity**: BIS issues Certificate of Conformity allowing equipment deployment.",
        ],
        "official_url": "https://www.bis.gov.in/product-certification/scheme-x-overview/?lang=en",
    },
    "hallmarking": {
        "title": "Gold & Silver Jewellery Hallmarking Scheme",
        "target_audience": "Jewellers, Gold Refineries, and Assaying & Hallmarking Centres (AHC)",
        "fee_schedule": {
            "jeweller_registration": "Free for Micro Enterprises; Tiered for Small/Medium/Large Jewellers",
            "hallmarking_charge": "₹45 per gold article / ₹35 per silver article (+ GST)",
            "huid_system": "Mandatory 6-digit alphanumeric Hallmark Unique Identification (HUID)",
            "registration_validity": "Lifetime / 5-Year Cycle",
        },
        "steps": [
            "1. **Jeweller Online Registration**: Jewellers register online on the `manakonline.in` portal under the BIS Hallmarking Scheme.",
            "2. **Submit to AHC**: Jeweller submits gold jewellery lots to a BIS Recognized Assaying & Hallmarking Centre (AHC).",
            "3. **Assaying & Laser Marking**: AHC performs fire assay testing for karat purity (e.g., 22K/916, 18K/750, 14K/585) and laser-marks the BIS logo, purity grade, and unique 6-digit HUID.",
            "4. **Verification on BIS CARE**: Consumer can scan and verify jewellery authenticity and jeweller details instantly via the **BIS CARE App**.",
        ],
        "official_url": "https://www.bis.gov.in/hallmarking-overview/?lang=en",
    },
}


class SchemeWalkthroughGuide:
    """
    Provides structured step-by-step guidance for official BIS certification schemes.
    """

    def get_walkthrough(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower() if query else ""
        log.info(f"Executing Scheme Walkthrough Guide for query: '{query}'")

        if "crs" in q_lower or "scheme-ii" in q_lower or "scheme 2" in q_lower or "electronic" in q_lower:
            scheme_key = "scheme_ii"
        elif "fmcs" in q_lower or "foreign" in q_lower or "overseas" in q_lower or "import" in q_lower:
            scheme_key = "fmcs"
        elif "scheme-x" in q_lower or "scheme x" in q_lower or "machinery" in q_lower or "capital goods" in q_lower:
            scheme_key = "scheme_x"
        elif "hallmark" in q_lower or "gold" in q_lower or "jewel" in q_lower or "silver" in q_lower or "huid" in q_lower:
            scheme_key = "hallmarking"
        else:
            # Default to Scheme-I (ISI Mark)
            scheme_key = "scheme_i"

        data = SCHEME_WALKTHROUGHS[scheme_key]

        # Format Markdown Output
        formatted = (
            f"### 📋 Official Step-by-Step Walkthrough: {data['title']}\n\n"
            f"**Target Audience**: {data['target_audience']}\n\n"
            f"#### 💰 Official Fee Schedule & Timelines:\n"
        )
        for fee_key, fee_val in data["fee_schedule"].items():
            readable_key = fee_key.replace("_", " ").title()
            formatted += f"- **{readable_key}**: {fee_val}\n"

        formatted += "\n#### 🚀 Step-by-Step Certification Procedure:\n"
        for step in data["steps"]:
            formatted += f"{step}\n"

        formatted += f"\n🔗 [Official BIS Portal Guide]({data['official_url']})\n"

        return {
            "intent": "certification_process",
            "flow": "scheme_walkthrough",
            "status": "success",
            "scheme_key": scheme_key,
            "title": data["title"],
            "fee_schedule": data["fee_schedule"],
            "steps": data["steps"],
            "formatted_text": formatted,
            "source": "official_scheme_walkthrough",
            "fallback_used": False,
        }


if __name__ == "__main__":
    guide = SchemeWalkthroughGuide()
    print(guide.get_walkthrough("how do I get BIS certification under Scheme I?")["formatted_text"])
