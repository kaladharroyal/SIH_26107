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
    directly to the official BIS System of Record.
    """

    def handle_complaint(self, query: str) -> Dict[str, Any]:
        q_clean = query.strip() if query else ""
        q_lower = q_clean.lower()
        log.info(f"Processing Consumer Complaint Query: '{q_clean}'")

        is_hallmarking = any(k in q_lower for k in ["gold", "jewel", "hallmark", "silver", "purity", "carat", "karat", "huid"])

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
            f"\n🔗 [Download Official BIS CARE App]({BIS_CARE_APP_URL})\n"
            f"🔗 [BIS Online Complaint Portal]({OFFICIAL_COMPLAINT_URL})\n"
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
    print(handler.handle_complaint("I bought a gold item and its purity is lower than promised")["formatted_text"])
