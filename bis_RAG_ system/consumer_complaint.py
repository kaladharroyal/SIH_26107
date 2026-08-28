"""
Phase 4, Step 15: Consumer Query & Complaint Handling Router (consumer_complaint.py)
Directs consumer complaints to official BIS CARE systems and provides legal compensation rights.
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("consumer_complaint")

OFFICIAL_COMPLAINT_URL = "https://www.bis.gov.in/consumer-overview/online-complaint-registration/?lang=en"
BIS_CARE_APP_URL = "https://play.google.com/store/apps/details?id=com.bis.bis_care"


class ConsumerComplaintHandler:
    def handle_complaint(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        log.info(f"Processing Consumer Complaint Query: '{query}'")

        is_hallmarking = "gold" in q_lower or "jewel" in q_lower or "hallmark" in q_lower

        if is_hallmarking:
            compensation_info = (
                "#### ⚖️ Official Consumer Compensation Rights (Hallmarking Regulations):\n"
                "- If hallmarked gold jewellery fails the purity test at a BIS Recognized Assaying & Hallmarking Centre (AHC), "
                "the consumer is legally entitled to **compensation equal to 2 TIMES the value of the purity shortfall**, "
                "plus full reimbursement of testing charges.\n"
            )
        else:
            compensation_info = (
                "#### ⚖️ Consumer Redressal Rights:\n"
                "- Products bearing a fake ISI mark or failing safety standards are subject to immediate BIS enforcement, "
                "product recall, and legal prosecution under the BIS Act, 2016.\n"
            )

        formatted_response = (
            f"### 🛡️ Consumer Complaint & Quality Issue Guidance\n\n"
            f"If you have purchased a product with a fake ISI mark, invalid C-number, or defective hallmarked jewellery, "
            f"you can lodge an official complaint directly into the BIS System of Record.\n\n"
            f"{compensation_info}\n"
            f"#### 📲 How to File an Official Complaint:\n"
            f"1. **BIS CARE Mobile App**: Download the official **BIS CARE App** to verify ISI/CRS/Hallmarking numbers instantly and file complaints.\n"
            f"2. **Online Portal**: Register your complaint online via the [Official BIS Online Complaint Portal]({OFFICIAL_COMPLAINT_URL}).\n"
            f"3. **Toll-Free Helpline**: Call BIS Consumer Helpline at `1800-11-4000` or email `complaints@bis.gov.in`.\n\n"
            f"🔗 [Download BIS CARE Mobile App]({BIS_CARE_APP_URL})\n"
        )

        return {
            "query": query,
            "category": "consumer_protection",
            "is_hallmarking": is_hallmarking,
            "formatted_text": formatted_response,
        }


if __name__ == "__main__":
    handler = ConsumerComplaintHandler()
    print(handler.handle_complaint("my gold hallmark ring is fake how to complain")["formatted_text"])
