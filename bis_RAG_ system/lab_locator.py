"""
Phase 4, Step 14: Filterable Lab Locator Engine (lab_locator.py)
Filters LIMS recognized and empaneled testing laboratories by city/state, IS number, and scope.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lab_locator")

RAW_DIR = Path("./raw_data")

# Core fallback lab directory database
FALLBACK_LABS = [
    {
        "lab_name": "Central Laboratory, Bureau of Indian Standards",
        "city": "Sahibabad",
        "state": "Uttar Pradesh",
        "supported_standards": ["IS 302", "IS 16102", "IS 14286", "IS 1786", "IS 14543"],
        "lab_type": "Central BIS Lab",
        "address": "Plot No. 20/9, Site IV, Sahibabad Industrial Area, Ghaziabad, UP",
        "contact": "cl-bis@bis.gov.in",
    },
    {
        "lab_name": "Northern Regional Office Laboratory (NROL)",
        "city": "Mohali",
        "state": "Punjab",
        "supported_standards": ["IS 1786", "IS 456", "IS 269", "IS 9873"],
        "lab_type": "Regional BIS Lab",
        "address": "Plot No. 4A, Sector 27B, Madhya Marg, Chandigarh / Mohali",
        "contact": "nrol@bis.gov.in",
    },
    {
        "lab_name": "Western Regional Office Laboratory (WROL)",
        "city": "Mumbai",
        "state": "Maharashtra",
        "supported_standards": ["IS 16102", "IS 1417", "IS 302", "IS 14286"],
        "lab_type": "Regional BIS Lab",
        "address": "E-9, MIDC, Andheri East, Mumbai, Maharashtra 400093",
        "contact": "wrol@bis.gov.in",
    },
    {
        "lab_name": "Southern Regional Office Laboratory (SROL)",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "supported_standards": ["IS 1786", "IS 1417", "IS 14543"],
        "lab_type": "Regional BIS Lab",
        "address": "CIT Campus, IV Cross Road, Taramani, Chennai, Tamil Nadu 600113",
        "contact": "srol@bis.gov.in",
    },
]


class LabLocator:
    def __init__(self):
        self.labs = self._load_labs()

    def _load_labs(self) -> List[Dict[str, Any]]:
        labs = list(FALLBACK_LABS)
        # Scan raw_data for scraped LIMS JSON files
        for json_file in RAW_DIR.glob("*lims*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        labs.append({
                            "lab_name": item.get("lab_name") or item.get("name", "Recognized Testing Lab"),
                            "city": item.get("city", "India"),
                            "state": item.get("state", "India"),
                            "supported_standards": [item.get("is_number", "IS Standard")],
                            "lab_type": "LIMS Recognized Lab",
                            "address": item.get("address", "Official Address on LIMS Portal"),
                            "contact": item.get("email", "lims@bis.gov.in"),
                        })
            except Exception:
                pass
        log.info(f"Loaded {len(labs)} testing laboratories into locator index.")
        return labs

    def search_labs(self, query: str, state: Optional[str] = None) -> Dict[str, Any]:
        q_lower = query.lower()
        log.info(f"Executing Lab Locator for query: '{query}' (state={state})")

        matches = []
        for lab in self.labs:
            # Check state/city match
            if state and state.lower() not in lab["state"].lower() and state.lower() not in lab["city"].lower():
                continue

            score = 0
            if lab["city"].lower() in q_lower or lab["state"].lower() in q_lower:
                score += 10

            for std in lab["supported_standards"]:
                if std.lower() in q_lower:
                    score += 15

            if "lab" in q_lower or "test" in q_lower:
                score += 2

            if score > 0 or not state:
                matches.append((score, lab))

        matches.sort(key=lambda x: x[0], reverse=True)
        top_matches = [m[1] for m in matches[:5]]

        if not top_matches:
            top_matches = self.labs[:3]

        formatted = f"### 🧪 BIS Testing Laboratory Search Results\n\n"
        for idx, lab in enumerate(top_matches, 1):
            stds = ", ".join(lab["supported_standards"])
            formatted += (
                f"**{idx}. {lab['lab_name']}** ({lab['lab_type']})\n"
                f"- **Location**: {lab['city']}, {lab['state']}\n"
                f"- **Address**: {lab['address']}\n"
                f"- **Testing Scope**: {stds}\n"
                f"- **Contact Email**: `{lab['contact']}`\n\n"
            )

        formatted += "🔗 Search complete LIMS lab directory at [BIS LIMS Portal](https://lims.bis.gov.in/)\n"

        return {
            "query": query,
            "total_found": len(top_matches),
            "labs": top_matches,
            "formatted_text": formatted,
        }


if __name__ == "__main__":
    locator = LabLocator()
    print(locator.search_labs("testing labs in Delhi for steel IS 1786")["formatted_text"])
