"""
Phase 4, Step 14: Filterable Lab Locator Engine (lab_locator.py)
Filters LIMS recognized and empaneled testing laboratories by city/state, IS number, and scope.
Loads dynamically from Phase 1 generated labs_directory.json with zero hard-coded laboratories.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lab_locator")

BASE_DIR = Path(__file__).resolve().parent.parent


class LabLocator:
    def __init__(self, labs_path: Optional[Path] = None):
        self.labs_path = labs_path or (BASE_DIR / "labs_directory.json")
        self.status = "unavailable"
        self.status_reason = ""
        self.labs: List[Dict[str, Any]] = self._load_labs()

    def _load_labs(self) -> List[Dict[str, Any]]:
        """Loads verified laboratory directory from Phase 1 generated dataset."""
        if not self.labs_path.exists():
            self.status = "unavailable"
            self.status_reason = "Laboratory directory file not found on disk."
            log.warning(f"Labs directory not found at {self.labs_path}")
            return []

        try:
            with open(self.labs_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.status = data.get("status", "unavailable")
            self.status_reason = data.get("status_reason", "")
            records = data.get("records", [])
            log.info(f"Loaded {len(records)} lab records from {self.labs_path.name} (Status: {self.status})")
            return records
        except Exception as e:
            self.status = "unavailable"
            self.status_reason = str(e)
            log.error(f"Error loading labs directory: {e}")
            return []

    def search_labs(self, query: str, state: Optional[str] = None) -> Dict[str, Any]:
        q_lower = query.lower()
        log.info(f"Executing Lab Locator for query: '{query}' (state={state})")

        if not self.labs or self.status == "unavailable":
            return {
                "query": query,
                "status": "unavailable",
                "total_found": 0,
                "labs": [],
                "formatted_text": (
                    "### 🧪 BIS Testing Laboratory Search\n\n"
                    "Static offline laboratory directory is currently unavailable or requires live BIS LIMS authentication.\n\n"
                    "🔗 Please consult the official live directory at [BIS LIMS Portal](https://lims.bis.gov.in/)\n"
                ),
            }

        matches = []
        for lab in self.labs:
            lab_state = lab.get("location") or lab.get("state") or ""
            lab_name = lab.get("lab_name", "")
            scopes = lab.get("testing_scope", [])

            if state and state.lower() not in lab_state.lower():
                continue

            score = 0
            if lab_state and lab_state.lower() in q_lower:
                score += 10
            if lab_name and lab_name.lower() in q_lower:
                score += 15

            for std in scopes:
                if str(std).lower() in q_lower:
                    score += 20

            if score > 0 or not state:
                matches.append((score, lab))

        matches.sort(key=lambda x: x[0], reverse=True)
        top_matches = [m[1] for m in matches[:5]]

        formatted = f"### 🧪 BIS Testing Laboratory Search Results\n\n"
        for idx, lab in enumerate(top_matches, 1):
            stds = ", ".join(str(s) for s in lab.get("testing_scope", [])) or "General Testing"
            formatted += (
                f"**{idx}. {lab.get('lab_name')}**\n"
                f"- **Location**: {lab.get('location', 'India')}\n"
                f"- **Address**: {lab.get('address', 'BIS Testing Centre')}\n"
                f"- **Testing Scope**: {stds}\n\n"
            )

        formatted += "🔗 Search complete LIMS lab directory at [BIS LIMS Portal](https://lims.bis.gov.in/)\n"

        return {
            "query": query,
            "status": "success",
            "total_found": len(top_matches),
            "labs": top_matches,
            "formatted_text": formatted,
        }
