"""
Phase 4, Step 14: Filterable Lab Locator Engine (lab_locator.py)
Filters BIS recognized and empaneled testing laboratories by city/state, IS number, and test scope.
Loads dynamically from labs_directory.json and falls back to searching verified lab_directory corpus chunks
with zero fabricated laboratories.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("lab_locator")

BASE_DIR = Path(__file__).resolve().parent.parent
OFFICIAL_LIMS_PORTAL = "https://lims.bis.gov.in/"


class LabLocator:
    """
    Filterable laboratory directory search with static file handling,
    state/scope filtering, and corpus search fallback.
    """

    def __init__(self, labs_path: Optional[Path] = None, retrieval_pipeline: Optional[Any] = None):
        self.labs_path = labs_path or (BASE_DIR / "labs_directory.json")
        self.status = "unavailable"
        self.status_reason = ""
        self.labs: List[Dict[str, Any]] = self._load_labs()
        self.retrieval = retrieval_pipeline

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
        """
        Searches and filters testing laboratories by state, name, or standard scope.
        If static dataset is empty/unavailable, falls back to corpus search or official LIMS portal.
        """
        q_clean = query.strip() if query else ""
        q_lower = q_clean.lower()
        state_filter = state.strip().lower() if state else None

        # Extract explicit state if present in query text (e.g. "labs in Delhi", "testing in Maharashtra")
        if not state_filter:
            for s_name in ["delhi", "maharashtra", "gujarat", "karnataka", "tamil nadu", "telangana", "uttar pradesh", "west bengal", "punjab", "haryana", "rajasthan"]:
                if re.search(r"\b" + re.escape(s_name) + r"\b", q_lower):
                    state_filter = s_name
                    break

        log.info(f"Executing Lab Locator -> Query: '{q_clean}' | State Filter: {state_filter}")

        # 1. Search in Static Dataset if records are available
        if self.labs and self.status == "available":
            matches = []
            for lab in self.labs:
                lab_state = str(lab.get("location") or lab.get("state") or "").lower()
                lab_name = str(lab.get("lab_name") or "").lower()
                scopes = [str(s).lower() for s in lab.get("testing_scope", [])]

                if state_filter and state_filter not in lab_state:
                    continue

                score = 0
                if state_filter and state_filter in lab_state:
                    score += 15
                if lab_name and any(w in lab_name for w in q_lower.split() if len(w) > 3):
                    score += 10
                for std in scopes:
                    if std in q_lower:
                        score += 25

                if score > 0 or not state_filter:
                    matches.append((score, lab))

            if matches:
                matches.sort(key=lambda x: x[0], reverse=True)
                top_matches = [m[1] for m in matches[:5]]

                formatted = "### 🧪 BIS Recognized Testing Laboratories\n\n"
                for idx, lab in enumerate(top_matches, 1):
                    stds = ", ".join(str(s) for s in lab.get("testing_scope", [])) or "General Product Testing"
                    formatted += (
                        f"**{idx}. {lab.get('lab_name')}**\n"
                        f"- **Location**: {lab.get('location', 'India')}\n"
                        f"- **Address**: {lab.get('address', 'BIS Testing Centre')}\n"
                        f"- **Scope of Testing**: {stds}\n\n"
                    )
                formatted += f"🔗 Consult the complete directory on the [Official BIS LIMS Portal]({OFFICIAL_LIMS_PORTAL})\n"

                return {
                    "intent": "lab_location",
                    "flow": "lab_locator",
                    "status": "success",
                    "total_found": len(top_matches),
                    "labs": top_matches,
                    "formatted_text": formatted,
                    "source": "labs_directory_json",
                    "fallback_used": False,
                }

        # 2. Fallback: Search in Phase 1/2 verified corpus (category == 'lab_directory')
        if self.retrieval is not None:
            log.info("Static lab directory is empty/unavailable. Querying corpus 'lab_directory' chunks...")
            try:
                corpus_hits = self.retrieval.retrieve(q_clean or "laboratory testing facility", top_n=3, category="lab_directory")
                if corpus_hits:
                    formatted_corpus = "### 🧪 BIS Testing Laboratory & Facility Guidance (Corpus Search)\n\n"
                    for idx, hit in enumerate(corpus_hits, 1):
                        doc = hit.get("doc", hit)
                        title = doc.get("clause_title") or doc.get("title") or "Testing Guidelines"
                        text_snippet = doc.get("text", "")[:250].strip()
                        formatted_corpus += f"**{idx}. {title}**\n{text_snippet}...\n\n"

                    formatted_corpus += (
                        f"🔗 For real-time, state-wise accredited lab listings and live testing scopes:\n"
                        f"👉 Search the [Official BIS LIMS Portal]({OFFICIAL_LIMS_PORTAL})\n"
                    )

                    return {
                        "intent": "lab_location",
                        "flow": "lab_locator",
                        "status": "success",
                        "total_found": len(corpus_hits),
                        "labs": [h.get("doc", h) for h in corpus_hits],
                        "formatted_text": formatted_corpus,
                        "source": "corpus_lab_directory",
                        "fallback_used": True,
                    }
            except Exception as e:
                log.warning(f"Lab corpus search fallback failed: {e}")

        # 3. Default structured redirection when no static/corpus lab records are found
        fallback_msg = (
            "### 🧪 BIS Testing Laboratory Search\n\n"
            "Official testing laboratories for Indian Standards compliance are managed dynamically through the "
            "BIS Laboratory Information Management System (LIMS).\n\n"
            "#### 🔍 How to Find an Accredited Testing Lab:\n"
            "1. Visit the official **BIS LIMS Portal**.\n"
            "2. Filter by your **State / City** and applicable **Indian Standard (IS Number)**.\n"
            "3. View recognized, empaneled, and government testing laboratory scopes.\n\n"
            f"🔗 [Official BIS LIMS Portal]({OFFICIAL_LIMS_PORTAL})\n"
        )
        return {
            "intent": "lab_location",
            "flow": "lab_locator",
            "status": "unavailable",
            "total_found": 0,
            "labs": [],
            "formatted_text": fallback_msg,
            "source": "official_lims_fallback",
            "fallback_used": True,
        }


if __name__ == "__main__":
    locator = LabLocator()
    print(locator.search_labs("which BIS lab can test my product in Delhi?")["formatted_text"])
