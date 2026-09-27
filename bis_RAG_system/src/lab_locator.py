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

    def search_labs(self, query: str, state: Optional[str] = None, language: str = "English") -> Dict[str, Any]:
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

        log.info(f"Executing Lab Locator -> Query: '{q_clean}' | State Filter: {state_filter} | Lang: '{language}'")

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
                    "citations": [
                        {
                            "label": "BIS Recognized Testing Laboratory Directory",
                            "url": OFFICIAL_LIMS_PORTAL,
                            "source_of_truth": "labs_directory_json",
                            "citation_type": "official_source",
                        }
                    ],
                    "fallback_used": False,
                }

        # 2. Fallback: Search in Phase 1/2 verified corpus (category == 'lab_directory')
        if self.retrieval is not None:
            log.info("Static lab directory is empty/unavailable. Querying corpus 'lab_directory' chunks...")
            try:
                if hasattr(self.retrieval, "retrieve_fast"):
                    corpus_hits = self.retrieval.retrieve_fast(q_clean or "laboratory testing facility", top_n=3, category="lab_directory")
                else:
                    corpus_hits = self.retrieval.retrieve(q_clean or "laboratory testing facility", top_n=3, category="lab_directory")
                if corpus_hits:
                    formatted_corpus = "### 🧪 BIS Testing Laboratory & Facility Guidance (Corpus Search)\n\n"
                    citations_list = []
                    for idx, hit in enumerate(corpus_hits, 1):
                        doc = hit.get("doc", hit)
                        title = doc.get("clause_title") or doc.get("title") or "Testing Guidelines"
                        text_snippet = doc.get("text", "")[:250].strip()
                        formatted_corpus += f"**{idx}. {title}**\n{text_snippet}...\n\n"
                        citations_list.append({
                            "label": title,
                            "url": doc.get("source_url") or OFFICIAL_LIMS_PORTAL,
                            "source_of_truth": doc.get("source_of_truth", "verified_bis_pdf"),
                            "citation_type": "corpus_record",
                            "chunk_id": doc.get("chunk_id", ""),
                        })

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
                        "citations": citations_list,
                        "fallback_used": True,
                    }
            except Exception as e:
                log.warning(f"Lab corpus search fallback failed: {e}")

        # 3. Default structured redirection when no static/corpus lab records are found
        lang_lower = (language or "english").lower()
        if "hindi" in lang_lower or lang_lower == "hi":
            fallback_msg = (
                "### 🧪 बीआईएस एलआईएमएस प्रयोगशाला खोज (BIS LIMS Discovery)\n\n"
                "भारतीय मानक अनुपालन हेतु आधिकारिक परीक्षण प्रयोगशालाएं बीआईएस प्रयोगशाला सूचना प्रबंधन प्रणाली (LIMS) के माध्यम से गतिशील रूप से प्रबंधित की जाती हैं। वर्तमान में अप्रचलित डेटा से बचने के लिए ऑफ़लाइन कॉर्पस में स्थानीय रिकॉर्ड संग्रहीत नहीं हैं।\n\n"
                "#### 🔍 अधिकृत प्रयोगशाला खोज प्रक्रिया:\n"
                "1. आधिकारिक **BIS LIMS पोर्टल** पर जाएं।\n"
                "2. अपने **राज्य / शहर** और लागू **भारतीय मानक (IS संख्या)** द्वारा फ़िल्टर करें।\n"
                "3. वर्तमान में मान्यता प्राप्त एवं पैनलबद्ध परीक्षण प्रयोगशालाओं के परीक्षण कार्यक्षेत्र (Test Scope) की जांच करें।\n\n"
                f"🔗 [आधिकारिक BIS LIMS पोर्टल]({OFFICIAL_LIMS_PORTAL})\n"
            )
        elif "telugu" in lang_lower or lang_lower == "te":
            fallback_msg = (
                "### 🧪 BIS LIMS ప్రయోగశాల గుర్తింపు (BIS LIMS Discovery)\n\n"
                "భారతీయ ప్రమాణాల అనుగుణ్యత కోసం అధికారిక పరీక్షా ప్రయోగశాలలు BIS ప్రయోగశాల సమాచార నిర్వహణ వ్యవస్థ (LIMS) ద్వారా నిర్వహించబడతాయి. పాతబడిపోయిన సమాచారాన్ని నివారించడానికి ఆఫ్‌లైన్ కార్పస్‌లో స్థానిక రికార్డులు నిల్వ చేయబడవు.\n\n"
                "#### 🔍 అధీకృత ప్రయోగశాల గుర్తింపు విధానం:\n"
                "1. అధికారిక **BIS LIMS పోర్టల్** ని సందర్శించండి.\n"
                "2. మీ **రాష్ట్రం / నగరం** మరియు వర్తించే **భారతీయ ప్రమాణం (IS సంఖ్య)** ద్వారా ఫిల్టర్ చేయండి.\n"
                "3. ప్రస్తుత గుర్తింపు పొందిన పరీక్షా ప్రయోగశాలల పరిధిని తనిఖీ చేయండి.\n\n"
                f"🔗 [అధికారిక BIS LIMS పోర్టల్]({OFFICIAL_LIMS_PORTAL})\n"
            )
        else:
            fallback_msg = (
                "### 🧪 BIS LIMS Laboratory Discovery\n\n"
                "Official testing laboratories for Indian Standards compliance are managed dynamically through the "
                "BIS Laboratory Information Management System (LIMS). Local verified laboratory records are not stored "
                "in the offline corpus to prevent outdated empanelment data.\n\n"
                "#### 🔍 Authoritative Laboratory Discovery:\n"
                "1. Visit the official **BIS LIMS Portal**.\n"
                "2. Filter by your **State / City** and applicable **Indian Standard (IS Number)**.\n"
                "3. View currently empaneled, recognized, and government testing laboratory scopes.\n\n"
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
            "citations": [
                {
                    "label": "Official BIS Laboratory Information Management System (LIMS)",
                    "url": OFFICIAL_LIMS_PORTAL,
                    "source_of_truth": "official_lims_portal",
                    "citation_type": "official_source",
                }
            ],
            "fallback_used": True,
        }


if __name__ == "__main__":
    locator = LabLocator()
    print(locator.search_labs("which BIS lab can test my product in Delhi?")["formatted_text"])
