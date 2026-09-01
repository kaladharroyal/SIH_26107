"""
Phase 4, Step 12: Product -> Standard Recommender (product_recommender.py)
Maps consumer and industry product queries to official Indian Standards (IS), mandatory status, and BIS schemes.
Supports consumer alias normalization, deterministic 81-record lookup from product_standard_map.json,
and corpus search fallback via Phase 2 hybrid retrieval.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("product_recommender")

BASE_DIR = Path(__file__).resolve().parent.parent

# Common consumer/industry terms mapped to official technical product categories
PRODUCT_ALIASES: Dict[str, str] = {
    "solar panel": "Photovoltaic Module",
    "solar panels": "Photovoltaic Module",
    "pv module": "Crystalline Silicon Terrestrial Photovoltaic",
    "pv modules": "Crystalline Silicon Terrestrial Photovoltaic",
    "tmt bar": "High strength deformed steel bars",
    "tmt bars": "High strength deformed steel bars",
    "steel bar": "High strength deformed steel bars",
    "steel bars": "High strength deformed steel bars",
    "steel reinforcement": "High strength deformed steel bars and wires for concrete",
    "reinforcement steel": "High strength deformed steel bars and wires for concrete",
    "led bulb": "Self-Ballasted LED Lamps",
    "led bulbs": "Self-Ballasted LED Lamps",
    "helmet": "Protective Helmets for Two Wheeler",
    "helmets": "Protective Helmets for Two Wheeler",
    "two wheeler helmet": "Protective Helmets for Two Wheeler",
    "cement": "Ordinary Portland Cement",
    "pvc pipe": "Unplasticized PVC Pipes for Potable Water Supplies",
    "pvc pipes": "Unplasticized PVC Pipes for Potable Water Supplies",
    "gold jewellery": "Gold and Gold Alloys Hallmarking",
    "gold jewelry": "Gold and Gold Alloys Hallmarking",
    "drinking water": "Packaged Drinking Water",
    "packaged water": "Packaged Drinking Water",
    "electric iron": "Electric Iron Safety Requirements",
}


class ProductRecommender:
    """
    Deterministic product-to-standard recommender with alias normalization,
    static dataset lookup, and corpus search fallback.
    """

    def __init__(self, map_path: Optional[Path] = None, retrieval_pipeline: Optional[Any] = None):
        self.map_path = map_path or (BASE_DIR / "product_standard_map.json")
        self.db: List[Dict[str, Any]] = self._load_product_map()
        self.retrieval = retrieval_pipeline

    def _load_product_map(self) -> List[Dict[str, Any]]:
        """Loads verified product-standard mappings from Phase 1 generated dataset."""
        if not self.map_path.exists():
            log.warning(f"Product standard map not found at {self.map_path}")
            return []

        try:
            with open(self.map_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            records = data.get("records", [])
            log.info(f"Loaded {len(records)} verified product-standard mappings from {self.map_path.name}")
            return records
        except Exception as e:
            log.error(f"Error loading product standard map: {e}")
            return []

    @staticmethod
    def normalize_query_with_aliases(query: str) -> str:
        """Normalizes user query with known consumer alias mappings."""
        q_clean = query.strip().lower()
        for alias, formal_term in PRODUCT_ALIASES.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q_clean):
                return formal_term
        return query.strip()

    def recommend(self, query: str, language: str = "English") -> Dict[str, Any]:
        """
        Recommends applicable Indian Standards for a product localized to requested language.
        First checks static product_standard_map.json; if no match, falls back to corpus search.
        """
        if not query or not query.strip():
            return {
                "intent": "product_recommendation",
                "flow": "product_recommender",
                "status": "invalid_query",
                "product_data": None,
                "formatted_text": "Please provide a valid product name or standard number.",
                "source": "product_recommender",
                "fallback_used": False,
            }

        q_orig = query.strip()
        normalized_query = self.normalize_query_with_aliases(q_orig)
        q_lower = normalized_query.lower()
        q_orig_lower = q_orig.lower()

        log.info(f"Product Recommender -> Original: '{q_orig}' | Normalized: '{normalized_query}' | Lang: '{language}'")

        # 1. Search in verified 81-record static map
        best_match = None
        highest_score = 0

        for item in self.db:
            product_name = item.get("product", "").lower()
            std_no = item.get("standard", "").lower()

            score = 0
            # Direct IS Number match
            if std_no and (std_no in q_orig_lower or std_no in q_lower):
                score += 50

            # Direct product title match
            if product_name:
                if product_name in q_lower or product_name in q_orig_lower:
                    score += 40 + len(product_name)
                elif len(q_lower) >= 15 and q_lower in product_name:
                    score += 40 + len(q_lower)
                elif len(q_orig_lower) >= 15 and q_orig_lower in product_name:
                    score += 40 + len(q_orig_lower)
                else:
                    unique_matching_words = {word for word in product_name.split() if len(word) > 3 and (word in q_lower or word in q_orig_lower)}
                    if len(unique_matching_words) >= 3:
                        score += 40

            if score > highest_score:
                highest_score = score
                best_match = item

        lang_lower = (language or "english").lower()

        if best_match and highest_score >= 40:
            std_str = best_match.get("standard", "")
            rev_year = best_match.get("revision_year")
            full_std = f"{std_str}:{rev_year}" if rev_year else std_str
            mandatory_tag = "Mandatory under QCO/CRS" if best_match.get("mandatory") else "Voluntary / Notified"
            scheme = best_match.get("scheme", "BIS Product Certification Scheme (Scheme-I)")
            url = best_match.get("source_url", "https://standardsbis.bsbedge.com/")

            if "hindi" in lang_lower or lang_lower == "hi":
                formatted = (
                    f"### बीआईएस मानक अनुशंसा\n\n"
                    f"**लागू मानक: {full_std}**\n\n"
                    f"प्राप्त बीआईएस दस्तावेजों के अनुसार, **{best_match.get('product')}** के लिए लागू भारतीय मानक **{full_std}** है।\n\n"
                    f"### मुख्य विवरण\n"
                    f"- **उत्पाद श्रेणी**: {best_match.get('product')}\n"
                    f"- **नियामक स्थिति**: {mandatory_tag}\n"
                    f"- **प्रमाणन योजना**: {scheme}\n\n"
                    f"### स्रोत\n\n"
                    f"📄 **BIS — {full_std}**\n"
                    f"[आधिकारिक BIS दस्तावेज खोलें ↗]({url})\n"
                )
            elif "telugu" in lang_lower or lang_lower == "te":
                formatted = (
                    f"### BIS ప్రమాణ సిఫార్సు\n\n"
                    f"**వర్తించే ప్రమాణం: {full_std}**\n\n"
                    f"పొందబడిన BIS రికార్డుల ప్రకారం, **{best_match.get('product')}** కి వర్తించే భారతీయ ప్రమాణం **{full_std}**.\n\n"
                    f"### ముఖ్యమైన వివరాలు\n"
                    f"- **ఉత్పత్తి వర్గం**: {best_match.get('product')}\n"
                    f"- **నియంత్రణ స్థితి**: {mandatory_tag}\n"
                    f"- **ధృవీకరణ పథకం**: {scheme}\n\n"
                    f"### మూలం\n\n"
                    f"📄 **BIS — {full_std}**\n"
                    f"[అధికారిక BIS పత్రాన్ని తెరవండి ↗]({url})\n"
                )
            elif "hinglish" in lang_lower:
                formatted = (
                    f"### BIS Standard Recommendation\n\n"
                    f"**Applicable Standard: {full_std}**\n\n"
                    f"Retrieved BIS documents ke anusaar, **{best_match.get('product')}** ke liye applicable Indian Standard **{full_std}** hai.\n\n"
                    f"### Important Details\n"
                    f"- **Product Category**: {best_match.get('product')}\n"
                    f"- **Regulatory Status**: {mandatory_tag}\n"
                    f"- **Certification Scheme**: {scheme}\n\n"
                    f"### Source\n\n"
                    f"📄 **BIS — {full_std}**\n"
                    f"[Open official BIS document ↗]({url})\n"
                )
            else:
                formatted = (
                    f"### BIS Standard Recommendation\n\n"
                    f"**Applicable Standard: {full_std}**\n\n"
                    f"The verified BIS material retrieved for this query identifies **{full_std}** as the applicable Indian Standard for **{best_match.get('product')}**.\n\n"
                    f"### Key Details\n"
                    f"- **Product Category**: {best_match.get('product')}\n"
                    f"- **Regulatory Status**: {mandatory_tag}\n"
                    f"- **Certification Scheme**: {scheme}\n\n"
                    f"### Source\n\n"
                    f"📄 **BIS — {full_std}**\n"
                    f"[Open official BIS document ↗]({url})\n"
                )

            return {
                "intent": "product_recommendation",
                "flow": "product_recommender",
                "status": "success",
                "product_data": best_match,
                "formatted_text": formatted,
                "source": "product_standard_map",
                "provenance": {
                    "source_url": url,
                    "source_hash": best_match.get("source_hash", ""),
                    "source_of_truth": best_match.get("source_of_truth", "verified_bis_api"),
                },
                "fallback_used": False,
            }

        # 2. Fallback to Phase 2 Corpus Retrieval if available
        if self.retrieval is not None:
            log.info(f"Static map had no match for '{q_orig}'. Triggering Phase 2 corpus retrieval fallback...")
            try:
                if hasattr(self.retrieval, "retrieve_fast"):
                    corpus_hits = self.retrieval.retrieve_fast(normalized_query, top_n=3, category="is_standard")
                    if not corpus_hits:
                        corpus_hits = self.retrieval.retrieve_fast(normalized_query, top_n=3, category=None)
                else:
                    corpus_hits = self.retrieval.retrieve(normalized_query, top_n=3, category="is_standard")
                    if not corpus_hits:
                        corpus_hits = self.retrieval.retrieve(normalized_query, top_n=3, category=None)

                if corpus_hits:
                    # Select the best hit that matches query specifics
                    best_doc = corpus_hits[0].get("doc", corpus_hits[0])
                    identified_std = None
                    identified_title = None

                    # Check for explicit standard patterns matching the query in retrieved context
                    q_lower_words = set(re.findall(r"\w+", normalized_query.lower()))
                    for hit in corpus_hits[:5]:
                        d = hit.get("doc", hit)
                        h_title = d.get("clause_title") or d.get("product") or d.get("title") or ""
                        h_text = d.get("text", "")
                        h_combined = f"{h_title}\n{h_text}"

                        # Check for cement matches
                        if "cement" in q_lower_words or "portland" in q_lower_words:
                            m_cem = re.search(r"\b(IS\s*269(?::\d{4})?|IS\s*1489(?:\s*\(Part\s*\d+\))?(?::\d{4})?|IS\s*455(?::\d{4})?)\b", h_combined, re.IGNORECASE)
                            if m_cem:
                                identified_std = m_cem.group(1).strip()
                                identified_title = "Ordinary Portland Cement (OPC)"
                                best_doc = d
                                break
                        # Check for steel / TMT reinforcement matches
                        if "tmt" in q_lower_words or "reinforcement" in q_lower_words or "steel" in q_lower_words:
                            m_steel = re.search(r"\b(IS\s*1786(?::\d{4})?|IS\s*432(?:\s*\(Part\s*\d+\))?(?::\d{4})?|IS\s*2062(?::\d{4})?)\b", h_combined, re.IGNORECASE)
                            if m_steel:
                                identified_std = m_steel.group(1).strip()
                                identified_title = "High strength deformed steel bars and wires for concrete reinforcement (TMT)"
                                best_doc = d
                                break

                    doc = best_doc
                    title = identified_title or doc.get("clause_title") or doc.get("product") or doc.get("title") or "Technical Specification"
                    std_no = identified_std or doc.get("is_number") or doc.get("standard")
                    rev_yr = doc.get("revision_year", "")

                    if not std_no and title:
                        match_is = re.search(r"\bIS\s*(\d+)(?::(\d{4}))?", title, re.IGNORECASE)
                        if match_is:
                            std_no = f"IS {match_is.group(1)}"
                            if match_is.group(2) and not rev_yr:
                                rev_yr = match_is.group(2)

                    std_no = std_no or "BIS Standard"
                    full_std = f"{std_no}:{rev_yr}" if rev_yr and rev_yr not in str(std_no) and ":" not in str(std_no) else str(std_no)
                    source_url = doc.get("source_url") or "https://www.bis.gov.in/"
                    source_hash = doc.get("source_hash", "")
                    source_of_truth = doc.get("source_of_truth", "verified_bis_pdf")
                    page_num = doc.get("page_start") or doc.get("page") or doc.get("pdf_page")

                    target_url = source_url
                    is_pdf = bool(source_url and ".pdf" in source_url.lower())
                    if page_num and str(page_num).isdigit() and int(page_num) > 1 and is_pdf and "#page=" not in target_url:
                        target_url = f"{target_url}#page={int(page_num)}"

                    action_txt = "Open official BIS PDF ↗" if is_pdf else "Open official BIS source ↗"

                    details_lines = []
                    if title and title != "Technical Specification":
                        details_lines.append(f"- **Product / Specification**: {title}")
                    if doc.get("clause_number"):
                        details_lines.append(f"- **Clause**: Clause {doc.get('clause_number')}")
                    if doc.get("text"):
                        clean_excerpt = " ".join(doc.get("text", "").split()[:35])
                        details_lines.append(f"- **Summary**: {clean_excerpt}...")

                    details_str = "\n".join(details_lines) if details_lines else "- Information identified in verified BIS regulatory records."

                    if "hindi" in lang_lower or lang_lower == "hi":
                        formatted = (
                            f"### बीआईएस मानक अनुशंसा\n\n"
                            f"**लागू मानक: {full_std}**\n\n"
                            f"प्राप्त बीआईएस दस्तावेजों के अनुसार, इस उत्पाद के लिए लागू भारतीय मानक **{full_std}** है।\n\n"
                            f"### मुख्य विवरण\n"
                            f"{details_str}\n\n"
                            f"### स्रोत\n\n"
                            f"📄 **BIS — {full_std}**\n"
                            f"[आधिकारिक BIS दस्तावेज खोलें ↗]({target_url})\n"
                        )
                    elif "telugu" in lang_lower or lang_lower == "te":
                        formatted = (
                            f"### BIS ప్రమాణ సిఫార్సు\n\n"
                            f"**వర్తించే ప్రమాణం: {full_std}**\n\n"
                            f"పొందబడిన BIS రికార్డుల ప్రకారం, ఈ ఉత్పత్తికి వర్తించే భారతీయ ప్రమాణం **{full_std}**.\n\n"
                            f"### ముఖ్యమైన వివరాలు\n"
                            f"{details_str}\n\n"
                            f"### మూలం\n\n"
                            f"📄 **BIS — {full_std}**\n"
                            f"[అధికారిక BIS పత్రాన్ని తెరవండి ↗]({target_url})\n"
                        )
                    elif "hinglish" in lang_lower:
                        formatted = (
                            f"### BIS Standard Recommendation\n\n"
                            f"**Applicable Standard: {full_std}**\n\n"
                            f"Retrieved BIS documents ke anusaar, is query ke liye applicable Indian Standard **{full_std}** hai.\n\n"
                            f"### Important Details\n"
                            f"{details_str}\n\n"
                            f"### Source\n\n"
                            f"📄 **BIS — {full_std}**\n"
                            f"[Open official BIS document ↗]({target_url})\n"
                        )
                    else:
                        formatted = (
                            f"### BIS Standard Recommendation\n\n"
                            f"**Applicable Standard: {full_std}**\n\n"
                            f"The verified BIS material retrieved for this query identifies **{full_std}** as the applicable Indian Standard.\n\n"
                            f"### Key Details\n"
                            f"{details_str}\n\n"
                            f"### Source\n\n"
                            f"📄 **BIS — {full_std}**\n"
                            f"[{action_txt}]({target_url})\n"
                        )

                    return {
                        "intent": "product_recommendation",
                        "flow": "product_recommender",
                        "status": "success",
                        "product_data": {
                            "product": title,
                            "standard": full_std,
                            "source_url": target_url,
                            "source_hash": source_hash,
                            "source_of_truth": source_of_truth,
                        },
                        "formatted_text": formatted,
                        "source": "corpus_search_fallback",
                        "provenance": {
                            "source_url": target_url,
                            "source_hash": source_hash,
                            "source_of_truth": source_of_truth,
                            "chunk_id": doc.get("chunk_id", ""),
                        },
                        "fallback_used": True,
                    }
            except Exception as e:
                log.warning(f"Corpus retrieval fallback failed: {e}")

        # 3. Default structured guidance if no match
        if "hindi" in lang_lower or lang_lower == "hi":
            formatted_guidance = (
                f"### बीआईएस स्रोत से पुष्टि करने में असमर्थ\n\n"
                f"उपलब्ध BIS सामग्री इस जानकारी की पुष्टि करने के लिए पर्याप्त नहीं है।\n\n"
                f"कृपया बीआईएस मानकों, प्रमाणन या उपभोक्ता सूचना से संबंधित प्रश्न पूछें।\n\n"
                f"### स्रोत\n\n"
                f"📄 **BIS अनिवार्य प्रमाणन निर्देशिका**\n"
                f"[आधिकारिक BIS पोर्टल खोलें ↗](https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en)\n"
            )
        elif "telugu" in lang_lower or lang_lower == "te":
            formatted_guidance = (
                f"### BIS మూలాల నుండి నిర్ధారించలేకపోయాము\n\n"
                f"లభ్యమైన BIS సమాచారం దీనిని నిర్ధారించడానికి సరిపోదు.\n\n"
                f"దయచేసి BIS ప్రమాణాలు, ధృవీకరణ లేదా వినియోగదారు సమాచారం గురించి అడగండి.\n\n"
                f"### మూలం\n\n"
                f"📄 **BIS తప్పనిసరి ధృవీకరణ డైరెక్టరీ**\n"
                f"[అధికారిక BIS పోర్టల్‌ను తెరవండి ↗](https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en)\n"
            )
        else:
            formatted_guidance = (
                f"### Unable to Confirm from BIS Sources\n\n"
                f"The retrieved BIS material does not provide enough information to confirm a specific Indian Standard for '{q_orig}'.\n\n"
                f"Please ask a BIS standards, certification, hallmarking, laboratory, or consumer-information question.\n\n"
                f"### Source\n\n"
                f"📄 **BIS Compulsory Certification Directory**\n"
                f"[Open official BIS portal ↗](https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en)\n"
            )
        return {
            "intent": "product_recommendation",
            "flow": "product_recommender",
            "status": "no_match",
            "product_data": None,
            "formatted_text": formatted_guidance,
            "source": "product_recommender",
            "fallback_used": False,
        }


if __name__ == "__main__":
    recommender = ProductRecommender()
    print("--- TEST 1: Alias Normalization (TMT Bar) ---")
    res1 = recommender.recommend("what BIS standard should I use for a TMT bar?")
    print(res1["formatted_text"])

    print("\n--- TEST 2: Direct IS lookup (IS 1070) ---")
    res2 = recommender.recommend("IS 1070")
    print(res2["formatted_text"])
