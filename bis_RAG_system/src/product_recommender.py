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

    def recommend(self, query: str) -> Dict[str, Any]:
        """
        Recommends applicable Indian Standards for a product.
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

        log.info(f"Product Recommender -> Original: '{q_orig}' | Normalized: '{normalized_query}'")

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

        if best_match and highest_score >= 40:
            std_str = best_match.get("standard", "")
            rev_year = best_match.get("revision_year")
            full_std = f"{std_str}:{rev_year}" if rev_year else std_str
            mandatory_tag = "MANDATORY UNDER QCO/CRS" if best_match.get("mandatory") else "VOLUNTARY / NOTIFIED"
            scheme = best_match.get("scheme", "BIS Product Certification Scheme (Scheme-I)")
            url = best_match.get("source_url", "https://standardsbis.bsbedge.com/")

            formatted = (
                f"### 📦 Official Product Standard Recommendation\n\n"
                f"- **Product Category**: {best_match.get('product')}\n"
                f"- **Applicable Indian Standard**: **{full_std}**\n"
                f"- **Regulatory Status**: 🛑 **{mandatory_tag}**\n"
                f"- **Certification Scheme**: {scheme}\n"
                f"- **Source of Truth**: `{best_match.get('source_of_truth', 'verified_bis_api')}`\n\n"
                f"🔗 [Official BIS Standard Preview Link]({url})\n"
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
                corpus_hits = self.retrieval.retrieve(normalized_query, top_n=3, category="is_standard")
                if not corpus_hits:
                    corpus_hits = self.retrieval.retrieve(normalized_query, top_n=3, category=None)

                if corpus_hits:
                    top_hit = corpus_hits[0]
                    doc = top_hit.get("doc", top_hit)
                    std_no = doc.get("is_number") or doc.get("standard") or "BIS Standard"
                    rev_yr = doc.get("revision_year", "")
                    full_std = f"{std_no}:{rev_yr}" if rev_yr else str(std_no)
                    title = doc.get("clause_title") or doc.get("product") or doc.get("title") or "Technical Specification"
                    source_url = doc.get("source_url") or "https://www.bis.gov.in/"
                    source_hash = doc.get("source_hash", "")
                    source_of_truth = doc.get("source_of_truth", "verified_bis_pdf")

                    formatted = (
                        f"### 📦 Product Standard Recommendation (Corpus Search Result)\n\n"
                        f"- **Search Term**: {q_orig}\n"
                        f"- **Applicable Indian Standard**: **{full_std}** ({title})\n"
                        f"- **Source File**: `{doc.get('source_file', 'BIS Archive')}`\n"
                        f"- **Source of Truth**: `{source_of_truth}`\n\n"
                        f"🔗 [Official BIS Source Link]({source_url})\n"
                    )

                    return {
                        "intent": "product_recommendation",
                        "flow": "product_recommender",
                        "status": "success",
                        "product_data": {
                            "product": title,
                            "standard": full_std,
                            "source_url": source_url,
                            "source_hash": source_hash,
                            "source_of_truth": source_of_truth,
                        },
                        "formatted_text": formatted,
                        "source": "corpus_search_fallback",
                        "provenance": {
                            "source_url": source_url,
                            "source_hash": source_hash,
                            "source_of_truth": source_of_truth,
                            "chunk_id": doc.get("chunk_id", ""),
                        },
                        "fallback_used": True,
                    }
            except Exception as e:
                log.warning(f"Corpus retrieval fallback failed: {e}")

        # 3. Default structured guidance if no match
        formatted_guidance = (
            f"### 📦 Product Standard Recommendation\n\n"
            f"Could not find an exact compulsory product mapping for '{q_orig}' in the offline database.\n\n"
            f"Please verify the mandatory standard directly on the official BIS portal:\n"
            f"🔗 [Official BIS Compulsory Certification Product List](https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en)\n"
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
