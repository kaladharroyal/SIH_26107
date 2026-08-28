"""
Phase 4, Step 12: Product -> Standard Recommender (product_recommender.py)
Deterministic tabular lookup mapping product names to mandatory Indian Standards (IS), QCO/CRS status, and scheme names.
Loads dynamically from Phase 1 generated product_standard_map.json with zero hard-coded production products.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("product_recommender")

BASE_DIR = Path(__file__).resolve().parent.parent


class ProductRecommender:
    def __init__(self, map_path: Optional[Path] = None):
        self.map_path = map_path or (BASE_DIR / "product_standard_map.json")
        self.db: List[Dict[str, Any]] = self._load_product_map()

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

    def recommend(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        log.info(f"Executing Product Recommender for: '{query}'")

        if not self.db:
            return {
                "status": "unavailable",
                "message": "Product standard mapping dataset is currently unavailable.",
                "formatted_text": "Product standard mapping dataset is currently unavailable. Please check the [Official BIS Compulsory Certification List](https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en).",
            }

        best_match = None
        highest_score = 0

        for item in self.db:
            product_name = item.get("product", "").lower()
            std_no = item.get("standard", "").lower()

            score = 0
            # Direct word and standard match scoring
            if product_name and product_name in q_lower:
                score += 20 + len(product_name)
            elif any(word in q_lower for word in product_name.split() if len(word) > 3):
                score += 10

            if std_no and std_no in q_lower:
                score += 30

            if score > highest_score:
                highest_score = score
                best_match = item

        if best_match and highest_score > 0:
            std_str = best_match.get("standard", "")
            rev_year = best_match.get("revision_year")
            full_std = f"{std_str}:{rev_year}" if rev_year else std_str
            mandatory_tag = "MANDATORY" if best_match.get("mandatory") else "VOLUNTARY / NOTIFIED"
            scheme = best_match.get("scheme", "BIS Certification Scheme")
            url = best_match.get("source_url", "https://www.bis.gov.in/")

            formatted_response = (
                f"### 📦 Product Standard Recommendation Results\n\n"
                f"**Product Category**: {best_match.get('product')}\n"
                f"**Applicable Indian Standard**: **{full_std}**\n"
                f"**Certification Status**: 🛑 **{mandatory_tag}**\n"
                f"**BIS Scheme**: {scheme}\n\n"
                f"🔗 [Official BIS Portal Link]({url})\n"
            )

            return {
                "status": "match_found",
                "product_data": best_match,
                "formatted_text": formatted_response,
            }

        log.warning(f"No direct product match found for query: '{query}'")
        return {
            "status": "no_match",
            "message": f"No direct product mapping found for '{query}'. Routing to general BIS Standards database search...",
            "formatted_text": f"Could not find an exact compulsory product mapping for '{query}'. Please check the [Official BIS Compulsory Certification List](https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en).",
        }
