"""
Phase 4, Step 12: Product -> Standard Recommender (product_recommender.py)
Deterministic tabular lookup mapping product names to mandatory Indian Standards (IS), QCO/CRS status, and scheme names.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("product_recommender")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw_data"

# Core product-to-standard mapping database
PRODUCT_DATABASE = [
    {
        "product_name": "LED Bulbs & Self-Ballasted LED Lamps",
        "aliases": ["led bulb", "led lamp", "led light", "bulb"],
        "is_number": "16102 (Part 1)",
        "revision_year": "2012",
        "mandatory_status": "MANDATORY_CRS",
        "scheme_name": "Compulsory Registration Scheme (Scheme-II)",
        "qco_order": "Electronics and IT Goods (Requirement for Compulsory Registration) Order",
        "official_url": "https://www.crsbis.in/BIS/",
    },
    {
        "product_name": "Steel Reinforcement Bars (TMT Bars)",
        "aliases": ["steel bar", "tmt bar", "steel reinforcement", "tmt steel", "iron rod"],
        "is_number": "1786",
        "revision_year": "2008",
        "mandatory_status": "MANDATORY_QCO",
        "scheme_name": "Product Certification Scheme (Scheme-I)",
        "qco_order": "Steel and Steel Products (Quality Control) Order",
        "official_url": "https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en",
    },
    {
        "product_name": "Solar Photovoltaic (PV) Modules & Panels",
        "aliases": ["solar panel", "pv module", "solar pv", "solar cell"],
        "is_number": "14286",
        "revision_year": "2010",
        "mandatory_status": "MANDATORY_CRS",
        "scheme_name": "Compulsory Registration Scheme (Scheme-II)",
        "qco_order": "Solar Photovoltaic Systems, Devices and Components QCO",
        "official_url": "https://www.crsbis.in/BIS/",
    },
    {
        "product_name": "Gold Jewellery and Gold Artefacts",
        "aliases": ["gold jewellery", "gold ornament", "hallmark gold", "gold ring", "gold chain"],
        "is_number": "1417",
        "revision_year": "2016",
        "mandatory_status": "MANDATORY_HALLMARKING",
        "scheme_name": "Hallmarking Scheme",
        "qco_order": "Hallmarking of Gold Jewellery and Gold Artefacts Order, 2020",
        "official_url": "https://www.bis.gov.in/hallmarking-overview/?lang=en",
    },
    {
        "product_name": "Packaged Drinking Water",
        "aliases": ["packaged water", "mineral water", "bottled water", "drinking water"],
        "is_number": "14543",
        "revision_year": "2004",
        "mandatory_status": "MANDATORY_QCO",
        "scheme_name": "Product Certification Scheme (Scheme-I)",
        "qco_order": "Food Safety and Standards (Packaging and Labelling) Regulations",
        "official_url": "https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en",
    },
    {
        "product_name": "Toys (Electric and Non-Electric)",
        "aliases": ["toy", "toys", "children toy", "electric toy"],
        "is_number": "9873 (Part 1)",
        "revision_year": "2019",
        "mandatory_status": "MANDATORY_QCO",
        "scheme_name": "Product Certification Scheme (Scheme-I)",
        "qco_order": "Toys (Quality Control) Order, 2020",
        "official_url": "https://www.bis.gov.in/product-certification/products-under-compulsory-certification/?lang=en",
    },
]


class ProductRecommender:
    def __init__(self):
        self.db = PRODUCT_DATABASE

    def recommend(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        log.info(f"Executing Product Recommender for: '{query}'")

        best_match = None
        highest_score = 0

        for item in self.db:
            score = 0
            for alias in item["aliases"]:
                if alias in q_lower:
                    score += len(alias) * 2
            if item["product_name"].lower() in q_lower:
                score += 20

            if score > highest_score:
                highest_score = score
                best_match = item

        if best_match and highest_score > 0:
            log.info(f"MATCH FOUND: '{best_match['product_name']}' (Score: {highest_score})")
            
            formatted_response = (
                f"### 📦 Product Standard Recommendation Results\n\n"
                f"**Product Category**: {best_match['product_name']}\n"
                f"**Applicable Indian Standard**: **IS {best_match['is_number']}:{best_match['revision_year']}**\n"
                f"**Certification Status**: 🛑 **{best_match['mandatory_status']}**\n"
                f"**BIS Scheme**: {best_match['scheme_name']}\n"
                f"**Quality Control Order (QCO)**: {best_match['qco_order']}\n\n"
                f"🔗 [Official BIS Scheme Link]({best_match['official_url']})\n"
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


if __name__ == "__main__":
    recommender = ProductRecommender()
    test_queries = ["is certification mandatory for LED bulbs", "do I need a licence for TMT steel bars", "gold ring hallmark"]
    for q in test_queries:
        res = recommender.recommend(q)
        print(res["formatted_text"])
