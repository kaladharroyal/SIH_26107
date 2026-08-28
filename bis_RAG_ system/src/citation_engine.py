"""
Phase 3, Part 3.3: Dual Citation Formatter Engine (citation_engine.py)
Parses generated responses and formats citations into clean, clickable links for PDFs and Web FAQs.
"""

import re
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("citation_engine")


class CitationEngine:
    def format_citations(self, response_text: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        formatted_text = response_text
        formatted_citations = []

        for idx, item in enumerate(context_chunks, 1):
            doc = item.get("doc", item)
            is_number = doc.get("is_number", "RAW")
            rev_year = doc.get("revision_year", "2026")
            clause_no = doc.get("clause_number", "N/A")
            page_start = doc.get("page_start", 1)
            source_file = doc.get("source_file", "")
            source_url = doc.get("source_url") or doc.get("source_file", "")
            content_type = doc.get("content_type", "pdf")

            if content_type == "html_page" or "faq" in source_file.lower():
                # FAQ-sourced citation (Simplified Guideline)
                citation_label = f"Per BIS FAQ Guidelines (Q.{clause_no})"
                citation_link = f"[{citation_label}]({source_url})"
                badge_type = "FAQ_GUIDELINE"
            else:
                # Official Standard Clause citation (Legal Regulation)
                citation_label = f"As per IS {is_number}:{rev_year}, Clause {clause_no}"
                if source_file.endswith(".pdf"):
                    citation_url = f"file:///{source_file}#page={page_start}"
                else:
                    citation_url = source_url
                citation_link = f"[{citation_label}]({citation_url})"
                badge_type = "OFFICIAL_STANDARD"

            formatted_citations.append({
                "chunk_id": doc.get("chunk_id", f"chunk_{idx}"),
                "label": citation_label,
                "link": citation_link,
                "badge": badge_type,
                "source": source_url,
                "page": page_start,
            })

        # Append structured Citations block at footer
        footer_markdown = "\n\n### 📖 Verification Sources & Citations:\n"
        for cite in formatted_citations:
            badge_icon = "📜 [Official Regulation]" if cite["badge"] == "OFFICIAL_STANDARD" else "💡 [FAQ Guideline]"
            footer_markdown += f"- {badge_icon} {cite['link']} *(Page {cite['page']})*\n"

        return {
            "formatted_text": formatted_text + footer_markdown,
            "citations_list": formatted_citations,
        }


if __name__ == "__main__":
    engine = CitationEngine()
    dummy_chunks = [
        {
            "doc": {
                "chunk_id": "IS1786_2008_C4.2",
                "is_number": "1786",
                "revision_year": "2008",
                "clause_number": "4.2",
                "page_start": 6,
                "source_file": "d:/kaladharroyal/bis_RAG_system/raw_data/IS_1786.pdf",
                "content_type": "pdf",
            }
        },
        {
            "doc": {
                "chunk_id": "CERT_FAQ_Q14",
                "is_number": "FAQ",
                "clause_number": "14",
                "page_start": 1,
                "source_url": "https://www.bis.gov.in/product-certification/product-certification-faq/?lang=en",
                "content_type": "html_page",
            }
        },
    ]
    res = engine.format_citations("Steel reinforcement bars must meet 1786 standards.", dummy_chunks)
    print("Formatted Text:\n", res["formatted_text"])
