"""
Phase 3, Part 3.3: Dual Citation Formatter & Validator Engine (citation_engine.py)
Parses generated response citations, binds them to Phase 1 provenance records,
validates citations against retrieved context to prevent hallucination, and renders clickable verification links.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("citation_engine")


class CitationEngine:
    """
    Handles citation formatting, provenance binding, and citation-to-context validation.
    """

    def __init__(self):
        pass

    @staticmethod
    def extract_inline_citations(text: str) -> List[str]:
        """Extracts inline citation tags from generated response text."""
        # Matches patterns like [IS 1786:2008, Clause 4.2], [As per IS 1417, Clause 3.1], [Per BIS Guidelines (Q.14)]
        matches = re.findall(r"\[(?:As per\s+|Per\s+)?([^\]]+)\]", text, re.IGNORECASE)
        return [m.strip() for m in matches if any(k in m.lower() for k in ["is ", "is:", "is-", "clause", "bis", "faq", "q."])]

    def validate_citations_against_context(
        self,
        citations_emitted: List[str],
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verifies that citations generated in the answer correspond directly
        to authentic chunks present in the retrieved candidate context.
        """
        valid_citations = []
        ungrounded_citations = []

        # Build context knowledge sets
        context_standards = set()
        context_clauses = set()
        context_chunk_ids = set()

        for item in context_chunks:
            doc = item.get("doc", item)
            is_no = (doc.get("is_number") or doc.get("standard") or "").lower().replace(" ", "").replace("-", "").replace(":", "")
            cl_no = str(doc.get("clause_number") or "").lower().strip()
            cid = str(doc.get("chunk_id") or "").lower()

            if is_no:
                context_standards.add(is_no)
            if cl_no:
                context_clauses.add(cl_no)
            if cid:
                context_chunk_ids.add(cid)

        for cite in citations_emitted:
            cite_clean = cite.lower().replace(" ", "").replace("-", "").replace(":", "")
            # Check standard match
            is_match = any(std in cite_clean for std in context_standards if len(std) > 2)
            # Check clause or FAQ match
            clause_match = any(cl in cite.lower() for cl in context_clauses if len(cl) >= 1)
            # Check general BIS match
            general_match = "bis" in cite.lower() or "faq" in cite.lower() or "guideline" in cite.lower()

            if is_match or (clause_match and general_match) or (general_match and len(context_chunks) > 0):
                valid_citations.append(cite)
            else:
                ungrounded_citations.append(cite)

        validation_result = {
            "total_emitted": len(citations_emitted),
            "valid_count": len(valid_citations),
            "ungrounded_count": len(ungrounded_citations),
            "valid_citations": valid_citations,
            "ungrounded_citations": ungrounded_citations,
            "all_valid": len(ungrounded_citations) == 0,
        }
        return validation_result

    def format_citations(
        self,
        response_text: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Binds retrieved context chunks to structured citations, validates inline citations,
        and appends a formatted verification footer.
        """
        formatted_citations = []
        seen_keys = set()

        for idx, item in enumerate(context_chunks, 1):
            doc = item.get("doc", item)
            chunk_id = doc.get("chunk_id", f"chunk_{idx}")
            is_number = doc.get("is_number") or doc.get("standard") or ""
            rev_year = doc.get("revision_year", "2026")
            clause_no = doc.get("clause_number")
            clause_title = doc.get("clause_title") or doc.get("product") or doc.get("title") or "Compliance Clause"
            page_start = doc.get("page_start", 1)
            source_file = doc.get("source_file", "")
            source_url = doc.get("source_url") or ""
            source_hash = doc.get("source_hash", "")
            source_of_truth = doc.get("source_of_truth", "official_bis")
            category = doc.get("category", "general")

            # Determine citation label and badge type
            if "faq" in category or "faq" in source_file.lower() or "faq" in source_url.lower():
                citation_label = f"Per BIS FAQ Guidelines (Q.{clause_no})" if clause_no else f"Per BIS FAQ Guidelines ({clause_title})"
                badge_type = "FAQ_GUIDELINE"
            elif category in ["is_standard", "standards"] or (is_number and (is_number.startswith("IS") or is_number.isdigit())):
                if clause_no:
                    citation_label = f"As per {is_number}:{rev_year}, Clause {clause_no}"
                else:
                    citation_label = f"As per {is_number}:{rev_year} ({clause_title})"
                badge_type = "OFFICIAL_STANDARD"
            elif category == "qco_order":
                citation_label = f"Per BIS Quality Control Order ({is_number or clause_title})"
                badge_type = "QCO_ORDER"
            else:
                citation_label = f"Per BIS Regulatory Record ({clause_title})"
                badge_type = "REGULATORY_RECORD"

            # Prefer verifiable official web URL if available, else clean relative PDF path
            if source_url and source_url.startswith("http"):
                target_url = source_url
            elif source_file:
                clean_file = Path(source_file.replace("\\", "/")).name
                target_url = f"raw_data/pdfs/{clean_file}#page={page_start}"
            else:
                target_url = "https://www.bis.gov.in/"

            citation_key = (citation_label, target_url)
            if citation_key not in seen_keys:
                seen_keys.add(citation_key)
                formatted_citations.append({
                    "chunk_id": chunk_id,
                    "label": citation_label,
                    "url": target_url,
                    "link": f"[{citation_label}]({target_url})",
                    "badge": badge_type,
                    "category": category,
                    "page": page_start,
                    "source_hash": source_hash,
                    "source_of_truth": source_of_truth,
                })

        # Validate inline citations present in generated response text
        emitted_tags = self.extract_inline_citations(response_text)
        validation_info = self.validate_citations_against_context(emitted_tags, context_chunks)

        # Build clean verification footer
        footer_lines = ["\n\n### 📖 Official Verification Sources & Citations:"]
        for cite in formatted_citations:
            if cite["badge"] == "OFFICIAL_STANDARD":
                icon = "📜 [Official Standard]"
            elif cite["badge"] == "QCO_ORDER":
                icon = "⚖️ [Mandatory QCO]"
            elif cite["badge"] == "FAQ_GUIDELINE":
                icon = "💡 [FAQ Guideline]"
            else:
                icon = "🏛️ [BIS Record]"

            page_str = f" *(Page {cite['page']})*" if cite["page"] and cite["page"] > 1 else ""
            footer_lines.append(f"- {icon} {cite['link']}{page_str}")

        footer_markdown = "\n".join(footer_lines)
        full_text = response_text.strip() + footer_markdown

        return {
            "formatted_text": full_text,
            "citations_list": formatted_citations,
            "inline_citations_emitted": emitted_tags,
            "validation": validation_info,
        }


if __name__ == "__main__":
    engine = CitationEngine()
    dummy_chunks = [
        {
            "doc": {
                "chunk_id": "test_01",
                "is_number": "IS 1786",
                "revision_year": "2008",
                "clause_number": "4.2",
                "clause_title": "Chemical Composition",
                "page_start": 6,
                "source_file": "raw_data/pdfs/IS_1786.pdf",
                "source_url": "https://standardsbis.bsbedge.com/BIS_Preview.aspx?id=1786_2008",
                "category": "is_standard",
                "source_hash": "abc123hash",
                "source_of_truth": "verified_bis_pdf",
            }
        }
    ]
    sample_text = "High strength deformed steel bars must satisfy composition limits [As per IS 1786:2008, Clause 4.2]."
    res = engine.format_citations(sample_text, dummy_chunks)
    print("Formatted Response:\n", res["formatted_text"])
    print("\nValidation Result:", res["validation"])
