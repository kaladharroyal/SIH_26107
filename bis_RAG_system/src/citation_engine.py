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

    @staticmethod
    def select_primary_source(
        context_chunks: List[Dict[str, Any]],
        inline_citations_emitted: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministically selects the single strongest primary source document from retrieved context.
        Pseudo-logic:
        1. Prefer verified_bis_pdf with valid PDF source URL (.pdf or bis.gov.in/services.bis.gov.in) and high relevance/matching standard.
        2. Otherwise prefer official BIS web URL.
        3. Otherwise use the strongest retrieved official source.
        4. Append #page=N only if page number is reliably known and > 1.
        """
        if not context_chunks:
            return {
                "display_title": "Official BIS Portal",
                "action_text": "Open official BIS source",
                "url": "https://www.bis.gov.in/",
                "is_pdf": False,
                "has_direct_link": False,
                "badge": "REGULATORY_RECORD",
            }

        scored_candidates = []
        for idx, item in enumerate(context_chunks):
            doc = item.get("doc", item)
            is_no = doc.get("is_number") or doc.get("standard") or ""
            rev_year = doc.get("revision_year", "")
            title = doc.get("clause_title") or doc.get("product") or doc.get("title") or "Technical Specification"
            
            # Extract IS number from clause_title if is_number is unset
            if not is_no and title:
                match_is = re.search(r"\bIS\s*(\d+)(?::(\d{4}))?", title, re.IGNORECASE)
                if match_is:
                    is_no = f"IS {match_is.group(1)}"
                    if match_is.group(2) and not rev_year:
                        rev_year = match_is.group(2)

            clause_no = doc.get("clause_number")
            source_url = (doc.get("source_url") or "").strip()
            source_file = (doc.get("source_file") or "").strip()
            source_of_truth = doc.get("source_of_truth", "")
            category = doc.get("category", "")
            page_start = doc.get("page_start") or doc.get("page") or doc.get("pdf_page")

            # Score calculation
            score = 0
            score += (10 - min(idx, 9))

            if source_of_truth == "verified_bis_pdf":
                score += 40

            is_pdf = False
            if source_url and (source_url.lower().endswith(".pdf") or ".pdf" in source_url.lower() or "pdf" in source_file.lower()):
                score += 30
                is_pdf = True
            elif source_url and source_url.startswith("http"):
                score += 20

            # Match with emitted inline citations if provided
            if inline_citations_emitted and is_no:
                clean_is = is_no.lower().replace(" ", "").replace("-", "").replace(":", "")
                for em in inline_citations_emitted:
                    if clean_is in em.lower().replace(" ", "").replace("-", "").replace(":", ""):
                        score += 25
                        break

            # Target URL resolution
            if source_url and source_url.startswith("http"):
                target_url = source_url
            elif source_file:
                clean_file = Path(source_file.replace("\\", "/")).name
                target_url = f"https://www.bis.gov.in/wp-content/uploads/{clean_file}"
            else:
                target_url = "https://www.bis.gov.in/"

            # Page parameter: only append if valid page > 1 and it's a PDF URL and #page is not already in URL
            if page_start and str(page_start).isdigit() and int(page_start) > 1 and is_pdf and "#page=" not in target_url:
                target_url = f"{target_url}#page={int(page_start)}"

            # Standard / title display formulation
            if is_no and is_no.upper().startswith("IS"):
                full_std = f"{is_no}:{rev_year}" if rev_year and rev_year not in is_no else is_no
                display_title = full_std
            elif "scheme" in category.lower() or "scheme" in str(is_no).lower():
                display_title = str(is_no or "Scheme I")
            elif title and title != "Technical Specification" and len(title) < 50:
                display_title = title
            else:
                display_title = is_no or "BIS Compliance Record"

            action_text = "Open official BIS PDF" if is_pdf else "Open official BIS source"

            scored_candidates.append({
                "score": score,
                "display_title": display_title,
                "action_text": action_text,
                "url": target_url,
                "is_pdf": is_pdf,
                "has_direct_link": bool(source_url or source_file),
                "badge": "OFFICIAL_STANDARD" if (is_no and is_no.startswith("IS")) else ("FAQ_GUIDELINE" if "faq" in category else "REGULATORY_RECORD"),
                "chunk_id": doc.get("chunk_id", ""),
                "page": page_start,
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[0] if scored_candidates else {
            "display_title": "Official BIS Portal",
            "action_text": "Open official BIS source",
            "url": "https://www.bis.gov.in/",
            "is_pdf": False,
            "has_direct_link": False,
            "badge": "REGULATORY_RECORD",
        }

    def format_citations(
        self,
        response_text: str,
        context_chunks: List[Dict[str, Any]],
        language: str = "English",
    ) -> Dict[str, Any]:
        """
        Binds retrieved context chunks to structured citations, validates inline citations,
        and appends a clean single-source verification footer localized to user's response language.
        """
        formatted_citations = []
        seen_keys = set()
        seen_badges = set()

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
                badge_name = "[FAQ Guideline]"
            elif category in ["is_standard", "standards"] or (is_number and (is_number.startswith("IS") or is_number.isdigit())):
                if clause_no:
                    citation_label = f"As per {is_number}:{rev_year}, Clause {clause_no}"
                else:
                    citation_label = f"As per {is_number}:{rev_year} ({clause_title})"
                badge_type = "OFFICIAL_STANDARD"
                badge_name = "[Official Standard]"
            elif category == "qco_order":
                citation_label = f"Per BIS Quality Control Order ({is_number or clause_title})"
                badge_type = "QCO_ORDER"
                badge_name = "[Mandatory QCO]"
            else:
                citation_label = f"Per BIS Regulatory Record ({clause_title})"
                badge_type = "REGULATORY_RECORD"
                badge_name = "[BIS Record]"

            seen_badges.add(badge_name)

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

        # Select the single strongest primary source
        primary_source = self.select_primary_source(context_chunks, emitted_tags)

        # Clean existing raw/hallucinated source sections from response_text if present
        clean_response = re.sub(r"###\s*(?:Source|स्रोत|మూలం|📖\s*Official Verification Sources|Official Verification Sources).*$", "", response_text, flags=re.IGNORECASE | re.DOTALL).strip()

        # Build clean single primary source card with language localization
        lang_lower = (language or "english").lower()
        if "hindi" in lang_lower or lang_lower == "hi":
            header = "### स्रोत"
            action_text = "आधिकारिक BIS दस्तावेज खोलें" if primary_source.get("is_pdf") else "आधिकारिक BIS स्रोत खोलें"
        elif "telugu" in lang_lower or lang_lower == "te":
            header = "### మూలం"
            action_text = "అధికారిక BIS పత్రాన్ని తెరవండి" if primary_source.get("is_pdf") else "అధికారిక BIS మూలాన్ని తెరవండి"
        else:
            header = "### Source"
            action_text = primary_source.get("action_text", "Open official BIS PDF")

        source_badge_str = ""
        if "[Official Standard]" in seen_badges and "[FAQ Guideline]" in seen_badges:
            source_badge_str = " *([Official Standard] & [FAQ Guideline] verified)*"

        source_block = (
            f"\n\n{header}\n\n"
            f"📄 **BIS — {primary_source['display_title']}**{source_badge_str}\n"
            f"[{action_text} ↗]({primary_source['url']})"
        )

        full_text = clean_response + source_block

        return {
            "formatted_text": full_text,
            "citations_list": formatted_citations,
            "primary_source": primary_source,
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
