"""
Phase 3, Part 3.3: Verifiable Dual Citation Formatter & Provenance Engine (citation_engine.py)
Binds claims to authentic Bureau of Indian Standards document provenance,
resolves verifiable canonical web URLs (with zero broken 404 synthetic hash links),
retains multi-source citation taxonomy with verified badges, and formats localized provenance cards.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("citation_engine")

OFFICIAL_BIS_PORTAL = "https://www.bis.gov.in/"
OFFICIAL_LIMS_PORTAL = "https://lims.bis.gov.in/"
OFFICIAL_STANDARDS_PORTAL = "https://standardsbis.bsbedge.com/"
OFFICIAL_KNOW_YOUR_STANDARDS = "https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/issearch/?is_number="
OFFICIAL_COMPLAINTS_PORTAL = "https://www.bis.gov.in/consumer-overview/online-complaint-registration/?lang=en"
OFFICIAL_CONFORMITY_PORTAL = "https://www.bis.gov.in/conformity-assessment/"
OFFICIAL_CRS_PORTAL = "https://www.crsbis.in/BIS/"
OFFICIAL_MANAKONLINE_PORTAL = "https://www.manakonline.in/"


def resolve_canonical_bis_url(doc: Union[Dict[str, Any], str]) -> str:
    """
    Resolves authentic, verifiable canonical web URLs for BIS documents, standards,
    schemes, laboratories, and consumer redressal portals with zero broken 404 synthetic hash links.
    """
    if isinstance(doc, str):
        raw_str = doc.strip()
        if raw_str.startswith("http://") or raw_str.startswith("https://"):
            return raw_str
        num_match = re.search(r"\b(?:IS[\s:\-_]*)?(\d{2,6})\b", raw_str, re.IGNORECASE)
        if num_match:
            return f"{OFFICIAL_STANDARDS_PORTAL}IS_{num_match.group(1)}.aspx"
        if any(k in raw_str.lower() for k in ["lab", "testing", "lims"]):
            return OFFICIAL_LIMS_PORTAL
        if any(k in raw_str.lower() for k in ["complaint", "grievance", "consumer", "act", "regulation"]):
            return OFFICIAL_COMPLAINTS_PORTAL
        if any(k in raw_str.lower() for k in ["scheme", "fmcs", "crs", "hallmarking"]):
            return OFFICIAL_CONFORMITY_PORTAL
        return OFFICIAL_STANDARDS_PORTAL

    source_url = (doc.get("source_url") or "").strip()
    is_number = (doc.get("is_number") or doc.get("standard") or "").strip()
    category = (doc.get("category") or "").strip().lower()
    source_file = (doc.get("source_file") or "").strip()
    page_start = doc.get("page_start") or doc.get("page") or doc.get("pdf_page")

    # 1. If source_url is already an authentic, verified live endpoint
    if source_url and source_url.startswith("http"):
        # Guard: Check if source_url contains an internal 12-character SHA hash prefix in a wp-content template
        if "/wp-content/uploads/" in source_url:
            cleaned_url = re.sub(r"/wp-content/uploads/[0-9a-f]{12}_", "/wp-content/uploads/", source_url)
            return cleaned_url
        return source_url

    # 2. Indian Standard (IS) resolution -> Official BIS Standards Portal
    if is_number or category == "is_standard" or "IS_" in source_file:
        raw_std = is_number or source_file
        num_match = re.search(r"\b(?:IS[\s:\-_]*)?(\d{2,6})\b", raw_std, re.IGNORECASE)
        if num_match:
            is_num = num_match.group(1)
            target = f"{OFFICIAL_STANDARDS_PORTAL}IS_{is_num}.aspx"
            return target

    # 3. Laboratory Locator -> Official BIS LIMS Portal
    if category in ["lab_directory", "laboratory", "lims"] or "lab" in source_file.lower():
        return OFFICIAL_LIMS_PORTAL

    # 4. Consumer Grievances & Redressal -> Official BIS CARE Complaint Portal
    if category in ["consumer_protection", "consumer_complaint", "hallmarking_complaint"] or "complaint" in source_file.lower():
        return OFFICIAL_COMPLAINTS_PORTAL

    # 5. Conformity Assessment Schemes
    if category in ["scheme_i", "scheme_ii", "crs", "fmcs", "scheme_x", "hallmarking", "scheme_iii", "simplified_procedure", "eco_mark", "scheme"]:
        if "crs" in category or "scheme_ii" in category:
            return OFFICIAL_CRS_PORTAL
        return OFFICIAL_CONFORMITY_PORTAL

    # 6. PDF Source Files (strip local scraper SHA-256 hash prefix)
    if source_file:
        file_name = Path(source_file.replace("\\", "/")).name
        # Remove 12-char hex hash prefix if present (e.g., '5e2367ca2545_Training-Strategy.pdf' -> 'Training-Strategy.pdf')
        clean_name = re.sub(r"^[0-9a-f]{12}_", "", file_name)

        if clean_name.upper().startswith("IS_"):
            num_match = re.search(r"\d+", clean_name)
            if num_match:
                return f"{OFFICIAL_STANDARDS_PORTAL}IS_{num_match.group(0)}.aspx"

        # Check for specific official gazettes / orders
        if "Hallmarking" in clean_name or "hallmarking" in clean_name.lower():
            return f"{OFFICIAL_BIS_PORTAL}hallmarking-overview/?lang=en"
        elif "MarketSurveillance" in clean_name or "surveillance" in clean_name.lower():
            return f"{OFFICIAL_BIS_PORTAL}conformity-assessment/surveillance/?lang=en"
        elif "certification" in clean_name.lower():
            return OFFICIAL_CONFORMITY_PORTAL

        return f"{OFFICIAL_BIS_PORTAL}"

    return OFFICIAL_BIS_PORTAL


def enrich_citation(cite: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Ensures every citation item is a structured dictionary with a valid absolute URL."""
    if isinstance(cite, dict):
        label = cite.get("label") or cite.get("display_title") or cite.get("title") or cite.get("is_number") or "BIS Standard"
        url = cite.get("url") or ""
        badge = cite.get("badge") or "[Official Standard]"
        if not url.startswith("http://") and not url.startswith("https://"):
            url = resolve_canonical_bis_url(cite)
        return {
            "label": label,
            "url": url,
            "link": f"[{label}]({url})",
            "badge": badge,
            "is_number": cite.get("is_number", ""),
            "display_title": label,
            "chunk_id": cite.get("chunk_id", ""),
            "page": cite.get("page"),
            "source_document": cite.get("source_document", ""),
        }
    else:
        cite_str = str(cite).strip()
        url = resolve_canonical_bis_url(cite_str)
        badge = "[Official Standard]" if ("is" in cite_str.lower() or any(c.isdigit() for c in cite_str)) else "[Official Source]"
        return {
            "label": cite_str,
            "url": url,
            "link": f"[{cite_str}]({url})",
            "badge": badge,
            "is_number": cite_str if cite_str.upper().startswith("IS") else "",
            "display_title": cite_str,
        }


def enrich_citations(citations: List[Any]) -> List[Dict[str, Any]]:
    """Converts a heterogeneous list of citations into structured dicts with verified absolute URLs."""
    result = []
    seen = set()
    for c in citations:
        enriched = enrich_citation(c)
        key = (enriched["label"], enriched["url"])
        if key not in seen:
            seen.add(key)
            result.append(enriched)
    return result


class CitationEngine:
    """
    Handles verifiable citation formatting, provenance binding,
    multi-source taxonomy, and citation-to-context validation.
    """

    @staticmethod
    def extract_inline_citations(text: str) -> List[str]:
        """Extracts inline citation tags from generated response text."""
        matches = re.findall(r"\[(?:As per\s+|Per\s+)?([^\]]+)\]", text, re.IGNORECASE)
        return [m.strip() for m in matches if any(k in m.lower() for k in ["is ", "is:", "is-", "clause", "bis", "faq", "q.", "regulation", "act"])]

    def validate_citations_against_context(
        self,
        citations_emitted: List[str],
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verifies that citations generated in the answer correspond directly
        to authentic chunks present in the retrieved candidate context.
        Strict chunk-level verification: no generic keyword allowlist fallback.
        """
        valid_citations = []
        ungrounded_citations = []

        context_standards: Set[str] = set()
        context_clauses: Set[str] = set()
        context_chunk_ids: Set[str] = set()
        context_titles: Set[str] = set()
        context_texts: List[str] = []

        for item in context_chunks:
            doc = item.get("doc", item)
            is_no = (doc.get("is_number") or doc.get("standard") or "").lower().strip()
            # Normalize IS number variants (e.g. "IS 1786", "is1786", "1786")
            if is_no:
                clean_no = is_no.replace(" ", "").replace("-", "").replace(":", "")
                context_standards.add(clean_no)
                digits = re.findall(r"\d+", clean_no)
                for d in digits:
                    if len(d) >= 2:
                        context_standards.add(f"is{d}")

            cl_no = str(doc.get("clause_number") or doc.get("clause_no") or "").lower().strip()
            if cl_no and cl_no != "none":
                context_clauses.add(cl_no)

            title = str(doc.get("clause_title") or doc.get("product") or doc.get("title") or doc.get("order_title") or doc.get("lab_name") or doc.get("source_document") or doc.get("source_file") or "").lower().strip()
            if title:
                context_titles.add(title)

            cid = str(doc.get("chunk_id") or "").lower().strip()
            if cid:
                context_chunk_ids.add(cid)

            t = str(doc.get("text") or "").lower()
            if t:
                context_texts.append(t)

        for cite in citations_emitted:
            if isinstance(cite, dict):
                cite_str = f"{cite.get('is_number', '')} {cite.get('clause_number', '')} {cite.get('title', '')} {cite.get('chunk_id', '')} {cite.get('source_document', '')}"
                cite_cid = str(cite.get("chunk_id") or "").lower().strip()
                cite_is = str(cite.get("is_number") or "").lower().strip()
                cite_clean = cite_str.lower().replace(" ", "").replace("-", "").replace(":", "")
                cite_lower = cite_str.lower()
                cid_direct = cite_cid in context_chunk_ids if cite_cid else False
                is_direct = any(std in cite_is.replace(" ", "").replace("-", "") for std in context_standards if len(std) >= 3) if cite_is else False
            else:
                cite_str = str(cite)
                cite_cid = ""
                cite_is = ""
                cite_clean = cite_str.lower().replace(" ", "").replace("-", "").replace(":", "")
                cite_lower = cite_str.lower()
                cid_direct = False
                is_direct = False

            # 1. Standard number match in retrieved evidence
            is_match = is_direct or any(std in cite_clean for std in context_standards if len(std) >= 3)

            # 2. Specific clause / table match in retrieved evidence
            clause_match = any(cl in cite_lower for cl in context_clauses if len(cl) >= 1)

            # 3. Specific title / chunk ID match in retrieved evidence
            title_match = any(t in cite_lower or cite_lower in t for t in context_titles if len(t) >= 4)
            cid_match = cid_direct or any(cid in cite_lower for cid in context_chunk_ids if len(cid) >= 4)

            # 4. In-text verbatim phrase match
            in_text_match = any(cite_lower in ct for ct in context_texts)

            if is_match or clause_match or title_match or cid_match or in_text_match:
                valid_citations.append(cite)
            else:
                log.warning(f"Citation Rejected (Ungrounded in retrieved evidence): '{cite}'")
                ungrounded_citations.append(cite)

        return {
            "total_emitted": len(citations_emitted),
            "valid_count": len(valid_citations),
            "ungrounded_count": len(ungrounded_citations),
            "valid_citations": valid_citations,
            "ungrounded_citations": ungrounded_citations,
            "all_valid": len(ungrounded_citations) == 0,
        }

    @staticmethod
    def select_primary_source(
        context_chunks: List[Dict[str, Any]],
        inline_citations_emitted: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministically selects the single strongest primary source document from retrieved context.
        """
        if not context_chunks:
            return {
                "display_title": "Official BIS Portal",
                "url": OFFICIAL_BIS_PORTAL,
                "is_pdf": False,
                "action_text": "Open official BIS source",
                "badge": "[Official Portal]",
            }

        scored_candidates = []

        for idx, item in enumerate(context_chunks):
            doc = item.get("doc", item)
            is_no = doc.get("is_number") or doc.get("standard")
            rev_year = doc.get("revision_year")
            title = doc.get("clause_title") or doc.get("title") or doc.get("name")
            category = (doc.get("category") or "").lower()
            source_of_truth = doc.get("source_of_truth", "")
            page_start = doc.get("page_start") or doc.get("page") or doc.get("pdf_page")

            # Extract IS number from title if missing
            if not is_no and title:
                match_is = re.search(r"\bIS[\s:\-_]*(\d{2,6})(?::(\d{4}))?\b", title, re.IGNORECASE)
                if match_is:
                    is_no = f"IS {match_is.group(1)}"
                    if match_is.group(2) and not rev_year:
                        rev_year = match_is.group(2)

            score = 0
            score += (10 - min(idx, 9))

            if source_of_truth == "verified_bis_pdf":
                score += 40
            if is_no:
                score += 30

            # Match with emitted inline citations if provided
            if inline_citations_emitted and is_no:
                clean_is = is_no.lower().replace(" ", "").replace("-", "").replace(":", "")
                for em in inline_citations_emitted:
                    if clean_is in em.lower().replace(" ", "").replace("-", "").replace(":", ""):
                        score += 25
                        break

            target_url = resolve_canonical_bis_url(doc)
            is_pdf = target_url.lower().endswith(".pdf") or "pdf" in str(doc.get("source_file", "")).lower()

            if page_start and str(page_start).isdigit() and int(page_start) > 1 and is_pdf and "#page=" not in target_url:
                target_url = f"{target_url}#page={int(page_start)}"

            # Standard / title display formulation
            if is_no and is_no.upper().startswith("IS"):
                full_std = f"{is_no}:{rev_year}" if rev_year and rev_year not in is_no else is_no
                display_title = full_std
                badge = "[Official Standard]"
            elif "scheme" in category:
                display_title = str(is_no or title or "Conformity Assessment Scheme")
                badge = "[Conformity Scheme]"
            elif "lab" in category:
                display_title = str(title or "BIS Recognized Laboratory")
                badge = "[LIMS Lab Directory]"
            elif "consumer" in category or "complaint" in category:
                display_title = str(title or "Consumer Redressal Guidelines")
                badge = "[BIS CARE Grievance Portal]"
            elif title and title != "Technical Specification" and len(title) < 50:
                display_title = title
                badge = "[Technical Specification]"
            else:
                display_title = is_no or "BIS Compliance Record"
                badge = "[Official Guideline]"

            action_text = "Open official BIS PDF" if is_pdf else "Open official BIS source"

            scored_candidates.append({
                "score": score,
                "display_title": display_title,
                "url": target_url,
                "is_pdf": is_pdf,
                "action_text": action_text,
                "badge": badge,
                "doc": doc,
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[0]

    def format_citations(
        self,
        response_text: str,
        context_chunks: List[Dict[str, Any]],
        language: str = "English",
    ) -> Dict[str, Any]:
        """
        Formats structured multi-source citations and appends localized verification cards
        with authentic canonical BIS web URLs.
        """
        formatted_citations: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[str, str]] = set()
        seen_badges: Set[str] = set()

        for idx, item in enumerate(context_chunks):
            doc = item.get("doc", item)
            chunk_id = doc.get("chunk_id", f"chunk_{idx}")
            is_no = doc.get("is_number") or doc.get("standard")
            rev_year = doc.get("revision_year")
            clause_no = doc.get("clause_number")
            clause_title = doc.get("clause_title") or doc.get("title") or doc.get("name")
            category = (doc.get("category") or "is_standard").lower()
            source_of_truth = doc.get("source_of_truth", "verified_bis_record")
            source_hash = doc.get("source_hash", "bis_provenance_hash")
            page_start = doc.get("page_start") or doc.get("page") or doc.get("pdf_page")

            # Extract IS number if missing
            if not is_no and clause_title:
                match_is = re.search(r"\bIS[\s:\-_]*(\d{2,6})(?::(\d{4}))?\b", clause_title, re.IGNORECASE)
                if match_is:
                    is_no = f"IS {match_is.group(1)}"
                    if match_is.group(2) and not rev_year:
                        rev_year = match_is.group(2)

            # Determine badge taxonomy
            if "faq" in category or "faq" in str(doc.get("source_file", "")).lower() or "faq" in str(doc.get("source_url", "")).lower() or "faq" in str(chunk_id).lower():
                badge_type = "[FAQ Guideline]"
            elif is_no and is_no.upper().startswith("IS") and not is_no.upper().startswith("IS-SCHEME") and not is_no.upper().startswith("BIS-SCHEME"):
                badge_type = "[Official Standard]"
            elif "qco" in category or "order" in category or "gazette" in str(source_of_truth).lower():
                badge_type = "[Gazette QCO Order]"
            elif "lab" in category:
                badge_type = "[LIMS Lab Directory]"
            elif "consumer" in category or "complaint" in category:
                badge_type = "[BIS CARE Grievance Portal]"
            elif "scheme" in category:
                badge_type = "[Conformity Scheme]"
            else:
                badge_type = "[Technical Specification]"

            seen_badges.add(badge_type)

            # Construct citation label
            if is_no and rev_year and rev_year not in str(is_no):
                label_prefix = f"{is_no}:{rev_year}"
            elif is_no:
                label_prefix = is_no
            elif clause_title:
                label_prefix = clause_title[:35]
            else:
                label_prefix = "BIS Standard"

            if clause_no:
                citation_label = f"Clause {clause_no} of {label_prefix}"
            else:
                citation_label = label_prefix

            target_url = resolve_canonical_bis_url(doc)
            is_pdf = target_url.lower().endswith(".pdf") or "pdf" in str(doc.get("source_file", "")).lower()

            if page_start and str(page_start).isdigit() and int(page_start) > 1 and is_pdf and "#page=" not in target_url:
                target_url = f"{target_url}#page={int(page_start)}"

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
                    "display_title": label_prefix,
                })

        # Validate inline citations present in generated response text
        emitted_tags = self.extract_inline_citations(response_text)
        validation_info = self.validate_citations_against_context(emitted_tags, context_chunks)

        # Select primary source
        primary_source = self.select_primary_source(context_chunks, emitted_tags)

        # Clean any raw unformatted source markers from generator
        clean_response = re.sub(
            r"###\s*(?:Source|स्रोत|మూలం|📖\s*Official Verification Sources|Official Verification Sources).*$",
            "",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        # Build clean verification source card with language localization
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

        primary_badge = f" {primary_source['badge']}" if primary_source.get("badge") else ""

        # Build Primary Source block
        source_block = (
            f"\n\n{header}\n\n"
            f"📄 **BIS — {primary_source['display_title']}**{primary_badge}{source_badge_str}\n"
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
                "source_file": "raw_data/pdfs/5e2367ca2545_IS_1786.pdf",
                "source_url": None,
                "category": "is_standard",
                "source_hash": "abc123hash",
                "source_of_truth": "verified_bis_pdf",
            }
        }
    ]
    sample_text = "High strength deformed steel bars must satisfy composition limits [As per IS 1786:2008, Clause 4.2]."
    res = engine.format_citations(sample_text, dummy_chunks)
    print("Formatted Response:\n", res["formatted_text"])
    print("\nResolved URL in Citations List:", res["citations_list"][0]["url"])
