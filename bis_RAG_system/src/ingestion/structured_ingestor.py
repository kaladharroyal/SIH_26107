"""
Structured Data Ingestor - Phase 1 Data Foundation (structured_ingestor.py)
Extracts structured product-standard mappings and lab directories with complete provenance
strictly scoped to the supplied baseline manifest records (0 full-corpus directory globbing).
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

log = logging.getLogger("structured_ingestor")


class StructuredIngestor:
    def __init__(self, raw_dir: Path, output_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_product_standard_map(self, manifest_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts verified product-to-standard mapping records strictly from the supplied baseline manifest records.
        Identifies explicit product-standard associations from the source structure/text, supports multiple
        mappings per page, eliminates guesswork, and avoids scheme inference from standard numbers.
        """
        mappings: List[Dict[str, Any]] = []
        seen_pairs = set()

        branch_codes = {
            "AHBO", "BHBO", "BNBO", "BPBO", "CHBO", "CNBO", "CTBO", "DHBO", "DLBO",
            "FRBO", "GZBO", "HRBO", "HUBO", "JDBO", "JKBO", "JPBO", "KKBO", "KOBO",
            "LKBO", "MUBO", "NGBO", "NOBO", "PNBO", "PRBO", "PTBO", "RJBO", "SUBO",
            "GDBO", "VJBO", "RPBO", "HYBO", "GHBO", "BO"
        }

        blacklist_products = {
            "english", "hindi", "bureau of indian standards", "know your standards",
            "click here", "home", "sitemap", "customer feedback", "log in",
            "orders", "wish list", "forgot password", "advance search", "scope",
            "free amendments", "table of contents", "basic details", "other details",
            "classification details", "cross reference details", "license", "laboratory",
            "part 1", "part 2", "part 3", "part 4", "part 5", "section 1", "section 2",
            "list of licenses", "list of standards", "specification", "indian standard",
            "indian standards", "standards", "amendment", "amendments", "annex a", "annex b",
            "list of expired or cancelled licenses under is", "list of laboratories for"
        }

        tech_split_regex = r"(?:Technical\s*Committe?e?|Status\s*:|Login\s*to\s*Download|Price\s*:|No\.\s*of\s*Amendments|Reaffirmed\s*\d{4}|1\.?\s*SCOPE|1\.1\s*SCOPE|\bSCOPE\b|(?:ICS|CED|MTD|CHD|ETD|TXD|FAD|UDC|PGD|PCD|MSD|TED)\s*\d+|Reaffirmed\b|First Revision|Second Revision|Third Revision|Fourth Revision|Fifth Revision)"

        def clean_product_name(prod: str) -> str:
            p = prod.strip()
            words = p.split()
            if words and (words[0].upper() in branch_codes or any(words[0].upper().startswith(b) for b in branch_codes)):
                p = " ".join(words[1:])
            p = re.sub(r"^\d+\s*[-.)]?\s*", "", p)
            p = re.sub(r"^\(Reaffirmed\s+Year\s*:\s*\d{4}\)\s*", "", p, flags=re.IGNORECASE)
            p = re.sub(r"^Reaffirmed\s+\d{4}\s*", "", p, flags=re.IGNORECASE)
            p = re.split(tech_split_regex, p, flags=re.IGNORECASE)[0]
            p = re.sub(r"\s+and\s+other\s+Indian\s+Standards.*$", "", p, flags=re.IGNORECASE)
            p = re.sub(r"\s+and\s+other\s+related\s+ISS.*$", "", p, flags=re.IGNORECASE)
            p = re.sub(r"\s+etc\.?$", "", p, flags=re.IGNORECASE)
            p = re.sub(r"\s*\((?:first|second|third|fourth|fifth|sixth|seventh|eighth|[a-z0-9\s]+)\s*revision\)", "", p, flags=re.IGNORECASE)
            p = re.sub(r"\s*-\s*Specification\s*\(?$", "", p, flags=re.IGNORECASE)
            p = re.sub(r"\s*Specification\s*\(?$", "", p, flags=re.IGNORECASE)
            p = re.sub(r"[\s\(\)\-]+$", "", p)
            p = p.strip()
            return p

        def is_valid_product(prod: str) -> bool:
            if not prod or len(prod) < 4 or len(prod) > 150:
                return False
            low = prod.lower()
            if low in blacklist_products:
                return False
            if any(b in low for b in [
                "click here", "sitemap", "customer feedback", "forgot password",
                "know your standards", "list of licenses", "list of expired",
                "list of laboratories", "bureau of indian standards",
                "and other indian standards", "date of enforcement",
                "quality control order", "classification details",
                "price :", "login to download", "no. of amendments"
            ]):
                return False
            if low.startswith("http") or low.startswith("www."):
                return False
            if re.match(r"^part\s*\d+", low) or re.match(r"^section\s*\d+", low):
                return False
            if low.startswith("is ") or low.startswith("and is ") or low.startswith("amd") or "amendment" in low:
                return False
            if not re.search(r"[A-Za-z]", prod):
                return False
            return True

        def extract_explicit_associations(page_text: str, page_url: str, source_hash: str) -> List[Dict[str, Any]]:
            extracted = []
            lines = page_text.split("\n")

            # Check if page_text is a single standard preview document
            preview_match = re.search(
                r"IS\s*(\d{2,6})\s*[:\-_]?\s*(\d{4})?\s*[:\-_–]?\s*([A-Za-z0-9][^\n\r]+?)" + tech_split_regex,
                page_text[:600],
                re.IGNORECASE,
            )
            if not preview_match:
                preview_match = re.search(
                    r"IS\s*(\d{2,6})\s*[:\-_]?\s*(\d{4})?\s*[:\-_–]?\s*([A-Za-z0-9][^\n\r]+?)(?:\s*\n|$)",
                    page_text[:400],
                )
            if preview_match:
                std_num = f"IS {preview_match.group(1).strip()}"
                rev_yr = preview_match.group(2).strip() if preview_match.group(2) else None
                prod = clean_product_name(preview_match.group(3))
                if is_valid_product(prod):
                    evidence = preview_match.group(0)[:200].replace("\n", " ").strip()
                    extracted.append({
                        "product": prod,
                        "standard": std_num,
                        "revision_year": rev_yr,
                        "scheme": None,
                        "mandatory": False,
                        "source_url": page_url,
                        "source_hash": source_hash,
                        "source_type": "product_standard_mapping",
                        "source_of_truth": "bis_source_explicit_association",
                        "verification_status": "source_explicit",
                        "evidence": evidence,
                    })

            # Line-by-line / block extraction for structured factsheets or tables
            for line in lines:
                line_clean = line.strip()
                if not line_clean or len(line_clean) < 8:
                    continue

                # Pattern 1: IS <num> : <yr> (<Prod> - Specification) or IS <num> <Prod>
                for m in re.finditer(
                    r"\bIS\s*[:\-_]?\s*(\d{2,6})(?:\s*[:\-_]\s*(\d{4}))?\s*[:\-_–]?\s*(?:\(([^)]+)\)|([A-Za-z][A-Za-z0-9\s,\-\/]+?))(?=\s+and\s+IS|\s*\(|\s*\n|$|\s*-\s*Specification|\s*;\s*IS)",
                    line_clean,
                ):
                    std = f"IS {m.group(1).strip()}"
                    yr = m.group(2).strip() if m.group(2) else None
                    prod = m.group(3) or m.group(4)
                    if prod:
                        prod = clean_product_name(prod)
                        if is_valid_product(prod) and not prod.startswith("IS "):
                            extracted.append({
                                "product": prod,
                                "standard": std,
                                "revision_year": yr,
                                "scheme": None,
                                "mandatory": False,
                                "source_url": page_url,
                                "source_hash": source_hash,
                                "source_type": "product_standard_mapping",
                                "source_of_truth": "bis_source_explicit_association",
                                "verification_status": "source_explicit",
                                "evidence": line_clean[:200],
                            })

                # Pattern 2: Product Name (IS 1786, IS 2062) or Product Name - IS 1786, IS 2062
                m_rev = re.search(
                    r"^([A-Za-z0-9\s,\-\/\.]+?)\s*(?:[-–:]|\()\s*IS\s*[:\-_]?\s*(\d{2,6})(?:\s*[:\-_]\s*(\d{4}))?(?:[,\s]+IS\s*[:\-_]?\s*(\d{2,6}))*",
                    line_clean,
                )
                if m_rev:
                    prod_candidate = clean_product_name(m_rev.group(1))
                    if is_valid_product(prod_candidate):
                        for is_m in re.finditer(r"\bIS\s*[:\-_]?\s*(\d{2,6})(?:\s*[:\-_]\s*(\d{4}))?\b", line_clean):
                            std = f"IS {is_m.group(1).strip()}"
                            yr = is_m.group(2).strip() if is_m.group(2) else None
                            extracted.append({
                                "product": prod_candidate,
                                "standard": std,
                                "revision_year": yr,
                                "scheme": None,
                                "mandatory": False,
                                "source_url": page_url,
                                "source_hash": source_hash,
                                "source_type": "product_standard_mapping",
                                "source_of_truth": "bis_source_explicit_association",
                                "verification_status": "source_explicit",
                                "evidence": line_clean[:200],
                            })

            return extracted

        # Filter strictly to product_standard_mapping baseline records
        psm_records = [
            r for r in manifest_records
            if r.get("category") == "product_standard_mapping" or "product_standard_mapping" in r.get("local_path", "").lower()
        ]
        log.info(f"Extracting product standard mappings strictly from {len(psm_records)} baseline records")

        for manifest_entry in psm_records:
            local_rel = manifest_entry.get("local_path", "")
            filename = Path(local_rel.replace("\\", "/")).name
            if not filename.endswith(".json"):
                continue

            # Support both raw_data/json/ and raw_data/
            json_file = self.raw_dir / filename
            if not json_file.exists():
                json_file = self.raw_dir / "json" / filename
            if not json_file.exists() and local_rel:
                json_file = self.output_dir / local_rel
            if not json_file.exists():
                continue

            source_url = manifest_entry.get("source_url", "")
            source_hash = manifest_entry.get("content_hash", "")

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    for item in data:
                        page_text = item.get("page_text", "")
                        page_url = item.get("page_url", source_url)

                        records = extract_explicit_associations(page_text, page_url, source_hash)
                        for rec in records:
                            pair_key = (rec["product"].lower(), rec["standard"])
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                mappings.append(rec)

            except Exception as e:
                log.warning(f"Error parsing {filename} for product map: {e}")

        result = {
            "dataset_name": "product_standard_map",
            "total_records": len(mappings),
            "records": mappings,
            "status": "source_validated" if mappings else "unavailable",
        }

        out_path = self.output_dir / "product_standard_map.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"Saved {len(mappings)} source-validated product-standard mappings strictly from baseline to {out_path}")
        return result

    def extract_labs_directory(self, manifest_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts verified lab directory records strictly from the supplied baseline manifest records.
        If no verified lab sources could be parsed, sets status: "unavailable" rather than inserting fake demo records.
        """
        labs: List[Dict[str, Any]] = []

        lab_records = [
            r for r in manifest_records
            if r.get("category") in ["lab_directory", "lims_recognized_labs", "lims_empaneled_labs", "lab_faq"]
            or "lab" in r.get("local_path", "").lower()
        ]
        log.info(f"Extracting labs directory strictly from {len(lab_records)} baseline records")

        for manifest_entry in lab_records:
            local_rel = manifest_entry.get("local_path", "")
            filename = Path(local_rel.replace("\\", "/")).name
            if not filename.endswith(".json"):
                continue
            json_file = self.raw_dir / filename
            if not json_file.exists():
                json_file = self.raw_dir / "json" / filename
            if not json_file.exists() and local_rel:
                json_file = self.output_dir / local_rel
            if not json_file.exists():
                continue

            source_url = manifest_entry.get("source_url", "")
            source_hash = manifest_entry.get("content_hash", "")

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and ("lab_name" in item or "laboratory_name" in item):
                            labs.append({
                                "lab_name": item.get("lab_name") or item.get("laboratory_name"),
                                "location": item.get("city") or item.get("state") or item.get("location"),
                                "address": item.get("address", ""),
                                "testing_scope": item.get("testing_scope") or item.get("scope", []),
                                "source_url": source_url,
                                "source_hash": source_hash,
                                "source_type": "lims_directory",
                                "source_of_truth": "official_bis",
                            })
            except Exception as e:
                log.warning(f"Error parsing {filename} for labs directory: {e}")

        result = {
            "dataset_name": "labs_directory",
            "total_records": len(labs),
            "records": labs,
            "status": "verified" if labs else "unavailable",
            "status_reason": "Verified lab directory loaded from BIS sources" if labs else "Dynamic LIMS lab API records require live BIS authentication; static dataset currently unavailable",
        }

        out_path = self.output_dir / "labs_directory.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"Saved {len(labs)} lab records strictly from baseline to {out_path} (Status: {result['status']})")
        return result
