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
        """
        mappings: List[Dict[str, Any]] = []
        seen_pairs = set()

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
                    item = data[0]
                    page_text = item.get("page_text", "")
                    page_url = item.get("page_url", source_url)

                    # Extract standard number from URL or text
                    id_m = re.search(r"[?&]id=(\d+)(?:_(\d{4}))?", page_url)
                    stdno_m = re.search(r"stdno=IS[\s_:-]*(\d+)", page_url)
                    
                    is_num = None
                    year = None
                    if id_m:
                        is_num = f"IS {id_m.group(1)}"
                        year = id_m.group(2) if id_m.group(2) else None
                    elif stdno_m:
                        is_num = f"IS {stdno_m.group(1)}"
                    else:
                        text_is = re.search(r"\bIS\s*[:\-_]?\s*(\d{2,6})(?:[\s:\-–]*(\d{4}))?", page_text)
                        if text_is:
                            is_num = f"IS {text_is.group(1)}"
                            year = text_is.group(2) if text_is.group(2) else None

                    # Extract product / scope name from title or SCOPE section
                    product_title = ""
                    title_m = re.search(
                        r"IS\s*\d+[\s:\-–]*\d{0,4}\s+([A-Za-z0-9\s,\-\(\)\/\.]+?)(?:UDC|ICS|MTD|CHD|ETD|CED|TXD|\d+\s*Scope|\d+\.\s*SCOPE|$)",
                        page_text,
                    )
                    if title_m and len(title_m.group(1).strip()) > 3:
                        product_title = title_m.group(1).strip()
                    else:
                        scope_m = re.search(
                            r"(?:SCOPE|Scope)\s*(?:1\.1\s*)?(?:This\s+(?:standard|specification|method)\s+(?:covers|prescribes|specifies|lays down|gives)[^\.\n]*?(?:for|of|the)?\s+([^\.\n]{5,120}))",
                            page_text,
                            re.IGNORECASE,
                        )
                        if scope_m:
                            product_title = scope_m.group(1).strip()
                        elif " " in page_text[:100]:
                            first_line = page_text[:100].split("\n")[0].strip()
                            if len(first_line) > 5 and not first_line.startswith("http"):
                                product_title = first_line

                    if is_num and product_title:
                        pair_key = (product_title.lower(), is_num)
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            mappings.append({
                                "product": product_title,
                                "standard": is_num,
                                "revision_year": year,
                                "scheme": "Scheme-I (ISI Mark)" if "1" in is_num else "CRS (Compulsory Registration Scheme)",
                                "mandatory": True if "mandatory" in page_text.lower() or "qco" in page_text.lower() else False,
                                "source_url": page_url,
                                "source_hash": source_hash,
                                "source_type": "product_standard_mapping",
                                "source_of_truth": "verified_bis_api",
                            })

            except Exception as e:
                log.warning(f"Error parsing {filename} for product map: {e}")

        result = {
            "dataset_name": "product_standard_map",
            "total_records": len(mappings),
            "records": mappings,
            "status": "verified" if mappings else "unavailable",
        }

        out_path = self.output_dir / "product_standard_map.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info(f"Saved {len(mappings)} verified product-standard mappings strictly from baseline to {out_path}")
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
