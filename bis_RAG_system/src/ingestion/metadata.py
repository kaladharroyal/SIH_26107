"""
Metadata Schema & Controlled Taxonomy - Phase 1 Data Foundation (metadata.py)
Defines standardized chunk record structures, controlled category taxonomy, and canonical hashing.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import hashlib

ALLOWED_CATEGORIES = {
    "is_standard",
    "qco_order",
    "amendment",
    "faq",
    "product_standard_mapping",
    "certification",
    "hallmarking",
    "consumer",
    "lab_directory",
    "licensing_fees",
    "annual_report",
    "general_policy",
    "other",
}

ALLOWED_PRODUCTION_SOURCE_OF_TRUTH = {
    "official_bis",
    "verified_bis_pdf",
    "verified_bis_html",
    "verified_bis_api",
    "derived_from_bis",
}

ALLOWED_IDENTITY_STATUS = {
    "verified",
    "non_standard",
    "unknown",
}


@dataclass
class ChunkRecord:
    chunk_id: str
    is_number: Optional[str]
    part: Optional[str]
    revision_year: Optional[str]
    identity_status: str
    identity_reason: str
    clause_number: str
    clause_title: Optional[str]
    text: str
    category: str
    content_type: str
    source_url: str
    source_file: str
    page_range: str
    source_hash: str
    source_of_truth: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_canonical_chunk_id(source_hash: str, clause_or_record_id: str, text: str) -> str:
    """
    Computes a canonical SHA-256 derived chunk ID using strict unit-separator delimiter (\x1f):
    chunk_id_input = source_hash + "\x1f" + clause_or_record_identity + "\x1f" + normalized_text
    """
    norm_text = " ".join(text.strip().split())
    chunk_id_input = f"{source_hash}\x1f{clause_or_record_id}\x1f{norm_text}"
    return hashlib.sha256(chunk_id_input.encode("utf-8")).hexdigest()
