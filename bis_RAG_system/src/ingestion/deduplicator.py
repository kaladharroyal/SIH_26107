"""
Deduplication & Canonical Ordering Engine - Phase 1 Data Foundation (deduplicator.py)
Guarantees deterministic SHA-256 derived chunk IDs, deduplication, and canonical stream hashing.
"""

import hashlib
import json
from typing import Dict, Any, List, Set

from metadata import ChunkRecord, generate_canonical_chunk_id


class Deduplicator:
    def __init__(self):
        self.seen_chunk_ids: Set[str] = set()
        self.duplicate_count = 0

    def is_duplicate(self, chunk_id: str) -> bool:
        if chunk_id in self.seen_chunk_ids:
            self.duplicate_count += 1
            return True
        self.seen_chunk_ids.add(chunk_id)
        return False

    @staticmethod
    def canonicalize_chunk(record: Dict[str, Any]) -> str:
        """Serializes record with sorted keys and normalized formatting for canonical comparison."""
        return json.dumps(record, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def compute_canonical_corpus_hash(chunks: List[Dict[str, Any]]) -> str:
        """
        Sorts chunks by chunk_id and computes a global SHA-256 hash over the canonical JSON lines.
        Used for Deterministic Canonical Comparison verification.
        """
        sorted_chunks = sorted(chunks, key=lambda c: c["chunk_id"])
        canonical_stream = "\n".join(Deduplicator.canonicalize_chunk(c) for c in sorted_chunks)
        return hashlib.sha256(canonical_stream.encode("utf-8")).hexdigest()
