"""
Loader — Native JSONL Chunk Utilities
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("loader")


def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Loads and returns verified JSONL document chunks."""
    path = Path(chunks_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    log.info(f"Loaded {len(chunks)} chunks from {path.name}")
    return chunks


def validate_chunk_provenance(chunk: Dict[str, Any]) -> bool:
    """Verifies that a chunk contains mandatory provenance metadata fields."""
    required_keys = ["chunk_id", "text", "category", "source_file", "source_url"]
    return all(k in chunk and chunk[k] for k in required_keys)
