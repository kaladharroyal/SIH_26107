"""
Ingestion, Clause-Level PDF Parsing, Normalization, Deduplication, and Validation.
"""

from src.ingestion.pdf_parser import Clause
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.loader import load_chunks, validate_chunk_provenance
from src.ingestion.metadata import ChunkRecord, ALLOWED_CATEGORIES
from src.ingestion.validator import IngestionValidator
from src.ingestion.deduplicator import Deduplicator

__all__ = [
    "Clause",
    "PDFIngestor",
    "load_chunks",
    "validate_chunk_provenance",
    "ChunkRecord",
    "ALLOWED_CATEGORIES",
    "IngestionValidator",
    "Deduplicator",
]
