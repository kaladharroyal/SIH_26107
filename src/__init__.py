"""
Bureau of Indian Standards (BIS) AI Assistant - Modular Package Root
Clean, minimal, and modular architecture.
"""

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    STATIC_DIR,
    DOCS_DIR,
    RAW_DATA_DIR,
    CLASSIFIED_DATA_DIR,
    VECTOR_INDEX_DIR,
    resolve_data_file,
    PRODUCT_STANDARD_MAP_PATH,
    LABS_DIRECTORY_PATH,
    PROCESSED_CHUNKS_PATH,
    FEEDBACK_DB_PATH,
)

# Core Orchestration & Pipeline
from src.core.rag_pipeline import BISRAGPipeline
from src.core.router import QueryIntentRouter
from src.core.generator import GroundedGenerator
from src.core.guardrails import GuardrailGate
from src.core.citation_engine import CitationEngine
from src.core.feedback_logger import FeedbackLogger
from src.core.multilingual import MultilingualHandler
from src.core.translation_engine import TranslationEngine

# Retrieval Engines
from src.retrieval.retrieval import HybridRetrievalPipeline, BM25Index
from src.retrieval.qdrant_retrieval import QdrantHybridRetriever

# Domain Services
from src.services.product_recommender import ProductRecommender
from src.services.lab_locator import LabLocator
from src.services.scheme_walkthrough import SchemeWalkthroughGuide
from src.services.consumer_complaint import ConsumerComplaintHandler

# Ingestion & Parsing
from src.ingestion.pdf_parser import Clause
from src.ingestion.pdf_ingestor import PDFIngestor
from src.ingestion.loader import load_chunks, validate_chunk_provenance

__all__ = [
    # Core
    "BISRAGPipeline",
    "QueryIntentRouter",
    "GroundedGenerator",
    "GuardrailGate",
    "CitationEngine",
    "FeedbackLogger",
    "MultilingualHandler",
    "TranslationEngine",
    # Retrieval
    "HybridRetrievalPipeline",
    "BM25Index",
    "QdrantHybridRetriever",
    # Services
    "ProductRecommender",
    "LabLocator",
    "SchemeWalkthroughGuide",
    "ConsumerComplaintHandler",
    # Ingestion
    "Clause",
    "PDFIngestor",
    "load_chunks",
    "validate_chunk_provenance",
    # Config
    "PROJECT_ROOT",
    "DATA_DIR",
    "STATIC_DIR",
    "DOCS_DIR",
    "RAW_DATA_DIR",
    "CLASSIFIED_DATA_DIR",
    "VECTOR_INDEX_DIR",
    "resolve_data_file",
    "PRODUCT_STANDARD_MAP_PATH",
    "LABS_DIRECTORY_PATH",
    "PROCESSED_CHUNKS_PATH",
    "FEEDBACK_DB_PATH",
]
