"""
BIS AI Assistant - Centralized Configuration & Path Manager (src/config.py)
Provides unified path resolution, environment loading, and runtime settings.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Resolve repository root directory (where app.py and requirements.txt live)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")

# Primary Directories
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data Subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
CLASSIFIED_DATA_DIR = DATA_DIR / "classified"
VECTOR_INDEX_DIR = DATA_DIR / "vector_index"

# Primary Data Files (with fallback discovery)
def resolve_data_file(filename: str, default_subdir: Optional[Path] = None) -> Path:
    """
    Resolves a data file path checking:
    1. data/<filename>
    2. data/<default_subdir>/<filename>
    3. root/<filename>
    4. fallback to data/<filename>
    """
    candidates = [
        DATA_DIR / filename,
        PROJECT_ROOT / filename,
    ]
    if default_subdir:
        candidates.insert(1, default_subdir / filename)
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DATA_DIR / filename


# Default Data File Paths
PRODUCT_STANDARD_MAP_PATH = resolve_data_file("product_standard_map.json")
LABS_DIRECTORY_PATH = resolve_data_file("labs_directory.json")
PROCESSED_CHUNKS_PATH = resolve_data_file("processed_chunks.jsonl")
DATA_QUALITY_REPORT_PATH = resolve_data_file("phase1_data_quality_report.json")
FEEDBACK_DB_PATH = resolve_data_file("bis_rag_telemetry.db")

# Service Configuration
SERVER_HOST = os.getenv("HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("PORT", "8000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
DEMO_MIN_RESPONSE_SECONDS = float(os.getenv("DEMO_MIN_RESPONSE_SECONDS", "0.0"))

# LLM & Embedding Providers
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Qdrant Vector Cloud Settings
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bis_standards_corpus")

# CORS Origins
def get_cors_origins() -> List[str]:
    cors_env = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
    )
    if cors_env.strip() == "*":
        return ["*"]
    return [o.strip() for o in cors_env.split(",") if o.strip()]
