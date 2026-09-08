"""
Incremental Indexer CLI & Module (src/ingestion/incremental_indexer.py)
Incrementally adds new PDF documents or chunks to the active FAISS dense vector store
and inverted BM25 index without requiring a complete corpus re-build.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("incremental_indexer")

# Ensure source directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = SRC_DIR.parent
for p in [str(SRC_DIR), str(BASE_DIR), str(SRC_DIR / "ingestion")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval import HybridRetrievalPipeline, BM25Index, DenseVectorStore, DEFAULT_INDEX_DIR, DEFAULT_CHUNKS_PATH


class IncrementalIndexer:
    """
    Manages online and offline incremental updates to the BIS RAG search indexes.
    Parses new PDF files or structured chunks, generates dense embeddings,
    updates postings lists, and serializes updated vector and BM25 store to disk.
    """

    def __init__(
        self,
        index_dir: Optional[Union[str, Path]] = None,
        chunks_path: Optional[Union[str, Path]] = None,
        use_mock_encoder: bool = False,
    ):
        self.index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        self.chunks_path = Path(chunks_path) if chunks_path else DEFAULT_CHUNKS_PATH
        self.use_mock = use_mock_encoder

        log.info(f"Initializing IncrementalIndexer on index_dir: {self.index_dir}")
        self.pipeline = HybridRetrievalPipeline(
            chunks_path=self.chunks_path,
            index_dir=self.index_dir,
            use_mock_encoder=self.use_mock,
        )

    def ingest_chunks(self, new_chunks: List[Dict[str, Any]], append_to_chunks_file: bool = True) -> Dict[str, Any]:
        """Incrementally ingests a list of chunk dictionaries into the active indices."""
        if not new_chunks:
            return {"status": "no_op", "ingested": 0}

        log.info(f"Ingesting {len(new_chunks)} new chunks incrementally...")
        result = self.pipeline.incremental_ingest(
            new_documents=new_chunks,
            save_to_disk=True,
            output_dir=self.index_dir,
        )

        # Optionally append new chunks to processed_chunks.jsonl
        if append_to_chunks_file and self.chunks_path.exists():
            try:
                with open(self.chunks_path, "a", encoding="utf-8") as f:
                    for c in new_chunks:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
                log.info(f"Appended {len(new_chunks)} records to {self.chunks_path.name}")
            except Exception as e:
                log.warning(f"Could not append to chunks file ({e})")

        return {
            "status": "success",
            "ingested": len(new_chunks),
            "total_indexed_documents": result.get("total_documents", 0),
        }

    def ingest_pdf_file(self, pdf_path: Union[str, Path], standard_code: Optional[str] = None) -> Dict[str, Any]:
        """Extracts text/clauses from a physical PDF and incrementally updates search indexes."""
        p_path = Path(pdf_path)
        if not p_path.exists():
            raise FileNotFoundError(f"PDF file not found at {p_path}")

        try:
            from pypdf import PdfReader
            reader = PdfReader(str(p_path))
            total_pages = len(reader.pages)
            if total_pages == 0:
                return {"status": "empty_pdf", "ingested": 0}

            new_chunks = []
            std_name = standard_code or p_path.stem.replace("_", " ").upper()

            # Simple clause chunker
            for page_idx, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                clean_text = " ".join(text.split()).strip()
                if len(clean_text) < 40:
                    continue

                chunk_id = f"inc_{p_path.stem}_p{page_idx}"
                chunk = {
                    "chunk_id": chunk_id,
                    "is_number": std_name if "IS" in std_name else f"IS {std_name}",
                    "clause_number": f"Page {page_idx}",
                    "clause_title": f"{std_name} Section (Page {page_idx})",
                    "text": clean_text,
                    "page_start": page_idx,
                    "page_end": page_idx,
                    "source_file": p_path.name,
                    "category": "is_standard",
                    "source_of_truth": "verified_bis_pdf",
                    "identity_status": "verified",
                }
                new_chunks.append(chunk)

            log.info(f"Extracted {len(new_chunks)} chunks from {p_path.name}")
            return self.ingest_chunks(new_chunks, append_to_chunks_file=True)

        except Exception as e:
            log.error(f"Error parsing PDF {p_path.name}: {e}")
            raise e


def main():
    parser = argparse.ArgumentParser(description="BIS Incremental PDF & Chunk Indexer")
    parser.add_argument("--pdf", type=str, help="Path to single PDF file to incrementally index")
    parser.add_argument("--chunks-json", type=str, help="Path to JSON file containing new chunks to index")
    parser.add_argument("--standard", type=str, default=None, help="Standard name / IS number override")
    parser.add_argument("--index-dir", type=str, default=str(DEFAULT_INDEX_DIR), help="Path to vector_index directory")
    parser.add_argument("--mock", action="store_true", help="Use mock encoder for fast offline testing")

    args = parser.parse_args()
    indexer = IncrementalIndexer(index_dir=args.index_dir, use_mock_encoder=args.mock)

    if args.pdf:
        res = indexer.ingest_pdf_file(args.pdf, standard_code=args.standard)
        print("\n--- Incremental Ingestion Result ---")
        print(json.dumps(res, indent=2))
    elif args.chunks_json:
        with open(args.chunks_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data if isinstance(data, list) else data.get("chunks", [])
        res = indexer.ingest_chunks(chunks)
        print("\n--- Incremental Ingestion Result ---")
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
