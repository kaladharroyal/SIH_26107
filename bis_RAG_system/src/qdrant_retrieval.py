"""
Isolated Qdrant + BM25 Hybrid Retrieval Module using Reciprocal Rank Fusion (RRF).
Strictly algorithmic: ZERO hardcoded domain knowledge, mappings, or shortcuts.
All knowledge is retrieved at runtime from Qdrant and BM25 databases.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Resolve project base directory
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

# Load environment configuration
load_dotenv(BASE_DIR / ".env")

from retrieval import BM25Index


class QdrantHybridRetriever:
    """
    Hybrid retriever combining:
    1. Lexical BM25 retrieval over disk index (39,082 chunks).
    2. Dense semantic BGE-M3 vector retrieval over persistent Qdrant Cloud (34,512 points).
    3. Reciprocal Rank Fusion (RRF) for deterministic score merging.
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        bm25_path: Optional[Path] = None,
        model_name: str = "BAAI/bge-m3",
        qdrant_path: Optional[Any] = None,
    ):
        if qdrant_path is not None:
            raise RuntimeError(
                f"Local Qdrant storage path ({qdrant_path}) is DECOMMISSIONED. "
                "Silent fallback to local storage is strictly forbidden. "
                "The application must connect exclusively to Qdrant Cloud via QDRANT_URL and QDRANT_API_KEY."
            )

        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")
        self.collection_name = collection_name or os.environ.get("QDRANT_COLLECTION", "bis_chunks_bge_m3_v1")
        self.bm25_path = Path(bm25_path) if bm25_path else (BASE_DIR / "vector_index" / "bm25_index.pkl")
        self.model_name = model_name

        if not self.qdrant_url:
            raise RuntimeError(
                "QDRANT_URL environment variable is missing. "
                "Qdrant Cloud connection required. Refusing to fall back to local storage."
            )
        if not self.qdrant_api_key:
            raise RuntimeError(
                "QDRANT_API_KEY environment variable is missing. "
                "Qdrant Cloud connection required. Refusing to fall back to local storage."
            )

        # 1. Connect to Qdrant Cloud
        from qdrant_client import QdrantClient

        try:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            col_info = self.client.get_collection(self.collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Qdrant Cloud collection '{self.collection_name}' at {self.qdrant_url}: {e}. "
                "Refusing to fall back to local storage."
            ) from e

        self.qdrant_points_count = col_info.points_count
        self.qdrant_vector_size = col_info.config.params.vectors.size
        c_dist = col_info.config.params.vectors.distance
        self.qdrant_distance = c_dist.value if hasattr(c_dist, "value") else str(c_dist)

        # 2. Load BGE-M3 Neural Embedding Model
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(self.model_name)
        if hasattr(self.encoder, "max_seq_length"):
            self.encoder.max_seq_length = 1024

        # 3. Load BM25 Index
        self.bm25_index = BM25Index.load(self.bm25_path)
        self.bm25_doc_count = self.bm25_index.n_docs

    def encode_query(self, query: str) -> Tuple[np.ndarray, float]:
        """Encodes query into normalized 1024-dim BGE-M3 vector and measures latency."""
        t0 = time.perf_counter()
        vec = self.encoder.encode(query, normalize_embeddings=True)
        t_embed = (time.perf_counter() - t0) * 1000
        vec = np.array(vec, dtype=np.float32).flatten()
        # Verify unit norm
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec, t_embed

    def retrieve_bm25(self, query: str, top_k: int = 10) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes sparse BM25 retrieval and maps to common result schema.
        Returns: (results, latency_ms)
        """
        t0 = time.perf_counter()
        raw_hits = self.bm25_index.search(query, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        results: List[Dict[str, Any]] = []
        for rank, hit in enumerate(raw_hits, 1):
            doc = hit.get("doc", {})
            results.append({
                "rank": rank,
                "chunk_id": doc.get("chunk_id") or hit.get("chunk_id", ""),
                "text": doc.get("text", ""),
                "source_pdf": doc.get("source_file", ""),
                "page": doc.get("page_range", ""),
                "standard": doc.get("is_number"),
                "category": doc.get("category", ""),
                "title": doc.get("clause_title") or doc.get("product") or "",
                "retriever": "bm25",
                "score": float(hit.get("score", 0.0)),
                "source_url": doc.get("source_url", ""),
                "source_of_truth": doc.get("source_of_truth", "verified_bis_pdf"),
            })
        return results, latency_ms

    def retrieve_bge_m3(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Encodes query via BGE-M3 and searches Qdrant collection.
        Returns: (results, embedding_latency_ms, qdrant_latency_ms)
        """
        vec, t_embed = self.encode_query(query)

        t0 = time.perf_counter()
        search_res = self.client.query_points(
            collection_name=self.collection_name,
            query=vec.tolist(),
            limit=top_k,
            with_payload=True,
        )
        t_qdrant = (time.perf_counter() - t0) * 1000

        results: List[Dict[str, Any]] = []
        for rank, pt in enumerate(search_res.points, 1):
            payload = pt.payload or {}
            meta = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
            results.append({
                "rank": rank,
                "chunk_id": payload.get("chunk_id", ""),
                "text": payload.get("text", ""),
                "source_pdf": payload.get("source_pdf", "") or meta.get("source_file", ""),
                "page": payload.get("page", "") or meta.get("page_range", ""),
                "standard": payload.get("standard_number") or meta.get("is_number"),
                "category": payload.get("category", "") or meta.get("category", ""),
                "title": payload.get("document_title", "") or meta.get("clause_title", ""),
                "retriever": "bge_m3",
                "score": float(pt.score),
                "source_url": meta.get("source_url", "") or payload.get("source_url", ""),
                "source_of_truth": meta.get("source_of_truth", "") or payload.get("source_of_truth", "verified_bis_pdf"),
            })
        return results, t_embed, t_qdrant

    @staticmethod
    def fuse_rrf(
        bm25_results: List[Dict[str, Any]],
        bge_results: List[Dict[str, Any]],
        rrf_k: int = 60,
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Merges BM25 and BGE-M3 results using Reciprocal Rank Fusion:
        RRF(chunk) = sum( 1 / (k + rank) ) across present retrievers.
        Deduplicates by chunk_id and preserves provenance and component ranks/scores.
        Returns: (fused_results, latency_ms)
        """
        t0 = time.perf_counter()
        scores: Dict[str, float] = {}
        bm25_ranks: Dict[str, int] = {}
        bge_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}
        bge_scores: Dict[str, float] = {}
        metadata_map: Dict[str, Dict[str, Any]] = {}

        # Process BM25 candidates
        for item in bm25_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
            bm25_ranks[cid] = rank
            bm25_scores[cid] = item["score"]
            if cid not in metadata_map:
                metadata_map[cid] = {
                    "text": item["text"],
                    "source_pdf": item["source_pdf"],
                    "page": item["page"],
                    "standard": item["standard"],
                    "category": item["category"],
                    "title": item["title"],
                    "source_url": item.get("source_url", ""),
                    "source_of_truth": item.get("source_of_truth", "verified_bis_pdf"),
                }

        # Process BGE-M3 candidates
        for item in bge_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
            bge_ranks[cid] = rank
            bge_scores[cid] = item["score"]
            if cid not in metadata_map:
                metadata_map[cid] = {
                    "text": item["text"],
                    "source_pdf": item["source_pdf"],
                    "page": item["page"],
                    "standard": item["standard"],
                    "category": item["category"],
                    "title": item["title"],
                    "source_url": item.get("source_url", ""),
                    "source_of_truth": item.get("source_of_truth", "verified_bis_pdf"),
                }

        # Sort descending by RRF score
        sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

        fused: List[Dict[str, Any]] = []
        for hybrid_rank, cid in enumerate(sorted_cids[:top_k], 1):
            meta = metadata_map[cid]
            fused.append({
                "rank": hybrid_rank,
                "chunk_id": cid,
                "rrf_score": float(scores[cid]),
                "bm25_rank": bm25_ranks.get(cid),
                "bge_rank": bge_ranks.get(cid),
                "bm25_score": bm25_scores.get(cid),
                "bge_score": bge_scores.get(cid),
                "text": meta["text"],
                "source_pdf": meta["source_pdf"],
                "page": meta["page"],
                "standard": meta["standard"],
                "category": meta["category"],
                "title": meta["title"],
                "source_url": meta.get("source_url", ""),
                "source_of_truth": meta.get("source_of_truth", "verified_bis_pdf"),
            })

        latency_ms = (time.perf_counter() - t0) * 1000
        return fused, latency_ms

    def retrieve_hybrid(
        self, query: str, top_k: int = 10, rrf_k: int = 60
    ) -> Dict[str, Any]:
        """
        Executes end-to-end hybrid retrieval:
        1. BM25 search
        2. BGE-M3 embedding + Qdrant search
        3. RRF score fusion and deduplication
        Returns complete evidence and latency breakdown.
        """
        t_total_start = time.perf_counter()

        bm25_hits, bm25_ms = self.retrieve_bm25(query, top_k=top_k)
        bge_hits, embed_ms, qdrant_ms = self.retrieve_bge_m3(query, top_k=top_k)
        fused_hits, rrf_ms = self.fuse_rrf(bm25_hits, bge_hits, rrf_k=rrf_k, top_k=top_k)

        total_ms = (time.perf_counter() - t_total_start) * 1000

        return {
            "query": query,
            "bm25_results": bm25_hits,
            "bge_results": bge_hits,
            "hybrid_results": fused_hits,
            "latencies": {
                "bm25_ms": round(bm25_ms, 2),
                "bge_embed_ms": round(embed_ms, 2),
                "qdrant_search_ms": round(qdrant_ms, 2),
                "bge_total_ms": round(embed_ms + qdrant_ms, 2),
                "rrf_ms": round(rrf_ms, 3),
                "total_hybrid_ms": round(total_ms, 2),
            },
        }

    def close(self):
        """Closes Qdrant client connection cleanly."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
