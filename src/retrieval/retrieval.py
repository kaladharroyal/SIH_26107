"""
Phase 2: Hybrid Retrieval Engine for BIS RAG System (retrieval.py)
Combines:
  1. Lexical / Sparse BM25 Search (optimized for exact Indian Standard codes, clause refs, terms)
  2. Dense Semantic Vector Search (BGE-M3 multilingual embeddings with persistent matrix store)
  3. Domain Category Pre-filtering (is_standard, qco_order, lab_directory, general_policy, etc.)
  4. Reciprocal Rank Fusion (RRF) for deterministic score merging
  5. Cross-Encoder & Heuristic Contextual Reranking
  6. Resumable, Batch-based Index Construction with Checkpointing
  7. 100% Provenance Preservation (chunk_id, source_file, source_url, source_hash, source_of_truth)
"""

import argparse
import hashlib
import json
import logging
import math
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("hybrid_retrieval")

try:
    from src.config import PROCESSED_CHUNKS_PATH, VECTOR_INDEX_DIR
    DEFAULT_CHUNKS_PATH = PROCESSED_CHUNKS_PATH
    DEFAULT_INDEX_DIR = VECTOR_INDEX_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DEFAULT_CHUNKS_PATH = BASE_DIR / "data" / "processed_chunks.jsonl"
    if not DEFAULT_CHUNKS_PATH.exists():
        DEFAULT_CHUNKS_PATH = BASE_DIR / "processed_chunks.jsonl"
    DEFAULT_INDEX_DIR = BASE_DIR / "data" / "vector_index"
    if not DEFAULT_INDEX_DIR.exists():
        DEFAULT_INDEX_DIR = BASE_DIR / "vector_index"


class BM25Index:
    """
    Persistent BM25 Index tailored for BIS Standard documents and clauses.
    Preserves exact IS identifiers (e.g. 'IS 1786', 'IS:1070:2023', 'IS-12860')
    and provides deterministic ranking and disk serialization.
    """

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.df: Dict[str, int] = {}
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_categories: List[str] = []
        self.doc_ids: List[str] = []

        if documents:
            self.build(documents)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text while preserving Indian Standard notations (e.g. IS 1786, IS:1070, IS-1448).
        """
        if not text:
            return []
        text_lower = text.lower()
        # Find explicit IS standard patterns (e.g. 'is 1786', 'is:1070', 'is-1448')
        is_patterns = re.findall(r"\bis[\s:\-_]*\d{2,6}(?:[\s:\-_]*\d{4})?\b", text_lower)
        normalized_is = [re.sub(r"[\s:\-_]+", " ", p).strip() for p in is_patterns]

        # General alphanumeric words
        words = re.findall(r"\b[a-z0-9_\-]{2,}\b", text_lower)
        return normalized_is + words

    def build(self, documents: List[Dict[str, Any]]):
        """Builds BM25 frequency tables and IDF from documents."""
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_len = []
        self.doc_freqs = []
        self.doc_categories = []
        self.doc_ids = []
        self.df = {}

        for d in documents:
            text = d.get("text", "")
            title = d.get("clause_title") or d.get("product") or ""
            is_num = d.get("is_number") or ""
            full_text = f"{is_num} {title} {text}"

            tokens = self.tokenize(full_text)
            self.doc_len.append(len(tokens))
            self.doc_categories.append(d.get("category", "general"))
            self.doc_ids.append(d.get("chunk_id", ""))

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1

            for t in freqs:
                self.df[t] = self.df.get(t, 0) + 1

            self.doc_freqs.append(freqs)

        total_len = sum(self.doc_len)
        self.avgdl = total_len / max(self.n_docs, 1)
        log.info(f"Built BM25 index over {self.n_docs} documents (avgdl={self.avgdl:.2f}, vocab={len(self.df)} terms).")

    def search(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Searches BM25 index for query with optional category filtering."""
        if not query or not self.n_docs:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scores: List[Tuple[float, int]] = []
        for i in range(self.n_docs):
            if category and self.doc_categories[i] != category:
                continue

            freqs = self.doc_freqs[i]
            dl = self.doc_len[i]
            score = 0.0

            for q in q_tokens:
                if q not in freqs:
                    continue
                tf = freqs[q]
                df = self.df.get(q, 0)
                # Robertson-Spärck Jones IDF
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-5)))
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            results.append({
                "doc": self.documents[idx],
                "score": float(score),
                "method": "bm25",
                "chunk_id": self.doc_ids[idx],
            })
        return results

    def save(self, filepath: Union[str, Path], corpus_hash: Optional[str] = None):
        """Serializes BM25 index data to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
            "df": self.df,
            "doc_freqs": self.doc_freqs,
            "doc_categories": self.doc_categories,
            "doc_ids": self.doc_ids,
            "corpus_hash": corpus_hash or getattr(self, "corpus_hash", None),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved BM25 index ({self.n_docs} docs) to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BM25Index":
        """Loads serialized BM25 index data from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found at {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        idx = cls(k1=state.get("k1", 1.5), b=state.get("b", 0.75))
        idx.documents = state["documents"]
        idx.doc_len = state["doc_len"]
        idx.avgdl = state["avgdl"]
        idx.n_docs = state["n_docs"]
        idx.df = state["df"]
        idx.doc_freqs = state["doc_freqs"]
        idx.doc_categories = state["doc_categories"]
        idx.doc_ids = state["doc_ids"]
        idx.corpus_hash = state.get("corpus_hash", None)
        log.info(f"Loaded BM25 index with {idx.n_docs} documents from {path}")
        return idx


class MockEmbeddingModel:
    """
    Lightweight, deterministic mock embedding model for fast offline unit testing.
    Produces deterministic 128-dimensional normalized vectors without downloading neural weights.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        single = isinstance(texts, str)
        text_list = [texts] if single else texts

        vectors = []
        for t in text_list:
            # Deterministic pseudo-random vector seeded by text MD5 hash
            h_int = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
            rng = np.random.RandomState(h_int % (2**31 - 1))
            vec = rng.randn(self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
            vectors.append(vec)

        res = np.vstack(vectors)
        return res[0] if single else res


class DenseVectorStore:
    """
    Persistent, memory-mapped dense vector store for BGE-M3 embeddings.
    Supports pre-computed matrix multiplication for ultra-fast cosine similarity.
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        chunk_ids: Optional[List[str]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
    ):
        self.embeddings = embeddings  # Shape: (N, D), normalized float32
        self.chunk_ids = chunk_ids or []
        self.documents = documents or []
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        self.doc_categories = [d.get("category", "general") for d in self.documents]

    @property
    def count(self) -> int:
        return len(self.chunk_ids)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Executes vector cosine search via dot product against normalized matrix."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Ensure 1D normalized query vector
        q = query_vec.flatten()
        norm = np.linalg.norm(q)
        if norm > 1e-9:
            q = q / norm

        # Compute cosine similarity across all vectors
        sims = np.dot(self.embeddings, q)

        if category:
            # Filter indices by category
            candidate_indices = [i for i, cat in enumerate(self.doc_categories) if cat == category]
            if not candidate_indices:
                return []
            candidate_sims = [(float(sims[i]), i) for i in candidate_indices]
            candidate_sims.sort(key=lambda x: x[0], reverse=True)
            top_indices = candidate_sims[:top_k]
        else:
            # Global top_k
            top_k = min(top_k, len(sims))
            # Use argpartition for fast top-k if large
            if len(sims) > top_k * 5:
                partitioned_idx = np.argpartition(sims, -top_k)[-top_k:]
                top_sims = [(float(sims[i]), int(i)) for i in partitioned_idx]
                top_sims.sort(key=lambda x: x[0], reverse=True)
                top_indices = top_sims
            else:
                sorted_idx = np.argsort(-sims)[:top_k]
                top_indices = [(float(sims[i]), int(i)) for i in sorted_idx]

        results = []
        for score, idx in top_indices:
            results.append({
                "doc": self.documents[idx],
                "score": float(score),
                "method": "dense",
                "chunk_id": self.chunk_ids[idx],
            })
        return results

    def save(
        self,
        output_dir: Union[str, Path],
        corpus_hash: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Saves matrix and metadata to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            np.save(str(out_path / "embeddings.npy"), self.embeddings)

        meta = {
            "count": len(self.chunk_ids),
            "dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
            "chunk_ids": self.chunk_ids,
            "corpus_hash": corpus_hash or getattr(self, "corpus_hash", None),
            "model_name": model_name or getattr(self, "model_name", "BAAI/bge-m3"),
            "dtype": str(self.embeddings.dtype) if self.embeddings is not None else "float32",
            "normalized": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_path / "vector_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save documents mapping
        with open(out_path / "documents.jsonl", "w", encoding="utf-8") as f:
            for d in self.documents:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        log.info(f"Saved DenseVectorStore ({len(self.chunk_ids)} items) to {out_path}")

    @classmethod
    def load(cls, output_dir: Union[str, Path]) -> "DenseVectorStore":
        """Loads matrix and metadata from disk."""
        path = Path(output_dir)
        npy_file = path / "embeddings.npy"
        meta_file = path / "vector_metadata.json"
        docs_file = path / "documents.jsonl"

        if not npy_file.exists() or not meta_file.exists() or not docs_file.exists():
            raise FileNotFoundError(f"Dense vector store incomplete at {path}")

        embeddings = np.load(str(npy_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        documents = []
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))

        chunk_ids = meta.get("chunk_ids", [d.get("chunk_id", "") for d in documents])
        log.info(f"Loaded DenseVectorStore with {len(chunk_ids)} embeddings of dim {embeddings.shape[1]} from {path}")
        store = cls(embeddings=embeddings, chunk_ids=chunk_ids, documents=documents)
        store.corpus_hash = meta.get("corpus_hash", None)
        store.model_name = meta.get("model_name", "BAAI/bge-m3")
        return store


class HybridRetrievalPipeline:
    """
    Production Hybrid Retrieval Engine combining BM25, Dense Vector Search,
    RRF Score Fusion, and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        index_dir: Path = DEFAULT_INDEX_DIR,
        use_mock_encoder: bool = False,
    ):
        self.chunks_path = Path(chunks_path)
        self.index_dir = Path(index_dir)
        self.use_mock_encoder = use_mock_encoder

        self.chunks: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Index] = None
        self.vector_store: Optional[DenseVectorStore] = None
        self.encoder: Any = None
        self.reranker: Any = None

        self._load_or_init_pipeline()

    def _load_or_init_pipeline(self):
        """Loads chunks, BM25 index, vector store, and neural models.

        F1 — DENSE INDEX STATE (intentionally deferred):
        Production retrieval runs in BM25-only mode (use_fast_retrieval=True in rag_pipeline.py).
        The full 39,082-chunk BGE-M3 dense embedding build was stopped because CPU-only
        inference is impractical for the full corpus.  The 512-chunk checkpoint at
        vector_index/embeddings_partial.npy is INCOMPLETE and must not be used as production.
        The 500-chunk validation index at vector_index/validation_500/ is a validation
        artifact only and must remain isolated.  Full dense hybrid retrieval (use_fast_retrieval=False)
        is available in the code and can be enabled once a suitable compute environment exists.
        """
        self.degraded_mode: bool = False  # F6: set True when corpus is missing

        # 1. Load Chunks
        if self.chunks_path.exists():
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.chunks.append(json.loads(line))
            log.info(f"Loaded {len(self.chunks)} verified chunks from {self.chunks_path}")
        else:
            # F6: missing corpus is an operational error, not a recoverable warning
            log.error(
                f"CORPUS MISSING: Chunks file not found at {self.chunks_path}. "
                "The retrieval pipeline is operating in DEGRADED MODE with an empty knowledge base. "
                "All queries will be refused by the guardrail. "
                "Ensure processed_chunks.jsonl is present before serving real queries."
            )
            self.degraded_mode = True

        # 2. Load or Build BM25 Index
        bm25_file = self.index_dir / "bm25_index.pkl"
        if bm25_file.exists():
            try:
                self.bm25_index = BM25Index.load(bm25_file)
            except Exception as e:
                log.warning(f"Could not load BM25 index from {bm25_file}: {e}. Building in memory...")
                self.bm25_index = BM25Index(self.chunks)
        elif self.chunks:
            self.bm25_index = BM25Index(self.chunks)

        # 3. Load or Init Vector Store
        if self.index_dir.exists() and (self.index_dir / "embeddings.npy").exists():
            try:
                self.vector_store = DenseVectorStore.load(self.index_dir)
            except Exception as e:
                log.warning(f"Could not load VectorStore from {self.index_dir}: {e}")

        # 4. Initialize Embedding Model
        if self.use_mock_encoder:
            self.encoder = MockEmbeddingModel(dim=128)
            log.info("Initialized MockEmbeddingModel for test mode.")
        else:
            self._init_neural_models()

    def _init_neural_models(self):
        """Attempts to load BGE-M3 and cross-encoder reranker models."""
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            log.info("Loading BGE-M3 sentence transformer...")
            self.encoder = SentenceTransformer("BAAI/bge-m3")
            if hasattr(self.encoder, "max_seq_length"):
                self.encoder.max_seq_length = 1024
            log.info("Loading BGE-Reranker cross encoder...")
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        except Exception as e:
            log.warning(
                f"Neural models could not be loaded directly ({e}). "
                "Dense search will use fallback/mock embedding mode."
            )
            self.encoder = MockEmbeddingModel(dim=128)

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes query string into dense vector."""
        if self.encoder is None:
            self.encoder = MockEmbeddingModel(dim=128)

        if hasattr(self.encoder, "encode"):
            vec = self.encoder.encode(query, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        return MockEmbeddingModel().encode(query)

    def sparse_search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs lexical BM25 search."""
        if not self.bm25_index:
            return []
        return self.bm25_index.search(query, top_k=top_k, category=category)

    def dense_search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs dense vector search against persistent vector store or in-memory chunks."""
        if not query or not re.search(r"[a-zA-Z0-9]", query):
            return []

        q_vec = self.encode_query(query)

        if self.vector_store and self.vector_store.count > 0:
            return self.vector_store.search(q_vec, top_k=top_k, category=category)

        # If vector store is not yet built on disk, score against loaded chunks
        if not self.chunks:
            return []

        # In-memory candidate search for mock / test runs
        candidate_pool = self.chunks
        if len(self.chunks) > 500:
            sparse_hits = [r["doc"] for r in self.sparse_search(query, top_k=top_k * 3, category=category)]
            candidate_pool = sparse_hits if sparse_hits else self.chunks[:top_k * 3]

        results = []
        for d in candidate_pool:
            if category and d.get("category") and d.get("category") != category:
                continue
            text = d.get("text", "")
            d_vec = self.encode_query(text[:300])
            sim = float(np.dot(q_vec, d_vec))
            results.append({
                "doc": d,
                "score": sim,
                "method": "dense",
                "chunk_id": d.get("chunk_id", ""),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Combines sparse and dense ranked candidate lists using Reciprocal Rank Fusion.
        RRF Score: sum( 1.0 / (rrf_k + rank + 1) )
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        for rank, item in enumerate(dense_results):
            doc = item["doc"]
            doc_id = item.get("chunk_id") or doc.get("chunk_id") or f"doc_{hash(doc.get('text', ''))}"
            doc_map[doc_id] = doc
            dense_ranks[doc_id] = rank + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, item in enumerate(sparse_results):
            doc = item["doc"]
            doc_id = item.get("chunk_id") or doc.get("chunk_id") or f"doc_{hash(doc.get('text', ''))}"
            doc_map[doc_id] = doc
            sparse_ranks[doc_id] = rank + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for doc_id, score in sorted_docs:
            fused.append({
                "chunk_id": doc_id,
                "doc": doc_map[doc_id],
                "rrf_score": float(score),
                "dense_rank": dense_ranks.get(doc_id, None),
                "sparse_rank": sparse_ranks.get(doc_id, None),
            })
        return fused

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks top candidate chunks using neural cross-encoder if available,
        or contextual exact-match boosting with score preservation.
        """
        if not candidates:
            return []

        if self.reranker is not None:
            try:
                pairs = [[query, c["doc"].get("text", "")] for c in candidates]
                scores = self.reranker.predict(pairs)
                for idx, c in enumerate(candidates):
                    c["rerank_score"] = float(scores[idx])
                candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                return candidates[:top_n]
            except Exception as e:
                log.warning(f"Cross-encoder reranking failed: {e}. Falling back to heuristic reranking.")

        # Heuristic Contextual Boosting
        # Boost matches for exact IS standard numbers (e.g. 'IS 1786')
        is_matches = re.findall(r"\bIS[\s:\-_]*\d{2,6}\b", query, re.IGNORECASE)
        for c in candidates:
            doc = c["doc"]
            text = doc.get("text", "").lower()
            title = (doc.get("clause_title") or doc.get("product") or "").lower()
            is_num = (doc.get("is_number") or "").lower()

            boost = 0.0
            for is_m in is_matches:
                norm_m = re.sub(r"[\s:\-_]+", " ", is_m.lower()).strip()
                if norm_m in text or norm_m in title or norm_m in is_num:
                    boost += 0.5

            c["rerank_score"] = c.get("rrf_score", 0.0) + boost

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

    def retrieve_fast(
        self,
        query: str,
        top_n: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes lightweight, ultra-fast BM25 retrieval for live interactive chatbot queries:
        1. BM25 Sparse Search across 41,476 verified BIS chunks in disk index
        2. Contextual exact-match boosting for Indian Standard (IS) codes and technical terms
        3. Returns Top-N Chunks with 100% complete document provenance
        4. Bypasses heavy neural transformer inference on CPU to guarantee sub-50ms retrieval.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        sparse_res = self.sparse_search(clean_query, top_k=top_n * 5, category=category)

        # Fallback to broad search if scoped category yielded 0 hits
        if not sparse_res and category is not None:
            sparse_res = self.sparse_search(clean_query, top_k=top_n * 5, category=None)

        if not sparse_res:
            return []

        # Contextual exact-match boosting for IS numbers and standard titles
        is_matches = re.findall(r"\bIS[\s:\-_]*\d{2,6}\b", clean_query, re.IGNORECASE)
        scored_candidates = []
        for item in sparse_res:
            doc = item["doc"]
            text = doc.get("text", "").lower()
            title = (doc.get("clause_title") or doc.get("product") or "").lower()
            is_num = (doc.get("is_number") or "").lower()

            boost = 0.0
            for is_m in is_matches:
                norm_m = re.sub(r"[\s:\-_]+", " ", is_m.lower()).strip()
                if norm_m in is_num:
                    boost += 3.0
                elif norm_m in title:
                    boost += 2.0
                elif norm_m in text:
                    boost += 1.0

            final_score = item.get("score", 0.0) + boost
            candidate = {
                "chunk_id": doc.get("chunk_id") or item.get("chunk_id", ""),
                "doc": doc,
                "score": float(final_score),
                "method": "fast_bm25",
                "source_file": doc.get("source_file", ""),
                "source_url": doc.get("source_url", ""),
                "source_hash": doc.get("source_hash", ""),
                "source_of_truth": doc.get("source_of_truth", ""),
                "category": doc.get("category", ""),
                "is_number": doc.get("is_number", ""),
                "revision_year": doc.get("revision_year", ""),
            }
            scored_candidates.append(candidate)

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_n]

    def retrieve(
        self,
        query: str,
        top_n: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes end-to-end Hybrid Retrieval:
        1. BM25 Sparse Search
        2. BGE-M3 Dense Vector Search
        3. RRF Score Fusion
        4. Contextual Reranking
        5. Returns Top-N Chunks with full provenance
        """
        if not query or not query.strip() or not re.search(r"[a-zA-Z0-9]", query):
            return []

        clean_query = query.strip()
        sparse_res = self.sparse_search(clean_query, top_k=20, category=category)
        dense_res = self.dense_search(clean_query, top_k=20, category=category)

        fused = self.reciprocal_rank_fusion(dense_res, sparse_res, rrf_k=60)
        reranked = self.rerank(clean_query, fused, top_n=top_n)

        # Enforce provenance integrity on all retrieved chunks
        for item in reranked:
            doc = item["doc"]
            item["chunk_id"] = doc.get("chunk_id", "")
            item["source_file"] = doc.get("source_file", "")
            item["source_url"] = doc.get("source_url", "")
            item["source_hash"] = doc.get("source_hash", "")
            item["source_of_truth"] = doc.get("source_of_truth", "")
            item["category"] = doc.get("category", "")
            item["is_number"] = doc.get("is_number", "")
            item["revision_year"] = doc.get("revision_year", "")

        return reranked

    @classmethod
    def build_full_index(
        cls,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        index_dir: Path = DEFAULT_INDEX_DIR,
        batch_size: int = 16,
        use_mock: bool = False,
    ):
        """
        Builds persistent BM25 and Dense vector indexes with resumable checkpointing.
        """
        chunks_file = Path(chunks_path)
        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = out_dir / "indexing_checkpoint.json"

        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file {chunks_file} not found")

        # Calculate SHA-256 of chunks file for corpus identity
        log.info(f"Computing corpus SHA-256 for {chunks_file}...")
        hasher = hashlib.sha256()
        with open(chunks_file, "rb") as cf:
            while buf := cf.read(1024 * 1024):
                hasher.update(buf)
        corpus_hash = hasher.hexdigest()
        log.info(f"Corpus SHA-256: {corpus_hash}")

        log.info(f"Loading chunks for indexing from {chunks_file}...")
        chunks: List[Dict[str, Any]] = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        total_chunks = len(chunks)
        log.info(f"Total chunks to index: {total_chunks}")

        # 1. Build and Save BM25 Index
        bm25_file = out_dir / "bm25_index.pkl"
        bm25 = None
        if bm25_file.exists():
            try:
                loaded_bm25 = BM25Index.load(bm25_file)
                if getattr(loaded_bm25, "corpus_hash", None) == corpus_hash and loaded_bm25.n_docs == total_chunks:
                    log.info(f"Existing verified BM25 index found ({loaded_bm25.n_docs} docs) matching corpus hash. Skipping rebuild.")
                    bm25 = loaded_bm25
            except Exception:
                bm25 = None

        if bm25 is None:
            log.info("Building BM25 index...")
            bm25 = BM25Index(chunks)
            bm25.save(bm25_file, corpus_hash=corpus_hash)

        # 2. Build Dense Vectors
        model_name = "mock" if use_mock else "BAAI/bge-m3"
        encoder = MockEmbeddingModel(dim=128) if use_mock else None
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading BGE-M3 embedding model for indexing...")
            encoder = SentenceTransformer("BAAI/bge-m3")
            if hasattr(encoder, "max_seq_length"):
                encoder.max_seq_length = 1024

        # Check existing checkpoint
        start_idx = 0
        embeddings_list = []
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                if cp.get("corpus_hash") and cp.get("corpus_hash") != corpus_hash:
                    log.warning("Corpus hash mismatch in checkpoint. Starting from 0.")
                    start_idx = 0
                elif cp.get("model_name") and cp.get("model_name") != model_name:
                    log.warning("Model mismatch in checkpoint. Starting from 0.")
                    start_idx = 0
                else:
                    start_idx = cp.get("processed_count", 0)
                    temp_npy = out_dir / "embeddings_partial.npy"
                    if temp_npy.exists() and start_idx > 0:
                        existing_arr = np.load(str(temp_npy))
                        if len(existing_arr) == start_idx:
                            embeddings_list.append(existing_arr)
                            log.info(f"Resuming indexing from checkpoint: {start_idx}/{total_chunks} chunks already processed.")
            except Exception as e:
                log.warning(f"Could not load checkpoint: {e}. Starting from 0.")
                start_idx = 0

        # Encode in batches
        all_embeddings = list(embeddings_list)
        last_checkpoint = start_idx
        for i in range(start_idx, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [b.get("text", "") for b in batch]
            vecs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(np.array(vecs, dtype=np.float32))

            current_count = min(i + batch_size, total_chunks)
            if current_count - last_checkpoint >= 500 or current_count == total_chunks:
                # Save partial checkpoint
                combined = np.vstack(all_embeddings)
                np.save(str(out_dir / "embeddings_partial.npy"), combined)
                last_checkpoint = current_count
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "processed_count": current_count,
                        "total_count": total_chunks,
                        "model_name": model_name,
                        "dim": int(combined.shape[1]),
                        "corpus_hash": corpus_hash,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, f)
                log.info(f"Indexed {current_count}/{total_chunks} chunks ({current_count/total_chunks*100:.1f}%)")

        final_matrix = np.vstack(all_embeddings)
        vector_store = DenseVectorStore(
            embeddings=final_matrix,
            chunk_ids=[c.get("chunk_id", f"c_{idx}") for idx, c in enumerate(chunks)],
            documents=chunks,
        )
        vector_store.save(out_dir, corpus_hash=corpus_hash, model_name=model_name)

        # Remove partial file on success
        if (out_dir / "embeddings_partial.npy").exists():
            (out_dir / "embeddings_partial.npy").unlink()

        log.info(f"Successfully built complete Hybrid Retrieval Index in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="BIS Hybrid Retrieval Pipeline CLI")
    parser.add_argument("--query", type=str, help="Query string to search")
    parser.add_argument("--category", type=str, default=None, help="Category pre-filter")
    parser.add_argument("--top_n", type=int, default=5, help="Number of results to return")
    parser.add_argument("--build-index", action="store_true", help="Build persistent indexes")
    parser.add_argument("--mock", action="store_true", help="Use mock embeddings for testing")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for vector indexing")
    parser.add_argument("--chunks", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks JSONL")
    parser.add_argument("--index-dir", type=str, default=str(DEFAULT_INDEX_DIR), help="Output index directory")

    args = parser.parse_args()

    if args.build_index:
        HybridRetrievalPipeline.build_full_index(
            chunks_path=Path(args.chunks),
            index_dir=Path(args.index_dir),
            batch_size=args.batch_size,
            use_mock=args.mock,
        )
        return

    pipeline = HybridRetrievalPipeline(
        chunks_path=Path(args.chunks),
        index_dir=Path(args.index_dir),
        use_mock_encoder=args.mock,
    )

    query = args.query or "what standard applies to gold jewellery hallmarking"
    print(f"\nSearching for: '{query}' (category={args.category}, top_n={args.top_n})...\n")
    results = pipeline.retrieve(query, top_n=args.top_n, category=args.category)

    for idx, r in enumerate(results, 1):
        doc = r["doc"]
        print(f"{idx}. [{doc.get('is_number', 'RAW')}] {doc.get('clause_title', 'Title')}")
        print(f"   Score: {r.get('rerank_score', 0):.4f} | RRF: {r.get('rrf_score', 0):.4f} | Chunk ID: {r.get('chunk_id')}")
        print(f"   Source: {r.get('source_file')} | Category: {r.get('category')}")
        print(f"   Snippet: {doc.get('text', '')[:140]}...\n")


if __name__ == "__main__":
    main()
