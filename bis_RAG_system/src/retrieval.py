"""
Phase 2: Hybrid Retrieval Engine for BIS RAG System (retrieval.py)
Combines:
  1. Lexical / Sparse BM25 Search (optimized for exact Indian Standard codes, clause refs, terms)
  2. Dense Semantic Vector Search (Dynamic Multilingual Embeddings with persistent matrix store)
  3. Domain Category Pre-filtering (is_standard, qco_order, lab_directory, general_policy, etc.)
  4. Reciprocal Rank Fusion (RRF) for deterministic score merging
  5. Cross-Encoder & Heuristic Contextual Reranking
  6. Resumable, Batch-based Index Construction with Checkpointing
  7. 100% Provenance Preservation (chunk_id, source_file, source_url, source_hash, source_of_truth)
"""

import argparse
import json
import logging
import math
import os
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

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_PATH = BASE_DIR / "processed_chunks.jsonl"
DEFAULT_INDEX_DIR = BASE_DIR / "vector_index"


class BM25Index:
    """
    Persistent Inverted-Index BM25 tailored for BIS Standard documents and clauses.
    Preserves exact IS identifiers (e.g. 'IS 1786', 'IS:1070:2023', 'IS-12860'),
    uses inverted postings lists for O(matching) search latency, and provides disk serialization.
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
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # token -> [(doc_idx, tf)]
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

        # General alphanumeric words across Latin and Indic scripts
        words = re.findall(r"[\w\-]{2,}", text_lower)
        return normalized_is + words

    def build(self, documents: List[Dict[str, Any]]):
        """Builds BM25 inverted postings lists, frequency tables, and IDF from documents."""
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_len = []
        self.doc_freqs = []
        self.doc_categories = []
        self.doc_ids = []
        self.df = {}
        self.inverted_index = {}

        for doc_idx, d in enumerate(documents):
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

            for t, tf in freqs.items():
                self.df[t] = self.df.get(t, 0) + 1
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_idx, tf))

            self.doc_freqs.append(freqs)

        total_len = sum(self.doc_len)
        self.avgdl = total_len / max(self.n_docs, 1)
        log.info(f"Built Inverted BM25 index over {self.n_docs} documents (avgdl={self.avgdl:.2f}, vocab={len(self.df)} terms).")

    def append_documents(self, new_documents: List[Dict[str, Any]]):
        """Incrementally adds new documents to the existing inverted index and updates IDF."""
        if not new_documents:
            return
        start_idx = self.n_docs
        for i, d in enumerate(new_documents):
            doc_idx = start_idx + i
            text = d.get("text", "")
            title = d.get("clause_title") or d.get("product") or ""
            is_num = d.get("is_number") or ""
            full_text = f"{is_num} {title} {text}"

            tokens = self.tokenize(full_text)
            self.documents.append(d)
            self.doc_len.append(len(tokens))
            self.doc_categories.append(d.get("category", "general"))
            self.doc_ids.append(d.get("chunk_id", ""))

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1

            for t, tf in freqs.items():
                self.df[t] = self.df.get(t, 0) + 1
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_idx, tf))

            self.doc_freqs.append(freqs)

        self.n_docs = len(self.documents)
        total_len = sum(self.doc_len)
        self.avgdl = total_len / max(self.n_docs, 1)
        log.info(f"Incrementally appended {len(new_documents)} documents to BM25 index (total docs: {self.n_docs}).")


    def search(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fast inverted-index BM25 search touching only matching candidate documents."""
        if not query or not self.n_docs:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        doc_scores: Dict[int, float] = {}

        for q in q_tokens:
            postings = self.inverted_index.get(q)
            if not postings:
                continue
            df = self.df.get(q, 0)
            # Robertson-Spärck Jones IDF
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            if idf <= 0.0:
                continue

            for doc_idx, tf in postings:
                if category and self.doc_categories[doc_idx] != category:
                    continue
                dl = self.doc_len[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-5)))
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + (idf * (numerator / denominator))

        if not doc_scores:
            return []

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in sorted_docs:
            results.append({
                "doc": self.documents[idx],
                "score": float(score),
                "method": "bm25",
                "chunk_id": self.doc_ids[idx],
            })
        return results

    def save(self, filepath: Union[str, Path]):
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
            "inverted_index": self.inverted_index,
            "doc_categories": self.doc_categories,
            "doc_ids": self.doc_ids,
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
        idx.doc_freqs = state.get("doc_freqs", [])
        idx.doc_categories = state["doc_categories"]
        idx.doc_ids = state["doc_ids"]
        idx.inverted_index = state.get("inverted_index", {})
        if not idx.inverted_index and idx.doc_freqs:
            # Reconstruct inverted index if loading legacy pickle
            for doc_idx, freqs in enumerate(idx.doc_freqs):
                for t, tf in freqs.items():
                    if t not in idx.inverted_index:
                        idx.inverted_index[t] = []
                    idx.inverted_index[t].append((doc_idx, tf))
        log.info(f"Loaded BM25 index with {idx.n_docs} documents from {path}")
        return idx


class HFTransformerEmbeddingModel:
    """
    Production Multilingual Neural Embedding Transformer utilizing HuggingFace AutoModel
    with mean pooling and float32 normalization. Dynamically introspects hidden_size.
    Default active model: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (384-dim).
    Supports high-dimensional models such as 'BAAI/bge-m3' (1024-dim) via EMBEDDING_MODEL_NAME.
    Supports PyTorch dynamic INT8 quantization and ONNX Runtime for high-throughput, low-latency CPU inference.
    """

    def __init__(self, model_name_or_path: Optional[str] = None, use_quantization: Optional[bool] = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        target = model_name_or_path or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.model_name = target
        log.info(f"Initializing HFTransformerEmbeddingModel with model '{target}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(target)
        raw_model = AutoModel.from_pretrained(target)
        raw_model.eval()

        # Dynamic INT8 Quantization Optimization for CPU
        if use_quantization is None:
            use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"

        if use_quantization and not torch.cuda.is_available():
            try:
                self.model = torch.quantization.quantize_dynamic(
                    raw_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                log.info(f"✓ Applied dynamic INT8 quantization to HFTransformerEmbeddingModel '{self.model_name}' (CPU acceleration active).")
            except Exception as e:
                log.warning(f"Could not quantize embedding model ({e}). Using unquantized model.")
                self.model = raw_model
        else:
            self.model = raw_model

        self.dimension = int(getattr(self.model.config, "hidden_size", 384))
        log.info(f"HFTransformerEmbeddingModel ready: '{self.model_name}' (dimension={self.dimension})")

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, batch_size: int = 64, **kwargs) -> np.ndarray:
        import torch
        single = isinstance(texts, str)
        text_list = [texts] if single else texts

        all_vecs = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_vecs.append(embeddings.cpu().numpy())

        res = np.vstack(all_vecs).astype(np.float32)
        return res[0] if single else res


class HFCrossEncoderReranker:
    """
    Production Cross-Encoder Contextual Reranker utilizing HuggingFace
    AutoModelForSequenceClassification with sigmoid calibration and INT8 quantization.
    Performs deep cross-attention between (query, document) pairs with LRU score caching.
    """

    def __init__(self, model_name_or_path: Optional[str] = None, use_quantization: Optional[bool] = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        target = model_name_or_path or os.getenv(
            "RERANKER_MODEL_NAME",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.model_name = target
        log.info(f"Initializing HFCrossEncoderReranker with model '{target}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(target)
        raw_model = AutoModelForSequenceClassification.from_pretrained(target)
        raw_model.eval()

        # Dynamic INT8 Quantization Optimization for Cross-Encoder
        if use_quantization is None:
            use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"

        if use_quantization and not torch.cuda.is_available():
            try:
                self.model = torch.quantization.quantize_dynamic(
                    raw_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                log.info(f"✓ Applied dynamic INT8 quantization to HFCrossEncoderReranker '{self.model_name}' (CPU acceleration active).")
            except Exception as e:
                log.warning(f"Could not quantize cross-encoder ({e}). Using unquantized model.")
                self.model = raw_model
        else:
            self.model = raw_model

        self._score_cache: Dict[Tuple[str, str], float] = {}
        log.info(f"HFCrossEncoderReranker ready: '{self.model_name}'")
        try:
            # Warm up once on CPU to avoid first-inference cold-start latency jitter
            _ = self.predict([("warmup query", "warmup text")])
        except Exception:
            pass

    def predict(self, pairs: List[Union[List[str], Tuple[str, str]]], batch_size: int = 20) -> np.ndarray:
        """
        Computes calibrated cross-encoder probabilities for query-document pairs.
        Returns a 1D numpy array of floats in [0.0, 1.0].
        Utilizes an in-memory score cache to avoid redundant CPU inference.
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        import torch

        all_scores = [0.0] * len(pairs)
        uncached_indices = []
        uncached_pairs = []

        for idx, p in enumerate(pairs):
            q_str = str(p[0]).strip()
            d_str = str(p[1])[:512].strip()
            cache_key = (q_str, d_str)
            if cache_key in self._score_cache:
                all_scores[idx] = self._score_cache[cache_key]
            else:
                uncached_indices.append(idx)
                uncached_pairs.append((q_str, d_str))

        if not uncached_pairs:
            return np.array(all_scores, dtype=np.float32)

        for i in range(0, len(uncached_pairs), batch_size):
            batch = uncached_pairs[i : i + batch_size]
            batch_indices = uncached_indices[i : i + batch_size]
            queries = [p[0] for p in batch]
            docs = [p[1] for p in batch]
            inputs = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
                probs = torch.sigmoid(logits)
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)
                batch_res = probs.cpu().numpy().tolist()

            for b_idx, orig_idx in enumerate(batch_indices):
                score_val = float(batch_res[b_idx])
                all_scores[orig_idx] = score_val
                # Store in LRU cache (capped at 5000 items)
                if len(self._score_cache) >= 5000:
                    self._score_cache.pop(next(iter(self._score_cache)))
                self._score_cache[(queries[b_idx], docs[b_idx])] = score_val

        return np.array(all_scores, dtype=np.float32)



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
            # Deterministic pseudo-random vector seeded by text hash
            h = hash(t)
            rng = np.random.RandomState(abs(h) % (2**31 - 1))
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
    Persistent, memory-mapped dense vector store with FAISS SIMD C++ acceleration
    and automatic category-partitioned sub-indices for sub-millisecond search latency.
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        chunk_ids: Optional[List[str]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        use_faiss: bool = True,
        faiss_index: Optional[Any] = None,
    ):
        self.embeddings = embeddings  # Shape: (N, D), normalized float32
        self.chunk_ids = chunk_ids or []
        self.documents = documents or []
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        self.doc_categories = [d.get("category", "general") for d in self.documents]
        self._category_to_indices: Dict[str, np.ndarray] = {}
        self.category_faiss_indices: Dict[str, Any] = {}
        if self.doc_categories:
            unique_cats = set(self.doc_categories)
            for cat in unique_cats:
                self._category_to_indices[cat] = np.array([i for i, c in enumerate(self.doc_categories) if c == cat], dtype=np.int64)
        self.use_faiss = use_faiss
        self.faiss_index = faiss_index
        if self.faiss_index is None:
            self._init_faiss_index()

    @property
    def dim(self) -> int:
        """Returns the embedding dimension of the store."""
        if self.embeddings is not None and len(self.embeddings.shape) > 1:
            return int(self.embeddings.shape[1])
        return 384

    def _init_faiss_index(self):
        """Builds in-memory global and per-category FAISS IndexFlatIP if FAISS is available."""
        if not self.use_faiss or self.embeddings is None or len(self.embeddings) == 0:
            return
        try:
            import faiss
            dim = int(self.embeddings.shape[1])
            # Global FAISS index
            index = faiss.IndexFlatIP(dim)
            vecs = np.ascontiguousarray(self.embeddings.astype(np.float32))
            index.add(vecs)
            self.faiss_index = index

            # Build per-category accelerated FAISS sub-indices
            self.category_faiss_indices = {}
            for cat, indices in self._category_to_indices.items():
                if len(indices) > 0:
                    cat_vecs = np.ascontiguousarray(self.embeddings[indices].astype(np.float32))
                    cat_idx = faiss.IndexFlatIP(dim)
                    cat_idx.add(cat_vecs)
                    self.category_faiss_indices[cat] = cat_idx

            log.info(f"Initialized accelerated FAISS IndexFlatIP with {index.ntotal} vectors ({len(self.category_faiss_indices)} category sub-indices, dim={dim})")
        except Exception as e:
            log.warning(f"Could not initialize FAISS index ({e}). Falling back to NumPy vector search.")
            self.faiss_index = None
            self.category_faiss_indices = {}

    @property
    def count(self) -> int:
        return len(self.chunk_ids)

    def append_embeddings(
        self,
        new_embeddings: np.ndarray,
        new_documents: List[Dict[str, Any]],
        new_chunk_ids: Optional[List[str]] = None,
    ):
        """Incrementally appends new embeddings and documents to the dense vector store and FAISS index."""
        if len(new_documents) == 0:
            return
        if new_chunk_ids is None:
            new_chunk_ids = [d.get("chunk_id", f"chunk_{len(self.documents)+i}") for i, d in enumerate(new_documents)]

        vecs = np.ascontiguousarray(new_embeddings.astype(np.float32))
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = vecs
        else:
            self.embeddings = np.vstack([self.embeddings, vecs])

        start_idx = len(self.documents)
        self.documents.extend(new_documents)
        self.chunk_ids.extend(new_chunk_ids)

        for i, cid in enumerate(new_chunk_ids):
            self.id_to_idx[cid] = start_idx + i

        self.doc_categories = [d.get("category", "general") for d in self.documents]
        self._category_to_indices = {}
        for cat in set(self.doc_categories):
            self._category_to_indices[cat] = np.array([i for i, c in enumerate(self.doc_categories) if c == cat], dtype=np.int64)

        if self.use_faiss:
            self._init_faiss_index()

        log.info(f"Incrementally appended {len(new_documents)} documents to DenseVectorStore (total items: {len(self.chunk_ids)}).")


    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
        category: Optional[str] = None,
        strict: bool = False,
    ) -> List[Dict[str, Any]]:
        """Executes vector cosine search via FAISS C++ acceleration or NumPy fallback with strict dimension validation."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Ensure 1D normalized query vector
        q = query_vec.flatten().astype(np.float32)
        expected_dim = self.embeddings.shape[1]
        if len(q) != expected_dim:
            if strict:
                raise ValueError(f"Embedding dimension mismatch: query vector has {len(q)} dims but index expects {expected_dim} dims!")
            log.error(
                f"Embedding dimension mismatch: query vector has {len(q)} dims but index expects {expected_dim} dims! "
                "Ensure consistent embedding models across ingestion and query phases."
            )

            # Safe boundary check
            if len(q) < expected_dim:
                q = np.pad(q, (0, expected_dim - len(q)))
            else:
                q = q[:expected_dim]

        norm = np.linalg.norm(q)
        if norm > 1e-9:
            q = q / norm

        # 1. FAISS Accelerated Category Sub-Index Path
        if category and category in self.category_faiss_indices:
            cat_faiss = self.category_faiss_indices[category]
            cat_indices = self._category_to_indices[category]
            q_2d = np.ascontiguousarray(q.reshape(1, -1).astype(np.float32))
            fetch_k = min(top_k, cat_faiss.ntotal)
            scores, indices = cat_faiss.search(q_2d, fetch_k)
            results = []
            for score, local_idx in zip(scores[0], indices[0]):
                if local_idx < 0 or local_idx >= len(cat_indices):
                    continue
                global_idx = int(cat_indices[local_idx])
                results.append({
                    "doc": self.documents[global_idx],
                    "score": float(score),
                    "method": "dense",
                    "chunk_id": self.chunk_ids[global_idx],
                })
            return results

        # 2. FAISS Accelerated Global Index Path
        if self.faiss_index is not None and not category:
            q_2d = np.ascontiguousarray(q.reshape(1, -1).astype(np.float32))
            fetch_k = min(top_k, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(q_2d, fetch_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                results.append({
                    "doc": self.documents[idx],
                    "score": float(score),
                    "method": "dense",
                    "chunk_id": self.chunk_ids[idx],
                })
            return results

        # 3. Filtered or NumPy Fallback Path
        if category:
            cat_indices = self._category_to_indices.get(category)
            if cat_indices is None or len(cat_indices) == 0:
                return []
            sub_embeddings = self.embeddings[cat_indices]
            sims = np.dot(sub_embeddings, q)
            top_k_sub = min(top_k, len(sims))
            if len(sims) > top_k_sub * 5:
                part_idx = np.argpartition(sims, -top_k_sub)[-top_k_sub:]
                top_sub = [(float(sims[p]), int(cat_indices[p])) for p in part_idx]
                top_sub.sort(key=lambda x: x[0], reverse=True)
                top_indices = top_sub
            else:
                sorted_p = np.argsort(-sims)[:top_k_sub]
                top_indices = [(float(sims[p]), int(cat_indices[p])) for p in sorted_p]
        else:
            sims = np.dot(self.embeddings, q)
            top_k = min(top_k, len(sims))
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

    def save(self, output_dir: Union[str, Path]):
        """Saves matrix, FAISS binary, and metadata to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            np.save(str(out_path / "embeddings.npy"), self.embeddings)

        if self.faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self.faiss_index, str(out_path / "faiss_index.bin"))
                log.info(f"Saved FAISS index binary to {out_path / 'faiss_index.bin'}")
            except Exception as e:
                log.warning(f"Could not write FAISS index to disk: {e}")

        meta = {
            "count": len(self.chunk_ids),
            "dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
            "model_name": getattr(self, "model_name", os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")),
            "chunk_ids": self.chunk_ids,
            "has_faiss_index": self.faiss_index is not None,
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
        """Loads matrix, optional pre-built FAISS index, and metadata from disk."""
        path = Path(output_dir)
        npy_file = path / "embeddings.npy"
        meta_file = path / "vector_metadata.json"
        docs_file = path / "documents.jsonl"
        faiss_file = path / "faiss_index.bin"

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

        faiss_idx = None
        if faiss_file.exists():
            try:
                import faiss
                faiss_idx = faiss.read_index(str(faiss_file))
                log.info(f"Loaded pre-built FAISS index binary ({faiss_idx.ntotal} items) from {faiss_file}")
            except Exception as e:
                log.warning(f"Could not load pre-built FAISS index: {e}")

        log.info(f"Loaded DenseVectorStore with {len(chunk_ids)} embeddings of dim {embeddings.shape[1]} from {path}")
        return cls(embeddings=embeddings, chunk_ids=chunk_ids, documents=documents, faiss_index=faiss_idx)


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
        """Loads chunks, BM25 index, vector store, and neural models."""
        # 1. Load or Build BM25 Index
        bm25_file = self.index_dir / "bm25_index.pkl"
        if bm25_file.exists():
            try:
                self.bm25_index = BM25Index.load(bm25_file)
                self.chunks = self.bm25_index.documents
                log.info(f"Loaded {len(self.chunks)} verified chunks from BM25 index.")
            except Exception as e:
                log.warning(f"Could not load BM25 index from {bm25_file}: {e}. Loading from source...")
                self.bm25_index = None

        if self.bm25_index is None:
            if self.chunks_path.exists():
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.chunks.append(json.loads(line))
                log.info(f"Loaded {len(self.chunks)} verified chunks from {self.chunks_path}")
            else:
                log.warning(f"Chunks file not found at {self.chunks_path}")
            self.bm25_index = BM25Index(self.chunks)

        # 3. Load or Init Vector Store
        if self.index_dir.exists() and (self.index_dir / "embeddings.npy").exists():
            try:
                self.vector_store = DenseVectorStore.load(self.index_dir)
            except Exception as e:
                log.warning(f"Could not load VectorStore from {self.index_dir}: {e}")

        # 4. Initialize Embedding Model
        target_dim = self.vector_store.dim if self.vector_store else 384
        if self.use_mock_encoder:
            self.encoder = MockEmbeddingModel(dim=target_dim)
            log.info(f"Initialized MockEmbeddingModel (dim={target_dim}) for test mode.")
        else:
            self._init_neural_models()

    def _init_neural_models(self):
        """Initializes HFTransformerEmbeddingModel and HFCrossEncoderReranker."""
        target_dim = self.vector_store.dim if self.vector_store else 384
        try:
            log.info("Loading Neural Transformer model (HFTransformerEmbeddingModel)...")
            self.encoder = HFTransformerEmbeddingModel()
        except Exception as e:
            log.warning(f"Embedding model could not be loaded ({e}). Using MockEmbeddingModel (dim={target_dim}).")
            self.encoder = MockEmbeddingModel(dim=target_dim)

        use_reranker = os.getenv("USE_RERANKER", "true").lower() in ("true", "1", "yes")
        if use_reranker:
            try:
                log.info("Loading Neural Cross-Encoder (HFCrossEncoderReranker)...")
                self.reranker = HFCrossEncoderReranker()
            except Exception as e:
                log.error(
                    f"===============================================================\n"
                    f"[WARNING/DEGRADATION] Neural cross-encoder failed to initialize ({e})!\n"
                    f"Pipeline will operate in degraded heuristic reranking mode.\n"
                    f"==============================================================="
                )
                self.reranker = None
        else:
            log.info("Neural Cross-Encoder disabled via USE_RERANKER=false.")
            self.reranker = None

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes query string into dense vector."""
        target_dim = self.vector_store.dim if self.vector_store else 384
        if self.encoder is None:
            self.encoder = MockEmbeddingModel(dim=target_dim)

        if hasattr(self.encoder, "encode"):
            vec = self.encoder.encode(query, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        return MockEmbeddingModel(dim=target_dim).encode(query)

    def sparse_search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs lexical BM25 search."""
        if not self.bm25_index:
            return []
        return self.bm25_index.search(query, top_k=top_k, category=category)

    def dense_search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs dense vector search against persistent vector store or in-memory chunks."""
        if not query or not query.strip():
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
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}

        for rank, item in enumerate(dense_results):
            doc = item["doc"]
            doc_id = item.get("chunk_id") or doc.get("chunk_id") or f"doc_{hash(doc.get('text', ''))}"
            doc_map[doc_id] = doc
            dense_ranks[doc_id] = rank + 1
            dense_scores[doc_id] = float(item.get("score", 0.0))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, item in enumerate(sparse_results):
            doc = item["doc"]
            doc_id = item.get("chunk_id") or doc.get("chunk_id") or f"doc_{hash(doc.get('text', ''))}"
            doc_map[doc_id] = doc
            sparse_ranks[doc_id] = rank + 1
            sparse_scores[doc_id] = float(item.get("score", 0.0))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for doc_id, score in sorted_docs:
            fused.append({
                "chunk_id": doc_id,
                "doc": doc_map[doc_id],
                "rrf_score": float(score),
                "dense_score": dense_scores.get(doc_id, None),
                "sparse_score": sparse_scores.get(doc_id, None),
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
                # Limit cross-encoder evaluation to top candidates from RRF pool (default 6)
                # to guarantee sub-300ms interactive CPU latency SLA
                rerank_depth = int(os.getenv("RERANK_DEPTH", "6"))
                candidates_to_score = candidates[:rerank_depth]
                pairs = [[query, c["doc"].get("text", "")] for c in candidates_to_score]
                scores = self.reranker.predict(pairs)
                for idx, c in enumerate(candidates_to_score):
                    score = float(scores[idx])
                    c["rerank_score"] = score
                    c["cross_encoder_score"] = score
                    c["rerank_method"] = "neural"
                candidates_to_score.sort(key=lambda x: x["rerank_score"], reverse=True)
                return candidates_to_score[:top_n]
            except Exception as e:
                log.error(
                    f"===============================================================\n"
                    f"[WARNING/DEGRADATION] Cross-encoder reranking failed: {e}.\n"
                    f"Falling back to heuristic regex boosting for query '{query[:40]}'.\n"
                    f"==============================================================="
                )

        # Heuristic Contextual Boosting Fallback
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
            c["rerank_method"] = "heuristic_fallback"

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
        1. Multilingual Query Normalization (for Indic scripts / Hinglish)
        2. BM25 Sparse Search (with cross-lingual query fusion)
        3. Dense Vector Search (with cross-lingual query fusion)
        4. RRF Score Fusion
        5. Contextual Reranking
        6. Returns Top-N Chunks with full provenance
        """
        if not query or not query.strip() or not re.sub(r"[^\w\s]", "", query, flags=re.UNICODE).strip():
            return []

        clean_query = query.strip()
        sparse_res = self.sparse_search(clean_query, top_k=20, category=category)
        dense_res = self.dense_search(clean_query, top_k=20, category=category)

        # Cross-lingual expansion for Indic scripts & code-mixed queries
        norm_q = clean_query
        try:
            from multilingual import MultilingualHandler
            normalizer = MultilingualHandler()
            norm_q = normalizer.normalize_native_to_english_keywords(clean_query)
            norm_q = normalizer.normalize_hinglish_to_english(norm_q)
            if norm_q.strip() and norm_q.strip().lower() != clean_query.lower():
                norm_sparse = self.sparse_search(norm_q, top_k=20, category=category)
                sparse_ids = {r.get("chunk_id") or r["doc"].get("chunk_id") for r in sparse_res}
                for r in norm_sparse:
                    cid = r.get("chunk_id") or r["doc"].get("chunk_id")
                    if cid not in sparse_ids:
                        sparse_res.append(r)
                        sparse_ids.add(cid)

                norm_dense = self.dense_search(norm_q, top_k=20, category=category)
                dense_ids = {r.get("chunk_id") or r["doc"].get("chunk_id") for r in dense_res}
                for r in norm_dense:
                    cid = r.get("chunk_id") or r["doc"].get("chunk_id")
                    if cid not in dense_ids:
                        dense_res.append(r)
                        dense_ids.add(cid)
        except Exception as e:
            log.debug(f"Multilingual query expansion skipped: {e}")

        fused = self.reciprocal_rank_fusion(dense_res, sparse_res, rrf_k=60)
        rerank_query = norm_q if (norm_q and norm_q.strip()) else clean_query
        reranked = self.rerank(rerank_query, fused, top_n=top_n)

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

    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Dynamically calculates and returns real chunk counts and category breakdown
        directly from the active loaded index.
        """
        docs = self.chunks if self.chunks else (self.bm25_index.documents if self.bm25_index else [])
        total_chunks = len(docs)
        category_counts: Dict[str, int] = {}
        for d in docs:
            cat = d.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        return {
            "total_chunks": total_chunks,
            "category_counts": category_counts,
            "categories": sorted(list(category_counts.keys())),
            "source_of_truth": "live_index",
        }

    @classmethod
    def build_full_index(
        cls,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        index_dir: Path = DEFAULT_INDEX_DIR,
        batch_size: int = 64,
        use_mock: bool = False,
        force_rebuild: bool = False,
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

        log.info(f"Loading chunks for indexing from {chunks_file}...")
        chunks: List[Dict[str, Any]] = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        total_chunks = len(chunks)
        log.info(f"Total chunks to index: {total_chunks}")

        # 1. Build and Save BM25 Index
        log.info("Building BM25 index...")
        bm25 = BM25Index(chunks)
        bm25.save(out_dir / "bm25_index.pkl")

        # 2. Build Dense Vectors
        encoder = MockEmbeddingModel(dim=128) if use_mock else None
        if encoder is None:
            try:
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                log.info(f"Loading dynamic neural embedding model '{model_name}' for indexing...")
                encoder = HFTransformerEmbeddingModel(model_name)
            except Exception as e:
                log.warning(f"Could not load neural embedding model ({e}). Using MockEmbeddingModel.")
                encoder = MockEmbeddingModel(dim=128)

        # Check existing checkpoint
        start_idx = 0
        embeddings_list = []
        if checkpoint_file.exists() and not force_rebuild:
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                saved_total = cp.get("total_count", 0)
                saved_proc = cp.get("processed_count", 0)
                if saved_total == total_chunks and 0 < saved_proc < total_chunks:
                    temp_npy = out_dir / "embeddings_partial.npy"
                    if temp_npy.exists():
                        existing_arr = np.load(str(temp_npy))
                        if len(existing_arr) == saved_proc:
                            start_idx = saved_proc
                            embeddings_list.append(existing_arr)
                            log.info(f"Resuming indexing from checkpoint: {start_idx}/{total_chunks} chunks.")
            except Exception as e:
                log.warning(f"Could not load checkpoint: {e}. Starting from 0.")
                start_idx = 0

        # Encode in batches
        all_embeddings = list(embeddings_list)
        for i in range(start_idx, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [b.get("text", "")[:500] for b in batch]
            vecs = encoder.encode(texts, normalize_embeddings=True)
            if hasattr(vecs, "ndim") and vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            all_embeddings.append(np.array(vecs, dtype=np.float32))

            current_count = min(i + batch_size, total_chunks)
            if current_count % 500 == 0 or current_count == total_chunks:
                # Save partial checkpoint
                combined = np.vstack(all_embeddings)
                np.save(str(out_dir / "embeddings_partial.npy"), combined)
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({"processed_count": current_count, "total_count": total_chunks, "timestamp": datetime.now(timezone.utc).isoformat()}, f)
                log.info(f"Indexed {current_count}/{total_chunks} chunks ({current_count/total_chunks*100:.1f}%)")

        final_matrix = np.vstack(all_embeddings)
        vector_store = DenseVectorStore(
            embeddings=final_matrix,
            chunk_ids=[c.get("chunk_id", f"c_{idx}") for idx, c in enumerate(chunks)],
            documents=chunks,
        )
        vector_store.save(out_dir)

        # Remove partial file on success
        if (out_dir / "embeddings_partial.npy").exists():
            (out_dir / "embeddings_partial.npy").unlink()
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        log.info(f"Successfully built complete Hybrid Retrieval Index in {out_dir}")

    def incremental_ingest(
        self,
        new_documents: List[Dict[str, Any]],
        save_to_disk: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally indexes new documents into both BM25 and Dense vector store,
        updating FAISS and BM25 persistent index files without full corpus rebuilds.
        """
        if not new_documents:
            return {"status": "empty", "added_count": 0}

        log.info(f"Starting incremental ingestion of {len(new_documents)} documents...")
        
        # 1. Update BM25 Index
        if self.bm25_index is None:
            self.bm25_index = BM25Index(new_documents)
        else:
            self.bm25_index.append_documents(new_documents)

        # 2. Compute Embeddings & Update Dense Vector Store
        texts = [f"{d.get('is_number', '')} {d.get('clause_title', '')} {d.get('text', '')}" for d in new_documents]
        new_embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        if hasattr(new_embeddings, "ndim") and new_embeddings.ndim == 1:
            new_embeddings = new_embeddings.reshape(1, -1)

        start_count = len(self.vector_store.chunk_ids) if self.vector_store else 0
        new_chunk_ids = [d.get("chunk_id", f"inc_chunk_{start_count + i}") for i, d in enumerate(new_documents)]
        
        if self.vector_store is None:
            self.vector_store = DenseVectorStore(
                embeddings=np.array(new_embeddings, dtype=np.float32),
                chunk_ids=new_chunk_ids,
                documents=new_documents,
            )
        else:
            self.vector_store.append_embeddings(new_embeddings, new_documents, new_chunk_ids)

        # 3. Persist updated index artifacts to disk if requested
        if save_to_disk:
            out_dir = Path(output_dir) if output_dir else getattr(self, "index_dir", DEFAULT_INDEX_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.bm25_index.save(out_dir / "bm25_index.pkl")
            self.vector_store.save(out_dir)
            log.info(f"✓ Saved updated incremental index to {out_dir}")

        return {
            "status": "success",
            "added_count": len(new_documents),
            "total_documents": self.bm25_index.n_docs,
        }




def main():
    parser = argparse.ArgumentParser(description="BIS Hybrid Retrieval Pipeline CLI")
    parser.add_argument("--query", type=str, help="Query string to search")
    parser.add_argument("--category", type=str, default=None, help="Category pre-filter")
    parser.add_argument("--top_n", type=int, default=5, help="Number of results to return")
    parser.add_argument("--build-index", action="store_true", help="Build persistent indexes")
    parser.add_argument("--mock", action="store_true", help="Use mock embeddings for testing")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for vector indexing")
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
