"""
Phase 2: Hybrid Retrieval Pipeline with BGE-M3, BM25, RRF, and Cross-Encoder Reranking
Supports dense semantic vector search, BM25 keyword matching, reciprocal rank fusion (RRF),
category pre-filtering, and cross-encoder reranking.
"""

import math
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hybrid_retrieval")

RAW_DIR = Path("./raw_data")
PROCESSED_CHUNKS_PATH = Path("./processed_chunks.jsonl")


class BM25Index:
    """In-memory BM25 index fallback for fast local keyword searching."""
    def __init__(self, documents: List[Dict[str, Any]]):
        self.docs = documents
        self.doc_len = [len(self._tokenize(d.get("text", ""))) for d in documents]
        self.avgdl = sum(self.doc_len) / max(len(documents), 1)
        self.n_docs = len(documents)
        self.df = {}
        self.doc_freqs = []

        for d in documents:
            freqs = {}
            tokens = self._tokenize(d.get("text", ""))
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            for t in freqs:
                self.df[t] = self.df.get(t, 0) + 1
            self.doc_freqs.append(freqs)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        scores = []
        k1 = 1.5
        b = 0.75

        for i, d in enumerate(self.docs):
            if category and d.get("category") and d.get("category") != category:
                continue
            
            score = 0.0
            freqs = self.doc_freqs[i]
            dl = self.doc_len[i]

            for q in q_tokens:
                if q not in freqs:
                    continue
                df = self.df.get(q, 0)
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = freqs[q]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (dl / self.avgdl))
                score += idf * (numerator / denominator)

            if score > 0:
                scores.append((score, d))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"doc": d, "score": s, "method": "bm25"} for s, d in scores[:top_k]]


class HybridRetrievalPipeline:
    def __init__(self, chunks_path: Path = PROCESSED_CHUNKS_PATH):
        self.chunks = self._load_chunks(chunks_path)
        self.bm25_index = BM25Index(self.chunks)
        self.encoder = None
        self.reranker = None
        self._init_models()

    def _load_chunks(self, path: Path) -> List[Dict[str, Any]]:
        chunks = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))
            log.info(f"Loaded {len(chunks)} chunks into memory for retrieval.")
        else:
            log.warning(f"Chunks file {path} not found. Scanning up to 5,000 raw_data JSONs...")
            count = 0
            for json_file in RAW_DIR.glob("*.json"):
                if json_file.name != "manifest.jsonl":
                    try:
                        data = json.loads(json_file.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            for idx, item in enumerate(data):
                                text = item.get("page_text", "")
                                if text.strip():
                                    chunks.append({
                                        "chunk_id": f"{json_file.stem}_{idx}",
                                        "is_number": "RAW",
                                        "revision_year": "2026",
                                        "clause_number": str(idx),
                                        "clause_title": item.get("page_url", json_file.name),
                                        "text": text,
                                        "source_file": json_file.name,
                                        "category": item.get("category", "general"),
                                    })
                        count += 1
                        if count >= 5000:
                            break
                    except Exception:
                        pass
            log.info(f"Fallback scanned {len(chunks)} raw page chunks from {count} JSON files.")
        return chunks

    def _init_models(self):
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                log.info("Loading BGE-M3 dense embedding model...")
                self.encoder = SentenceTransformer("BAAI/bge-m3")
                log.info("Loading BGE Reranker model...")
                self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            except Exception as e:
                log.warning(f"Could not load neural embedding models directly: {e}. Falling back to BM25 + keyword search.")

    def dense_search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.encoder:
            return []
        
        q_vec = self.encoder.encode(query, normalize_embeddings=True)
        # Fast dot-product / cosine similarity
        results = []
        for d in self.chunks:
            if category and d.get("category") and d.get("category") != category:
                continue
            # Fallback simple string matching weight + dense representation
            text = d.get("text", "")
            d_vec = self.encoder.encode(text[:500], normalize_embeddings=True)
            sim = float(q_vec @ d_vec)
            results.append((sim, d))

        results.sort(key=lambda x: x[0], reverse=True)
        return [{"doc": d, "score": s, "method": "dense"} for s, d in results[:top_k]]

    def sparse_search(self, query: str, top_k: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.bm25_index.search(query, top_k=top_k, category=category)

    def reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], rrf_k: int = 60) -> List[Dict[str, Any]]:
        rrf_scores = {}
        doc_map = {}

        for rank, item in enumerate(dense_results):
            doc = item["doc"]
            doc_id = doc.get("chunk_id") or f"{doc.get('is_number', '')}_{doc.get('clause_number', '')}_{hash(doc.get('text', ''))}"
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, item in enumerate(sparse_results):
            doc = item["doc"]
            doc_id = doc.get("chunk_id") or f"{doc.get('is_number', '')}_{doc.get('clause_number', '')}_{hash(doc.get('text', ''))}"
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"doc": doc_map[doc_id], "rrf_score": score} for doc_id, score in sorted_docs]

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if self.reranker and HAS_SENTENCE_TRANSFORMERS:
            pairs = [[query, c["doc"]["text"]] for c in candidates]
            scores = self.reranker.predict(pairs)
            for idx, c in enumerate(candidates):
                c["rerank_score"] = float(scores[idx])
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates[:top_n]
        else:
            # Fallback boost for IS number exact matches in title/text
            for c in candidates:
                text = c["doc"]["text"]
                boost = 0.0
                if re.search(r"IS\s*\d+", query, re.IGNORECASE):
                    is_match = re.findall(r"IS\s*\d+", query, re.IGNORECASE)[0]
                    if is_match.lower() in text.lower():
                        boost += 0.5
                c["rerank_score"] = c.get("rrf_score", 0.0) + boost

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates[:top_n]

    def retrieve(self, query: str, top_n: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        log.info(f"Executing Hybrid Retrieval for query: '{query}' (category={category})")
        dense_res = self.dense_search(query, top_k=20, category=category)
        sparse_res = self.sparse_search(query, top_k=20, category=category)
        fused = self.reciprocal_rank_fusion(dense_res, sparse_res)
        reranked = self.rerank(query, fused, top_n=top_n)
        return reranked


if __name__ == "__main__":
    pipeline = HybridRetrievalPipeline()
    test_queries = [
        "what standard applies to gold jewellery hallmarking",
        "IS 1786 steel reinforcement requirements",
        "how to register as a jeweller under BIS scheme",
    ]
    for q in test_queries:
        print(f"\n--- QUERY: {q} ---")
        results = pipeline.retrieve(q, top_n=3)
        for idx, res in enumerate(results, 1):
            doc = res["doc"]
            print(f"{idx}. [{doc.get('is_number', 'RAW')}] {doc.get('clause_title', 'Title')} (Score: {res.get('rerank_score', 0):.4f})")
            print(f"   Snippet: {doc.get('text', '')[:120]}...")
