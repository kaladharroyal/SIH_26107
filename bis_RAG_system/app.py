"""
BIS AI Compliance Assistant - Web Application Server (app.py)
FastAPI server connecting the Unified BIS RAG Pipeline to an interactive browser UI.
Provides REST endpoints for Chat, Product Recommendation, Lab Locator, Scheme Walkthrough, and Feedback.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Add src, tests, and root to python path for modular imports
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"
for path in [SRC_DIR, TESTS_DIR, BASE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from cachetools import TTLCache

from generator import _load_env
from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("bis_web_app")

_load_env()

app = FastAPI(title="Bureau of Indian Standards (BIS) AI Assistant", version="2.0.0")

# Configurable CORS origins for browser security
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
if "*" in allowed_origins:
    allow_all = True
else:
    allow_all = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

log.info("Initializing BIS RAG Pipeline for Web Server...")
use_fast = os.getenv("USE_FAST_RETRIEVAL", "false").lower() == "true"
if use_fast:
    log.warning("⚠️  [RETRIEVAL MODE OVERRIDE] USE_FAST_RETRIEVAL=true is active. Dense semantic neural search is BYPASSED (sparse BM25 only).")
else:
    log.info("✅ [RETRIEVAL MODE ACTIVE] True Hybrid Retrieval enabled: BM25 Sparse Search + 384-dim Dense Neural Vector Search with Reciprocal Rank Fusion.")

pipeline = BISRAGPipeline(use_fast_retrieval=use_fast)

# Distributed Response Cache Abstraction (Redis + in-memory TTLCache fallback)
class DistributedResponseCache:
    """
    Production Distributed Cache with Redis backend support and seamless in-memory TTLCache fallback.
    Maintains dict-like syntax (__contains__, __getitem__, __setitem__) for 100% backward compatibility.
    """

    def __init__(self, ttl: int = 3600, maxsize: int = 1000):
        self.ttl = ttl
        self.maxsize = maxsize
        self._memory_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.redis_client = None
        self.is_redis_active = False

        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2.0)
                self.redis_client.ping()
                self.is_redis_active = True
                log.info(f"✓ Connected to Redis Distributed Cache at {redis_url.split('@')[-1] if '@' in redis_url else 'Redis'}")
            except Exception as e:
                log.warning(f"Could not connect to Redis ({e}). Falling back to in-memory TTLCache.")
                self.redis_client = None
                self.is_redis_active = False
        else:
            log.info("Redis URL not configured. Using high-performance in-memory TTLCache (TTL=3600s, maxsize=1000).")

    def _format_key(self, key: Any) -> str:
        if isinstance(key, tuple):
            return f"bis_cache:{':'.join(str(k) for k in key)}"
        return f"bis_cache:{str(key)}"

    def __contains__(self, key: Any) -> bool:
        if self.is_redis_active and self.redis_client:
            try:
                return bool(self.redis_client.exists(self._format_key(key)))
            except Exception:
                pass
        return key in self._memory_cache

    def __getitem__(self, key: Any) -> Dict[str, Any]:
        if self.is_redis_active and self.redis_client:
            try:
                data = self.redis_client.get(self._format_key(key))
                if data is not None:
                    return json.loads(data)
            except Exception:
                pass
        return self._memory_cache[key]

    def __setitem__(self, key: Any, value: Dict[str, Any]):
        # Always update local memory cache
        self._memory_cache[key] = value
        if self.is_redis_active and self.redis_client:
            try:
                r_key = self._format_key(key)
                self.redis_client.setex(r_key, self.ttl, json.dumps(value, ensure_ascii=False))
            except Exception as e:
                log.warning(f"Error writing to Redis cache: {e}")

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        if key in self:
            return self[key]
        return default

    def clear(self):
        self._memory_cache.clear()
        if self.is_redis_active and self.redis_client:
            try:
                keys = self.redis_client.keys("bis_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception:
                pass


# Initialize distributed cache instance
QUERY_CACHE = DistributedResponseCache(ttl=3600, maxsize=1000)

import copy
from concurrent.futures import ThreadPoolExecutor

STREAM_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="bis_sse_worker")



class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = None
    language: Optional[str] = None


class FeedbackRequest(BaseModel):
    log_id: Optional[Union[int, str]] = None
    query: str
    rating: int
    notes: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>BIS AI Assistant Web Server is Running</h1>"


@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)
    if len(query_text) > 1000:
        return JSONResponse({"error": "Query exceeds maximum permitted length of 1000 characters"}, status_code=400)

    # Check cache for identical requests
    cache_key = (query_text.lower(), req.category or "", req.language or "")
    if cache_key in QUERY_CACHE:
        cached_result = copy.deepcopy(QUERY_CACHE[cache_key])
        cached_result["cached"] = True
        log.info(f"Cache Hit for query: '{query_text[:40]}'")
        return JSONResponse(cached_result)

    log.info(f"Received Web Chat Query: '{query_text}' (Language: {req.language})")
    req_start = time.monotonic()

    # Offload heavy synchronous pipeline execution to thread pool
    result = await asyncio.to_thread(
        pipeline.query,
        query_text,
        category=req.category,
        target_language=req.language,
    )
    total_elapsed_ms = round((time.monotonic() - req_start) * 1000, 2)

    response_payload = {
        "query": result.get("query"),
        "log_id": result.get("log_id"),
        "intent": result.get("intent", "general_rag"),
        "flow_used": result.get("flow_used", "general_rag"),
        "status": result.get("status", "success"),
        "confidence_score": result.get("confidence_score", 0.0),
        "response": result.get("response", ""),
        "results": result.get("results"),
        "citations": result.get("citations", []),
        "fallback_used": result.get("fallback_used", False),
        "provider": result.get("provider"),
        "model_used": result.get("model_used"),
        "lang_code": result.get("lang_code"),
        "detected_language": result.get("detected_language") or result.get("response_language"),
        "response_language": result.get("response_language"),
        "retrieval_ms": result.get("retrieval_ms"),
        "generation_ms": result.get("generation_ms"),
        "total_ms": total_elapsed_ms,
        "cached": False,
    }

    # Store in response cache if successful
    if result.get("status") == "success":
        QUERY_CACHE[cache_key] = copy.deepcopy(response_payload)

    return JSONResponse(response_payload)


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: QueryRequest, request: Request):
    query_text = req.query.strip()
    if not query_text:
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

    log.info(f"Received Web Chat Stream Query: '{query_text}' (Language: {req.language})")

    async def sse_event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()

        def background_worker():
            try:
                for event in pipeline.query_stream(
                    query_text,
                    category=req.category,
                    target_language=req.language,
                ):
                    if stop_event.is_set():
                        log.info("SSE client disconnected; stopping stream pipeline worker.")
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            except Exception as e:
                log.error(f"Error during SSE stream generation: {e}")
                err_event = {
                    "stage": "completed",
                    "percent": 100,
                    "message": "Error occurred during stream processing.",
                    "result": {
                        "query": query_text,
                        "status": "error",
                        "response": f"An error occurred while processing the request: {str(e)}",
                        "citations": [],
                    },
                }
                asyncio.run_coroutine_threadsafe(queue.put(err_event), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Submit task to bounded ThreadPoolExecutor instead of unconstrained raw OS thread
        future = STREAM_EXECUTOR.submit(background_worker)

        try:
            while True:
                if await request.is_disconnected():
                    stop_event.set()
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if item is None:
                    break
                payload_str = json.dumps(item, ensure_ascii=False)
                yield f"data: {payload_str}\n\n"
        finally:
            stop_event.set()

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/corpus-stats")
async def corpus_stats_endpoint():
    """Returns real, live corpus chunk counts and category breakdown directly from the active index."""
    stats = await asyncio.to_thread(pipeline.retrieval.get_corpus_stats)
    return JSONResponse(stats)


@app.get("/api/recommend")
async def recommend_endpoint(query: str, language: str = "English"):
    q_clean = query.strip() if query else ""
    if not q_clean:
        return JSONResponse({"error": "Query parameter required"}, status_code=400)
    res = await asyncio.to_thread(pipeline.product_recommender.recommend, q_clean, language=language)
    return JSONResponse(res)


@app.get("/api/labs")
async def labs_endpoint(query: str = "", state: Optional[str] = None, standard: Optional[str] = None, language: str = "English"):
    res = await asyncio.to_thread(pipeline.lab_locator.search_labs, query=query, state=state, standard=standard, language=language)
    return JSONResponse(res)


@app.get("/api/schemes")
async def schemes_endpoint(scheme: str = "scheme_i", language: str = "English"):
    res = await asyncio.to_thread(pipeline.scheme_walkthrough.get_walkthrough, scheme, language=language)
    return JSONResponse(res)


@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    log.info(f"User Feedback received for query '{req.query}': LogID={req.log_id}, Rating={req.rating}, Notes={req.notes}")
    qid = None
    if req.log_id is not None:
        try:
            parsed_id = int(req.log_id)
            if parsed_id > 0:
                qid = parsed_id
        except (ValueError, TypeError):
            qid = None

    success = await asyncio.to_thread(
        pipeline.feedback_logger.submit_feedback,
        query_id=qid,
        rating=req.rating,
        feedback_notes=req.notes,
    )
    return JSONResponse({
        "success": success,
        "message": "Feedback recorded successfully in telemetry database.",
        "log_id": qid,
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log.info(f"Starting server on http://localhost:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)

