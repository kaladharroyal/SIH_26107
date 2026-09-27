"""
BIS AI Compliance Assistant - Web Application Server (app.py)
FastAPI server connecting the Unified BIS RAG Pipeline to an interactive browser UI.
Provides REST endpoints for Chat, Product Recommendation, Lab Locator, Scheme Walkthrough, and Feedback.
"""

import asyncio
from collections import defaultdict
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Optional
import uuid

# Add src, tests, and root to python path for modular imports
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"
for path in [SRC_DIR, TESTS_DIR, BASE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from feedback_logger import FeedbackLogger
from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("bis_web_app")

app = FastAPI(title="Bureau of Indian Standards (BIS) AI Assistant", version="2.0.0")

# F6: Configurable CORS
cors_env = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000")
if cors_env.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# F6: Production default is 0.0 seconds (no artificial delay)
DEMO_MIN_RESPONSE_SECONDS = float(os.getenv("DEMO_MIN_RESPONSE_SECONDS", "0.0"))

# F6: In-process rate limiting (default 60 req/min per IP, set <= 0 to disable)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_ip_request_timestamps: Dict[str, list] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()


@app.middleware("http")
async def request_observability_and_rate_limit_middleware(request: Request, call_next):
    # F5: Generate or propagate safe Request ID
    req_id_raw = request.headers.get("X-Request-ID", "").strip()
    if req_id_raw and re.match(r"^[a-zA-Z0-9_\-]{1,64}$", req_id_raw):
        req_id = req_id_raw
    else:
        req_id = uuid.uuid4().hex
    request.state.request_id = req_id

    # F6: In-process rate limiting (skip /health, /, and static docs)
    if RATE_LIMIT_PER_MINUTE > 0 and request.url.path not in ["/health", "/"]:
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        async with _rate_limit_lock:
            timestamps = [t for t in _ip_request_timestamps[client_ip] if now - t < 60.0]
            if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
                _ip_request_timestamps[client_ip] = timestamps
                log.warning(f"[{req_id}] Rate limit exceeded for IP {client_ip}")
                return JSONResponse(
                    {"error": "Too Many Requests", "message": "In-process rate limit exceeded. Please retry later."},
                    status_code=429,
                    headers={"X-Request-ID": req_id, "Retry-After": "60"},
                )
            timestamps.append(now)
            _ip_request_timestamps[client_ip] = timestamps

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


async def enforce_demo_latency(start_time: float, min_seconds: float = DEMO_MIN_RESPONSE_SECONDS) -> float:
    """Centralized helper for demo latency pacing. When min_seconds is 0.0, adds zero delay."""
    elapsed = time.monotonic() - start_time
    remaining = min_seconds - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)
    return round((time.monotonic() - start_time) * 1000, 2)


log.info(f"Initializing BIS RAG Pipeline for Web Server (Demo Min Latency: {DEMO_MIN_RESPONSE_SECONDS}s)...")
pipeline = BISRAGPipeline()
feedback_logger = FeedbackLogger()


class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = None


class FeedbackRequest(BaseModel):
    log_id: Optional[str] = "query_log"
    query: str
    rating: int
    notes: Optional[str] = ""


@app.get("/health")
async def health_check():
    """F6: Production health check endpoint (bypasses rate limiting)."""
    return JSONResponse({
        "status": "healthy",
        "service": "bis_ai_assistant",
        "version": "2.0.0",
    })


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>BIS AI Assistant Web Server is Running</h1>"


@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest, request: Request):
    req_id = getattr(request.state, "request_id", "local")
    query_text = req.query.strip()
    if not query_text:
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

    log.info(f"[{req_id}] Received Web Chat Query: '{query_text}'")
    req_start = time.monotonic()
    result = pipeline.query(query_text, category=req.category)
    total_elapsed_ms = await enforce_demo_latency(req_start)

    return JSONResponse({
        "query": result.get("query"),
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
        "retrieval_ms": result.get("retrieval_ms"),
        "generation_ms": result.get("generation_ms"),
        "total_ms": total_elapsed_ms,
        "request_id": req_id,
    })


@app.get("/api/recommend")
async def recommend_endpoint(query: str):
    q_clean = query.strip() if query else ""
    if not q_clean:
        return JSONResponse({"error": "Query parameter required"}, status_code=400)
    res = pipeline.product_recommender.recommend(q_clean)
    return JSONResponse(res)


@app.get("/api/labs")
async def labs_endpoint(query: str = "", state: str = ""):
    res = pipeline.lab_locator.search_labs(query=query, state=state if state else None)
    return JSONResponse(res)


@app.get("/api/schemes")
async def schemes_endpoint(scheme: str = "scheme_i"):
    res = pipeline.scheme_walkthrough.get_walkthrough(scheme)
    return JSONResponse(res)


@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest, request: Request):
    req_id = getattr(request.state, "request_id", "local")
    q_clean = req.query.strip() if req.query else ""
    if not q_clean:
        return JSONResponse({"success": False, "error": "Query cannot be empty"}, status_code=400)
    if not isinstance(req.rating, int) or req.rating < 1 or req.rating > 5:
        return JSONResponse({"success": False, "error": "Rating must be an integer between 1 and 5"}, status_code=400)

    log.info(f"[{req_id}] User Feedback received for query '{q_clean[:40]}': Rating={req.rating}, Notes={req.notes}")
    success = feedback_logger.log_feedback(
        query=q_clean,
        rating=req.rating,
        notes=req.notes,
        log_id=req.log_id,
    )
    if success:
        return JSONResponse({"success": True, "message": "Feedback recorded successfully."})
    else:
        return JSONResponse({"success": False, "error": "Internal feedback storage error."}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log.info(f"Starting server on http://localhost:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
