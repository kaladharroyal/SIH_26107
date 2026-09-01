"""
BIS AI Compliance Assistant - Web Application Server (app.py)
FastAPI server connecting the Unified BIS RAG Pipeline to an interactive browser UI.
Provides REST endpoints for Chat, Product Recommendation, Lab Locator, Scheme Walkthrough, and Feedback.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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

from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("bis_web_app")

app = FastAPI(title="Bureau of Indian Standards (BIS) AI Assistant", version="2.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import asyncio
import time

DEMO_MIN_RESPONSE_SECONDS = float(os.getenv("DEMO_MIN_RESPONSE_SECONDS", "10.0"))


async def enforce_demo_latency(start_time: float, min_seconds: float = DEMO_MIN_RESPONSE_SECONDS) -> float:
    """
    Centralized helper to ensure consistent minimum demo latency without slowing down retrieval or generation.
    If actual processing took 2s, waits remaining 8s (total 10s).
    If actual processing took 12s, waits 0s (total 12s).
    """
    elapsed = time.monotonic() - start_time
    remaining = min_seconds - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)
    return round((time.monotonic() - start_time) * 1000, 2)


log.info(f"Initializing BIS RAG Pipeline for Web Server (Demo Min Latency: {DEMO_MIN_RESPONSE_SECONDS}s)...")
pipeline = BISRAGPipeline(use_fast_retrieval=True)


class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = None


class FeedbackRequest(BaseModel):
    log_id: Optional[str] = "query_log"
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

    log.info(f"Received Web Chat Query: '{query_text}'")
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
async def feedback_endpoint(req: FeedbackRequest):
    log.info(f"User Feedback received for query '{req.query}': Rating={req.rating}, Notes={req.notes}")
    return JSONResponse({"success": True, "message": "Feedback recorded successfully."})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log.info(f"Starting server on http://localhost:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
