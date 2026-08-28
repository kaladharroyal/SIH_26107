"""
Phase 6: Web UI & Full System Integration Server (app.py)
FastAPI server providing REST endpoints for Chat, Sub-Flow Routing, Citation Viewing, and Feedback Logging.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from test_phase5 import MultilingualBISPipelne
from feedback_logger import FeedbackLogger
from lab_locator import LabLocator
from product_recommender import ProductRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bis_web_app")

app = FastAPI(title="BIS AI Compliance Assistant", version="1.0.0")

pipeline = MultilingualBISPipelne()
feedback_logger = FeedbackLogger()
lab_locator = LabLocator()
product_recommender = ProductRecommender()


class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = "general"


class FeedbackRequest(BaseModel):
    log_id: int
    rating: int
    feedback_notes: Optional[str] = ""


@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    log.info(f"API Chat request received: '{query_text}'")
    res = pipeline.process_multilingual_query(query_text)

    log_id = feedback_logger.log_query(
        query=query_text,
        intent=res.get("sub_flow", "general"),
        confidence_score=res.get("confidence_score", 0.95),
        response_text=res.get("response", ""),
        retrieved_chunks=res.get("retrieved_chunks", []),
    )

    return JSONResponse({
        "log_id": log_id,
        "query": query_text,
        "detected_language": res.get("detected_language", "English"),
        "sub_flow": res.get("sub_flow", "general_rag"),
        "response": res.get("response", ""),
    })


@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    success = feedback_logger.submit_feedback(req.log_id, req.rating, req.feedback_notes)
    return JSONResponse({"success": success})


@app.get("/api/labs")
async def labs_endpoint(query: str = "", state: str = ""):
    res = lab_locator.search_labs(query, state=state)
    return JSONResponse(res)


@app.get("/api/recommend")
async def recommend_endpoint(query: str):
    res = product_recommender.recommend(query)
    return JSONResponse(res)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
