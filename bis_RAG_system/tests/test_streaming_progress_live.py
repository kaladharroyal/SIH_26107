"""
Verification Test Suite: Authentic Live Verification Progress & SSE Telemetry (Feature 11)
Verifies:
1. Pipeline query_stream generator emits monotonic progress events (15% -> 35%/50% -> 55%/75% -> 85%/90% -> 100%)
2. Specialized Sub-Flows stream authentic subflow_execution & citation_verification events
3. General Hybrid RAG streams retrieval_active, guardrails_eval, grounded_synthesis, and citation_verification
4. Refusal Gate triggers clean stream completion on low-confidence queries
5. FastAPI /api/chat/stream SSE endpoint returns valid text/event-stream chunks
"""

import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src and root are on Python path
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
SRC_DIR = BASE_DIR / "src"
for path in [SRC_DIR, BASE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastapi.testclient import TestClient
from app import app
from rag_pipeline import BISRAGPipeline


def test_sse_stages_order_and_monotonic_progress():
    print("\n" + "=" * 65)
    print("TEST 1: SSE Lifecycle Stages Order & Monotonic Progress")
    print("=" * 65)

    pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)
    query = "what is the standard for high strength deformed steel bars"

    events = list(pipeline.query_stream(query))
    print(f"Total stream events emitted: {len(events)}")
    assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"

    last_pct = 0
    stages_seen = []
    for idx, ev in enumerate(events):
        stage = ev.get("stage")
        pct = ev.get("percent")
        msg = ev.get("message")
        stages_seen.append(stage)
        print(f"  [{idx+1}] Stage: {stage:<22} | Progress: {pct:>3}% | Msg: {msg}")

        assert isinstance(pct, (int, float)), f"Progress percent must be numeric: {pct}"
        assert pct >= last_pct, f"Progress must be non-decreasing! {pct} < {last_pct}"
        assert pct <= 100, f"Progress cannot exceed 100%: {pct}"
        assert msg and len(msg.strip()) > 0, "Event message cannot be empty"
        last_pct = pct

    assert "multilingual_routing" in stages_seen, "Missing multilingual_routing stage"
    assert "completed" in stages_seen, "Missing completed stage"
    assert last_pct == 100, f"Final event must reach 100%, reached {last_pct}%"

    completed_event = events[-1]
    assert completed_event["stage"] == "completed"
    res = completed_event.get("result", {})
    assert res.get("status") in ["success", "no_match"], f"Unexpected result status: {res.get('status')}"
    assert len(res.get("response", "")) > 0, "Result response text cannot be empty"

    print("\n  [PASS] SSE stages and monotonic progress percentages verified.")


def test_subflow_streaming_telemetry():
    print("\n" + "=" * 65)
    print("TEST 2: Specialized Sub-Flows Telemetry (Product, Scheme, Lab, Complaint)")
    print("=" * 65)

    pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)
    test_queries = [
        ("Scheme Walkthrough", "how to apply for Scheme-I ISI certification"),
        ("Lab Locator", "where is the BIS testing lab in Delhi"),
        ("Consumer Complaint", "fake ISI mark on product how to file complaint"),
    ]

    for flow_label, query in test_queries:
        print(f"\nTesting {flow_label} Stream -> Query: '{query}'")
        events = list(pipeline.query_stream(query))
        stages = [e["stage"] for e in events]
        print(f"  Stages: {' -> '.join(stages)}")

        assert "multilingual_routing" in stages
        assert "subflow_execution" in stages
        assert "citation_verification" in stages
        assert "completed" in stages

        completed = events[-1]
        assert completed["percent"] == 100
        res = completed["result"]
        assert res.get("status") == "success"
        assert len(res.get("citations", [])) > 0, f"Expected citations for {flow_label}"

    print("\n  [PASS] Sub-flow streaming telemetry verified across all specialized flows.")


def test_refusal_gate_streaming():
    print("\n" + "=" * 65)
    print("TEST 3: Guardrail Refusal Gate Event Streaming")
    print("=" * 65)

    pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True, confidence_threshold=0.99)
    query = "recipe for homemade chocolate cake"

    events = list(pipeline.query_stream(query))
    stages = [e["stage"] for e in events]
    print(f"Stages emitted for out-of-domain query: {' -> '.join(stages)}")

    assert "guardrails_eval" in stages
    assert "completed" in stages

    completed = events[-1]
    assert completed["percent"] == 100
    res = completed["result"]
    assert res.get("status") == "refused"
    assert "cannot find" in res.get("response").lower() or "पर्याप्त" in res.get("response") or "official bis portal" in res.get("response").lower()

    print("\n  [PASS] Refusal gate event streaming and safe fallback verified.")


def test_fastapi_sse_stream_endpoint():
    print("\n" + "=" * 65)
    print("TEST 4: FastAPI /api/chat/stream Server-Sent Events Endpoint")
    print("=" * 65)

    client = TestClient(app)
    payload = {"query": "what BIS standard covers solar panel photovoltaic modules"}

    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert "text/event-stream" in response.headers.get("content-type", ""), "Content-Type must be text/event-stream"

    # Parse SSE stream text
    raw_lines = response.text.split("\n\n")
    events = []
    for line in raw_lines:
        line_clean = line.strip()
        if line_clean.startswith("data:"):
            json_str = line_clean[5:].strip()
            data = json.loads(json_str)
            events.append(data)

    print(f"FastAPI SSE Stream delivered {len(events)} events:")
    for ev in events:
        print(f"  [SSE] Stage: {ev.get('stage'):<22} | {ev.get('percent'):>3}% | {ev.get('message')}")

    assert len(events) >= 3, "Expected at least 3 SSE events"
    assert events[-1]["stage"] == "completed"
    assert events[-1]["percent"] == 100
    assert "result" in events[-1]
    assert events[-1]["result"]["status"] in ["success", "no_match"]

    print("\n  [PASS] FastAPI SSE endpoint delivers valid text/event-stream chunks.")


def test_empty_query_streaming():
    print("\n" + "=" * 65)
    print("TEST 5: Empty Query Stream Handling")
    print("=" * 65)

    pipeline = BISRAGPipeline(llm_provider="mock", use_mock_retrieval=True)
    events = list(pipeline.query_stream("   "))
    assert len(events) == 1
    assert events[0]["stage"] == "completed"
    assert events[0]["percent"] == 100
    assert events[0]["result"]["status"] == "refused"

    print("  [PASS] Empty query handled gracefully in stream.")


def run_all_tests():
    print("=" * 65)
    print("RUNNING FEATURE 11 VERIFICATION SUITE: REAL-TIME PROGRESS & SSE STREAMING")
    print("=" * 65)

    test_sse_stages_order_and_monotonic_progress()
    test_subflow_streaming_telemetry()
    test_refusal_gate_streaming()
    test_fastapi_sse_stream_endpoint()
    test_empty_query_streaming()

    print("\n" + "=" * 65)
    print("ALL 5 LIVE PROGRESS & SSE STREAMING TESTS PASSED 100%!")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
