"""
Phase 6: Automated Concurrency & Stress Testing Suite (stress_test.py)
Evaluates pipeline throughput, latency percentiles (p50, p95, p99), and error rates
under multi-threaded concurrent load with memory profiling.
"""

import concurrent.futures
import logging
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Dict, List
import pytest

# Add src and root to python path for clean test execution
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"
for path in [SRC_DIR, TESTS_DIR, BASE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_pipeline import BISRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stress_test")

STRESS_QUERIES_POOL = [
    # 1. Ambiguous Product Terms
    "do I need certification for solar panel",
    "what is the process for tmt bar testing",
    # 2. Outdated / Revised Standard Handling
    "IS 1786 requirements for steel",
    "IS 12860 metallic coating thickness measurement",
    # 3. Code-Mixed Hinglish Inputs
    "mera LED bulb ke liye BIS certification chahiye",
    "Scheme-I me apply kaise kare kitna fee lagiga",
    # 4. Consumer Complaints
    "my gold hallmark jewellery is fake how to complain",
    "substandard ISI marked product complaint compensation",
    # 5. Lab Lookups & Schemes
    "find certified testing labs in Maharashtra",
    "what is foreign manufacturers certification FMCS Scheme-IV",
]


def execute_stress_run(
    num_workers: int = 4,
    num_requests: int = 20,
) -> Dict[str, Any]:
    """Executes multi-threaded concurrent queries against BISRAGPipeline."""
    log.info(f"Starting Stress Test: {num_requests} requests across {num_workers} workers...")
    pipeline = BISRAGPipeline(llm_provider="mock", use_fast_retrieval=True)

    queries = [STRESS_QUERIES_POOL[i % len(STRESS_QUERIES_POOL)] for i in range(num_requests)]
    latencies: List[float] = []
    successes = 0
    failures = 0
    errors: List[str] = []

    tracemalloc.start()
    t_start = time.time()

    def _worker(q: str):
        t0 = time.time()
        try:
            res = pipeline.query(q)
            dt = round((time.time() - t0) * 1000, 2)
            if res.get("status") in ["success", "refused", "unavailable"]:
                return (True, dt, None)
            else:
                return (False, dt, f"Unexpected status: {res.get('status')}")
        except Exception as e:
            dt = round((time.time() - t0) * 1000, 2)
            return (False, dt, str(e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, q) for q in queries]
        for f in concurrent.futures.as_completed(futures):
            ok, dt, err = f.result()
            latencies.append(dt)
            if ok:
                successes += 1
            else:
                failures += 1
                errors.append(err)

    total_wall_time = time.time() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def _percentile(p: float) -> float:
        if n == 0:
            return 0.0
        k = (n - 1) * p
        f = int(k)
        c = f + 1
        if c < n:
            return round(latencies_sorted[f] + (k - f) * (latencies_sorted[c] - latencies_sorted[f]), 2)
        return round(latencies_sorted[f], 2)

    rps = round(num_requests / total_wall_time, 2) if total_wall_time > 0 else 0.0
    error_rate = round((failures / num_requests) * 100, 2) if num_requests > 0 else 0.0

    report = {
        "config": {
            "workers": num_workers,
            "total_requests": num_requests,
        },
        "summary": {
            "successful_requests": successes,
            "failed_requests": failures,
            "error_rate_pct": error_rate,
            "throughput_rps": rps,
            "total_wall_time_sec": round(total_wall_time, 3),
        },
        "latency_ms": {
            "min": round(latencies_sorted[0], 2) if n > 0 else 0.0,
            "average": round(sum(latencies) / max(1, n), 2),
            "p50": _percentile(0.50),
            "p95": _percentile(0.95),
            "p99": _percentile(0.99),
            "max": round(latencies_sorted[-1], 2) if n > 0 else 0.0,
        },
        "memory_mb": {
            "peak_allocated": round(peak_mem / (1024 * 1024), 2),
            "final_allocated": round(current_mem / (1024 * 1024), 2),
        },
        "errors": errors,
    }
    return report


def test_stress_concurrency():
    """Pytest test asserting that pipeline handles concurrent queries with 0% error rate."""
    workers = int(os.getenv("STRESS_WORKERS", "4"))
    requests = int(os.getenv("STRESS_QUERIES", "20"))
    report = execute_stress_run(num_workers=workers, num_requests=requests)

    assert report["summary"]["failed_requests"] == 0, f"Stress test encountered failures: {report['errors']}"
    assert report["summary"]["error_rate_pct"] == 0.0
    assert report["summary"]["throughput_rps"] > 0.0
    assert report["latency_ms"]["p95"] < 3000.0, f"p95 latency too high: {report['latency_ms']['p95']}ms"


def run_stress_testing():
    """CLI runner preserving backwards compatibility."""
    workers = int(os.getenv("STRESS_WORKERS", "4"))
    requests = int(os.getenv("STRESS_QUERIES", "24"))
    rep = execute_stress_run(num_workers=workers, num_requests=requests)

    print("\n" + "=" * 70)
    print("      PHASE 6: CONCURRENT STRESS & LATENCY PERCENTILE REPORT")
    print("=" * 70)
    print(f"Workers Configured       : {rep['config']['workers']}")
    print(f"Total Requests Executed  : {rep['config']['total_requests']}")
    print(f"Successful Requests      : {rep['summary']['successful_requests']}")
    print(f"Failed Requests          : {rep['summary']['failed_requests']}")
    print(f"Error Rate               : {rep['summary']['error_rate_pct']}%")
    print(f"Throughput (RPS)         : {rep['summary']['throughput_rps']} req/sec")
    print(f"Total Elapsed Wall Time  : {rep['summary']['total_wall_time_sec']}s")
    print("-" * 70)
    print(f"Latency Min              : {rep['latency_ms']['min']} ms")
    print(f"Latency Average          : {rep['latency_ms']['average']} ms")
    print(f"Latency p50 (Median)     : {rep['latency_ms']['p50']} ms")
    print(f"Latency p95              : {rep['latency_ms']['p95']} ms")
    print(f"Latency p99              : {rep['latency_ms']['p99']} ms")
    print(f"Latency Max              : {rep['latency_ms']['max']} ms")
    print("-" * 70)
    print(f"Peak Memory Allocated    : {rep['memory_mb']['peak_allocated']} MB")
    print("=" * 70 + "\n")

    if rep["summary"]["failed_requests"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_stress_testing()
