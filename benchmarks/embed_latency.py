"""Measure embedding latency against a real OpenAI-compatible /v1/embeddings server.

Usage:
    TREE_RAG_EMBED_URL=http://127.0.0.1:8089/v1 python benchmarks/embed_latency.py

Reports p50/p95/min latency (ms) for batch=1 and batch=10. Requires a running
embedding server; the script does not fall back to any mock embedder.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request

MODEL = os.environ.get("TREE_RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
URL = os.environ.get("TREE_RAG_EMBED_URL", "")


def embed(texts: list[str]) -> tuple[float, list[list[float]]]:
    body = json.dumps({"input": texts, "model": MODEL}).encode()
    req = urllib.request.Request(
        URL.rstrip("/") + "/embeddings", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, [row["embedding"] for row in payload["data"]]


def bench(texts: list[str], n: int) -> dict[str, float]:
    for _ in range(5):  # warmup
        embed(texts)
    times = sorted(embed(texts)[0] for _ in range(n))
    return {
        "batch": len(texts),
        "n": n,
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[int(0.95 * (n - 1))], 2),
        "min_ms": round(times[0], 2),
    }


def main() -> int:
    if not URL:
        print("Set TREE_RAG_EMBED_URL to a running embedding server.", file=sys.stderr)
        return 2
    dim = len(embed(["query: warmup"])[1][0])
    print(json.dumps({"model": MODEL, "dim": dim}))
    print(json.dumps(bench(["query: how to configure the embedding server"], 200)))
    batch = [f"passage: sample note {i} about memory layer configuration" for i in range(10)]
    print(json.dumps(bench(batch, 100)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
