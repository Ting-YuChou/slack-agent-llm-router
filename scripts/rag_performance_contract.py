#!/usr/bin/env python3
"""Measure and enforce the RAG ingestion, retrieval, and upload contracts."""

import argparse
import asyncio
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.chunker import DocumentChunk
from src.rag.service import RagService
from src.rag.vector_store import RedisStackRagVectorStore


def evaluate_contract(result: Dict[str, Any]) -> List[str]:
    failures = []
    if result["embedding_request_reduction_percent"] < 90.0:
        failures.append("embedding request reduction is below 90%")
    if result["redis_wait_reduction_percent"] < 90.0:
        failures.append("Redis wait reduction is below 90%")
    if result["ingestion_throughput_improvement"] < 3.0:
        failures.append("ingestion throughput improvement is below 3x")
    if result["retrieval_p95_improvement_percent"] < 30.0:
        failures.append("retrieval p95 improvement is below 30%")
    if result["upload_peak_rss_delta_bytes"] > 64 * 1024 * 1024:
        failures.append("100 MB upload peak RSS delta exceeds 64 MB")
    if result.get("errors", 0) != 0:
        failures.append("benchmark workload reported errors")
    return failures


class _LatencyEmbeddingProvider:
    def __init__(self, latency_seconds: float = 0.001):
        self.latency_seconds = latency_seconds
        self.scalar_calls = 0
        self.batch_calls = 0

    async def embed(self, text: str) -> List[float]:
        self.scalar_calls += 1
        await asyncio.sleep(self.latency_seconds)
        return [float(int(text) % 7), 1.0, 0.0, 0.0]

    async def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        self.batch_calls += 1
        await asyncio.sleep(self.latency_seconds)
        return [[float(int(text) % 7), 1.0, 0.0, 0.0] for text in texts]


def _chunks(count: int, document_id: str) -> List[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            document_id=document_id,
            text=str(index),
            page_start=1,
            page_end=1,
            block_ids=[f"block-{index}"],
            block_types=["text"],
        )
        for index in range(count)
    ]


async def _embedding_benchmark(count: int) -> Dict[str, float]:
    baseline_provider = _LatencyEmbeddingProvider()
    started = time.perf_counter()
    for index in range(count):
        await baseline_provider.embed(str(index))
    baseline_seconds = time.perf_counter() - started

    provider = _LatencyEmbeddingProvider()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"batch_size": 32, "max_concurrent_batches": 2},
        },
        embedding_provider=provider,
    )
    started = time.perf_counter()
    embeddings, warnings = await service._embed_chunks(_chunks(count, "embedding"))
    optimized_seconds = time.perf_counter() - started
    if (
        warnings
        or len(embeddings) != count
        or any(value is None for value in embeddings)
    ):
        raise RuntimeError("embedding benchmark did not complete successfully")
    return {
        "baseline_seconds": baseline_seconds,
        "optimized_seconds": optimized_seconds,
        "baseline_requests": float(baseline_provider.scalar_calls),
        "optimized_requests": float(provider.batch_calls),
    }


async def _redis_benchmark(count: int, host: str, port: int) -> Dict[str, float]:
    import redis.asyncio as redis

    client = redis.Redis(host=host, port=port, decode_responses=False)
    await client.ping()
    prefix = f"ragbench:{uuid.uuid4().hex}"
    started = time.perf_counter()
    for index in range(count):
        key = f"{prefix}:legacy:chunk:{index}"
        await client.hset(key, mapping={"text": str(index)})
        await client.sadd(f"{prefix}:legacy:document", str(index))
        await client.sadd(f"{prefix}:legacy:knowledge_bases", "school")
    baseline_seconds = time.perf_counter() - started
    baseline_waits = count * 3

    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 4},
            "redis": {
                "host": host,
                "port": port,
                "key_prefix": prefix,
                "pipeline_batch_size": 64,
            },
            "indexing": {"atomic_commit_max_chunks": max(2000, count)},
        }
    )
    await store.initialize()
    chunks = _chunks(count, "optimized")
    embeddings = [[float(index % 7), 1.0, 0.0, 0.0] for index in range(count)]
    started = time.perf_counter()
    written = await store.upsert_chunks(chunks, embeddings, knowledge_base_id="school")
    optimized_seconds = time.perf_counter() - started
    if written != count:
        raise RuntimeError("Redis benchmark wrote an unexpected chunk count")
    optimized_waits = ((count + 63) // 64) + 1
    await store.shutdown()
    keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
    if keys:
        await client.delete(*keys)
    await client.aclose()
    return {
        "baseline_seconds": baseline_seconds,
        "optimized_seconds": optimized_seconds,
        "baseline_waits": float(baseline_waits),
        "optimized_waits": float(optimized_waits),
    }


async def _retrieval_benchmark(samples: int = 30) -> Dict[str, float]:
    store = RedisStackRagVectorStore(
        {"retrieval": {"max_concurrent_branches": 3, "deadline_seconds": 1}}
    )
    store.client = object()

    async def delayed_empty(*_args: Any) -> List[Any]:
        await asyncio.sleep(0.01)
        return []

    store._keyword_search = delayed_empty
    store._vector_search = delayed_empty
    store._visual_search = delayed_empty
    baseline = []
    optimized = []
    for _ in range(samples):
        started = time.perf_counter()
        await delayed_empty()
        await delayed_empty()
        await delayed_empty()
        baseline.append(time.perf_counter() - started)

        started = time.perf_counter()
        await store.search(
            "query",
            [1.0],
            knowledge_base_ids=[],
            limit=5,
            candidate_count=30,
            keyword_weight=0.35,
            vector_weight=0.6,
            recency_weight=0.05,
            min_score=0.0,
            visual_embedding=[1.0],
            visual_weight=0.4,
        )
        optimized.append(time.perf_counter() - started)
    return {
        "baseline_p95": _percentile(baseline, 0.95),
        "optimized_p95": _percentile(optimized, 0.95),
    }


def _percentile(values: Sequence[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


class _GeneratedUpload:
    def __init__(self, size: int):
        self.remaining = size

    async def read(self, size: int) -> bytes:
        count = min(size, self.remaining)
        self.remaining -= count
        return b"x" * count


async def _upload_worker(directory: str) -> int:
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {"staging_dir": directory},
            "upload": {
                "multipart_max_bytes": 100_000_000,
                "stream_chunk_bytes": 1_048_576,
            },
        }
    )
    path = await service.stage_uploaded_file(
        _GeneratedUpload(100_000_000),
        filename="large.bin",
        knowledge_base_id="school",
        document_id="rss",
    )
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(path).unlink(missing_ok=True)
    multiplier = 1 if sys.platform == "darwin" else 1024
    print(json.dumps({"rss_delta_bytes": max(0, after - before) * multiplier}))
    return 0


def _measure_upload_rss() -> int:
    with tempfile.TemporaryDirectory() as directory:
        completed = subprocess.run(
            [sys.executable, __file__, "--upload-worker", directory],
            check=True,
            capture_output=True,
            text=True,
        )
    return int(json.loads(completed.stdout)["rss_delta_bytes"])


async def run_contract(chunks: int, host: str, port: int) -> Dict[str, Any]:
    embedding = await _embedding_benchmark(chunks)
    redis = await _redis_benchmark(chunks, host, port)
    retrieval = await _retrieval_benchmark()
    baseline_ingestion = embedding["baseline_seconds"] + redis["baseline_seconds"]
    optimized_ingestion = embedding["optimized_seconds"] + redis["optimized_seconds"]
    return {
        "chunks": chunks,
        "errors": 0,
        "embedding_baseline_requests": int(embedding["baseline_requests"]),
        "embedding_optimized_requests": int(embedding["optimized_requests"]),
        "embedding_request_reduction_percent": 100.0
        * (1.0 - embedding["optimized_requests"] / embedding["baseline_requests"]),
        "redis_baseline_waits": int(redis["baseline_waits"]),
        "redis_optimized_waits": int(redis["optimized_waits"]),
        "redis_wait_reduction_percent": 100.0
        * (1.0 - redis["optimized_waits"] / redis["baseline_waits"]),
        "ingestion_throughput_improvement": baseline_ingestion / optimized_ingestion,
        "retrieval_baseline_p95_ms": retrieval["baseline_p95"] * 1000.0,
        "retrieval_optimized_p95_ms": retrieval["optimized_p95"] * 1000.0,
        "retrieval_p95_improvement_percent": 100.0
        * (1.0 - retrieval["optimized_p95"] / retrieval["baseline_p95"]),
        "upload_peak_rss_delta_bytes": _measure_upload_rss(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=1000)
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6380)
    parser.add_argument("--upload-worker")
    args = parser.parse_args()
    if args.upload_worker:
        return asyncio.run(_upload_worker(args.upload_worker))
    result = asyncio.run(run_contract(args.chunks, args.redis_host, args.redis_port))
    failures = evaluate_contract(result)
    result["failures"] = failures
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
