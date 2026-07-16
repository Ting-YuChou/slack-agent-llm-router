#!/usr/bin/env python3
"""Measure and enforce the RAG ingestion, retrieval, and upload contracts."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.chunker import DocumentChunk
from src.rag.service import IngestionJob, RagService
from src.rag.vector_store import RedisStackRagVectorStore


class _MeasuredPipeline:
    def __init__(self, pipeline: Any, counter: "RedisOperationCounter"):
        self._pipeline = pipeline
        self._counter = counter

    def __getattr__(self, name: str):
        target = getattr(self._pipeline, name)
        if name == "execute":

            async def execute(*args: Any, **kwargs: Any):
                self._counter._record("pipeline.execute")
                return await target(*args, **kwargs)

            return execute

        def enqueue(*args: Any, **kwargs: Any):
            self._counter.operations[f"pipeline.{name}"] = (
                self._counter.operations.get(f"pipeline.{name}", 0) + 1
            )
            target(*args, **kwargs)
            return self

        return enqueue


class RedisOperationCounter:
    """Transparent Redis client proxy that counts observed await boundaries.

    This intentionally records calls made by the production vector-store code;
    it does not infer waits from chunk count or pipeline configuration.
    """

    def __init__(self, client: Any, *, latency_seconds: float = 0.0):
        self._client = client
        self._latency_seconds = latency_seconds
        self.network_waits = 0
        self.operations: Dict[str, int] = {}

    def _record(self, name: str) -> None:
        self.network_waits += 1
        self.operations[name] = self.operations.get(name, 0) + 1

    def pipeline(self, **kwargs: Any) -> _MeasuredPipeline:
        return _MeasuredPipeline(self._client.pipeline(**kwargs), self)

    def __getattr__(self, name: str):
        target = getattr(self._client, name)
        if not callable(target):
            return target

        async def measured(*args: Any, **kwargs: Any):
            self._record(name)
            if self._latency_seconds:
                await asyncio.sleep(self._latency_seconds)
            return await target(*args, **kwargs)

        return measured


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
        return [float(sum(text.encode()) % 7), 1.0, 0.0, 0.0]

    async def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        self.batch_calls += 1
        await asyncio.sleep(self.latency_seconds)
        return [[float(sum(text.encode()) % 7), 1.0, 0.0, 0.0] for text in texts]


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

    raw_client = redis.Redis(host=host, port=port, decode_responses=False)
    await raw_client.ping()
    baseline_counter = RedisOperationCounter(raw_client)
    prefix = f"ragbench:{uuid.uuid4().hex}"
    started = time.perf_counter()
    for index in range(count):
        key = f"{prefix}:legacy:chunk:{index}"
        await baseline_counter.hset(key, mapping={"text": str(index)})
        await baseline_counter.sadd(f"{prefix}:legacy:document", str(index))
        await baseline_counter.sadd(f"{prefix}:legacy:knowledge_bases", "school")
    baseline_seconds = time.perf_counter() - started
    baseline_waits = baseline_counter.network_waits

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
    optimized_counter = RedisOperationCounter(store.client)
    store.client = optimized_counter
    chunks = _chunks(count, "optimized")
    embeddings = [[float(index % 7), 1.0, 0.0, 0.0] for index in range(count)]
    started = time.perf_counter()
    written = await store.upsert_chunks(chunks, embeddings, knowledge_base_id="school")
    optimized_seconds = time.perf_counter() - started
    if written != count:
        raise RuntimeError("Redis benchmark wrote an unexpected chunk count")
    optimized_waits = optimized_counter.network_waits
    optimized_operations = dict(optimized_counter.operations)
    await store.shutdown()
    keys = [key async for key in raw_client.scan_iter(match=f"{prefix}:*")]
    if keys:
        await raw_client.delete(*keys)
    await raw_client.aclose()
    return {
        "baseline_seconds": baseline_seconds,
        "optimized_seconds": optimized_seconds,
        "baseline_waits": float(baseline_waits),
        "optimized_waits": float(optimized_waits),
        "optimized_operations": optimized_operations,
    }


class _StaticParser:
    def parse_bytes(self, **_kwargs: Any) -> object:
        return object()


class _StaticChunker:
    def __init__(self, chunks: Sequence[DocumentChunk]):
        self._chunks = list(chunks)

    def chunk(self, _parsed: object) -> List[DocumentChunk]:
        return list(self._chunks)


async def _production_ingestion_run(
    count: int,
    host: str,
    port: int,
    *,
    embedding_batch_size: int,
    pipeline_batch_size: int,
) -> Dict[str, Any]:
    prefix = f"ragingestionbench:{uuid.uuid4().hex}"
    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 4},
            "redis": {
                "host": host,
                "port": port,
                "key_prefix": prefix,
                "pipeline_batch_size": pipeline_batch_size,
            },
            "indexing": {"atomic_commit_max_chunks": max(2000, count)},
        }
    )
    await store.initialize()
    counter = RedisOperationCounter(store.client)
    store.client = counter
    provider = _LatencyEmbeddingProvider()
    service = RagService(
        {
            "enabled": True,
            "backend": "redis",
            "default_knowledge_base_ids": ["school"],
            "embedding": {
                "dimensions": 4,
                "batch_size": embedding_batch_size,
                "max_concurrent_batches": 1,
            },
        },
        parser=_StaticParser(),
        vector_store=store,
        embedding_provider=provider,
    )
    service.chunker = _StaticChunker(_chunks(count, "ingestion-document"))
    started = time.perf_counter()
    job = await service.ingest_document(
        content=b"benchmark",
        filename="benchmark.txt",
        knowledge_base_id="school",
        document_id="ingestion-document",
    )
    elapsed = time.perf_counter() - started
    if job.status != "completed" or job.chunks_indexed != count:
        raise RuntimeError(f"production ingestion did not complete: {job.to_dict()}")
    raw_client = counter._client
    operations = dict(counter.operations)
    waits = counter.network_waits
    await raw_client.execute_command("FT.DROPINDEX", store.index_name, "DD")
    await service.shutdown()
    return {
        "seconds": elapsed,
        "embedding_requests": provider.batch_calls,
        "redis_waits": waits,
        "redis_operations": operations,
    }


async def _production_ingestion_benchmark(
    count: int, host: str, port: int
) -> Dict[str, Any]:
    baseline = await _production_ingestion_run(
        count,
        host,
        port,
        embedding_batch_size=1,
        pipeline_batch_size=1,
    )
    optimized = await _production_ingestion_run(
        count,
        host,
        port,
        embedding_batch_size=32,
        pipeline_batch_size=64,
    )
    return {"baseline": baseline, "optimized": optimized}


async def _retrieval_benchmark(
    host: str, port: int, samples: int = 30
) -> Dict[str, float]:
    prefix = f"ragretrievalbench:{uuid.uuid4().hex}"
    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 4},
            "redis": {"host": host, "port": port, "key_prefix": prefix},
            "retrieval": {
                "max_concurrent_branches": 3,
                "deadline_seconds": 1,
                "keyword_scorers": [],
            },
        }
    )
    await store.initialize()
    # Exercise real Redis commands while applying a fixed 10 ms network RTT so
    # the concurrency contract is stable across very fast local machines.
    simulated_network_rtt_seconds = 0.01
    store.client = RedisOperationCounter(
        store.client, latency_seconds=simulated_network_rtt_seconds
    )
    chunks = [
        DocumentChunk(
            chunk_id=f"retrieve-{index}",
            document_id="retrieval-document",
            text=f"tuition policy section {index}",
            page_start=1,
            page_end=1,
            block_ids=[f"block-{index}"],
            block_types=["text"],
        )
        for index in range(30)
    ]
    embeddings = [[float(index % 7), 1.0, 0.0, 0.0] for index in range(30)]
    await store.upsert_chunks(chunks, embeddings, knowledge_base_id="school")
    service = RagService(
        {
            "enabled": True,
            "backend": "redis",
            "default_knowledge_base_ids": ["school"],
            "embedding": {"dimensions": 4},
            "retrieval": {"top_k": 5, "candidate_count": 30},
        },
        vector_store=store,
        embedding_provider=_LatencyEmbeddingProvider(),
    )
    baseline = []
    optimized = []
    baseline_order: List[str] = []
    optimized_order: List[str] = []
    for _ in range(samples):
        store.retrieval_config["max_concurrent_branches"] = 1
        started = time.perf_counter()
        baseline_results = await service.retrieve("tuition policy")
        baseline.append(time.perf_counter() - started)

        store.retrieval_config["max_concurrent_branches"] = 3
        started = time.perf_counter()
        optimized_results = await service.retrieve("tuition policy")
        optimized.append(time.perf_counter() - started)
        baseline_order = [item.chunk.chunk_id for item in baseline_results]
        optimized_order = [item.chunk.chunk_id for item in optimized_results]
        if baseline_order != optimized_order:
            raise RuntimeError("parallel retrieval changed production result ordering")
    result = {
        "baseline_p95": _percentile(baseline, 0.95),
        "optimized_p95": _percentile(optimized, 0.95),
        "result_count": float(len(optimized_order)),
        "simulated_network_rtt_seconds": simulated_network_rtt_seconds,
    }
    raw_client = store.client._client
    await raw_client.execute_command("FT.DROPINDEX", store.index_name, "DD")
    await store.shutdown()
    return result


def _percentile(values: Sequence[float], ratio: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


async def _upload_worker(directory: str) -> int:
    # Import and construct the application before the RSS baseline so the
    # measurement covers multipart parsing + the production endpoint/stager,
    # rather than module import or app setup.
    import main as router_main
    import uvicorn

    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {"staging_dir": directory},
            "ingestion_queue": {"enabled": True},
            "upload": {
                "multipart_max_bytes": 100_000_000,
                "stream_chunk_bytes": 1_048_576,
            },
        }
    )

    async def queue_document_ingestion(**payload: Any) -> IngestionJob:
        return IngestionJob(
            job_id="rss-job",
            document_id=payload["document_id"],
            filename=payload["filename"],
            knowledge_base_id=payload.get("knowledge_base_id") or "school",
            storage_ref=payload.get("storage_ref"),
        )

    service.queue_document_ingestion = queue_document_ingestion
    platform = router_main.LLMRouterPlatform()
    platform.config.setdefault("security", {}).setdefault("api_keys", {})[
        "enabled"
    ] = False
    platform.config.setdefault("api", {}).setdefault("rate_limiting", {})[
        "enabled"
    ] = False
    platform.services["rag"] = service
    app = platform._create_fastapi_app()
    source_path = Path(directory) / "multipart-source.bin"
    with source_path.open("wb") as source:
        source.truncate(100_000_000)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="error",
            access_log=False,
        )
    )
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        if server_task.done():
            await server_task
            raise RuntimeError("multipart benchmark API failed to start")
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    import psutil

    process_metrics = psutil.Process(os.getpid())
    before = process_metrics.memory_info().rss
    peak = before
    sampling_done = asyncio.Event()

    async def sample_rss() -> None:
        nonlocal peak
        while not sampling_done.is_set():
            peak = max(peak, process_metrics.memory_info().rss)
            await asyncio.sleep(0.005)

    sampler = asyncio.create_task(sample_rss())
    process = await asyncio.create_subprocess_exec(
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--form",
        f"file=@{source_path};type=application/octet-stream;filename=large.bin",
        "--form",
        "knowledge_base_id=school",
        "--form",
        "document_id=rss",
        f"http://127.0.0.1:{port}/rag/documents",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    sampling_done.set()
    await sampler
    if process.returncode != 0:
        raise RuntimeError(
            "multipart API benchmark failed: "
            f"curl={process.returncode} stdout={stdout.decode()} stderr={stderr.decode()}"
        )
    server.should_exit = True
    await server_task
    print(json.dumps({"rss_delta_bytes": max(0, peak - before)}))
    return 0


def _measure_upload_rss() -> int:
    with tempfile.TemporaryDirectory() as directory:
        completed = subprocess.run(
            [sys.executable, __file__, "--upload-worker", directory],
            check=True,
            capture_output=True,
            text=True,
        )
    payload = json.loads(
        next(line for line in reversed(completed.stdout.splitlines()) if line.strip())
    )
    return int(payload["rss_delta_bytes"])


async def run_contract(chunks: int, host: str, port: int) -> Dict[str, Any]:
    ingestion = await _production_ingestion_benchmark(chunks, host, port)
    baseline = ingestion["baseline"]
    optimized = ingestion["optimized"]
    retrieval = await _retrieval_benchmark(host, port)
    return {
        "chunks": chunks,
        "errors": 0,
        "embedding_baseline_requests": int(baseline["embedding_requests"]),
        "embedding_optimized_requests": int(optimized["embedding_requests"]),
        "embedding_request_reduction_percent": 100.0
        * (1.0 - optimized["embedding_requests"] / baseline["embedding_requests"]),
        "redis_baseline_waits": int(baseline["redis_waits"]),
        "redis_optimized_waits": int(optimized["redis_waits"]),
        "redis_optimized_operations": optimized["redis_operations"],
        "redis_wait_reduction_percent": 100.0
        * (1.0 - optimized["redis_waits"] / baseline["redis_waits"]),
        "ingestion_baseline_seconds": baseline["seconds"],
        "ingestion_optimized_seconds": optimized["seconds"],
        "ingestion_throughput_improvement": baseline["seconds"] / optimized["seconds"],
        "retrieval_baseline_p95_ms": retrieval["baseline_p95"] * 1000.0,
        "retrieval_optimized_p95_ms": retrieval["optimized_p95"] * 1000.0,
        "retrieval_simulated_network_rtt_ms": (
            retrieval["simulated_network_rtt_seconds"] * 1000.0
        ),
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
