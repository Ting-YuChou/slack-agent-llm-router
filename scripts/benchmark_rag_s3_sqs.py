#!/usr/bin/env python3
"""Benchmark the multi-stage S3 + SQS RAG ingestion path.

The corpus is uploaded before the measured interval. The measured clock starts
immediately before the concurrent ``/complete`` burst and stops when every job
is terminal. AWS execution is opt-in; calculation and report helpers are kept
importable for unit tests.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import httpx


TERMINAL_STATUSES = {"completed", "completed_with_warnings", "dead_lettered"}
SUCCESS_STATUSES = {"completed", "completed_with_warnings"}


def measured_worker_orders() -> List[List[int]]:
    return [[1, 2, 4, 8], [8, 4, 2, 1], [2, 8, 1, 4]]


def run_timeout_seconds(single_worker_pilot_seconds: float) -> float:
    return max(1800.0, 3.0 * float(single_worker_pilot_seconds) * 10.0)


@dataclass(frozen=True)
class CorpusEntry:
    path: Path
    filename: str
    format: str
    size_bytes: int
    checksum_sha256: str
    page_count: int


@dataclass
class PreparedJob:
    entry: CorpusEntry
    job: Dict[str, Any]
    upload_seconds: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without a heavy dependency."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def job_latency_seconds(job: Mapping[str, Any]) -> Optional[float]:
    started = _parse_timestamp(job.get("dispatch_started_at"))
    terminal = _parse_timestamp(job.get("terminal_at"))
    if started is None or terminal is None:
        return None
    latency = (terminal - started).total_seconds()
    if latency < 0:
        raise ValueError("terminal_at precedes dispatch_started_at")
    return latency


def summarize_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    completion_seconds: float,
    duplicate_commit_count: float,
    duplicate_noop_count: float = 0,
    duplicate_delivery_count: float = 0,
    index_commit_count: float = 0,
    peak_visible_depth: int,
    peak_total_depth: int,
) -> Dict[str, Any]:
    successful = [job for job in jobs if job.get("status") in SUCCESS_STATUSES]
    failures = [
        job
        for job in jobs
        if job.get("status") in {"dead_lettered", "benchmark_timeout"}
    ]
    dead_lettered = [job for job in jobs if job.get("status") == "dead_lettered"]
    timed_out = [job for job in jobs if job.get("status") == "benchmark_timeout"]
    latencies = []
    latency_errors = []
    for job in jobs:
        try:
            latency = job_latency_seconds(job)
        except (TypeError, ValueError) as exc:
            latency_errors.append(f"{job.get('job_id')}: {exc}")
            continue
        if latency is None:
            latency_errors.append(f"{job.get('job_id')}: missing benchmark timestamp")
            continue
        latencies.append(latency)
    minutes = completion_seconds / 60.0
    successful_bytes = sum(int(job.get("size_bytes") or 0) for job in successful)
    total = len(jobs)
    return {
        "total_documents": total,
        "successful_documents": len(successful),
        "failed_documents": len(failures),
        "dead_lettered_documents": len(dead_lettered),
        "timeout_documents": len(timed_out),
        "dead_lettered_attempts": [
            int(job.get("attempts") or 0) for job in dead_lettered
        ],
        "completion_seconds": float(completion_seconds),
        "docs_per_minute": len(successful) / minutes if minutes > 0 else 0.0,
        "mb_per_minute": (
            successful_bytes / 1_000_000.0 / minutes if minutes > 0 else 0.0
        ),
        "latency_seconds": {
            "p50": calculate_percentile(latencies, 50),
            "p95": calculate_percentile(latencies, 95),
            "p99": calculate_percentile(latencies, 99),
        },
        "latencies": latencies,
        "latency_sample_count": len(latencies),
        "latency_validation_errors": latency_errors,
        "retry_rate": (
            sum(int(job.get("attempts") or 0) > 1 for job in jobs) / total
            if total
            else 0.0
        ),
        "failure_rate": len(failures) / total if total else 0.0,
        "duplicate_commit_count": max(
            float(duplicate_commit_count),
            float(index_commit_count) - len(successful),
        ),
        "duplicate_noop_count": float(duplicate_noop_count),
        "duplicate_delivery_count": float(duplicate_delivery_count),
        "index_commit_count": float(index_commit_count),
        "peak_visible_queue_depth": int(peak_visible_depth),
        "peak_total_queue_depth": int(peak_total_depth),
    }


def summarize_worker_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not runs:
        raise ValueError("at least one measured run is required")

    def distribution(name: str) -> Dict[str, Any]:
        values = [float(run[name]) for run in runs]
        return {
            "median": statistics.median(values),
            "range": [min(values), max(values)],
        }

    pooled = [
        float(value) for run in runs for value in list(run.get("latencies") or [])
    ]
    return {
        "run_count": len(runs),
        "docs_per_minute": distribution("docs_per_minute"),
        "mb_per_minute": distribution("mb_per_minute")
        if all("mb_per_minute" in run for run in runs)
        else None,
        "completion_seconds": distribution("completion_seconds"),
        "pooled_latency_seconds": {
            "p50": calculate_percentile(pooled, 50),
            "p95": calculate_percentile(pooled, 95),
            "p99": calculate_percentile(pooled, 99),
        },
    }


def determine_scaling_knee(
    docs_per_minute: Mapping[int, float],
) -> Union[int, str]:
    if not docs_per_minute:
        raise ValueError("worker throughput is required")
    workers = sorted(docs_per_minute)
    baseline_workers = workers[0]
    baseline = float(docs_per_minute[baseline_workers])
    for current, doubled in zip(workers, workers[1:]):
        if doubled != current * 2:
            raise ValueError("worker counts must be consecutive doublings")
        current_rate = float(docs_per_minute[current])
        doubled_rate = float(docs_per_minute[doubled])
        gain = doubled_rate / current_rate - 1.0 if current_rate > 0 else 0.0
        efficiency = (
            doubled_rate / (baseline * doubled / baseline_workers)
            if baseline > 0
            else 0.0
        )
        if gain < 0.30 or efficiency < 0.60:
            return current
    return f">{workers[-1]}"


def parse_sqs_attributes(
    attributes: Mapping[str, Any], *, sampled_at: Optional[str] = None
) -> Dict[str, Any]:
    visible = int(attributes.get("ApproximateNumberOfMessages") or 0)
    in_flight = int(attributes.get("ApproximateNumberOfMessagesNotVisible") or 0)
    return {
        "sampled_at": sampled_at or utc_now(),
        "visible_depth": visible,
        "in_flight_depth": in_flight,
        "total_depth": visible + in_flight,
        "delayed_depth": int(attributes.get("ApproximateNumberOfMessagesDelayed") or 0),
        "oldest_message_age_seconds": int(
            attributes.get("ApproximateAgeOfOldestMessage") or 0
        ),
    }


def parse_prometheus_counter(text: str, metric_name: str) -> float:
    """Sum every label-set for a Prometheus counter family."""
    total = 0.0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name != metric_name:
            continue
        try:
            total += float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid Prometheus sample: {line}") from exc
    return total


def validate_manifest(
    manifest_path: Union[str, Path], *, expected_documents: Optional[int] = None
) -> List[CorpusEntry]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("manifest must contain a non-empty documents list")
    if expected_documents is not None and len(documents) != expected_documents:
        raise ValueError(
            f"manifest contains {len(documents)} documents; "
            f"expected {expected_documents}"
        )
    entries: List[CorpusEntry] = []
    for index, raw in enumerate(documents):
        document_path = Path(str(raw["path"])).expanduser().resolve()
        if not document_path.is_file():
            raise ValueError(f"document {index} does not exist: {document_path}")
        actual_size = document_path.stat().st_size
        expected_size = int(raw["size_bytes"])
        if actual_size != expected_size:
            raise ValueError(
                f"document {index} size mismatch: {actual_size} != {expected_size}"
            )
        actual_checksum = base64.b64encode(
            hashlib.sha256(document_path.read_bytes()).digest()
        ).decode("ascii")
        expected_checksum = str(raw["checksum_sha256"])
        if actual_checksum != expected_checksum:
            raise ValueError(f"document {index} checksum mismatch")
        entries.append(
            CorpusEntry(
                path=document_path,
                filename=str(raw.get("filename") or document_path.name),
                format=str(raw.get("format") or document_path.suffix.lstrip(".")),
                size_bytes=actual_size,
                checksum_sha256=actual_checksum,
                page_count=int(raw.get("page_count") or 0),
            )
        )
    return entries


REQUIRED_RESOURCE_KEYS = {
    "ecs_cpu",
    "ecs_memory",
    "redis_cpu",
    "redis_latency",
    "embedding_qps",
    "embedding_p95",
    "embedding_errors",
}


def load_cloudwatch_queries(path: Union[str, Path]) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("CloudWatch config must contain a non-empty queries list")
    normalized = []
    for query in queries:
        item = dict(query)
        for required in ("key", "namespace", "metric_name", "dimensions"):
            if required not in item:
                raise ValueError(f"CloudWatch query is missing {required}")
        item["stat"] = str(item.get("stat") or "Average")
        item["period_seconds"] = int(item.get("period_seconds") or 60)
        normalized.append(item)
    missing = REQUIRED_RESOURCE_KEYS - {str(query["key"]) for query in normalized}
    if missing:
        raise ValueError(
            "CloudWatch config is missing required signals: "
            + ", ".join(sorted(missing))
        )
    return normalized


class CloudWatchResourceSampler:
    def __init__(
        self,
        queries: Sequence[Mapping[str, Any]],
        *,
        region: Optional[str] = None,
    ):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "CloudWatch resource sampling requires the boto3 dependency"
            ) from exc
        self.client = boto3.client("cloudwatch", region_name=region)
        self.queries = [dict(query) for query in queries]
        self.interval_started_at: Optional[datetime] = None
        self.interval_ended_at: Optional[datetime] = None

    def start_interval(self, started_at: datetime) -> None:
        self.interval_started_at = started_at
        self.interval_ended_at = None

    def finish_interval(self, ended_at: datetime) -> None:
        self.interval_ended_at = ended_at

    async def sample(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        interval_start = self.interval_started_at or now
        row: Dict[str, Any] = {"sampled_at": now.isoformat()}
        for query in self.queries:
            stat = str(query["stat"])
            period_seconds = int(query["period_seconds"])
            kwargs: Dict[str, Any] = {
                "Namespace": str(query["namespace"]),
                "MetricName": str(query["metric_name"]),
                "Dimensions": [
                    {"Name": str(name), "Value": str(value)}
                    for name, value in dict(query["dimensions"]).items()
                ],
                "StartTime": interval_start - timedelta(seconds=period_seconds),
                "EndTime": now,
                "Period": period_seconds,
            }
            if stat.lower().startswith("p"):
                kwargs["ExtendedStatistics"] = [stat]
            else:
                kwargs["Statistics"] = [stat]
            response = await asyncio.to_thread(
                self.client.get_metric_statistics, **kwargs
            )
            datapoints = sorted(
                (
                    point
                    for point in (response.get("Datapoints") or [])
                    if point.get("Timestamp") is not None
                    and point["Timestamp"] + timedelta(seconds=period_seconds)
                    > interval_start
                    and (
                        self.interval_ended_at is None
                        or point["Timestamp"] < self.interval_ended_at
                    )
                ),
                key=lambda point: point.get("Timestamp"),
            )
            if datapoints:
                point = datapoints[-1]
                if stat.lower().startswith("p"):
                    value = dict(point.get("ExtendedStatistics") or {}).get(stat)
                else:
                    value = point.get(stat)
                if value is not None:
                    key = str(query["key"])
                    row[key] = value
                    row[f"{key}_timestamp"] = point["Timestamp"].isoformat()
        return row


class AwsControlPlane:
    def __init__(self, *, region: Optional[str] = None):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "AWS benchmark execution requires the boto3 dependency"
            ) from exc
        self.sqs = boto3.client("sqs", region_name=region)
        self.ecs = boto3.client("ecs", region_name=region)
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self._age_cache: Dict[str, tuple[float, int]] = {}

    async def _oldest_message_age(self, queue_url: str) -> int:
        cached = self._age_cache.get(queue_url)
        now_monotonic = time.monotonic()
        if cached and now_monotonic - cached[0] < 15.0:
            return cached[1]
        now = datetime.now(timezone.utc)
        response = await asyncio.to_thread(
            self.cloudwatch.get_metric_statistics,
            Namespace="AWS/SQS",
            MetricName="ApproximateAgeOfOldestMessage",
            Dimensions=[{"Name": "QueueName", "Value": queue_url.rsplit("/", 1)[-1]}],
            StartTime=now - timedelta(minutes=5),
            EndTime=now,
            Period=60,
            Statistics=["Maximum"],
        )
        datapoints = sorted(
            response.get("Datapoints") or [],
            key=lambda point: point.get("Timestamp"),
        )
        age = int(datapoints[-1].get("Maximum") or 0) if datapoints else 0
        self._age_cache[queue_url] = (now_monotonic, age)
        return age

    async def queue_sample(self, queue_url: str) -> Dict[str, Any]:
        names = [
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
            "ApproximateNumberOfMessagesDelayed",
        ]
        response = await asyncio.to_thread(
            self.sqs.get_queue_attributes,
            QueueUrl=queue_url,
            AttributeNames=names,
        )
        attributes = dict(response.get("Attributes") or {})
        attributes["ApproximateAgeOfOldestMessage"] = str(
            await self._oldest_message_age(queue_url)
        )
        return parse_sqs_attributes(attributes)

    async def assert_queue_empty(self, queue_url: str, label: str) -> None:
        sample = await self.queue_sample(queue_url)
        if sample["total_depth"] or sample["delayed_depth"]:
            raise RuntimeError(f"{label} is not empty: {sample}")

    async def wait_for_in_flight(
        self,
        queue_url: str,
        *,
        minimum_depth: int = 1,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        latest: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            latest = await self.queue_sample(queue_url)
            if int(latest.get("in_flight_depth") or 0) >= minimum_depth:
                return latest
            await asyncio.sleep(poll_interval_seconds)
        raise RuntimeError(
            "no in-flight SQS delivery appeared before worker-stop timeout: "
            f"{latest}"
        )

    async def scale_ecs(
        self,
        *,
        cluster: str,
        service: str,
        workers: int,
        task_definition: Optional[str],
    ) -> None:
        kwargs: Dict[str, Any] = {
            "cluster": cluster,
            "service": service,
            "desiredCount": workers,
        }
        if task_definition:
            kwargs["taskDefinition"] = task_definition
            kwargs["forceNewDeployment"] = True
        await asyncio.to_thread(self.ecs.update_service, **kwargs)
        waiter = self.ecs.get_waiter("services_stable")
        await asyncio.to_thread(
            waiter.wait,
            cluster=cluster,
            services=[service],
            WaiterConfig={"Delay": 15, "MaxAttempts": 80},
        )
        response = await asyncio.to_thread(
            self.ecs.describe_services, cluster=cluster, services=[service]
        )
        current = response["services"][0]
        if int(current.get("runningCount") or 0) != workers:
            raise RuntimeError(f"ECS service did not reach {workers} healthy tasks")

    async def send_duplicate(self, queue_url: str, payload: Mapping[str, Any]) -> None:
        await asyncio.to_thread(
            self.sqs.send_message,
            QueueUrl=queue_url,
            MessageBody=json.dumps(payload, separators=(",", ":")),
        )

    async def stop_one_task(self, *, cluster: str, service: str) -> str:
        response = await asyncio.to_thread(
            self.ecs.list_tasks, cluster=cluster, serviceName=service
        )
        tasks = list(response.get("taskArns") or [])
        if not tasks:
            raise RuntimeError("no ECS worker task is available to stop")
        task = tasks[0]
        await asyncio.to_thread(
            self.ecs.stop_task,
            cluster=cluster,
            task=task,
            reason="RAG benchmark visibility-redelivery scenario",
        )
        return task


class RagBenchmarkRunner:
    def __init__(
        self,
        *,
        api_url: str,
        queue_url: str,
        dlq_url: str,
        api_headers: Optional[Mapping[str, str]] = None,
        region: Optional[str] = None,
        poll_interval_seconds: float = 0.25,
        request_timeout_seconds: float = 60.0,
        job_poll_interval_seconds: float = 2.0,
        job_poll_concurrency: int = 50,
        metrics_url: Optional[str] = None,
        http_transport: Optional[httpx.AsyncBaseTransport] = None,
        resource_sampler: Optional[CloudWatchResourceSampler] = None,
        resource_interval_seconds: float = 15.0,
        resource_backfill_timeout_seconds: float = 120.0,
        aws: Optional[AwsControlPlane] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.queue_url = queue_url
        self.dlq_url = dlq_url
        self.api_headers = dict(api_headers or {})
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.job_poll_interval_seconds = job_poll_interval_seconds
        self.job_poll_concurrency = job_poll_concurrency
        self.metrics_url = metrics_url
        self.http_transport = http_transport
        self.resource_sampler = resource_sampler
        self.resource_interval_seconds = resource_interval_seconds
        self.resource_backfill_timeout_seconds = resource_backfill_timeout_seconds
        self.aws = aws or AwsControlPlane(region=region)

    def _client(self, *, upload: bool = False) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout_seconds),
            transport=self.http_transport,
            headers={} if upload else self.api_headers,
        )

    async def _duplicate_counters(self) -> Dict[str, float]:
        if not self.metrics_url:
            return {"noops": 0.0, "deliveries": 0.0, "index_commits": 0.0}
        async with self._client() as client:
            response = await client.get(self.metrics_url)
            response.raise_for_status()
            return {
                "noops": parse_prometheus_counter(
                    response.text, "llm_router_rag_duplicate_noops_total"
                ),
                "deliveries": parse_prometheus_counter(
                    response.text, "llm_router_rag_duplicate_deliveries_total"
                ),
                "index_commits": parse_prometheus_counter(
                    response.text, "llm_router_rag_index_commits_total"
                ),
            }

    async def _wait_for_queue_settle(
        self, *, expected_dlq_count: int, timeout_seconds: float
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        source: Dict[str, Any] = {}
        dlq: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            source = await self.aws.queue_sample(self.queue_url)
            dlq = await self.aws.queue_sample(self.dlq_url)
            if (
                source["total_depth"] == 0
                and source["delayed_depth"] == 0
                and dlq["total_depth"] == expected_dlq_count
            ):
                return source, dlq
            await asyncio.sleep(self.poll_interval_seconds)
        return source, dlq

    async def _create_and_upload(
        self,
        client: httpx.AsyncClient,
        upload_client: httpx.AsyncClient,
        entry: CorpusEntry,
        *,
        run_id: str,
        knowledge_base_id: str,
        ordinal: int,
        created_jobs_sink: List[Dict[str, Any]],
    ) -> PreparedJob:
        response = await client.post(
            f"{self.api_url}/rag/uploads",
            headers={
                **self.api_headers,
                "Idempotency-Key": f"benchmark:{run_id}:{ordinal}",
            },
            json={
                "filename": entry.filename,
                "size_bytes": entry.size_bytes,
                "checksum_sha256": entry.checksum_sha256,
                "knowledge_base_id": knowledge_base_id,
                "document_id": f"benchmark-{run_id}-{ordinal}",
                "metadata": {
                    "benchmark_run_id": run_id,
                    "benchmark_ordinal": ordinal,
                    "format": entry.format,
                    "page_count": entry.page_count,
                },
            },
        )
        response.raise_for_status()
        created = response.json()
        created_jobs_sink.append(dict(created["job"]))
        upload = created["upload"]
        started = time.perf_counter()
        payload = await asyncio.to_thread(entry.path.read_bytes)
        put = await upload_client.request(
            str(upload.get("method") or "PUT"),
            str(upload["url"]),
            headers=dict(upload.get("headers") or {}),
            content=payload,
        )
        put.raise_for_status()
        return PreparedJob(
            entry=entry,
            job=dict(created["job"]),
            upload_seconds=time.perf_counter() - started,
        )

    async def prepare_uploads(
        self,
        entries: Sequence[CorpusEntry],
        *,
        run_id: str,
        knowledge_base_id: str,
        concurrency: int = 16,
        created_jobs_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PreparedJob]:
        created_jobs_sink = created_jobs_sink if created_jobs_sink is not None else []
        semaphore = asyncio.Semaphore(concurrency)
        async with self._client() as client, self._client(upload=True) as upload_client:

            async def prepare(index: int, entry: CorpusEntry) -> PreparedJob:
                async with semaphore:
                    return await self._create_and_upload(
                        client,
                        upload_client,
                        entry,
                        run_id=run_id,
                        knowledge_base_id=knowledge_base_id,
                        ordinal=index,
                        created_jobs_sink=created_jobs_sink,
                    )

            tasks = [
                asyncio.create_task(prepare(index, entry))
                for index, entry in enumerate(entries)
            ]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    async def _sample_queue_until(
        self, stop: asyncio.Event, samples: List[Dict[str, Any]]
    ) -> None:
        while not stop.is_set():
            samples.append(await self.aws.queue_sample(self.queue_url))
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _sample_resources_until(
        self, stop: asyncio.Event, samples: List[Dict[str, Any]]
    ) -> None:
        if self.resource_sampler is None:
            return
        while not stop.is_set():
            samples.append(await self.resource_sampler.sample())
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self.resource_interval_seconds
                )
            except asyncio.TimeoutError:
                pass

    async def _backfill_resource_signals(self, samples: List[Dict[str, Any]]) -> None:
        if self.resource_sampler is None:
            return
        deadline = time.monotonic() + self.resource_backfill_timeout_seconds
        while True:
            coverage = {
                key
                for sample in samples
                for key, value in sample.items()
                if value is not None
            }
            if REQUIRED_RESOURCE_KEYS <= coverage or time.monotonic() >= deadline:
                return
            samples.append(await self.resource_sampler.sample())
            refreshed_coverage = {
                key
                for sample in samples
                for key, value in sample.items()
                if value is not None
            }
            if REQUIRED_RESOURCE_KEYS <= refreshed_coverage:
                return
            await asyncio.sleep(min(15.0, max(0.0, deadline - time.monotonic())))

    async def _complete_all(
        self,
        client: httpx.AsyncClient,
        prepared: Sequence[PreparedJob],
        *,
        dispatch_started_at: str,
    ) -> List[Dict[str, Any]]:
        async def complete(item: PreparedJob) -> Dict[str, Any]:
            job_id = item.job["job_id"]
            response = await client.post(
                f"{self.api_url}/rag/uploads/{job_id}/complete",
                headers={
                    **self.api_headers,
                    "X-RAG-Dispatch-Started-At": dispatch_started_at,
                },
            )
            response.raise_for_status()
            return dict(response.json())

        return await asyncio.gather(*(complete(item) for item in prepared))

    async def _poll_jobs(
        self,
        client: httpx.AsyncClient,
        prepared: Sequence[PreparedJob],
        *,
        timeout_seconds: float,
    ) -> List[Dict[str, Any]]:
        pending = {str(item.job["job_id"]): item for item in prepared}
        terminal: Dict[str, Dict[str, Any]] = {}
        deadline = time.monotonic() + timeout_seconds
        semaphore = asyncio.Semaphore(self.job_poll_concurrency)
        while pending and time.monotonic() < deadline:
            ids = list(pending)

            async def fetch(job_id: str) -> tuple[str, Dict[str, Any]]:
                async with semaphore:
                    response = await client.get(
                        f"{self.api_url}/rag/jobs/{job_id}",
                        headers=self.api_headers,
                    )
                    response.raise_for_status()
                    return job_id, dict(response.json())

            for job_id, job in await asyncio.gather(*(fetch(job_id) for job_id in ids)):
                if job.get("status") in TERMINAL_STATUSES:
                    job["size_bytes"] = pending[job_id].entry.size_bytes
                    terminal[job_id] = job
                    pending.pop(job_id)
            if pending:
                await asyncio.sleep(self.job_poll_interval_seconds)
        for job_id, item in pending.items():
            terminal[job_id] = {
                **item.job,
                "status": "benchmark_timeout",
                "size_bytes": item.entry.size_bytes,
            }
        return [terminal[str(item.job["job_id"])] for item in prepared]

    async def _fetch_jobs_once(
        self,
        client: httpx.AsyncClient,
        prepared: Sequence[PreparedJob],
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.job_poll_concurrency)

        async def fetch(item: PreparedJob) -> Dict[str, Any]:
            async with semaphore:
                response = await client.get(
                    f"{self.api_url}/rag/jobs/{item.job['job_id']}",
                    headers=self.api_headers,
                )
                response.raise_for_status()
                job = dict(response.json())
                job["size_bytes"] = item.entry.size_bytes
                return job

        return await asyncio.gather(*(fetch(item) for item in prepared))

    async def _cleanup_vector_documents(
        self, client: httpx.AsyncClient, jobs: Sequence[Mapping[str, Any]]
    ) -> int:
        semaphore = asyncio.Semaphore(16)

        async def delete(job: Mapping[str, Any]) -> int:
            async with semaphore:
                if job.get("status") not in TERMINAL_STATUSES:
                    return 0
                document_id = job.get("document_id")
                if not document_id:
                    return 0
                response = await client.delete(
                    f"{self.api_url}/rag/documents/{document_id}",
                    headers=self.api_headers,
                    params={"knowledge_base_id": job.get("knowledge_base_id")},
                )
                response.raise_for_status()
                return int(response.json().get("deleted_chunks") or 0)

        return sum(await asyncio.gather(*(delete(job) for job in jobs)))

    async def run(
        self,
        entries: Sequence[CorpusEntry],
        *,
        output_dir: Path,
        run_id: str,
        layer: str,
        workers: int,
        timeout_seconds: float,
        ecs_cluster: Optional[str] = None,
        ecs_service: Optional[str] = None,
        task_definition: Optional[str] = None,
        duplicate_messages: int = 0,
        stop_worker: bool = False,
        mode: str = "measured",
        repetition: Optional[int] = None,
        expected_dlq_count: int = 0,
    ) -> Dict[str, Any]:
        if not entries:
            raise ValueError("the selected corpus is empty")
        if bool(ecs_cluster) != bool(ecs_service):
            raise ValueError("ECS cluster and service must be supplied together")
        await self.aws.assert_queue_empty(self.queue_url, "source queue")
        await self.aws.assert_queue_empty(self.dlq_url, "DLQ")
        if ecs_cluster and ecs_service:
            await self.aws.scale_ecs(
                cluster=ecs_cluster,
                service=ecs_service,
                workers=workers,
                task_definition=task_definition,
            )
        knowledge_base_id = f"benchmark-{layer}-{run_id}"
        upload_started = time.perf_counter()
        created_upload_jobs: List[Dict[str, Any]] = []
        try:
            prepared = await self.prepare_uploads(
                entries,
                run_id=run_id,
                knowledge_base_id=knowledge_base_id,
                created_jobs_sink=created_upload_jobs,
            )
        except BaseException as exc:
            partial = {
                "schema_version": 1,
                "run_id": run_id,
                "layer": layer,
                "workers": workers,
                "mode": mode,
                "status": "upload_preparation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_objects_preserved": True,
                "created_upload_jobs": len(created_upload_jobs),
            }
            write_run_outputs(output_dir, partial, created_upload_jobs, [], [])
            raise
        upload_seconds = time.perf_counter() - upload_started
        queue_samples: List[Dict[str, Any]] = []
        resource_samples: List[Dict[str, Any]] = []
        stop_sampling = asyncio.Event()
        duplicate_before = await self._duplicate_counters()
        measured_wall_started_at = utc_now()
        if self.resource_sampler is not None:
            self.resource_sampler.start_interval(
                _parse_timestamp(measured_wall_started_at) or datetime.now(timezone.utc)
            )
        sampler = asyncio.create_task(
            self._sample_queue_until(stop_sampling, queue_samples)
        )
        resource_task = asyncio.create_task(
            self._sample_resources_until(stop_sampling, resource_samples)
        )
        jobs: List[Dict[str, Any]] = []
        completion_seconds = 0.0
        deleted_chunks = 0
        stopped_task = None
        final_queue: Dict[str, Any] = {}
        final_dlq: Dict[str, Any] = {}
        duplicate_after = dict(duplicate_before)
        failure: Optional[BaseException] = None
        try:
            async with self._client() as client:
                measured_start = time.perf_counter()
                completed_jobs = await self._complete_all(
                    client,
                    prepared,
                    dispatch_started_at=measured_wall_started_at,
                )
                if duplicate_messages:
                    for job in completed_jobs[:duplicate_messages]:
                        await self.aws.send_duplicate(
                            self.queue_url, _sqs_payload_from_job(job)
                        )
                if stop_worker:
                    if not ecs_cluster or not ecs_service:
                        raise ValueError(
                            "worker-stop scenario requires ECS cluster/service"
                        )
                    await self.aws.wait_for_in_flight(
                        self.queue_url, minimum_depth=workers
                    )
                    stopped_task = await self.aws.stop_one_task(
                        cluster=ecs_cluster, service=ecs_service
                    )
                jobs = await self._poll_jobs(
                    client, prepared, timeout_seconds=timeout_seconds
                )
                completion_seconds = time.perf_counter() - measured_start
                if self.resource_sampler is not None:
                    self.resource_sampler.finish_interval(datetime.now(timezone.utc))
            final_queue, final_dlq = await self._wait_for_queue_settle(
                expected_dlq_count=expected_dlq_count,
                timeout_seconds=timeout_seconds,
            )
            async with self._client() as client:
                if expected_dlq_count:
                    jobs = await self._fetch_jobs_once(client, prepared)
                deleted_chunks = await self._cleanup_vector_documents(client, jobs)
            duplicate_after = await self._duplicate_counters()
            if mode == "measured":
                await self._backfill_resource_signals(resource_samples)
        except BaseException as exc:
            failure = exc
        finally:
            stop_sampling.set()
            sampling_results = await asyncio.gather(
                sampler, resource_task, return_exceptions=True
            )
            if failure is None:
                sampling_failure = next(
                    (
                        result
                        for result in sampling_results
                        if isinstance(result, BaseException)
                    ),
                    None,
                )
                if sampling_failure is not None:
                    failure = sampling_failure
        if failure is not None:
            partial_jobs = jobs or [dict(item.job) for item in prepared]
            partial = {
                "schema_version": 1,
                "run_id": run_id,
                "layer": layer,
                "workers": workers,
                "mode": mode,
                "status": "benchmark_failed",
                "error_type": type(failure).__name__,
                "error": str(failure),
                "source_objects_preserved": True,
            }
            write_run_outputs(
                output_dir, partial, partial_jobs, queue_samples, resource_samples
            )
            raise failure
        queue_samples.append(await self.aws.queue_sample(self.queue_url))
        peak_visible = max(
            (sample["visible_depth"] for sample in queue_samples), default=0
        )
        peak_total = max((sample["total_depth"] for sample in queue_samples), default=0)
        summary = summarize_jobs(
            jobs,
            completion_seconds=completion_seconds,
            duplicate_commit_count=0,
            duplicate_noop_count=max(
                0.0, duplicate_after["noops"] - duplicate_before["noops"]
            ),
            duplicate_delivery_count=max(
                0.0,
                duplicate_after["deliveries"] - duplicate_before["deliveries"],
            ),
            index_commit_count=max(
                0.0,
                duplicate_after["index_commits"] - duplicate_before["index_commits"],
            ),
            peak_visible_depth=peak_visible,
            peak_total_depth=peak_total,
        )
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "layer": layer,
            "workers": workers,
            "mode": mode,
            "repetition": repetition,
            "knowledge_base_id": knowledge_base_id,
            "started_at": measured_wall_started_at,
            "upload": {
                "documents": len(prepared),
                "bytes": sum(item.entry.size_bytes for item in prepared),
                "elapsed_seconds": upload_seconds,
                "mb_per_minute": (
                    sum(item.entry.size_bytes for item in prepared)
                    / 1_000_000.0
                    / (upload_seconds / 60.0)
                    if upload_seconds > 0
                    else 0.0
                ),
                "max_single_upload_seconds": max(
                    item.upload_seconds for item in prepared
                ),
            },
            "summary": summary,
            "final_queue": final_queue,
            "final_dlq": final_dlq,
            "stopped_task": stopped_task,
            "cleanup": {"deleted_chunks": deleted_chunks},
            "injected_duplicate_messages": duplicate_messages,
            "expected_dlq_count": expected_dlq_count,
            "resource_signal_coverage": sorted(
                REQUIRED_RESOURCE_KEYS
                & {
                    key
                    for sample in resource_samples
                    for key, value in sample.items()
                    if value is not None
                }
            ),
        }
        write_run_outputs(output_dir, result, jobs, queue_samples, resource_samples)
        return result


def _sqs_payload_from_job(job: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "dispatch_id": job["dispatch_id"],
        "index_version": job.get("index_version"),
        "batch_id": job.get("batch_id") or "",
        "document_id": job["document_id"],
        "knowledge_base_id": job["knowledge_base_id"],
        "filename": job["filename"],
        "source": dict(job.get("source") or {}),
        "attempt": 1,
        "created_at": utc_now(),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def write_run_outputs(
    output_dir: Path,
    result: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    queue_samples: Sequence[Mapping[str, Any]],
    resource_samples: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "jobs.csv", jobs)
    _write_csv(output_dir / "queue.csv", queue_samples)
    _write_csv(output_dir / "resources.csv", resource_samples)
    _write_timeline_svg(output_dir / "queue_timeline.svg", queue_samples)
    _write_queue_resource_timeline_svg(
        output_dir / "queue_resource_timeline.svg",
        queue_samples,
        resource_samples,
    )


def _write_timeline_svg(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    width, height, pad = 900, 360, 45
    if not samples:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        return
    maximum = max(1, max(int(sample.get("total_depth") or 0) for sample in samples))

    def points(key: str) -> str:
        values = []
        denominator = max(1, len(samples) - 1)
        for index, sample in enumerate(samples):
            x = pad + index * (width - 2 * pad) / denominator
            y = height - pad - int(sample.get(key) or 0) * (height - 2 * pad) / maximum
            values.append(f"{x:.1f},{y:.1f}")
        return " ".join(values)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="25" font-family="sans-serif" font-size="16">SQS queue timeline</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#555"/>
<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points('visible_depth')}"/>
<polyline fill="none" stroke="#ea580c" stroke-width="2" points="{points('in_flight_depth')}"/>
<text x="{width-220}" y="25" fill="#2563eb" font-family="sans-serif">visible</text>
<text x="{width-135}" y="25" fill="#ea580c" font-family="sans-serif">in-flight</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_queue_resource_timeline_svg(
    path: Path,
    queue_samples: Sequence[Mapping[str, Any]],
    resource_samples: Sequence[Mapping[str, Any]],
) -> None:
    """Render normalized timelines; the CSV files retain exact units."""
    series: Dict[str, List[float]] = {
        "queue_visible": [
            float(sample.get("visible_depth") or 0) for sample in queue_samples
        ],
        "queue_in_flight": [
            float(sample.get("in_flight_depth") or 0) for sample in queue_samples
        ],
    }
    resource_keys = sorted(
        {
            key
            for sample in resource_samples
            for key, value in sample.items()
            if key != "sampled_at" and isinstance(value, (int, float))
        }
    )
    for key in resource_keys:
        series[key] = [
            float(sample[key])
            for sample in resource_samples
            if sample.get(key) is not None
        ]
    width, height, pad = 1000, 500, 55
    colors = [
        "#2563eb",
        "#ea580c",
        "#16a34a",
        "#9333ea",
        "#dc2626",
        "#0891b2",
        "#4f46e5",
        "#65a30d",
        "#c026d3",
    ]
    elements = [
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="28" font-family="sans-serif" font-size="18">Queue and resource saturation timeline (normalized)</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#555"/>',
    ]
    for index, (name, values) in enumerate(series.items()):
        if not values:
            continue
        maximum = max(1.0, max(values))
        denominator = max(1, len(values) - 1)
        points = " ".join(
            f"{pad + point_index * (width - 2 * pad) / denominator:.1f},"
            f"{height - pad - value * (height - 2 * pad) / maximum:.1f}"
            for point_index, value in enumerate(values)
        )
        color = colors[index % len(colors)]
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
        )
        elements.append(
            f'<text x="{width-235}" y="{25+index*17}" fill="{color}" font-family="sans-serif">{name}</text>'
        )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">' + "".join(elements) + "</svg>",
        encoding="utf-8",
    )


def _write_series_svg(
    path: Path,
    *,
    title: str,
    series: Mapping[str, Sequence[tuple[int, float]]],
) -> None:
    width, height, pad = 900, 420, 55
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        return
    xs = sorted({point[0] for point in all_points})
    maximum = max(1.0, max(point[1] for point in all_points))
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626"]

    def x_position(worker: int) -> float:
        return pad + xs.index(worker) * (width - 2 * pad) / max(1, len(xs) - 1)

    def y_position(value: float) -> float:
        return height - pad - value * (height - 2 * pad) / maximum

    elements = [
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="28" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#555"/>',
    ]
    for index, (name, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        polyline = " ".join(
            f"{x_position(worker):.1f},{y_position(value):.1f}"
            for worker, value in sorted(points)
        )
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>'
        )
        elements.append(
            f'<text x="{width-210}" y="{25+index*18}" fill="{color}" font-family="sans-serif">{name}</text>'
        )
    for worker in xs:
        elements.append(
            f'<text x="{x_position(worker)-4:.1f}" y="{height-25}" font-family="sans-serif">{worker}</text>'
        )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}">' + "".join(elements) + "</svg>",
        encoding="utf-8",
    )


def analyze_result_tree(output_root: Union[str, Path]) -> Dict[str, Any]:
    root = Path(output_root)
    grouped: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for summary_path in root.rglob("summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("mode", "measured") != "measured":
            continue
        violations = healthy_run_violations(payload, expected_documents=500)
        if violations:
            raise ValueError(
                f"unhealthy measured result {summary_path}: {', '.join(violations)}"
            )
        layer = str(payload["layer"])
        workers = int(payload["workers"])
        grouped.setdefault(layer, {}).setdefault(workers, []).append(
            dict(payload["summary"])
        )
    if not grouped:
        raise ValueError("no measured summary.json files were found")
    if set(grouped) != {"pipeline", "product"}:
        raise ValueError("aggregate analysis requires pipeline and product layers")
    for layer, worker_runs in grouped.items():
        if set(worker_runs) != {1, 2, 4, 8}:
            raise ValueError(f"{layer} is missing a 1/2/4/8 worker result")
        for workers, runs in worker_runs.items():
            if len(runs) != 3:
                raise ValueError(
                    f"{layer}/{workers} requires exactly three healthy runs"
                )
    report: Dict[str, Any] = {"layers": {}}
    throughput: Dict[str, List[tuple[int, float]]] = {}
    completion: Dict[str, List[tuple[int, float]]] = {}
    latency: Dict[str, List[tuple[int, float]]] = {}
    for layer, worker_runs in grouped.items():
        layer_workers: Dict[str, Any] = {}
        rates: Dict[int, float] = {}
        for workers, runs in sorted(worker_runs.items()):
            summarized = summarize_worker_runs(runs)
            layer_workers[str(workers)] = summarized
            median_rate = float(summarized["docs_per_minute"]["median"])
            rates[workers] = median_rate
            throughput.setdefault(f"{layer} docs/min", []).append(
                (workers, median_rate)
            )
            mb_summary = summarized.get("mb_per_minute")
            if mb_summary:
                throughput.setdefault(f"{layer} MB/min", []).append(
                    (workers, float(mb_summary["median"]))
                )
            completion.setdefault(layer, []).append(
                (workers, float(summarized["completion_seconds"]["median"]))
            )
            for percentile in ("p50", "p95", "p99"):
                latency.setdefault(f"{layer} {percentile}", []).append(
                    (workers, float(summarized["pooled_latency_seconds"][percentile]))
                )
        knee: Union[int, str, None] = None
        if len(rates) > 1:
            knee = determine_scaling_knee(rates)
        report["layers"][layer] = {"workers": layer_workers, "scaling_knee": knee}
    (root / "aggregate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_series_svg(
        root / "workers_throughput.svg",
        title="Workers vs throughput",
        series=throughput,
    )
    _write_series_svg(
        root / "workers_completion.svg",
        title="Workers vs completion time",
        series=completion,
    )
    _write_series_svg(
        root / "workers_latency.svg",
        title="Workers vs pooled latency percentiles",
        series=latency,
    )
    return report


def healthy_run_violations(
    result: Mapping[str, Any], *, expected_documents: int
) -> List[str]:
    summary = result["summary"]
    violations = []
    if int(summary["total_documents"]) != expected_documents:
        violations.append("not every benchmark job reached a recorded terminal result")
    expected_dlq_count = int(result.get("expected_dlq_count") or 0)
    if expected_dlq_count:
        if int(summary.get("dead_lettered_documents") or 0) != expected_dlq_count:
            violations.append("dead-lettered job count does not match expectation")
        if int(summary.get("timeout_documents") or 0) != 0:
            violations.append("DLQ resilience still contains benchmark timeouts")
        if any(
            int(attempts) < 3
            for attempts in (summary.get("dead_lettered_attempts") or [])
        ):
            violations.append("a DLQ job did not reach the third receive attempt")
    elif float(summary["failure_rate"]) != 0:
        violations.append("failure rate is non-zero")
    if float(summary["duplicate_commit_count"]) != 0:
        violations.append("duplicate commit count is non-zero")
    injected_duplicates = int(result.get("injected_duplicate_messages") or 0)
    observed_duplicates = float(summary.get("duplicate_noop_count") or 0) + float(
        summary.get("duplicate_delivery_count") or 0
    )
    if injected_duplicates and observed_duplicates < injected_duplicates:
        violations.append("not every injected duplicate delivery was observed")
    if not expected_dlq_count and float(summary["retry_rate"]) > 0.005:
        violations.append("retry rate exceeds 0.5%")
    if (
        int(summary.get("latency_sample_count") or 0) != expected_documents
        or len(summary.get("latencies") or []) != expected_documents
    ):
        violations.append("one or more jobs lack a valid end-to-end latency")
    if summary.get("latency_validation_errors"):
        violations.append("one or more job timestamps are invalid")
    final_queue = result.get("final_queue") or {}
    if (
        int(final_queue.get("total_depth") or 0) != 0
        or int(final_queue.get("delayed_depth") or 0) != 0
    ):
        violations.append("source queue did not drain")
    final_dlq = result.get("final_dlq") or {}
    if int(final_dlq.get("total_depth") or 0) != expected_dlq_count:
        violations.append("DLQ depth does not match the expected resilience outcome")
    if result.get("mode") == "measured":
        coverage = set(result.get("resource_signal_coverage") or [])
        missing_resources = REQUIRED_RESOURCE_KEYS - coverage
        if missing_resources:
            violations.append(
                "missing measured resource signals: "
                + ", ".join(sorted(missing_resources))
            )
    return violations


def _api_headers(args: argparse.Namespace) -> Dict[str, str]:
    if not args.api_key_env:
        return {}
    value = os.getenv(args.api_key_env)
    if not value:
        raise RuntimeError(f"{args.api_key_env} is not set")
    return {args.api_key_header: value}


def _build_runner(args: argparse.Namespace) -> RagBenchmarkRunner:
    resource_sampler = None
    if args.cloudwatch_config:
        resource_sampler = CloudWatchResourceSampler(
            load_cloudwatch_queries(args.cloudwatch_config), region=args.region
        )
    return RagBenchmarkRunner(
        api_url=args.api_url,
        queue_url=args.queue_url,
        dlq_url=args.dlq_url,
        api_headers=_api_headers(args),
        region=args.region,
        metrics_url=args.metrics_url,
        resource_sampler=resource_sampler,
        resource_backfill_timeout_seconds=args.resource_backfill_timeout_seconds,
    )


async def _run_command(args: argparse.Namespace) -> int:
    entries = validate_manifest(args.manifest, expected_documents=args.jobs)
    runner = _build_runner(args)
    result = await runner.run(
        entries,
        output_dir=Path(args.output_dir),
        run_id=args.run_id or uuid.uuid4().hex,
        layer=args.layer,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        ecs_cluster=args.ecs_cluster,
        ecs_service=args.ecs_service,
        task_definition=args.task_definition,
        duplicate_messages=args.duplicate_messages,
        stop_worker=args.stop_worker,
        mode=args.mode,
        repetition=args.repetition,
        expected_dlq_count=args.expected_dlq_count,
    )
    violations = healthy_run_violations(result, expected_documents=args.jobs)
    print(json.dumps(result, indent=2, sort_keys=True))
    if violations:
        print(json.dumps({"healthy_run_violations": violations}, indent=2))
        return 2
    return 0


async def _matrix_command(args: argparse.Namespace) -> int:
    entries = validate_manifest(args.manifest, expected_documents=500)
    if not args.cloudwatch_config:
        raise ValueError("matrix execution requires --cloudwatch-config")
    runner = _build_runner(args)
    root = Path(args.output_dir)
    timeout = run_timeout_seconds(args.pilot_seconds)
    task_definitions = {
        "pipeline": args.pipeline_task_definition,
        "product": args.product_task_definition,
    }
    for layer in ("pipeline", "product"):
        task_definition = task_definitions[layer]
        for workers in (1, 2, 4, 8):
            run_id = f"{layer}-warmup-w{workers}-{uuid.uuid4().hex[:8]}"
            await runner.run(
                entries[:20],
                output_dir=root / layer / f"warmup-w{workers}",
                run_id=run_id,
                layer=layer,
                workers=workers,
                timeout_seconds=timeout,
                ecs_cluster=args.ecs_cluster,
                ecs_service=args.ecs_service,
                task_definition=task_definition,
                mode="warmup",
            )
        for repetition, order in enumerate(measured_worker_orders(), start=1):
            for workers in order:
                run_id = f"{layer}-r{repetition}-w{workers}-{uuid.uuid4().hex[:8]}"
                result = await runner.run(
                    entries,
                    output_dir=(root / layer / f"measured-r{repetition}-w{workers}"),
                    run_id=run_id,
                    layer=layer,
                    workers=workers,
                    timeout_seconds=timeout,
                    ecs_cluster=args.ecs_cluster,
                    ecs_service=args.ecs_service,
                    task_definition=task_definition,
                    mode="measured",
                    repetition=repetition,
                )
                violations = healthy_run_violations(result, expected_documents=500)
                if violations:
                    raise RuntimeError(
                        f"unhealthy measured run {run_id}: {', '.join(violations)}"
                    )
    print(json.dumps(analyze_result_tree(root), indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--jobs", type=int, default=500)

    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--api-url", required=True)
    run.add_argument("--queue-url", required=True)
    run.add_argument("--dlq-url", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--layer", choices=("pipeline", "product"), required=True)
    run.add_argument("--workers", type=int, choices=(1, 2, 4, 8), required=True)
    run.add_argument("--jobs", type=int, default=500)
    run.add_argument("--timeout-seconds", type=float, default=1800)
    run.add_argument("--run-id")
    run.add_argument("--region")
    run.add_argument("--metrics-url")
    run.add_argument("--cloudwatch-config")
    run.add_argument("--resource-backfill-timeout-seconds", type=float, default=120)
    run.add_argument("--ecs-cluster")
    run.add_argument("--ecs-service")
    run.add_argument("--task-definition")
    run.add_argument("--api-key-env")
    run.add_argument("--api-key-header", default="X-API-Key")
    run.add_argument("--duplicate-messages", type=int, default=0)
    run.add_argument("--stop-worker", action="store_true")
    run.add_argument(
        "--mode",
        choices=("measured", "warmup", "pilot", "dry_run", "resilience"),
        default="measured",
    )
    run.add_argument("--repetition", type=int)
    run.add_argument("--expected-dlq-count", type=int, default=0)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output-dir", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--manifest", required=True)
    matrix.add_argument("--api-url", required=True)
    matrix.add_argument("--queue-url", required=True)
    matrix.add_argument("--dlq-url", required=True)
    matrix.add_argument("--output-dir", required=True)
    matrix.add_argument("--region")
    matrix.add_argument("--metrics-url", required=True)
    matrix.add_argument("--cloudwatch-config", required=True)
    matrix.add_argument("--resource-backfill-timeout-seconds", type=float, default=120)
    matrix.add_argument("--ecs-cluster", required=True)
    matrix.add_argument("--ecs-service", required=True)
    matrix.add_argument("--pipeline-task-definition", required=True)
    matrix.add_argument("--product-task-definition", required=True)
    matrix.add_argument("--pilot-seconds", type=float, required=True)
    matrix.add_argument("--api-key-env")
    matrix.add_argument("--api-key-header", default="X-API-Key")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-manifest":
        entries = validate_manifest(args.manifest, expected_documents=args.jobs)
        print(
            json.dumps(
                {
                    "documents": len(entries),
                    "bytes": sum(entry.size_bytes for entry in entries),
                }
            )
        )
        return 0
    if args.command == "analyze":
        print(json.dumps(analyze_result_tree(args.output_dir), indent=2))
        return 0
    if args.command == "matrix":
        return asyncio.run(_matrix_command(args))
    return asyncio.run(_run_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
