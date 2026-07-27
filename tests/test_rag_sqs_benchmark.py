import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
import httpx

from scripts.benchmark_rag_s3_sqs import (
    calculate_percentile,
    determine_scaling_knee,
    parse_sqs_attributes,
    parse_prometheus_counter,
    RagBenchmarkRunner,
    analyze_result_tree,
    load_cloudwatch_queries,
    measured_worker_orders,
    run_timeout_seconds,
    AwsControlPlane,
    CloudWatchResourceSampler,
    summarize_jobs,
    summarize_worker_runs,
    validate_manifest,
    healthy_run_violations,
)


def _job(index, *, latency, attempts=1, status="completed", size=5_000_000):
    started = datetime(2026, 1, 1, 0, 0, 0) + timedelta(seconds=index)
    return {
        "job_id": f"job-{index}",
        "status": status,
        "attempts": attempts,
        "size_bytes": size,
        "dispatch_started_at": started.isoformat(),
        "terminal_at": (started + timedelta(seconds=latency)).isoformat(),
    }


def test_percentile_uses_linear_interpolation():
    samples = [1.0, 2.0, 3.0, 4.0]

    assert calculate_percentile(samples, 50) == pytest.approx(2.5)
    assert calculate_percentile(samples, 95) == pytest.approx(3.85)
    assert calculate_percentile(samples, 99) == pytest.approx(3.97)


def test_summarize_jobs_applies_benchmark_metric_definitions():
    jobs = [
        _job(0, latency=10),
        _job(1, latency=20, attempts=2),
        _job(2, latency=30, status="dead_lettered"),
        _job(3, latency=40, status="benchmark_timeout"),
    ]

    summary = summarize_jobs(
        jobs,
        completion_seconds=120,
        duplicate_commit_count=0,
        peak_visible_depth=3,
        peak_total_depth=4,
    )

    assert summary["successful_documents"] == 2
    assert summary["docs_per_minute"] == pytest.approx(1.0)
    assert summary["mb_per_minute"] == pytest.approx(5.0)
    assert summary["retry_rate"] == pytest.approx(0.25)
    assert summary["failure_rate"] == pytest.approx(0.5)
    assert summary["latency_seconds"]["p50"] == pytest.approx(25.0)
    assert summary["peak_visible_queue_depth"] == 3
    assert summary["peak_total_queue_depth"] == 4


def test_healthy_run_rejects_missing_or_negative_latency_timestamps():
    jobs = [
        {
            "job_id": "missing",
            "status": "completed",
            "attempts": 1,
            "size_bytes": 1,
        },
        {
            **_job(1, latency=1),
            "terminal_at": "2025-12-31T23:59:00+00:00",
        },
    ]
    summary = summarize_jobs(
        jobs,
        completion_seconds=1,
        duplicate_commit_count=0,
        peak_visible_depth=0,
        peak_total_depth=0,
    )
    result = {
        "mode": "dry_run",
        "expected_dlq_count": 0,
        "summary": summary,
        "final_queue": {"total_depth": 0, "delayed_depth": 0},
        "final_dlq": {"total_depth": 0},
    }

    violations = healthy_run_violations(result, expected_documents=2)

    assert "one or more jobs lack a valid end-to-end latency" in violations
    assert "one or more job timestamps are invalid" in violations


def test_extra_atomic_index_commit_is_counted_as_effective_duplicate():
    jobs = [_job(0, latency=1), _job(1, latency=1)]

    summary = summarize_jobs(
        jobs,
        completion_seconds=1,
        duplicate_commit_count=0,
        index_commit_count=3,
        peak_visible_depth=0,
        peak_total_depth=0,
    )

    assert summary["duplicate_commit_count"] == 1


def test_expected_dlq_does_not_accept_benchmark_timeouts():
    jobs = [
        _job(index, latency=3, attempts=3, status="benchmark_timeout")
        for index in range(5)
    ]
    summary = summarize_jobs(
        jobs,
        completion_seconds=10,
        duplicate_commit_count=0,
        peak_visible_depth=5,
        peak_total_depth=5,
    )
    result = {
        "mode": "resilience",
        "expected_dlq_count": 5,
        "summary": summary,
        "final_queue": {"total_depth": 0, "delayed_depth": 0},
        "final_dlq": {"total_depth": 5},
    }

    violations = healthy_run_violations(result, expected_documents=5)

    assert "dead-lettered job count does not match expectation" in violations
    assert "DLQ resilience still contains benchmark timeouts" in violations


def test_worker_summary_pools_latency_samples_instead_of_averaging_percentiles():
    runs = [
        {"docs_per_minute": 10, "completion_seconds": 20, "latencies": [1, 2]},
        {"docs_per_minute": 30, "completion_seconds": 40, "latencies": [100, 101]},
        {"docs_per_minute": 20, "completion_seconds": 30, "latencies": [3, 4]},
    ]

    summary = summarize_worker_runs(runs)

    assert summary["docs_per_minute"]["median"] == 20
    assert summary["docs_per_minute"]["range"] == [10, 30]
    assert summary["pooled_latency_seconds"]["p50"] == pytest.approx(3.5)


def test_scaling_knee_uses_gain_and_parallel_efficiency_thresholds():
    assert determine_scaling_knee({1: 100, 2: 180, 4: 225, 8: 250}) == 2
    assert determine_scaling_knee({1: 100, 2: 175, 4: 310, 8: 520}) == ">8"


def test_parse_sqs_attributes_reports_visible_and_total_depth():
    sample = parse_sqs_attributes(
        {
            "ApproximateNumberOfMessages": "7",
            "ApproximateNumberOfMessagesNotVisible": "3",
            "ApproximateNumberOfMessagesDelayed": "2",
            "ApproximateAgeOfOldestMessage": "11",
        },
        sampled_at="2026-01-01T00:00:00",
    )

    assert sample["visible_depth"] == 7
    assert sample["in_flight_depth"] == 3
    assert sample["total_depth"] == 10
    assert sample["delayed_depth"] == 2
    assert sample["oldest_message_age_seconds"] == 11


def test_prometheus_counter_sums_all_backend_reason_series():
    text = """
# HELP llm_router_rag_duplicate_commits_total duplicate commits
llm_router_rag_duplicate_commits_total{backend="memory",reason="same_generation"} 2
llm_router_rag_duplicate_commits_total{backend="redis_stack",reason="same_generation"} 3
other_total 99
"""

    assert parse_prometheus_counter(text, "llm_router_rag_duplicate_commits_total") == 5


def test_manifest_validation_checks_size_and_checksum(tmp_path):
    document = tmp_path / "handbook.txt"
    document.write_bytes(b"handbook")
    checksum = base64.b64encode(hashlib.sha256(b"handbook").digest()).decode()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "path": str(document),
                        "filename": "handbook.txt",
                        "format": "txt",
                        "size_bytes": 8,
                        "checksum_sha256": checksum,
                        "page_count": 1,
                    }
                ]
            }
        )
    )

    entries = validate_manifest(manifest, expected_documents=1)

    assert entries[0].size_bytes == 8
    assert entries[0].checksum_sha256 == checksum


def test_manifest_validation_rejects_changed_corpus_file(tmp_path):
    document = tmp_path / "changed.txt"
    document.write_bytes(b"changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "path": str(document),
                        "filename": "changed.txt",
                        "format": "txt",
                        "size_bytes": 999,
                        "checksum_sha256": "wrong",
                        "page_count": 1,
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="size mismatch"):
        validate_manifest(manifest, expected_documents=1)


@pytest.mark.asyncio
async def test_runner_executes_presign_put_complete_poll_and_writes_outputs(tmp_path):
    document = tmp_path / "handbook.txt"
    document.write_bytes(b"handbook")
    checksum = base64.b64encode(hashlib.sha256(b"handbook").digest()).decode()
    entry = type("Entry", (), {})()
    entry.path = document
    entry.filename = "handbook.txt"
    entry.format = "txt"
    entry.size_bytes = 8
    entry.checksum_sha256 = checksum
    entry.page_count = 1
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text='llm_router_rag_duplicate_commits_total{backend="redis_stack",reason="same_generation"} 0\n',
            )
        if request.method == "POST" and request.url.path == "/rag/uploads":
            return httpx.Response(
                201,
                json={
                    "job": {"job_id": "job-1"},
                    "upload": {
                        "method": "PUT",
                        "url": "https://upload.test/object",
                        "headers": {"x-amz-checksum-sha256": checksum},
                    },
                },
            )
        if request.method == "PUT" and request.url.path == "/object":
            assert request.content == b"handbook"
            return httpx.Response(200)
        if request.method == "POST" and request.url.path.endswith("/complete"):
            assert request.headers["X-RAG-Dispatch-Started-At"].endswith("+00:00")
            return httpx.Response(202, json={"job_id": "job-1", "status": "queued"})
        if request.method == "GET" and request.url.path == "/rag/jobs/job-1":
            return httpx.Response(
                200,
                json={
                    "job_id": "job-1",
                    "document_id": "doc-1",
                    "filename": "handbook.txt",
                    "knowledge_base_id": "kb-1",
                    "status": "completed",
                    "attempts": 1,
                    "dispatch_started_at": "2026-01-01T00:00:00+00:00",
                    "terminal_at": "2026-01-01T00:00:02+00:00",
                },
            )
        if request.method == "DELETE" and request.url.path == "/rag/documents/doc-1":
            return httpx.Response(200, json={"deleted_chunks": 1})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    class FakeAws:
        async def assert_queue_empty(self, _url, _label):
            return None

        async def queue_sample(self, _url):
            return parse_sqs_attributes({})

    output_dir = tmp_path / "output"
    runner = RagBenchmarkRunner(
        api_url="https://api.test",
        queue_url="queue",
        dlq_url="dlq",
        metrics_url="https://api.test/metrics",
        http_transport=httpx.MockTransport(handler),
        aws=FakeAws(),
        poll_interval_seconds=0.001,
    )

    result = await runner.run(
        [entry],
        output_dir=output_dir,
        run_id="run-1",
        layer="pipeline",
        workers=1,
        timeout_seconds=1,
    )

    assert result["summary"]["successful_documents"] == 1
    assert result["summary"]["latency_seconds"]["p50"] == 2
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "jobs.csv").exists()
    assert (output_dir / "queue.csv").exists()
    assert (output_dir / "queue_resource_timeline.svg").exists()
    assert ("PUT", "/object") in calls
    assert ("DELETE", "/rag/documents/doc-1") in calls


@pytest.mark.asyncio
async def test_runner_stops_sampling_and_writes_partial_report_on_complete_failure(
    tmp_path,
):
    document = tmp_path / "failure.txt"
    document.write_bytes(b"failure")
    checksum = base64.b64encode(hashlib.sha256(b"failure").digest()).decode()
    entry = type("Entry", (), {})()
    entry.path = document
    entry.filename = "failure.txt"
    entry.format = "txt"
    entry.size_bytes = 7
    entry.checksum_sha256 = checksum
    entry.page_count = 1

    def handler(request):
        if request.method == "POST" and request.url.path == "/rag/uploads":
            return httpx.Response(
                201,
                json={
                    "job": {"job_id": "job-failure"},
                    "upload": {
                        "method": "PUT",
                        "url": "https://upload.test/failure",
                        "headers": {},
                    },
                },
            )
        if request.method == "PUT":
            return httpx.Response(200)
        if request.method == "POST" and request.url.path.endswith("/complete"):
            return httpx.Response(503, json={"error": "unavailable"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    class FakeAws:
        async def assert_queue_empty(self, _url, _label):
            return None

        async def queue_sample(self, _url):
            return parse_sqs_attributes({})

    output_dir = tmp_path / "partial"
    runner = RagBenchmarkRunner(
        api_url="https://api.test",
        queue_url="queue",
        dlq_url="dlq",
        http_transport=httpx.MockTransport(handler),
        aws=FakeAws(),
        poll_interval_seconds=0.001,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await runner.run(
            [entry],
            output_dir=output_dir,
            run_id="failed-run",
            layer="pipeline",
            workers=1,
            timeout_seconds=1,
            mode="dry_run",
        )

    partial = json.loads((output_dir / "summary.json").read_text())
    assert partial["status"] == "benchmark_failed"
    assert partial["source_objects_preserved"] is True


@pytest.mark.asyncio
async def test_runner_reports_created_job_when_presigned_put_fails(tmp_path):
    document = tmp_path / "put-failure.txt"
    document.write_bytes(b"failure")
    checksum = base64.b64encode(hashlib.sha256(b"failure").digest()).decode()
    entry = type("Entry", (), {})()
    entry.path = document
    entry.filename = "put-failure.txt"
    entry.format = "txt"
    entry.size_bytes = 7
    entry.checksum_sha256 = checksum
    entry.page_count = 1

    def handler(request):
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "job": {"job_id": "created-before-put-failure"},
                    "upload": {
                        "method": "PUT",
                        "url": "https://upload.test/failure",
                        "headers": {},
                    },
                },
            )
        return httpx.Response(503)

    class FakeAws:
        async def assert_queue_empty(self, _url, _label):
            return None

    output_dir = tmp_path / "upload-partial"
    runner = RagBenchmarkRunner(
        api_url="https://api.test",
        queue_url="queue",
        dlq_url="dlq",
        http_transport=httpx.MockTransport(handler),
        aws=FakeAws(),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await runner.run(
            [entry],
            output_dir=output_dir,
            run_id="put-failed-run",
            layer="pipeline",
            workers=1,
            timeout_seconds=1,
            mode="dry_run",
        )

    partial = json.loads((output_dir / "summary.json").read_text())
    assert partial["status"] == "upload_preparation_failed"
    assert partial["created_upload_jobs"] == 1
    assert "created-before-put-failure" in (output_dir / "jobs.csv").read_text()


def test_analysis_reports_three_run_medians_pooled_percentiles_and_charts(tmp_path):
    rates = (100, 120, 110)
    resources = [
        "ecs_cpu",
        "ecs_memory",
        "redis_cpu",
        "redis_latency",
        "embedding_qps",
        "embedding_p95",
        "embedding_errors",
    ]
    for layer in ("pipeline", "product"):
        for workers in (1, 2, 4, 8):
            for repetition, rate in enumerate(rates, start=1):
                run_dir = tmp_path / f"{layer}-{workers}-{repetition}"
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "run_id": f"{layer}-{workers}-{repetition}",
                            "layer": layer,
                            "workers": workers,
                            "mode": "measured",
                            "expected_dlq_count": 0,
                            "resource_signal_coverage": resources,
                            "final_queue": {"total_depth": 0, "delayed_depth": 0},
                            "final_dlq": {"total_depth": 0},
                            "summary": {
                                "total_documents": 500,
                                "successful_documents": 500,
                                "failed_documents": 0,
                                "failure_rate": 0,
                                "retry_rate": 0,
                                "duplicate_commit_count": 0,
                                "docs_per_minute": rate * workers,
                                "mb_per_minute": rate * workers * 5,
                                "completion_seconds": (300 - rate) / workers,
                                "latencies": [float(repetition)] * 500,
                                "latency_sample_count": 500,
                                "latency_validation_errors": [],
                            },
                        }
                    )
                )

    report = analyze_result_tree(tmp_path)

    worker = report["layers"]["pipeline"]["workers"]["1"]
    assert worker["docs_per_minute"]["median"] == 110
    assert worker["pooled_latency_seconds"]["p50"] == 2
    assert (tmp_path / "workers_throughput.svg").exists()
    assert (tmp_path / "workers_completion.svg").exists()
    assert (tmp_path / "workers_latency.svg").exists()


def test_cloudwatch_query_config_covers_required_resource_signals(tmp_path):
    config = tmp_path / "cloudwatch.json"
    config.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "key": "ecs_cpu",
                        "namespace": "AWS/ECS",
                        "metric_name": "CPUUtilization",
                        "dimensions": {},
                    },
                    {
                        "key": "ecs_memory",
                        "namespace": "AWS/ECS",
                        "metric_name": "MemoryUtilization",
                        "dimensions": {},
                    },
                    {
                        "key": "redis_cpu",
                        "namespace": "AWS/ElastiCache",
                        "metric_name": "EngineCPUUtilization",
                        "dimensions": {},
                    },
                    {
                        "key": "redis_latency",
                        "namespace": "AWS/ElastiCache",
                        "metric_name": "SuccessfulReadRequestLatency",
                        "dimensions": {},
                    },
                    {
                        "key": "embedding_qps",
                        "namespace": "RAG/Embedding",
                        "metric_name": "Requests",
                        "dimensions": {},
                    },
                    {
                        "key": "embedding_p95",
                        "namespace": "RAG/Embedding",
                        "metric_name": "Latency",
                        "dimensions": {},
                        "stat": "p95",
                    },
                    {
                        "key": "embedding_errors",
                        "namespace": "RAG/Embedding",
                        "metric_name": "Errors",
                        "dimensions": {},
                    },
                ]
            }
        )
    )

    queries = load_cloudwatch_queries(config)

    assert {query["key"] for query in queries} == {
        "ecs_cpu",
        "ecs_memory",
        "redis_cpu",
        "redis_latency",
        "embedding_qps",
        "embedding_p95",
        "embedding_errors",
    }


def test_matrix_orders_and_timeout_follow_benchmark_protocol():
    assert measured_worker_orders() == [
        [1, 2, 4, 8],
        [8, 4, 2, 1],
        [2, 8, 1, 4],
    ]
    assert run_timeout_seconds(50) == 1800
    assert run_timeout_seconds(120) == 3600


@pytest.mark.asyncio
async def test_worker_crash_waits_until_a_message_is_in_flight(monkeypatch):
    control = object.__new__(AwsControlPlane)
    samples = iter(
        [
            {"in_flight_depth": 0},
            {"in_flight_depth": 0},
            {"in_flight_depth": 2},
        ]
    )

    async def queue_sample(_url):
        return next(samples)

    monkeypatch.setattr(control, "queue_sample", queue_sample)

    sample = await control.wait_for_in_flight(
        "queue", timeout_seconds=1, poll_interval_seconds=0
    )

    assert sample["in_flight_depth"] == 2


@pytest.mark.asyncio
async def test_cloudwatch_sampler_reads_extended_stat_and_ignores_stale_points():
    started = datetime.now(timezone.utc) - timedelta(seconds=30)

    class FakeCloudWatch:
        def get_metric_statistics(self, **_kwargs):
            return {
                "Datapoints": [
                    {
                        "Timestamp": started - timedelta(minutes=1),
                        "ExtendedStatistics": {"p95": 999},
                    },
                    {
                        "Timestamp": started + timedelta(seconds=5),
                        "ExtendedStatistics": {"p95": 0.25},
                    },
                ]
            }

    sampler = object.__new__(CloudWatchResourceSampler)
    sampler.client = FakeCloudWatch()
    sampler.queries = [
        {
            "key": "embedding_p95",
            "namespace": "RAG/Embedding",
            "metric_name": "Latency",
            "dimensions": {},
            "stat": "p95",
            "period_seconds": 60,
        }
    ]
    sampler.interval_started_at = started
    sampler.interval_ended_at = started + timedelta(seconds=30)

    sample = await sampler.sample()

    assert sample["embedding_p95"] == 0.25
    assert (
        sample["embedding_p95_timestamp"]
        == (started + timedelta(seconds=5)).isoformat()
    )


@pytest.mark.asyncio
async def test_resource_backfill_accepts_late_required_signals():
    class LateSampler:
        async def sample(self):
            return {
                "sampled_at": datetime.now(timezone.utc).isoformat(),
                "ecs_cpu": 1,
                "ecs_memory": 2,
                "redis_cpu": 3,
                "redis_latency": 4,
                "embedding_qps": 5,
                "embedding_p95": 6,
                "embedding_errors": 0,
            }

    runner = RagBenchmarkRunner(
        api_url="https://api.test",
        queue_url="queue",
        dlq_url="dlq",
        aws=object(),
        resource_sampler=LateSampler(),
        resource_backfill_timeout_seconds=1,
    )
    samples = []

    await runner._backfill_resource_signals(samples)

    assert samples[0]["embedding_p95"] == 6
