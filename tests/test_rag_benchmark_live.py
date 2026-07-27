"""Opt-in 10-document AWS dry run for the benchmark harness."""

import os
import uuid

import pytest

from scripts.benchmark_rag_s3_sqs import (
    RagBenchmarkRunner,
    healthy_run_violations,
    validate_manifest,
)


REQUIRED_ENV = (
    "RAG_BENCHMARK_API_URL",
    "RAG_BENCHMARK_MANIFEST_10",
    "RAG_AWS_QUEUE_URL",
    "RAG_AWS_DLQ_URL",
    "RAG_ECS_CLUSTER",
    "RAG_ECS_SERVICE",
    "RAG_PIPELINE_TASK_DEFINITION",
    "AWS_REGION",
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RAG_BENCHMARK_CONFIRM") != "1"
        or any(not os.getenv(name) for name in REQUIRED_ENV),
        reason="isolated 10-document AWS benchmark dry run was not enabled",
    ),
]


@pytest.mark.asyncio
async def test_ten_document_aws_benchmark_dry_run(tmp_path):
    entries = validate_manifest(
        os.environ["RAG_BENCHMARK_MANIFEST_10"], expected_documents=10
    )
    api_headers = {}
    if os.getenv("RAG_BENCHMARK_API_KEY"):
        api_headers["X-API-Key"] = os.environ["RAG_BENCHMARK_API_KEY"]
    runner = RagBenchmarkRunner(
        api_url=os.environ["RAG_BENCHMARK_API_URL"],
        queue_url=os.environ["RAG_AWS_QUEUE_URL"],
        dlq_url=os.environ["RAG_AWS_DLQ_URL"],
        api_headers=api_headers,
        region=os.environ["AWS_REGION"],
        metrics_url=os.getenv("RAG_METRICS_URL"),
    )

    result = await runner.run(
        entries,
        output_dir=tmp_path / "aws-dry-run",
        run_id=f"dry-run-{uuid.uuid4().hex}",
        layer="pipeline",
        workers=1,
        timeout_seconds=float(os.getenv("RAG_BENCHMARK_TIMEOUT_SECONDS", "1800")),
        ecs_cluster=os.environ["RAG_ECS_CLUSTER"],
        ecs_service=os.environ["RAG_ECS_SERVICE"],
        task_definition=os.environ["RAG_PIPELINE_TASK_DEFINITION"],
        mode="dry_run",
    )

    assert healthy_run_violations(result, expected_documents=10) == []
