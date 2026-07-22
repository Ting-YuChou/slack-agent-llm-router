"""Opt-in smoke test for an isolated AWS RAG test bucket and queue."""

import asyncio
import os
import uuid

import pytest

from src.rag.queue import SqsIngestionQueue
from src.rag.storage import S3ObjectStore


REQUIRED_ENV = ("RAG_AWS_BUCKET", "RAG_AWS_QUEUE_URL", "AWS_REGION")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RAG_AWS_SMOKE_CONFIRM") != "1"
        or any(not os.getenv(name) for name in REQUIRED_ENV),
        reason="isolated AWS RAG smoke resources were not explicitly enabled",
    ),
]


@pytest.mark.asyncio
async def test_isolated_aws_s3_sqs_round_trip():
    run_id = str(uuid.uuid4())
    store = S3ObjectStore(
        {
            "bucket": os.environ["RAG_AWS_BUCKET"],
            "region": os.environ["AWS_REGION"],
            "environment": "smoke",
            "prefix": "rag-smoke",
        }
    )
    queue = SqsIngestionQueue(
        {
            "queue_url": os.environ["RAG_AWS_QUEUE_URL"],
            "region": os.environ["AWS_REGION"],
            "wait_time_seconds": 2,
            "visibility_timeout_seconds": 30,
        }
    )
    ref = await store.put_bytes(job_id=run_id, content=b"aws-rag-smoke")
    try:
        assert await store.load(ref) == b"aws-rag-smoke"
        await queue.publish(
            {
                "schema_version": 1,
                "job_id": run_id,
                "dispatch_id": run_id,
                "smoke_test": True,
            }
        )
        matched = None
        for _ in range(10):
            deliveries = await queue.receive(max_messages=10)
            for delivery in deliveries:
                if delivery.payload.get("job_id") == run_id:
                    matched = delivery
                    break
                await queue.extend_visibility(delivery, 0)
            if matched:
                break
            await asyncio.sleep(0.25)
        assert matched is not None
        await queue.ack(matched)
    finally:
        await store.delete(ref)
