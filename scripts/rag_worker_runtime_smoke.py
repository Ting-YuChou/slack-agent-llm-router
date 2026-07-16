#!/usr/bin/env python3
"""Exercise RAG queue success, retry, and dead-letter paths against real Redis."""

import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.service import RagService


async def _process_next(service: RagService, consumer: str):
    entries = await service.read_new_jobs(consumer)
    if not entries:
        raise RuntimeError("RAG smoke worker did not receive an expected job")
    message_id, fields = entries[0]
    return await service.process_stream_message(message_id, fields, consumer)


async def run() -> dict:
    run_id = uuid.uuid4().hex
    key_prefix = f"ragsmoke:{run_id}"
    staging_dir = Path(f"/app/data/rag/smoke/{run_id}")
    service = RagService(
        {
            "enabled": True,
            "backend": "redis_stack",
            "parser": {"provider": "text", "max_file_size_bytes": 1_000_000},
            "chunking": {"strategy": "hybrid", "chunk_size_tokens": 128},
            "embedding": {"provider": "hash", "dimensions": 16},
            "redis": {
                "host": "redis-stack",
                "port": 6379,
                "key_prefix": key_prefix,
                "pipeline_batch_size": 16,
            },
            "indexing": {"control_plane_version": "v2"},
            "ingestion_queue": {
                "enabled": True,
                "stream_key": f"{key_prefix}:stream",
                "group_name": f"{key_prefix}:workers",
                "dead_letter_stream_key": f"{key_prefix}:dead-letter",
                "retry_zset_key": f"{key_prefix}:retry",
                "consumer_count": 1,
                "concurrency": 1,
                "block_ms": 100,
                "pending_idle_ms": 100,
                "max_attempts": 2,
                "retry_backoff_seconds": 0,
            },
            "storage": {"staging_dir": str(staging_dir)},
        }
    )
    await service.initialize()
    client = service.vector_store.client
    try:
        success = await service.queue_document_ingestion(
            content=b"Tuition policy applications close on May 1.",
            filename="success.txt",
            knowledge_base_id="school",
            document_id="success-document",
        )
        completed = await _process_next(service, "smoke-consumer")
        if completed is None or completed.status != "completed":
            raise RuntimeError("RAG smoke success job did not complete")

        failure = await service.queue_document_ingestion(
            content=b"This staged file will be removed before processing.",
            filename="failure.txt",
            knowledge_base_id="school",
            document_id="failure-document",
        )
        Path(failure.storage_ref or "").unlink()
        first_failure = await _process_next(service, "smoke-consumer")
        if first_failure is None or first_failure.status != "retrying":
            raise RuntimeError("RAG smoke job did not enter retry state")
        if await service.promote_due_retries() != 1:
            raise RuntimeError("RAG smoke retry was not promoted")
        dead = await _process_next(service, "smoke-consumer")
        if dead is None or dead.status != "dead_lettered":
            raise RuntimeError("RAG smoke job did not enter dead-letter state")
        dead_letters = await client.xlen(service.dead_letter_stream_key)
        if int(dead_letters) != 1:
            raise RuntimeError("RAG smoke dead-letter stream count is not one")
        return {
            "success_job": success.job_id,
            "success_status": completed.status,
            "retry_job": failure.job_id,
            "retry_attempts": dead.attempts,
            "retry_status": dead.status,
            "dead_letters": int(dead_letters),
        }
    finally:
        keys = [key async for key in client.scan_iter(match=f"{key_prefix}:*")]
        if keys:
            await client.delete(*keys)
        await service.shutdown()
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), sort_keys=True))
