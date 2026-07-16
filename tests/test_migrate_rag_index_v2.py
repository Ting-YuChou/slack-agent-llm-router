import os
import uuid

import pytest

from scripts.migrate_rag_index_v2 import MigrationRunner, MigrationSpec


class _Redis:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value):
        self.values[key] = value


@pytest.mark.asyncio
async def test_migration_dry_run_does_not_touch_redis():
    redis = _Redis()
    runner = MigrationRunner(redis, MigrationSpec(key_prefix="rag"))

    plan = await runner.run(apply=False)

    assert plan[0]["phase"] == "copy"
    assert plan[-1]["phase"] == "resume"
    assert redis.values == {}


@pytest.mark.asyncio
async def test_apply_requires_pause_and_resume_hooks():
    runner = MigrationRunner(_Redis(), MigrationSpec(key_prefix="rag"))

    with pytest.raises(ValueError, match="pause-command"):
        await runner.run(apply=True)


@pytest.mark.asyncio
async def test_resume_continues_from_persisted_phase(monkeypatch):
    redis = _Redis()
    spec = MigrationSpec(
        key_prefix="rag",
        pause_command="pause",
        resume_command="resume",
        switch_command="switch",
    )
    runner = MigrationRunner(redis, spec)
    calls = []

    async def record(phase):
        calls.append(phase)

    monkeypatch.setattr(runner, "_run_phase", record)
    await redis.set(runner.state_key, "validate_initial")

    await runner.run(apply=True)

    assert calls[0] == "validate_initial"
    assert calls[-1] == "resume"


def test_tagged_key_mapping_is_deterministic_and_preserves_source_suffix():
    spec = MigrationSpec(key_prefix="rag")

    assert spec.target_chunk_key("school", "doc-1", "chunk-2") == (
        "rag:v2:chunk:{school:doc-1}:chunk-2"
    )
    assert spec.target_document_key("school", "doc-1", "chunks") == (
        "rag:document:v2:{school:doc-1}:chunks"
    )


@pytest.mark.asyncio
async def test_failed_rollback_stays_paused(monkeypatch):
    redis = _Redis()
    spec = MigrationSpec(
        key_prefix="rag",
        pause_command="pause",
        resume_command="resume",
        rollback_command="rollback",
    )
    runner = MigrationRunner(redis, spec)
    calls = []
    await redis.set(runner.state_key, "complete")

    async def hook(command):
        calls.append(command)
        if command == "rollback":
            raise RuntimeError("switch failed")

    monkeypatch.setattr(runner, "_run_hook", hook)
    monkeypatch.setattr(runner, "_copy_hashes_to_legacy", lambda: _async_value({}))
    monkeypatch.setattr(runner, "_validate", lambda: _async_value({}))

    with pytest.raises(RuntimeError, match="switch failed"):
        await runner.rollback()

    assert calls == ["pause", "rollback"]
    assert redis.values[runner.state_key] == "rollback_copied"


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_rollback_switched_retries_only_resume(monkeypatch):
    redis = _Redis()
    spec = MigrationSpec(
        key_prefix="rag",
        pause_command="pause",
        resume_command="resume",
        rollback_command="rollback",
    )
    runner = MigrationRunner(redis, spec)
    await redis.set(runner.state_key, "rollback_switched")
    calls = []

    async def hook(command):
        calls.append(command)

    monkeypatch.setattr(runner, "_run_hook", hook)

    await runner.rollback()

    assert calls == ["resume"]
    assert redis.values[runner.state_key] == "rolled_back"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_copy_and_checksum_validation():
    redis_asyncio = pytest.importorskip("redis.asyncio")
    client = redis_asyncio.Redis(
        host=os.getenv("RAG_REDIS_HOST", "localhost"),
        port=int(os.getenv("RAG_REDIS_PORT", "6380")),
        decode_responses=False,
    )
    try:
        await client.ping()
    except Exception as exc:
        await client.aclose()
        pytest.skip(f"Redis is not available: {exc}")
    prefix = f"ragmigration:{uuid.uuid4().hex}"
    runner = MigrationRunner(client, MigrationSpec(key_prefix=prefix))
    try:
        await client.hset(
            f"{prefix}:chunk:school:chunk-1",
            mapping={
                "knowledge_base_id": "school",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "text": "policy",
            },
        )
        await client.hset(
            f"{prefix}:visual_chunk:school:chunk-1",
            mapping={
                "knowledge_base_id": "school",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "visual_embedding": b"\x00\x01",
            },
        )
        await client.hset(
            f"{prefix}:table:school:doc-1:table-1",
            mapping={
                "knowledge_base_id": "school",
                "document_id": "doc-1",
                "table_id": "table-1",
                "table_json": "{}",
            },
        )
        await client.hset(
            f"{prefix}:table_row:school:doc-1:table-1:row-1",
            mapping={
                "knowledge_base_id": "school",
                "document_id": "doc-1",
                "table_id": "table-1",
                "row_id": "row-1",
                "row_json": "{}",
            },
        )
        await client.hset(
            f"{prefix}:figure:school:doc-1:figure-1",
            mapping={
                "knowledge_base_id": "school",
                "document_id": "doc-1",
                "figure_id": "figure-1",
                "figure_json": "{}",
            },
        )
        await client.sadd(f"{prefix}:document:school:doc-1:chunks", "chunk-1")
        await client.sadd(f"{prefix}:document:school:doc-1:visual_chunks", "chunk-1")
        await client.sadd(f"{prefix}:tables:school:doc-1", "table-1")
        await client.sadd(f"{prefix}:figures:school:doc-1", "figure-1")
        await client.sadd(f"{prefix}:table_rows:school:doc-1:table-1", "row-1")
        await client.sadd(f"{prefix}:document:doc-1:knowledge_bases", "school")
        await client.sadd(
            f"{prefix}:document:v2:{{school:doc-1}}:chunks",
            "stale-chunk",
        )
        await client.sadd(f"{prefix}:document:doc-1:knowledge_bases", "stale-kb")

        copied = await runner._copy_hashes()
        validated = await runner._validate()

        assert copied["copied_chunk"] == 1
        assert copied["copied_visual_chunk"] == 1
        assert copied["copied_table"] == 1
        assert copied["copied_table_row"] == 1
        assert copied["copied_figure"] == 1
        assert validated["count"] == 5
        assert await client.hgetall(f"{prefix}:v2:chunk:{{school:doc-1}}:chunk-1")
        assert await client.smembers(
            f"{prefix}:document:v2:{{school:doc-1}}:chunks"
        ) == {b"chunk-1"}
        assert await client.smembers(f"{prefix}:document:doc-1:knowledge_bases") == {
            b"school"
        }

        await client.hset(
            f"{prefix}:v2:chunk:{{school:doc-1}}:chunk-1",
            mapping={
                "knowledge_base_id": "school",
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "text": "updated after cutover",
            },
        )
        await client.sadd(f"{prefix}:document:school:doc-1:chunks", "stale-chunk")
        await client.sadd(f"{prefix}:document:doc-1:knowledge_bases", "stale-kb")
        await runner._copy_hashes_to_legacy()
        await runner._validate()
        assert (
            await client.hget(f"{prefix}:chunk:school:chunk-1", "text")
            == b"updated after cutover"
        )
        assert await client.smembers(f"{prefix}:document:school:doc-1:chunks") == {
            b"chunk-1"
        }
        assert await client.smembers(f"{prefix}:document:doc-1:knowledge_bases") == {
            b"school"
        }
        assert await client.smembers(
            f"{prefix}:table_rows:v2:{{school:doc-1}}:table-1"
        ) == {b"row-1"}
        assert await client.smembers(f"{prefix}:tables:v2:{{school:doc-1}}") == {
            b"table-1"
        }
        assert await client.smembers(f"{prefix}:figures:v2:{{school:doc-1}}") == {
            b"figure-1"
        }
        assert await client.smembers(
            f"{prefix}:document:v2:{{school:doc-1}}:visual_chunks"
        ) == {b"chunk-1"}
    finally:
        keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()
