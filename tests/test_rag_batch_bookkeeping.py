"""Batch progress bookkeeping: round trips, atomic counts and compatibility."""

import asyncio
import base64
import json

import pytest

from src.memory import HashEmbeddingProvider
from src.rag.parser import TextDocumentParser
from src.rag.service import JOB_STATUSES, IngestionBatch, IngestionJob, RagService
from src.rag.vector_store import InMemoryRagVectorStore

JOB_TTL = 4321


def _encode(value):
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


class _CountingPipeline:
    def __init__(self, client, transaction):
        self._client = client
        self.transaction = transaction
        self._queued = []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self._queued.append((name, args, kwargs))
            return self

        return queue

    async def execute(self):
        queued, self._queued = self._queued, []
        await self._client._round_trip()
        return [
            self._client._apply(name, *args, **kwargs) for name, args, kwargs in queued
        ]


class CountingRedis:
    """Redis double: one awaited command or one pipeline execute is one round trip.

    Every round trip yields to the event loop first, the way a socket write
    does, so unlocked read-modify-write sequences interleave.
    """

    def __init__(self):
        self.values = {}
        self.sets = {}
        self.hashes = {}
        self.ttls = {}
        self.streams = {}
        self.groups = set()
        self.sequence = 0
        self.round_trips = 0
        self.commands = []
        self.pipelines = []

    async def _round_trip(self):
        self.round_trips += 1
        await asyncio.sleep(0)

    def _apply(self, name, *args, **kwargs):
        self.commands.append((name, args[0] if args else None))
        return getattr(self, f"_do_{name}")(*args, **kwargs)

    def pipeline(self, transaction=True):
        pipeline = _CountingPipeline(self, transaction)
        self.pipelines.append(pipeline)
        return pipeline

    def __getattr__(self, name):
        if name.startswith("_") or not hasattr(type(self), f"_do_{name}"):
            raise AttributeError(name)

        async def command(*args, **kwargs):
            await self._round_trip()
            return self._apply(name, *args, **kwargs)

        return command

    def count(self, name, key=None):
        return sum(
            1
            for command, command_key in self.commands
            if command == name and (key is None or command_key == key)
        )

    def reset_counters(self):
        self.round_trips = 0
        self.commands = []
        self.pipelines = []

    def _do_get(self, key):
        value = self.values.get(key)
        return None if value is None else _encode(value)

    def _do_mget(self, keys, *more):
        names = [keys, *more] if isinstance(keys, (str, bytes)) else list(keys)
        return [self._do_get(name) for name in names]

    def _do_setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = int(ttl)
        return True

    def _do_set(self, key, value, **kwargs):
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = value
        if kwargs.get("ex") is not None:
            self.ttls[key] = int(kwargs["ex"])
        return True

    def _do_delete(self, *keys):
        removed = 0
        for key in keys:
            for store in (self.values, self.sets, self.hashes):
                if store.pop(key, None) is not None:
                    removed += 1
            self.ttls.pop(key, None)
        return removed

    def _do_sadd(self, key, *members):
        target = self.sets.setdefault(key, set())
        before = len(target)
        target.update(members)
        return len(target) - before

    def _do_smembers(self, key):
        return {_encode(member) for member in self.sets.get(key, set())}

    def _do_hincrby(self, key, field, amount=1):
        target = self.hashes.setdefault(key, {})
        target[field] = int(target.get(field, 0)) + int(amount)
        return target[field]

    def _do_hset(self, key, field=None, value=None, mapping=None):
        target = self.hashes.setdefault(key, {})
        if field is not None:
            target[field] = value
        target.update(mapping or {})
        return 1

    def _do_hgetall(self, key):
        return {
            _encode(field): _encode(value)
            for field, value in self.hashes.get(key, {}).items()
        }

    def _do_expire(self, key, ttl):
        if key in self.values or key in self.sets or key in self.hashes:
            self.ttls[key] = int(ttl)
            return True
        return False

    def _do_eval(self, script, _numkeys, key, token, *args):
        if "INCR" in script:
            self.sequence = max(self.sequence, int(token)) + 1
            return self.sequence
        return 0

    def _do_xgroup_create(self, stream, group, id="0", mkstream=True):
        self.groups.add((stream, group))
        self.streams.setdefault(stream, [])
        return True

    def _do_xadd(self, stream, fields, **kwargs):
        entries = self.streams.setdefault(stream, [])
        message_id = f"{len(entries) + 1}-0"
        entries.append((message_id, dict(fields)))
        return message_id


def _service(tmp_path, *, client=None, queue_enabled=True):
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "job_ttl_seconds": JOB_TTL,
            "local_job_cache_max_entries": 5000,
            "parser": {"provider": "text"},
            "embedding": {"provider": "hash", "dimensions": 16},
            "ingestion_queue": {"enabled": queue_enabled},
            "storage": {"staging_dir": str(tmp_path / "rag_uploads")},
        },
        parser=TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    if client is not None:
        service.vector_store.client = client
    return service


def _documents(count):
    return [
        {
            "filename": f"doc-{index}.md",
            "document_id": f"doc-{index}",
            "content_base64": base64.b64encode(f"Rule {index}".encode()).decode(),
        }
        for index in range(count)
    ]


async def _creation_round_trips(tmp_path, count):
    client = CountingRedis()
    service = _service(tmp_path / f"n{count}", client=client)
    await service.initialize()
    client.reset_counters()
    batch = await service.create_batch(
        knowledge_base_id="school", documents=_documents(count)
    )
    return service, client, batch, client.round_trips


async def _seed_batch(service, count, *, status="queued"):
    """Create a batch and its jobs through the bookkeeping path, without staging."""
    batch = IngestionBatch(
        batch_id="batch-1",
        knowledge_base_id="school",
        total=count,
        status_counts={name: 0 for name in JOB_STATUSES},
    )
    await service._save_batch(batch)
    jobs = []
    for index in range(count):
        job = IngestionJob(
            job_id=f"job-{index:04d}",
            document_id=f"doc-{index}",
            filename=f"doc-{index}.md",
            knowledge_base_id="school",
            status=status,
            batch_id=batch.batch_id,
        )
        await service._save_job(job)
        jobs.append(job)
    return batch, jobs


@pytest.mark.asyncio
async def test_batch_creation_round_trips_grow_linearly_with_documents(tmp_path):
    _, _, _, trips_10 = await _creation_round_trips(tmp_path, 10)
    _, _, _, trips_20 = await _creation_round_trips(tmp_path, 20)
    service, _, batch, trips_100 = await _creation_round_trips(tmp_path, 100)

    per_document = (trips_20 - trips_10) / 10
    assert (trips_100 - trips_20) / 80 == per_document
    # dispatch id, previous status, job SETEX, batch pipeline, XGROUP, XADD,
    # post-enqueue job SETEX.
    assert per_document == 7
    progress = await service.get_batch(batch.batch_id)
    assert progress["status_counts"]["queued"] == 100
    assert len(progress["job_ids"]) == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [5, 100])
async def test_status_change_is_one_round_trip_regardless_of_batch_size(tmp_path, size):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    _, jobs = await _seed_batch(service, size)
    client.reset_counters()

    await service._update_batch_for_status_change(
        jobs[0], previous_status="queued", new_status="running"
    )

    assert client.round_trips == 1
    assert [pipeline.transaction for pipeline in client.pipelines] == [True]
    assert client.count("sadd") == 1
    progress = await service.get_batch("batch-1")
    assert progress["status_counts"]["queued"] == size - 1
    assert progress["status_counts"]["running"] == 1


@pytest.mark.asyncio
async def test_saving_a_job_does_not_rewrite_the_batch_manifest(tmp_path):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    _, jobs = await _seed_batch(service, 3)
    meta_key = service._batch_key("batch-1")

    jobs[0].status = "running"
    await service._save_job(jobs[0])

    assert client.count("setex", meta_key) == 1


@pytest.mark.asyncio
async def test_concurrent_status_changes_do_not_lose_counts(tmp_path):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    _, jobs = await _seed_batch(service, 20)

    async def complete(job):
        job.status = "completed"
        await service._save_job(job)

    await asyncio.gather(*(complete(job) for job in jobs))

    progress = await service.get_batch("batch-1")
    assert progress["status_counts"]["completed"] == 20
    assert progress["status_counts"]["queued"] == 0
    assert progress["status"] == "completed"
    assert len(progress["job_ids"]) == 20


@pytest.mark.asyncio
async def test_get_batch_reads_jobs_with_mget_in_constant_round_trips(tmp_path):
    trips = {}
    for size in (10, 60):
        client = CountingRedis()
        service = _service(tmp_path / str(size), client=client, queue_enabled=False)
        _, jobs = await _seed_batch(service, size)
        jobs[0].status = "completed"
        await service._save_job(jobs[0])
        service.jobs.clear()
        client.reset_counters()

        progress = await service.get_batch("batch-1")

        trips[size] = client.round_trips
        assert client.count("mget") == 1
        assert client.count("get") <= 1
        assert set(progress) == {
            "batch_id",
            "knowledge_base_id",
            "total",
            "status",
            "status_counts",
            "job_ids",
            "jobs",
            "created_at",
            "updated_at",
        }
        assert progress["total"] == size
        assert progress["status"] == "running"
        assert list(progress["status_counts"]) == sorted(JOB_STATUSES)
        assert sorted(progress["job_ids"]) == [job.job_id for job in jobs]
        assert sorted(progress["jobs"], key=lambda item: item["job_id"]) == [
            job.to_public_dict() for job in jobs
        ]
        # Same side effects as get_job: local cache and the persisted status.
        cached = service.jobs.get(jobs[0].job_id)
        assert cached is not None
        assert getattr(cached, "_persisted_status") == "completed"
    assert trips[10] == trips[60] == 2


@pytest.mark.asyncio
async def test_get_batch_reads_large_batches_in_chunks_of_500_keys(tmp_path):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    await _seed_batch(service, 501)
    client.reset_counters()

    progress = await service.get_batch("batch-1")

    assert len(progress["jobs"]) == 501
    assert client.count("mget") == 2
    assert client.round_trips == 3


@pytest.mark.asyncio
async def test_get_batch_falls_back_to_the_local_job_when_redis_has_none(tmp_path):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    _, jobs = await _seed_batch(service, 3)
    client.values.pop(service._job_key(jobs[1].job_id))
    client.values[service._job_key(jobs[2].job_id)] = "{not json"

    progress = await service.get_batch("batch-1")

    assert sorted(item["job_id"] for item in progress["jobs"]) == [
        job.job_id for job in jobs
    ]


@pytest.mark.asyncio
async def test_batch_written_by_the_previous_format_stays_readable_and_updatable(
    tmp_path,
):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    job_ids = [f"job-{index}" for index in range(5)]
    legacy = IngestionBatch(
        batch_id="legacy",
        knowledge_base_id="school",
        total=5,
        status_counts={
            **{name: 0 for name in JOB_STATUSES},
            "queued": 3,
            "completed": 2,
        },
        job_ids=list(job_ids),
    )
    # The previous code kept everything in the JSON manifest and the job set.
    client.values[service._batch_key("legacy")] = json.dumps(
        legacy.to_dict(jobs=[]), sort_keys=True
    )
    client.sets[service._batch_jobs_key("legacy")] = set(job_ids)
    for index, job_id in enumerate(job_ids):
        job = IngestionJob(
            job_id=job_id,
            document_id=f"doc-{index}",
            filename=f"doc-{index}.md",
            knowledge_base_id="school",
            status="queued" if index < 3 else "completed",
            batch_id="legacy",
        )
        client.values[service._job_key(job_id)] = json.dumps(job.to_dict())

    before = await service.get_batch("legacy")
    job = await service.get_job("job-0")
    job.status = "completed"
    await service._save_job(job)
    after = await service.get_batch("legacy")

    assert before["status_counts"]["queued"] == 3
    assert before["status_counts"]["completed"] == 2
    assert after["status_counts"]["queued"] == 2
    assert after["status_counts"]["completed"] == 3
    assert sorted(after["job_ids"]) == job_ids
    counts = client.hashes[service._batch_key("legacy") + ":counts"]
    assert counts["queued"] == -1
    assert counts["completed"] == 1


@pytest.mark.asyncio
async def test_counts_never_go_negative(tmp_path):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    _, jobs = await _seed_batch(service, 1)

    # A status the batch never counted, e.g. after a lost counter.
    await service._update_batch_for_status_change(
        jobs[0], previous_status="retrying", new_status="failed"
    )

    progress = await service.get_batch("batch-1")
    assert progress["status_counts"]["retrying"] == 0
    assert progress["status_counts"]["failed"] == 1
    assert all(count >= 0 for count in progress["status_counts"].values())


@pytest.mark.asyncio
async def test_manifest_counts_and_job_set_all_carry_the_job_ttl(tmp_path):
    service, client, batch, _ = await _creation_round_trips(tmp_path, 2)
    meta_key = service._batch_key(batch.batch_id)

    assert client.ttls[meta_key] == JOB_TTL
    assert client.ttls[meta_key + ":counts"] == JOB_TTL
    assert client.ttls[meta_key + ":jobs"] == JOB_TTL


@pytest.mark.asyncio
async def test_updated_at_still_advances_with_status_changes(tmp_path):
    client = CountingRedis()
    service = _service(tmp_path, client=client, queue_enabled=False)
    _, jobs = await _seed_batch(service, 1)
    before = await service.get_batch("batch-1")
    await asyncio.sleep(0.01)

    jobs[0].status = "running"
    await service._save_job(jobs[0])

    after = await service.get_batch("batch-1")
    assert after["updated_at"] > before["updated_at"]
    assert after["created_at"] == before["created_at"]


@pytest.mark.asyncio
async def test_in_memory_batches_work_without_a_redis_client(tmp_path):
    service = _service(tmp_path, queue_enabled=False)
    assert service._job_client() is None
    _, jobs = await _seed_batch(service, 3)

    jobs[0].status = "completed"
    await service._save_job(jobs[0])
    jobs[1].status = "failed"
    await service._save_job(jobs[1])

    progress = await service.get_batch("batch-1")
    assert progress["status_counts"]["queued"] == 1
    assert progress["status_counts"]["completed"] == 1
    assert progress["status_counts"]["failed"] == 1
    assert progress["job_ids"] == [job.job_id for job in jobs]
    assert [item["job_id"] for item in progress["jobs"]] == [job.job_id for job in jobs]
