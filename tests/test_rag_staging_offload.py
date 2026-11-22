"""Staging file I/O and chunking stay off the event loop."""

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from src.rag.service import IngestionJob, RagService, _safe_path_component
from src.rag.staging import RagStagingStore


class _StreamingUpload:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    async def read(self, _size):
        return self.chunks.pop(0) if self.chunks else b""


class _QueueRedis:
    """The few commands a non-batch queued ingestion needs."""

    def __init__(self):
        self.values = {}
        self.streams = {}
        self.sequence = 0

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, _ttl, value):
        self.values[key] = value

    async def eval(self, _script, _numkeys, _key, token, *_args):
        self.sequence = max(self.sequence, int(token)) + 1
        return self.sequence

    async def xgroup_create(self, stream, _group, id="0", mkstream=True):
        self.streams.setdefault(stream, [])

    async def xadd(self, stream, fields, **_kwargs):
        entries = self.streams.setdefault(stream, [])
        entries.append((f"{len(entries) + 1}-0", dict(fields)))
        return entries[-1][0]


def _service(tmp_path, **overrides):
    config = {
        "enabled": True,
        "backend": "memory",
        "parser": {"provider": "text"},
        "embedding": {"provider": "hash", "dimensions": 16},
        "storage": {"staging_dir": str(tmp_path / "staging")},
        "upload": {"multipart_max_bytes": 1000, "stream_chunk_bytes": 4},
    }
    config.update(overrides)
    return RagService(config)


def _record_thread(monkeypatch, target, name, calls):
    real = getattr(target, name)

    def recording(*args, **kwargs):
        calls.append((name, threading.get_ident(), args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, recording)


def _count_fsync(monkeypatch):
    calls = []
    real_fsync = os.fsync

    def counting(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting)
    return calls


async def _upload(service, chunks, job_id="job-1"):
    return await service.stage_uploaded_file(
        _StreamingUpload(chunks),
        filename="handbook.pdf",
        knowledge_base_id="school",
        document_id="doc-1",
        job_id=job_id,
    )


@pytest.mark.asyncio
async def test_upload_runs_begin_and_reserve_off_the_event_loop(tmp_path, monkeypatch):
    service = _service(tmp_path)
    calls = []
    for name in ("begin", "reserve"):
        _record_thread(monkeypatch, service._staging_store, name, calls)
    loop_thread = threading.get_ident()

    storage_ref = await _upload(service, [b"aaaa", b"bbbb", b"cccc", b"dd"])

    assert Path(storage_ref).read_bytes() == b"aaaabbbbccccdd"
    assert [name for name, *_ in calls] == ["begin", *["reserve"] * 4]
    # One reservation per chunk, sized to that chunk.
    assert [args[1] for name, _, args, _ in calls if name == "reserve"] == [
        4,
        4,
        4,
        2,
    ]
    assert all(thread != loop_thread for _, thread, _, _ in calls)
    assert service._staging_store.usage_bytes() == 14


@pytest.mark.asyncio
async def test_event_loop_keeps_running_while_reserve_is_slow(tmp_path, monkeypatch):
    service = _service(tmp_path)
    real_reserve = service._staging_store.reserve

    def slow_reserve(job_id, amount):
        time.sleep(0.05)
        return real_reserve(job_id, amount)

    monkeypatch.setattr(service._staging_store, "reserve", slow_reserve)
    ticks = []

    async def heartbeat():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.002)

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    started = time.monotonic()
    try:
        await _upload(service, [b"aaaa", b"bbbb", b"cccc", b"dddd"])
    finally:
        finished = time.monotonic()
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)

    during = [tick for tick in ticks if started <= tick <= finished]
    assert finished - started >= 0.2
    # A blocked loop manages at most a tick or two per chunk.
    assert len(during) >= 20


@pytest.mark.asyncio
async def test_json_staging_runs_off_the_event_loop(tmp_path, monkeypatch):
    service = _service(tmp_path, ingestion_queue={"enabled": True})
    service.vector_store.client = _QueueRedis()
    calls = []
    _record_thread(monkeypatch, service._staging_store, "stage_bytes", calls)

    job = await service.queue_document_ingestion(
        content=b"Tuition is due in August.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )

    assert Path(job.storage_ref).read_bytes() == b"Tuition is due in August."
    assert len(calls) == 1
    assert calls[0][1] != threading.get_ident()


def test_reserve_fsyncs_the_manifest_but_not_the_ledger(tmp_path, monkeypatch):
    store = RagStagingStore(tmp_path, {})
    store.begin("job-1", "upload.pdf")
    fsyncs = _count_fsync(monkeypatch)

    store.reserve("job-1", 4)

    assert len(fsyncs) == 1
    assert json.loads((tmp_path / ".usage.json").read_text())["bytes"] == 4
    assert json.loads((tmp_path / "job-1.manifest.json").read_text())["bytes"] == 4
    store.rollback("job-1")


@pytest.mark.asyncio
async def test_four_chunk_upload_fsync_count(tmp_path, monkeypatch):
    service = _service(tmp_path)
    await asyncio.to_thread(service._staging_store.ensure_reconciled)
    fsyncs = _count_fsync(monkeypatch)

    await _upload(service, [b"aaaa", b"bbbb", b"cccc", b"dddd"])

    # begin manifest, one manifest write per chunk, the data file, publish.
    assert len(fsyncs) == 1 + 4 + 1 + 1


@pytest.mark.parametrize("damage", ["missing", "empty", "garbage", "wrong_total"])
def test_reconcile_rebuilds_a_lost_or_wrong_ledger(tmp_path, damage):
    first = RagStagingStore(tmp_path, {})
    first.stage_bytes("job-1", "one.pdf", b"12345")
    first.stage_bytes("job-2", "two.pdf", b"1234567")
    ledger = tmp_path / ".usage.json"
    if damage == "missing":
        ledger.unlink()
    elif damage == "empty":
        ledger.write_text("")
    elif damage == "garbage":
        ledger.write_text('{"bytes": 1')
    else:
        ledger.write_text('{"bytes": 999}')

    second = RagStagingStore(tmp_path, {})

    assert second.usage_bytes() == 12
    assert json.loads(ledger.read_text())["bytes"] == 12


def _stage_many(store, count):
    return {
        f"job-{index}": store.stage_bytes(f"job-{index}", "doc.pdf", b"x")
        for index in range(count)
    }


def _count_manifest_reads(monkeypatch, store):
    reads = []
    real_read = store._read_json

    def counting(path, default):
        if str(path).endswith(".manifest.json"):
            reads.append(Path(path).name)
        return real_read(path, default)

    monkeypatch.setattr(store, "_read_json", counting)
    return reads


def _manifest(root, job_id):
    return json.loads((Path(root) / f"{job_id}.manifest.json").read_text())


def test_update_status_with_job_id_reads_only_that_manifest(tmp_path, monkeypatch):
    now = [1000.0]
    store = RagStagingStore(
        tmp_path, {"completed_file_ttl_seconds": 10}, clock=lambda: now[0]
    )
    _stage_many(store, 50)
    storage_ref = store.stage_bytes("target", "doc.pdf", b"target")
    now[0] = 2000.0
    reads = _count_manifest_reads(monkeypatch, store)

    store.update_status(storage_ref, "completed", job_id="target")

    assert reads == ["target.manifest.json"]
    manifest = _manifest(tmp_path, "target")
    assert manifest["status"] == "completed"
    assert manifest["updated_at"] == 2000.0
    assert manifest["expires_at"] == 2010.0


@pytest.mark.parametrize("hint", ["job-3", "no-such-job", "../job-3", ""])
def test_update_status_falls_back_to_the_scan_when_the_hint_is_wrong(tmp_path, hint):
    store = RagStagingStore(tmp_path, {})
    _stage_many(store, 5)
    storage_ref = store.stage_bytes("target", "doc.pdf", b"target")

    store.update_status(storage_ref, "running", job_id=hint)

    assert _manifest(tmp_path, "target")["status"] == "running"
    assert _manifest(tmp_path, "job-3")["status"] == "queued"


def test_update_status_rejects_a_hint_whose_manifest_points_elsewhere(tmp_path):
    store = RagStagingStore(tmp_path, {})
    refs = _stage_many(store, 2)

    with pytest.raises(FileNotFoundError):
        store.update_status(str(tmp_path / "unknown.pdf"), "running", job_id="job-0")

    assert _manifest(tmp_path, "job-0")["status"] == "queued"
    assert Path(refs["job-0"]).exists()


def test_update_status_without_job_id_still_scans(tmp_path, monkeypatch):
    store = RagStagingStore(tmp_path, {})
    _stage_many(store, 5)
    storage_ref = store.stage_bytes("target", "doc.pdf", b"target")

    store.update_status(storage_ref, "failed")

    assert _manifest(tmp_path, "target")["status"] == "failed"
    with pytest.raises(FileNotFoundError):
        store.update_status(str(tmp_path / "unknown.pdf"), "failed")


def test_refresh_lease_accepts_the_job_id_hint(tmp_path, monkeypatch):
    now = [1000.0]
    store = RagStagingStore(
        tmp_path, {"orphan_file_ttl_seconds": 30}, clock=lambda: now[0]
    )
    _stage_many(store, 5)
    storage_ref = store.stage_bytes("target", "doc.pdf", b"target")
    now[0] = 1500.0
    reads = _count_manifest_reads(monkeypatch, store)

    store.refresh_lease(storage_ref, "processing", job_id="target")

    assert reads == ["target.manifest.json"]
    assert _manifest(tmp_path, "target")["expires_at"] == 1530.0


@pytest.mark.asyncio
@pytest.mark.parametrize("job_id", ["job-1", "weird job/id:1"])
async def test_save_job_updates_the_manifest_directly_and_off_the_loop(
    tmp_path, monkeypatch, job_id
):
    service = _service(tmp_path)
    store = service._staging_store
    for index in range(10):
        store.stage_bytes(f"other-{index}", "doc.pdf", b"x")
    storage_ref = await asyncio.to_thread(
        service.stage_document_content,
        b"Tuition is due in August.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
        job_id=job_id,
    )
    calls = []
    _record_thread(monkeypatch, store, "update_status", calls)
    reads = _count_manifest_reads(monkeypatch, store)
    job = IngestionJob(
        job_id=job_id,
        document_id="doc-1",
        filename="handbook.md",
        knowledge_base_id="school",
        status="running",
        storage_ref=storage_ref,
    )

    await service._save_job(job)

    staged_id = _safe_path_component(job_id)
    assert len(calls) == 1
    assert calls[0][1] != threading.get_ident()
    assert reads == [f"{staged_id}.manifest.json"]
    assert _manifest(service.staging_dir, staged_id)["status"] == "running"


@pytest.mark.asyncio
async def test_save_job_without_a_staged_file_does_not_touch_staging(
    tmp_path, monkeypatch
):
    service = _service(tmp_path)
    calls = []
    _record_thread(monkeypatch, service._staging_store, "update_status", calls)
    job = IngestionJob(
        job_id="job-1",
        document_id="doc-1",
        filename="handbook.md",
        knowledge_base_id="school",
    )

    await service._save_job(job)

    assert calls == []


@pytest.mark.asyncio
async def test_ingestion_loads_and_chunks_off_the_event_loop(tmp_path, monkeypatch):
    service = _service(tmp_path)
    await service.initialize()
    storage_ref = await asyncio.to_thread(
        service.stage_document_content,
        b"Tuition is due in August.\n\nRefunds close in September.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
        job_id="job-1",
    )
    calls = []
    _record_thread(monkeypatch, service, "load_staged_content", calls)
    _record_thread(monkeypatch, service.chunker, "chunk", calls)
    loop_thread = threading.get_ident()

    try:
        job = await service.process_ingestion_job(
            "job-1",
            content=None,
            filename="handbook.md",
            knowledge_base_id="school",
            document_id="doc-1",
            storage_ref=storage_ref,
        )
    finally:
        await service.shutdown()

    assert job.status == "completed", job.error
    assert job.chunks_indexed >= 1
    assert [name for name, *_ in calls] == ["load_staged_content", "chunk"]
    assert all(thread != loop_thread for _, thread, _, _ in calls)


async def _cancel_while_blocked(service, monkeypatch, method):
    store = service._staging_store
    entered = threading.Event()
    release = threading.Event()
    real = getattr(store, method)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=1)
        return real(*args, **kwargs)

    monkeypatch.setattr(store, method, blocked)
    task = asyncio.create_task(_upload(service, [b"aaaa", b"bbbb"]))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["begin", "reserve"])
async def test_upload_cancelled_inside_a_staging_thread_leaves_nothing_behind(
    tmp_path, monkeypatch, method
):
    service = _service(tmp_path)

    await _cancel_while_blocked(service, monkeypatch, method)

    staging = service.staging_dir
    assert list(staging.rglob("*.part")) == []
    assert list(staging.glob("*.manifest.json")) == []
    assert list(staging.glob("*.pdf")) == []
    assert list(staging.glob(".*.lease.lock")) == []
    assert service._staging_store._leases == {}
    assert service._staging_store.usage_bytes() == 0
