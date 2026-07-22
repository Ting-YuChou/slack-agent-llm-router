import asyncio
import base64
import hashlib
import json
import time

import pytest

from src.rag.queue import QueueDelivery, SqsIngestionQueue
from src.rag.chunker import DocumentChunk
from src.rag.service import IngestionJob, RagService
from src.rag.storage import (
    LocalObjectStore,
    RagObjectIntegrityError,
    RagObjectRef,
    S3ObjectStore,
)
from src.utils.metrics import RAG_METRICS
from src.rag.vector_store import InMemoryRagVectorStore


class FakeS3Body:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload


class FakeS3Client:
    def __init__(self, payload=b"handbook"):
        self.payload = payload
        self.tags = []
        self.presign_params = None
        self.puts = []

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.presign_params = (operation, Params, ExpiresIn)
        return "https://uploads.example.test/presigned"

    def head_object(self, **kwargs):
        return {
            "ContentLength": len(self.payload),
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(self.payload).digest()
            ).decode("ascii"),
            "VersionId": "version-1",
            "ETag": '"etag-1"',
        }

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {
            "VersionId": "version-put",
            "ETag": '"etag-put"',
            "ChecksumSHA256": kwargs["ChecksumSHA256"],
        }

    def get_object(self, **kwargs):
        return {"Body": FakeS3Body(self.payload)}

    def put_object_tagging(self, **kwargs):
        self.tags.append(kwargs)

    def delete_object(self, **kwargs):
        return {}


@pytest.mark.asyncio
async def test_s3_upload_completion_pins_validated_object_version():
    payload = b"handbook"
    checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    client = FakeS3Client(payload)
    store = S3ObjectStore(
        {
            "bucket": "rag-documents",
            "environment": "test",
            "presign_ttl_seconds": 600,
        },
        client=client,
    )

    pending_ref, upload = await store.create_upload(
        job_id="job-1",
        size_bytes=len(payload),
        checksum_sha256=checksum,
    )
    completed_ref = await store.complete_upload(pending_ref)

    assert pending_ref.uri == "s3://rag-documents/rag/test/job-1/source"
    assert upload == {
        "method": "PUT",
        "url": "https://uploads.example.test/presigned",
        "headers": {
            "content-length": str(len(payload)),
            "x-amz-checksum-sha256": checksum,
            "x-amz-tagging": "state=pending",
        },
        "expires_in": 600,
    }
    assert completed_ref.version_id == "version-1"
    assert completed_ref.etag == "etag-1"
    assert await store.load(completed_ref) == payload


@pytest.mark.asyncio
async def test_s3_upload_completion_rejects_size_or_checksum_mismatch():
    payload = b"actual"
    client = FakeS3Client(payload)
    store = S3ObjectStore(
        {"bucket": "rag-documents", "environment": "test"}, client=client
    )
    ref = RagObjectRef(
        backend="s3",
        uri="s3://rag-documents/rag/test/job-1/source",
        bucket="rag-documents",
        key="rag/test/job-1/source",
        size_bytes=999,
        checksum_sha256="wrong",
    )

    with pytest.raises(RagObjectIntegrityError):
        await store.complete_upload(ref)


@pytest.mark.asyncio
async def test_s3_put_bytes_returns_a_versioned_object_reference():
    client = FakeS3Client()
    store = S3ObjectStore(
        {"bucket": "rag-documents", "environment": "test"}, client=client
    )

    ref = await store.put_bytes(job_id="job-1", content=b"handbook")

    assert ref.version_id == "version-put"
    assert ref.size_bytes == 8
    assert client.puts[0]["Key"] == "rag/test/job-1/source"
    assert client.puts[0]["ContentLength"] == 8


@pytest.mark.asyncio
async def test_local_object_store_matches_put_load_tag_delete_contract(tmp_path):
    store = LocalObjectStore({"staging_dir": str(tmp_path)})

    ref = await store.put_bytes(job_id="job-1", content=b"handbook")
    await store.tag(ref, "completed")

    assert ref.backend == "local"
    assert await store.load(ref) == b"handbook"
    await store.delete(ref)
    assert not list(tmp_path.glob("*.manifest.json"))


class FakeSqsClient:
    def __init__(self):
        self.sent = []
        self.deleted = []
        self.visibility = []

    def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return {"MessageId": "message-1"}

    def receive_message(self, **kwargs):
        return {
            "Messages": [
                {
                    "MessageId": "message-1",
                    "ReceiptHandle": "receipt-1",
                    "Body": json.dumps(
                        {
                            "schema_version": 1,
                            "job_id": "job-1",
                            "dispatch_id": "dispatch-1",
                        }
                    ),
                    "Attributes": {"ApproximateReceiveCount": "2"},
                }
            ]
        }

    def delete_message(self, **kwargs):
        self.deleted.append(kwargs)

    def change_message_visibility(self, **kwargs):
        self.visibility.append(kwargs)


@pytest.mark.asyncio
async def test_sqs_queue_preserves_delivery_handle_and_receive_count():
    client = FakeSqsClient()
    queue = SqsIngestionQueue(
        {
            "queue_url": "https://sqs.example.test/source",
            "wait_time_seconds": 20,
            "visibility_timeout_seconds": 900,
        },
        client=client,
    )

    message_id = await queue.publish(
        {"schema_version": 1, "job_id": "job-1", "dispatch_id": "dispatch-1"}
    )
    deliveries = await queue.receive(max_messages=1)
    await queue.extend_visibility(deliveries[0], 120)
    await queue.ack(deliveries[0])

    assert message_id == "message-1"
    assert deliveries[0].receive_count == 2
    assert deliveries[0].payload["dispatch_id"] == "dispatch-1"
    assert client.visibility[0]["ReceiptHandle"] == "receipt-1"
    assert client.deleted[0]["ReceiptHandle"] == "receipt-1"


class FakeObjectStore:
    def __init__(self):
        self.completed = []
        self.tags = []

    async def create_upload(self, *, job_id, size_bytes, checksum_sha256):
        ref = RagObjectRef(
            backend="s3",
            uri=f"s3://bucket/rag/test/{job_id}/source",
            bucket="bucket",
            key=f"rag/test/{job_id}/source",
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
        )
        return ref, {
            "method": "PUT",
            "url": "https://upload.test",
            "headers": {},
            "expires_in": 900,
        }

    async def complete_upload(self, ref):
        self.completed.append(ref)
        return RagObjectRef(
            **{
                **ref.to_dict(),
                "version_id": "v1",
                "etag": "etag-1",
            }
        )

    async def put_bytes(self, *, job_id, content):
        checksum = base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")
        return RagObjectRef(
            backend="s3",
            uri=f"s3://bucket/rag/test/{job_id}/source",
            bucket="bucket",
            key=f"rag/test/{job_id}/source",
            version_id="put-v1",
            checksum_sha256=checksum,
            size_bytes=len(content),
        )

    async def tag(self, ref, status):
        self.tags.append((ref.uri, status))


class FakeIngestionQueue:
    def __init__(self, *, publish_error=None, ack_error=None, visibility_error=None):
        self.published = []
        self.acked = []
        self.visibility = []
        self.publish_error = publish_error
        self.ack_error = ack_error
        self.visibility_error = visibility_error

    async def publish(self, payload):
        self.published.append(payload)
        if self.publish_error:
            raise self.publish_error
        return f"message-{len(self.published)}"

    async def ack(self, delivery):
        if self.ack_error:
            raise self.ack_error
        self.acked.append(delivery.message_id)

    async def extend_visibility(self, delivery, timeout_seconds):
        if self.visibility_error:
            raise self.visibility_error
        self.visibility.append((delivery.message_id, timeout_seconds))


@pytest.mark.asyncio
async def test_presigned_upload_completion_is_idempotent_and_enqueues_once(tmp_path):
    object_store = FakeObjectStore()
    queue = FakeIngestionQueue()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {
                "backend": "s3",
                "staging_dir": str(tmp_path / "must-not-exist"),
                "s3": {"bucket": "bucket", "environment": "test"},
            },
            "ingestion_queue": {
                "enabled": True,
                "backend": "sqs",
                "sqs": {"queue_url": "https://sqs.test/source"},
            },
        },
        object_store=object_store,
        ingestion_queue=queue,
    )
    await service.initialize()

    created = await service.create_presigned_upload(
        filename="handbook.pdf",
        size_bytes=8,
        checksum_sha256=base64.b64encode(hashlib.sha256(b"handbook").digest()).decode(
            "ascii"
        ),
        knowledge_base_id="school",
        metadata={"title": "Handbook"},
        idempotency_key="upload-once",
    )
    assert created["job"].status == "awaiting_upload"
    completed = await service.complete_presigned_upload(created["job"].job_id)
    repeated = await service.complete_presigned_upload(created["job"].job_id)

    assert completed.status == "queued"
    assert repeated.status == "queued"
    assert completed.source["version_id"] == "v1"
    assert len(queue.published) == 1
    assert queue.published[0]["dispatch_id"] == completed.dispatch_id
    assert object_store.tags[-1][1] == "queued"
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.asyncio
async def test_presigned_upload_requires_base64_sha256(tmp_path):
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {
                "backend": "s3",
                "staging_dir": str(tmp_path / "unused"),
                "s3": {"bucket": "bucket", "environment": "test"},
            },
            "ingestion_queue": {
                "enabled": True,
                "backend": "sqs",
                "sqs": {"queue_url": "https://sqs.test/source"},
            },
        },
        object_store=FakeObjectStore(),
        ingestion_queue=FakeIngestionQueue(),
    )

    with pytest.raises(ValueError, match="SHA-256"):
        await service.create_presigned_upload(
            filename="handbook.pdf",
            size_bytes=8,
            checksum_sha256="not-a-checksum",
        )


@pytest.mark.asyncio
async def test_inline_document_uses_s3_without_local_staging(tmp_path):
    object_store = FakeObjectStore()
    queue = FakeIngestionQueue()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {
                "backend": "s3",
                "staging_dir": str(tmp_path / "must-not-exist"),
                "s3": {"bucket": "bucket", "environment": "test"},
            },
            "ingestion_queue": {
                "enabled": True,
                "backend": "sqs",
                "sqs": {"queue_url": "https://sqs.test/source"},
            },
        },
        object_store=object_store,
        ingestion_queue=queue,
    )

    job = await service.queue_document_ingestion(
        content=b"handbook",
        filename="handbook.pdf",
        knowledge_base_id="school",
    )

    assert job.source["backend"] == "s3"
    assert job.source["version_id"] == "put-v1"
    assert queue.published[0]["source"] == job.source
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.asyncio
async def test_existing_s3_storage_ref_is_validated_before_enqueue(tmp_path):
    object_store = FakeObjectStore()
    queue = FakeIngestionQueue()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {
                "backend": "s3",
                "staging_dir": str(tmp_path / "unused"),
                "s3": {"bucket": "bucket", "environment": "test"},
            },
            "ingestion_queue": {
                "enabled": True,
                "backend": "sqs",
                "sqs": {"queue_url": "https://sqs.test/source"},
            },
        },
        object_store=object_store,
        ingestion_queue=queue,
    )

    job = await service.queue_document_ingestion(
        storage_ref="s3://bucket/rag/test/job-external/source",
        filename="handbook.pdf",
        knowledge_base_id="school",
    )

    assert job.source["version_id"] == "v1"
    assert object_store.completed[0].uri == job.storage_ref


def test_local_storage_cannot_be_combined_with_sqs(tmp_path):
    with pytest.raises(ValueError, match="local.*sqs"):
        RagService(
            {
                "enabled": True,
                "backend": "memory",
                "storage": {
                    "backend": "local",
                    "staging_dir": str(tmp_path),
                },
                "ingestion_queue": {
                    "enabled": True,
                    "backend": "sqs",
                    "sqs": {"queue_url": "https://sqs.test/source"},
                },
            }
        )


class FakeLeaseRedis:
    def __init__(self, *, lease_available=True):
        self.values = {}
        self.lease_available = lease_available
        self.eval_calls = []
        self.sequence = 0

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, _ttl, value):
        self.values[key] = value

    async def set(self, key, value, **kwargs):
        if kwargs.get("nx") and (not self.lease_available or key in self.values):
            return False
        self.values[key] = value
        return True

    async def incr(self, _key):
        self.sequence += 1
        return self.sequence

    async def expire(self, _key, _ttl):
        return True

    async def delete(self, key):
        self.values.pop(key, None)

    async def eval(self, script, numkeys, key, token, *args):
        self.eval_calls.append((script, numkeys, key, token, *args))
        if "INCR" in script:
            self.sequence = max(self.sequence, int(token)) + 1
            return self.sequence
        if self.values.get(key) == token:
            if "expire" in script:
                return 1
            self.values.pop(key, None)
            return 1
        return 0


class FakeJobVectorStore:
    def __init__(self, client):
        self.client = client

    async def initialize(self):
        return None

    async def shutdown(self):
        return None


def _sqs_service(tmp_path, *, lease_available=True):
    queue = FakeIngestionQueue()
    client = FakeLeaseRedis(lease_available=lease_available)
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {
                "backend": "s3",
                "staging_dir": str(tmp_path / "unused"),
                "s3": {"bucket": "bucket", "environment": "test"},
            },
            "ingestion_queue": {
                "enabled": True,
                "backend": "sqs",
                "max_attempts": 3,
                "retry_backoff_seconds": 30,
                "sqs": {
                    "queue_url": "https://sqs.test/source",
                    "visibility_timeout_seconds": 900,
                    "heartbeat_interval_seconds": 60,
                },
            },
        },
        vector_store=FakeJobVectorStore(client),
        object_store=FakeObjectStore(),
        ingestion_queue=queue,
    )
    return service, queue


def _delivery(job, *, dispatch_id=None, receive_count=1):
    return QueueDelivery(
        message_id="message-1",
        receipt_handle="receipt-1",
        receive_count=receive_count,
        payload={
            "schema_version": 1,
            "job_id": job.job_id,
            "dispatch_id": dispatch_id or job.dispatch_id,
            "source": dict(job.source),
        },
    )


@pytest.mark.asyncio
async def test_sqs_worker_acks_stale_dispatch_without_processing(tmp_path, monkeypatch):
    service, queue = _sqs_service(tmp_path)
    job = IngestionJob(
        job_id="job-1",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        dispatch_id="current-dispatch",
        source={"backend": "s3", "uri": "s3://bucket/rag/test/job-1/source"},
    )
    await service._save_job(job)
    processed = []

    async def unexpected(*_args, **_kwargs):
        processed.append(True)

    monkeypatch.setattr(service, "process_ingestion_job", unexpected)

    await service.process_sqs_delivery(
        _delivery(job, dispatch_id="stale-dispatch"), "worker-1"
    )

    assert processed == []
    assert queue.acked == ["message-1"]


@pytest.mark.asyncio
async def test_sqs_terminal_ack_failure_does_not_crash_worker(tmp_path):
    service, queue = _sqs_service(tmp_path)
    queue.ack_error = RuntimeError("delete failed")
    job = IngestionJob(
        job_id="job-terminal",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        status="completed",
        dispatch_id="dispatch-1",
    )
    await service._save_job(job)

    result = await service.process_sqs_delivery(_delivery(job), "worker-1")

    assert result.status == "completed"
    assert queue.acked == []


@pytest.mark.asyncio
async def test_sqs_redelivery_resumes_a_failed_job_after_worker_crash(
    tmp_path, monkeypatch
):
    service, queue = _sqs_service(tmp_path)
    job = IngestionJob(
        job_id="job-crashed",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        status="failed",
        dispatch_id="dispatch-1",
        source={"backend": "s3", "uri": "s3://bucket/rag/test/job-crashed/source"},
    )
    await service._save_job(job)
    processed = []

    async def complete(*_args, **_kwargs):
        processed.append(True)
        job.status = "completed"
        await service._save_job(job)
        return job

    monkeypatch.setattr(service, "process_ingestion_job", complete)
    result = await service.process_sqs_delivery(_delivery(job), "worker-1")

    assert processed == [True]
    assert result.status == "completed"
    assert queue.acked == ["message-1"]


@pytest.mark.asyncio
async def test_sqs_worker_does_not_process_without_dispatch_lease(
    tmp_path, monkeypatch
):
    service, queue = _sqs_service(tmp_path, lease_available=False)
    job = IngestionJob(
        job_id="job-1",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        dispatch_id="dispatch-1",
        source={"backend": "s3", "uri": "s3://bucket/rag/test/job-1/source"},
    )
    await service._save_job(job)
    processed = []
    monkeypatch.setattr(
        service,
        "process_ingestion_job",
        lambda *_args, **_kwargs: processed.append(True),
    )

    await service.process_sqs_delivery(_delivery(job), "worker-1")

    assert processed == []
    assert queue.acked == []
    assert queue.visibility == [("message-1", 30)]


@pytest.mark.asyncio
async def test_sqs_worker_marks_last_failure_for_broker_redrive(tmp_path, monkeypatch):
    service, queue = _sqs_service(tmp_path)
    job = IngestionJob(
        job_id="job-1",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        dispatch_id="dispatch-1",
        source={"backend": "s3", "uri": "s3://bucket/rag/test/job-1/source"},
    )
    await service._save_job(job)

    async def fail(*_args, **_kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(service, "process_ingestion_job", fail)

    result = await service.process_sqs_delivery(
        _delivery(job, receive_count=3), "worker-1"
    )

    assert result.status == "dead_lettered"
    assert result.attempts == 3
    assert "parser exploded" in result.error
    assert queue.acked == []


@pytest.mark.asyncio
async def test_sqs_processing_lease_release_is_compare_and_delete(tmp_path):
    service, _queue = _sqs_service(tmp_path)
    job = IngestionJob(
        job_id="job-1",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        dispatch_id="dispatch-1",
    )

    lease = await service._acquire_sqs_processing_lease(
        job, dispatch_id="dispatch-1", consumer_name="worker-1"
    )
    await service._release_sqs_processing_lease(*lease)

    assert service.vector_store.client.eval_calls


@pytest.mark.asyncio
async def test_manual_retry_uses_a_new_dispatch_id_for_sqs(tmp_path):
    service, queue = _sqs_service(tmp_path)
    job = IngestionJob(
        job_id="job-1",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        status="dead_lettered",
        dispatch_id="old-dispatch",
        storage_ref="s3://bucket/rag/test/job-1/source",
        source={"backend": "s3", "uri": "s3://bucket/rag/test/job-1/source"},
    )
    await service._save_job(job)

    retried = await service.retry_job(job.job_id)

    assert retried.dispatch_id != "old-dispatch"
    assert queue.published[-1]["dispatch_id"] == retried.dispatch_id


@pytest.mark.asyncio
async def test_sqs_publish_failure_leaves_recoverable_enqueue_pending_job(tmp_path):
    service, queue = _sqs_service(tmp_path)
    queue.publish_error = RuntimeError("send failed")

    with pytest.raises(RuntimeError, match="send failed"):
        await service.queue_document_ingestion(
            content=b"handbook",
            filename="handbook.pdf",
            knowledge_base_id="school",
        )

    job = next(iter(service.jobs.values()))
    assert job.status == "enqueue_pending"
    dispatch_id = job.dispatch_id
    queue.publish_error = None
    retried = await service.retry_job(job.job_id)
    assert retried.status == "queued"
    assert retried.dispatch_id == dispatch_id


@pytest.mark.asyncio
async def test_sqs_jobs_use_monotonic_index_generation_separate_from_dispatch(tmp_path):
    service, _queue = _sqs_service(tmp_path)

    first = await service.queue_document_ingestion(
        content=b"first", filename="first.pdf", document_id="doc-1"
    )
    second = await service.queue_document_ingestion(
        content=b"second", filename="second.pdf", document_id="doc-1"
    )

    assert first.dispatch_id != second.dispatch_id
    assert first.index_version != first.dispatch_id
    assert first.index_version < second.index_version


@pytest.mark.asyncio
async def test_sqs_generation_advances_past_legacy_timestamp_generation(tmp_path):
    service, _queue = _sqs_service(tmp_path)
    legacy_generation = f"{time.time_ns() - 1:020d}-00000001"

    _dispatch_id, index_version = await service._new_sqs_dispatch()

    assert index_version > legacy_generation


@pytest.mark.asyncio
async def test_sqs_ack_failure_preserves_completed_terminal_state(
    tmp_path, monkeypatch
):
    service, queue = _sqs_service(tmp_path)
    queue.ack_error = RuntimeError("delete failed")
    job = IngestionJob(
        job_id="job-ack",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        dispatch_id="dispatch-1",
        source={"backend": "s3", "uri": "s3://bucket/rag/test/job-ack/source"},
    )
    await service._save_job(job)

    async def complete(*_args, **_kwargs):
        job.status = "completed"
        await service._save_job(job)
        return job

    monkeypatch.setattr(service, "process_ingestion_job", complete)
    result = await service.process_sqs_delivery(_delivery(job), "worker-1")

    assert result.status == "completed"
    assert (await service.get_job(job.job_id)).status == "completed"
    assert queue.visibility == []


@pytest.mark.asyncio
async def test_sqs_heartbeat_failure_cancels_processing(tmp_path, monkeypatch):
    service, queue = _sqs_service(tmp_path)
    service.sqs_heartbeat_interval_seconds = 0.001
    queue.visibility_error = RuntimeError("visibility lost")
    job = IngestionJob(
        job_id="job-heartbeat",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        dispatch_id="dispatch-1",
        source={
            "backend": "s3",
            "uri": "s3://bucket/rag/test/job-heartbeat/source",
        },
    )
    await service._save_job(job)
    cancelled = []

    async def slow(*_args, **_kwargs):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    monkeypatch.setattr(service, "process_ingestion_job", slow)
    result = await service.process_sqs_delivery(_delivery(job), "worker-1")

    assert cancelled == [True]
    assert result.status == "retrying"


@pytest.mark.asyncio
async def test_redis_stream_completion_tags_s3_source(tmp_path, monkeypatch):
    service, _queue = _sqs_service(tmp_path)
    service.queue_backend = "redis_stream"
    job = IngestionJob(
        job_id="job-stream",
        document_id="doc-1",
        filename="handbook.pdf",
        knowledge_base_id="school",
        source={"backend": "s3", "uri": "s3://bucket/rag/test/job-stream/source"},
    )
    await service._save_job(job)

    async def complete(*_args, **_kwargs):
        job.status = "completed"
        await service._save_job(job)
        return job

    monkeypatch.setattr(service, "process_ingestion_job", complete)

    async def ack(*_args):
        return 1

    monkeypatch.setattr(service, "ack_stream_message", ack)
    await service.process_stream_message("1-0", {"job_id": job.job_id}, "worker-1")

    assert service.object_store.tags[-1][1] == "completed"


@pytest.mark.asyncio
async def test_upload_idempotency_is_shared_through_redis(tmp_path):
    client = FakeLeaseRedis()
    object_store = FakeObjectStore()
    config = {
        "enabled": True,
        "backend": "memory",
        "storage": {
            "backend": "s3",
            "staging_dir": str(tmp_path / "unused"),
            "s3": {"bucket": "bucket", "environment": "test"},
        },
        "ingestion_queue": {
            "enabled": True,
            "backend": "sqs",
            "sqs": {"queue_url": "https://sqs.test/source"},
        },
    }
    first = RagService(
        config,
        vector_store=FakeJobVectorStore(client),
        object_store=object_store,
        ingestion_queue=FakeIngestionQueue(),
    )
    second = RagService(
        config,
        vector_store=FakeJobVectorStore(client),
        object_store=object_store,
        ingestion_queue=FakeIngestionQueue(),
    )
    checksum = base64.b64encode(hashlib.sha256(b"handbook").digest()).decode("ascii")

    first_result = await first.create_presigned_upload(
        filename="handbook.pdf",
        size_bytes=8,
        checksum_sha256=checksum,
        idempotency_key="shared-key",
    )
    second_result = await second.create_presigned_upload(
        filename="handbook.pdf",
        size_bytes=8,
        checksum_sha256=checksum,
        idempotency_key="shared-key",
    )

    assert second_result["job"].job_id == first_result["job"].job_id


@pytest.mark.asyncio
async def test_presigned_idempotency_key_rejects_a_different_request(tmp_path):
    service, _queue = _sqs_service(tmp_path)
    checksum = base64.b64encode(hashlib.sha256(b"handbook").digest()).decode("ascii")
    await service.create_presigned_upload(
        filename="first.pdf",
        size_bytes=8,
        checksum_sha256=checksum,
        idempotency_key="same-key",
    )

    with pytest.raises(ValueError, match="Idempotency-Key"):
        await service.create_presigned_upload(
            filename="different.pdf",
            size_bytes=8,
            checksum_sha256=checksum,
            idempotency_key="same-key",
        )


def test_rag_aws_metrics_cover_storage_queue_and_delivery_outcomes():
    assert RAG_METRICS.object_operations
    assert RAG_METRICS.queue_operations
    assert RAG_METRICS.delivery_outcomes
    assert RAG_METRICS.processing_lease_contention


@pytest.mark.asyncio
async def test_vector_index_version_can_be_pinned_to_dispatch_id():
    store = InMemoryRagVectorStore()
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="handbook",
        page_start=1,
        page_end=1,
        block_ids=[],
        block_types=[],
        metadata={},
    )

    await store.upsert_chunks(
        [chunk],
        [[1.0]],
        knowledge_base_id="school",
        index_version="dispatch-1",
    )
    changed = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="changed content must not replace an already committed dispatch",
        page_start=1,
        page_end=1,
        block_ids=[],
        block_types=[],
        metadata={},
    )
    await store.upsert_chunks(
        [changed],
        [[2.0]],
        knowledge_base_id="school",
        index_version="dispatch-1",
    )

    assert next(iter(store.items.values())).index_version == "dispatch-1"
    assert next(iter(store.items.values())).chunk.text == "handbook"
