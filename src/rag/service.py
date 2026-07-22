"""High-level school-document RAG ingestion and retrieval service."""

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.memory import EmbeddingProvider, build_embedding_provider
from src.rag.chunker import DocumentChunk, build_chunker
from src.rag.parser import build_document_parser
from src.rag.reranker import Reranker, build_reranker
from src.rag.queue import QueueDelivery, SqsIngestionQueue
from src.rag.staging import RagStagingStore, StagingCapacityError
from src.rag.storage import RagObjectRef, S3ObjectStore
from src.rag.vector_store import RagSearchResult, build_vector_store
from src.rag.visual import (
    FigureCropper,
    VisualProcessor,
    build_visual_processor,
    compose_figure_text,
)
from src.utils.metrics import RAG_METRICS
from src.utils.bounded_state import BoundedTTLMap
from src.utils.schema import ResponseSource, ToolCall

logger = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = {
    "completed",
    "completed_with_warnings",
    "failed",
    "dead_lettered",
}

JOB_STATUSES = {
    "awaiting_upload",
    "enqueue_pending",
    "queued",
    "running",
    "retrying",
    *TERMINAL_JOB_STATUSES,
}


class RagPayloadTooLarge(ValueError):
    """Raised before an oversized RAG upload can enter parsing or indexing."""


class RagIdempotencyConflict(ValueError):
    """Raised when an idempotency key is reused for a different upload request."""


class RagStorageCapacityError(StagingCapacityError):
    """Raised when the shared RAG staging volume is at capacity."""


class RagQueuePreDurabilityError(RuntimeError):
    """A queue submission failure proven to precede every durable side effect."""

    safe_to_cleanup_staging = True


@dataclass
class IngestionJob:
    """Ingestion job record surfaced by the API."""

    job_id: str
    document_id: str
    filename: str
    knowledge_base_id: str
    status: str = "queued"
    chunks_indexed: int = 0
    error: Optional[str] = None
    last_error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    worker_id: Optional[str] = None
    stream_message_id: Optional[str] = None
    batch_id: Optional[str] = None
    storage_ref: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)
    dispatch_id: Optional[str] = None
    index_version: Optional[str] = None
    upload_request_fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "knowledge_base_id": self.knowledge_base_id,
            "status": self.status,
            "chunks_indexed": self.chunks_indexed,
            "error": self.error,
            "last_error": self.last_error,
            "warnings": list(self.warnings),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "worker_id": self.worker_id,
            "stream_message_id": self.stream_message_id,
            "batch_id": self.batch_id,
            "storage_ref": self.storage_ref,
            "source": dict(self.source),
            "dispatch_id": self.dispatch_id,
            "index_version": self.index_version,
            "upload_request_fingerprint": self.upload_request_fingerprint,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IngestionJob":
        return cls(
            job_id=str(payload["job_id"]),
            document_id=str(payload["document_id"]),
            filename=str(payload["filename"]),
            knowledge_base_id=str(payload["knowledge_base_id"]),
            status=str(payload.get("status") or "queued"),
            chunks_indexed=int(payload.get("chunks_indexed") or 0),
            error=payload.get("error"),
            last_error=payload.get("last_error"),
            warnings=list(payload.get("warnings") or []),
            attempts=int(payload.get("attempts") or 0),
            max_attempts=int(payload.get("max_attempts") or 3),
            worker_id=payload.get("worker_id"),
            stream_message_id=payload.get("stream_message_id"),
            batch_id=payload.get("batch_id"),
            storage_ref=payload.get("storage_ref"),
            source=dict(payload.get("source") or {}),
            dispatch_id=payload.get("dispatch_id"),
            index_version=payload.get("index_version"),
            upload_request_fingerprint=payload.get("upload_request_fingerprint"),
            metadata=dict(payload.get("metadata") or {}),
            created_at=_parse_datetime(payload.get("created_at")),
            updated_at=_parse_datetime(payload.get("updated_at")),
        )


@dataclass
class IngestionBatch:
    """Batch manifest progress record."""

    batch_id: str
    knowledge_base_id: str
    total: int
    status_counts: Dict[str, int] = field(default_factory=dict)
    job_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self, jobs: Optional[Sequence[IngestionJob]] = None) -> Dict[str, Any]:
        terminal_count = sum(
            int(self.status_counts.get(status, 0)) for status in TERMINAL_JOB_STATUSES
        )
        return {
            "batch_id": self.batch_id,
            "knowledge_base_id": self.knowledge_base_id,
            "total": self.total,
            "status": "completed"
            if self.total and terminal_count >= self.total
            else "running",
            "status_counts": {
                status: int(self.status_counts.get(status, 0))
                for status in sorted(JOB_STATUSES)
            },
            "job_ids": list(self.job_ids),
            "jobs": [job.to_dict() for job in jobs] if jobs is not None else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IngestionBatch":
        return cls(
            batch_id=str(payload["batch_id"]),
            knowledge_base_id=str(payload.get("knowledge_base_id") or "default"),
            total=int(payload.get("total") or 0),
            status_counts=dict(payload.get("status_counts") or {}),
            job_ids=list(payload.get("job_ids") or []),
            created_at=_parse_datetime(payload.get("created_at")),
            updated_at=_parse_datetime(payload.get("updated_at")),
        )


class RagService:
    """Coordinates parser, chunker, embedding provider, and vector store."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        parser: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
        visual_processor: Optional[VisualProcessor] = None,
        figure_cropper: Optional[FigureCropper] = None,
        object_store: Optional[Any] = None,
        ingestion_queue: Optional[Any] = None,
    ):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.auto_retrieve = bool(self.config.get("auto_retrieve", True))
        self.default_knowledge_base_ids = list(
            self.config.get("default_knowledge_base_ids", []) or []
        )
        self.parser_config = dict(self.config.get("parser", {}) or {})
        self.chunking_config = dict(self.config.get("chunking", {}) or {})
        self.embedding_config = dict(self.config.get("embedding", {}) or {})
        self.retrieval_config = dict(self.config.get("retrieval", {}) or {})
        self.rerank_config = dict(self.config.get("rerank", {}) or {})
        self.visual_config = dict(self.config.get("visual", {}) or {})
        self.redis_config = dict(self.config.get("redis", {}) or {})
        self.intent_gate_config = dict(self.config.get("intent_gate", {}) or {})
        self.queue_config = dict(self.config.get("ingestion_queue", {}) or {})
        self.storage_config = dict(self.config.get("storage", {}) or {})
        self.upload_config = dict(self.config.get("upload", {}) or {})
        self.job_ttl_seconds = int(self.config.get("job_ttl_seconds", 86400))
        self.queue_enabled = bool(self.queue_config.get("enabled", False))
        self.storage_backend = str(
            self.storage_config.get("backend") or "local"
        ).lower()
        self.queue_backend = str(
            self.queue_config.get("backend") or "redis_stream"
        ).lower()
        if (
            self.queue_enabled
            and self.queue_backend == "sqs"
            and self.storage_backend != "s3"
        ):
            raise ValueError("local storage cannot be combined with sqs ingestion")
        if self.storage_backend not in {"local", "s3"}:
            raise ValueError("rag.storage.backend must be local or s3")
        if self.queue_backend not in {"redis_stream", "sqs"}:
            raise ValueError("rag.ingestion_queue.backend must be redis_stream or sqs")
        self.stream_key = str(
            self.queue_config.get("stream_key")
            or self._prefixed_key("ingestion:stream")
        )
        self.group_name = str(
            self.queue_config.get("group_name") or "rag-ingestion-workers"
        )
        self.dead_letter_stream_key = str(
            self.queue_config.get("dead_letter_stream_key")
            or self._prefixed_key("ingestion:dead_letter")
        )
        self.retry_zset_key = str(
            self.queue_config.get("retry_zset_key")
            or self._prefixed_key("ingestion:retry")
        )
        self.worker_concurrency = max(1, int(self.queue_config.get("concurrency", 1)))
        self.worker_count = max(1, int(self.queue_config.get("consumer_count", 1)))
        self.stream_block_ms = int(self.queue_config.get("block_ms", 5000))
        self.pending_idle_ms = int(self.queue_config.get("pending_idle_ms", 300000))
        self.max_attempts = max(1, int(self.queue_config.get("max_attempts", 3)))
        self.retry_backoff_seconds = max(
            0.0, float(self.queue_config.get("retry_backoff_seconds", 30))
        )
        self.stream_maxlen = int(self.queue_config.get("stream_maxlen", 10000))
        self.dead_letter_maxlen = int(
            self.queue_config.get("dead_letter_maxlen", 10000)
        )
        self.heartbeat_interval_seconds = float(
            self.queue_config.get("heartbeat_interval_seconds", 5)
        )
        self.heartbeat_ttl_seconds = int(
            self.queue_config.get("heartbeat_ttl_seconds", 15)
        )
        self.sqs_config = dict(self.queue_config.get("sqs", {}) or {})
        self.sqs_visibility_timeout_seconds = int(
            self.sqs_config.get("visibility_timeout_seconds", 900)
        )
        self.sqs_heartbeat_interval_seconds = float(
            self.sqs_config.get("heartbeat_interval_seconds", 60)
        )
        self.staging_dir = Path(
            self.storage_config.get("staging_dir") or "data/rag/uploads"
        )
        self.cleanup_completed_files = bool(
            self.storage_config.get("cleanup_completed_files", False)
        )
        self.failed_file_ttl_seconds = int(
            self.storage_config.get("failed_file_ttl_seconds", 604800)
        )
        self.orphan_file_ttl_seconds = int(
            self.storage_config.get("orphan_file_ttl_seconds", 604800)
        )
        self.janitor_interval_seconds = float(
            self.storage_config.get("janitor_interval_seconds", 300)
        )
        self._staging_store = (
            RagStagingStore(self.staging_dir, self.storage_config)
            if self.storage_backend == "local"
            else None
        )
        self.object_store = object_store
        if self.storage_backend == "s3" and self.object_store is None:
            self.object_store = S3ObjectStore(
                dict(self.storage_config.get("s3", {}) or {})
            )
        self.ingestion_queue = ingestion_queue
        if self.queue_backend == "sqs" and self.ingestion_queue is None:
            self.ingestion_queue = SqsIngestionQueue(
                dict(self.queue_config.get("sqs", {}) or {})
            )
        self._janitor_task: Optional[asyncio.Task] = None
        self.parser = parser or build_document_parser(self.parser_config)
        self.chunker = build_chunker(self.chunking_config)
        self.embedding_provider = embedding_provider or build_embedding_provider(
            self.embedding_config
        )
        self.reranker = reranker or build_reranker(self.rerank_config)
        self.visual_processor = visual_processor or build_visual_processor(
            self.visual_config
        )
        self.figure_cropper = figure_cropper or FigureCropper(self.visual_config)
        self.vector_store = vector_store or build_vector_store(self.config)
        self._index_generation_lock = asyncio.Lock()
        self._last_local_index_generation = 0
        self.jobs: BoundedTTLMap[str, IngestionJob] = BoundedTTLMap(
            max_entries=int(self.config.get("local_job_cache_max_entries", 1000)),
            ttl_seconds=max(1, self.job_ttl_seconds),
            metric_name="rag_jobs",
        )
        self.batches: BoundedTTLMap[str, IngestionBatch] = BoundedTTLMap(
            max_entries=int(self.config.get("local_batch_cache_max_entries", 100)),
            ttl_seconds=max(1, self.job_ttl_seconds),
            metric_name="rag_batches",
        )
        self._initialized = False
        self._upload_idempotency: BoundedTTLMap[str, str] = BoundedTTLMap(
            max_entries=int(self.config.get("upload_idempotency_max_entries", 1000)),
            ttl_seconds=max(1, self.job_ttl_seconds),
            metric_name="rag_upload_idempotency",
        )

    async def initialize(self):
        if not self.enabled:
            return
        if self._staging_store is not None:
            await asyncio.to_thread(self._staging_store.ensure_reconciled)
        await self.vector_store.initialize()
        if self.queue_enabled:
            if self.storage_backend == "local":
                self.staging_dir.mkdir(parents=True, exist_ok=True)
            if self.queue_backend == "redis_stream":
                await self._ensure_stream_group()
        if self._staging_store is not None:
            self._janitor_task = asyncio.create_task(
                self._staging_janitor_loop(), name="rag_staging_janitor"
            )
        if self._visual_enabled() and hasattr(self.figure_cropper, "assets_dir"):
            self.figure_cropper.assets_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    async def shutdown(self):
        if not self.enabled:
            return
        if self._janitor_task:
            self._janitor_task.cancel()
            await asyncio.gather(self._janitor_task, return_exceptions=True)
            self._janitor_task = None
        await self.vector_store.shutdown()
        close_embedding_provider = getattr(self.embedding_provider, "close", None)
        if callable(close_embedding_provider):
            await close_embedding_provider()
        self._initialized = False

    def is_healthy(self) -> bool:
        return (not self.enabled) or self._initialized

    async def ingest_document(
        self,
        *,
        content: Optional[bytes] = None,
        filename: str,
        knowledge_base_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> IngestionJob:
        job = await self.queue_document_ingestion(
            content=content,
            filename=filename,
            knowledge_base_id=knowledge_base_id,
            metadata=metadata,
            document_id=document_id,
        )
        return await self.process_ingestion_job(
            job.job_id,
            content=content,
            filename=filename,
            knowledge_base_id=job.knowledge_base_id,
            metadata=metadata,
            document_id=job.document_id,
            raise_on_error=True,
        )

    async def queue_document_ingestion(
        self,
        *,
        content: Optional[bytes] = None,
        filename: str,
        knowledge_base_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        storage_ref: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> IngestionJob:
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        max_size = int(self.parser_config.get("max_file_size_bytes", 100_000_000))
        if content is not None and len(content) > max_size:
            raise ValueError("Document is larger than the configured RAG limit")
        if content is None and not storage_ref:
            raise ValueError("content or storage_ref is required")

        resolved_job_id = job_id or str(uuid.uuid4())
        resolved_document_id = document_id or str(uuid.uuid4())
        resolved_kb_id = knowledge_base_id or self._default_write_kb()
        resolved_storage_ref = storage_ref
        resolved_source: Dict[str, Any] = {}
        if self.queue_enabled and content is not None:
            if self.storage_backend == "s3":
                if self.object_store is None:
                    raise RuntimeError("S3 object store is unavailable")
                ref = await self.object_store.put_bytes(
                    job_id=resolved_job_id, content=content
                )
                resolved_storage_ref = ref.uri
                resolved_source = ref.to_dict()
            else:
                resolved_storage_ref = self.stage_document_content(
                    content,
                    filename=filename,
                    knowledge_base_id=resolved_kb_id,
                    document_id=resolved_document_id,
                    job_id=resolved_job_id,
                )
        elif self.queue_enabled and resolved_storage_ref:
            if self.storage_backend == "local":
                self._resolve_staged_path(resolved_storage_ref)
            else:
                if self.object_store is None:
                    raise RuntimeError("S3 object store is unavailable")
                ref = await self.object_store.complete_upload(
                    RagObjectRef(backend="s3", uri=resolved_storage_ref)
                )
                resolved_storage_ref = ref.uri
                resolved_source = ref.to_dict()
        dispatch_id = None
        index_version = None
        if self.queue_backend == "sqs":
            dispatch_id, index_version = await self._new_sqs_dispatch()
        job = IngestionJob(
            job_id=resolved_job_id,
            document_id=resolved_document_id,
            filename=filename,
            knowledge_base_id=resolved_kb_id,
            max_attempts=self.max_attempts,
            batch_id=batch_id,
            storage_ref=resolved_storage_ref,
            source=resolved_source,
            dispatch_id=dispatch_id,
            index_version=index_version,
            metadata=dict(metadata or {}),
        )
        if self.queue_enabled and self.queue_backend == "sqs":
            job.status = "enqueue_pending"
        await self._save_job(job)
        if self.queue_enabled:
            await self.enqueue_ingestion_job(job)
        return job

    async def create_presigned_upload(
        self,
        *,
        filename: str,
        size_bytes: int,
        checksum_sha256: str,
        knowledge_base_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.storage_backend != "s3" or self.object_store is None:
            raise RuntimeError("presigned uploads require the S3 storage backend")
        if not self.queue_enabled:
            raise RuntimeError("presigned uploads require the ingestion queue")
        max_size = int(self.upload_config.get("multipart_max_bytes", 100_000_000))
        if size_bytes < 1 or size_bytes > max_size:
            raise RagPayloadTooLarge("Document exceeds the configured RAG upload limit")
        try:
            decoded_checksum = base64.b64decode(
                checksum_sha256.encode("ascii"), validate=True
            )
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise ValueError("checksum_sha256 must be a base64 SHA-256 digest") from exc
        if len(decoded_checksum) != 32:
            raise ValueError("checksum_sha256 must be a base64 SHA-256 digest")
        resolved_kb_id = knowledge_base_id or self._default_write_kb()
        request_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "filename": filename,
                    "size_bytes": int(size_bytes),
                    "checksum_sha256": checksum_sha256,
                    "knowledge_base_id": resolved_kb_id,
                    "metadata": dict(metadata or {}),
                    "document_id": document_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        job_id = str(uuid.uuid4())
        idempotency_redis_key = None
        claimed_idempotency = False
        if idempotency_key:
            local_job_id = self._upload_idempotency.get(idempotency_key)
            client = self._job_client()
            if client is not None and hasattr(client, "set"):
                key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
                idempotency_redis_key = self._prefixed_key(
                    f"upload:idempotency:{key_hash}"
                )
                claimed_idempotency = bool(
                    await client.set(
                        idempotency_redis_key,
                        job_id,
                        nx=True,
                        ex=self.job_ttl_seconds,
                    )
                )
                if not claimed_idempotency:
                    existing = await client.get(idempotency_redis_key)
                    local_job_id = _decode(existing) if existing else None
            if local_job_id:
                prior = await self.get_job(local_job_id)
                if prior is None:
                    raise RuntimeError("idempotent upload is still being initialized")
                if prior.upload_request_fingerprint != request_fingerprint:
                    raise RagIdempotencyConflict(
                        "Idempotency-Key was already used for a different upload request"
                    )
                ref = RagObjectRef.from_dict(prior.source)
                _, upload = await self.object_store.create_upload(
                    job_id=prior.job_id,
                    size_bytes=int(ref.size_bytes or size_bytes),
                    checksum_sha256=str(ref.checksum_sha256 or checksum_sha256),
                )
                return {"job": prior, "upload": upload}

        resolved_document_id = document_id or str(uuid.uuid4())
        try:
            dispatch_id, index_version = await self._new_sqs_dispatch()
            ref, upload = await self.object_store.create_upload(
                job_id=job_id,
                size_bytes=int(size_bytes),
                checksum_sha256=checksum_sha256,
            )
            job = IngestionJob(
                job_id=job_id,
                document_id=resolved_document_id,
                filename=filename,
                knowledge_base_id=resolved_kb_id,
                status="awaiting_upload",
                max_attempts=self.max_attempts,
                storage_ref=ref.uri,
                source=ref.to_dict(),
                dispatch_id=dispatch_id,
                index_version=index_version,
                upload_request_fingerprint=request_fingerprint,
                metadata=dict(metadata or {}),
            )
            await self._save_job(job)
        except BaseException:
            if claimed_idempotency and idempotency_redis_key:
                await self._release_idempotency_claim(idempotency_redis_key, job_id)
            raise
        if idempotency_key:
            self._upload_idempotency[idempotency_key] = job.job_id
        return {"job": job, "upload": upload}

    async def _release_idempotency_claim(self, key: str, job_id: str) -> None:
        client = self._job_client()
        if client is None:
            return
        current = await client.get(key)
        if current is not None and _decode(current) == job_id:
            await client.delete(key)

    async def complete_presigned_upload(self, job_id: str) -> IngestionJob:
        if self.storage_backend != "s3" or self.object_store is None:
            raise RuntimeError("presigned uploads require the S3 storage backend")
        job = await self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in {"awaiting_upload", "enqueue_pending"}:
            return job
        ref = await self.object_store.complete_upload(
            RagObjectRef.from_dict(job.source)
        )
        job.source = ref.to_dict()
        job.storage_ref = ref.uri
        job.status = "enqueue_pending"
        job.updated_at = datetime.now()
        await self._save_job(job)
        await self._tag_job_source(job, "queued")
        await self.enqueue_ingestion_job(job)
        return job

    async def process_ingestion_job(
        self,
        job_id: str,
        *,
        content: Optional[bytes],
        filename: str,
        knowledge_base_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: str,
        storage_ref: Optional[str] = None,
        worker_id: Optional[str] = None,
        stream_message_id: Optional[str] = None,
        attempt: Optional[int] = None,
        raise_on_error: bool = False,
    ) -> IngestionJob:
        job = await self.get_job(job_id)
        if job is None:
            job = IngestionJob(
                job_id=job_id,
                document_id=document_id,
                filename=filename,
                knowledge_base_id=knowledge_base_id,
                storage_ref=storage_ref,
            )
        try:
            job.status = "running"
            job.worker_id = worker_id
            job.stream_message_id = stream_message_id
            if attempt is not None:
                job.attempts = max(job.attempts, int(attempt))
            job.updated_at = datetime.now()
            await self._save_job(job)
            if content is None:
                if job.source.get("backend") == "s3":
                    if self.object_store is None:
                        raise RuntimeError("S3 object store is unavailable")
                    content = await self.object_store.load(
                        RagObjectRef.from_dict(job.source)
                    )
                else:
                    content = self.load_staged_content(storage_ref or job.storage_ref)
            parsed = await asyncio.to_thread(
                self.parser.parse_bytes,
                content=content,
                filename=filename,
                document_id=document_id,
                metadata={
                    **dict(metadata or job.metadata or {}),
                    "knowledge_base_id": knowledge_base_id,
                },
            )
            visual_warnings = await self._process_visual_blocks(
                content=content,
                filename=filename,
                parsed=parsed,
                knowledge_base_id=knowledge_base_id,
            )
            chunks = self.chunker.chunk(parsed)
            if not chunks:
                raise ValueError("Document parsing produced no indexable text chunks")
            embeddings, warnings = await self._embed_chunks(chunks)
            (
                visual_embeddings,
                visual_embedding_warnings,
            ) = await self._embed_visual_chunks(chunks)
            job.warnings = [*visual_warnings, *warnings, *visual_embedding_warnings]
            job.chunks_indexed = await self.vector_store.upsert_chunks(
                chunks,
                embeddings,
                knowledge_base_id=knowledge_base_id,
                visual_embeddings=visual_embeddings,
                index_version=job.index_version or job.dispatch_id,
            )
            job.status = "completed_with_warnings" if job.warnings else "completed"
            job.error = None
            job.last_error = None
            job.updated_at = datetime.now()
            await self._save_job(job)
            return job
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.last_error = str(exc)
            job.updated_at = datetime.now()
            await self._save_job(job)
            logger.exception("RAG ingestion failed for %s", filename)
            if raise_on_error:
                raise
            return job

    async def retrieve(
        self,
        query: str,
        *,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        max_results: Optional[int] = None,
        candidate_count: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RagSearchResult]:
        if not self.enabled:
            return []
        deadline_seconds = max(
            0.001, float(self.retrieval_config.get("deadline_seconds", 30))
        )
        try:
            async with asyncio.timeout(deadline_seconds):
                return await self._retrieve_within_deadline(
                    query,
                    knowledge_base_ids=knowledge_base_ids,
                    max_results=max_results,
                    candidate_count=candidate_count,
                    min_score=min_score,
                )
        except TimeoutError:
            logger.warning(
                "RAG retrieval exceeded its %.3fs shared deadline",
                deadline_seconds,
            )
            return []

    async def _retrieve_within_deadline(
        self,
        query: str,
        *,
        knowledge_base_ids: Optional[Sequence[str]] = None,
        max_results: Optional[int] = None,
        candidate_count: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RagSearchResult]:
        embedding, visual_embedding = await asyncio.gather(
            self._safe_embed(query),
            self._safe_visual_query_embed(query),
        )
        effective_kbs = self._effective_knowledge_base_ids(knowledge_base_ids)
        top_k = int(max_results or self.retrieval_config.get("top_k", 5))
        first_stage_candidate_count = int(
            candidate_count or self.retrieval_config.get("candidate_count", 30)
        )
        visual_retrieval = dict(self.visual_config.get("retrieval", {}) or {})
        rerank_top_n = int(self.rerank_config.get("top_n", top_k))
        rerank_enabled = self._rerank_enabled()
        first_stage_limit = (
            max(top_k, first_stage_candidate_count, rerank_top_n)
            if rerank_enabled
            else top_k
        )
        results = await self.vector_store.search(
            query,
            embedding,
            knowledge_base_ids=effective_kbs,
            limit=first_stage_limit,
            candidate_count=max(first_stage_candidate_count, first_stage_limit),
            keyword_weight=float(self.retrieval_config.get("keyword_weight", 0.35)),
            vector_weight=float(self.retrieval_config.get("vector_weight", 0.6)),
            recency_weight=float(self.retrieval_config.get("recency_weight", 0.05)),
            min_score=float(
                min_score
                if min_score is not None
                else self.retrieval_config.get("min_score", 0.0)
            ),
            visual_embedding=visual_embedding,
            visual_weight=float(visual_retrieval.get("weight", 0.4)),
            visual_min_score=float(visual_retrieval.get("min_score", 0.0)),
            visual_candidate_count=int(
                visual_retrieval.get("top_k", first_stage_candidate_count)
            ),
        )
        if rerank_enabled:
            results = await self._rerank_results(query, results, top_k, rerank_top_n)
        return results[:top_k]

    def build_context(self, results: Sequence[RagSearchResult]) -> str:
        if not results:
            return ""
        max_context_chars = int(self.retrieval_config.get("max_context_chars", 6000))
        lines = [
            "School document search results (trusted indexed school documents):",
            "Answer only from these sources when they are relevant. Cite sources inline as [S1], [S2], etc. If the sources do not contain the answer, say the school document library does not have enough information.",
        ]
        remaining = max_context_chars - len("\n\n".join(lines)) - 2
        if remaining <= 0:
            return "\n\n".join(lines)[:max_context_chars]
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            title = str(
                chunk.metadata.get("title")
                or chunk.metadata.get("filename")
                or chunk.document_id
            )
            page = (
                f"p. {chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"pp. {chunk.page_start}-{chunk.page_end}"
            )
            text = self._context_text_for_chunk(chunk)
            source_header = (
                f"[S{index}] {title} ({page}, kb={result.knowledge_base_id})\n"
                f"Chunk: {chunk.chunk_id}\n"
            )
            budget = remaining - len(source_header) - len("Content: ") - 2
            if budget <= 0:
                break
            if len(text) > budget:
                text = text[: max(budget - 3, 0)].rstrip() + "..."
            entry = source_header + f"Content: {text}"
            if len(entry) + 2 > remaining:
                break
            lines.append(entry)
            remaining -= len(entry) + 2
        return "\n\n".join(lines)

    def _context_text_for_chunk(self, chunk: DocumentChunk) -> str:
        metadata = dict(chunk.metadata or {})
        if metadata.get("is_figure"):
            parts = []
            if chunk.section_path:
                parts.append(f"Section: {' > '.join(chunk.section_path)}")
            if metadata.get("figure_id"):
                parts.append(f"Figure: {metadata.get('figure_id')}")
            if metadata.get("image_ref"):
                parts.append(f"Image: {metadata.get('image_ref')}")
            if metadata.get("bbox"):
                parts.append(f"Bounding box: {metadata.get('bbox')}")
            for key, title in (
                ("figure_caption", "Caption"),
                ("figure_ocr_text", "OCR text"),
                ("figure_chart_summary", "Chart summary"),
                ("figure_diagram_summary", "Diagram summary"),
            ):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(f"{title}: {' '.join(value.split())}")
            if chunk.text.strip():
                parts.append("Text: " + " ".join(chunk.text.split()))
            return "\n".join(parts)
        if metadata.get("is_table_row"):
            values = dict(metadata.get("table_row_values") or {})
            value_text = "; ".join(
                f"{column} = {value}" for column, value in values.items()
            )
            columns = ", ".join(
                str(value) for value in metadata.get("table_columns") or []
            )
            parts = []
            if chunk.section_path:
                parts.append(f"Section: {' > '.join(chunk.section_path)}")
            if metadata.get("table_id"):
                parts.append(f"Table: {metadata.get('table_id')}")
            if columns:
                parts.append(f"Columns: {columns}")
            parts.append(
                f"Table row: {metadata.get('table_row_label') or 'Untitled row'}"
            )
            if value_text:
                parts.append(f"Values: {value_text}")
            parts.append("Text: " + " ".join(chunk.text.split()))
            return "\n".join(parts)
        if metadata.get("is_table_summary"):
            return "Table summary: " + " ".join(chunk.text.split())
        return " ".join(chunk.text.split())

    def sources_from_results(
        self, results: Sequence[RagSearchResult]
    ) -> List[ResponseSource]:
        sources = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            title = str(
                chunk.metadata.get("title")
                or chunk.metadata.get("filename")
                or chunk.document_id
            )
            sources.append(
                ResponseSource(
                    title=title,
                    url=f"rag://{result.knowledge_base_id}/{chunk.document_id}#{chunk.chunk_id}",
                    snippet=" ".join(chunk.text.split())[:1000],
                    score=result.score,
                    rank=index,
                    source_type="rag",
                    document_id=chunk.document_id,
                    page=chunk.page_start,
                    bbox=chunk.bboxes[0] if chunk.bboxes else None,
                    chunk_id=chunk.chunk_id,
                    index_version=result.index_version,
                    figure_id=chunk.metadata.get("figure_id"),
                    image_ref=chunk.metadata.get("image_ref"),
                )
            )
        return sources

    def tool_call_from_results(
        self,
        query: str,
        results: Sequence[RagSearchResult],
        latency_ms: int,
    ) -> ToolCall:
        return ToolCall(
            name="rag_search",
            provider=str(self.config.get("backend", "redis_stack")),
            arguments={"query": query},
            result_count=len(results),
            latency_ms=latency_ms,
            error=None if results else "rag_no_hits",
        )

    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        client = self._job_client()
        if client:
            raw = await client.get(self._job_key(job_id))
            if raw:
                try:
                    job = IngestionJob.from_dict(json.loads(_decode(raw)))
                    setattr(job, "_persisted_status", job.status)
                    self.jobs[job.job_id] = job
                    return job
                except Exception:
                    logger.warning("Failed to decode RAG ingestion job %s", job_id)
        return self.jobs.get(job_id)

    async def delete_document(
        self, document_id: str, knowledge_base_id: Optional[str] = None
    ) -> int:
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        deleted = await self.vector_store.delete_document(
            document_id, knowledge_base_id
        )
        if self._visual_enabled():
            if knowledge_base_id:
                self.figure_cropper.delete_document_assets(
                    document_id, knowledge_base_id
                )
            else:
                for kb_id in self._effective_knowledge_base_ids(None):
                    self.figure_cropper.delete_document_assets(document_id, kb_id)
        return deleted

    async def create_batch(
        self,
        *,
        documents: Sequence[Dict[str, Any]],
        knowledge_base_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionBatch:
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        if not self.queue_enabled:
            raise RuntimeError("RAG ingestion queue is required for batch ingestion")
        if not documents:
            raise ValueError("batch documents must not be empty")

        decoded_limit = int(
            self.upload_config.get("json_max_decoded_bytes", 10_000_000)
        )
        # Validate every inline document before creating the manifest or enqueueing
        # any jobs. This prevents an oversized later item from leaving a partial
        # batch behind.
        for item in documents:
            encoded = item.get("content_base64")
            if encoded:
                decode_base64_document(str(encoded), max_decoded_bytes=decoded_limit)

        batch_id = str(uuid.uuid4())
        resolved_kb_id = knowledge_base_id or self._default_write_kb()
        batch = IngestionBatch(
            batch_id=batch_id,
            knowledge_base_id=resolved_kb_id,
            total=len(documents),
            status_counts={status: 0 for status in JOB_STATUSES},
        )
        await self._save_batch(batch)

        job_ids = []
        for item in documents:
            filename = str(item.get("filename") or "").strip()
            if not filename:
                raise ValueError("each batch document requires filename")
            item_metadata = {
                **dict(metadata or {}),
                **dict(item.get("metadata") or {}),
            }
            content = item.get("content")
            if content is None and item.get("content_base64"):
                content = decode_base64_document(
                    str(item["content_base64"]),
                    max_decoded_bytes=decoded_limit,
                )
            job = await self.queue_document_ingestion(
                content=content,
                storage_ref=item.get("storage_ref"),
                filename=filename,
                knowledge_base_id=item.get("knowledge_base_id") or resolved_kb_id,
                metadata=item_metadata,
                document_id=item.get("document_id"),
                batch_id=batch_id,
            )
            job_ids.append(job.job_id)

        updated_batch = await self._load_batch(batch_id)
        if updated_batch is None:
            batch.job_ids = job_ids
            batch.updated_at = datetime.now()
            await self._save_batch(batch)
            return batch
        return updated_batch

    async def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        batch = await self._load_batch(batch_id)
        if batch is None:
            return None
        jobs = []
        for job_id in batch.job_ids:
            job = await self.get_job(job_id)
            if job:
                jobs.append(job)
        return batch.to_dict(jobs=jobs)

    async def retry_job(self, job_id: str) -> Optional[IngestionJob]:
        job = await self.get_job(job_id)
        if job is None:
            return None
        if job.status not in {"enqueue_pending", "failed", "dead_lettered", "retrying"}:
            raise ValueError(
                "only enqueue-pending, failed, retrying, or dead-lettered jobs can be retried"
            )
        if not job.storage_ref:
            raise ValueError("job has no staged storage_ref to retry")
        reuse_dispatch = job.status == "enqueue_pending"
        job.status = "enqueue_pending" if self.queue_backend == "sqs" else "queued"
        job.error = None
        job.last_error = None
        if self.queue_backend == "sqs" and not reuse_dispatch:
            job.dispatch_id, job.index_version = await self._new_sqs_dispatch()
        job.updated_at = datetime.now()
        await self._save_job(job)
        await self.enqueue_ingestion_job(job, attempt=job.attempts + 1)
        return job

    async def enqueue_ingestion_job(
        self, job: IngestionJob, attempt: Optional[int] = None
    ) -> str:
        if self.queue_backend == "sqs":
            if self.ingestion_queue is None:
                raise RagQueuePreDurabilityError(
                    "SQS client is required for RAG ingestion queue"
                )
            if not job.dispatch_id:
                job.dispatch_id, job.index_version = await self._new_sqs_dispatch()
            payload = {
                "schema_version": 1,
                "job_id": job.job_id,
                "dispatch_id": job.dispatch_id,
                "index_version": job.index_version,
                "batch_id": job.batch_id or "",
                "document_id": job.document_id,
                "knowledge_base_id": job.knowledge_base_id,
                "filename": job.filename,
                "source": dict(job.source),
                "attempt": int(
                    attempt if attempt is not None else max(job.attempts + 1, 1)
                ),
                "created_at": datetime.now().isoformat(),
            }
            message_id = await self.ingestion_queue.publish(payload)
            job.stream_message_id = str(message_id)
            job.status = "queued"
            job.updated_at = datetime.now()
            try:
                await self._save_job(job)
            except Exception:
                logger.warning(
                    "RAG job %s was sent to SQS but metadata refresh failed",
                    job.job_id,
                    exc_info=True,
                )
            return job.stream_message_id

        client = self._job_client()
        if not client:
            raise RagQueuePreDurabilityError(
                "Redis client is required for RAG ingestion queue"
            )
        await self._ensure_stream_group()
        resolved_attempt = int(
            attempt if attempt is not None else max(job.attempts + 1, 1)
        )
        fields = {
            "job_id": job.job_id,
            "batch_id": job.batch_id or "",
            "document_id": job.document_id,
            "knowledge_base_id": job.knowledge_base_id,
            "storage_ref": job.storage_ref or "",
            "filename": job.filename,
            "attempt": str(resolved_attempt),
            "created_at": datetime.now().isoformat(),
        }
        kwargs: Dict[str, Any] = {}
        if self.stream_maxlen > 0:
            kwargs = {"maxlen": self.stream_maxlen, "approximate": True}
        message_id = await client.xadd(self.stream_key, fields, **kwargs)
        job.stream_message_id = _decode(message_id)
        job.status = "queued"
        job.updated_at = datetime.now()
        try:
            await self._save_job(job)
        except Exception:
            # The stream entry is already durable and contains every field the
            # worker needs. Deleting its staged file would turn a recoverable
            # metadata refresh failure into guaranteed data loss. The initial
            # job record written before XADD remains the worker/API fallback.
            logger.warning(
                "RAG job %s was enqueued but its stream metadata refresh failed",
                job.job_id,
                exc_info=True,
            )
        return job.stream_message_id

    async def run_ingestion_workers(self):
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        if not self.queue_enabled:
            raise RuntimeError("RAG ingestion queue is disabled")
        if self.queue_backend == "redis_stream":
            await self._ensure_stream_group()
        workers = [
            asyncio.create_task(
                (
                    self._sqs_worker_loop(self._consumer_name(index))
                    if self.queue_backend == "sqs"
                    else self._worker_loop(self._consumer_name(index))
                ),
                name=f"rag_ingestion_worker_{index}",
            )
            for index in range(self.worker_count)
        ]
        worker_id = f"{socket.gethostname()}-{os.getpid()}"
        heartbeat = asyncio.create_task(
            self._worker_heartbeat_loop(worker_id),
            name="rag_ingestion_worker_heartbeat",
        )
        tasks = [*workers, heartbeat]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _worker_heartbeat_loop(self, worker_id: str) -> None:
        client = self._job_client()
        if not client:
            raise RuntimeError("Redis client is required for RAG worker heartbeat")
        key = self._prefixed_key(f"worker:heartbeat:{worker_id}")
        consecutive_failures = 0
        failure_limit = max(
            2,
            int(self.heartbeat_ttl_seconds / self.heartbeat_interval_seconds),
        )
        while True:
            try:
                await client.set(
                    key,
                    datetime.now().isoformat(),
                    ex=self.heartbeat_ttl_seconds,
                )
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    "RAG worker heartbeat failed (%d/%d)",
                    consecutive_failures,
                    failure_limit,
                    exc_info=True,
                )
                if consecutive_failures >= failure_limit:
                    raise RuntimeError(
                        "RAG worker heartbeat exceeded its Redis failure limit"
                    )
            await asyncio.sleep(self.heartbeat_interval_seconds)

    async def _worker_loop(self, consumer_name: str):
        semaphore = asyncio.Semaphore(self.worker_concurrency)
        while True:
            await self.promote_due_retries()
            claimed = await self.claim_pending_jobs(consumer_name)
            if claimed:
                await self._process_stream_entries(claimed, consumer_name, semaphore)
                continue

            entries = await self.read_new_jobs(consumer_name)
            if entries:
                await self._process_stream_entries(entries, consumer_name, semaphore)

    async def _process_stream_entries(
        self,
        entries: Sequence[tuple[str, Dict[str, Any]]],
        consumer_name: str,
        semaphore: asyncio.Semaphore,
    ):
        async def run_one(message_id: str, fields: Dict[str, Any]):
            async with semaphore:
                await self.process_stream_message(message_id, fields, consumer_name)

        await asyncio.gather(
            *(run_one(message_id, fields) for message_id, fields in entries)
        )

    async def process_stream_message(
        self, message_id: str, fields: Dict[str, Any], consumer_name: str
    ) -> Optional[IngestionJob]:
        decoded_fields = {_decode(key): _decode(value) for key, value in fields.items()}
        job_id = decoded_fields.get("job_id")
        if not job_id:
            await self.ack_stream_message(message_id)
            return None
        job = await self.get_job(job_id)
        if job is None:
            await self.ack_stream_message(message_id)
            return None
        if job.status in {"completed", "completed_with_warnings", "dead_lettered"}:
            await self.ack_stream_message(message_id)
            return job

        attempt = int(decoded_fields.get("attempt") or job.attempts + 1)
        try:
            job = await self.process_ingestion_job(
                job.job_id,
                content=None,
                filename=job.filename,
                knowledge_base_id=job.knowledge_base_id,
                metadata=job.metadata,
                document_id=job.document_id,
                storage_ref=decoded_fields.get("storage_ref") or job.storage_ref,
                worker_id=consumer_name,
                stream_message_id=message_id,
                attempt=attempt,
                raise_on_error=True,
            )
        except Exception as exc:
            latest_job = await self.get_job(job.job_id)
            await self._handle_stream_failure(
                latest_job or job,
                message_id=message_id,
                error=str(exc),
                attempt=attempt,
            )
            return await self.get_job(job.job_id)
        await self._tag_job_source_best_effort(job, job.status)
        await self.ack_stream_message(message_id)
        return job

    async def _sqs_worker_loop(self, consumer_name: str) -> None:
        if self.ingestion_queue is None:
            raise RuntimeError("SQS ingestion queue is unavailable")
        while True:
            deliveries = await self.ingestion_queue.receive(
                max_messages=self.worker_concurrency
            )
            if not deliveries:
                continue
            semaphore = asyncio.Semaphore(self.worker_concurrency)

            async def run_one(delivery: QueueDelivery):
                async with semaphore:
                    await self.process_sqs_delivery(delivery, consumer_name)

            await asyncio.gather(*(run_one(delivery) for delivery in deliveries))

    async def process_sqs_delivery(
        self, delivery: QueueDelivery, consumer_name: str
    ) -> Optional[IngestionJob]:
        if self.ingestion_queue is None:
            raise RuntimeError("SQS ingestion queue is unavailable")
        payload = dict(delivery.payload)
        job_id = str(payload.get("job_id") or "")
        dispatch_id = str(payload.get("dispatch_id") or "")
        if not job_id or not dispatch_id:
            RAG_METRICS.delivery_outcomes.labels("sqs", "invalid").inc()
            await self.ingestion_queue.extend_visibility(
                delivery, int(self._retry_delay_seconds(delivery.receive_count))
            )
            return None
        job = await self.get_job(job_id)
        if job is None:
            RAG_METRICS.delivery_outcomes.labels("sqs", "missing_job").inc()
            await self._ack_sqs_best_effort(delivery)
            return None
        if job.dispatch_id != dispatch_id:
            RAG_METRICS.delivery_outcomes.labels("sqs", "stale_dispatch").inc()
            await self._ack_sqs_best_effort(delivery)
            return job
        if job.status in {"completed", "completed_with_warnings", "dead_lettered"}:
            if job.status != "dead_lettered":
                await self._ack_sqs_best_effort(delivery)
            RAG_METRICS.delivery_outcomes.labels("sqs", "terminal").inc()
            return job

        lease = await self._acquire_sqs_processing_lease(
            job, dispatch_id=dispatch_id, consumer_name=consumer_name
        )
        if lease is None:
            RAG_METRICS.processing_lease_contention.inc()
            RAG_METRICS.delivery_outcomes.labels("sqs", "lease_contended").inc()
            await self.ingestion_queue.extend_visibility(
                delivery, int(self._retry_delay_seconds(delivery.receive_count))
            )
            return job

        processing = asyncio.create_task(
            self.process_ingestion_job(
                job.job_id,
                content=None,
                filename=job.filename,
                knowledge_base_id=job.knowledge_base_id,
                metadata=job.metadata,
                document_id=job.document_id,
                storage_ref=job.storage_ref,
                worker_id=consumer_name,
                stream_message_id=delivery.message_id,
                attempt=delivery.receive_count,
                raise_on_error=True,
            ),
            name=f"rag_sqs_processing:{job.job_id}",
        )
        heartbeat = asyncio.create_task(
            self._sqs_delivery_heartbeat(delivery, lease[0], lease[1]),
            name=f"rag_sqs_visibility:{job.job_id}",
        )
        processing_started = time.perf_counter()
        try:
            done, _ = await asyncio.wait(
                {processing, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if processing in done:
                result = await processing
            else:
                heartbeat_error = heartbeat.exception()
                processing.cancel()
                await asyncio.gather(processing, return_exceptions=True)
                if heartbeat_error is not None:
                    raise heartbeat_error
                raise RuntimeError("SQS visibility heartbeat stopped unexpectedly")
        except Exception as exc:
            latest = await self.get_job(job.job_id) or job
            latest.attempts = max(latest.attempts, delivery.receive_count)
            latest.error = str(exc)
            latest.last_error = str(exc)
            latest.updated_at = datetime.now()
            if delivery.receive_count >= latest.max_attempts:
                latest.status = "dead_lettered"
                await self._save_job(latest)
                await self._tag_job_source_best_effort(latest, "dead_lettered")
                RAG_METRICS.delivery_outcomes.labels("sqs", "dead_lettered").inc()
                duration_outcome = "dead_lettered"
            else:
                latest.status = "retrying"
                await self._save_job(latest)
                try:
                    await self.ingestion_queue.extend_visibility(
                        delivery,
                        int(self._retry_delay_seconds(delivery.receive_count)),
                    )
                except Exception:
                    logger.warning(
                        "Failed to defer retry for SQS message %s",
                        delivery.message_id,
                        exc_info=True,
                    )
                RAG_METRICS.delivery_outcomes.labels("sqs", "retrying").inc()
                duration_outcome = "retrying"
            RAG_METRICS.processing_duration.labels("sqs", duration_outcome).observe(
                time.perf_counter() - processing_started
            )
            return latest
        else:
            await self._tag_job_source_best_effort(result, result.status)
            if await self._ack_sqs_best_effort(delivery):
                RAG_METRICS.delivery_outcomes.labels("sqs", "completed").inc()
            RAG_METRICS.processing_duration.labels("sqs", "completed").observe(
                time.perf_counter() - processing_started
            )
            return result
        finally:
            processing.cancel()
            heartbeat.cancel()
            await asyncio.gather(processing, heartbeat, return_exceptions=True)
            await self._release_sqs_processing_lease(*lease)

    async def _acquire_sqs_processing_lease(
        self,
        job: IngestionJob,
        *,
        dispatch_id: str,
        consumer_name: str,
    ) -> Optional[tuple[str, str]]:
        client = self._job_client()
        if client is None or not hasattr(client, "set"):
            raise RuntimeError("Redis is required for SQS processing leases")
        key = self._prefixed_key(f"processing:{job.job_id}:{dispatch_id}")
        token = f"{consumer_name}:{uuid.uuid4()}"
        acquired = await client.set(
            key,
            token,
            nx=True,
            ex=max(
                self.sqs_visibility_timeout_seconds,
                int(self.sqs_heartbeat_interval_seconds * 3),
            ),
        )
        return (key, token) if acquired else None

    async def _release_sqs_processing_lease(self, key: str, token: str) -> None:
        client = self._job_client()
        if client is None:
            return
        if hasattr(client, "eval"):
            await client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end",
                1,
                key,
                token,
            )
            return
        current = await client.get(key)
        if current is not None and _decode(current) == token:
            await client.delete(key)

    async def _sqs_delivery_heartbeat(
        self, delivery: QueueDelivery, lease_key: str, lease_token: str
    ) -> None:
        queue = self.ingestion_queue
        if queue is None:
            raise RuntimeError("SQS ingestion queue is unavailable")
        while True:
            await asyncio.sleep(self.sqs_heartbeat_interval_seconds)
            await queue.extend_visibility(delivery, self.sqs_visibility_timeout_seconds)
            client = self._job_client()
            if client is None or not hasattr(client, "eval"):
                raise RuntimeError(
                    "Redis Lua support is required for SQS lease renewal"
                )
            renewed = await client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end",
                1,
                lease_key,
                lease_token,
                max(
                    self.sqs_visibility_timeout_seconds,
                    int(self.sqs_heartbeat_interval_seconds * 3),
                ),
            )
            if not renewed:
                raise RuntimeError(
                    "SQS processing lease was lost while extending visibility"
                )

    async def _tag_job_source(self, job: IngestionJob, status: str) -> None:
        if job.source.get("backend") != "s3" or self.object_store is None:
            return
        tag = getattr(self.object_store, "tag", None)
        if callable(tag):
            await tag(RagObjectRef.from_dict(job.source), status)

    async def _ack_sqs_best_effort(self, delivery: QueueDelivery) -> bool:
        if self.ingestion_queue is None:
            return False
        try:
            await self.ingestion_queue.ack(delivery)
        except Exception:
            logger.warning(
                "SQS DeleteMessage failed for %s; leaving it for ack-only redelivery",
                delivery.message_id,
                exc_info=True,
            )
            RAG_METRICS.delivery_outcomes.labels("sqs", "ack_failed").inc()
            return False
        return True

    async def _tag_job_source_best_effort(self, job: IngestionJob, status: str) -> None:
        try:
            await self._tag_job_source(job, status)
        except Exception:
            logger.warning(
                "Failed to tag RAG source for job %s as %s",
                job.job_id,
                status,
                exc_info=True,
            )

    async def promote_due_retries(self) -> int:
        client = self._job_client()
        if not client:
            return 0
        now = time.time()
        due = await client.zrangebyscore(self.retry_zset_key, 0, now)
        count = 0
        for raw_job_id in due:
            job_id = _decode(raw_job_id)
            job = await self.get_job(job_id)
            await client.zrem(self.retry_zset_key, job_id)
            if job is None or job.status in TERMINAL_JOB_STATUSES:
                continue
            job.status = "queued"
            job.updated_at = datetime.now()
            await self._save_job(job)
            await self.enqueue_ingestion_job(job, attempt=job.attempts + 1)
            count += 1
        return count

    async def read_new_jobs(
        self, consumer_name: str
    ) -> List[tuple[str, Dict[str, Any]]]:
        client = self._job_client()
        if not client:
            return []
        raw = await client.xreadgroup(
            self.group_name,
            consumer_name,
            {self.stream_key: ">"},
            count=self.worker_concurrency,
            block=self.stream_block_ms,
        )
        return _parse_stream_read(raw)

    async def claim_pending_jobs(
        self, consumer_name: str
    ) -> List[tuple[str, Dict[str, Any]]]:
        client = self._job_client()
        if not client:
            return []
        try:
            raw = await client.xautoclaim(
                self.stream_key,
                self.group_name,
                consumer_name,
                self.pending_idle_ms,
                "0-0",
                count=self.worker_concurrency,
            )
        except Exception as exc:
            logger.debug("RAG pending claim skipped: %s", exc)
            return []
        return _parse_autoclaim(raw)

    async def ack_stream_message(self, message_id: str) -> int:
        client = self._job_client()
        if not client:
            return 0
        return int(await client.xack(self.stream_key, self.group_name, message_id))

    async def _handle_stream_failure(
        self,
        job: IngestionJob,
        *,
        message_id: str,
        error: str,
        attempt: int,
    ) -> None:
        client = self._job_client()
        if not client:
            return
        job.attempts = max(job.attempts, attempt)
        job.last_error = error
        job.error = error
        if job.attempts < job.max_attempts:
            job.status = "retrying"
            job.updated_at = datetime.now()
            await self._save_job(job)
            due_at = time.time() + self._retry_delay_seconds(job.attempts)
            await client.zadd(self.retry_zset_key, {job.job_id: due_at})
            await self.ack_stream_message(message_id)
            return

        job.status = "dead_lettered"
        job.updated_at = datetime.now()
        await self._save_job(job)
        await self._tag_job_source_best_effort(job, "dead_lettered")
        await client.xadd(
            self.dead_letter_stream_key,
            {
                "job_id": job.job_id,
                "batch_id": job.batch_id or "",
                "document_id": job.document_id,
                "knowledge_base_id": job.knowledge_base_id,
                "filename": job.filename,
                "storage_ref": job.storage_ref or "",
                "attempts": str(job.attempts),
                "error": error,
                "created_at": datetime.now().isoformat(),
            },
            maxlen=self.dead_letter_maxlen,
            approximate=True,
        )
        await self.ack_stream_message(message_id)

    def serialize_results(
        self, results: Sequence[RagSearchResult]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "rank": index,
                "score": result.score,
                "match_source": result.match_source,
                "knowledge_base_id": result.knowledge_base_id,
                "chunk_id": result.chunk.chunk_id,
                "document_id": result.chunk.document_id,
                "page_start": result.chunk.page_start,
                "page_end": result.chunk.page_end,
                "section_path": result.chunk.section_path,
                "snippet": " ".join(result.chunk.text.split())[:1000],
                "metadata": result.chunk.metadata,
                "index_version": result.index_version,
            }
            for index, result in enumerate(results, start=1)
        ]

    async def _embed_chunks(
        self, chunks: Sequence[DocumentChunk]
    ) -> tuple[List[Optional[List[float]]], List[str]]:
        embeddings: List[Optional[List[float]]] = [None] * len(chunks)
        eligible = [
            (index, chunk.text)
            for index, chunk in enumerate(chunks)
            if not (chunk.metadata.get("is_figure") and not chunk.text.strip())
        ]
        embed_many = getattr(self.embedding_provider, "embed_many", None)
        batch_size = max(1, int(self.embedding_config.get("batch_size", 32)))
        max_concurrency = max(
            1, int(self.embedding_config.get("max_concurrent_batches", 2))
        )
        semaphore = asyncio.Semaphore(max_concurrency)

        async def embed_batch(
            batch: Sequence[tuple[int, str]],
        ) -> tuple[Sequence[tuple[int, str]], List[Optional[List[float]]]]:
            async with semaphore:
                if callable(embed_many):
                    try:
                        values = await asyncio.wait_for(
                            embed_many([text for _, text in batch]),
                            timeout=float(self.embedding_config.get("timeout", 30)),
                        )
                        if len(values) != len(batch):
                            raise ValueError("embedding batch response length mismatch")
                        return batch, [list(value) for value in values]
                    except Exception as exc:
                        logger.info(
                            "RAG embedding batch unavailable; retrying scalar: %s",
                            exc,
                        )
                        scalar_values: List[Optional[List[float]]] = []
                        for _, text in batch:
                            try:
                                value = await asyncio.wait_for(
                                    self.embedding_provider.embed(text),
                                    timeout=float(
                                        self.embedding_config.get("timeout", 30)
                                    ),
                                )
                            except Exception as scalar_exc:
                                raise RuntimeError(
                                    "scalar embedding fallback failed after batch error"
                                ) from scalar_exc
                            if value is None:
                                raise RuntimeError(
                                    "scalar embedding fallback failed after batch error"
                                )
                            scalar_values.append(list(value))
                        return batch, scalar_values
                values = []
                for _, text in batch:
                    values.append(await self._safe_embed(text))
                return batch, values

        batches = [
            eligible[index : index + batch_size]
            for index in range(0, len(eligible), batch_size)
        ]
        for batch, values in await asyncio.gather(
            *(embed_batch(batch) for batch in batches)
        ):
            for (chunk_index, _), value in zip(batch, values):
                embeddings[chunk_index] = value

        failures = sum(1 for index, _ in eligible if embeddings[index] is None)
        warnings = []
        if failures:
            warnings.append(
                f"Embedding unavailable for {failures} of {len(chunks)} chunks; "
                "indexed affected chunks for keyword-only retrieval."
            )
        return embeddings, warnings

    async def _process_visual_blocks(
        self,
        *,
        content: bytes,
        filename: str,
        parsed: Any,
        knowledge_base_id: str,
    ) -> List[str]:
        if not self._visual_enabled():
            return []
        warnings = await asyncio.to_thread(
            self.figure_cropper.crop_figures,
            content=content,
            filename=filename,
            parsed_document=parsed,
            knowledge_base_id=knowledge_base_id,
        )
        figure_blocks = [
            block
            for block in getattr(parsed, "blocks", [])
            if getattr(block, "block_type", "") == "figure"
            and isinstance(getattr(block, "metadata", None), dict)
            and isinstance(block.metadata.get("figure"), dict)
        ]
        for block in figure_blocks:
            figure = block.metadata["figure"]
            try:
                result = await self.visual_processor.process_figure(figure)
            except Exception as exc:
                warning = (
                    f"Visual processing failed for {figure.get('figure_id')}: {exc}"
                )
                warnings.append(warning)
                figure.setdefault("visual", {})["warnings"] = [warning]
                continue
            figure["visual"] = result.to_metadata()
            warnings.extend(result.warnings)
            block.text = compose_figure_text(block)
        if warnings and self.visual_config.get("required", False):
            raise RuntimeError("; ".join(warnings))
        return warnings

    async def _embed_visual_chunks(
        self, chunks: Sequence[DocumentChunk]
    ) -> tuple[List[Optional[List[float]]], List[str]]:
        visual_embeddings: List[Optional[List[float]]] = []
        failures = 0
        for chunk in chunks:
            embedding = chunk.metadata.get("visual_embedding")
            if isinstance(embedding, list):
                try:
                    visual_embeddings.append([float(value) for value in embedding])
                    continue
                except (TypeError, ValueError):
                    pass
            if chunk.metadata.get("is_figure"):
                failures += 1
            visual_embeddings.append(None)
        warnings = []
        if failures and self._visual_embedding_enabled():
            warnings.append(
                f"Visual embedding unavailable for {failures} figure chunks; "
                "indexed affected chunks for OCR/caption text retrieval only."
            )
        return visual_embeddings, warnings

    async def _safe_embed(self, text: str) -> Optional[List[float]]:
        try:
            return await asyncio.wait_for(
                self.embedding_provider.embed(text),
                timeout=float(self.embedding_config.get("timeout", 30)),
            )
        except Exception as exc:
            logger.info("RAG embedding unavailable; using keyword-only search: %s", exc)
            return None

    async def _safe_visual_query_embed(self, query: str) -> Optional[List[float]]:
        if not self._visual_embedding_enabled():
            return None
        try:
            visual_embedding_config = dict(
                self.visual_config.get("embedding", {}) or {}
            )
            return await asyncio.wait_for(
                self.visual_processor.embed_query(query),
                timeout=float(visual_embedding_config.get("timeout", 30)),
            )
        except Exception as exc:
            logger.info("RAG visual embedding unavailable: %s", exc)
            return None

    def _visual_enabled(self) -> bool:
        return bool(self.visual_config.get("enabled", False))

    def _visual_embedding_enabled(self) -> bool:
        return self._visual_enabled() and bool(
            dict(self.visual_config.get("embedding", {}) or {}).get("enabled", False)
        )

    def _rerank_enabled(self) -> bool:
        return bool(self.rerank_config.get("enabled", False))

    async def _rerank_results(
        self,
        query: str,
        results: Sequence[RagSearchResult],
        top_k: int,
        rerank_top_n: int,
    ) -> List[RagSearchResult]:
        if len(results) <= 1:
            return list(results)
        head_count = max(top_k, min(len(results), rerank_top_n))
        head = list(results[:head_count])
        tail = list(results[head_count:])
        try:
            reranked = await asyncio.wait_for(
                self.reranker.rerank(query, head),
                timeout=float(self.rerank_config.get("timeout", 30)),
            )
        except Exception as exc:
            logger.warning("RAG rerank failed; using first-stage order: %s", exc)
            return list(results)
        if len(reranked) != len(head):
            logger.warning("RAG rerank returned an unexpected number of results")
            return list(results)
        return list(reranked) + tail

    def _effective_knowledge_base_ids(
        self, knowledge_base_ids: Optional[Sequence[str]]
    ) -> List[str]:
        provided = [str(value) for value in (knowledge_base_ids or []) if str(value)]
        if provided:
            return provided
        return list(self.default_knowledge_base_ids)

    def _default_write_kb(self) -> str:
        if self.default_knowledge_base_ids:
            return self.default_knowledge_base_ids[0]
        return "default"

    async def _new_sqs_dispatch(self) -> tuple[str, str]:
        dispatch_id = str(uuid.uuid4())
        client = self._job_client()
        if client is not None and hasattr(client, "eval"):
            sequence = int(
                await client.eval(
                    "local current = redis.call('GET', KEYS[1]); "
                    "local floor = ARGV[1]; "
                    "if (not current) or (#current < #floor) or "
                    "(#current == #floor and current < floor) then "
                    "redis.call('SET', KEYS[1], floor); end; "
                    "return redis.call('INCR', KEYS[1])",
                    1,
                    self._prefixed_key("index:generation_sequence"),
                    str(time.time_ns()),
                )
            )
        else:
            async with self._index_generation_lock:
                sequence = max(time.time_ns(), self._last_local_index_generation + 1)
                self._last_local_index_generation = sequence
        return dispatch_id, f"{sequence:020d}:{dispatch_id}"

    async def _save_job(self, job: IngestionJob) -> None:
        previous_status = getattr(job, "_persisted_status", None)
        if previous_status is None:
            previous_status = await self._stored_job_status(job.job_id)
        self.jobs[job.job_id] = job
        client = self._job_client()
        if not client:
            await self._update_batch_for_status_change(
                job,
                previous_status=previous_status,
                new_status=job.status,
            )
            setattr(job, "_persisted_status", job.status)
            self._update_staging_manifest(job)
            return
        job_ttl = self._job_retention_ttl(job.status)
        await client.setex(
            self._job_key(job.job_id),
            job_ttl,
            json.dumps(job.to_dict(), sort_keys=True),
        )
        await self._update_batch_for_status_change(
            job,
            previous_status=previous_status,
            new_status=job.status,
        )
        setattr(job, "_persisted_status", job.status)
        self._update_staging_manifest(job)

    def _job_retention_ttl(self, status: str) -> int:
        if status in {"completed", "completed_with_warnings"}:
            return int(self.storage_config.get("completed_file_ttl_seconds", 86400))
        return max(self.job_ttl_seconds, self.failed_file_ttl_seconds)

    def _update_staging_manifest(self, job: IngestionJob) -> None:
        if not job.storage_ref or self._staging_store is None:
            return
        try:
            self._staging_store.update_status(job.storage_ref, job.status)
        except FileNotFoundError:
            logger.debug("No staging manifest for job %s", job.job_id)

    async def _stored_job_status(self, job_id: str) -> Optional[str]:
        existing = self.jobs.get(job_id)
        if existing:
            return existing.status
        client = self._job_client()
        if client:
            raw = await client.get(self._job_key(job_id))
            if raw:
                try:
                    payload = json.loads(_decode(raw))
                    return str(payload.get("status") or "")
                except Exception:
                    return None
        return None

    def _job_client(self) -> Optional[Any]:
        client = getattr(self.vector_store, "client", None)
        if client and hasattr(client, "get") and hasattr(client, "setex"):
            return client
        return None

    def _job_key(self, job_id: str) -> str:
        key_prefix = str(self.redis_config.get("key_prefix") or "rag")
        return f"{key_prefix}:jobs:{job_id}"

    def _prefixed_key(self, suffix: str) -> str:
        key_prefix = str(self.redis_config.get("key_prefix") or "rag")
        return f"{key_prefix}:{suffix}"

    def stage_document_content(
        self,
        content: bytes,
        *,
        filename: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: Optional[str] = None,
    ) -> str:
        if self._staging_store is None:
            raise RuntimeError("local staging is unavailable for the S3 backend")
        safe_filename = _safe_path_component(filename or "document.pdf")
        resolved_job_id = _safe_path_component(job_id or str(uuid.uuid4()))
        try:
            return self._staging_store.stage_bytes(
                resolved_job_id, safe_filename, content
            )
        except StagingCapacityError as exc:
            raise RagStorageCapacityError(str(exc)) from exc

    async def stage_uploaded_file(
        self,
        upload: Any,
        *,
        filename: str,
        knowledge_base_id: str,
        document_id: str,
        job_id: Optional[str] = None,
    ) -> str:
        """Stream an upload to disk and publish it with an atomic rename."""
        if self._staging_store is None:
            raise RuntimeError("local staging is unavailable for the S3 backend")
        safe_filename = _safe_path_component(filename or "document.pdf")
        resolved_job_id = _safe_path_component(job_id or str(uuid.uuid4()))
        max_bytes = int(self.upload_config.get("multipart_max_bytes", 100_000_000))
        chunk_bytes = max(
            1, int(self.upload_config.get("stream_chunk_bytes", 1_048_576))
        )
        size = 0
        published = False
        try:
            temporary, target = self._staging_store.begin(
                resolved_job_id, safe_filename
            )
        except StagingCapacityError as exc:
            raise RagStorageCapacityError(str(exc)) from exc
        try:
            with temporary.open("xb") as staged_file:
                while True:
                    chunk = await upload.read(chunk_bytes)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise RagPayloadTooLarge(
                            "Multipart document exceeds the configured RAG upload limit"
                        )
                    try:
                        self._staging_store.reserve(resolved_job_id, len(chunk))
                    except StagingCapacityError as exc:
                        raise RagStorageCapacityError(str(exc)) from exc
                    await asyncio.to_thread(staged_file.write, chunk)
                await asyncio.to_thread(staged_file.flush)
                await asyncio.to_thread(os.fsync, staged_file.fileno())
            publish_task = asyncio.create_task(
                asyncio.to_thread(
                    self._staging_store.publish,
                    resolved_job_id,
                    temporary,
                    target,
                )
            )
            try:
                await asyncio.shield(publish_task)
            except asyncio.CancelledError:
                # The thread cannot be cancelled once os.replace has started. Wait
                # for its deterministic outcome, then remove an otherwise
                # unreferenced target before propagating cancellation.
                await publish_task
                self._staging_store.rollback(resolved_job_id)
                raise
            published = True
            return publish_task.result()
        finally:
            if not published:
                self._staging_store.rollback(resolved_job_id)

    def load_staged_content(self, storage_ref: Optional[str]) -> bytes:
        if self._staging_store is None:
            raise RuntimeError("local staging is unavailable for the S3 backend")
        if not storage_ref:
            raise ValueError("storage_ref is required for queued ingestion")
        path = self._resolve_staged_path(storage_ref)
        max_size = int(self.parser_config.get("max_file_size_bytes", 100_000_000))
        if path.stat().st_size > max_size:
            raise ValueError("Document is larger than the configured RAG limit")
        return path.read_bytes()

    def cleanup_staged_file(self, storage_ref: str) -> None:
        if self._staging_store is None:
            return
        try:
            self._staging_store.delete(storage_ref)
        except Exception:
            logger.debug(
                "Failed to clean staged RAG file %s", storage_ref, exc_info=True
            )

    async def _staging_janitor_loop(self) -> None:
        if self._staging_store is None:
            return
        while True:
            await asyncio.sleep(self.janitor_interval_seconds)
            try:
                await asyncio.to_thread(self._staging_store.run_janitor)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("RAG staging janitor failed", exc_info=True)

    def _resolve_staged_path(self, storage_ref: str) -> Path:
        base = self.staging_dir.resolve()
        path = Path(storage_ref)
        if not path.is_absolute():
            path = Path(storage_ref)
        resolved = path.resolve()
        if base != resolved and base not in resolved.parents:
            raise ValueError(
                "storage_ref must be inside the configured RAG staging_dir"
            )
        return resolved

    async def _ensure_stream_group(self) -> None:
        client = self._job_client()
        if not client:
            return
        try:
            await client.xgroup_create(
                self.stream_key,
                self.group_name,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _consumer_name(self, index: int) -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{index}:{uuid.uuid4().hex[:8]}"

    def _retry_delay_seconds(self, attempts: int) -> float:
        return self.retry_backoff_seconds * max(1, attempts)

    async def _save_batch(self, batch: IngestionBatch) -> None:
        self.batches[batch.batch_id] = batch
        client = self._job_client()
        if not client:
            return
        await client.setex(
            self._batch_key(batch.batch_id),
            self.job_ttl_seconds,
            json.dumps(batch.to_dict(jobs=[]), sort_keys=True),
        )
        if batch.job_ids:
            await client.delete(self._batch_jobs_key(batch.batch_id))
            for job_id in batch.job_ids:
                await client.sadd(self._batch_jobs_key(batch.batch_id), job_id)
            await client.expire(
                self._batch_jobs_key(batch.batch_id), self.job_ttl_seconds
            )

    async def _load_batch(self, batch_id: str) -> Optional[IngestionBatch]:
        client = self._job_client()
        if client:
            raw = await client.get(self._batch_key(batch_id))
            if raw:
                try:
                    payload = json.loads(_decode(raw))
                    job_ids = await client.smembers(self._batch_jobs_key(batch_id))
                    if job_ids:
                        payload["job_ids"] = [_decode(job_id) for job_id in job_ids]
                    return IngestionBatch.from_dict(payload)
                except Exception:
                    logger.warning("Failed to decode RAG batch %s", batch_id)
        return self.batches.get(batch_id)

    async def _update_batch_for_status_change(
        self,
        job: IngestionJob,
        *,
        previous_status: Optional[str],
        new_status: str,
    ) -> None:
        if not job.batch_id or previous_status == new_status:
            return
        batch = await self._load_batch(job.batch_id)
        if batch is None:
            return
        if previous_status in JOB_STATUSES:
            batch.status_counts[previous_status] = max(
                int(batch.status_counts.get(previous_status, 0)) - 1,
                0,
            )
        if new_status in JOB_STATUSES:
            batch.status_counts[new_status] = (
                int(batch.status_counts.get(new_status, 0)) + 1
            )
        if job.job_id not in batch.job_ids:
            batch.job_ids.append(job.job_id)
        batch.updated_at = datetime.now()
        await self._save_batch(batch)

    def _batch_key(self, batch_id: str) -> str:
        return self._prefixed_key(f"batch:{batch_id}")

    def _batch_jobs_key(self, batch_id: str) -> str:
        return self._prefixed_key(f"batch:{batch_id}:jobs")


def decode_base64_document(
    payload: str, *, max_decoded_bytes: Optional[int] = None
) -> bytes:
    if max_decoded_bytes is not None:
        padding = len(payload) - len(payload.rstrip("="))
        estimated_size = (len(payload) * 3) // 4 - padding
        if estimated_size > max_decoded_bytes:
            raise RagPayloadTooLarge(
                "JSON document exceeds the configured decoded RAG upload limit"
            )
    try:
        decoded = base64.b64decode(payload.encode("utf-8"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("content_base64 must be valid base64") from exc
    if max_decoded_bytes is not None and len(decoded) > max_decoded_bytes:
        raise RagPayloadTooLarge(
            "JSON document exceeds the configured decoded RAG upload limit"
        )
    return decoded


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value:
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            pass
    return datetime.now()


def _safe_path_component(value: str) -> str:
    name = Path(str(value or "document")).name.strip()
    sanitized = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in name
    ).strip("._")
    return sanitized or "document"


def _parse_stream_read(raw: Any) -> List[tuple[str, Dict[str, Any]]]:
    entries: List[tuple[str, Dict[str, Any]]] = []
    for _stream_name, stream_entries in raw or []:
        for message_id, fields in stream_entries or []:
            entries.append((_decode(message_id), dict(fields or {})))
    return entries


def _parse_autoclaim(raw: Any) -> List[tuple[str, Dict[str, Any]]]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        claimed_entries = raw[1]
    else:
        claimed_entries = []
    entries: List[tuple[str, Dict[str, Any]]] = []
    for message_id, fields in claimed_entries or []:
        entries.append((_decode(message_id), dict(fields or {})))
    return entries
