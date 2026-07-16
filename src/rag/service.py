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
from src.rag.vector_store import RagSearchResult, build_vector_store
from src.rag.visual import (
    FigureCropper,
    VisualProcessor,
    build_visual_processor,
    compose_figure_text,
)
from src.utils.schema import ResponseSource, ToolCall

logger = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = {
    "completed",
    "completed_with_warnings",
    "failed",
    "dead_lettered",
}

JOB_STATUSES = {
    "queued",
    "running",
    "retrying",
    *TERMINAL_JOB_STATUSES,
}


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
        self.job_ttl_seconds = int(self.config.get("job_ttl_seconds", 86400))
        self.queue_enabled = bool(self.queue_config.get("enabled", False))
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
        self.heartbeat_interval_seconds = float(
            self.queue_config.get("heartbeat_interval_seconds", 5)
        )
        self.heartbeat_ttl_seconds = int(
            self.queue_config.get("heartbeat_ttl_seconds", 15)
        )
        self.staging_dir = Path(
            self.storage_config.get("staging_dir") or "data/rag/uploads"
        )
        self.cleanup_completed_files = bool(
            self.storage_config.get("cleanup_completed_files", False)
        )
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
        self.jobs: Dict[str, IngestionJob] = {}
        self.batches: Dict[str, IngestionBatch] = {}
        self._initialized = False

    async def initialize(self):
        if not self.enabled:
            return
        await self.vector_store.initialize()
        if self.queue_enabled:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            await self._ensure_stream_group()
        if self._visual_enabled() and hasattr(self.figure_cropper, "assets_dir"):
            self.figure_cropper.assets_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    async def shutdown(self):
        if not self.enabled:
            return
        await self.vector_store.shutdown()
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
    ) -> IngestionJob:
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        max_size = int(self.parser_config.get("max_file_size_bytes", 100_000_000))
        if content is not None and len(content) > max_size:
            raise ValueError("Document is larger than the configured RAG limit")
        if content is None and not storage_ref:
            raise ValueError("content or storage_ref is required")

        resolved_document_id = document_id or str(uuid.uuid4())
        resolved_kb_id = knowledge_base_id or self._default_write_kb()
        resolved_storage_ref = storage_ref
        if self.queue_enabled and content is not None:
            resolved_storage_ref = self.stage_document_content(
                content,
                filename=filename,
                knowledge_base_id=resolved_kb_id,
                document_id=resolved_document_id,
            )
        elif self.queue_enabled and resolved_storage_ref:
            self._resolve_staged_path(resolved_storage_ref)
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            document_id=resolved_document_id,
            filename=filename,
            knowledge_base_id=resolved_kb_id,
            max_attempts=self.max_attempts,
            batch_id=batch_id,
            storage_ref=resolved_storage_ref,
            metadata=dict(metadata or {}),
        )
        await self._save_job(job)
        if self.queue_enabled:
            await self.enqueue_ingestion_job(job)
        return job

    async def process_ingestion_job(
        self,
        job_id: str,
        *,
        content: bytes,
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
            )
            job.status = "completed_with_warnings" if job.warnings else "completed"
            job.error = None
            job.last_error = None
            job.updated_at = datetime.now()
            await self._save_job(job)
            if job.storage_ref and self.cleanup_completed_files:
                self.cleanup_staged_file(job.storage_ref)
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
        embedding = await self._safe_embed(query)
        visual_embedding = await self._safe_visual_query_embed(query)
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
                content = decode_base64_document(str(item["content_base64"]))
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
        if job.status not in {"failed", "dead_lettered", "retrying"}:
            raise ValueError(
                "only failed, retrying, or dead_lettered jobs can be retried"
            )
        if not job.storage_ref:
            raise ValueError("job has no staged storage_ref to retry")
        job.status = "queued"
        job.error = None
        job.last_error = None
        job.updated_at = datetime.now()
        await self._save_job(job)
        await self.enqueue_ingestion_job(job, attempt=job.attempts + 1)
        return job

    async def enqueue_ingestion_job(
        self, job: IngestionJob, attempt: Optional[int] = None
    ) -> str:
        client = self._job_client()
        if not client:
            raise RuntimeError("Redis client is required for RAG ingestion queue")
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
        await self._save_job(job)
        return job.stream_message_id

    async def run_ingestion_workers(self):
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        if not self.queue_enabled:
            raise RuntimeError("RAG ingestion queue is disabled")
        await self._ensure_stream_group()
        workers = [
            asyncio.create_task(
                self._worker_loop(self._consumer_name(index)),
                name=f"rag_ingestion_worker_{index}",
            )
            for index in range(self.worker_count)
        ]
        worker_id = f"{socket.gethostname()}-{os.getpid()}"
        heartbeat = asyncio.create_task(
            self._worker_heartbeat_loop(worker_id),
            name="rag_ingestion_worker_heartbeat",
        )
        try:
            await asyncio.gather(*workers)
        finally:
            heartbeat.cancel()
            for worker in workers:
                worker.cancel()

    async def _worker_heartbeat_loop(self, worker_id: str) -> None:
        client = self._job_client()
        if not client:
            raise RuntimeError("Redis client is required for RAG worker heartbeat")
        key = self._prefixed_key(f"worker:heartbeat:{worker_id}")
        while True:
            await client.set(
                key,
                datetime.now().isoformat(),
                ex=self.heartbeat_ttl_seconds,
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
        if job.status in TERMINAL_JOB_STATUSES:
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
            await self.ack_stream_message(message_id)
            return job
        except Exception as exc:
            latest_job = await self.get_job(job.job_id)
            await self._handle_stream_failure(
                latest_job or job,
                message_id=message_id,
                error=str(exc),
                attempt=attempt,
            )
            return await self.get_job(job.job_id)

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
        embeddings = []
        failures = 0
        for chunk in chunks:
            if chunk.metadata.get("is_figure") and not chunk.text.strip():
                embeddings.append(None)
                continue
            embedding = await self._safe_embed(chunk.text)
            if embedding is None:
                failures += 1
            embeddings.append(embedding)
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
            return
        await client.setex(
            self._job_key(job.job_id),
            self.job_ttl_seconds,
            json.dumps(job.to_dict(), sort_keys=True),
        )
        await self._update_batch_for_status_change(
            job,
            previous_status=previous_status,
            new_status=job.status,
        )
        setattr(job, "_persisted_status", job.status)

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
    ) -> str:
        safe_filename = _safe_path_component(filename or "document.pdf")
        safe_kb = _safe_path_component(knowledge_base_id)
        safe_document = _safe_path_component(document_id)
        content_hash = hashlib.sha256(content).hexdigest()
        target_dir = self.staging_dir / safe_kb / safe_document
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{content_hash}-{safe_filename}"
        target.write_bytes(content)
        return str(target)

    def load_staged_content(self, storage_ref: Optional[str]) -> bytes:
        if not storage_ref:
            raise ValueError("storage_ref is required for queued ingestion")
        path = self._resolve_staged_path(storage_ref)
        max_size = int(self.parser_config.get("max_file_size_bytes", 100_000_000))
        if path.stat().st_size > max_size:
            raise ValueError("Document is larger than the configured RAG limit")
        return path.read_bytes()

    def cleanup_staged_file(self, storage_ref: str) -> None:
        try:
            self._resolve_staged_path(storage_ref).unlink(missing_ok=True)
        except Exception:
            logger.debug(
                "Failed to clean staged RAG file %s", storage_ref, exc_info=True
            )

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


def decode_base64_document(payload: str) -> bytes:
    try:
        return base64.b64decode(payload.encode("utf-8"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("content_base64 must be valid base64") from exc


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
