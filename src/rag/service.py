"""High-level school-document RAG ingestion and retrieval service."""

import asyncio
import base64
import binascii
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from src.memory import EmbeddingProvider, build_embedding_provider
from src.rag.chunker import DocumentChunk, build_chunker
from src.rag.parser import build_document_parser
from src.rag.vector_store import RagSearchResult, build_vector_store
from src.utils.schema import ResponseSource, ToolCall

logger = logging.getLogger(__name__)


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
    warnings: List[str] = field(default_factory=list)
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
            "warnings": list(self.warnings),
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
            warnings=list(payload.get("warnings") or []),
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
        self.redis_config = dict(self.config.get("redis", {}) or {})
        self.intent_gate_config = dict(self.config.get("intent_gate", {}) or {})
        self.job_ttl_seconds = int(self.config.get("job_ttl_seconds", 86400))
        self.parser = parser or build_document_parser(self.parser_config)
        self.chunker = build_chunker(self.chunking_config)
        self.embedding_provider = embedding_provider or build_embedding_provider(
            self.embedding_config
        )
        self.vector_store = vector_store or build_vector_store(self.config)
        self.jobs: Dict[str, IngestionJob] = {}
        self._initialized = False

    async def initialize(self):
        if not self.enabled:
            return
        await self.vector_store.initialize()
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
        content: bytes,
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
        content: bytes,
        filename: str,
        knowledge_base_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> IngestionJob:
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        max_size = int(self.parser_config.get("max_file_size_bytes", 100_000_000))
        if len(content) > max_size:
            raise ValueError("Document is larger than the configured RAG limit")

        resolved_document_id = document_id or str(uuid.uuid4())
        resolved_kb_id = knowledge_base_id or self._default_write_kb()
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            document_id=resolved_document_id,
            filename=filename,
            knowledge_base_id=resolved_kb_id,
        )
        await self._save_job(job)
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
        raise_on_error: bool = False,
    ) -> IngestionJob:
        job = await self.get_job(job_id)
        if job is None:
            job = IngestionJob(
                job_id=job_id,
                document_id=document_id,
                filename=filename,
                knowledge_base_id=knowledge_base_id,
            )
        try:
            job.status = "running"
            job.updated_at = datetime.now()
            await self._save_job(job)
            parsed = await asyncio.to_thread(
                self.parser.parse_bytes,
                content=content,
                filename=filename,
                document_id=document_id,
                metadata={
                    **dict(metadata or {}),
                    "knowledge_base_id": knowledge_base_id,
                },
            )
            chunks = self.chunker.chunk(parsed)
            if not chunks:
                raise ValueError("Document parsing produced no indexable text chunks")
            embeddings, warnings = await self._embed_chunks(chunks)
            job.warnings = warnings
            job.chunks_indexed = await self.vector_store.upsert_chunks(
                chunks,
                embeddings,
                knowledge_base_id=knowledge_base_id,
            )
            job.status = "completed_with_warnings" if warnings else "completed"
            job.updated_at = datetime.now()
            await self._save_job(job)
            return job
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
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
        effective_kbs = self._effective_knowledge_base_ids(knowledge_base_ids)
        return await self.vector_store.search(
            query,
            embedding,
            knowledge_base_ids=effective_kbs,
            limit=int(max_results or self.retrieval_config.get("top_k", 5)),
            candidate_count=int(
                candidate_count or self.retrieval_config.get("candidate_count", 30)
            ),
            keyword_weight=float(self.retrieval_config.get("keyword_weight", 0.35)),
            vector_weight=float(self.retrieval_config.get("vector_weight", 0.6)),
            recency_weight=float(self.retrieval_config.get("recency_weight", 0.05)),
            min_score=float(
                min_score
                if min_score is not None
                else self.retrieval_config.get("min_score", 0.0)
            ),
        )

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
            text = " ".join(chunk.text.split())
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
                    return IngestionJob.from_dict(json.loads(_decode(raw)))
                except Exception:
                    logger.warning("Failed to decode RAG ingestion job %s", job_id)
        return self.jobs.get(job_id)

    async def delete_document(
        self, document_id: str, knowledge_base_id: Optional[str] = None
    ) -> int:
        if not self.enabled:
            raise RuntimeError("RAG service is disabled")
        return await self.vector_store.delete_document(document_id, knowledge_base_id)

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

    async def _safe_embed(self, text: str) -> Optional[List[float]]:
        try:
            return await asyncio.wait_for(
                self.embedding_provider.embed(text),
                timeout=float(self.embedding_config.get("timeout", 30)),
            )
        except Exception as exc:
            logger.info("RAG embedding unavailable; using keyword-only search: %s", exc)
            return None

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
        self.jobs[job.job_id] = job
        client = self._job_client()
        if not client:
            return
        await client.setex(
            self._job_key(job.job_id),
            self.job_ttl_seconds,
            json.dumps(job.to_dict(), sort_keys=True),
        )

    def _job_client(self) -> Optional[Any]:
        client = getattr(self.vector_store, "client", None)
        if client and hasattr(client, "get") and hasattr(client, "setex"):
            return client
        return None

    def _job_key(self, job_id: str) -> str:
        key_prefix = str(self.redis_config.get("key_prefix") or "rag")
        return f"{key_prefix}:jobs:{job_id}"


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
