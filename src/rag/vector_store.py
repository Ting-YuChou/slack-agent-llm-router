"""Vector stores for school-document RAG chunks."""

import array
import asyncio
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.memory import tokenize
from src.rag.chunker import DocumentChunk

logger = logging.getLogger(__name__)

SIDECAR_METADATA_KEYS = {
    "figure_sidecar",
    "table_sidecar",
    "table_row_sidecar",
    "visual_embedding",
}

ATOMIC_GENERATION_COMMIT_LUA = """
local staging_set = KEYS[1]
local document_set = KEYS[2]
local active_generation = KEYS[3]
local document_tables = KEYS[4]
local document_figures = KEYS[5]
local document_visuals = KEYS[6]
local expected = tonumber(ARGV[1])
local knowledge_base_id = ARGV[2]
local generation = ARGV[3]
local chunk_prefix = ARGV[4]
local resource_count = tonumber(ARGV[5])
local table_prefix = ARGV[6]
local table_rows_prefix = ARGV[7]
local table_row_prefix = ARGV[8]
local figure_prefix = ARGV[9]
local visual_prefix = ARGV[10]
local chunk_args_start = 11
local chunk_keys_start = 7
local resource_keys_start = chunk_keys_start + (expected * 2)

if redis.call('SCARD', staging_set) ~= expected then
    return redis.error_reply('incomplete RAG staging generation')
end

for index = 1, expected do
    local key_offset = chunk_keys_start + ((index - 1) * 2)
    if redis.call('EXISTS', KEYS[key_offset]) ~= 1 then
        return redis.error_reply('missing RAG staging chunk')
    end
end
for index = 1, resource_count do
    local key_offset = resource_keys_start + ((index - 1) * 2)
    if redis.call('EXISTS', KEYS[key_offset]) ~= 1 then
        return redis.error_reply('missing RAG staging sidecar')
    end
end

local active = redis.call('GET', active_generation)
if active and active > generation then
    for index = 1, expected do
        local key_offset = chunk_keys_start + ((index - 1) * 2)
        redis.call('DEL', KEYS[key_offset])
    end
    for index = 1, resource_count do
        local key_offset = resource_keys_start + ((index - 1) * 2)
        redis.call('DEL', KEYS[key_offset])
    end
    redis.call('DEL', staging_set)
    return -expected
end

local previous = redis.call('SMEMBERS', document_set)
for _, chunk_id in ipairs(previous) do
    redis.call('DEL', chunk_prefix .. chunk_id)
end
redis.call('DEL', document_set)

for index = 1, expected do
    local key_offset = chunk_keys_start + ((index - 1) * 2)
    local staging_key = KEYS[key_offset]
    local live_key = KEYS[key_offset + 1]
    local chunk_id = ARGV[chunk_args_start + index - 1]
    redis.call('RENAME', staging_key, live_key)
    redis.call('PERSIST', live_key)
    redis.call('SADD', document_set, chunk_id)
end

local previous_tables = redis.call('SMEMBERS', document_tables)
for _, table_id in ipairs(previous_tables) do
    local rows_key = table_rows_prefix .. table_id
    local previous_rows = redis.call('SMEMBERS', rows_key)
    for _, row_id in ipairs(previous_rows) do
        redis.call('DEL', table_row_prefix .. table_id .. ':' .. row_id)
    end
    redis.call('DEL', rows_key)
    redis.call('DEL', table_prefix .. table_id)
end
local previous_figures = redis.call('SMEMBERS', document_figures)
for _, figure_id in ipairs(previous_figures) do
    redis.call('DEL', figure_prefix .. figure_id)
end
local previous_visuals = redis.call('SMEMBERS', document_visuals)
for _, chunk_id in ipairs(previous_visuals) do
    redis.call('DEL', visual_prefix .. chunk_id)
end
redis.call('DEL', document_tables, document_figures, document_visuals)

for index = 1, resource_count do
    local key_offset = resource_keys_start + ((index - 1) * 2)
    redis.call('RENAME', KEYS[key_offset], KEYS[key_offset + 1])
    redis.call('PERSIST', KEYS[key_offset + 1])
end
redis.call('SET', active_generation, generation)
redis.call('DEL', staging_set)
return expected
"""


@dataclass
class StoredRagChunk:
    """A chunk plus index metadata."""

    chunk: DocumentChunk
    knowledge_base_id: str
    embedding: Optional[List[float]] = None
    visual_embedding: Optional[List[float]] = None
    index_version: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RagSearchResult:
    """One RAG retrieval hit."""

    chunk: DocumentChunk
    score: float
    match_source: str
    knowledge_base_id: str
    index_version: Optional[str] = None
    updated_at: Optional[datetime] = None


class InMemoryRagVectorStore:
    """Deterministic vector store for local tests and lightweight development."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.retrieval_config = dict(self.config.get("retrieval", {}) or {})
        self.items: Dict[str, StoredRagChunk] = {}

    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Optional[Sequence[float]]],
        *,
        knowledge_base_id: str,
        visual_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
        index_version: Optional[str] = None,
    ) -> int:
        if not chunks:
            return 0
        now = datetime.now()
        document_id = chunks[0].document_id
        index_version = index_version or _new_index_version()
        existing_items = [
            item
            for item in self.items.values()
            if item.chunk.document_id == document_id
            and item.knowledge_base_id == knowledge_base_id
        ]
        if existing_items and all(
            item.index_version == index_version for item in existing_items
        ):
            return len(existing_items)
        await self.delete_document(document_id, knowledge_base_id)
        visual_embeddings = visual_embeddings or [None] * len(chunks)
        for chunk, embedding, visual_embedding in zip(
            chunks, embeddings, visual_embeddings
        ):
            storage_id = self._storage_id(knowledge_base_id, chunk.chunk_id)
            existing = self.items.get(storage_id)
            self.items[storage_id] = StoredRagChunk(
                chunk=chunk,
                knowledge_base_id=knowledge_base_id,
                embedding=list(embedding) if embedding else None,
                visual_embedding=list(visual_embedding) if visual_embedding else None,
                index_version=index_version,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
        return len(chunks)

    async def delete_document(
        self, document_id: str, knowledge_base_id: Optional[str] = None
    ) -> int:
        delete_ids = [
            storage_id
            for storage_id, item in self.items.items()
            if item.chunk.document_id == document_id
            and (
                knowledge_base_id is None or item.knowledge_base_id == knowledge_base_id
            )
        ]
        for storage_id in delete_ids:
            self.items.pop(storage_id, None)
        return len(delete_ids)

    async def search(
        self,
        query: str,
        embedding: Optional[Sequence[float]],
        *,
        knowledge_base_ids: Sequence[str],
        limit: int,
        candidate_count: int,
        keyword_weight: float,
        vector_weight: float,
        recency_weight: float,
        min_score: float,
        visual_embedding: Optional[Sequence[float]] = None,
        visual_weight: float = 0.0,
        visual_min_score: float = 0.0,
        visual_candidate_count: Optional[int] = None,
    ) -> List[RagSearchResult]:
        tokens = tokenize(query)
        allowed_kbs = set(knowledge_base_ids or [])
        searchable_items = [
            (storage_id, item)
            for storage_id, item in self.items.items()
            if not allowed_kbs or item.knowledge_base_id in allowed_kbs
        ]
        keyword_scores = _bm25_scores(
            tokens,
            [(storage_id, item.chunk.text) for storage_id, item in searchable_items],
        )
        results: Dict[str, Tuple[StoredRagChunk, float, str]] = {}
        for storage_id, item in searchable_items:
            keyword_score = keyword_scores.get(storage_id, 0.0)
            vector_score = (
                _cosine_similarity(embedding, item.embedding)
                if embedding and item.embedding
                else 0.0
            )
            visual_score = (
                _cosine_similarity(visual_embedding, item.visual_embedding)
                if visual_embedding and item.visual_embedding
                else 0.0
            )
            score = (
                keyword_weight * keyword_score
                + vector_weight * vector_score
                + visual_weight * visual_score
                + recency_weight * _recency_score(item.updated_at)
            )
            effective_min_score = (
                min(min_score, visual_min_score)
                if visual_score > 0 and visual_min_score > 0
                else min_score
            )
            if score <= 0 or score < effective_min_score:
                continue
            weighted_keyword = keyword_weight * keyword_score
            weighted_vector = vector_weight * vector_score
            weighted_visual = visual_weight * visual_score
            source = "hybrid"
            if weighted_visual > 0 and weighted_keyword <= 0 and weighted_vector <= 0:
                source = "visual"
            elif weighted_keyword > 0 and weighted_vector <= 0 and weighted_visual <= 0:
                source = "keyword"
            elif weighted_vector > 0 and weighted_keyword <= 0 and weighted_visual <= 0:
                source = "vector"
            elif weighted_visual > 0:
                source = "hybrid_visual"
            results[storage_id] = (item, score, source)

        ranked = sorted(results.values(), key=lambda value: -value[1])[
            : max(candidate_count, limit)
        ]
        return [
            RagSearchResult(
                chunk=item.chunk,
                score=score,
                match_source=source,
                knowledge_base_id=item.knowledge_base_id,
                index_version=item.index_version,
                updated_at=item.updated_at,
            )
            for item, score, source in ranked[:limit]
        ]

    def _storage_id(self, knowledge_base_id: str, chunk_id: str) -> str:
        return f"{knowledge_base_id}:{chunk_id}"


class RedisStackRagVectorStore:
    """Redis Stack-backed vector store using RediSearch HNSW."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.redis_config = dict(self.config.get("redis", {}) or {})
        self.indexing_config = dict(self.config.get("indexing", {}) or {})
        self.embedding_config = dict(self.config.get("embedding", {}) or {})
        self.retrieval_config = dict(self.config.get("retrieval", {}) or {})
        self.visual_config = dict(self.config.get("visual", {}) or {})
        self.visual_embedding_config = dict(
            self.visual_config.get("embedding", {}) or {}
        )
        self.key_prefix = self.redis_config.get("key_prefix", "rag")
        self.control_plane_version = str(
            self.indexing_config.get("control_plane_version", "v2")
        )
        index_suffix = (
            f"{self.control_plane_version}:" if self.control_plane_version else ""
        )
        self.index_name = f"{self.key_prefix}:idx:{index_suffix}chunks"
        self.visual_index_name = f"{self.key_prefix}:idx:{index_suffix}visual_chunks"
        self.dimensions = int(self.embedding_config.get("dimensions", 1024))
        self.visual_dimensions = int(
            self.visual_embedding_config.get("dimensions", 1024)
        )
        self.client = None
        self.pipeline_batch_size = max(
            1, int(self.redis_config.get("pipeline_batch_size", 64))
        )
        self.atomic_commit_max_chunks = max(
            1, int(self.indexing_config.get("atomic_commit_max_chunks", 2000))
        )
        self.staging_ttl_seconds = max(
            1, int(self.indexing_config.get("staging_ttl_seconds", 86400))
        )

    async def initialize(self):
        import redis.asyncio as redis_asyncio

        redis_url = self.redis_config.get("url")
        password = None
        password_env = self.redis_config.get("password_env")
        if password_env:
            password = os.getenv(str(password_env))
        if redis_url and hasattr(redis_asyncio.Redis, "from_url"):
            self.client = redis_asyncio.Redis.from_url(
                redis_url, decode_responses=False
            )
        else:
            self.client = redis_asyncio.Redis(
                host=self.redis_config.get("host", "localhost"),
                port=int(self.redis_config.get("port", 6380)),
                db=int(self.redis_config.get("db", 0)),
                password=password,
                decode_responses=False,
            )
        await self.client.ping()
        await self._ensure_index()
        if self._visual_enabled():
            await self._ensure_visual_index()

    async def shutdown(self):
        if self.client and hasattr(self.client, "aclose"):
            await self.client.aclose()
        elif self.client and hasattr(self.client, "close"):
            await self.client.close()

    async def _ensure_index(self):
        try:
            await self.client.execute_command("FT.INFO", self.index_name)
            return
        except Exception:
            pass

        try:
            await self.client.execute_command(
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._chunk_prefix(),
                "SCHEMA",
                "knowledge_base_id",
                "TAG",
                "document_id",
                "TAG",
                "chunk_id",
                "TAG",
                "index_version",
                "TAG",
                "text",
                "TEXT",
                "keywords",
                "TAG",
                "block_types",
                "TAG",
                "page_start",
                "NUMERIC",
                "SORTABLE",
                "page_end",
                "NUMERIC",
                "created_at",
                "NUMERIC",
                "updated_at",
                "NUMERIC",
                "SORTABLE",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.dimensions),
                "DISTANCE_METRIC",
                "COSINE",
            )
        except Exception as exc:
            raise RuntimeError(
                "RAG Redis backend requires Redis Stack RediSearch/vector support"
            ) from exc

    async def _ensure_visual_index(self):
        try:
            await self.client.execute_command("FT.INFO", self.visual_index_name)
            return
        except Exception:
            pass

        try:
            await self.client.execute_command(
                "FT.CREATE",
                self.visual_index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._visual_chunk_prefix(),
                "SCHEMA",
                "knowledge_base_id",
                "TAG",
                "document_id",
                "TAG",
                "chunk_id",
                "TAG",
                "index_version",
                "TAG",
                "text",
                "TEXT",
                "block_types",
                "TAG",
                "page_start",
                "NUMERIC",
                "SORTABLE",
                "page_end",
                "NUMERIC",
                "updated_at",
                "NUMERIC",
                "SORTABLE",
                "visual_embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self.visual_dimensions),
                "DISTANCE_METRIC",
                "COSINE",
            )
        except Exception as exc:
            raise RuntimeError(
                "RAG visual Redis backend requires Redis Stack vector support"
            ) from exc

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Optional[Sequence[float]]],
        *,
        knowledge_base_id: str,
        visual_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
        index_version: Optional[str] = None,
    ) -> int:
        if not self.client or not chunks:
            return 0
        if len(chunks) > self.atomic_commit_max_chunks:
            raise ValueError(
                "Document exceeds the configured RAG atomic commit limit "
                f"({self.atomic_commit_max_chunks} chunks)"
            )
        now_ts = time.time()
        document_id = chunks[0].document_id
        requested_index_version = index_version
        index_version = index_version or _new_index_version()
        if requested_index_version is not None:
            active_generation = await self.client.get(
                self._active_generation_key(document_id, knowledge_base_id)
            )
            if (
                active_generation is not None
                and _decode(active_generation) == index_version
            ):
                return len(chunks)
        visual_embeddings = visual_embeddings or [None] * len(chunks)
        staging_set = self._staging_generation_key(
            document_id, knowledge_base_id, index_version
        )
        staged_keys: List[Tuple[str, str, str]] = []
        for batch_start in range(0, len(chunks), self.pipeline_batch_size):
            pipeline = self.client.pipeline(transaction=False)
            batch_end = batch_start + self.pipeline_batch_size
            for chunk, embedding in zip(
                chunks[batch_start:batch_end], embeddings[batch_start:batch_end]
            ):
                mapping = self._chunk_mapping(
                    chunk,
                    embedding,
                    knowledge_base_id=knowledge_base_id,
                    index_version=index_version,
                    updated_at=now_ts,
                )
                staging_key = self._staging_chunk_key(
                    chunk.chunk_id,
                    knowledge_base_id,
                    document_id,
                    index_version,
                )
                live_key = self._chunk_key(
                    chunk.chunk_id,
                    knowledge_base_id,
                    document_id=document_id,
                )
                pipeline.hset(staging_key, mapping=mapping)
                pipeline.expire(staging_key, self.staging_ttl_seconds)
                pipeline.sadd(staging_set, chunk.chunk_id)
                pipeline.expire(staging_set, self.staging_ttl_seconds)
                staged_keys.append((staging_key, live_key, chunk.chunk_id))
            if batch_start == 0:
                pipeline.sadd(self._document_kbs_key(document_id), knowledge_base_id)
            await pipeline.execute()

        sidecar_resources = self._generation_sidecar_resources(
            chunks,
            visual_embeddings,
            knowledge_base_id=knowledge_base_id,
            index_version=index_version,
            updated_at=now_ts,
        )
        staged_resources: List[Tuple[str, str]] = []
        resource_items = list(sidecar_resources.items())
        for batch_start in range(0, len(resource_items), self.pipeline_batch_size):
            pipeline = self.client.pipeline(transaction=False)
            batch_end = batch_start + self.pipeline_batch_size
            for live_key, (resource_type, value) in resource_items[
                batch_start:batch_end
            ]:
                staging_key = self._staging_resource_key(
                    document_id,
                    knowledge_base_id,
                    index_version,
                    live_key,
                )
                if resource_type == "hash":
                    pipeline.hset(staging_key, mapping=value)
                else:
                    pipeline.sadd(staging_key, *sorted(value))
                pipeline.expire(staging_key, self.staging_ttl_seconds)
                staged_resources.append((staging_key, live_key))
            await pipeline.execute()

        arguments: List[Any] = [
            len(chunks),
            knowledge_base_id,
            index_version,
            self._document_chunk_prefix(document_id, knowledge_base_id),
            len(staged_resources),
            self._table_prefix(document_id, knowledge_base_id),
            self._table_rows_prefix(document_id, knowledge_base_id),
            self._table_row_prefix(document_id, knowledge_base_id),
            self._figure_prefix(document_id, knowledge_base_id),
            self._visual_chunk_document_prefix(document_id, knowledge_base_id),
        ]
        arguments.extend(chunk_id for _, _, chunk_id in staged_keys)
        keys: List[str] = [
            staging_set,
            self._document_key(document_id, knowledge_base_id),
            self._active_generation_key(document_id, knowledge_base_id),
            self._document_tables_key(document_id, knowledge_base_id),
            self._document_figures_key(document_id, knowledge_base_id),
            self._document_visual_key(document_id, knowledge_base_id),
        ]
        for staging_key, live_key, _ in staged_keys:
            keys.extend([staging_key, live_key])
        for staging_key, live_key in staged_resources:
            keys.extend([staging_key, live_key])
        committed = await self.client.eval(
            ATOMIC_GENERATION_COMMIT_LUA,
            len(keys),
            *keys,
            *arguments,
        )
        if int(committed) == -len(chunks):
            return len(chunks)
        if int(committed) != len(chunks):
            raise RuntimeError("RAG generation cutover returned an unexpected count")
        return len(chunks)

    def _generation_sidecar_resources(
        self,
        chunks: Sequence[DocumentChunk],
        visual_embeddings: Sequence[Optional[Sequence[float]]],
        *,
        knowledge_base_id: str,
        index_version: str,
        updated_at: float,
    ) -> Dict[str, Tuple[str, Any]]:
        resources: Dict[str, Tuple[str, Any]] = {}

        def add_set(key: str, member: str) -> None:
            resource = resources.setdefault(key, ("set", set()))
            resource[1].add(member)

        for chunk, visual_embedding in zip(chunks, visual_embeddings):
            metadata = dict(chunk.metadata or {})
            table_id = str(metadata.get("table_id") or "")
            table_sidecar = metadata.get("table_sidecar")
            row_sidecar = metadata.get("table_row_sidecar")
            row_id = str(metadata.get("table_row_id") or "")
            if table_id and isinstance(table_sidecar, dict):
                resources[
                    self._table_key(knowledge_base_id, chunk.document_id, table_id)
                ] = (
                    "hash",
                    {
                        "knowledge_base_id": knowledge_base_id,
                        "document_id": chunk.document_id,
                        "table_id": table_id,
                        "index_version": index_version,
                        "updated_at": updated_at,
                        "table_json": json.dumps(table_sidecar, sort_keys=True),
                    },
                )
                add_set(
                    self._document_tables_key(chunk.document_id, knowledge_base_id),
                    table_id,
                )
            if table_id and row_id and isinstance(row_sidecar, dict):
                add_set(
                    self._document_tables_key(chunk.document_id, knowledge_base_id),
                    table_id,
                )
                resources[
                    self._table_row_key(
                        knowledge_base_id, chunk.document_id, table_id, row_id
                    )
                ] = (
                    "hash",
                    {
                        "knowledge_base_id": knowledge_base_id,
                        "document_id": chunk.document_id,
                        "table_id": table_id,
                        "row_id": row_id,
                        "index_version": index_version,
                        "updated_at": updated_at,
                        "row_json": json.dumps(row_sidecar, sort_keys=True),
                    },
                )
                add_set(
                    self._table_rows_key(
                        knowledge_base_id, chunk.document_id, table_id
                    ),
                    row_id,
                )

            figure_id = str(metadata.get("figure_id") or "")
            figure_sidecar = metadata.get("figure_sidecar")
            if figure_id and isinstance(figure_sidecar, dict):
                resources[
                    self._figure_key(knowledge_base_id, chunk.document_id, figure_id)
                ] = (
                    "hash",
                    {
                        "knowledge_base_id": knowledge_base_id,
                        "document_id": chunk.document_id,
                        "figure_id": figure_id,
                        "index_version": index_version,
                        "updated_at": updated_at,
                        "figure_json": json.dumps(
                            _figure_sidecar_for_storage(figure_sidecar), sort_keys=True
                        ),
                    },
                )
                add_set(
                    self._document_figures_key(chunk.document_id, knowledge_base_id),
                    figure_id,
                )

            if visual_embedding and self._visual_enabled():
                if len(visual_embedding) != self.visual_dimensions:
                    raise ValueError(
                        "Visual embedding dimensions do not match RAG visual index"
                    )
                resources[
                    self._visual_chunk_key(
                        chunk.chunk_id,
                        knowledge_base_id,
                        document_id=chunk.document_id,
                    )
                ] = (
                    "hash",
                    {
                        "knowledge_base_id": knowledge_base_id,
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "index_version": index_version,
                        "text": chunk.text,
                        "block_types": ",".join(chunk.block_types),
                        "metadata": json.dumps(
                            _metadata_for_chunk_storage(chunk.metadata), sort_keys=True
                        ),
                        "section_path": json.dumps(chunk.section_path),
                        "block_ids": json.dumps(chunk.block_ids),
                        "bboxes": json.dumps(chunk.bboxes),
                        "page_start": int(chunk.page_start),
                        "page_end": int(chunk.page_end),
                        "created_at": updated_at,
                        "updated_at": updated_at,
                        "visual_embedding": _vector_to_bytes(visual_embedding),
                    },
                )
                add_set(
                    self._document_visual_key(chunk.document_id, knowledge_base_id),
                    chunk.chunk_id,
                )
        return resources

    def _chunk_mapping(
        self,
        chunk: DocumentChunk,
        embedding: Optional[Sequence[float]],
        *,
        knowledge_base_id: str,
        index_version: str,
        updated_at: float,
    ) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {
            "knowledge_base_id": knowledge_base_id,
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "index_version": index_version,
            "text": chunk.text,
            "keywords": ",".join(tokenize(chunk.text)),
            "block_types": ",".join(chunk.block_types),
            "metadata": json.dumps(
                _metadata_for_chunk_storage(chunk.metadata), sort_keys=True
            ),
            "section_path": json.dumps(chunk.section_path),
            "block_ids": json.dumps(chunk.block_ids),
            "bboxes": json.dumps(chunk.bboxes),
            "page_start": int(chunk.page_start),
            "page_end": int(chunk.page_end),
            "created_at": updated_at,
            "updated_at": updated_at,
        }
        if embedding:
            if len(embedding) != self.dimensions:
                raise ValueError("Embedding dimensions do not match RAG index")
            mapping["embedding"] = _vector_to_bytes(embedding)
        return mapping

    async def delete_document(
        self, document_id: str, knowledge_base_id: Optional[str] = None
    ) -> int:
        if not self.client:
            return 0
        if knowledge_base_id:
            return await self._delete_document_for_kb(document_id, knowledge_base_id)

        kb_ids = await self.client.smembers(self._document_kbs_key(document_id))
        count = 0
        for raw_kb_id in kb_ids:
            count += await self._delete_document_for_kb(document_id, _decode(raw_kb_id))

        count += await self._delete_legacy_document_chunks(document_id, None)
        await self.client.delete(self._document_kbs_key(document_id))
        return count

    async def _delete_document_for_kb(
        self, document_id: str, knowledge_base_id: str
    ) -> int:
        chunk_ids = await self.client.smembers(
            self._document_key(document_id, knowledge_base_id)
        )
        count = 0
        for raw_chunk_id in chunk_ids:
            chunk_id = _decode(raw_chunk_id)
            deleted = await self.client.delete(
                self._chunk_key(chunk_id, knowledge_base_id, document_id=document_id)
            )
            count += int(bool(deleted))
        await self.client.delete(self._document_key(document_id, knowledge_base_id))
        await self._delete_table_sidecars(document_id, knowledge_base_id)
        await self._delete_figure_sidecars(document_id, knowledge_base_id)
        count += await self._delete_visual_chunks(document_id, knowledge_base_id)

        remaining_kbs = await self.client.smembers(self._document_kbs_key(document_id))
        await self.client.srem(self._document_kbs_key(document_id), knowledge_base_id)
        if not [kb for kb in remaining_kbs if _decode(kb) != knowledge_base_id]:
            await self.client.delete(self._document_kbs_key(document_id))

        count += await self._delete_legacy_document_chunks(
            document_id, knowledge_base_id
        )
        return count

    async def _delete_legacy_document_chunks(
        self, document_id: str, knowledge_base_id: Optional[str]
    ) -> int:
        chunk_ids = await self.client.smembers(self._legacy_document_key(document_id))
        count = 0
        for raw_chunk_id in chunk_ids:
            chunk_id = _decode(raw_chunk_id)
            payload = await self.client.hgetall(self._legacy_chunk_key(chunk_id))
            if knowledge_base_id and payload:
                item = self._deserialize(payload)
                if item.knowledge_base_id != knowledge_base_id:
                    continue
            deleted = await self.client.delete(self._legacy_chunk_key(chunk_id))
            count += int(bool(deleted))
        if not knowledge_base_id:
            await self.client.delete(self._legacy_document_key(document_id))
        return count

    async def search(
        self,
        query: str,
        embedding: Optional[Sequence[float]],
        *,
        knowledge_base_ids: Sequence[str],
        limit: int,
        candidate_count: int,
        keyword_weight: float,
        vector_weight: float,
        recency_weight: float,
        min_score: float,
        visual_embedding: Optional[Sequence[float]] = None,
        visual_weight: float = 0.0,
        visual_min_score: float = 0.0,
        visual_candidate_count: Optional[int] = None,
    ) -> List[RagSearchResult]:
        semaphore = asyncio.Semaphore(
            max(1, int(self.retrieval_config.get("max_concurrent_branches", 3)))
        )

        async def run_branch(name: str, factory: Any) -> List[RagSearchResult]:
            async with semaphore:
                try:
                    return await factory()
                except Exception as exc:
                    logger.warning("Redis RAG %s retrieval failed: %s", name, exc)
                    return []

        tasks = [
            asyncio.create_task(
                run_branch(
                    "keyword",
                    lambda: self._keyword_search(
                        query, knowledge_base_ids, candidate_count
                    ),
                )
            ),
            asyncio.create_task(
                run_branch(
                    "vector",
                    lambda: self._vector_search(
                        embedding, knowledge_base_ids, candidate_count
                    ),
                )
            ),
            asyncio.create_task(
                run_branch(
                    "visual",
                    lambda: self._visual_search(
                        visual_embedding,
                        knowledge_base_ids,
                        int(visual_candidate_count or candidate_count),
                    ),
                )
            ),
        ]
        done: set[asyncio.Task] = set()
        pending: set[asyncio.Task] = set(tasks)
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=float(self.retrieval_config.get("deadline_seconds", 30)),
            )
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
        if pending:
            logger.warning(
                "Redis RAG retrieval deadline expired for %s branches", len(pending)
            )
        branch_results: List[List[RagSearchResult]] = []
        for task in tasks:
            if task in done and not task.cancelled():
                branch_results.append(task.result())
            else:
                branch_results.append([])
        keyword_results, vector_results, visual_results = branch_results
        merged = _merge_results(
            keyword_results,
            vector_results,
            visual_results,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            visual_weight=visual_weight,
            recency_weight=recency_weight,
            min_score=min_score,
            visual_min_score=visual_min_score,
        )
        return merged[:limit]

    async def _keyword_search(
        self, query: str, knowledge_base_ids: Sequence[str], limit: int
    ) -> List[RagSearchResult]:
        tokens = tokenize(query)
        if not self.client or not tokens:
            return []
        filter_query = self._filter_query(knowledge_base_ids)
        text_query = "|".join(_escape_text_token(token) for token in tokens)
        redis_query = (
            f"{filter_query} @text:({text_query})"
            if filter_query != "*"
            else f"@text:({text_query})"
        )
        results = await self._keyword_text_search(redis_query, limit)
        if results:
            return results

        token_query = "|".join(_escape_tag(token) for token in tokens)
        fallback_query = (
            f"{filter_query} @keywords:{{{token_query}}}"
            if filter_query != "*"
            else f"@keywords:{{{token_query}}}"
        )
        return await self._search(fallback_query, limit, "keyword", tokens=tokens)

    async def _keyword_text_search(
        self, query: str, limit: int
    ) -> List[RagSearchResult]:
        for scorer in self._keyword_scorer_candidates():
            try:
                return await self._search(
                    query,
                    limit,
                    "keyword",
                    with_scores=True,
                    scorer=scorer,
                    normalize_scores=self._keyword_normalization() == "max",
                    raise_on_error=scorer is not None,
                )
            except Exception as exc:
                logger.debug("Redis RAG keyword scorer %s failed: %s", scorer, exc)
                continue
        return await self._search(
            query,
            limit,
            "keyword",
            with_scores=True,
            normalize_scores=self._keyword_normalization() == "max",
        )

    async def _vector_search(
        self,
        embedding: Optional[Sequence[float]],
        knowledge_base_ids: Sequence[str],
        limit: int,
    ) -> List[RagSearchResult]:
        if not self.client or not embedding:
            return []
        if len(embedding) != self.dimensions:
            logger.warning("Skipping RAG vector search due to dimension mismatch")
            return []
        filter_query = self._filter_query(knowledge_base_ids)
        base_query = "*" if filter_query == "*" else f"({filter_query})"
        redis_query = f"{base_query}=>[KNN {limit} @embedding $vec AS distance]"
        try:
            raw_results = await self.client.execute_command(
                "FT.SEARCH",
                self.index_name,
                redis_query,
                "PARAMS",
                "2",
                "vec",
                _vector_to_bytes(embedding),
                "SORTBY",
                "distance",
                "ASC",
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "15",
                "knowledge_base_id",
                "document_id",
                "chunk_id",
                "text",
                "metadata",
                "section_path",
                "block_ids",
                "block_types",
                "bboxes",
                "page_start",
                "page_end",
                "created_at",
                "updated_at",
                "index_version",
                "distance",
                "DIALECT",
                "2",
            )
        except Exception as exc:
            logger.warning("Redis RAG vector search failed: %s", exc)
            return []
        return self._parse_results(raw_results, "vector")

    async def _visual_search(
        self,
        visual_embedding: Optional[Sequence[float]],
        knowledge_base_ids: Sequence[str],
        limit: int,
    ) -> List[RagSearchResult]:
        if not self.client or not visual_embedding or not self._visual_enabled():
            return []
        if len(visual_embedding) != self.visual_dimensions:
            logger.warning("Skipping RAG visual search due to dimension mismatch")
            return []
        filter_query = self._filter_query(knowledge_base_ids)
        base_query = "*" if filter_query == "*" else f"({filter_query})"
        redis_query = f"{base_query}=>[KNN {limit} @visual_embedding $vec AS distance]"
        try:
            raw_results = await self.client.execute_command(
                "FT.SEARCH",
                self.visual_index_name,
                redis_query,
                "PARAMS",
                "2",
                "vec",
                _vector_to_bytes(visual_embedding),
                "SORTBY",
                "distance",
                "ASC",
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "15",
                "knowledge_base_id",
                "document_id",
                "chunk_id",
                "text",
                "metadata",
                "section_path",
                "block_ids",
                "block_types",
                "bboxes",
                "page_start",
                "page_end",
                "created_at",
                "updated_at",
                "index_version",
                "distance",
                "DIALECT",
                "2",
            )
        except Exception as exc:
            logger.warning("Redis RAG visual search failed: %s", exc)
            return []
        return self._parse_results(raw_results, "visual")

    async def _search(
        self,
        query: str,
        limit: int,
        source: str,
        tokens: Optional[Sequence[str]] = None,
        with_scores: bool = False,
        scorer: Optional[str] = None,
        normalize_scores: bool = False,
        raise_on_error: bool = False,
    ) -> List[RagSearchResult]:
        command: List[Any] = ["FT.SEARCH", self.index_name, query]
        if with_scores:
            command.append("WITHSCORES")
        if scorer:
            command.extend(["SCORER", scorer])
        command.extend(
            [
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "14",
                "knowledge_base_id",
                "document_id",
                "chunk_id",
                "index_version",
                "text",
                "metadata",
                "section_path",
                "block_ids",
                "block_types",
                "bboxes",
                "page_start",
                "page_end",
                "created_at",
                "updated_at",
            ]
        )
        try:
            raw_results = await self.client.execute_command(*command)
        except Exception as exc:
            if raise_on_error:
                raise
            logger.warning("Redis RAG keyword search failed: %s", exc)
            return []
        return self._parse_results(
            raw_results,
            source,
            tokens=tokens,
            with_scores=with_scores,
            normalize_scores=normalize_scores,
        )

    def _parse_results(
        self,
        raw_results: Any,
        source: str,
        tokens: Optional[Sequence[str]] = None,
        with_scores: bool = False,
        normalize_scores: bool = False,
    ) -> List[RagSearchResult]:
        if not raw_results or len(raw_results) < 2:
            return []
        parsed = []
        entries = raw_results[1:]
        index = 0
        while index < len(entries):
            index += 1
            raw_score = None
            if with_scores:
                raw_score = entries[index]
                index += 1
            fields = entries[index]
            index += 1
            payload = {}
            for field_index in range(0, len(fields), 2):
                payload[_decode(fields[field_index])] = fields[field_index + 1]
            item = self._deserialize(payload)
            distance = payload.get("distance")
            score = 1.0
            if distance is not None:
                try:
                    score = max(0.0, 1.0 - float(_decode(distance)))
                except ValueError:
                    score = 0.0
            elif raw_score is not None:
                try:
                    score = max(0.0, float(_decode(raw_score)))
                except ValueError:
                    score = 0.0
            elif tokens:
                score = _keyword_overlap_score(tokens, item.chunk.text)
                if score <= 0:
                    continue
            parsed.append(
                RagSearchResult(
                    chunk=item.chunk,
                    score=score,
                    match_source=source,
                    knowledge_base_id=item.knowledge_base_id,
                    index_version=item.index_version,
                    updated_at=item.updated_at,
                )
            )
        if normalize_scores:
            parsed = _normalize_result_scores(parsed)
        return parsed

    def _deserialize(self, payload: Dict[Any, Any]) -> StoredRagChunk:
        decoded = {_decode(key): value for key, value in payload.items()}
        metadata = _json_loads(decoded.get("metadata"), {})
        section_path = _json_loads(decoded.get("section_path"), [])
        block_ids = _json_loads(decoded.get("block_ids"), [])
        bboxes = _json_loads(decoded.get("bboxes"), [])
        block_types = [
            value
            for value in _decode(decoded.get("block_types", "")).split(",")
            if value
        ]
        chunk = DocumentChunk(
            chunk_id=_decode(decoded.get("chunk_id", "")),
            document_id=_decode(decoded.get("document_id", "")),
            text=_decode(decoded.get("text", "")),
            page_start=int(float(_decode(decoded.get("page_start", "1")))),
            page_end=int(float(_decode(decoded.get("page_end", "1")))),
            block_ids=list(block_ids),
            block_types=block_types,
            bboxes=list(bboxes),
            section_path=list(section_path),
            metadata=dict(metadata),
        )
        return StoredRagChunk(
            chunk=chunk,
            knowledge_base_id=_decode(decoded.get("knowledge_base_id", "")),
            embedding=_bytes_to_vector(decoded.get("embedding")),
            index_version=_decode(decoded.get("index_version", "")),
            created_at=datetime.fromtimestamp(
                float(_decode(decoded.get("created_at", "0")) or 0)
            ),
            updated_at=datetime.fromtimestamp(
                float(_decode(decoded.get("updated_at", "0")) or 0)
            ),
        )

    def _filter_query(self, knowledge_base_ids: Sequence[str]) -> str:
        if not knowledge_base_ids:
            return "*"
        values = "|".join(_escape_tag(value) for value in knowledge_base_ids)
        return f"@knowledge_base_id:{{{values}}}"

    def _chunk_prefix(self) -> str:
        return f"{self.key_prefix}:v2:chunk:"

    def _legacy_chunk_prefix(self) -> str:
        return f"{self.key_prefix}:chunk:"

    def _document_prefix(self, document_id: str, knowledge_base_id: str) -> str:
        return f"{self.key_prefix}:v2:{self._generation_tag(document_id, knowledge_base_id)}"

    def _document_chunk_prefix(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self._chunk_prefix()}{tag}:"

    def _chunk_key(
        self,
        chunk_id: str,
        knowledge_base_id: str,
        *,
        document_id: Optional[str] = None,
    ) -> str:
        if document_id:
            return f"{self._document_chunk_prefix(document_id, knowledge_base_id)}{chunk_id}"
        return f"{self._chunk_prefix()}{knowledge_base_id}:{chunk_id}"

    def _generation_tag(self, document_id: str, knowledge_base_id: str) -> str:
        return f"{{{knowledge_base_id}:{document_id}}}"

    def _staging_generation_key(
        self, document_id: str, knowledge_base_id: str, generation: str
    ) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:v2:{tag}:staging:{generation}:chunks"

    def _staging_chunk_key(
        self,
        chunk_id: str,
        knowledge_base_id: str,
        document_id: str,
        generation: str,
    ) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:v2:{tag}:staging:{generation}:chunk:{chunk_id}"

    def _staging_resource_key(
        self,
        document_id: str,
        knowledge_base_id: str,
        generation: str,
        live_key: str,
    ) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        digest = hashlib.sha1(live_key.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:v2:{tag}:staging:{generation}:resource:{digest}"

    def _active_generation_key(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:v2:{tag}:active_generation"

    def _visual_chunk_prefix(self) -> str:
        return f"{self.key_prefix}:v2:visual_chunk:"

    def _legacy_visual_chunk_prefix(self) -> str:
        return f"{self.key_prefix}:visual_chunk:"

    def _visual_chunk_document_prefix(
        self, document_id: str, knowledge_base_id: str
    ) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self._visual_chunk_prefix()}{tag}:"

    def _visual_chunk_key(
        self,
        chunk_id: str,
        knowledge_base_id: str,
        *,
        document_id: Optional[str] = None,
    ) -> str:
        if document_id:
            return (
                f"{self._visual_chunk_document_prefix(document_id, knowledge_base_id)}"
                f"{chunk_id}"
            )
        return f"{self._visual_chunk_prefix()}{knowledge_base_id}:{chunk_id}"

    def _legacy_chunk_key(self, chunk_id: str) -> str:
        return f"{self._legacy_chunk_prefix()}{chunk_id}"

    def _document_key(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:document:v2:{tag}:chunks"

    def _legacy_document_key(self, document_id: str) -> str:
        return f"{self.key_prefix}:document:{document_id}:chunks"

    def _document_kbs_key(self, document_id: str) -> str:
        return f"{self.key_prefix}:document:{document_id}:knowledge_bases"

    def _document_tables_key(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:tables:v2:{tag}"

    def _document_visual_key(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:document:v2:{tag}:visual_chunks"

    def _document_figures_key(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:figures:v2:{tag}"

    def _figure_prefix(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:figure:v2:{tag}:"

    def _figure_key(
        self, knowledge_base_id: str, document_id: str, figure_id: str
    ) -> str:
        return f"{self._figure_prefix(document_id, knowledge_base_id)}{figure_id}"

    def _table_prefix(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:table:v2:{tag}:"

    def _table_key(
        self, knowledge_base_id: str, document_id: str, table_id: str
    ) -> str:
        return f"{self._table_prefix(document_id, knowledge_base_id)}{table_id}"

    def _table_rows_prefix(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:table_rows:v2:{tag}:"

    def _table_rows_key(
        self, knowledge_base_id: str, document_id: str, table_id: str
    ) -> str:
        return f"{self._table_rows_prefix(document_id, knowledge_base_id)}{table_id}"

    def _table_row_prefix(self, document_id: str, knowledge_base_id: str) -> str:
        tag = self._generation_tag(document_id, knowledge_base_id)
        return f"{self.key_prefix}:table_row:v2:{tag}:"

    def _table_row_key(
        self,
        knowledge_base_id: str,
        document_id: str,
        table_id: str,
        row_id: str,
    ) -> str:
        return f"{self._table_row_prefix(document_id, knowledge_base_id)}{table_id}:{row_id}"

    async def _upsert_table_sidecars_from_chunk(
        self,
        chunk: DocumentChunk,
        *,
        knowledge_base_id: str,
        index_version: str,
        updated_at: float,
    ) -> None:
        metadata = dict(chunk.metadata or {})
        table_id = str(metadata.get("table_id") or "")
        if not table_id:
            return
        table_sidecar = metadata.get("table_sidecar")
        if isinstance(table_sidecar, dict):
            await self.client.hset(
                self._table_key(knowledge_base_id, chunk.document_id, table_id),
                mapping={
                    "knowledge_base_id": knowledge_base_id,
                    "document_id": chunk.document_id,
                    "table_id": table_id,
                    "index_version": index_version,
                    "updated_at": updated_at,
                    "table_json": json.dumps(table_sidecar, sort_keys=True),
                },
            )
            await self.client.sadd(
                self._document_tables_key(chunk.document_id, knowledge_base_id),
                table_id,
            )

        row_sidecar = metadata.get("table_row_sidecar")
        row_id = str(metadata.get("table_row_id") or "")
        if isinstance(row_sidecar, dict) and row_id:
            await self.client.sadd(
                self._document_tables_key(chunk.document_id, knowledge_base_id),
                table_id,
            )
            await self.client.hset(
                self._table_row_key(
                    knowledge_base_id,
                    chunk.document_id,
                    table_id,
                    row_id,
                ),
                mapping={
                    "knowledge_base_id": knowledge_base_id,
                    "document_id": chunk.document_id,
                    "table_id": table_id,
                    "row_id": row_id,
                    "index_version": index_version,
                    "updated_at": updated_at,
                    "row_json": json.dumps(row_sidecar, sort_keys=True),
                },
            )
            await self.client.sadd(
                self._table_rows_key(knowledge_base_id, chunk.document_id, table_id),
                row_id,
            )

    async def _upsert_figure_sidecars_from_chunk(
        self,
        chunk: DocumentChunk,
        *,
        knowledge_base_id: str,
        index_version: str,
        updated_at: float,
    ) -> None:
        metadata = dict(chunk.metadata or {})
        figure_id = str(metadata.get("figure_id") or "")
        figure_sidecar = metadata.get("figure_sidecar")
        if not figure_id or not isinstance(figure_sidecar, dict):
            return
        await self.client.hset(
            self._figure_key(knowledge_base_id, chunk.document_id, figure_id),
            mapping={
                "knowledge_base_id": knowledge_base_id,
                "document_id": chunk.document_id,
                "figure_id": figure_id,
                "index_version": index_version,
                "updated_at": updated_at,
                "figure_json": json.dumps(
                    _figure_sidecar_for_storage(figure_sidecar), sort_keys=True
                ),
            },
        )
        await self.client.sadd(
            self._document_figures_key(chunk.document_id, knowledge_base_id),
            figure_id,
        )

    async def _upsert_visual_chunk(
        self,
        chunk: DocumentChunk,
        visual_embedding: Optional[Sequence[float]],
        *,
        knowledge_base_id: str,
        index_version: str,
        updated_at: float,
    ) -> None:
        if not self._visual_enabled() or not visual_embedding:
            return
        if len(visual_embedding) != self.visual_dimensions:
            raise ValueError(
                "Visual embedding dimensions do not match RAG visual index"
            )
        now_ts = updated_at
        mapping: Dict[str, Any] = {
            "knowledge_base_id": knowledge_base_id,
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "index_version": index_version,
            "text": chunk.text,
            "block_types": ",".join(chunk.block_types),
            "metadata": json.dumps(
                _metadata_for_chunk_storage(chunk.metadata), sort_keys=True
            ),
            "section_path": json.dumps(chunk.section_path),
            "block_ids": json.dumps(chunk.block_ids),
            "bboxes": json.dumps(chunk.bboxes),
            "page_start": int(chunk.page_start),
            "page_end": int(chunk.page_end),
            "created_at": now_ts,
            "updated_at": now_ts,
            "visual_embedding": _vector_to_bytes(visual_embedding),
        }
        await self.client.hset(
            self._visual_chunk_key(
                chunk.chunk_id,
                knowledge_base_id,
                document_id=chunk.document_id,
            ),
            mapping=mapping,
        )
        await self.client.sadd(
            self._document_visual_key(chunk.document_id, knowledge_base_id),
            chunk.chunk_id,
        )

    async def _delete_table_sidecars(
        self, document_id: str, knowledge_base_id: str
    ) -> None:
        table_ids = await self.client.smembers(
            self._document_tables_key(document_id, knowledge_base_id)
        )
        for raw_table_id in table_ids:
            table_id = _decode(raw_table_id)
            row_ids = await self.client.smembers(
                self._table_rows_key(knowledge_base_id, document_id, table_id)
            )
            if row_ids:
                await self.client.delete(
                    *[
                        self._table_row_key(
                            knowledge_base_id,
                            document_id,
                            table_id,
                            _decode(row_id),
                        )
                        for row_id in row_ids
                    ]
                )
            await self.client.delete(
                self._table_rows_key(knowledge_base_id, document_id, table_id),
                self._table_key(knowledge_base_id, document_id, table_id),
            )
        await self.client.delete(
            self._document_tables_key(document_id, knowledge_base_id)
        )

    async def _delete_figure_sidecars(
        self, document_id: str, knowledge_base_id: str
    ) -> None:
        figure_ids = await self.client.smembers(
            self._document_figures_key(document_id, knowledge_base_id)
        )
        for raw_figure_id in figure_ids:
            await self.client.delete(
                self._figure_key(
                    knowledge_base_id,
                    document_id,
                    _decode(raw_figure_id),
                )
            )
        await self.client.delete(
            self._document_figures_key(document_id, knowledge_base_id)
        )

    async def _delete_visual_chunks(
        self, document_id: str, knowledge_base_id: str
    ) -> int:
        chunk_ids = await self.client.smembers(
            self._document_visual_key(document_id, knowledge_base_id)
        )
        count = 0
        for raw_chunk_id in chunk_ids:
            chunk_id = _decode(raw_chunk_id)
            deleted = await self.client.delete(
                self._visual_chunk_key(
                    chunk_id,
                    knowledge_base_id,
                    document_id=document_id,
                )
            )
            count += int(bool(deleted))
        await self.client.delete(
            self._document_visual_key(document_id, knowledge_base_id)
        )
        return count

    def _keyword_scorer_candidates(self) -> List[Optional[str]]:
        configured = str(
            self.retrieval_config.get("keyword_scorer") or "BM25STD"
        ).strip()
        candidates: List[Optional[str]] = []
        for scorer in [configured, "BM25", None]:
            if scorer and scorer.upper() == "DEFAULT":
                scorer = None
            if scorer not in candidates:
                candidates.append(scorer)
        return candidates

    def _keyword_normalization(self) -> str:
        return (
            str(self.retrieval_config.get("keyword_score_normalization") or "max")
            .strip()
            .lower()
        )

    def _visual_enabled(self) -> bool:
        return bool(self.visual_config.get("enabled", False)) and bool(
            self.visual_embedding_config.get("enabled", False)
        )


def build_vector_store(config: Dict[str, Any]):
    backend = str(config.get("backend", "redis_stack")).lower()
    if backend == "memory":
        return InMemoryRagVectorStore(config)
    return RedisStackRagVectorStore(config)


def _keyword_overlap_score(tokens: Sequence[str], text: str) -> float:
    token_set = set(tokens)
    if not token_set:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(token_set.intersection(text_tokens)) / len(token_set)


def _metadata_for_chunk_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(metadata or {}).items()
        if key not in SIDECAR_METADATA_KEYS
    }


def _figure_sidecar_for_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(metadata or {})
    visual = dict(payload.get("visual") or {})
    visual.pop("visual_embedding", None)
    if visual:
        payload["visual"] = visual
    return payload


def _bm25_scores(
    query_tokens: Sequence[str],
    documents: Sequence[Tuple[str, str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, float]:
    if not query_tokens or not documents:
        return {}
    tokenized_documents = [(doc_id, tokenize(text)) for doc_id, text in documents]
    document_count = len(tokenized_documents)
    lengths = [len(tokens) for _doc_id, tokens in tokenized_documents]
    avg_length = sum(lengths) / document_count if document_count else 0.0
    if avg_length <= 0:
        return {}

    query_terms = list(dict.fromkeys(query_tokens))
    document_frequency: Dict[str, int] = {term: 0 for term in query_terms}
    for _doc_id, doc_tokens in tokenized_documents:
        token_set = set(doc_tokens)
        for term in query_terms:
            if term in token_set:
                document_frequency[term] += 1

    scores: Dict[str, float] = {}
    for doc_id, doc_tokens in tokenized_documents:
        if not doc_tokens:
            continue
        term_counts: Dict[str, int] = {}
        for token in doc_tokens:
            if token in document_frequency:
                term_counts[token] = term_counts.get(token, 0) + 1
        score = 0.0
        doc_length = len(doc_tokens)
        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = term_frequency + k1 * (1.0 - b + b * doc_length / avg_length)
            score += idf * (term_frequency * (k1 + 1.0)) / denominator
        if score > 0:
            scores[doc_id] = score
    return _normalize_scores(scores)


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    max_score = max(scores.values(), default=0.0)
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    return {key: max(0.0, value / max_score) for key, value in scores.items()}


def _normalize_result_scores(
    results: Sequence[RagSearchResult],
) -> List[RagSearchResult]:
    max_score = max((result.score for result in results), default=0.0)
    if max_score <= 0:
        return list(results)
    return [
        RagSearchResult(
            chunk=result.chunk,
            score=max(0.0, result.score / max_score),
            match_source=result.match_source,
            knowledge_base_id=result.knowledge_base_id,
            index_version=result.index_version,
            updated_at=result.updated_at,
        )
        for result in results
    ]


def _cosine_similarity(
    left: Optional[Sequence[float]], right: Optional[Sequence[float]]
) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def _recency_score(updated_at: datetime) -> float:
    age_seconds = max((datetime.now() - updated_at).total_seconds(), 0)
    return 1.0 / (1.0 + age_seconds / 86400.0)


def _merge_results(
    keyword_results: Sequence[RagSearchResult],
    vector_results: Sequence[RagSearchResult],
    visual_results: Sequence[RagSearchResult] = (),
    *,
    keyword_weight: float,
    vector_weight: float,
    visual_weight: float = 0.0,
    recency_weight: float,
    min_score: float,
    visual_min_score: float = 0.0,
) -> List[RagSearchResult]:
    by_id: Dict[str, Tuple[RagSearchResult, float, str]] = {}

    def add(result: RagSearchResult, weight: float):
        if weight <= 0:
            return
        chunk_id = result.chunk.chunk_id
        existing = by_id.get(chunk_id)
        source = result.match_source
        score = result.score * weight
        if existing:
            result = existing[0]
            source = "hybrid_visual" if "visual" in {source, existing[2]} else "hybrid"
            score += existing[1]
        by_id[chunk_id] = (result, score, source)

    for result in keyword_results:
        add(result, keyword_weight)
    for result in vector_results:
        add(result, vector_weight)
    for result in visual_results:
        add(result, visual_weight)

    merged = []
    for result, score, source in by_id.values():
        if result.updated_at:
            score = score + recency_weight * _recency_score(result.updated_at)
        threshold = (
            min(visual_min_score, min_score)
            if source in {"visual", "hybrid_visual"} and visual_min_score > 0
            else min_score
        )
        if score < threshold:
            continue
        merged.append(
            RagSearchResult(
                chunk=result.chunk,
                score=score,
                match_source=source,
                knowledge_base_id=result.knowledge_base_id,
                index_version=result.index_version,
                updated_at=result.updated_at,
            )
        )
    merged.sort(key=lambda item: -item.score)
    return merged


def _escape_tag(value: Any) -> str:
    special_chars = set(",.<>[]{}\"':;!@#$%^&*()-+=~| \\")
    return "".join(
        f"\\{char}" if char in special_chars else char for char in str(value)
    )


def _escape_text_token(value: Any) -> str:
    special_chars = set(",.<>[]{}\"':;!@#$%^&*()-+=~|\\/")
    return "".join(
        f"\\{char}" if char in special_chars else char for char in str(value)
    )


def _new_index_version() -> str:
    return f"{time.time_ns():020d}-{os.getpid():08d}"


def _vector_to_bytes(values: Sequence[float]) -> bytes:
    return array.array("f", [float(value) for value in values]).tobytes()


def _bytes_to_vector(value: Any) -> Optional[List[float]]:
    if value in (None, b"", ""):
        return None
    if isinstance(value, str):
        value = value.encode("latin1")
    vector = array.array("f")
    vector.frombytes(value)
    return list(vector)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(_decode(value))
    except Exception:
        return default
