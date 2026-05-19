"""Vector stores for school-document RAG chunks."""

import array
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


@dataclass
class StoredRagChunk:
    """A chunk plus index metadata."""

    chunk: DocumentChunk
    knowledge_base_id: str
    embedding: Optional[List[float]] = None
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
    ) -> int:
        if not chunks:
            return 0
        now = datetime.now()
        document_id = chunks[0].document_id
        await self.delete_document(document_id, knowledge_base_id)
        index_version = _new_index_version()
        for chunk, embedding in zip(chunks, embeddings):
            storage_id = self._storage_id(knowledge_base_id, chunk.chunk_id)
            existing = self.items.get(storage_id)
            self.items[storage_id] = StoredRagChunk(
                chunk=chunk,
                knowledge_base_id=knowledge_base_id,
                embedding=list(embedding) if embedding else None,
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
    ) -> List[RagSearchResult]:
        tokens = tokenize(query)
        allowed_kbs = set(knowledge_base_ids or [])
        results: Dict[str, Tuple[StoredRagChunk, float, str]] = {}
        for storage_id, item in self.items.items():
            if allowed_kbs and item.knowledge_base_id not in allowed_kbs:
                continue
            keyword_score = _keyword_score(tokens, item.chunk.text)
            vector_score = (
                _cosine_similarity(embedding, item.embedding)
                if embedding and item.embedding
                else 0.0
            )
            score = (
                keyword_weight * keyword_score
                + vector_weight * vector_score
                + recency_weight * _recency_score(item.updated_at)
            )
            if score <= 0 or score < min_score:
                continue
            source = "hybrid"
            if keyword_score > 0 and vector_score <= 0:
                source = "keyword"
            elif vector_score > 0 and keyword_score <= 0:
                source = "vector"
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
        self.embedding_config = dict(self.config.get("embedding", {}) or {})
        self.key_prefix = self.redis_config.get("key_prefix", "rag")
        self.index_name = f"{self.key_prefix}:idx:chunks"
        self.dimensions = int(self.embedding_config.get("dimensions", 1024))
        self.client = None

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

    async def shutdown(self):
        if self.client and hasattr(self.client, "close"):
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

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Optional[Sequence[float]]],
        *,
        knowledge_base_id: str,
    ) -> int:
        if not self.client or not chunks:
            return 0
        now_ts = time.time()
        document_id = chunks[0].document_id
        await self.delete_document(document_id, knowledge_base_id)
        index_version = _new_index_version()
        count = 0
        for chunk, embedding in zip(chunks, embeddings):
            mapping: Dict[str, Any] = {
                "knowledge_base_id": knowledge_base_id,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "index_version": index_version,
                "text": chunk.text,
                "keywords": ",".join(tokenize(chunk.text)),
                "block_types": ",".join(chunk.block_types),
                "metadata": json.dumps(chunk.metadata, sort_keys=True),
                "section_path": json.dumps(chunk.section_path),
                "block_ids": json.dumps(chunk.block_ids),
                "bboxes": json.dumps(chunk.bboxes),
                "page_start": int(chunk.page_start),
                "page_end": int(chunk.page_end),
                "created_at": now_ts,
                "updated_at": now_ts,
            }
            if embedding:
                if len(embedding) != self.dimensions:
                    raise ValueError("Embedding dimensions do not match RAG index")
                mapping["embedding"] = _vector_to_bytes(embedding)
            await self.client.hset(
                self._chunk_key(chunk.chunk_id, knowledge_base_id), mapping=mapping
            )
            await self.client.sadd(
                self._document_key(chunk.document_id, knowledge_base_id), chunk.chunk_id
            )
            await self.client.sadd(
                self._document_kbs_key(chunk.document_id), knowledge_base_id
            )
            count += 1
        return count

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
                self._chunk_key(chunk_id, knowledge_base_id)
            )
            count += int(bool(deleted))
        await self.client.delete(self._document_key(document_id, knowledge_base_id))

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
    ) -> List[RagSearchResult]:
        keyword_results = await self._keyword_search(
            query, knowledge_base_ids, candidate_count
        )
        vector_results = await self._vector_search(
            embedding, knowledge_base_ids, candidate_count
        )
        merged = _merge_results(
            keyword_results,
            vector_results,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            recency_weight=recency_weight,
            min_score=min_score,
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
        results = await self._search(redis_query, limit, "keyword", tokens=tokens)
        if results:
            return results

        token_query = "|".join(_escape_tag(token) for token in tokens)
        fallback_query = (
            f"{filter_query} @keywords:{{{token_query}}}"
            if filter_query != "*"
            else f"@keywords:{{{token_query}}}"
        )
        return await self._search(fallback_query, limit, "keyword", tokens=tokens)

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
                "16",
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
                "embedding",
                "index_version",
                "distance",
                "DIALECT",
                "2",
            )
        except Exception as exc:
            logger.warning("Redis RAG vector search failed: %s", exc)
            return []
        return self._parse_results(raw_results, "vector")

    async def _search(
        self,
        query: str,
        limit: int,
        source: str,
        tokens: Optional[Sequence[str]] = None,
    ) -> List[RagSearchResult]:
        try:
            raw_results = await self.client.execute_command(
                "FT.SEARCH",
                self.index_name,
                query,
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "15",
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
                "embedding",
            )
        except Exception as exc:
            logger.warning("Redis RAG keyword search failed: %s", exc)
            return []
        return self._parse_results(raw_results, source, tokens=tokens)

    def _parse_results(
        self,
        raw_results: Any,
        source: str,
        tokens: Optional[Sequence[str]] = None,
    ) -> List[RagSearchResult]:
        if not raw_results or len(raw_results) < 2:
            return []
        parsed = []
        entries = raw_results[1:]
        for index in range(0, len(entries), 2):
            fields = entries[index + 1]
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
            elif tokens:
                score = _keyword_score(tokens, item.chunk.text)
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
        return f"{self.key_prefix}:chunk:"

    def _chunk_key(self, chunk_id: str, knowledge_base_id: str) -> str:
        return f"{self._chunk_prefix()}{knowledge_base_id}:{chunk_id}"

    def _legacy_chunk_key(self, chunk_id: str) -> str:
        return f"{self._chunk_prefix()}{chunk_id}"

    def _document_key(self, document_id: str, knowledge_base_id: str) -> str:
        return f"{self.key_prefix}:document:{knowledge_base_id}:{document_id}:chunks"

    def _legacy_document_key(self, document_id: str) -> str:
        return f"{self.key_prefix}:document:{document_id}:chunks"

    def _document_kbs_key(self, document_id: str) -> str:
        return f"{self.key_prefix}:document:{document_id}:knowledge_bases"


def build_vector_store(config: Dict[str, Any]):
    backend = str(config.get("backend", "redis_stack")).lower()
    if backend == "memory":
        return InMemoryRagVectorStore(config)
    return RedisStackRagVectorStore(config)


def _keyword_score(tokens: Sequence[str], text: str) -> float:
    token_set = set(tokens)
    if not token_set:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(token_set.intersection(text_tokens)) / len(token_set)


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
    *,
    keyword_weight: float,
    vector_weight: float,
    recency_weight: float,
    min_score: float,
) -> List[RagSearchResult]:
    by_id: Dict[str, Tuple[RagSearchResult, float, str]] = {}

    def add(result: RagSearchResult, weight: float):
        chunk_id = result.chunk.chunk_id
        existing = by_id.get(chunk_id)
        source = result.match_source
        score = result.score * weight
        if existing:
            result = existing[0]
            source = "hybrid"
            score += existing[1]
        by_id[chunk_id] = (result, score, source)

    for result in keyword_results:
        add(result, keyword_weight)
    for result in vector_results:
        add(result, vector_weight)

    merged = []
    for result, score, source in by_id.values():
        if result.updated_at:
            score = score + recency_weight * _recency_score(result.updated_at)
        if score < min_score:
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
    payload = f"{time.time_ns()}:{os.getpid()}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


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
