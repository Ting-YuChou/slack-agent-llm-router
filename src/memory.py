"""
Per-user long-term memory with hybrid keyword and vector retrieval.

The Slack integration uses this module as an optional layer before building
QueryRequest.context. Memory failures are intentionally non-fatal.
"""

import array
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

MEMORY_CONTEXT_HEADER = "Long-term user memory:"
RECENT_CONTEXT_HEADER = "Recent conversation:"
DEFAULT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
    "you",
}


@dataclass
class MemoryItem:
    """A single durable memory item scoped to one Slack workspace/user."""

    scope: str
    text: str
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    expires_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        return self.expires_at <= (now or datetime.now())


@dataclass
class MemorySearchResult:
    """Search result with internal scoring information."""

    item: MemoryItem
    score: float
    match_source: str


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> List[float]:
        """Return an embedding for the supplied text."""


class NoOpEmbeddingProvider:
    """Embedding provider used when vector search is disabled or unavailable."""

    async def embed(self, text: str) -> List[float]:
        raise RuntimeError("Embedding provider is disabled")


class HashEmbeddingProvider:
    """Deterministic local embedding provider for tests and development."""

    def __init__(self, dimensions: int = 128):
        self.dimensions = max(1, int(dimensions))

    async def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    """OpenAI embeddings provider with a narrow dependency surface."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "text-embedding-3-small")
        self.api_key = config.get("api_key")
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        if not self.api_key and api_key_env:
            self.api_key = os.getenv(str(api_key_env))
        self.base_url = config.get("base_url")
        self.timeout = float(config.get("timeout", 10))
        self.client = None

    async def embed(self, text: str) -> List[float]:
        if not self.api_key:
            raise RuntimeError("OpenAI embedding API key is not configured")
        if self.client is None:
            import openai

            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = openai.AsyncOpenAI(**kwargs)

        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return list(response.data[0].embedding)


class LocalHttpEmbeddingProvider:
    """Embedding provider for OpenAI-compatible local embedding endpoints."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config.get("url") or config.get("base_url")
        self.model = config.get("model")
        self.timeout = float(config.get("timeout", 10))

    async def embed(self, text: str) -> List[float]:
        if not self.url:
            raise RuntimeError("Local embedding URL is not configured")
        import httpx

        payload = {"input": text}
        if self.model:
            payload["model"] = self.model
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                str(self.url).rstrip("/") + "/embeddings", json=payload
            )
            response.raise_for_status()
            data = response.json()
        return list(data["data"][0]["embedding"])


class MemoryStore(Protocol):
    async def initialize(self):
        """Initialize backing resources."""

    async def shutdown(self):
        """Release resources."""

    async def add(self, item: MemoryItem):
        """Add or update one memory item."""

    async def get(self, scope: str, memory_id: str) -> Optional[MemoryItem]:
        """Fetch one memory item."""

    async def list(
        self, scope: str, limit: int = 50, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """List memory items for one scope."""

    async def delete(self, scope: str, memory_id: str) -> bool:
        """Delete one memory item."""

    async def delete_all(self, scope: str) -> int:
        """Delete all memory items for one scope."""

    async def keyword_search(
        self, scope: str, tokens: Sequence[str], filters: Dict[str, Any], limit: int
    ) -> List[MemorySearchResult]:
        """Search by keywords and metadata."""

    async def vector_search(
        self,
        scope: str,
        embedding: Sequence[float],
        filters: Dict[str, Any],
        limit: int,
    ) -> List[MemorySearchResult]:
        """Search by vector similarity and metadata."""


def build_embedding_provider(config: Dict[str, Any]) -> EmbeddingProvider:
    provider = str(config.get("provider", "none")).lower()
    if provider in {"none", "disabled", ""}:
        return NoOpEmbeddingProvider()
    if provider in {"hash", "local_hash"}:
        return HashEmbeddingProvider(dimensions=int(config.get("dimensions", 128)))
    if provider == "openai":
        return OpenAIEmbeddingProvider(config)
    if provider in {"local_http", "http"}:
        return LocalHttpEmbeddingProvider(config)
    logger.warning(
        "Unknown embedding provider '%s'; using keyword-only memory", provider
    )
    return NoOpEmbeddingProvider()


def build_memory_store(config: Dict[str, Any]) -> MemoryStore:
    backend = str(config.get("backend", "memory")).lower()
    if backend == "redis_stack":
        return RedisStackMemoryStore(config)
    if backend == "memory":
        return InMemoryMemoryStore()
    logger.warning("Unknown memory backend '%s'; using in-memory memory store", backend)
    return InMemoryMemoryStore()


def tokenize(text: str, min_length: int = 2) -> List[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*", (text or "").lower()):
        if len(token) < min_length or token in DEFAULT_STOP_WORDS:
            continue
        tokens.append(token)
    return sorted(set(tokens))


def build_memory_scope(team_id: Optional[str], user_id: str) -> str:
    normalized_team = (team_id or "default").strip() or "default"
    return f"{normalized_team}:{user_id}"


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _timestamp(value: Optional[datetime]) -> float:
    if value is None:
        return 0.0
    return value.timestamp()


def _escape_redis_tag(value: str) -> str:
    special_chars = set(",.<>[]{}\"':;!@#$%^&*()-+=~| \\")
    return "".join(f"\\{char}" if char in special_chars else char for char in value)


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


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def _keyword_score(tokens: Sequence[str], item: MemoryItem) -> float:
    token_set = set(tokens)
    if not token_set:
        return 0.0
    overlap = token_set.intersection(item.keywords)
    return len(overlap) / len(token_set)


def _matches_filters(item: MemoryItem, filters: Dict[str, Any]) -> bool:
    now = datetime.now()
    if item.is_expired(now):
        return False

    visibility = str(item.metadata.get("visibility") or "channel").lower()
    visibility_scope = filters.get("visibility_scope")
    if visibility_scope == "channel_or_global":
        channel_id = filters.get("channel_id")
        if visibility == "global":
            pass
        elif visibility == "channel" and channel_id:
            if item.metadata.get("channel_id") != channel_id:
                return False
        else:
            return False
    elif filters.get("visibility"):
        if visibility != str(filters["visibility"]).lower():
            return False

    exact_filter_keys = ("thread_ts", "source")
    if visibility_scope != "channel_or_global":
        exact_filter_keys = ("channel_id", "thread_ts", "source")

    for key in exact_filter_keys:
        expected = filters.get(key)
        if expected and item.metadata.get(key) != expected:
            return False
    return True


def _recency_score(item: MemoryItem, now: Optional[datetime] = None) -> float:
    now = now or datetime.now()
    age_seconds = max((now - item.updated_at).total_seconds(), 0)
    return 1.0 / (1.0 + age_seconds / 86400.0)


class InMemoryMemoryStore:
    """In-memory store used for tests and local development."""

    def __init__(self):
        self.items: Dict[str, Dict[str, MemoryItem]] = {}

    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    async def add(self, item: MemoryItem):
        self.items.setdefault(item.scope, {})[item.memory_id] = item

    async def get(self, scope: str, memory_id: str) -> Optional[MemoryItem]:
        item = self.items.get(scope, {}).get(memory_id)
        if item and item.is_expired():
            await self.delete(scope, memory_id)
            return None
        return item

    async def list(
        self, scope: str, limit: int = 50, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        filters = filters or {}
        memories = [
            item
            for item in self.items.get(scope, {}).values()
            if _matches_filters(item, filters)
        ]
        memories.sort(
            key=lambda item: (-item.importance, item.created_at, item.memory_id)
        )
        return memories[:limit]

    async def delete(self, scope: str, memory_id: str) -> bool:
        return self.items.get(scope, {}).pop(memory_id, None) is not None

    async def delete_all(self, scope: str) -> int:
        count = len(self.items.get(scope, {}))
        self.items[scope] = {}
        return count

    async def keyword_search(
        self, scope: str, tokens: Sequence[str], filters: Dict[str, Any], limit: int
    ) -> List[MemorySearchResult]:
        results = []
        for item in self.items.get(scope, {}).values():
            if not _matches_filters(item, filters):
                continue
            score = _keyword_score(tokens, item)
            if score <= 0:
                continue
            results.append(
                MemorySearchResult(item=item, score=score, match_source="keyword")
            )
        results.sort(
            key=lambda result: (
                -result.score,
                -result.item.importance,
                result.item.created_at,
                result.item.memory_id,
            )
        )
        return results[:limit]

    async def vector_search(
        self,
        scope: str,
        embedding: Sequence[float],
        filters: Dict[str, Any],
        limit: int,
    ) -> List[MemorySearchResult]:
        results = []
        for item in self.items.get(scope, {}).values():
            if not item.embedding or not _matches_filters(item, filters):
                continue
            score = _cosine_similarity(embedding, item.embedding)
            if score <= 0:
                continue
            results.append(
                MemorySearchResult(item=item, score=score, match_source="vector")
            )
        results.sort(
            key=lambda result: (
                -result.score,
                -result.item.importance,
                result.item.created_at,
                result.item.memory_id,
            )
        )
        return results[:limit]


class RedisStackMemoryStore:
    """Redis Stack-backed memory store using RediSearch for hybrid retrieval."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_config = dict(config.get("redis", {}) or {})
        self.key_prefix = (
            self.redis_config.get("key_prefix")
            or config.get("key_prefix")
            or "slack_memory"
        )
        self.index_name = f"{self.key_prefix}:idx:memory"
        self.dimensions = int(config.get("embedding", {}).get("dimensions", 1536))
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
                port=int(self.redis_config.get("port", 6379)),
                db=int(self.redis_config.get("db", 3)),
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

        prefix = f"{self.key_prefix}:item:"
        try:
            await self.client.execute_command(
                "FT.CREATE",
                self.index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                prefix,
                "SCHEMA",
                "scope",
                "TAG",
                "SORTABLE",
                "memory_id",
                "TAG",
                "text",
                "TEXT",
                "keywords",
                "TAG",
                "channel_id",
                "TAG",
                "thread_ts",
                "TAG",
                "source",
                "TAG",
                "visibility",
                "TAG",
                "created_at",
                "NUMERIC",
                "SORTABLE",
                "expires_at",
                "NUMERIC",
                "importance",
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
                "Redis Stack memory backend requires RediSearch/vector support"
            ) from exc

    def _item_key(self, scope: str, memory_id: str) -> str:
        return f"{self.key_prefix}:item:{scope}:{memory_id}"

    def _user_index_key(self, scope: str) -> str:
        return f"{self.key_prefix}:user:{scope}:items"

    async def add(self, item: MemoryItem):
        if not self.client:
            return
        key = self._item_key(item.scope, item.memory_id)
        mapping: Dict[str, Any] = {
            "scope": item.scope,
            "memory_id": item.memory_id,
            "text": item.text,
            "keywords": ",".join(item.keywords),
            "metadata": json.dumps(item.metadata, sort_keys=True),
            "channel_id": item.metadata.get("channel_id", ""),
            "thread_ts": item.metadata.get("thread_ts", ""),
            "source": item.metadata.get("source", ""),
            "visibility": item.metadata.get("visibility", "channel"),
            "created_at": _timestamp(item.created_at),
            "updated_at": _timestamp(item.updated_at),
            "expires_at": _timestamp(item.expires_at),
            "importance": float(item.importance),
        }
        if item.embedding:
            if len(item.embedding) != self.dimensions:
                raise ValueError("Embedding dimensions do not match Redis vector index")
            mapping["embedding"] = _vector_to_bytes(item.embedding)
        await self.client.hset(key, mapping=mapping)
        await self.client.sadd(self._user_index_key(item.scope), item.memory_id)

    async def get(self, scope: str, memory_id: str) -> Optional[MemoryItem]:
        if not self.client:
            return None
        payload = await self.client.hgetall(self._item_key(scope, memory_id))
        if not payload:
            return None
        item = self._deserialize(payload)
        if item.is_expired():
            await self.delete(scope, memory_id)
            return None
        return item

    async def list(
        self, scope: str, limit: int = 50, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        if not self.client:
            return []
        filters = filters or {}
        memory_ids = await self.client.smembers(self._user_index_key(scope))
        items = []
        for raw_id in memory_ids:
            memory_id = self._decode(raw_id)
            item = await self.get(scope, memory_id)
            if item and _matches_filters(item, filters):
                items.append(item)
        items.sort(key=lambda item: (-item.importance, item.created_at, item.memory_id))
        return items[:limit]

    async def delete(self, scope: str, memory_id: str) -> bool:
        if not self.client:
            return False
        deleted = await self.client.delete(self._item_key(scope, memory_id))
        await self.client.srem(self._user_index_key(scope), memory_id)
        return bool(deleted)

    async def delete_all(self, scope: str) -> int:
        if not self.client:
            return 0
        memory_ids = await self.client.smembers(self._user_index_key(scope))
        count = 0
        for raw_id in memory_ids:
            if await self.delete(scope, self._decode(raw_id)):
                count += 1
        await self.client.delete(self._user_index_key(scope))
        return count

    async def keyword_search(
        self, scope: str, tokens: Sequence[str], filters: Dict[str, Any], limit: int
    ) -> List[MemorySearchResult]:
        if not self.client or not tokens:
            return []
        query = self._build_filter_query(scope, filters)
        token_query = "|".join(_escape_redis_tag(token) for token in tokens)
        query = f"{query} (@keywords:{{{token_query}}})"
        results = await self._search(query, limit=limit, source="keyword")
        rescored = []
        for result in results:
            score = _keyword_score(tokens, result.item)
            if score <= 0:
                continue
            rescored.append(
                MemorySearchResult(
                    item=result.item,
                    score=score,
                    match_source=result.match_source,
                )
            )
        rescored.sort(
            key=lambda result: (
                -result.score,
                -result.item.importance,
                result.item.created_at,
                result.item.memory_id,
            )
        )
        return rescored[:limit]

    async def vector_search(
        self,
        scope: str,
        embedding: Sequence[float],
        filters: Dict[str, Any],
        limit: int,
    ) -> List[MemorySearchResult]:
        if not self.client or not embedding:
            return []
        if len(embedding) != self.dimensions:
            logger.warning("Skipping vector search due to embedding dimension mismatch")
            return []
        query = (
            f"({self._build_filter_query(scope, filters)})"
            f"=>[KNN {limit} @embedding $vec AS distance]"
        )
        try:
            raw_results = await self.client.execute_command(
                "FT.SEARCH",
                self.index_name,
                query,
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
                "11",
                "scope",
                "memory_id",
                "text",
                "keywords",
                "metadata",
                "created_at",
                "updated_at",
                "expires_at",
                "importance",
                "embedding",
                "distance",
                "DIALECT",
                "2",
            )
        except Exception as exc:
            logger.warning("Redis vector memory search failed: %s", exc)
            return []
        return self._parse_search_results(raw_results, source="vector")

    def _build_filter_query(self, scope: str, filters: Dict[str, Any]) -> str:
        parts = [f"@scope:{{{_escape_redis_tag(scope)}}}"]
        for key in ("thread_ts", "source"):
            value = filters.get(key)
            if value:
                parts.append(f"@{key}:{{{_escape_redis_tag(str(value))}}}")
        visibility_scope = filters.get("visibility_scope")
        if visibility_scope == "channel_or_global":
            channel_id = filters.get("channel_id")
            if channel_id:
                parts.append(
                    "(@visibility:{global}|"
                    f"(@visibility:{{channel}} @channel_id:{{{_escape_redis_tag(str(channel_id))}}}))"
                )
            else:
                parts.append("@visibility:{global}")
        elif filters.get("visibility"):
            parts.append(
                f"@visibility:{{{_escape_redis_tag(str(filters['visibility']))}}}"
            )
            channel_id = filters.get("channel_id")
            if channel_id and str(filters["visibility"]).lower() == "channel":
                parts.append(f"@channel_id:{{{_escape_redis_tag(str(channel_id))}}}")
        else:
            channel_id = filters.get("channel_id")
            if channel_id:
                parts.append(f"@channel_id:{{{_escape_redis_tag(str(channel_id))}}}")
        now_ts = int(time.time())
        parts.append(f"(@expires_at:[0 0]|@expires_at:[{now_ts} +inf])")
        return " ".join(parts)

    async def _search(
        self, query: str, limit: int, source: str
    ) -> List[MemorySearchResult]:
        try:
            raw_results = await self.client.execute_command(
                "FT.SEARCH",
                self.index_name,
                query,
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "10",
                "scope",
                "memory_id",
                "text",
                "keywords",
                "metadata",
                "created_at",
                "updated_at",
                "expires_at",
                "importance",
                "embedding",
            )
        except Exception as exc:
            logger.warning("Redis keyword memory search failed: %s", exc)
            return []
        return self._parse_search_results(raw_results, source=source)

    def _parse_search_results(
        self, raw_results: Any, source: str
    ) -> List[MemorySearchResult]:
        if not raw_results or len(raw_results) < 2:
            return []
        parsed = []
        entries = raw_results[1:]
        for index in range(0, len(entries), 2):
            fields = entries[index + 1]
            payload = {}
            for field_index in range(0, len(fields), 2):
                payload[self._decode(fields[field_index])] = fields[field_index + 1]
            item = self._deserialize(payload)
            distance = payload.get("distance")
            if distance is not None:
                try:
                    score = max(0.0, 1.0 - float(self._decode(distance)))
                except ValueError:
                    score = 0.0
            else:
                score = 1.0
            parsed.append(
                MemorySearchResult(item=item, score=score, match_source=source)
            )
        return parsed

    def _deserialize(self, payload: Dict[Any, Any]) -> MemoryItem:
        decoded = {self._decode(key): value for key, value in payload.items()}
        metadata_raw = self._decode(decoded.get("metadata", "{}"))
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
        keywords = [
            keyword
            for keyword in self._decode(decoded.get("keywords", "")).split(",")
            if keyword
        ]
        return MemoryItem(
            scope=self._decode(decoded.get("scope", "")),
            text=self._decode(decoded.get("text", "")),
            memory_id=self._decode(decoded.get("memory_id", "")),
            keywords=keywords,
            metadata=metadata,
            created_at=_coerce_datetime(
                float(self._decode(decoded.get("created_at", "0")))
            )
            or datetime.now(),
            updated_at=_coerce_datetime(
                float(self._decode(decoded.get("updated_at", "0")))
            )
            or datetime.now(),
            importance=float(self._decode(decoded.get("importance", "0.5"))),
            expires_at=_coerce_datetime(
                float(self._decode(decoded.get("expires_at", "0")))
            )
            if float(self._decode(decoded.get("expires_at", "0"))) > 0
            else None,
            embedding=_bytes_to_vector(decoded.get("embedding")),
        )

    def _decode(self, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)


class MemoryManager:
    """High-level memory API used by Slack handlers."""

    def __init__(
        self,
        config: Dict[str, Any],
        store: Optional[MemoryStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.max_items_per_user = int(self.config.get("max_items_per_user", 500))
        self.retention_days = self.config.get("retention_days") or self.config.get(
            "ttl_days"
        )
        self.search_config = dict(self.config.get("search", {}) or {})
        self.max_results = int(self.search_config.get("max_results", 5))
        self.max_context_chars = int(self.search_config.get("max_context_chars", 2000))
        self.max_item_chars = int(self.search_config.get("max_item_chars", 500))
        self.keyword_weight = float(self.search_config.get("keyword_weight", 0.45))
        self.vector_weight = float(self.search_config.get("vector_weight", 0.45))
        self.recency_weight = float(self.search_config.get("recency_weight", 0.05))
        self.importance_weight = float(
            self.search_config.get("importance_weight", 0.05)
        )
        self.store = store or build_memory_store(self.config)
        self.embedding_provider = embedding_provider or build_embedding_provider(
            dict(self.config.get("embedding", {}) or {})
        )

    async def initialize(self):
        if not self.enabled:
            return
        await self.store.initialize()

    async def shutdown(self):
        if not self.enabled:
            return
        await self.store.shutdown()

    async def remember(
        self,
        scope: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> Optional[MemoryItem]:
        if not self.enabled:
            return None
        normalized_text = (text or "").strip()
        if not normalized_text:
            return None

        existing_count = len(
            await self.store.list(scope, limit=self.max_items_per_user + 1)
        )
        if existing_count >= self.max_items_per_user:
            raise ValueError("Memory limit reached for this user")

        now = datetime.now()
        expires_at = None
        if self.retention_days:
            expires_at = now + timedelta(days=int(self.retention_days))
        embedding = await self._safe_embed(normalized_text)
        item = MemoryItem(
            scope=scope,
            text=normalized_text,
            keywords=tokenize(normalized_text),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
            importance=min(max(float(importance), 0.0), 1.0),
            expires_at=expires_at,
            embedding=embedding,
        )
        await self.store.add(item)
        return item

    async def list_memories(
        self,
        scope: str,
        limit: int = 20,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryItem]:
        if not self.enabled:
            return []
        filters = self._search_filters(metadata)
        return await self.store.list(scope, limit=limit, filters=filters)

    async def forget(self, scope: str, memory_id: str) -> bool:
        if not self.enabled:
            return False
        return await self.store.delete(scope, memory_id)

    async def forget_all(self, scope: str) -> int:
        if not self.enabled:
            return 0
        return await self.store.delete_all(scope)

    async def search(
        self, scope: str, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[MemorySearchResult]:
        if not self.enabled:
            return []
        filters = self._search_filters(metadata)
        tokens = tokenize(query)
        candidate_limit = max(self.max_results * 3, self.max_results)
        keyword_results = await self.store.keyword_search(
            scope, tokens, filters, candidate_limit
        )
        embedding = await self._safe_embed(query)
        vector_results = []
        if embedding:
            vector_results = await self.store.vector_search(
                scope, embedding, filters, candidate_limit
            )
        merged = self._merge_results(keyword_results, vector_results)
        return merged[: self.max_results]

    def build_context(
        self, results: Sequence[MemorySearchResult], conversation_context: str
    ) -> str:
        if not results:
            return conversation_context

        remaining = self.max_context_chars
        memory_lines = []
        for index, result in enumerate(results, start=1):
            text = " ".join(result.item.text.split())
            if len(text) > self.max_item_chars:
                text = text[: self.max_item_chars].rstrip() + "..."
            line = f"{index}. {text}"
            if len(line) + 1 > remaining:
                break
            memory_lines.append(line)
            remaining -= len(line) + 1

        if not memory_lines:
            return conversation_context

        sections = [MEMORY_CONTEXT_HEADER, *memory_lines]
        if conversation_context:
            sections.extend(["", RECENT_CONTEXT_HEADER, conversation_context])
        return "\n".join(sections)

    def _merge_results(
        self,
        keyword_results: Sequence[MemorySearchResult],
        vector_results: Sequence[MemorySearchResult],
    ) -> List[MemorySearchResult]:
        by_id: Dict[str, Tuple[MemoryItem, float, str]] = {}
        now = datetime.now()

        def add(result: MemorySearchResult, weight: float):
            memory_id = result.item.memory_id
            item, current_score, source = by_id.get(
                memory_id, (result.item, 0.0, result.match_source)
            )
            combined_source = source
            if source != result.match_source:
                combined_source = "hybrid"
            score = current_score + weight * result.score
            by_id[memory_id] = (item, score, combined_source)

        for result in keyword_results:
            add(result, self.keyword_weight)
        for result in vector_results:
            add(result, self.vector_weight)

        merged = []
        for item, base_score, source in by_id.values():
            score = (
                base_score
                + self.recency_weight * _recency_score(item, now)
                + self.importance_weight * item.importance
            )
            merged.append(
                MemorySearchResult(item=item, score=score, match_source=source)
            )

        merged.sort(
            key=lambda result: (
                -result.score,
                -result.item.importance,
                result.item.created_at,
                result.item.memory_id,
            )
        )
        return merged

    def _search_filters(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        allowed_keys = {
            "channel_id",
            "thread_ts",
            "source",
            "visibility",
            "visibility_scope",
        }
        return {
            key: value
            for key, value in (metadata or {}).items()
            if key in allowed_keys and value
        }

    async def _safe_embed(self, text: str) -> Optional[List[float]]:
        try:
            return await asyncio.wait_for(
                self.embedding_provider.embed(text),
                timeout=float(self.config.get("embedding", {}).get("timeout", 10)),
            )
        except Exception as exc:
            logger.info(
                "Memory embedding unavailable; falling back to keyword-only: %s", exc
            )
            return None
