from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.memory import (
    HashEmbeddingProvider,
    InMemoryMemoryStore,
    MemoryItem,
    MemoryManager,
    MemorySearchResult,
    RedisStackMemoryStore,
    build_memory_scope,
    tokenize,
)


class FailingEmbeddingProvider:
    async def embed(self, text):
        raise RuntimeError("embedding unavailable")


@pytest.mark.asyncio
async def test_tokenizer_and_keyword_search_are_user_scoped():
    manager = MemoryManager(
        {
            "enabled": True,
            "search": {"max_results": 5},
            "embedding": {"provider": "none"},
        },
        store=InMemoryMemoryStore(),
        embedding_provider=FailingEmbeddingProvider(),
    )
    await manager.initialize()

    await manager.remember(
        "T1:U1",
        "The billing project uses ClickHouse for analytics",
        metadata={"source": "slack"},
    )
    await manager.remember(
        "T1:U2",
        "The billing project uses Postgres",
        metadata={"source": "slack"},
    )

    results = await manager.search("T1:U1", "billing ClickHouse", {"source": "slack"})

    assert tokenize("The billing project uses ClickHouse") == [
        "billing",
        "clickhouse",
        "project",
        "uses",
    ]
    assert [result.item.text for result in results] == [
        "The billing project uses ClickHouse for analytics"
    ]


@pytest.mark.asyncio
async def test_hybrid_search_uses_vector_when_keywords_do_not_overlap():
    manager = MemoryManager(
        {
            "enabled": True,
            "search": {
                "max_results": 5,
                "keyword_weight": 0.45,
                "vector_weight": 0.45,
            },
            "embedding": {"provider": "hash", "dimensions": 16},
        },
        store=InMemoryMemoryStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    await manager.initialize()

    item = MemoryItem(
        scope="T1:U1",
        text="Use terse answers for deployment questions",
        keywords=["terse", "answers"],
        metadata={"source": "slack"},
        embedding=await manager.embedding_provider.embed("concise deployment help"),
    )
    await manager.store.add(item)

    results = await manager.search("T1:U1", "concise deployment help", {"source": "slack"})

    assert results
    assert results[0].item.memory_id == item.memory_id
    assert results[0].match_source == "vector"


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_to_keyword_only():
    manager = MemoryManager(
        {"enabled": True, "search": {"max_results": 5}},
        store=InMemoryMemoryStore(),
        embedding_provider=FailingEmbeddingProvider(),
    )
    await manager.initialize()
    item = await manager.remember(
        "T1:U1",
        "Flink analytics jobs write rolling metrics",
        metadata={"source": "slack"},
    )

    results = await manager.search("T1:U1", "Flink metrics", {"source": "slack"})

    assert item is not None
    assert item.embedding is None
    assert [result.item.memory_id for result in results] == [item.memory_id]
    assert results[0].match_source == "keyword"


@pytest.mark.asyncio
async def test_visibility_filter_allows_global_and_same_channel_only():
    manager = MemoryManager(
        {"enabled": True, "search": {"max_results": 5}},
        store=InMemoryMemoryStore(),
        embedding_provider=FailingEmbeddingProvider(),
    )
    await manager.initialize()
    await manager.remember(
        "T1:U1",
        "Use Python examples in C1",
        metadata={"source": "slack", "visibility": "channel", "channel_id": "C1"},
    )
    await manager.remember(
        "T1:U1",
        "Use concise answers everywhere",
        metadata={"source": "slack", "visibility": "global"},
    )

    same_channel = await manager.search(
        "T1:U1",
        "Use Python concise examples",
        {"source": "slack", "visibility_scope": "channel_or_global", "channel_id": "C1"},
    )
    other_channel = await manager.search(
        "T1:U1",
        "Use Python concise examples",
        {"source": "slack", "visibility_scope": "channel_or_global", "channel_id": "C2"},
    )

    assert {result.item.text for result in same_channel} == {
        "Use Python examples in C1",
        "Use concise answers everywhere",
    }
    assert [result.item.text for result in other_channel] == [
        "Use concise answers everywhere"
    ]


@pytest.mark.asyncio
async def test_redis_keyword_search_rescores_by_token_overlap():
    store = RedisStackMemoryStore({"embedding": {"dimensions": 16}})
    store.client = object()
    weak = MemoryItem(
        scope="T1:U1",
        text="billing only",
        keywords=["billing"],
        memory_id="weak",
    )
    strong = MemoryItem(
        scope="T1:U1",
        text="billing clickhouse analytics",
        keywords=["billing", "clickhouse", "analytics"],
        memory_id="strong",
    )
    store._search = AsyncMock(
        return_value=[
            MemorySearchResult(item=weak, score=1.0, match_source="keyword"),
            MemorySearchResult(item=strong, score=1.0, match_source="keyword"),
        ]
    )

    results = await store.keyword_search(
        "T1:U1", ["billing", "clickhouse", "analytics"], {}, limit=5
    )

    assert [result.item.memory_id for result in results] == ["strong", "weak"]
    assert results[0].score == 1.0
    assert results[1].score == pytest.approx(1 / 3)


def test_context_format_is_deterministic_and_excludes_scores():
    manager = MemoryManager(
        {"enabled": True, "search": {"max_context_chars": 200, "max_item_chars": 80}},
        store=InMemoryMemoryStore(),
        embedding_provider=FailingEmbeddingProvider(),
    )
    item = MemoryItem(
        scope="T1:U1",
        text="Always answer API design questions with concrete endpoint examples.",
        memory_id="m1",
        created_at=datetime.now() - timedelta(days=1),
        metadata={"source": "slack"},
    )
    result = MemorySearchResult(item=item, score=0.1234, match_source="hybrid")

    context = manager.build_context([result], "user: previous message")

    assert context == (
        "Long-term user memory:\n"
        "1. Always answer API design questions with concrete endpoint examples.\n"
        "\n"
        "Recent conversation:\n"
        "user: previous message"
    )
    assert "0.1234" not in context
    assert "hybrid" not in context


def test_context_unchanged_when_no_memory_hits():
    manager = MemoryManager({"enabled": True}, store=InMemoryMemoryStore())

    assert manager.build_context([], "user: hello") == "user: hello"


@pytest.mark.asyncio
async def test_forget_and_forget_all():
    manager = MemoryManager(
        {"enabled": True},
        store=InMemoryMemoryStore(),
        embedding_provider=FailingEmbeddingProvider(),
    )
    first = await manager.remember("T1:U1", "first", metadata={"source": "slack"})
    await manager.remember("T1:U1", "second", metadata={"source": "slack"})

    assert first is not None
    assert await manager.forget("T1:U1", first.memory_id) is True
    assert len(await manager.list_memories("T1:U1")) == 1
    assert await manager.forget_all("T1:U1") == 1
    assert await manager.list_memories("T1:U1") == []


def test_memory_scope_uses_team_id_when_available():
    assert build_memory_scope("T1", "U1") == "T1:U1"
    assert build_memory_scope(None, "U1") == "default:U1"
