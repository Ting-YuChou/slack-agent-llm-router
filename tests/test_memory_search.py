"""Memory search: skip work for users without memories, drop weak matches."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from src.memory import (
    HashEmbeddingProvider,
    InMemoryMemoryStore,
    MemoryItem,
    MemoryManager,
    MemorySearchResult,
    RedisStackMemoryStore,
)
from src.utils.schema import SlackMemorySearchConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class CountingEmbeddingProvider:
    def __init__(self, dimensions=16):
        self.inner = HashEmbeddingProvider(dimensions=dimensions)
        self.calls = []

    async def embed(self, text):
        self.calls.append(text)
        return await self.inner.embed(text)

    async def embed_many(self, texts):
        return [await self.embed(text) for text in texts]


class CountingStore(InMemoryMemoryStore):
    def __init__(self):
        super().__init__()
        self.has_items_calls = 0
        self.keyword_calls = 0

    async def has_items(self, scope):
        self.has_items_calls += 1
        return await super().has_items(scope)

    async def keyword_search(self, scope, tokens, filters, limit):
        self.keyword_calls += 1
        return await super().keyword_search(scope, tokens, filters, limit)


class ScoredVectorStore(InMemoryMemoryStore):
    """Returns vector neighbours with fixed similarity scores, like a KNN."""

    def __init__(self, scores):
        super().__init__()
        self.scored = [
            MemoryItem(scope="T1:U1", text=f"memory {score}", keywords=[])
            for score in scores
        ]
        self.scores = scores
        for item in self.scored:
            self.items.setdefault("T1:U1", {})[item.memory_id] = item

    async def vector_search(self, scope, embedding, filters, limit):
        return [
            MemorySearchResult(item=item, score=score, match_source="vector")
            for item, score in zip(self.scored, self.scores)
        ]


def manager_for(store, embedder=None, **search):
    return MemoryManager(
        {
            "enabled": True,
            "search": {"max_results": 5, **search},
            "embedding": {"provider": "hash", "dimensions": 16},
        },
        store=store,
        embedding_provider=embedder or CountingEmbeddingProvider(),
    )


# S16: no embedding for a user without memories


@pytest.mark.asyncio
async def test_search_without_any_memory_never_calls_the_embedder():
    embedder = CountingEmbeddingProvider()
    store = CountingStore()
    manager = manager_for(store, embedder)

    assert await manager.search("T1:U1", "how do I deploy?") == []
    assert await manager.search("T1:U1", "and roll back?") == []

    assert embedder.calls == []
    assert store.keyword_calls == 0
    # The empty answer is cached for a little while.
    assert store.has_items_calls == 1


@pytest.mark.asyncio
async def test_remember_invalidates_the_cached_empty_answer():
    embedder = CountingEmbeddingProvider()
    manager = manager_for(CountingStore(), embedder)

    assert await manager.search("T1:U1", "deploy region") == []
    await manager.remember("T1:U1", "We deploy to the eu-west region")
    embedder.calls.clear()

    results = await manager.search("T1:U1", "deploy region")

    assert [result.item.text for result in results] == [
        "We deploy to the eu-west region"
    ]
    assert embedder.calls == ["deploy region"]


@pytest.mark.asyncio
async def test_forget_all_invalidates_the_cached_answer():
    embedder = CountingEmbeddingProvider()
    manager = manager_for(CountingStore(), embedder)
    await manager.remember("T1:U1", "We deploy to the eu-west region")
    assert await manager.search("T1:U1", "deploy region")

    await manager.forget_all("T1:U1")
    embedder.calls.clear()

    assert await manager.search("T1:U1", "deploy region") == []
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_forget_invalidates_the_cached_answer():
    embedder = CountingEmbeddingProvider()
    manager = manager_for(CountingStore(), embedder)
    item = await manager.remember("T1:U1", "We deploy to the eu-west region")
    assert await manager.search("T1:U1", "deploy region")

    await manager.forget("T1:U1", item.memory_id)
    embedder.calls.clear()

    assert await manager.search("T1:U1", "deploy region") == []
    assert embedder.calls == []


@pytest.mark.asyncio
async def test_a_store_without_has_items_is_still_searched():
    class LegacyStore(InMemoryMemoryStore):
        has_items = None

    embedder = CountingEmbeddingProvider()
    manager = manager_for(LegacyStore(), embedder)
    await manager.remember("T1:U1", "We deploy to the eu-west region")

    assert await manager.search("T1:U1", "deploy region")


@pytest.mark.asyncio
async def test_a_stale_empty_answer_is_not_cached_over_a_new_memory():
    """search reads "no items", remember adds one, then search would cache it."""
    counted = asyncio.Event()
    release = asyncio.Event()

    class SlowCountStore(InMemoryMemoryStore):
        async def has_items(self, scope):
            answer = await super().has_items(scope)
            counted.set()
            await release.wait()
            return answer

    manager = manager_for(SlowCountStore())
    first = asyncio.create_task(manager.search("T1:U1", "deploy region"))
    await asyncio.wait_for(counted.wait(), timeout=1)
    await manager.remember("T1:U1", "We deploy to the eu-west region")
    release.set()
    assert await first == []

    results = await manager.search("T1:U1", "deploy region")

    assert [result.item.text for result in results] == [
        "We deploy to the eu-west region"
    ]


@pytest.mark.asyncio
async def test_a_failed_remember_still_drops_the_cached_answer():
    class PartialWriteStore(InMemoryMemoryStore):
        async def add(self, item):
            # Like Redis: the item hash lands, then the index write fails.
            await super().add(item)
            raise ConnectionError("redis went away")

    manager = manager_for(PartialWriteStore())
    assert await manager.search("T1:U1", "deploy region") == []

    with pytest.raises(ConnectionError):
        await manager.remember("T1:U1", "We deploy to the eu-west region")

    assert await manager.search("T1:U1", "deploy region")


@pytest.mark.asyncio
async def test_in_memory_store_has_items_counts_one_scope():
    store = InMemoryMemoryStore()
    await store.add(MemoryItem(scope="T1:U1", text="hello", keywords=["hello"]))

    assert await store.has_items("T1:U1")
    assert not await store.has_items("T1:U2")


@pytest.mark.asyncio
async def test_redis_store_has_items_uses_scard_on_the_user_index():
    store = RedisStackMemoryStore({"embedding": {"dimensions": 16}})
    store.client = AsyncMock()
    store.client.scard = AsyncMock(side_effect=[0, 3])

    assert not await store.has_items("T1:U1")
    assert await store.has_items("T1:U1")
    store.client.scard.assert_awaited_with(store._user_index_key("T1:U1"))


@pytest.mark.asyncio
async def test_redis_store_has_items_fails_open():
    store = RedisStackMemoryStore({"embedding": {"dimensions": 16}})
    store.client = AsyncMock()
    store.client.scard = AsyncMock(side_effect=ConnectionError("redis down"))

    assert await store.has_items("T1:U1")


# S16: a vector-similarity floor


@pytest.mark.asyncio
async def test_low_scoring_vector_neighbours_are_not_returned():
    manager = manager_for(ScoredVectorStore([0.82, 0.12]), min_score=0.3)

    results = await manager.search("T1:U1", "unrelated question")

    assert [result.item.text for result in results] == ["memory 0.82"]


@pytest.mark.asyncio
async def test_min_score_defaults_to_off():
    manager = manager_for(ScoredVectorStore([0.82, 0.12]))

    results = await manager.search("T1:U1", "unrelated question")

    assert [result.item.text for result in results] == ["memory 0.82", "memory 0.12"]


@pytest.mark.asyncio
async def test_keyword_search_and_embedding_run_concurrently():
    """S6c: each side waits for the other to start; in sequence it times out."""
    keyword_started = asyncio.Event()
    embed_started = asyncio.Event()
    overlapped = []

    async def wait_for(event):
        try:
            await asyncio.wait_for(event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            overlapped.append(False)
        else:
            overlapped.append(True)

    class WaitingStore(InMemoryMemoryStore):
        async def keyword_search(self, scope, tokens, filters, limit):
            keyword_started.set()
            await wait_for(embed_started)
            return await super().keyword_search(scope, tokens, filters, limit)

    class WaitingEmbedder(CountingEmbeddingProvider):
        async def embed(self, text):
            embed_started.set()
            await wait_for(keyword_started)
            return await super().embed(text)

    store = WaitingStore()
    embedder = WaitingEmbedder()
    item = MemoryItem(
        scope="T1:U1",
        text="We deploy to the eu-west region",
        keywords=["deploy", "region"],
        embedding=await embedder.inner.embed("deploy region"),
    )
    await store.add(item)
    manager = manager_for(store, embedder)

    results = await manager.search("T1:U1", "deploy region")

    assert overlapped == [True, True]
    assert results[0].item.memory_id == item.memory_id
    assert results[0].match_source == "hybrid"


def test_search_config_min_score_defaults_to_zero():
    assert SlackMemorySearchConfig().min_score == 0.0
    with pytest.raises(ValueError):
        SlackMemorySearchConfig(min_score=1.5)


@pytest.mark.parametrize("name", ["config.yaml", "config.compose.yaml"])
def test_shipped_configs_set_a_vector_floor_with_memory_still_disabled(name):
    config = yaml.safe_load((CONFIG_DIR / name).read_text())
    memory = config["slack"]["memory"]

    assert memory["enabled"] is False
    assert memory["search"]["min_score"] == 0.3
