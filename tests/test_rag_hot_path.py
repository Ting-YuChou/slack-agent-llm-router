import asyncio
from pathlib import Path

import pytest
import yaml

import main
from src.memory import LocalHttpEmbeddingProvider, OpenAIEmbeddingProvider
from src.rag.chunker import DocumentChunk
from src.rag.service import RagService
from src.rag.vector_store import RedisStackRagVectorStore
from src.utils.schema import PlatformConfig


class _HeartbeatRedis:
    def __init__(self):
        self.calls = []

    async def set(self, key, value, *, ex):
        self.calls.append((key, value, ex))

    async def get(self, key):
        return None

    async def setex(self, key, ttl, value):
        return True


class _HeartbeatStore:
    def __init__(self, client):
        self.client = client

    async def initialize(self):
        return None

    async def shutdown(self):
        return None


def test_rag_environment_overrides_enable_service_and_queue(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "logging": {"file": str(tmp_path / "router.log")},
                "rag": {
                    "enabled": False,
                    "ingestion_queue": {"enabled": False},
                },
            }
        )
    )
    monkeypatch.setenv("LLM_ROUTER_RAG_ENABLED", "true")
    monkeypatch.setenv("LLM_ROUTER_RAG_QUEUE_ENABLED", "true")

    platform = main.LLMRouterPlatform(str(config_path))

    assert platform.config["rag"]["enabled"] is True
    assert platform.config["rag"]["ingestion_queue"]["enabled"] is True


def test_rag_compose_overlay_enables_api_and_worker():
    overlay = yaml.safe_load(Path("docker-compose.rag.yml").read_text())

    for service_name in ("api", "rag-workers"):
        environment = overlay["services"][service_name]["environment"]
        assert environment["LLM_ROUTER_RAG_ENABLED"] == "true"
        assert environment["LLM_ROUTER_RAG_QUEUE_ENABLED"] == "true"
        assert "redis-stack" in overlay["services"][service_name]["depends_on"]
    assert "healthcheck" in overlay["services"]["rag-workers"]


@pytest.mark.asyncio
async def test_worker_heartbeat_is_refreshed_with_ttl(tmp_path):
    redis = _HeartbeatRedis()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "ingestion_queue": {
                "enabled": True,
                "heartbeat_interval_seconds": 0.01,
                "heartbeat_ttl_seconds": 1,
            },
            "storage": {"staging_dir": str(tmp_path)},
        },
        vector_store=_HeartbeatStore(redis),
    )

    task = asyncio.create_task(service._worker_heartbeat_loop("worker-a"))
    await asyncio.sleep(0.025)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(redis.calls) >= 2
    assert all(call[0].endswith(":heartbeat:worker-a") for call in redis.calls)
    assert all(call[2] == 1 for call in redis.calls)


class _BatchEmbeddingProvider:
    def __init__(self):
        self.batch_sizes = []
        self.active = 0
        self.max_active = 0

    async def embed(self, text):
        return [float(text)]

    async def embed_many(self, texts):
        self.batch_sizes.append(len(texts))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        return [[float(text)] for text in texts]


def _chunks(count):
    return [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            document_id="doc",
            text=str(index),
            page_start=1,
            page_end=1,
            block_ids=[f"block-{index}"],
            block_types=["text"],
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_embedding_batches_are_bounded_and_ordered():
    provider = _BatchEmbeddingProvider()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"batch_size": 32, "max_concurrent_batches": 2},
        },
        embedding_provider=provider,
    )

    embeddings, warnings = await service._embed_chunks(_chunks(1000))

    assert len(provider.batch_sizes) == 32
    assert max(provider.batch_sizes) == 32
    assert provider.max_active == 2
    assert embeddings == [[float(index)] for index in range(1000)]
    assert warnings == []


@pytest.mark.asyncio
async def test_embedding_batch_failure_only_degrades_that_batch():
    class Provider(_BatchEmbeddingProvider):
        async def embed_many(self, texts):
            if texts[0] == "32":
                raise RuntimeError("batch unavailable")
            return await super().embed_many(texts)

    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"batch_size": 32, "max_concurrent_batches": 2},
        },
        embedding_provider=Provider(),
    )

    embeddings, warnings = await service._embed_chunks(_chunks(70))

    assert embeddings[:32] == [[float(index)] for index in range(32)]
    assert embeddings[32:64] == [None] * 32
    assert embeddings[64:] == [[float(index)] for index in range(64, 70)]
    assert "32 of 70" in warnings[0]


@pytest.mark.asyncio
async def test_openai_embedding_provider_sends_one_batch_and_restores_order():
    provider = OpenAIEmbeddingProvider({"api_key": "test", "model": "embedding"})

    class Embeddings:
        async def create(self, **kwargs):
            assert kwargs["input"] == ["zero", "one"]
            return type(
                "Response",
                (),
                {
                    "data": [
                        type("Item", (), {"index": 1, "embedding": [1.0]})(),
                        type("Item", (), {"index": 0, "embedding": [0.0]})(),
                    ]
                },
            )()

    provider.client = type("Client", (), {"embeddings": Embeddings()})()

    assert await provider.embed_many(["zero", "one"]) == [[0.0], [1.0]]


@pytest.mark.asyncio
async def test_local_http_embedding_provider_reuses_client_for_batches():
    provider = LocalHttpEmbeddingProvider({"url": "http://embedding/v1"})
    posts = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [1.0]}]}

    class Client:
        async def post(self, url, json):
            posts.append((url, json))
            return Response()

    provider.client = Client()

    assert await provider.embed_many(["hello"]) == [[1.0]]
    assert await provider.embed("again") == [1.0]
    assert len(posts) == 2


class _RecordingPipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def hset(self, key, *, mapping):
        self.commands.append(("hset", key, mapping))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    def sadd(self, key, *members):
        self.commands.append(("sadd", key, members))
        return self

    async def execute(self):
        self.redis.pipeline_executes.append(list(self.commands))
        return [1] * len(self.commands)


class _AtomicRedis:
    def __init__(self):
        self.pipeline_executes = []
        self.eval_calls = []
        self.direct_deletes = []

    def pipeline(self, transaction=False):
        assert transaction is False
        return _RecordingPipeline(self)

    async def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        return int(args[numkeys])

    async def delete(self, *keys):
        self.direct_deletes.extend(keys)
        return len(keys)


@pytest.mark.asyncio
async def test_redis_generation_stages_in_bounded_pipelines_before_atomic_cutover():
    redis = _AtomicRedis()
    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 1},
            "redis": {"pipeline_batch_size": 64},
            "indexing": {
                "control_plane_version": "v2",
                "atomic_commit_max_chunks": 2000,
                "staging_ttl_seconds": 86400,
            },
        }
    )
    store.client = redis

    count = await store.upsert_chunks(
        _chunks(1000),
        [[float(index)] for index in range(1000)],
        knowledge_base_id="school",
    )

    assert count == 1000
    assert len(redis.pipeline_executes) == 16
    assert len(redis.eval_calls) == 1
    assert redis.direct_deletes == []
    staged_hsets = [
        command
        for batch in redis.pipeline_executes
        for command in batch
        if command[0] == "hset"
    ]
    assert len(staged_hsets) == 1000
    assert all(":staging:" in command[1] for command in staged_hsets)


@pytest.mark.asyncio
async def test_redis_generation_rejects_documents_over_atomic_limit():
    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 1},
            "indexing": {"atomic_commit_max_chunks": 2},
        }
    )
    store.client = _AtomicRedis()

    with pytest.raises(ValueError, match="atomic commit limit"):
        await store.upsert_chunks(
            _chunks(3),
            [[1.0], [2.0], [3.0]],
            knowledge_base_id="school",
        )

    assert store.client.eval_calls == []


def test_rag_batch_and_indexing_config_survive_validation():
    config = PlatformConfig.model_validate(
        {
            "rag": {
                "embedding": {"batch_size": 32, "max_concurrent_batches": 2},
                "redis": {"pipeline_batch_size": 64},
                "indexing": {
                    "control_plane_version": "v2",
                    "atomic_commit_max_chunks": 2000,
                    "staging_ttl_seconds": 86400,
                },
            }
        }
    )

    assert config.rag.embedding.batch_size == 32
    assert config.rag.embedding.max_concurrent_batches == 2
    assert config.rag.redis.pipeline_batch_size == 64
    assert config.rag.indexing.control_plane_version == "v2"
    assert config.rag.indexing.atomic_commit_max_chunks == 2000
