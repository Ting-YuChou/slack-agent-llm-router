import asyncio
import base64
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

import main
from src.memory import LocalHttpEmbeddingProvider, OpenAIEmbeddingProvider
from src.rag.chunker import DocumentChunk
from src.rag.service import RagPayloadTooLarge, RagService, decode_base64_document
from src.rag.vector_store import (
    ATOMIC_GENERATION_COMMIT_LUA,
    RedisStackRagVectorStore,
)
from src.utils.schema import PlatformConfig


class _HeartbeatRedis:
    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    async def set(self, key, value, *, ex):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("temporary Redis failure")
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


def test_rag_local_job_and_batch_caches_are_bounded():
    service = RagService(
        {
            "enabled": False,
            "backend": "memory",
            "local_job_cache_max_entries": 2,
            "local_batch_cache_max_entries": 1,
        },
        vector_store=_HeartbeatStore(_HeartbeatRedis()),
    )

    for index in range(5):
        service.jobs[f"job-{index}"] = object()
        service.batches[f"batch-{index}"] = object()

    assert len(service.jobs) == 2
    assert len(service.batches) == 1
    assert service.jobs.get("job-0") is None


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


def test_rag_compose_profile_enables_api_and_worker():
    result = subprocess.run(
        ["docker", "compose", "--profile", "rag", "config"],
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = yaml.safe_load(result.stdout)

    for service_name in ("api", "rag-workers"):
        environment = resolved["services"][service_name]["environment"]
        assert environment["LLM_ROUTER_RAG_ENABLED"] == "true"
        assert environment["LLM_ROUTER_RAG_QUEUE_ENABLED"] == "true"
        assert "redis-stack" in resolved["services"][service_name]["depends_on"]
    assert "healthcheck" in resolved["services"]["rag-workers"]

    base = yaml.safe_load(Path("docker-compose.yml").read_text())
    assert "LLM_ROUTER_RAG_ENABLED" not in base["services"]["api"].get(
        "environment", {}
    )


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


@pytest.mark.asyncio
async def test_worker_heartbeat_survives_a_transient_redis_failure(tmp_path):
    redis = _HeartbeatRedis(failures=1)
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

    assert redis.calls


@pytest.mark.asyncio
async def test_worker_supervisor_propagates_a_terminal_heartbeat_failure(
    tmp_path, monkeypatch
):
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "ingestion_queue": {"enabled": True, "consumer_count": 1},
            "storage": {"staging_dir": str(tmp_path)},
        },
        vector_store=_HeartbeatStore(_HeartbeatRedis()),
    )

    async def wait_forever(*_args):
        await asyncio.Event().wait()

    async def fail_heartbeat(*_args):
        raise RuntimeError("heartbeat stopped")

    monkeypatch.setattr(service, "_worker_loop", wait_forever)
    monkeypatch.setattr(service, "_worker_heartbeat_loop", fail_heartbeat)

    # Stream-group setup completes before tasks are spawned.
    async def ready():
        return None

    monkeypatch.setattr(service, "_ensure_stream_group", ready)
    with pytest.raises(RuntimeError, match="heartbeat stopped"):
        await asyncio.wait_for(service.run_ingestion_workers(), timeout=0.2)


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
async def test_embedding_batch_failure_retries_that_batch_scalar():
    class Provider(_BatchEmbeddingProvider):
        def __init__(self):
            super().__init__()
            self.scalar_calls = []

        async def embed_many(self, texts):
            if texts[0] == "32":
                raise RuntimeError("batch unavailable")
            return await super().embed_many(texts)

        async def embed(self, text):
            self.scalar_calls.append(text)
            return await super().embed(text)

    provider = Provider()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"batch_size": 32, "max_concurrent_batches": 2},
        },
        embedding_provider=provider,
    )

    embeddings, warnings = await service._embed_chunks(_chunks(70))

    assert embeddings[:32] == [[float(index)] for index in range(32)]
    assert embeddings[32:64] == [[float(index)] for index in range(32, 64)]
    assert embeddings[64:] == [[float(index)] for index in range(64, 70)]
    assert provider.scalar_calls == [str(index) for index in range(32, 64)]
    assert warnings == []


@pytest.mark.asyncio
async def test_embedding_batch_and_scalar_failure_aborts_before_indexing():
    class Provider(_BatchEmbeddingProvider):
        async def embed_many(self, texts):
            raise RuntimeError("batch unavailable")

        async def embed(self, text):
            if text == "3":
                raise RuntimeError("scalar unavailable")
            return await super().embed(text)

    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"batch_size": 8, "max_concurrent_batches": 1},
        },
        embedding_provider=Provider(),
    )

    with pytest.raises(RuntimeError, match="scalar embedding fallback failed"):
        await service._embed_chunks(_chunks(8))


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
        if self.redis.fail_on_resource and any(
            "resource:" in command[1] for command in self.commands
        ):
            raise RuntimeError("sidecar staging failed")
        self.redis.pipeline_executes.append(list(self.commands))
        return [1] * len(self.commands)


class _AtomicRedis:
    def __init__(self, eval_result=None, *, fail_on_resource=False):
        self.pipeline_executes = []
        self.eval_calls = []
        self.direct_deletes = []
        self.eval_result = eval_result
        self.fail_on_resource = fail_on_resource

    def pipeline(self, transaction=False):
        assert transaction is False
        return _RecordingPipeline(self)

    async def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        if self.eval_result is not None:
            return self.eval_result
        return int(args[numkeys])

    async def delete(self, *keys):
        self.direct_deletes.append((bool(self.eval_calls), keys))
        return len(keys)

    async def smembers(self, _key):
        return set()


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
    assert all(after_cutover for after_cutover, _keys in redis.direct_deletes)
    staged_hsets = [
        command
        for batch in redis.pipeline_executes
        for command in batch
        if command[0] == "hset"
    ]
    assert len(staged_hsets) == 1000
    assert all(":staging:" in command[1] for command in staged_hsets)


@pytest.mark.asyncio
async def test_redis_generation_stages_sidecars_and_stale_writer_does_not_mutate_live():
    redis = _AtomicRedis(eval_result=-1)
    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 1},
            "redis": {"pipeline_batch_size": 64},
            "indexing": {"control_plane_version": "v2"},
        }
    )
    store.client = redis
    chunk = _chunks(1)[0]
    chunk.metadata.update(
        {
            "table_id": "table-1",
            "table_sidecar": {"headers": ["name"]},
            "table_row_id": "row-1",
            "table_row_sidecar": {"name": "Ada"},
            "figure_id": "figure-1",
            "figure_sidecar": {"caption": "Chart"},
        }
    )

    assert await store.upsert_chunks([chunk], [[1.0]], knowledge_base_id="school") == 1

    assert len(redis.eval_calls) == 1
    assert redis.direct_deletes == []
    sidecar_hsets = [
        command
        for batch in redis.pipeline_executes
        for command in batch
        if command[0] == "hset" and "resource:" in command[1]
    ]
    assert len(sidecar_hsets) == 3
    _, numkeys, args = redis.eval_calls[0]
    eval_keys = args[:numkeys]
    assert all("{school:doc}" in key for key in eval_keys)


def test_atomic_generation_lua_renames_sidecars_before_setting_active_generation():
    sidecar_rename = ATOMIC_GENERATION_COMMIT_LUA.index(
        "redis.call('RENAME', KEYS[key_offset], KEYS[key_offset + 1])"
    )
    active_switch = ATOMIC_GENERATION_COMMIT_LUA.index(
        "redis.call('SET', active_generation, generation)"
    )

    assert sidecar_rename < active_switch


@pytest.mark.asyncio
async def test_sidecar_staging_failure_aborts_before_generation_cutover():
    redis = _AtomicRedis(fail_on_resource=True)
    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 1},
            "indexing": {"control_plane_version": "v2"},
        }
    )
    store.client = redis
    chunk = _chunks(1)[0]
    chunk.metadata.update(
        {"table_id": "table-1", "table_sidecar": {"headers": ["name"]}}
    )

    with pytest.raises(RuntimeError, match="sidecar staging failed"):
        await store.upsert_chunks([chunk], [[1.0]], knowledge_base_id="school")

    assert redis.eval_calls == []
    assert redis.direct_deletes == []


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


@pytest.mark.asyncio
async def test_query_text_and_visual_embeddings_start_concurrently():
    both_started = asyncio.Event()
    started = set()

    async def mark_started(name, value):
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)
        return value

    class TextProvider:
        async def embed(self, _text):
            return await mark_started("text", [1.0])

    class VisualProvider:
        async def embed_query(self, _text):
            return await mark_started("visual", [1.0])

    class Store:
        def __init__(self):
            self.embedding = None
            self.visual_embedding = None

        async def search(self, _query, embedding, **kwargs):
            self.embedding = embedding
            self.visual_embedding = kwargs["visual_embedding"]
            return []

    store = Store()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "visual": {"enabled": True, "embedding": {"enabled": True}},
        },
        embedding_provider=TextProvider(),
        visual_processor=VisualProvider(),
        vector_store=store,
    )

    assert await service.retrieve("tuition") == []
    assert started == {"text", "visual"}
    assert store.embedding == [1.0]
    assert store.visual_embedding == [1.0]


@pytest.mark.asyncio
async def test_retrieval_deadline_covers_embeddings_search_and_rerank():
    class Provider:
        async def embed(self, _text):
            await asyncio.sleep(0.03)
            return [1.0]

    class Store:
        async def search(self, *_args, **_kwargs):
            await asyncio.sleep(0.03)
            return []

    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "retrieval": {"deadline_seconds": 0.04},
        },
        embedding_provider=Provider(),
        vector_store=Store(),
    )

    started = time.monotonic()
    assert await service.retrieve("tuition") == []
    assert time.monotonic() - started < 0.07


@pytest.mark.asyncio
async def test_redis_retrieval_branches_run_concurrently_and_degrade_independently():
    store = RedisStackRagVectorStore(
        {
            "retrieval": {"max_concurrent_branches": 3, "deadline_seconds": 1},
            "embedding": {"dimensions": 1},
        }
    )
    store.client = object()
    all_started = asyncio.Event()
    started = set()

    async def branch(name, *, fail=False):
        started.add(name)
        if len(started) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=0.1)
        if fail:
            raise RuntimeError("branch failed")
        return []

    store._keyword_search = lambda *_args: branch("keyword")
    store._vector_search = lambda *_args: branch("vector", fail=True)
    store._visual_search = lambda *_args: branch("visual")

    results = await store.search(
        "tuition",
        [1.0],
        knowledge_base_ids=[],
        limit=5,
        candidate_count=30,
        keyword_weight=0.35,
        vector_weight=0.6,
        recency_weight=0.05,
        min_score=0.0,
        visual_embedding=[1.0],
        visual_weight=0.4,
    )

    assert results == []
    assert started == {"keyword", "vector", "visual"}


@pytest.mark.asyncio
async def test_redis_search_commands_do_not_return_embedding_payloads():
    class SearchRedis:
        def __init__(self):
            self.commands = []

        async def execute_command(self, *command):
            self.commands.append(command)
            return [0]

    store = RedisStackRagVectorStore({"embedding": {"dimensions": 1}})
    store.client = SearchRedis()

    await store._search("*", 30, "keyword")
    await store._vector_search([1.0], [], 30)

    assert len(store.client.commands) == 2
    assert all("embedding" not in command for command in store.client.commands)


class _StreamingUpload:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.read_sizes = []

    async def read(self, size):
        self.read_sizes.append(size)
        return self.chunks.pop(0) if self.chunks else b""


@pytest.mark.asyncio
async def test_multipart_upload_streams_to_atomic_staging_file(tmp_path):
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {"staging_dir": str(tmp_path)},
            "upload": {
                "multipart_max_bytes": 100,
                "stream_chunk_bytes": 4,
            },
        }
    )
    upload = _StreamingUpload([b"abcd", b"ef", b""])

    storage_ref = await service.stage_uploaded_file(
        upload,
        filename="handbook.pdf",
        knowledge_base_id="school",
        document_id="doc-1",
    )

    assert Path(storage_ref).read_bytes() == b"abcdef"
    assert upload.read_sizes == [4, 4, 4]
    assert list(tmp_path.rglob("*.part")) == []


@pytest.mark.asyncio
async def test_multipart_upload_overflow_removes_partial_file(tmp_path):
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {"staging_dir": str(tmp_path)},
            "upload": {"multipart_max_bytes": 5, "stream_chunk_bytes": 4},
        }
    )

    with pytest.raises(RagPayloadTooLarge):
        await service.stage_uploaded_file(
            _StreamingUpload([b"abcd", b"ef"]),
            filename="handbook.pdf",
            knowledge_base_id="school",
            document_id="doc-1",
        )

    assert list(tmp_path.rglob("*.part")) == []
    assert list(tmp_path.rglob("*.pdf")) == []


@pytest.mark.asyncio
async def test_multipart_upload_cancellation_during_publish_removes_new_target(
    tmp_path, monkeypatch
):
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "storage": {"staging_dir": str(tmp_path)},
        }
    )
    replace_started = threading.Event()
    allow_replace = threading.Event()
    real_replace = __import__("os").replace

    def blocked_replace(source, target):
        replace_started.set()
        assert allow_replace.wait(timeout=1)
        real_replace(source, target)

    monkeypatch.setattr("src.rag.service.os.replace", blocked_replace)
    task = asyncio.create_task(
        service.stage_uploaded_file(
            _StreamingUpload([b"contents", b""]),
            filename="handbook.pdf",
            knowledge_base_id="school",
            document_id="doc-1",
        )
    )
    assert await asyncio.to_thread(replace_started.wait, 1)
    task.cancel()
    allow_replace.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert list(tmp_path.rglob("*.part")) == []
    assert list(tmp_path.rglob("*.pdf")) == []


def test_json_base64_limit_checks_decoded_size():
    encoded = base64.b64encode(b"12345").decode("ascii")

    with pytest.raises(RagPayloadTooLarge):
        decode_base64_document(encoded, max_decoded_bytes=4)

    assert decode_base64_document(encoded, max_decoded_bytes=5) == b"12345"


@pytest.mark.asyncio
async def test_batch_json_documents_enforce_decoded_upload_limit_before_creation():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "ingestion_queue": {"enabled": True},
            "upload": {"json_max_decoded_bytes": 4},
        }
    )

    with pytest.raises(RagPayloadTooLarge):
        await service.create_batch(
            documents=[
                {
                    "filename": "large.pdf",
                    "content_base64": base64.b64encode(b"12345").decode("ascii"),
                }
            ]
        )

    assert service.batches == {}


def test_rag_upload_config_survives_validation():
    config = PlatformConfig.model_validate(
        {
            "rag": {
                "upload": {
                    "json_max_decoded_bytes": 10_000_000,
                    "multipart_max_bytes": 100_000_000,
                    "stream_chunk_bytes": 1_048_576,
                }
            }
        }
    )

    assert config.rag.upload.json_max_decoded_bytes == 10_000_000
    assert config.rag.upload.multipart_max_bytes == 100_000_000
