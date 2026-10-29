"""I10: retrieval failures are told apart from "no hits"."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part2_inference import InferenceEngine, route_response_status
from src.llm_router_part3_pipeline import build_inference_completed_event
from src.rag.chunker import DocumentChunk
from src.rag.service import RagService
from src.rag.vector_store import RagSearchResult, RedisStackRagVectorStore
from src.utils.schema import RagPolicy


class _Embedder:
    async def embed(self, _text):
        return [1.0]


def _hit():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Tuition payment is due on May 1.",
        page_start=1,
        page_end=1,
        block_ids=["b1"],
        block_types=["text"],
        metadata={"title": "School Handbook"},
    )
    return RagSearchResult(
        chunk=chunk, score=0.9, match_source="hybrid", knowledge_base_id="school"
    )


class _Store:
    """Vector store double: answers, sleeps past the deadline or fails."""

    def __init__(self, *, results=None, delay=0.0, error=None):
        self.results = list(results or [])
        self.delay = delay
        self.error = error

    async def search(self, *_args, **_kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return list(self.results)


def _rag_service(store, *, deadline_seconds=1.0):
    return RagService(
        {
            "enabled": True,
            "backend": "memory",
            "retrieval": {"deadline_seconds": deadline_seconds},
        },
        embedding_provider=_Embedder(),
        vector_store=store,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store", "expected_status", "expected_hits"),
    [
        pytest.param(_Store(results=[_hit()]), "hits", 1, id="hits"),
        pytest.param(_Store(), "no_hits", 0, id="no-hits"),
        pytest.param(_Store(delay=0.5), "timeout", 0, id="timeout"),
        pytest.param(
            _Store(error=ConnectionError("redis down")), "error", 0, id="error"
        ),
    ],
)
async def test_retrieve_with_outcome_types_the_result(
    store, expected_status, expected_hits
):
    service = _rag_service(store, deadline_seconds=0.05)

    outcome = await service.retrieve_with_outcome("tuition")

    assert outcome.status == expected_status
    assert len(outcome.results) == expected_hits


class _DownRedis:
    async def execute_command(self, *_command):
        raise ConnectionError("Error 111 connecting to redis:6379")


class _EmptyRedis:
    async def execute_command(self, *_command):
        return [0]


class _NoTextFieldRedis:
    """Full-text query fails (old index), tag and vector queries answer."""

    async def execute_command(self, *command):
        if "@text:" in str(command[2]):
            raise RuntimeError("Unknown field at offset 0 near text")
        return [0]


def _redis_rag_service(client):
    store = RedisStackRagVectorStore(
        {
            "retrieval": {"deadline_seconds": 1},
            "embedding": {"dimensions": 1},
        }
    )
    store.client = client
    return _rag_service(store)


@pytest.mark.asyncio
async def test_redis_store_failure_is_an_error_not_no_hits():
    service = _redis_rag_service(_DownRedis())

    outcome = await service.retrieve_with_outcome("tuition deadline")

    assert outcome.status == "error"
    # The old entry point still answers [] for the same failure.
    assert await service.retrieve("tuition deadline") == []


@pytest.mark.asyncio
async def test_redis_store_that_answers_empty_is_no_hits():
    service = _redis_rag_service(_EmptyRedis())

    outcome = await service.retrieve_with_outcome("tuition deadline")

    assert outcome.status == "no_hits"


@pytest.mark.asyncio
async def test_redis_keyword_fallback_that_answers_is_no_hits():
    service = _redis_rag_service(_NoTextFieldRedis())

    outcome = await service.retrieve_with_outcome("tuition deadline")

    assert outcome.status == "no_hits"


def _engine(rag_service, provider_response):
    router = MagicMock()
    router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    router.token_counter.count_tokens.return_value = 5
    event_producer = AsyncMock()
    engine = InferenceEngine(
        {"cache": {"enabled": True}, "batching": {"enabled": False}},
        router,
        event_producer=event_producer,
        rag_service=rag_service,
    )
    engine.providers = {"openai": AsyncMock()}
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=provider_response
    )
    engine._get_local_fallback_models = MagicMock(return_value=[])
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()
    return engine, event_producer


_UNAVAILABLE_STORES = [
    pytest.param(lambda: _Store(delay=0.5), "timeout", id="timeout"),
    pytest.param(
        lambda: _Store(error=ConnectionError("redis down")), "error", id="error"
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_store", "status"), _UNAVAILABLE_STORES)
async def test_auto_rag_unavailable_answers_without_rag(
    sample_query_request, inference_response_factory, make_store, status
):
    engine, event_producer = _engine(
        _rag_service(make_store(), deadline_seconds=0.05),
        inference_response_factory(response_text="General answer."),
    )
    request = sample_query_request.model_copy(update={"knowledge_base_ids": ["school"]})

    response = await engine.process_query(request)

    assert response.error is None
    assert response.response_text == "General answer."
    assert "rag_unavailable" in response.warnings
    assert response.tool_calls[0].name == "rag_search"
    assert response.tool_calls[0].error == f"rag_{status}"
    assert "rag_context" not in request.metadata
    engine.providers["openai"].generate_response.assert_awaited_once()
    engine.cache.cache_response.assert_not_awaited()
    published = event_producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(request, published)
    assert event["status"] == "success"
    assert event["error_type"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_store", "status"), _UNAVAILABLE_STORES)
async def test_required_rag_unavailable_is_a_retrieval_error(
    sample_query_request, inference_response_factory, make_store, status
):
    engine, event_producer = _engine(
        _rag_service(make_store(), deadline_seconds=0.05),
        inference_response_factory(),
    )
    request = sample_query_request.model_copy(update={"rag_policy": RagPolicy.REQUIRED})

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    assert response.error_code == "retrieval_unavailable"
    assert "文件庫沒有足夠資訊" not in response.response_text
    assert route_response_status(response) == (503, {})
    engine.providers["openai"].generate_response.assert_not_awaited()
    # Neither the model nor its provider is blamed for a retrieval outage.
    engine.router.update_model_stats.assert_not_called()
    published = event_producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(request, published)
    assert event["error_type"] not in {"timeout", "provider_error", "rate_limited"}
    assert event["error_code"] == "retrieval_unavailable"


@pytest.mark.asyncio
async def test_required_rag_genuine_no_hits_keeps_the_grounded_reply(
    sample_query_request, inference_response_factory
):
    engine, _ = _engine(_rag_service(_Store()), inference_response_factory())
    request = sample_query_request.model_copy(update={"rag_policy": RagPolicy.REQUIRED})

    response = await engine.process_query(request)

    assert response.error is None
    assert response.finish_reason == "rag_no_hits"
    assert "文件庫沒有足夠資訊" in response.response_text


@pytest.mark.asyncio
async def test_auto_rag_with_legacy_retrieve_that_raises_answers_without_rag(
    sample_query_request, inference_response_factory
):
    rag_service = SimpleNamespace(
        enabled=True,
        auto_retrieve=True,
        retrieve=AsyncMock(side_effect=ConnectionError("redis down")),
        sources_from_results=MagicMock(return_value=[]),
        tool_call_from_results=MagicMock(return_value=None),
        build_context=MagicMock(return_value=""),
    )
    engine, event_producer = _engine(rag_service, inference_response_factory())
    request = sample_query_request.model_copy(update={"knowledge_base_ids": ["school"]})

    response = await engine.process_query(request)

    assert response.error is None
    assert "rag_unavailable" in response.warnings
    engine.cache.cache_response.assert_not_awaited()
