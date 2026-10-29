"""I10 x I5: retrieval steps get a share of the request deadline, not all of it.

Before this, only the request deadline (``_within_request_deadline``) cut a
slow RAG store or web search off, from outside: the typed retrieval
outcomes never ran and every case ended as a 504.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.llm_router_part2_inference import (
    InferenceEngine,
    OpenAIProvider,
    route_response_status,
)
from src.llm_router_part3_pipeline import build_inference_completed_event
from src.rag.service import RagService
from src.tools.web_search import WebSearchTool
from src.utils.schema import (
    InferenceConfig,
    InferenceResponse,
    QueryRequest,
    QueryType,
    RagPolicy,
    ToolPolicy,
)

_PROVIDER_BLAMING_ERROR_TYPES = {"timeout", "provider_error", "rate_limited"}


class _Embedder:
    async def embed(self, _text):
        return [1.0]


class _SlowStore:
    async def search(self, *_args, **_kwargs):
        await asyncio.sleep(5)
        return []


class _SlowSearchBackend:
    provider_name = "tavily"

    async def search(self, query, options=None):
        await asyncio.sleep(5)
        return []


def _slow_rag_service():
    # The store's own deadline (30 s) is far beyond the request deadline,
    # so only the retrieval share of the request deadline can stop it.
    return RagService(
        {
            "enabled": True,
            "backend": "memory",
            "retrieval": {"deadline_seconds": 30},
        },
        embedding_provider=_Embedder(),
        vector_store=_SlowStore(),
    )


def _answer():
    return InferenceResponse(
        response_text="General answer.",
        model_name="gpt-5",
        provider="openai",
        token_count_input=1,
        token_count_output=1,
        total_tokens=2,
        latency_ms=1,
        tokens_per_second=1.0,
        cost_usd=0.0,
    )


def _engine(*, rag_service=None, web_search_tool=None, query_type=None, scheduler=None):
    router = MagicMock()
    router.route_query = AsyncMock(
        return_value=SimpleNamespace(selected_model="gpt-5", query_type=query_type)
    )
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    router.token_counter.count_tokens.return_value = 5
    engine = InferenceEngine(
        {
            "scheduler": scheduler or {"request_deadline_seconds": 1.0},
            "cache": {"enabled": True},
            "batching": {"enabled": False},
        },
        router,
        event_producer=AsyncMock(),
        rag_service=rag_service,
    )
    engine.web_search_tool = web_search_tool
    engine.providers = {
        "openai": SimpleNamespace(generate_response=AsyncMock(return_value=_answer()))
    }
    engine._get_local_fallback_models = MagicMock(return_value=[])
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()
    return engine


async def _process(engine, request):
    started = time.monotonic()
    response = await asyncio.wait_for(engine.process_query(request), timeout=3)
    return response, time.monotonic() - started


def _published_event(engine, request):
    published = engine.event_producer.produce_inference_completed.await_args.args[1]
    return build_inference_completed_event(request, published)


def _model_failures(engine):
    return [
        call
        for call in engine.router.update_model_stats.call_args_list
        if call.kwargs.get("success") is False
    ]


def _request(**overrides):
    payload = {"query": "When is tuition due?", "user_id": "u1"}
    payload.update(overrides)
    return QueryRequest(**payload)


@pytest.mark.asyncio
async def test_slow_rag_under_auto_answers_without_documents():
    engine = _engine(rag_service=_slow_rag_service())
    request = _request(knowledge_base_ids=["school"])

    response, elapsed = await _process(engine, request)

    assert response.error is None
    assert response.response_text == "General answer."
    assert "rag_unavailable" in response.warnings
    # Half of the 1 s deadline went to retrieval; the model got the rest.
    assert elapsed < 0.9
    engine.providers["openai"].generate_response.assert_awaited_once()
    engine.cache.cache_response.assert_not_awaited()
    assert _model_failures(engine) == []
    assert _published_event(engine, request)["error_type"] is None


@pytest.mark.asyncio
async def test_slow_rag_under_required_is_retrieval_unavailable():
    engine = _engine(rag_service=_slow_rag_service())
    request = _request(rag_policy=RagPolicy.REQUIRED)

    response, elapsed = await _process(engine, request)

    assert response.error == "inference_failed"
    assert response.error_code == "retrieval_unavailable"
    assert route_response_status(response) == (503, {})
    assert elapsed < 0.9
    engine.providers["openai"].generate_response.assert_not_awaited()
    assert _model_failures(engine) == []
    event = _published_event(engine, request)
    assert event["error_type"] not in _PROVIDER_BLAMING_ERROR_TYPES
    assert event["error_code"] == "retrieval_unavailable"


@pytest.mark.asyncio
async def test_slow_web_search_under_auto_warns_and_answers():
    tool = WebSearchTool(
        {"enabled": True, "cache_ttl_seconds": 0}, backend=_SlowSearchBackend()
    )
    engine = _engine(web_search_tool=tool, query_type=QueryType.WEB_RESEARCH)
    request = _request(tool_policy=ToolPolicy.AUTO)

    response, elapsed = await _process(engine, request)

    assert response.error is None
    assert "web_search_unavailable" in response.warnings
    assert elapsed < 0.9
    sent = engine.providers["openai"].generate_response.await_args.args[0]
    assert "cannot verify the latest information" in OpenAIProvider({})._build_prompt(
        sent
    )
    engine.cache.cache_response.assert_not_awaited()
    assert _model_failures(engine) == []
    assert _published_event(engine, request)["error_type"] is None


def test_retrieval_budget_ratio_defaults_to_half_and_is_validated():
    assert InferenceConfig().retrieval_budget_ratio == 0.5
    assert InferenceConfig(retrieval_budget_ratio=0.25).retrieval_budget_ratio == 0.25
    for invalid in (0, 1.5, -0.1):
        with pytest.raises(ValidationError):
            InferenceConfig(retrieval_budget_ratio=invalid)


@pytest.mark.asyncio
async def test_retrieval_budget_ratio_is_configurable():
    engine = _engine(rag_service=_slow_rag_service())
    engine.config["retrieval_budget_ratio"] = 0.2
    request = _request(knowledge_base_ids=["school"])

    response, elapsed = await _process(engine, request)

    assert "rag_unavailable" in response.warnings
    assert elapsed < 0.5
