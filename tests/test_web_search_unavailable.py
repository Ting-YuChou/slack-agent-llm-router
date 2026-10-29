"""I11: a failed live web search is visible to the caller and the model."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part2_inference import InferenceEngine, OpenAIProvider
from src.tools.web_search import WebSearchResult, WebSearchTool
from src.utils.schema import QueryType, ResponseSource, ToolPolicy

DISCLAIMER = "cannot verify the latest information"


class _FailingBackend:
    provider_name = "tavily"

    async def search(self, query, options=None):
        raise RuntimeError("Tavily returned HTTP 502")


class _EmptyBackend:
    provider_name = "tavily"

    async def search(self, query, options=None):
        return []


class _Backend:
    provider_name = "tavily"

    async def search(self, query, options=None):
        return [
            ResponseSource(
                title="Example",
                url="https://example.com/news",
                snippet="Fresh context",
                source_domain="example.com",
                rank=1,
            )
        ]


def _tool(backend):
    return WebSearchTool({"enabled": True, "cache_ttl_seconds": 0}, backend=backend)


@pytest.mark.asyncio
async def test_backend_failure_result_carries_a_warning():
    result = await _tool(_FailingBackend()).search("latest AI news", user_id="u1")

    assert result.warnings == ["web_search_unavailable"]
    assert result.blocked_reason == "web_search_error"


@pytest.mark.asyncio
async def test_successful_search_has_no_warnings():
    result = await _tool(_Backend()).search("latest AI news", user_id="u1")

    assert result.warnings == []


def test_web_search_result_defaults_to_no_warnings():
    assert WebSearchResult().warnings == []


def _engine(web_search_tool, provider_response):
    router = MagicMock()
    router.route_query = AsyncMock(
        return_value=SimpleNamespace(
            selected_model="gpt-5", query_type=QueryType.WEB_RESEARCH
        )
    )
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    engine = InferenceEngine(
        {"cache": {"enabled": True}, "batching": {"enabled": False}}, router
    )
    engine.web_search_tool = web_search_tool
    engine.providers = {"openai": AsyncMock()}
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=provider_response
    )
    engine._get_local_fallback_models = MagicMock(return_value=[])
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()
    return engine


def _prompt_sent(engine):
    request = engine.providers["openai"].generate_response.await_args.args[0]
    return OpenAIProvider({})._build_prompt(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_policy", [ToolPolicy.AUTO, ToolPolicy.REQUIRED], ids=["auto", "required"]
)
async def test_backend_failure_warns_and_tells_the_model(
    sample_query_request, inference_response_factory, tool_policy
):
    engine = _engine(_tool(_FailingBackend()), inference_response_factory())
    request = sample_query_request.model_copy(update={"tool_policy": tool_policy})

    response = await engine.process_query(request)

    assert response.error is None
    assert "web_search_unavailable" in response.warnings
    assert DISCLAIMER in _prompt_sent(engine)
    engine.cache.cache_response.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "web_search_tool",
    [
        pytest.param(None, id="not-configured"),
        pytest.param(_tool(_EmptyBackend()), id="no-results"),
    ],
)
async def test_required_search_without_results_gets_the_disclaimer(
    sample_query_request, inference_response_factory, web_search_tool
):
    engine = _engine(web_search_tool, inference_response_factory())
    request = sample_query_request.model_copy(
        update={"tool_policy": ToolPolicy.REQUIRED, "allowed_tools": ["web_search"]}
    )

    await engine.process_query(request)

    assert DISCLAIMER in _prompt_sent(engine)
    engine.cache.cache_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_search_has_no_disclaimer(
    sample_query_request, inference_response_factory
):
    engine = _engine(_tool(_Backend()), inference_response_factory())
    request = sample_query_request.model_copy(
        update={"tool_policy": ToolPolicy.REQUIRED}
    )

    response = await engine.process_query(request)

    prompt = _prompt_sent(engine)
    assert "Fresh context" in prompt
    assert DISCLAIMER not in prompt
    assert "web_search_unavailable" not in response.warnings
