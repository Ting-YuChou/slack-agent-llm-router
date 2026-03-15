from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm_router_part2_inference import (
    AnthropicProvider,
    BatchProcessor,
    InferenceEngine,
    OpenAIProvider,
    ResponseCache,
)
from src.utils.schema import QueryRequest, UserTier


@pytest.fixture
def inference_config():
    return {
        "compression": {
            "max_context_tokens": 10,
            "compression_ratio": 0.5,
            "method": "semantic_graph",
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "max_size": "1GB",
        },
        "batching": {
            "enabled": False,
        },
    }


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_generate_response_returns_complete_inference_response(self, sample_query_request):
        with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.output_text = "Use a simple helper."
            mock_response.usage.input_tokens = 12
            mock_response.usage.output_tokens = 18
            mock_response.usage.total_tokens = 30

            mock_client = AsyncMock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider({"api_key": "test-key"})
            await provider.initialize()

            response = await provider.generate_response(sample_query_request, "gpt-5")

            assert response.response_text == "Use a simple helper."
            assert response.provider == "openai"
            assert response.total_tokens == 30
            assert response.tokens_per_second > 0
            assert response.cached is False


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_generate_response_returns_complete_inference_response(self, sample_query_request):
        with patch("src.llm_router_part2_inference.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Anthropic answer")]
            mock_response.usage.input_tokens = 9
            mock_response.usage.output_tokens = 21

            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            provider = AnthropicProvider({"api_key": "test-key"})
            await provider.initialize()

            response = await provider.generate_response(sample_query_request, "claude-3.5-sonnet")

            assert response.response_text == "Anthropic answer"
            assert response.provider == "anthropic"
            assert response.total_tokens == 30
            assert response.tokens_per_second > 0


class TestResponseCache:
    def test_generate_cache_key_changes_with_request_parameters(self):
        cache = ResponseCache({"enabled": True})
        request_a = QueryRequest(
            query="hello",
            user_id="u1",
            user_tier=UserTier.FREE,
            temperature=0.1,
            max_tokens=10,
        )
        request_b = QueryRequest(
            query="hello",
            user_id="u1",
            user_tier=UserTier.FREE,
            temperature=0.8,
            max_tokens=10,
        )

        key_a = cache.generate_cache_key(request_a, "gpt-5")
        key_b = cache.generate_cache_key(request_b, "gpt-5")

        assert key_a != key_b


class TestBatchProcessor:
    @pytest.mark.asyncio
    async def test_disabled_batching_processes_immediately(self, sample_query_request, inference_response_factory):
        provider = AsyncMock()
        provider.generate_response.return_value = inference_response_factory()

        processor = BatchProcessor({"enabled": False})
        response = await processor.add_request(sample_query_request, provider, "gpt-5")

        provider.generate_response.assert_awaited_once_with(sample_query_request, "gpt-5")
        assert response.model_name == "gpt-5"


class TestInferenceEngine:
    @pytest.mark.asyncio
    async def test_process_query_uses_provider_and_caches_result(
        self,
        inference_config,
        sample_query_request,
        inference_response_factory,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
        router.get_model_info.return_value = {"config": {"provider": "openai"}}
        router.update_model_stats = MagicMock()

        engine = InferenceEngine(inference_config, router)
        engine.providers = {"openai": AsyncMock()}
        engine.providers["openai"].generate_response = AsyncMock(return_value=inference_response_factory())
        engine.cache.get_cached_response = AsyncMock(return_value=None)
        engine.cache.cache_response = AsyncMock()
        engine.context_compressor.compress_context = AsyncMock(return_value="short ctx")

        request = sample_query_request.model_copy(update={"context": "x" * 32})
        response = await engine.process_query(request)

        assert response.response_text == "generated response"
        assert request.context == "short ctx"
        router.update_model_stats.assert_called_once()
        engine.cache.cache_response.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_query_returns_cached_response(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
        router.get_model_info.return_value = {"config": {"provider": "openai"}}
        router.update_model_stats = MagicMock()

        engine = InferenceEngine(inference_config, router)
        cached_payload = {
            "response_text": "from cache",
            "model_name": "gpt-5",
            "provider": "openai",
            "token_count_input": 3,
            "token_count_output": 7,
            "total_tokens": 10,
            "latency_ms": 5,
            "tokens_per_second": 200.0,
            "cost_usd": 0.0,
            "cached": False,
        }
        engine.cache.get_cached_response = AsyncMock(return_value=cached_payload)
        engine.cache.cache_response = AsyncMock()

        response = await engine.process_query(sample_query_request)

        assert response.cached is True
        assert response.response_text == "from cache"
        engine.cache.cache_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_query_returns_error_response_when_provider_missing(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="unknown-model"))
        router.get_model_info.return_value = None
        router.update_model_stats = MagicMock()

        engine = InferenceEngine(inference_config, router)
        engine.cache.get_cached_response = AsyncMock(return_value=None)

        response = await engine.process_query(sample_query_request)

        assert response.provider == "error"
        assert response.error is not None
        assert response.total_tokens == 0
        assert response.tokens_per_second == 0.0
