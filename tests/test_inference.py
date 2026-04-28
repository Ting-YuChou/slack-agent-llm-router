import asyncio
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from src.llm_router_part2_inference import (
    AnthropicProvider,
    BatchProcessor,
    InferenceEngine,
    OpenAIProvider,
    ResponseCache,
    vLLMProvider,
)
from src.utils.schema import Attachment, AttachmentType, QueryRequest, UserTier


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
    async def test_generate_response_returns_complete_inference_response(
        self, sample_query_request
    ):
        with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.output_text = "Use a simple helper."
            mock_response.usage.input_tokens = 12
            mock_response.usage.output_tokens = 18
            mock_response.usage.total_tokens = 30
            mock_response.usage.prompt_tokens_details.cached_tokens = 4

            mock_client = AsyncMock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider({"api_key": "test-key"})
            await provider.initialize()

            response = await provider.generate_response(sample_query_request, "gpt-5")

            assert response.response_text == "Use a simple helper."
            assert response.provider == "openai"
            assert response.total_tokens == 30
            assert response.provider_cached_input_tokens == 4
            assert response.tokens_per_second > 0
            assert response.cached is False

    @pytest.mark.asyncio
    async def test_generate_response_sends_openai_prompt_cache_hints(
        self, sample_query_request
    ):
        with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.output_text = "Cached prefix answer"
            mock_response.usage.input_tokens = 1200
            mock_response.usage.output_tokens = 20
            mock_response.usage.total_tokens = 1220
            mock_response.usage.prompt_tokens_details.cached_tokens = 1024

            mock_client = AsyncMock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(
                {
                    "api_key": "test-key",
                    "prompt_cache": {
                        "enabled": True,
                        "key_strategy": "conversation",
                        "retention": "24h",
                    },
                }
            )
            await provider.initialize()

            response = await provider.generate_response(sample_query_request, "gpt-5")

            kwargs = mock_client.responses.create.await_args.kwargs
            assert "prompt_cache_key" in kwargs
            assert kwargs["prompt_cache_retention"] == "24h"
            assert "user" not in kwargs
            assert response.provider_cached_input_tokens == 1024

    @pytest.mark.asyncio
    async def test_generate_response_includes_attachment_text_in_prompt(
        self, sample_query_request
    ):
        with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.output_text = "Used attachment content"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 10
            mock_response.usage.total_tokens = 20

            mock_client = AsyncMock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider({"api_key": "test-key"})
            await provider.initialize()

            request = sample_query_request.model_copy(
                update={
                    "attachments": [
                        Attachment(
                            name="report.csv",
                            type=AttachmentType.DOCUMENT,
                            size_bytes=11,
                            mime_type="text/csv",
                            content=b"a,b\n1,2\n",
                        )
                    ]
                }
            )

            await provider.generate_response(request, "gpt-5")

            prompt = mock_client.responses.create.await_args.kwargs["input"]
            assert "report.csv" in prompt
            assert "a,b\n1,2" in prompt

    @pytest.mark.asyncio
    async def test_generate_response_includes_response_style_instructions(
        self, sample_query_request
    ):
        with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.output_text = "Styled answer"
            mock_response.usage.input_tokens = 8
            mock_response.usage.output_tokens = 12
            mock_response.usage.total_tokens = 20

            mock_client = AsyncMock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider({"api_key": "test-key"})
            await provider.initialize()

            request = sample_query_request.model_copy(
                update={
                    "metadata": {
                        "response_style_instructions": "Use precise technical terminology."
                    }
                }
            )

            await provider.generate_response(request, "gpt-5")

            prompt = mock_client.responses.create.await_args.kwargs["input"]
            assert "Response style instructions" in prompt
            assert "Use precise technical terminology." in prompt


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_generate_response_returns_complete_inference_response(
        self, sample_query_request
    ):
        with patch(
            "src.llm_router_part2_inference.anthropic.AsyncAnthropic"
        ) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Anthropic answer")]
            mock_response.usage.input_tokens = 9
            mock_response.usage.output_tokens = 21

            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            provider = AnthropicProvider({"api_key": "test-key"})
            await provider.initialize()

            response = await provider.generate_response(
                sample_query_request, "claude-sonnet-4-6"
            )

            assert response.response_text == "Anthropic answer"
            assert response.provider == "anthropic"
            assert response.total_tokens == 30
            assert response.tokens_per_second > 0

    @pytest.mark.asyncio
    async def test_generate_response_uses_anthropic_cache_control(
        self, sample_query_request
    ):
        with patch(
            "src.llm_router_part2_inference.anthropic.AsyncAnthropic"
        ) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Anthropic cached answer")]
            mock_response.usage.input_tokens = 9
            mock_response.usage.output_tokens = 21
            mock_response.usage.cache_creation_input_tokens = 5
            mock_response.usage.cache_read_input_tokens = 7

            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            provider = AnthropicProvider(
                {
                    "api_key": "test-key",
                    "prompt_cache": {
                        "enabled": True,
                        "cache_control": {"type": "ephemeral"},
                    },
                }
            )
            await provider.initialize()

            response = await provider.generate_response(
                sample_query_request, "claude-sonnet-4-6"
            )

            messages = mock_client.messages.create.await_args.kwargs["messages"]
            content = messages[0]["content"]
            assert content[0]["cache_control"] == {"type": "ephemeral"}
            assert "Previous discussion context" in content[0]["text"]
            assert content[1]["text"].startswith("Query:")
            assert response.token_count_input == 21
            assert response.provider_cache_creation_input_tokens == 5
            assert response.provider_cache_read_input_tokens == 7


class TestVLLMProvider:
    def test_uses_explicit_base_url_when_configured(self):
        provider = vLLMProvider({"base_url": "http://127.0.0.1:8001/"})

        assert provider.base_url == "http://127.0.0.1:8001"

    def test_falls_back_to_host_and_port_when_base_url_missing(self):
        provider = vLLMProvider({"host": "vllm.internal", "port": 9000})

        assert provider.base_url == "http://vllm.internal:9000"


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

    def test_generate_cache_key_can_use_shared_scope(self):
        cache = ResponseCache({"enabled": True, "scope": "shared"})
        request_a = QueryRequest(query="hello", user_id="u1", user_tier=UserTier.FREE)
        request_b = QueryRequest(query="hello", user_id="u2", user_tier=UserTier.FREE)

        assert cache.generate_cache_key(request_a, "gpt-5") == cache.generate_cache_key(
            request_b, "gpt-5"
        )

    @pytest.mark.asyncio
    async def test_initialize_uses_configured_redis_connection(self):
        redis_client = AsyncMock()
        redis_client.ping = AsyncMock(return_value=True)

        with patch(
            "src.llm_router_part2_inference.redis.Redis", return_value=redis_client
        ) as mock_redis:
            cache = ResponseCache(
                {
                    "enabled": True,
                    "backend": "redis",
                    "redis": {
                        "host": "redis",
                        "port": 6380,
                        "db": 4,
                        "password_env": "REDIS_PASSWORD",
                    },
                }
            )

            with patch.dict("os.environ", {"REDIS_PASSWORD": "secret"}, clear=False):
                await cache.initialize()

            mock_redis.assert_called_once_with(
                host="redis",
                port=6380,
                db=4,
                password="secret",
                decode_responses=True,
            )
            assert cache.enabled is True


class TestBatchProcessor:
    @pytest.mark.asyncio
    async def test_disabled_batching_processes_immediately(
        self, sample_query_request, inference_response_factory
    ):
        provider = AsyncMock()
        provider.generate_response.return_value = inference_response_factory()

        processor = BatchProcessor({"enabled": False})
        response = await processor.add_request(sample_query_request, provider, "gpt-5")

        provider.generate_response.assert_awaited_once_with(
            sample_query_request, "gpt-5"
        )
        assert response.model_name == "gpt-5"

    @pytest.mark.asyncio
    async def test_enabled_batching_coalesces_identical_requests(
        self, sample_query_request, inference_response_factory
    ):
        provider = AsyncMock()
        request_started = asyncio.Event()
        release_request = asyncio.Event()

        async def generate_response(_request, _model_name):
            request_started.set()
            await release_request.wait()
            return inference_response_factory(response_text="coalesced")

        provider.generate_response.side_effect = generate_response

        processor = BatchProcessor(
            {"enabled": True, "max_batch_size": 8, "max_wait_time_ms": 1}
        )

        first_task = asyncio.create_task(
            processor.add_request(sample_query_request.model_copy(), provider, "gpt-5")
        )
        await request_started.wait()
        second_task = asyncio.create_task(
            processor.add_request(sample_query_request.model_copy(), provider, "gpt-5")
        )
        release_request.set()

        response_a, response_b = await asyncio.gather(first_task, second_task)

        assert provider.generate_response.await_count == 1
        assert response_a.response_text == "coalesced"
        assert response_b.response_text == "coalesced"
        assert response_a is not response_b

    @pytest.mark.asyncio
    async def test_enabled_batching_dispatches_distinct_requests_as_batch(
        self, sample_query_request, inference_response_factory
    ):
        provider = AsyncMock()
        provider.generate_batch_responses = AsyncMock(
            return_value=[
                inference_response_factory(response_text="first"),
                inference_response_factory(response_text="second"),
            ]
        )

        processor = BatchProcessor(
            {"enabled": True, "max_batch_size": 2, "max_wait_time_ms": 50}
        )
        request_b = sample_query_request.model_copy(
            update={"query": "Write a Rust function to add two numbers"}
        )

        response_a, response_b = await asyncio.gather(
            processor.add_request(sample_query_request, provider, "gpt-5"),
            processor.add_request(request_b, provider, "gpt-5"),
        )

        provider.generate_batch_responses.assert_awaited_once()
        provider.generate_response.assert_not_called()
        assert response_a.response_text == "first"
        assert response_b.response_text == "second"


class TestResponseCache:
    def test_cache_key_ignores_memory_analytics_metadata(self, sample_query_request):
        cache = ResponseCache({"enabled": True})
        base_request = sample_query_request.model_copy(
            update={
                "metadata": {
                    "source": "slack",
                    "memory_hit_count": 1,
                    "memory_match_sources": ["hybrid"],
                }
            }
        )
        equivalent_request = sample_query_request.model_copy(
            update={
                "metadata": {
                    "source": "slack",
                    "memory_hit_count": 2,
                    "memory_match_sources": ["keyword"],
                }
            }
        )

        assert cache.generate_cache_key(base_request, "gpt-5") == cache.generate_cache_key(
            equivalent_request, "gpt-5"
        )


class TestInferenceEngine:
    @pytest.mark.asyncio
    async def test_initialize_skips_vllm_when_endpoint_not_configured(self):
        router = MagicMock()
        engine = InferenceEngine({"vllm": {}}, router)
        engine.context_compressor.initialize = AsyncMock()
        engine.cache.initialize = AsyncMock()

        await engine.initialize()

        assert "vllm" not in engine.providers

    @pytest.mark.asyncio
    async def test_process_query_uses_provider_and_caches_result(
        self,
        inference_config,
        sample_query_request,
        inference_response_factory,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(selected_model="gpt-5")
        )
        router.get_model_info.return_value = {"config": {"provider": "openai"}}
        router.update_model_stats = MagicMock()

        engine = InferenceEngine(inference_config, router)
        engine.providers = {"openai": AsyncMock()}
        engine.providers["openai"].generate_response = AsyncMock(
            return_value=inference_response_factory()
        )
        engine.cache.get_cached_response = AsyncMock(return_value=None)
        engine.cache.cache_response = AsyncMock()
        engine.context_compressor.compress_context = AsyncMock(return_value="short ctx")

        request = sample_query_request.model_copy(update={"context": "x" * 32})
        response = await engine.process_query(request)

        assert response.response_text == "generated response"
        assert request.context == "short ctx"
        router.update_model_stats.assert_called_once()
        engine.cache.cache_response.assert_awaited_once()
        cached_payload = engine.cache.cache_response.await_args.args[1]
        assert isinstance(cached_payload["timestamp"], str)

    @pytest.mark.asyncio
    async def test_process_query_publishes_inference_completed_event(
        self,
        inference_config,
        sample_query_request,
        inference_response_factory,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(
                selected_model="gpt-5",
                query_type=SimpleNamespace(value="analysis"),
                routing_reason="Rule-based selection",
            )
        )
        router.get_model_info.return_value = {"config": {"provider": "openai"}}
        router.update_model_stats = MagicMock()

        event_producer = AsyncMock()
        event_producer.produce_inference_completed = AsyncMock()
        event_producer.produce_query_log = AsyncMock()

        engine = InferenceEngine(
            inference_config,
            router,
            event_producer=event_producer,
        )
        engine.providers = {"openai": AsyncMock()}
        engine.providers["openai"].generate_response = AsyncMock(
            return_value=inference_response_factory()
        )
        engine.cache.get_cached_response = AsyncMock(return_value=None)
        engine.cache.cache_response = AsyncMock()

        await engine.process_query(sample_query_request)

        event_producer.produce_inference_completed.assert_awaited_once()
        event_producer.produce_query_log.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_query_returns_cached_response(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(selected_model="gpt-5")
        )
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
    async def test_process_query_checks_cache_after_context_compression(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(selected_model="gpt-5")
        )
        router.get_model_info.return_value = {"config": {"provider": "openai"}}

        engine = InferenceEngine(inference_config, router)
        engine.context_compressor.compress_context = AsyncMock(return_value="short ctx")
        seen_contexts = []
        engine.cache.generate_cache_key = MagicMock(
            side_effect=lambda request, _model: seen_contexts.append(request.context)
            or "compressed-key"
        )
        engine.cache.get_cached_response = AsyncMock(
            return_value={
                "response_text": "from compressed cache",
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
        )

        request = sample_query_request.model_copy(update={"context": "x" * 32})
        response = await engine.process_query(request)

        assert response.cached is True
        assert seen_contexts == ["short ctx"]

    @pytest.mark.asyncio
    async def test_stream_query_serves_cached_response(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(selected_model="gpt-5")
        )
        router.get_model_info.return_value = {"config": {"provider": "openai"}}

        engine = InferenceEngine(
            {**inference_config, "stream_cache_chunk_chars": 4},
            router,
        )
        provider = AsyncMock()
        engine.providers = {"openai": provider}
        engine.cache.get_cached_response = AsyncMock(
            return_value={
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
        )

        chunks = [chunk async for chunk in engine.stream_query(sample_query_request)]

        assert "".join(chunks) == "from cache"
        provider.stream_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_query_publishes_error_completion_event_when_provider_missing(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(selected_model="unknown-model")
        )
        router.get_model_info.return_value = None
        router.update_model_stats = MagicMock()

        event_producer = AsyncMock()
        event_producer.produce_inference_completed = AsyncMock()
        event_producer.produce_query_log = AsyncMock()

        engine = InferenceEngine(
            inference_config,
            router,
            event_producer=event_producer,
        )
        engine.cache.get_cached_response = AsyncMock(return_value=None)

        response = await engine.process_query(sample_query_request)

        assert response.provider == "error"
        event_producer.produce_inference_completed.assert_awaited_once()
        event_producer.produce_query_log.assert_awaited_once()
        published_response = event_producer.produce_inference_completed.await_args.args[
            1
        ]
        assert published_response.provider == "error"

    @pytest.mark.asyncio
    async def test_process_query_returns_error_response_when_provider_missing(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(selected_model="unknown-model")
        )
        router.get_model_info.return_value = None
        router.update_model_stats = MagicMock()

        engine = InferenceEngine(inference_config, router)
        engine.cache.get_cached_response = AsyncMock(return_value=None)

        response = await engine.process_query(sample_query_request)

        assert response.provider == "error"
        assert response.error is not None
        assert response.total_tokens == 0
        assert response.tokens_per_second == 0.0

    @pytest.mark.asyncio
    async def test_process_query_falls_back_to_local_model_when_cloud_provider_fails(
        self,
        inference_config,
        sample_query_request,
        inference_response_factory,
    ):
        from src.llm_router_part1_router import ModelRouter
        from src.utils.schema import QueryType

        router = ModelRouter(
            {
                "default_model": "gpt-5",
                "routing_strategy": "intelligent",
                "models": {
                    "gpt-5": {
                        "provider": "openai",
                        "max_tokens": 128000,
                        "cost_per_token": 0.00003,
                        "priority": 2,
                        "capabilities": ["general", "reasoning", "coding", "analysis"],
                        "api_key_env": "OPENAI_API_KEY",
                    },
                    "mistral-7b": {
                        "provider": "vllm",
                        "model_path": "/models/mistral",
                        "max_tokens": 4096,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "coding"],
                        "gpu_memory_gb": 16,
                    },
                },
                "routing_rules": [],
            }
        )
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(
                selected_model="gpt-5",
                query_type=QueryType.GENERAL,
                token_count=64,
                routing_reason="Capability-based selection",
            )
        )
        router.update_model_stats = MagicMock()

        engine = InferenceEngine(inference_config, router)
        engine.providers = {
            "openai": AsyncMock(),
            "vllm": AsyncMock(),
        }
        engine.providers["openai"].generate_response = AsyncMock(
            side_effect=RuntimeError("upstream overloaded")
        )
        engine.providers["vllm"].generate_response = AsyncMock(
            return_value=inference_response_factory(
                model_name="mistral-7b",
                provider="vllm",
                cost_usd=0.0,
            )
        )
        engine.cache.get_cached_response = AsyncMock(side_effect=[None, None])
        engine.cache.cache_response = AsyncMock()

        response = await engine.process_query(sample_query_request)

        assert response.model_name == "mistral-7b"
        assert response.provider == "vllm"
        router.update_model_stats.assert_any_call(
            model_name="gpt-5",
            success=False,
            latency_ms=ANY,
        )
        router.update_model_stats.assert_any_call(
            model_name="mistral-7b",
            success=True,
            latency_ms=response.latency_ms,
        )
