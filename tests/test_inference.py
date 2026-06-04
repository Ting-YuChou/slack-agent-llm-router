import asyncio
import time
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
from src.utils.schema import (
    Attachment,
    AttachmentType,
    QueryRequest,
    QueryType,
    RagPolicy,
    ResponseSource,
    ToolCall,
    ToolPolicy,
    UserTier,
)
from src.rag.chunker import DocumentChunk
from src.rag.vector_store import RagSearchResult


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
            assert "prompt_cache_retention" not in kwargs
            assert "temperature" not in kwargs
            assert kwargs["reasoning"] == {"effort": "minimal"}
            assert "user" not in kwargs
            assert response.provider_cached_input_tokens == 1024

    def test_openai_request_kwargs_include_temperature_for_supported_models(
        self, sample_query_request
    ):
        provider = OpenAIProvider({"api_key": "test-key"})

        kwargs = provider._build_response_request_kwargs(
            sample_query_request, "gpt-4.1", "Prompt"
        )

        assert kwargs["temperature"] == sample_query_request.temperature
        assert "reasoning" not in kwargs

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
    def _mock_vllm_response(self, text="ok", prompt_tokens=3, completion_tokens=2):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"text": text}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return mock_response

    def test_uses_explicit_base_url_when_configured(self):
        provider = vLLMProvider({"base_url": "http://127.0.0.1:8001/"})

        assert provider.base_url == "http://127.0.0.1:8001"

    def test_falls_back_to_host_and_port_when_base_url_missing(self):
        provider = vLLMProvider({"host": "vllm.internal", "port": 9000})

        assert provider.base_url == "http://vllm.internal:9000"

    def test_single_base_url_normalizes_to_default_endpoint(self):
        provider = vLLMProvider({"base_url": "http://127.0.0.1:8001/"})

        assert len(provider.endpoints) == 1
        assert provider.endpoints[0].name == "default"
        assert provider.endpoints[0].base_url == "http://127.0.0.1:8001"
        assert provider.endpoints[0].models == []

    @pytest.mark.asyncio
    async def test_generate_response_preserves_completions_mode(
        self, sample_query_request
    ):
        provider = vLLMProvider({"base_url": "http://127.0.0.1:8001"})
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"text": "Use a helper."}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
            },
        }
        provider.http_client = AsyncMock()
        provider.http_client.post.return_value = mock_response

        response = await provider.generate_response(sample_query_request, "mistral-7b")

        endpoint, call_kwargs = provider.http_client.post.call_args.args[0], (
            provider.http_client.post.call_args.kwargs
        )
        assert endpoint == "/v1/completions"
        assert call_kwargs["json"]["model"] == "mistral-7b"
        assert call_kwargs["json"]["prompt"]
        assert response.response_text == "Use a helper."
        assert response.total_tokens == 15

    @pytest.mark.asyncio
    async def test_generate_response_supports_chat_completions_mode(
        self, sample_query_request
    ):
        provider = vLLMProvider(
            {"base_url": "http://127.0.0.1:8001", "api_mode": "chat_completions"}
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Use a chat template."}}],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "total_tokens": 18,
            },
        }
        provider.http_client = AsyncMock()
        provider.http_client.post.return_value = mock_response

        response = await provider.generate_response(
            sample_query_request, "qwen3.6-27b-fast"
        )

        endpoint, call_kwargs = provider.http_client.post.call_args.args[0], (
            provider.http_client.post.call_args.kwargs
        )
        payload = call_kwargs["json"]
        assert endpoint == "/v1/chat/completions"
        assert payload["model"] == "qwen3.6-27b-fast"
        assert payload["messages"][0]["role"] == "user"
        assert (
            "Context: Previous discussion context" in payload["messages"][0]["content"]
        )
        assert (
            "Query: Write a Python function to add two numbers"
            in payload["messages"][0]["content"]
        )
        assert payload["stream"] is False
        assert response.response_text == "Use a chat template."
        assert response.token_count_input == 13
        assert response.token_count_output == 5
        assert response.total_tokens == 18

    @pytest.mark.asyncio
    async def test_stream_response_supports_chat_completions_mode(
        self, sample_query_request
    ):
        class StreamResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def aiter_lines(self):
                lines = [
                    'data: {"choices":[{"delta":{"content":"Use "}}]}',
                    'data: {"choices":[{"delta":{"content":"streaming."}}]}',
                    "data: [DONE]",
                ]
                for line in lines:
                    yield line

        provider = vLLMProvider(
            {"base_url": "http://127.0.0.1:8001", "api_mode": "chat_completions"}
        )
        provider.http_client = MagicMock()
        provider.http_client.stream.return_value = StreamResponse()

        chunks = [
            chunk
            async for chunk in provider.stream_response(
                sample_query_request, "qwen3.6-27b-fast"
            )
        ]

        endpoint, call_kwargs = provider.http_client.stream.call_args.args[:2], (
            provider.http_client.stream.call_args.kwargs
        )
        assert endpoint == ("POST", "/v1/chat/completions")
        assert call_kwargs["json"]["stream"] is True
        assert chunks == ["Use ", "streaming."]

    @pytest.mark.asyncio
    async def test_pool_selects_least_outstanding_endpoint(self, sample_query_request):
        provider = vLLMProvider(
            {
                "endpoints": [
                    {"name": "qwen-a", "base_url": "http://127.0.0.1:8001"},
                    {"name": "qwen-b", "base_url": "http://127.0.0.1:8002"},
                ]
            }
        )
        provider.endpoints[0].outstanding = 2
        provider.endpoints[0].client = AsyncMock()
        provider.endpoints[1].client = AsyncMock()
        provider.endpoints[1].client.post.return_value = self._mock_vllm_response(
            "endpoint b"
        )

        response = await provider.generate_response(sample_query_request, "mistral-7b")

        assert response.response_text == "endpoint b"
        provider.endpoints[0].client.post.assert_not_called()
        provider.endpoints[1].client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_pool_filters_endpoints_by_model(self, sample_query_request):
        provider = vLLMProvider(
            {
                "endpoints": [
                    {
                        "name": "mistral",
                        "base_url": "http://127.0.0.1:8001",
                        "models": ["mistral-7b"],
                    },
                    {
                        "name": "qwen",
                        "base_url": "http://127.0.0.1:8002",
                        "models": ["qwen3.6-27b-fast"],
                    },
                ]
            }
        )
        provider.endpoints[0].client = AsyncMock()
        provider.endpoints[1].client = AsyncMock()
        provider.endpoints[1].client.post.return_value = self._mock_vllm_response(
            "qwen endpoint"
        )

        response = await provider.generate_response(
            sample_query_request, "qwen3.6-27b-fast"
        )

        assert response.response_text == "qwen endpoint"
        provider.endpoints[0].client.post.assert_not_called()
        provider.endpoints[1].client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_pool_fails_over_from_unhealthy_endpoint(self, sample_query_request):
        provider = vLLMProvider(
            {
                "failure_cooldown_seconds": 60,
                "endpoints": [
                    {"name": "qwen-a", "base_url": "http://127.0.0.1:8001"},
                    {"name": "qwen-b", "base_url": "http://127.0.0.1:8002"},
                ],
            }
        )
        provider.endpoints[0].client = AsyncMock()
        provider.endpoints[0].client.post.side_effect = RuntimeError("endpoint down")
        provider.endpoints[1].client = AsyncMock()
        provider.endpoints[1].client.post.return_value = self._mock_vllm_response(
            "failover"
        )

        response = await provider.generate_response(sample_query_request, "mistral-7b")

        assert response.response_text == "failover"
        assert provider.endpoints[0].healthy is False
        assert provider.endpoints[0].cooldown_until > time.time()
        assert "endpoint down" in provider.endpoints[0].last_error
        provider.endpoints[1].client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_prefix_affinity_reuses_prior_endpoint(self, sample_query_request):
        provider = vLLMProvider(
            {
                "endpoints": [
                    {"name": "qwen-a", "base_url": "http://127.0.0.1:8001"},
                    {"name": "qwen-b", "base_url": "http://127.0.0.1:8002"},
                ]
            }
        )
        provider.endpoints[0].outstanding = 1
        provider.endpoints[0].client = AsyncMock()
        provider.endpoints[0].client.post.return_value = self._mock_vllm_response(
            "endpoint a"
        )
        provider.endpoints[1].client = AsyncMock()
        provider.endpoints[1].client.post.return_value = self._mock_vllm_response(
            "endpoint b"
        )

        first = await provider.generate_response(sample_query_request, "mistral-7b")
        provider.endpoints[0].outstanding = 0
        second = await provider.generate_response(sample_query_request, "mistral-7b")

        assert first.response_text == "endpoint b"
        assert second.response_text == "endpoint b"
        provider.endpoints[0].client.post.assert_not_called()
        assert provider.endpoints[1].client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_prefix_affinity_falls_back_when_endpoint_saturated(
        self, sample_query_request
    ):
        provider = vLLMProvider(
            {
                "endpoints": [
                    {"name": "qwen-a", "base_url": "http://127.0.0.1:8001"},
                    {
                        "name": "qwen-b",
                        "base_url": "http://127.0.0.1:8002",
                        "max_outstanding": 1,
                    },
                ]
            }
        )
        prompt = provider._build_prompt(sample_query_request)
        prefix_key = provider._prompt_prefix_fingerprint("mistral-7b", prompt)
        provider._prefix_affinity[prefix_key] = ("qwen-b", time.time() + 60)
        provider.endpoints[0].client = AsyncMock()
        provider.endpoints[0].client.post.return_value = self._mock_vllm_response(
            "endpoint a"
        )
        provider.endpoints[1].outstanding = 1
        provider.endpoints[1].client = AsyncMock()

        response = await provider.generate_response(sample_query_request, "mistral-7b")

        assert response.response_text == "endpoint a"
        provider.endpoints[0].client.post.assert_called_once()
        provider.endpoints[1].client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_response_tracks_outstanding_until_stream_finishes(
        self, sample_query_request
    ):
        observed_outstanding = []

        class StreamResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def aiter_lines(self):
                observed_outstanding.append(provider.endpoints[0].outstanding)
                lines = [
                    'data: {"choices":[{"text":"first "}]}\n',
                    'data: {"choices":[{"text":"chunk"}]}\n',
                    "data: [DONE]",
                ]
                for line in lines:
                    yield line

        provider = vLLMProvider({"base_url": "http://127.0.0.1:8001"})
        provider.http_client = MagicMock()
        provider.http_client.stream.return_value = StreamResponse()

        chunks = [
            chunk
            async for chunk in provider.stream_response(
                sample_query_request, "mistral-7b"
            )
        ]

        assert chunks == ["first ", "chunk"]
        assert observed_outstanding == [1]
        assert provider.endpoints[0].outstanding == 0

    def test_parse_vllm_metrics_extracts_capacity_signals(self):
        provider = vLLMProvider({"base_url": "http://127.0.0.1:8001"})
        metrics = provider._parse_vllm_metrics(
            """
            # HELP vllm:num_requests_running Number of running requests.
            vllm:num_requests_running{model_name="qwen"} 2
            vllm:num_requests_waiting{model_name="qwen"} 3
            vllm:gpu_cache_usage_perc{gpu="0"} 0.42
            """
        )

        assert metrics == {
            "running_requests": 2,
            "queue_depth": 3,
            "kv_cache_usage": 0.42,
        }


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

        assert cache.generate_cache_key(
            base_request, "gpt-5"
        ) == cache.generate_cache_key(equivalent_request, "gpt-5")

    def test_cache_key_includes_tool_policy(self, sample_query_request):
        cache = ResponseCache({"enabled": True})
        auto_request = sample_query_request.model_copy()
        required_request = sample_query_request.model_copy(
            update={"tool_policy": ToolPolicy.REQUIRED}
        )

        assert cache.generate_cache_key(
            auto_request, "gpt-5"
        ) != cache.generate_cache_key(required_request, "gpt-5")


class TestInferenceEngine:
    @pytest.mark.asyncio
    async def test_initialize_skips_vllm_when_endpoint_not_configured(self):
        router = MagicMock()
        engine = InferenceEngine({"vllm": {}}, router)
        engine.context_compressor.initialize = AsyncMock()
        engine.cache.initialize = AsyncMock()

        await engine.initialize()

        assert "vllm" not in engine.providers

    def test_vllm_pool_endpoints_count_as_configured(self):
        router = MagicMock()
        engine = InferenceEngine({}, router)

        assert engine._vllm_configured(
            {"endpoints": [{"base_url": "http://127.0.0.1:8001"}]}
        )

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
    async def test_process_query_runs_web_search_before_provider_call(
        self,
        inference_config,
        sample_query_request,
        inference_response_factory,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(
                selected_model="gpt-5",
                query_type=QueryType.WEB_RESEARCH,
            )
        )
        router.get_model_info.return_value = {"config": {"provider": "openai"}}
        router.update_model_stats = MagicMock()

        source = ResponseSource(
            title="Example",
            url="https://example.com/news",
            snippet="Fresh context",
            source_domain="example.com",
            rank=1,
        )
        tool_call = ToolCall(
            name="web_search",
            provider="tavily",
            result_count=1,
            latency_ms=8,
        )
        web_search_result = SimpleNamespace(
            sources=[source],
            tool_call=tool_call,
            context="Web search results:\n[1] Example",
            blocked_reason=None,
        )

        engine = InferenceEngine(inference_config, router)
        engine.providers = {"openai": AsyncMock()}

        async def generate_response(request, _model_name):
            assert "Web search results" in request.metadata["web_search_context"]
            return inference_response_factory(response_text="answer [1]")

        engine.providers["openai"].generate_response = AsyncMock(
            side_effect=generate_response
        )
        engine.web_search_tool = SimpleNamespace(
            search=AsyncMock(return_value=web_search_result)
        )
        engine.cache.get_cached_response = AsyncMock(return_value=None)
        engine.cache.cache_response = AsyncMock()

        request = sample_query_request.model_copy(
            update={"tool_policy": ToolPolicy.REQUIRED, "allowed_tools": ["web_search"]}
        )
        response = await engine.process_query(request)

        assert response.sources == [source]
        assert response.tool_calls == [tool_call]
        assert response.tool_latency_ms == 8

    @pytest.mark.asyncio
    async def test_process_query_attaches_structured_web_search_error(
        self,
        inference_config,
        sample_query_request,
        inference_response_factory,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(
                selected_model="gpt-5",
                query_type=QueryType.WEB_RESEARCH,
            )
        )
        router.get_model_info.return_value = {"config": {"provider": "openai"}}
        router.update_model_stats = MagicMock()

        engine = InferenceEngine(inference_config, router)
        engine.providers = {"openai": AsyncMock()}
        engine.providers["openai"].generate_response = AsyncMock(
            return_value=inference_response_factory(response_text="answer without web")
        )
        engine.web_search_tool = None
        engine.cache.get_cached_response = AsyncMock(return_value=None)
        engine.cache.cache_response = AsyncMock()

        request = sample_query_request.model_copy(
            update={"tool_policy": ToolPolicy.REQUIRED, "allowed_tools": ["web_search"]}
        )
        response = await engine.process_query(request)

        assert response.sources == []
        assert response.tool_calls[0].error == "web_search_not_configured"
        assert (
            request.metadata["web_search_blocked_reason"] == "web_search_not_configured"
        )
        engine.cache.cache_response.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_query_injects_rag_context_before_cache_and_provider(
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

        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Tuition payment is due on May 1.",
            page_start=2,
            page_end=2,
            block_ids=["b1"],
            block_types=["text"],
            metadata={"title": "School Handbook"},
        )
        rag_result = RagSearchResult(
            chunk=chunk,
            score=0.9,
            match_source="hybrid",
            knowledge_base_id="school",
        )
        source = ResponseSource(
            title="School Handbook",
            url="rag://school/doc-1#chunk-1",
            rank=1,
            source_type="rag",
            document_id="doc-1",
            page=2,
            chunk_id="chunk-1",
        )
        tool_call = ToolCall(
            name="rag_search",
            provider="memory",
            result_count=1,
            latency_ms=4,
        )
        rag_service = SimpleNamespace(
            enabled=True,
            auto_retrieve=True,
            retrieve=AsyncMock(return_value=[rag_result]),
            build_context=MagicMock(
                return_value="School document search results:\n[S1] May 1"
            ),
            sources_from_results=MagicMock(return_value=[source]),
            tool_call_from_results=MagicMock(return_value=tool_call),
        )
        engine = InferenceEngine(inference_config, router, rag_service=rag_service)
        engine.providers = {"openai": AsyncMock()}

        async def generate_response(request, _model_name):
            assert "School document search results" in request.metadata["rag_context"]
            assert request.metadata["rag_chunk_ids"] == ["chunk-1"]
            return inference_response_factory(
                response_text="Tuition is due May 1 [S1]."
            )

        engine.providers["openai"].generate_response = AsyncMock(
            side_effect=generate_response
        )
        engine.cache.get_cached_response = AsyncMock(return_value=None)
        engine.cache.cache_response = AsyncMock()

        request = sample_query_request.model_copy(
            update={"knowledge_base_ids": ["school"]}
        )
        response = await engine.process_query(request)

        assert response.sources == [source]
        assert response.tool_calls == [tool_call]
        engine.cache.get_cached_response.assert_awaited_once()
        assert request.metadata["rag_chunk_ids"] == ["chunk-1"]

    @pytest.mark.asyncio
    async def test_required_rag_no_hits_returns_grounded_no_answer(
        self,
        inference_config,
        sample_query_request,
    ):
        router = MagicMock()
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(selected_model="gpt-5")
        )
        router.update_model_stats = MagicMock()
        router.token_counter.count_tokens.return_value = 5
        rag_service = SimpleNamespace(
            enabled=True,
            auto_retrieve=True,
            retrieve=AsyncMock(return_value=[]),
            sources_from_results=MagicMock(return_value=[]),
            tool_call_from_results=MagicMock(
                return_value=ToolCall(
                    name="rag_search",
                    provider="memory",
                    result_count=0,
                    latency_ms=3,
                    error="rag_no_hits",
                )
            ),
        )
        engine = InferenceEngine(inference_config, router, rag_service=rag_service)
        engine.providers = {"openai": AsyncMock()}
        engine.cache.get_cached_response = AsyncMock(return_value=None)

        request = sample_query_request.model_copy(
            update={"rag_policy": RagPolicy.REQUIRED}
        )
        response = await engine.process_query(request)

        assert response.provider == "rag"
        assert response.finish_reason == "rag_no_hits"
        assert "文件庫沒有足夠資訊" in response.response_text
        engine.providers["openai"].generate_response.assert_not_called()
        router.update_model_stats.assert_called_once_with(
            model_name="gpt-5",
            success=True,
            latency_ms=response.latency_ms,
        )
        assert engine.inference_stats["gpt-5"]["total_requests"] == 1

    def test_rag_auto_gate_only_runs_for_school_document_intent(
        self,
        inference_config,
    ):
        router = MagicMock()
        rag_service = SimpleNamespace(enabled=True, auto_retrieve=True)
        engine = InferenceEngine(inference_config, router, rag_service=rag_service)
        rag_service.intent_gate_config = {
            "strong_terms": ["registrar"],
            "exclude_terms": ["ignore-rag"],
        }
        routing_decision = SimpleNamespace(query_type=QueryType.GENERAL)

        assert (
            engine._should_run_rag(
                QueryRequest(query="How do I contact the registrar?", user_id="u1"),
                routing_decision,
            )
            is True
        )
        assert (
            engine._should_run_rag(
                QueryRequest(query="When is tuition due?", user_id="u1"),
                routing_decision,
            )
            is False
        )
        assert (
            engine._should_run_rag(
                QueryRequest(query="What is the company policy?", user_id="u1"),
                routing_decision,
            )
            is False
        )
        assert (
            engine._should_run_rag(
                QueryRequest(query="How does Python async await work?", user_id="u1"),
                routing_decision,
            )
            is False
        )

    def test_rag_explicit_triggers_bypass_auto_intent_gate(self, inference_config):
        router = MagicMock()
        rag_service = SimpleNamespace(enabled=True, auto_retrieve=False)
        engine = InferenceEngine(inference_config, router, rag_service=rag_service)
        routing_decision = SimpleNamespace(query_type=QueryType.CODE_GENERATION)

        assert (
            engine._should_run_rag(
                QueryRequest(
                    query="Find this in the uploaded docs",
                    user_id="u1",
                    knowledge_base_ids=["school"],
                ),
                routing_decision,
            )
            is True
        )
        assert (
            engine._should_run_rag(
                QueryRequest(
                    query="Find this in the uploaded docs",
                    user_id="u1",
                    rag_policy=RagPolicy.REQUIRED,
                ),
                routing_decision,
            )
            is True
        )
        assert (
            engine._should_run_rag(
                QueryRequest(
                    query="When is tuition due?",
                    user_id="u1",
                    rag_policy=RagPolicy.DISABLED,
                    knowledge_base_ids=["school"],
                ),
                routing_decision,
            )
            is False
        )

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

    @pytest.mark.asyncio
    async def test_process_query_falls_back_between_configured_vllm_models(
        self,
        inference_config,
        sample_query_request,
        inference_response_factory,
    ):
        from src.llm_router_part1_router import ModelRouter

        router = ModelRouter(
            {
                "default_model": "qwen3.6-27b-fast",
                "routing_strategy": "intelligent",
                "models": {
                    "qwen3.6-27b-fast": {
                        "provider": "vllm",
                        "model_path": "Qwen/Qwen3.6-27B-FP8",
                        "max_tokens": 32768,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "coding"],
                    },
                    "mistral-7b": {
                        "provider": "vllm",
                        "model_path": "/models/mistral",
                        "max_tokens": 8192,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "coding"],
                    },
                },
                "routing_rules": [],
            }
        )
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(
                selected_model="qwen3.6-27b-fast",
                query_type=QueryType.GENERAL,
                token_count=64,
                routing_reason="Fast local selection",
            )
        )
        router.update_model_stats = MagicMock()
        config = {
            **inference_config,
            "vllm": {
                "model_fallback": {
                    "enabled": True,
                    "fallbacks": {"qwen3.6-27b-fast": ["mistral-7b"]},
                    "allowed_query_types": ["general", "code_generation"],
                    "max_input_tokens": 2048,
                    "max_output_tokens": 1024,
                    "disallow_attachments": True,
                    "disallow_complex_reasoning": True,
                }
            },
        }

        engine = InferenceEngine(config, router)
        engine.providers = {"vllm": AsyncMock()}
        engine.providers["vllm"].generate_response = AsyncMock(
            side_effect=[
                RuntimeError("qwen temporarily unavailable"),
                inference_response_factory(
                    model_name="mistral-7b",
                    provider="vllm",
                    cost_usd=0.0,
                ),
            ]
        )
        engine.cache.get_cached_response = AsyncMock(side_effect=[None, None])
        engine.cache.cache_response = AsyncMock()

        response = await engine.process_query(sample_query_request)

        assert response.model_name == "mistral-7b"
        assert response.provider == "vllm"
        assert [
            call.args[1]
            for call in engine.providers["vllm"].generate_response.await_args_list
        ] == ["qwen3.6-27b-fast", "mistral-7b"]
        router.update_model_stats.assert_any_call(
            model_name="qwen3.6-27b-fast",
            success=False,
            latency_ms=ANY,
        )
        router.update_model_stats.assert_any_call(
            model_name="mistral-7b",
            success=True,
            latency_ms=response.latency_ms,
        )

    @pytest.mark.asyncio
    async def test_vllm_model_fallback_rejects_complex_reasoning(
        self,
        inference_config,
        sample_query_request,
    ):
        from src.llm_router_part1_router import ModelRouter

        router = ModelRouter(
            {
                "default_model": "qwen3.6-27b-fast",
                "routing_strategy": "intelligent",
                "models": {
                    "qwen3.6-27b-fast": {
                        "provider": "vllm",
                        "model_path": "Qwen/Qwen3.6-27B-FP8",
                        "max_tokens": 32768,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "coding", "reasoning"],
                    },
                    "mistral-7b": {
                        "provider": "vllm",
                        "model_path": "/models/mistral",
                        "max_tokens": 8192,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "coding"],
                    },
                },
                "routing_rules": [],
            }
        )
        router.route_query = AsyncMock(
            return_value=SimpleNamespace(
                selected_model="qwen3.6-27b-fast",
                query_type=QueryType.REASONING,
                query_complexity="complex",
                token_count=512,
                routing_reason="Fast local selection",
            )
        )
        router.update_model_stats = MagicMock()
        config = {
            **inference_config,
            "vllm": {
                "model_fallback": {
                    "enabled": True,
                    "fallbacks": {"qwen3.6-27b-fast": ["mistral-7b"]},
                    "allowed_query_types": ["general", "code_generation"],
                    "max_input_tokens": 2048,
                    "max_output_tokens": 1024,
                    "disallow_complex_reasoning": True,
                }
            },
        }

        engine = InferenceEngine(config, router)
        engine.providers = {"vllm": AsyncMock()}
        engine.providers["vllm"].generate_response = AsyncMock(
            side_effect=RuntimeError("qwen temporarily unavailable")
        )
        engine.cache.get_cached_response = AsyncMock(return_value=None)

        response = await engine.process_query(sample_query_request)

        assert response.provider == "error"
        assert response.error == "inference_failed"
        assert engine.providers["vllm"].generate_response.await_count == 1
