from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.schema import (
    ApiRateLimitingConfig,
    Attachment,
    AttachmentType,
    InferenceResponse,
    ModelConfig,
    PlatformConfig,
    ProviderEndpointConfig,
    QueryRequest,
    RagPolicy,
    ResponseSource,
    SystemMetric,
    ToolCall,
    ToolPolicy,
)


def test_attachment_size_validation():
    with pytest.raises(ValidationError):
        Attachment(
            name="bad.bin",
            type=AttachmentType.FILE,
            size_bytes=0,
            mime_type="application/octet-stream",
        )


def test_query_request_strips_whitespace():
    request = QueryRequest(query="  hello world  ", user_id="u1")

    assert request.query == "hello world"


def test_query_request_rejects_too_many_attachments():
    attachments = [
        Attachment(
            name=f"file-{index}.txt",
            type=AttachmentType.FILE,
            size_bytes=10,
            mime_type="text/plain",
        )
        for index in range(11)
    ]

    with pytest.raises(ValidationError):
        QueryRequest(query="hello", user_id="u1", attachments=attachments)


def test_query_request_accepts_web_search_tool_options():
    request = QueryRequest(
        query="latest AI news",
        user_id="u1",
        tool_policy=ToolPolicy.REQUIRED,
        allowed_tools=["web_search"],
        web_search_options={"max_results": 3, "search_depth": "basic"},
    )

    assert request.tool_policy == ToolPolicy.REQUIRED
    assert request.web_search_options.max_results == 3


def test_query_request_accepts_rag_options():
    request = QueryRequest(
        query="When is tuition due?",
        user_id="u1",
        rag_policy=RagPolicy.REQUIRED,
        knowledge_base_ids=["school"],
        rag_options={"max_results": 4, "candidate_count": 20, "min_score": 0.1},
    )

    assert request.rag_policy == RagPolicy.REQUIRED
    assert request.knowledge_base_ids == ["school"]
    assert request.rag_options.max_results == 4


def test_inference_response_auto_calculates_total_tokens_when_zero():
    response = InferenceResponse(
        response_text="ok",
        model_name="gpt-5",
        provider="openai",
        token_count_input=11,
        token_count_output=13,
        total_tokens=0,
        latency_ms=20,
        tokens_per_second=650.0,
        cost_usd=0.01,
    )

    assert response.total_tokens == 24


def test_inference_response_accepts_sources_and_tool_calls():
    response = InferenceResponse(
        response_text="ok [1]",
        model_name="gpt-5",
        provider="openai",
        token_count_input=11,
        token_count_output=13,
        total_tokens=24,
        latency_ms=20,
        tokens_per_second=650.0,
        cost_usd=0.01,
        sources=[
            ResponseSource(
                title="Example",
                url="https://example.com",
                snippet="snippet",
                score=0.8,
                rank=1,
                source_type="rag",
                document_id="doc-1",
                page=3,
                bbox=[0, 0, 10, 10],
                chunk_id="chunk-1",
                index_version="v1",
            )
        ],
        tool_calls=[
            ToolCall(name="web_search", provider="tavily", result_count=1, latency_ms=5)
        ],
    )
    assert response.sources[0].index_version == "v1"

    assert response.sources[0].url == "https://example.com"
    assert response.sources[0].score == 0.8
    assert response.sources[0].chunk_id == "chunk-1"
    assert response.tool_calls[0].name == "web_search"


def test_model_config_rejects_unknown_capability():
    with pytest.raises(ValidationError):
        ModelConfig(
            name="bad-model",
            provider="test",
            max_tokens=1000,
            cost_per_token=0.0,
            priority=1,
            capabilities=["unknown-capability"],
        )


def test_model_config_accepts_tool_capabilities():
    model = ModelConfig(
        name="gpt-5",
        provider="openai",
        max_tokens=1000,
        cost_per_token=0.0,
        priority=1,
        capabilities=["general", "tool_use", "web_search"],
    )

    assert "web_search" in model.capabilities


def test_system_metric_name_validation():
    with pytest.raises(ValidationError):
        SystemMetric(name="bad metric name", value=1.0)


def test_platform_config_rejects_invalid_api_port():
    with pytest.raises(ValidationError):
        PlatformConfig(api={"port": 70000})


def test_api_rate_limiting_accepts_legacy_request_bucket_config():
    config = ApiRateLimitingConfig(
        enabled=True,
        requests_per_minute=500,
        burst_size=50,
    )

    assert config.enabled is True
    assert config.requests_per_minute == 500
    assert config.burst_size == 50
    assert config.redis.db == 4
    assert config.redis.key_prefix == "api_gateway"
    assert config.failure_mode == "closed"


def test_api_rate_limiting_accepts_nested_admission_config():
    config = ApiRateLimitingConfig(
        enabled=True,
        failure_mode="open",
        redis={"host": "redis", "db": 4, "key_prefix": "gateway"},
        queue={"max_depth": 25, "timeout_ms": 50},
        global_limits={"active_requests": 10, "tokens_per_minute": 1000},
        per_user={"requests_per_minute": 5, "burst_size": 2},
        by_tier={"premium": {"tokens_per_minute": 2000, "burst_tokens": 500}},
        providers={"openai": {"active_requests": 3}},
        models={"gpt-5": {"requests_per_minute": 10, "burst_size": 4}},
        token_budget={"enabled": True, "tokens_per_minute": 5000, "burst_tokens": 500},
    )

    assert config.failure_mode == "open"
    assert config.redis.host == "redis"
    assert config.queue.max_depth == 25
    assert config.global_limits.active_requests == 10
    assert config.per_user.requests_per_minute == 5
    assert config.by_tier["premium"].burst_tokens == 500
    assert config.providers["openai"].active_requests == 3
    assert config.models["gpt-5"].burst_size == 4
    assert config.token_budget.enabled is True


def test_api_rate_limiting_rejects_unknown_failure_mode():
    with pytest.raises(ValidationError):
        ApiRateLimitingConfig(failure_mode="maybe")


def test_provider_endpoint_config_accepts_chat_completions_api_mode():
    config = ProviderEndpointConfig(api_mode="chat_completions")

    assert config.api_mode == "chat_completions"


def test_provider_endpoint_config_accepts_vllm_pool_endpoints():
    config = ProviderEndpointConfig(
        api_mode="chat_completions",
        routing_strategy="least_outstanding_prefix_aware",
        health_check_interval_seconds=10,
        failure_cooldown_seconds=30,
        metrics_refresh_seconds=5,
        prefix_affinity_ttl_seconds=120,
        metrics_scrape_enabled=True,
        endpoints=[
            {
                "name": "qwen-a",
                "base_url": "http://127.0.0.1:8001",
                "models": ["qwen3.6-27b-fast"],
                "weight": 2.0,
                "max_outstanding": 4,
                "health_path": "/health",
                "metrics_path": "/metrics",
                "prefix_cache_enabled": True,
                "enabled": True,
            }
        ],
    )

    assert config.endpoints[0].name == "qwen-a"
    assert config.endpoints[0].models == ["qwen3.6-27b-fast"]
    assert config.endpoints[0].max_outstanding == 4
    assert config.metrics_scrape_enabled is True


def test_provider_endpoint_config_accepts_vllm_model_fallback_policy():
    config = ProviderEndpointConfig(
        model_fallback={
            "enabled": True,
            "fallbacks": {"qwen3.6-27b-fast": ["mistral-7b"]},
            "allowed_query_types": ["general", "code_generation"],
            "max_input_tokens": 2048,
            "max_output_tokens": 1024,
            "disallow_attachments": True,
            "disallow_complex_reasoning": True,
            "disallow_required_tools": True,
            "disallow_required_rag": True,
        }
    )

    assert config.model_fallback.enabled is True
    assert config.model_fallback.fallbacks == {"qwen3.6-27b-fast": ["mistral-7b"]}
    assert config.model_fallback.max_input_tokens == 2048


def test_provider_endpoint_config_rejects_unknown_api_mode():
    with pytest.raises(ValidationError):
        ProviderEndpointConfig(api_mode="responses")


def test_provider_endpoint_config_rejects_unknown_routing_strategy():
    with pytest.raises(ValidationError):
        ProviderEndpointConfig(routing_strategy="random")


def test_platform_config_rejects_unknown_slack_state_backend():
    with pytest.raises(ValidationError):
        PlatformConfig(slack={"state_backend": "sqlite"})


def test_platform_config_rejects_unknown_slack_memory_backend():
    with pytest.raises(ValidationError):
        PlatformConfig(slack={"memory": {"backend": "sqlite"}})


def test_platform_config_defaults_file_state_path():
    config = PlatformConfig(slack={"state_backend": "file"})

    assert config.slack.state_file == "data/slack_state.json"


def test_platform_config_accepts_redis_slack_state_backend():
    config = PlatformConfig(
        slack={
            "state_backend": "redis",
            "state_key_prefix": "slack_state_test",
            "redis": {"host": "localhost", "port": 6379, "db": 5},
        }
    )

    assert config.slack.state_backend == "redis"
    assert config.slack.state_key_prefix == "slack_state_test"
    assert config.slack.redis["db"] == 5


def test_platform_config_accepts_slack_context_config():
    config = PlatformConfig(
        slack={
            "context": {
                "enabled": True,
                "strategy": "thread_first",
                "max_thread_messages": 25,
                "max_channel_messages": 8,
                "max_context_chars": 3000,
                "include_bot_messages": False,
                "timeout_seconds": 2.5,
                "fail_open": True,
            }
        }
    )

    assert config.slack.context.enabled is True
    assert config.slack.context.strategy == "thread_first"
    assert config.slack.context.max_thread_messages == 25
    assert config.slack.context.max_channel_messages == 8
    assert config.slack.context.max_context_chars == 3000
    assert config.slack.context.timeout_seconds == 2.5


def test_platform_config_rejects_unknown_slack_context_strategy():
    with pytest.raises(ValidationError):
        PlatformConfig(slack={"context": {"strategy": "channel_first"}})


def test_platform_config_accepts_slack_memory_config():
    config = PlatformConfig(
        slack={
            "memory": {
                "enabled": True,
                "backend": "redis_stack",
                "redis": {"host": "redis-stack", "db": 3},
                "embedding": {"provider": "openai", "dimensions": 1536},
            }
        }
    )

    assert config.slack.memory.enabled is True
    assert config.slack.memory.backend == "redis_stack"
    assert config.slack.memory.redis.db == 3


def test_platform_config_accepts_tavily_web_search_config():
    config = PlatformConfig(
        tools={
            "web_search": {
                "enabled": True,
                "provider": "tavily",
                "api_key_env": "TAVILY_API_KEY",
                "max_results_per_domain": 1,
            }
        }
    )

    assert config.tools.web_search.enabled is True
    assert config.tools.web_search.provider == "tavily"
    assert config.tools.web_search.max_results_per_domain == 1


def test_platform_config_accepts_rag_config():
    config = PlatformConfig(
        rag={
            "enabled": True,
            "backend": "memory",
            "default_knowledge_base_ids": ["school"],
            "embedding": {
                "provider": "local_http",
                "model": "BAAI/bge-m3",
                "dimensions": 1024,
            },
            "retrieval": {
                "keyword_scorer": "BM25STD",
                "keyword_score_normalization": "max",
            },
            "visual": {
                "enabled": True,
                "crop_dpi": 180,
                "embedding": {
                    "enabled": True,
                    "provider": "nomic_multimodal",
                    "dimensions": 1024,
                },
            },
            "ingestion_queue": {"enabled": True, "concurrency": 2},
            "storage": {"staging_dir": "data/rag/uploads"},
        }
    )

    assert config.rag.enabled is True
    assert config.rag.backend == "memory"
    assert config.rag.default_knowledge_base_ids == ["school"]
    assert config.rag.retrieval.keyword_scorer == "BM25STD"
    assert config.rag.retrieval.keyword_score_normalization == "max"
    assert config.rag.visual.enabled is True
    assert config.rag.visual.embedding.provider == "nomic_multimodal"
    assert config.rag.visual.embedding.dimensions == 1024
    assert config.rag.ingestion_queue.enabled is True
    assert config.rag.ingestion_queue.concurrency == 2
    assert config.rag.storage.staging_dir == "data/rag/uploads"


def test_checked_in_compose_config_validates():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.compose.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = PlatformConfig.model_validate(config)

    assert validated.pipeline.enabled is True
    assert validated.kafka.bootstrap_servers == ["kafka:29092"]
    assert validated.clickhouse.host == "clickhouse"


def test_checked_in_default_config_uses_qwen_fast_lane():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = PlatformConfig.model_validate(config)

    assert validated.router.fast_lane_models == ["qwen3.6-27b-fast"]
    assert validated.router.models["qwen3.6-27b-fast"].provider == "vllm"
    assert validated.router.models["qwen3.6-27b-fast"].max_tokens == 32768
    assert validated.inference.vllm.api_mode == "chat_completions"
    assert validated.inference.vllm.model_fallback.enabled is True
    assert validated.inference.vllm.model_fallback.fallbacks == {
        "qwen3.6-27b-fast": ["mistral-7b"]
    }


def test_compose_and_default_config_keep_flink_topics_in_sync():
    root = Path(__file__).resolve().parents[1]
    default_config = yaml.safe_load(
        (root / "config" / "config.yaml").read_text(encoding="utf-8")
    )
    compose_config = yaml.safe_load(
        (root / "config" / "config.compose.yaml").read_text(encoding="utf-8")
    )

    expected_topic_keys = {
        "requests_raw",
        "inference_completed",
        "requests_enriched",
        "fast_lane_hints",
        "analytics_model_metrics_1m",
        "routing_guardrails",
        "routing_policy_state",
        "alerts",
    }

    default_topics = set(
        (default_config.get("kafka", {}) or {}).get("topics", {}).keys()
    )
    compose_topics = set(
        (compose_config.get("kafka", {}) or {}).get("topics", {}).keys()
    )

    assert expected_topic_keys.issubset(default_topics)
    assert expected_topic_keys.issubset(compose_topics)
