from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.utils.schema import (
    Attachment,
    AttachmentType,
    InferenceResponse,
    ModelConfig,
    PlatformConfig,
    QueryRequest,
    SystemMetric,
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


def test_system_metric_name_validation():
    with pytest.raises(ValidationError):
        SystemMetric(name="bad metric name", value=1.0)


def test_platform_config_rejects_invalid_api_port():
    with pytest.raises(ValidationError):
        PlatformConfig(api={"port": 70000})


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


def test_checked_in_compose_config_validates():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.compose.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = PlatformConfig.model_validate(config)

    assert validated.pipeline.enabled is True
    assert validated.kafka.bootstrap_servers == ["kafka:29092"]
    assert validated.clickhouse.host == "clickhouse"


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
