import pytest
from pydantic import ValidationError

from src.utils.schema import (
    Attachment,
    AttachmentType,
    InferenceResponse,
    ModelConfig,
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
