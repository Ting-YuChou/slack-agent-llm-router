"""I6: truncated, filtered and empty answers are flagged and never cached."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm_router_part2_inference import (
    AnthropicProvider,
    InferenceEngine,
    OpenAIProvider,
    vLLMProvider,
)
from src.llm_router_part3_pipeline import build_inference_completed_event


def _openai_usage():
    return SimpleNamespace(
        input_tokens=12,
        output_tokens=18,
        total_tokens=30,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )


async def _openai_generate(api_response, request):
    with patch("src.llm_router_part2_inference.openai.AsyncOpenAI") as mock_openai:
        client = AsyncMock()
        client.responses.create.return_value = api_response
        mock_openai.return_value = client
        provider = OpenAIProvider({"api_key": "test-key"})
        await provider.initialize()
        return await provider.generate_response(request, "gpt-5")


def _openai_response(**fields):
    payload = {
        "output_text": "partial answer",
        "usage": _openai_usage(),
        "status": "completed",
        "incomplete_details": None,
        "output": [],
    }
    payload.update(fields)
    return SimpleNamespace(**payload)


_REFUSAL_OUTPUT = [
    SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="refusal", refusal="I can't help with that.")],
    )
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param({}, "stop", id="completed"),
        pytest.param(
            {
                "status": "incomplete",
                "incomplete_details": SimpleNamespace(reason="max_output_tokens"),
            },
            "length",
            id="max-output-tokens",
        ),
        pytest.param(
            {
                "status": "incomplete",
                "incomplete_details": SimpleNamespace(reason="content_filter"),
            },
            "content_filter",
            id="content-filter",
        ),
        pytest.param(
            {"output_text": "", "output": _REFUSAL_OUTPUT},
            "content_filter",
            id="refusal",
        ),
        # Unknown shapes (mocked SDK objects) keep the old default.
        pytest.param({"status": MagicMock()}, "stop", id="non-string-status"),
    ],
)
async def test_openai_finish_reason(sample_query_request, fields, expected):
    response = await _openai_generate(_openai_response(**fields), sample_query_request)

    assert response.finish_reason == expected


@pytest.mark.asyncio
async def test_openai_provider_request_id(sample_query_request):
    api_response = _openai_response(_request_id="req_openai_1")

    response = await _openai_generate(api_response, sample_query_request)

    assert response.provider_request_id == "req_openai_1"


@pytest.mark.asyncio
async def test_openai_mock_request_id_is_ignored(sample_query_request):
    api_response = MagicMock()
    api_response.output_text = "answer"
    api_response.usage = _openai_usage()

    response = await _openai_generate(api_response, sample_query_request)

    assert response.provider_request_id is None
    assert response.finish_reason == "stop"


async def _anthropic_generate(api_response, request):
    with patch(
        "src.llm_router_part2_inference.anthropic.AsyncAnthropic"
    ) as mock_anthropic:
        client = AsyncMock()
        client.messages.create.return_value = api_response
        mock_anthropic.return_value = client
        provider = AnthropicProvider({"api_key": "test-key"})
        await provider.initialize()
        return await provider.generate_response(request, "claude-sonnet-4-6")


def _anthropic_response(**fields):
    payload = {
        "content": [SimpleNamespace(type="text", text="partial answer")],
        "usage": SimpleNamespace(input_tokens=9, output_tokens=21),
        "stop_reason": "end_turn",
    }
    payload.update(fields)
    return SimpleNamespace(**payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        pytest.param("end_turn", "stop", id="end-turn"),
        pytest.param("stop_sequence", "stop", id="stop-sequence"),
        pytest.param("max_tokens", "length", id="max-tokens"),
        pytest.param(
            "model_context_window_exceeded", "length", id="context-window-exceeded"
        ),
        pytest.param("refusal", "content_filter", id="refusal"),
        pytest.param(MagicMock(), "stop", id="non-string"),
    ],
)
async def test_anthropic_finish_reason(sample_query_request, stop_reason, expected):
    api_response = _anthropic_response(stop_reason=stop_reason)

    response = await _anthropic_generate(api_response, sample_query_request)

    assert response.finish_reason == expected


@pytest.mark.asyncio
async def test_anthropic_provider_request_id(sample_query_request):
    api_response = _anthropic_response(_request_id="req_anthropic_1")

    response = await _anthropic_generate(api_response, sample_query_request)

    assert response.provider_request_id == "req_anthropic_1"


def _vllm_http_response(choice, *, headers=None):
    return httpx.Response(
        200,
        json={
            "choices": [choice],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        },
        headers=headers or {},
        request=httpx.Request("POST", "http://127.0.0.1:8001/v1/completions"),
    )


async def _vllm_generate(http_response, request):
    provider = vLLMProvider({"base_url": "http://127.0.0.1:8001"})
    provider.http_client = AsyncMock()
    provider.http_client.post.return_value = http_response
    return await provider.generate_response(request, "mistral-7b")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        pytest.param({"text": "cut off", "finish_reason": "length"}, "length"),
        pytest.param({"text": "done", "finish_reason": "stop"}, "stop"),
        pytest.param({"text": "done", "finish_reason": None}, "stop"),
        pytest.param({"text": "done"}, "stop", id="missing"),
    ],
)
async def test_vllm_finish_reason(sample_query_request, choice, expected):
    response = await _vllm_generate(_vllm_http_response(choice), sample_query_request)

    assert response.finish_reason == expected


@pytest.mark.asyncio
async def test_vllm_provider_request_id_comes_from_the_header(sample_query_request):
    http_response = _vllm_http_response(
        {"text": "done", "finish_reason": "stop"},
        headers={"X-Request-Id": "vllm-req-1"},
    )

    response = await _vllm_generate(http_response, sample_query_request)

    assert response.provider_request_id == "vllm-req-1"


@pytest.mark.asyncio
async def test_vllm_mock_response_has_no_request_id(sample_query_request):
    http_response = MagicMock()
    http_response.json.return_value = {"choices": [{"text": "done"}], "usage": {}}

    response = await _vllm_generate(http_response, sample_query_request)

    assert response.provider_request_id is None
    assert response.finish_reason == "stop"


def _engine(provider_response=None, *, side_effect=None):
    router = MagicMock()
    router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    event_producer = AsyncMock()
    engine = InferenceEngine(
        {"cache": {"enabled": True}, "batching": {"enabled": False}},
        router,
        event_producer=event_producer,
    )
    engine.providers = {"openai": AsyncMock()}
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=provider_response, side_effect=side_effect
    )
    engine._get_local_fallback_models = MagicMock(return_value=[])
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()
    return engine, event_producer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "expected_warning"),
    [
        pytest.param({"finish_reason": "length"}, "response_truncated", id="length"),
        pytest.param(
            {"finish_reason": "content_filter"},
            "response_filtered",
            id="content-filter",
        ),
        pytest.param({"response_text": "   "}, "empty_response", id="empty"),
    ],
)
async def test_incomplete_answers_get_a_warning_and_skip_the_cache(
    sample_query_request, inference_response_factory, overrides, expected_warning
):
    engine, event_producer = _engine(
        inference_response_factory(provider_request_id="req_1", **overrides)
    )

    response = await engine.process_query(sample_query_request)

    assert response.error is None
    assert expected_warning in response.warnings
    engine.cache.cache_response.assert_not_awaited()
    published = event_producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(sample_query_request, published)
    assert event["finish_reason"] == response.finish_reason
    assert event["provider_request_id"] == "req_1"


@pytest.mark.asyncio
async def test_complete_answer_is_cached_without_warnings(
    sample_query_request, inference_response_factory
):
    engine, event_producer = _engine(
        inference_response_factory(provider_request_id="req_2")
    )

    response = await engine.process_query(sample_query_request)

    assert response.warnings == []
    engine.cache.cache_response.assert_awaited_once()
    # A cache hit makes no provider call, so it must not replay this id.
    cached_payload = engine.cache.cache_response.await_args.args[1]
    assert "provider_request_id" not in cached_payload
    published = event_producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(sample_query_request, published)
    assert event["finish_reason"] == "stop"
    assert event["provider_request_id"] == "req_2"


class _SDKStatusError(Exception):
    def __init__(self, status_code, request_id):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.request_id = request_id
        self.response = SimpleNamespace(status_code=status_code, headers={})


@pytest.mark.asyncio
async def test_error_response_keeps_the_provider_request_id(sample_query_request):
    engine, _ = _engine(side_effect=_SDKStatusError(503, "req_failed_1"))

    response = await engine.process_query(sample_query_request)

    assert response.error == "inference_failed"
    assert response.provider_request_id == "req_failed_1"


@pytest.mark.asyncio
async def test_error_response_reads_x_request_id_from_httpx_errors(
    sample_query_request,
):
    request = httpx.Request("POST", "http://vllm:8000/v1/completions")
    http_response = httpx.Response(
        503, request=request, headers={"x-request-id": "vllm-failed-1"}
    )
    exc = httpx.HTTPStatusError("busy", request=request, response=http_response)
    engine, _ = _engine(side_effect=exc)

    response = await engine.process_query(sample_query_request)

    assert response.provider_request_id == "vllm-failed-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason",
    ["abort", "failed", "cancelled", "incomplete", "tool_calls"],
)
async def test_other_unfinished_answers_get_a_generic_warning(
    sample_query_request, inference_response_factory, finish_reason
):
    engine, event_producer = _engine(
        inference_response_factory(finish_reason=finish_reason)
    )

    response = await engine.process_query(sample_query_request)

    assert response.finish_reason == finish_reason
    assert "response_incomplete" in response.warnings
    assert "response_truncated" not in response.warnings
    engine.cache.cache_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_request_reports_finish_reason_error(sample_query_request):
    engine, event_producer = _engine(side_effect=_SDKStatusError(503, "req_3"))

    response = await engine.process_query(sample_query_request)

    assert response.finish_reason == "error"
    published = event_producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(sample_query_request, published)
    assert event["status"] == "error"
    assert event["finish_reason"] == "error"


# Real SDK objects: CI has no openai/anthropic SDK, so these run locally only.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "content": [
                    {"type": "output_text", "text": "partial", "annotations": []}
                ],
            },
            "length",
            id="max-output-tokens",
        ),
        pytest.param(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "content": [],
            },
            "content_filter",
            id="content-filter",
        ),
        pytest.param(
            {
                "status": "completed",
                "incomplete_details": None,
                "content": [{"type": "refusal", "refusal": "I can't help."}],
            },
            "content_filter",
            id="refusal",
        ),
        pytest.param(
            {
                "status": "completed",
                "incomplete_details": None,
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            },
            "stop",
            id="completed",
        ),
    ],
)
async def test_openai_sdk_response_finish_reason(
    sample_query_request, fields, expected
):
    pytest.importorskip("openai", minversion="1.66")
    from openai.types.responses import Response

    api_response = Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-5",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": fields["content"],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": fields["status"],
            "incomplete_details": fields["incomplete_details"],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
        }
    )
    api_response._request_id = "req_sdk_openai"

    response = await _openai_generate(api_response, sample_query_request)

    assert response.finish_reason == expected
    assert response.provider_request_id == "req_sdk_openai"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("end_turn", "stop"),
    ],
)
async def test_anthropic_sdk_message_finish_reason(
    sample_query_request, stop_reason, expected
):
    pytest.importorskip("anthropic", minversion="0.69")
    from anthropic.types import Message

    api_response = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    api_response._request_id = "req_sdk_anthropic"

    response = await _anthropic_generate(api_response, sample_query_request)

    assert response.finish_reason == expected
    assert response.provider_request_id == "req_sdk_anthropic"
