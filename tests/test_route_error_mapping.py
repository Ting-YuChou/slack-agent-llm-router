"""I12: specific error codes for failed inference and their /route HTTP status."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

import main
import src.llm_router_part2_inference as inference_module
import src.provider_scheduler as provider_scheduler
from src.admission import AdmissionDecision, AdmissionRejectedError
from src.llm_router_part2_inference import (
    InferenceEngine,
    classify_inference_error,
    inference_error_code,
    route_response_status,
)
from src.llm_router_part3_pipeline import build_inference_completed_event
from src.utils.schema import InferenceResponse

# test_main's platform doubles; the fixture is used through pytestmark below.
from test_main import (  # noqa: F401
    DummyInferenceEngine,
    DummyRouter,
    _write_config,
    patched_platform_deps,
)


class _ProviderStatusError(Exception):
    """SDK APIStatusError shape without the SDK: status, response and code."""

    def __init__(self, status_code, message=None, *, code=None, headers=None):
        super().__init__(message or f"Error code: {status_code}")
        self.status_code = status_code
        self.code = code
        self.response = SimpleNamespace(
            status_code=status_code, headers=dict(headers or {})
        )


class _TypedRejection(ValueError):
    """Shape of the router's typed rejections (ModelNotEntitledError and friends)."""

    def __init__(self, message, *, error_code, http_status, error_type=None):
        super().__init__(message)
        self.error_code = error_code
        self.http_status = http_status
        if error_type is not None:
            self.error_type = error_type


_APIConnectionError = type("APIConnectionError", (Exception,), {})


def _admission_rejected(reason, *, status_code=429, retry_after_seconds=1):
    return AdmissionRejectedError(
        AdmissionDecision.reject(
            status_code=status_code,
            error="scheduler_unavailable",
            reason=reason,
            message="denied",
            retry_after_seconds=retry_after_seconds,
        )
    )


_ANTHROPIC_PROMPT_TOO_LONG = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'prompt is too long: "
    "210000 tokens > 200000 maximum'}}"
)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        pytest.param(
            _ProviderStatusError(400, code="context_length_exceeded"),
            "context_length_exceeded",
            id="openai-context-length",
        ),
        pytest.param(
            _ProviderStatusError(400, _ANTHROPIC_PROMPT_TOO_LONG),
            "context_length_exceeded",
            id="anthropic-prompt-too-long",
        ),
        pytest.param(
            _ProviderStatusError(
                400,
                "This model's maximum context length is 4096 tokens. However, "
                "you requested 5000 tokens.",
            ),
            "context_length_exceeded",
            id="vllm-maximum-context-length",
        ),
        pytest.param(_ProviderStatusError(413), "prompt_too_large", id="status-413"),
        pytest.param(
            _TypedRejection(
                "too large", error_code="prompt_too_large", http_status=413
            ),
            "prompt_too_large",
            id="typed-prompt-too-large",
        ),
        pytest.param(
            _TypedRejection("tier", error_code="model_not_entitled", http_status=403),
            "model_not_entitled",
            id="typed-model-not-entitled",
        ),
        pytest.param(_ProviderStatusError(429), "rate_limited", id="status-429"),
        pytest.param(
            _admission_rejected("provider_active_requests_exceeded"),
            "rate_limited",
            id="admission-capacity",
        ),
        pytest.param(
            _ProviderStatusError(503), "provider_unavailable", id="status-503"
        ),
        pytest.param(
            _ProviderStatusError(529), "provider_unavailable", id="status-529"
        ),
        pytest.param(ConnectionError("reset"), "provider_unavailable", id="connection"),
        pytest.param(
            _APIConnectionError(), "provider_unavailable", id="sdk-connection"
        ),
        pytest.param(
            _admission_rejected("circuit_open", status_code=503),
            "provider_unavailable",
            id="circuit-open",
        ),
        pytest.param(
            RuntimeError("No healthy vLLM endpoint can serve model mistral-7b"),
            "provider_unavailable",
            id="no-healthy-vllm-endpoint",
        ),
        pytest.param(asyncio.TimeoutError(), "timeout", id="asyncio-timeout"),
        pytest.param(_ProviderStatusError(408), "timeout", id="status-408"),
        # Nothing specific to say: /route keeps answering 502 for these.
        pytest.param(_ProviderStatusError(400), None, id="plain-400"),
        pytest.param(_ProviderStatusError(401), None, id="revoked-key-401"),
        pytest.param(_ProviderStatusError(403), None, id="provider-403"),
        pytest.param(KeyError("bug"), None, id="internal-bug"),
        pytest.param(ValueError("no provider"), None, id="value-error"),
    ],
)
def test_inference_error_code(exc, expected):
    assert inference_error_code(exc) == expected


def test_vllm_context_length_error_is_read_from_the_response_body():
    request = httpx.Request("POST", "http://vllm:8000/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={
            "object": "error",
            "message": "This model's maximum context length is 4096 tokens. "
            "However, you requested 5000 tokens.",
            "type": "BadRequestError",
            "code": 400,
        },
    )
    exc = httpx.HTTPStatusError("Client error", request=request, response=response)

    assert inference_error_code(exc) == "context_length_exceeded"


def test_retry_after_http_date_and_unparseable_values():
    future = _ProviderStatusError(
        429, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}
    )
    garbage = _ProviderStatusError(429, headers={"retry-after": "soon"})

    assert inference_module._retry_after_seconds(future) > 0
    assert inference_module._retry_after_seconds(garbage) is None


def test_typed_rejection_error_type_attribute_wins():
    exc = _TypedRejection(
        "tier",
        error_code="model_not_entitled",
        http_status=403,
        error_type="user_or_policy",
    )
    assert classify_inference_error(exc) == "user_or_policy"

    internal = _TypedRejection(
        "store", error_code="retrieval_unavailable", http_status=503
    )
    internal.error_type = "internal"
    assert classify_inference_error(internal) == "internal"


def test_status_code_helper_and_request_shaped_statuses_are_shared():
    assert (
        inference_module._exception_status_code
        is provider_scheduler._exception_status_code
    )
    assert provider_scheduler.REQUEST_SHAPED_STATUS_CODES == frozenset({400, 413, 422})
    assert (
        inference_module.REQUEST_SHAPED_STATUS_CODES
        is provider_scheduler.REQUEST_SHAPED_STATUS_CODES
    )


def _error_response(**overrides):
    payload = {
        "response_text": "Request processing failed",
        "model_name": "gpt-5",
        "provider": "error",
        "token_count_input": 0,
        "token_count_output": 0,
        "total_tokens": 0,
        "latency_ms": 5,
        "tokens_per_second": 0.0,
        "cost_usd": 0.0,
        "error": "inference_failed",
    }
    payload.update(overrides)
    return InferenceResponse(**payload)


@pytest.mark.parametrize(
    ("error_code", "expected_status"),
    [
        ("context_length_exceeded", 413),
        ("prompt_too_large", 413),
        ("model_not_entitled", 403),
        ("rate_limited", 429),
        ("provider_unavailable", 503),
        ("retrieval_unavailable", 503),
        ("timeout", 504),
        (None, 502),
        ("something_new", 502),
    ],
)
def test_route_response_status_maps_error_codes(error_code, expected_status):
    status, headers = route_response_status(_error_response(error_code=error_code))

    assert status == expected_status
    assert headers == {}


def test_route_response_status_success_is_200():
    response = _error_response(error=None, response_text="ok")

    assert route_response_status(response) == (200, {})


@pytest.mark.parametrize("error_code", ["rate_limited", "provider_unavailable"])
def test_route_response_status_sends_retry_after_when_provider_gave_one(error_code):
    response = _error_response(error_code=error_code, retry_after_seconds=7)

    status, headers = route_response_status(response)

    assert headers == {"Retry-After": "7"}


def _failing_engine(exc, *, event_producer=None):
    router = MagicMock()
    router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    engine = InferenceEngine(
        {"cache": {"enabled": True}, "batching": {"enabled": False}},
        router,
        event_producer=event_producer,
    )
    engine.providers = {"openai": AsyncMock()}
    engine.providers["openai"].generate_response = AsyncMock(side_effect=exc)
    engine._get_local_fallback_models = MagicMock(return_value=[])
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    return engine


@pytest.mark.asyncio
async def test_process_query_error_response_carries_error_code_and_retry_after(
    sample_query_request,
):
    event_producer = AsyncMock()
    exc = _ProviderStatusError(429, headers={"retry-after": "7"})
    engine = _failing_engine(exc, event_producer=event_producer)

    response = await engine.process_query(sample_query_request)

    # The existing contract stays: generic error, "error" provider sentinel.
    assert response.error == "inference_failed"
    assert response.provider == "error"
    assert response.error_type == "rate_limited"
    assert response.error_code == "rate_limited"
    assert response.retry_after_seconds == 7
    published = event_producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(sample_query_request, published)
    assert event["error_type"] == "rate_limited"
    assert event["error_code"] == "rate_limited"


@pytest.mark.asyncio
async def test_process_query_reads_retry_after_ms(sample_query_request):
    exc = _ProviderStatusError(429, headers={"retry-after-ms": "1500"})
    engine = _failing_engine(exc)

    response = await engine.process_query(sample_query_request)

    assert response.retry_after_seconds == 2


@pytest.mark.asyncio
async def test_process_query_context_length_error_code(sample_query_request):
    engine = _failing_engine(_ProviderStatusError(400, _ANTHROPIC_PROMPT_TOO_LONG))

    response = await engine.process_query(sample_query_request)

    assert response.error_type == "user_or_policy"
    assert response.error_code == "context_length_exceeded"
    assert response.retry_after_seconds is None


@pytest.mark.asyncio
async def test_process_query_error_code_comes_from_typed_router_rejection(
    sample_query_request,
):
    engine = _failing_engine(RuntimeError("unused"))
    engine.router.route_query = AsyncMock(
        side_effect=_TypedRejection(
            "tier", error_code="model_not_entitled", http_status=403
        )
    )

    response = await engine.process_query(sample_query_request)

    assert response.error_code == "model_not_entitled"
    assert response.error_type == "user_or_policy"


@pytest.mark.asyncio
async def test_process_query_logs_the_request_id(sample_query_request, caplog):
    engine = _failing_engine(_ProviderStatusError(503))

    with caplog.at_level(logging.ERROR, logger=inference_module.__name__):
        await engine.process_query(sample_query_request)

    failures = [
        record.getMessage()
        for record in caplog.records
        if "Inference failed" in record.getMessage()
    ]
    assert failures == [
        f"Inference failed for request {sample_query_request.request_id}"
    ]


_uses_platform_doubles = pytest.mark.usefixtures("patched_platform_deps")


def _route_client(tmp_path, response):
    config_path = _write_config(tmp_path)
    platform = main.LLMRouterPlatform(config_path=str(config_path))
    inference = DummyInferenceEngine({}, DummyRouter({}))
    inference.response = response
    platform.services["inference"] = inference
    return TestClient(platform._create_fastapi_app())


@pytest.mark.parametrize(
    ("error_code", "retry_after", "expected_status", "expected_retry_after"),
    [
        ("context_length_exceeded", None, 413, None),
        ("prompt_too_large", None, 413, None),
        ("model_not_entitled", None, 403, None),
        ("rate_limited", 7, 429, "7"),
        ("rate_limited", None, 429, None),
        ("provider_unavailable", None, 503, None),
        ("retrieval_unavailable", None, 503, None),
        ("timeout", None, 504, None),
        (None, None, 502, None),
    ],
)
@_uses_platform_doubles
def test_route_endpoint_maps_error_code_to_http_status(
    tmp_path, error_code, retry_after, expected_status, expected_retry_after
):
    response = _error_response(error_code=error_code, retry_after_seconds=retry_after)

    with _route_client(tmp_path, response) as client:
        result = client.post("/route", json={"query": "hello", "user_id": "u1"})

    assert result.status_code == expected_status
    assert result.headers.get("retry-after") == expected_retry_after
    body = result.json()
    assert body["error"] == "inference_failed"
    assert body["provider"] == "error"
    assert body["error_code"] == error_code
