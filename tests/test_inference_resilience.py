"""Resilience of the chat inference path: deadlines, fallbacks and guardrails."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from src.admission import AdmissionDecision
from src.llm_router_part2_inference import (
    AnthropicProvider,
    InferenceEngine,
    OpenAIProvider,
    vLLMProvider,
)
from src.provider_scheduler import (
    ProviderCapacityScheduler,
    RequestExecutionBudget,
    SchedulerLease,
)
from src.utils.schema import InferenceResponse, ProviderSchedulerConfig, QueryRequest


class _StatusError(Exception):
    """Provider SDK error shape: the HTTP status is an attribute (no SDK in CI)."""

    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _request(**overrides):
    payload = {"query": "Explain the outage", "user_id": "u1"}
    payload.update(overrides)
    return QueryRequest(**payload)


def _response(model_name, provider="openai"):
    return InferenceResponse(
        response_text=f"answer from {model_name}",
        model_name=model_name,
        provider=provider,
        token_count_input=1,
        token_count_output=1,
        total_tokens=2,
        latency_ms=1,
        tokens_per_second=1.0,
        cost_usd=0.0,
    )


async def _hang(*_args):
    await asyncio.Event().wait()


class _Admission:
    """Admits every provider call; with no Redis eval, circuits stay local."""

    def __init__(self):
        self.calls = []

    async def acquire_provider(self, **kwargs):
        self.calls.append(kwargs)
        return AdmissionDecision.allow(
            SimpleNamespace(reservation_id=f"r{len(self.calls)}", stage="provider")
        )

    async def release(self, _reservation, *, actual_tokens=None):
        return True


_CIRCUIT = {
    "failure_mode": "open",
    "circuit_breaker": {"enabled": True, "failure_threshold": 1},
}
# The shipped configs allow 3 transport attempts per request. The count is
# shared, so a fallback only runs while the primary has left some unused.
_RETRY = {"retry": {"max_attempts_per_request": 3, "initial_backoff_ms": 0}}


def _mock_engine(providers, *, scheduler, admission=None):
    """Engine over a mocked router: providers maps model name to provider name."""
    router = MagicMock()
    router.get_model_info.side_effect = lambda name: (
        {"config": {"provider": providers[name]}} if name in providers else None
    )
    router.update_model_stats = MagicMock()
    router.route_query = AsyncMock(
        return_value=SimpleNamespace(selected_model=next(iter(providers)))
    )
    return InferenceEngine(
        {"scheduler": scheduler}, router, admission_controller=admission
    )


# --- I5: per-attempt timeouts leave time for the fallback ---------------------


def test_scheduler_config_reserves_thirty_percent_for_fallback_by_default():
    assert ProviderSchedulerConfig().fallback_reserve_ratio == 0.3
    with pytest.raises(ValidationError):
        ProviderSchedulerConfig(fallback_reserve_ratio=1.0)

    assert (
        ProviderCapacityScheduler({}).create_execution_budget().fallback_reserve_ratio
        == 0.3
    )
    budget = ProviderCapacityScheduler(
        {"fallback_reserve_ratio": 0.5, "request_deadline_seconds": 20}
    ).create_execution_budget()
    assert budget.fallback_reserve_ratio == 0.5
    assert budget.deadline_seconds == 20


def test_attempt_timeout_is_provider_timeout_capped_by_remaining_minus_reserve():
    budget = RequestExecutionBudget(
        max_attempts=3,
        deadline_seconds=10,
        started_at=time.monotonic() - 2,
        fallback_reserve_ratio=0.3,
    )

    reserve = budget.fallback_reserve_seconds()

    assert reserve == pytest.approx(2.4, abs=0.05)
    assert budget.attempt_timeout_seconds(60, reserve) == pytest.approx(5.6, abs=0.05)
    assert budget.attempt_timeout_seconds(None, reserve) == pytest.approx(5.6, abs=0.05)
    assert budget.attempt_timeout_seconds(4, reserve) == 4
    # Without a fallback route the primary may use all the remaining time.
    assert budget.attempt_timeout_seconds(60) == pytest.approx(8, abs=0.05)
    assert budget.attempt_timeout_seconds(60, 9) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallbacks", "expected_reserve"),
    [([], 0.0), (["mistral-7b"], 3.0)],
    ids=["no-fallback", "fallback"],
)
async def test_primary_attempt_reserves_time_only_when_a_fallback_exists(
    fallbacks, expected_reserve
):
    engine = _mock_engine(
        {"gpt-5": "openai", "mistral-7b": "vllm"},
        scheduler={"request_deadline_seconds": 10, "fallback_reserve_ratio": 0.3},
    )
    engine.providers = {"openai": OpenAIProvider({"timeout": 45})}
    engine._get_local_fallback_models = MagicMock(return_value=fallbacks)
    calls = []

    async def run_provider_call(**kwargs):
        calls.append(kwargs)
        return _response("gpt-5")

    engine.scheduler.run_provider_call = run_provider_call

    await engine._generate_response_with_fallback(
        _request(),
        SimpleNamespace(selected_model="gpt-5"),
        execution_budget=engine.scheduler.create_execution_budget(),
    )

    (call,) = calls
    assert call["attempt_timeout_seconds"] == 45
    assert call["fallback_reserve_seconds"] == pytest.approx(expected_reserve, abs=0.05)


@pytest.mark.asyncio
async def test_hung_primary_leaves_time_for_fallback_within_deadline():
    engine = _mock_engine(
        {"gpt-5": "openai", "mistral-7b": "vllm"},
        scheduler={
            "request_deadline_seconds": 0.5,
            "fallback_reserve_ratio": 0.3,
            **_RETRY,
        },
    )
    fallback = AsyncMock(return_value=_response("mistral-7b", "vllm"))
    engine.providers = {
        "openai": SimpleNamespace(generate_response=_hang),
        "vllm": SimpleNamespace(generate_response=fallback),
    }
    engine._get_local_fallback_models = MagicMock(return_value=["mistral-7b"])

    started = time.monotonic()
    response = await asyncio.wait_for(engine.process_query(_request()), timeout=3)
    elapsed = time.monotonic() - started

    assert response.error is None
    assert response.model_name == "mistral-7b"
    assert elapsed < 0.5
    fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_hung_primary_attempt_counts_against_its_circuit():
    engine = _mock_engine(
        {"gpt-5": "openai", "mistral-7b": "vllm"},
        scheduler={"request_deadline_seconds": 0.3, **_CIRCUIT, **_RETRY},
        admission=_Admission(),
    )
    engine.providers = {
        "openai": SimpleNamespace(generate_response=_hang),
        "vllm": SimpleNamespace(
            generate_response=AsyncMock(return_value=_response("mistral-7b", "vllm"))
        ),
    }
    engine._get_local_fallback_models = MagicMock(return_value=["mistral-7b"])

    response, model_name, _ = await engine._generate_response_with_fallback(
        _request(),
        SimpleNamespace(selected_model="gpt-5"),
        execution_budget=engine.scheduler.create_execution_budget(),
    )

    assert model_name == "mistral-7b"
    assert response.model_name == "mistral-7b"
    state = await engine.scheduler._read_circuit_state("openai", "gpt-5")
    assert state["state"] == "open"
    vllm_state = await engine.scheduler._read_circuit_state("vllm", "mistral-7b")
    assert vllm_state.get("failure_count", 0) == 0


@pytest.mark.asyncio
async def test_attempt_timeout_counts_as_circuit_failure():
    scheduler = ProviderCapacityScheduler(_CIRCUIT, _Admission())

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run_provider_call(
            request=_request(),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=_hang,
            attempt_timeout_seconds=0.02,
        )

    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state["state"] == "open"


@pytest.mark.asyncio
async def test_retry_timing_out_in_the_queue_is_not_a_provider_failure():
    scheduler = ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {"enabled": True, "failure_threshold": 2},
            **_RETRY,
        },
        _Admission(),
    )
    acquires = []

    async def acquire(**_kwargs):
        acquires.append(_kwargs)
        if len(acquires) > 1:
            await _hang()
        return SchedulerLease(provider="openai", model_name="gpt-5")

    scheduler.acquire = acquire
    operation = AsyncMock(side_effect=_StatusError(503))

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run_provider_call(
            request=_request(),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=operation,
            execution_budget=RequestExecutionBudget(
                max_attempts=3, deadline_seconds=0.2
            ),
            attempt_timeout_seconds=0.05,
        )

    # Only the 503 reached the provider; later tries never left the queue.
    assert operation.await_count == 1
    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state["failure_count"] == 1
    assert state["state"] == "closed"


class _ControlledBudget(RequestExecutionBudget):
    """A budget whose deadline passes when the test says so."""

    expired = False

    @property
    def remaining_seconds(self) -> float:
        return 0.0 if self.expired else 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deadline_passed", "expected_state"),
    [(True, "open"), (False, None)],
    ids=["deadline", "caller-went-away"],
)
async def test_deadline_cancellation_of_a_hung_call_counts_as_circuit_failure(
    deadline_passed, expected_state
):
    scheduler = ProviderCapacityScheduler(_CIRCUIT, _Admission())
    budget = _ControlledBudget(max_attempts=1, deadline_seconds=5)
    started = asyncio.Event()

    async def hang():
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        scheduler.run_provider_call(
            request=_request(),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=hang,
            execution_budget=budget,
        )
    )
    await started.wait()
    budget.expired = deadline_passed
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state.get("state") == expected_state


@pytest.mark.asyncio
async def test_primary_retries_stop_at_the_fallback_reserve():
    scheduler = ProviderCapacityScheduler(
        {"retry": {"max_attempts_per_request": 3, "initial_backoff_ms": 0}},
        _Admission(),
    )
    budget = RequestExecutionBudget(max_attempts=3, deadline_seconds=0.3)
    operation = AsyncMock(side_effect=_hang)

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run_provider_call(
            request=_request(),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=operation,
            execution_budget=budget,
            fallback_reserve_seconds=0.1,
        )

    assert operation.await_count == 1
    assert budget.attempts_used == 1
    assert budget.remaining_seconds > 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_cls", "client_path"),
    [
        (OpenAIProvider, "src.llm_router_part2_inference.openai.AsyncOpenAI"),
        (
            AnthropicProvider,
            "src.llm_router_part2_inference.anthropic.AsyncAnthropic",
        ),
    ],
)
@pytest.mark.parametrize(("configured", "read"), [(45, 45), (None, 60)])
async def test_cloud_clients_get_short_connect_and_provider_read_timeout(
    provider_cls, client_path, configured, read
):
    with patch(client_path) as client:
        await provider_cls({"api_key": "test", "timeout": configured}).initialize()

    assert client.call_args.kwargs["timeout"] == httpx.Timeout(
        connect=3, read=read, write=10, pool=2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("configured", "read"), [(120, 120), (None, 300)])
async def test_vllm_client_gets_short_connect_and_provider_read_timeout(
    configured, read
):
    provider = vLLMProvider(
        {"base_url": "http://127.0.0.1:8001", "timeout": configured}
    )
    provider._refresh_endpoint_health = AsyncMock()
    provider._refresh_endpoint_metrics = AsyncMock()

    with patch("src.llm_router_part2_inference.httpx.AsyncClient") as client:
        await provider.initialize()

    assert client.call_args.kwargs["timeout"] == httpx.Timeout(
        connect=3, read=read, write=10, pool=2
    )


@pytest.mark.asyncio
async def test_slow_rag_shortens_the_llm_window_instead_of_extending_the_deadline():
    engine = _mock_engine(
        {"gpt-5": "openai"}, scheduler={"request_deadline_seconds": 0.3}
    )
    engine.providers = {"openai": SimpleNamespace(generate_response=_hang)}
    engine._get_local_fallback_models = MagicMock(return_value=[])

    async def slow_rag(*_args):
        await asyncio.sleep(0.2)
        return None

    engine._run_rag_if_needed = slow_rag

    started = time.monotonic()
    response = await asyncio.wait_for(engine.process_query(_request()), timeout=3)
    elapsed = time.monotonic() - started

    assert response.error == "inference_failed"
    assert elapsed < 0.42


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["_run_rag_if_needed", "_run_web_search_if_needed"])
async def test_hung_retrieval_is_capped_by_the_request_deadline(stage):
    engine = _mock_engine(
        {"gpt-5": "openai"}, scheduler={"request_deadline_seconds": 0.2}
    )
    provider = AsyncMock(return_value=_response("gpt-5"))
    engine.providers = {"openai": SimpleNamespace(generate_response=provider)}
    setattr(engine, stage, _hang)

    started = time.monotonic()
    response = await asyncio.wait_for(engine.process_query(_request()), timeout=2)
    elapsed = time.monotonic() - started

    assert response.error == "inference_failed"
    assert elapsed < 0.35
    provider.assert_not_awaited()
