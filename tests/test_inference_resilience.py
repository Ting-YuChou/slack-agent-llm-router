"""Resilience of the chat inference path: deadlines, fallbacks and guardrails."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml
from pydantic import ValidationError

from src.admission import AdmissionDecision
from src.llm_router_part2_inference import (
    AnthropicProvider,
    BaseInferenceProvider,
    InferenceEngine,
    OpenAIProvider,
    vLLMProvider,
)
from src.provider_scheduler import (
    ProviderCapacityScheduler,
    RequestExecutionBudget,
    SchedulerLease,
)
from src.utils.schema import (
    InferenceConfig,
    InferenceResponse,
    PlatformConfig,
    ProviderSchedulerConfig,
    QueryRequest,
    QueryType,
    UserTier,
)


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


class _TimedProvider(BaseInferenceProvider):
    """A provider with a real provider timeout; delay=None hangs."""

    def __init__(self, provider_name, delay, *, timeout=60):
        super().__init__({"timeout": timeout})
        self.provider_name = provider_name
        self.delay = delay
        self.calls = []

    async def initialize(self):
        return None

    async def generate_response(self, request, model_name):
        self.calls.append(model_name)
        if self.delay is None:
            await _hang()
        await asyncio.sleep(self.delay)
        return _response(model_name, self.provider_name)

    async def stream_response(self, request, model_name):
        yield ""

    def get_health_status(self):
        return {}


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
    assert ProviderSchedulerConfig().min_attempt_window_seconds == 1.0
    assert ProviderSchedulerConfig().circuit_timeout_min_window_ratio == 0.5
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
    ("fallbacks", "max_attempts", "expected_reserve", "expected_reserve_attempts"),
    [
        ([], 3, 0.0, 0),
        (["mistral-7b"], 3, 3.0, 1),
        # One attempt per request: the fallback can never run, so holding
        # time back would only cut the primary short.
        (["mistral-7b"], 1, 0.0, 0),
    ],
    ids=["no-fallback", "fallback", "fallback-but-one-attempt"],
)
async def test_primary_attempt_reserves_time_only_when_a_fallback_can_run(
    fallbacks, max_attempts, expected_reserve, expected_reserve_attempts
):
    engine = _mock_engine(
        {"gpt-5": "openai", "mistral-7b": "vllm"},
        scheduler={
            "request_deadline_seconds": 10,
            "fallback_reserve_ratio": 0.3,
            "retry": {"max_attempts_per_request": max_attempts},
        },
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
    assert call["reserve_attempts"] == expected_reserve_attempts


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
@pytest.mark.parametrize(
    ("provider_timeout", "expected_failures"),
    [
        # The attempt had its whole provider timeout: a real hang.
        pytest.param(0.05, 1, id="full-provider-timeout"),
        # The reserve left a sliver of a 60 s timeout: says nothing about it.
        # A timeout far beyond the deadline (vLLM's 300 s against 60 s): the
        # hang used every second the request could give it.
        pytest.param(60, 1, id="provider-timeout-beyond-deadline"),
    ],
)
async def test_hung_primary_counts_against_its_circuit_only_with_a_long_window(
    provider_timeout, expected_failures
):
    engine = _mock_engine(
        {"gpt-5": "openai", "mistral-7b": "vllm"},
        scheduler={
            "request_deadline_seconds": 0.3,
            "min_attempt_window_seconds": 0,
            **_CIRCUIT,
            **_RETRY,
        },
        admission=_Admission(),
    )
    engine.providers = {
        "openai": _TimedProvider("openai", None, timeout=provider_timeout),
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
    assert state.get("failure_count", 0) == expected_failures
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
    """A budget with 5 s left until the test says the deadline passed."""

    expired = False

    @property
    def remaining_seconds(self) -> float:
        return 0.0 if self.expired else 5.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deadline_seconds", "deadline_passed", "expected_failures"),
    [
        # The deadline cancelled a call that had the whole 5 s deadline.
        pytest.param(5, True, 1, id="deadline-after-full-window"),
        # Earlier steps had spent 45 s of a 50 s deadline: a sliver.
        pytest.param(50, True, 0, id="deadline-after-cut-short-window"),
        # Cancelled with time left: the caller went away, not the provider.
        pytest.param(5, False, 0, id="caller-went-away"),
    ],
)
async def test_cancelled_call_counts_only_when_the_deadline_ended_a_full_window(
    deadline_seconds, deadline_passed, expected_failures
):
    scheduler = ProviderCapacityScheduler(_CIRCUIT, _Admission())
    budget = _ControlledBudget(max_attempts=1, deadline_seconds=deadline_seconds)
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
    assert state.get("failure_count", 0) == expected_failures


@pytest.mark.asyncio
async def test_hung_primary_without_fallback_counts_when_the_outer_deadline_wins():
    # Single-flight off (the default) and no fallback: the scheduler's attempt
    # timeout and the request's own deadline fire together, and the outer
    # cancel usually wins.
    counted = 0
    for _ in range(5):
        engine = _chain_engine(
            _router(("gpt-5",)),
            None,
            chains={},
            scheduler={
                "request_deadline_seconds": 0.1,
                "min_attempt_window_seconds": 0,
                **_CIRCUIT,
                **_RETRY,
            },
            admission=_Admission(),
        )
        assert engine.single_flight.enabled is False
        engine.providers = {"openai": _TimedProvider("openai", None, timeout=0.1)}

        response = await asyncio.wait_for(
            engine.process_query(_request(**_ENTERPRISE)), timeout=3
        )

        assert response.error == "inference_failed"
        state = await engine.scheduler._read_circuit_state("openai", "gpt-5")
        counted += state.get("failure_count", 0)

    assert counted == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spent_share", "expected_failures"),
    [
        # A 300 s timeout against a 60 s deadline, scaled down: the hang had
        # the whole deadline, the longest window any attempt could get.
        pytest.param(0.0, 1, id="whole-deadline"),
        # Earlier steps spent 70% of the deadline: the 30% left is a sliver.
        pytest.param(0.7, 0, id="deadline-mostly-spent"),
    ],
)
async def test_timeout_counts_against_the_circuit_by_the_window_the_request_allowed(
    spent_share, expected_failures
):
    scheduler = ProviderCapacityScheduler(
        {**_CIRCUIT, "min_attempt_window_seconds": 0}, _Admission()
    )
    deadline = 0.1

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run_provider_call(
            request=_request(),
            model_name="mistral-7b",
            provider="vllm",
            estimated_input_tokens=1,
            operation=_hang,
            execution_budget=RequestExecutionBudget(
                max_attempts=1,
                deadline_seconds=deadline,
                started_at=time.monotonic() - deadline * spent_share,
            ),
            attempt_timeout_seconds=deadline * 5,
        )

    state = await scheduler._read_circuit_state("vllm", "mistral-7b")
    assert state.get("failure_count", 0) == expected_failures


@pytest.mark.asyncio
async def test_attempt_with_less_than_the_minimum_window_never_starts():
    admission = _Admission()
    scheduler = ProviderCapacityScheduler(
        {**_CIRCUIT, "min_attempt_window_seconds": 1.0}, admission
    )
    operation = AsyncMock(side_effect=_hang)

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run_provider_call(
            request=_request(),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=operation,
            execution_budget=RequestExecutionBudget(
                max_attempts=3, deadline_seconds=0.5
            ),
            attempt_timeout_seconds=5,
        )

    assert time.monotonic() - started < 0.1
    operation.assert_not_awaited()
    assert admission.calls == []
    state = await scheduler._read_circuit_state("openai", "gpt-5")
    assert state.get("failure_count", 0) == 0


@pytest.mark.asyncio
async def test_retry_is_not_started_with_less_than_the_minimum_window_left():
    scheduler = ProviderCapacityScheduler(
        {
            "min_attempt_window_seconds": 0.1,
            "retry": {"max_attempts_per_request": 3, "initial_backoff_ms": 50},
        },
        _Admission(),
    )
    operation = AsyncMock(side_effect=_StatusError(503))

    # After the 50 ms backoff only ~70 ms would be left for the retry.
    with pytest.raises(_StatusError):
        await scheduler.run_provider_call(
            request=_request(),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=operation,
            execution_budget=RequestExecutionBudget(
                max_attempts=3, deadline_seconds=0.12
            ),
            attempt_timeout_seconds=5,
        )

    assert operation.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("left_after_wait", [0.03, 0.0], ids=["sliver", "nothing"])
async def test_retry_that_loses_its_window_while_waiting_raises_the_real_failure(
    left_after_wait,
):
    scheduler = ProviderCapacityScheduler(
        {
            "min_attempt_window_seconds": 0.05,
            "retry": {"max_attempts_per_request": 3, "initial_backoff_ms": 0},
        },
        _Admission(),
    )
    deadline = 0.2
    operation = AsyncMock(side_effect=_StatusError(503))

    async def slow_retry_budget(*_args):
        # The retry-budget check (a Redis round trip) eats the window.
        await asyncio.sleep(deadline - left_after_wait)
        return True

    scheduler._consume_retry_budget = slow_retry_budget

    with pytest.raises(_StatusError):
        await scheduler.run_provider_call(
            request=_request(),
            model_name="gpt-5",
            provider="openai",
            estimated_input_tokens=1,
            operation=operation,
            execution_budget=RequestExecutionBudget(
                max_attempts=3, deadline_seconds=deadline
            ),
            attempt_timeout_seconds=5,
        )

    assert operation.await_count == 1


@pytest.mark.asyncio
async def test_slow_rag_does_not_open_healthy_providers_circuits():
    engine = _chain_engine(
        _router(),
        None,
        scheduler=_shipped_scheduler(request_deadline_seconds=1.0),
        admission=_Admission(),
    )
    openai_provider = _TimedProvider("openai", 0.1)
    anthropic_provider = _TimedProvider("anthropic", 0.1)
    engine.providers = {"openai": openai_provider, "anthropic": anthropic_provider}

    async def slow_rag(*_args):
        await asyncio.sleep(0.93)
        return None

    engine._run_rag_if_needed = slow_rag

    response = await engine.process_query(_request(**_ENTERPRISE))

    # ~70 ms were left: too little for either provider, so neither is called.
    assert response.error == "inference_failed"
    assert openai_provider.calls == []
    assert anthropic_provider.calls == []
    for provider, model in (
        ("openai", "gpt-5"),
        ("anthropic", "claude-sonnet-4-6"),
    ):
        state = await engine.scheduler._read_circuit_state(provider, model)
        assert state.get("failure_count", 0) == 0, provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rag_share", "expected_failures"),
    [
        # Window 0.7 x the deadline = 0.7 x the provider timeout: a real hang.
        pytest.param(0.0, 1, id="no-rag"),
        # Window 0.7 x 0.4 = 0.28 x the provider timeout: says nothing.
        pytest.param(0.6, 0, id="rag-used-60-percent"),
    ],
)
async def test_hung_gpt5_trips_its_circuit_under_shipped_timeouts(
    rag_share, expected_failures
):
    # The shipped numbers, 100 times smaller: 60 s provider timeout and
    # deadline, 0.3 reserve, 1 s minimum window.
    scale = 0.01
    scheduler = _shipped_scheduler(
        request_deadline_seconds=60 * scale, min_attempt_window_seconds=1 * scale
    )
    assert scheduler["fallback_reserve_ratio"] == 0.3
    assert scheduler["circuit_timeout_min_window_ratio"] == 0.5
    engine = _chain_engine(_router(), None, scheduler=scheduler, admission=_Admission())
    engine.providers = {
        "openai": _TimedProvider("openai", None, timeout=60 * scale),
        "anthropic": _TimedProvider("anthropic", 0, timeout=60 * scale),
    }

    async def rag(*_args):
        await asyncio.sleep(60 * scale * rag_share)
        return None

    engine._run_rag_if_needed = rag

    response = await asyncio.wait_for(
        engine.process_query(_request(**_ENTERPRISE)), timeout=3
    )

    assert response.error is None
    assert response.model_name == "claude-sonnet-4-6"
    assert response.fallback_from_model == "gpt-5"
    state = await engine.scheduler._read_circuit_state("openai", "gpt-5")
    assert state.get("failure_count", 0) == expected_failures


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
async def test_vllm_client_gets_short_connect_and_provider_timeout_elsewhere(
    configured, read
):
    # Local endpoints queue many concurrent generations on one connection
    # pool: a 2 s pool wait would fail healthy requests (and trip circuits).
    provider = vLLMProvider(
        {"base_url": "http://127.0.0.1:8001", "timeout": configured}
    )
    provider._refresh_endpoint_health = AsyncMock()
    provider._refresh_endpoint_metrics = AsyncMock()

    with patch("src.llm_router_part2_inference.httpx.AsyncClient") as client:
        await provider.initialize()

    assert client.call_args.kwargs["timeout"] == httpx.Timeout(
        connect=3, read=read, write=read, pool=read
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


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["_run_rag_if_needed", "_run_web_search_if_needed"])
async def test_retrieval_cut_off_by_the_deadline_is_not_the_models_failure(stage):
    from src.llm_router_part3_pipeline import build_inference_completed_event

    router = MagicMock()
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
    engine = InferenceEngine(
        {"scheduler": {"request_deadline_seconds": 0.2}},
        router,
        event_producer=AsyncMock(),
    )
    engine.providers = {"openai": SimpleNamespace(generate_response=AsyncMock())}
    setattr(engine, stage, _hang)
    request = _request()

    response = await asyncio.wait_for(engine.process_query(request), timeout=2)

    assert response.error == "inference_failed"
    router.update_model_stats.assert_not_called()
    published = engine.event_producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(request, published)
    assert event["error_type"] not in {"timeout", "provider_error", "rate_limited"}


def test_request_stage_deadline_error_carries_its_http_mapping():
    from src.llm_router_part2_inference import RequestStageDeadlineError

    error = RequestStageDeadlineError("request deadline reached during rag")

    assert not isinstance(error, (TimeoutError, asyncio.TimeoutError, RuntimeError))
    assert (error.error_type, error.error_code, error.http_status) == (
        "internal",
        "timeout",
        504,
    )


def _blame(router, engine, request):
    """Router-stat failures, inference_stats errors and the published event."""
    from src.llm_router_part3_pipeline import build_inference_completed_event

    failures = [
        call.kwargs["model_name"]
        for call in router.update_model_stats.call_args_list
        if call.kwargs.get("success") is False
    ]
    errors = {
        model: stats["error_count"] for model, stats in engine.inference_stats.items()
    }
    published = engine.event_producer.produce_inference_completed.await_args.args[1]
    return failures, errors, build_inference_completed_event(request, published)


@pytest.mark.asyncio
async def test_models_skipped_for_lack_of_time_are_not_blamed():
    from flink.logic import counts_toward_provider_health

    router = _router()
    engine = _chain_engine(
        router,
        None,
        scheduler=_shipped_scheduler(request_deadline_seconds=1.0),
        admission=_Admission(),
    )
    openai_provider = _TimedProvider("openai", 0.1)
    anthropic_provider = _TimedProvider("anthropic", 0.1)
    engine.providers = {"openai": openai_provider, "anthropic": anthropic_provider}

    async def slow_rag(*_args):
        await asyncio.sleep(0.93)  # Ends in time, but leaves less than 1 s.
        return None

    engine._run_rag_if_needed = slow_rag
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    assert openai_provider.calls == anthropic_provider.calls == []
    failures, errors, event = _blame(router, engine, request)
    assert failures == []
    assert errors == {}
    assert not counts_toward_provider_health(event)


@pytest.mark.asyncio
async def test_deadline_spent_before_any_model_call_is_a_stage_deadline():
    from flink.logic import counts_toward_provider_health
    from src.llm_router_part2_inference import RequestStageDeadlineError

    router = _router()
    engine = _chain_engine(router, None, scheduler={"request_deadline_seconds": 0.2})

    async def slow_cache_lookup(*_args):
        await asyncio.sleep(0.25)
        return None

    engine._get_cached_inference_response = slow_cache_lookup
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    engine.providers["openai"].generate_response.assert_not_awaited()
    failures, errors, event = _blame(router, engine, request)
    assert failures == []
    assert not counts_toward_provider_health(event)

    budget = RequestExecutionBudget(
        max_attempts=3, deadline_seconds=1, started_at=time.monotonic() - 2
    )
    with pytest.raises(RequestStageDeadlineError):
        await engine._generate_response_with_fallback(
            _request(), SimpleNamespace(selected_model="gpt-5"), budget
        )


# --- I4: config-driven cross-provider fallback chains --------------------------

_APIConnectionError = type("APIConnectionError", (Exception,), {})
_CHAINS = {"gpt-5": ["claude-sonnet-4-6"], "claude-sonnet-4-6": ["gpt-5"]}
_MODELS = {
    "gpt-5": {
        "provider": "openai",
        "max_tokens": 128000,
        "cost_per_token": 0.00003,
        "priority": 2,
        "capabilities": ["general", "reasoning", "coding", "analysis"],
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "max_tokens": 200000,
        "cost_per_token": 0.00003,
        "priority": 1,
        "capabilities": ["general", "reasoning", "coding", "analysis"],
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "max_tokens": 200000,
        "cost_per_token": 0.00001,
        "priority": 1,
        "capabilities": ["general"],
    },
    "mistral-7b": {
        "provider": "vllm",
        "model_path": "/models/mistral",
        "max_tokens": 4096,
        "cost_per_token": 0.0,
        "priority": 3,
        "capabilities": ["general", "coding"],
    },
}
_ENTERPRISE = {"user_tier": UserTier.ENTERPRISE}


class _PolicyCache:
    """Routing policy as Flink would materialize it: guardrail blocks."""

    def __init__(self, **policy):
        self.policy = policy

    async def get_effective_policy(self, _request_id, _user_id, _session_id=None):
        return dict(self.policy)


class _RejectFirstAdmission(_Admission):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason

    async def acquire_provider(self, **kwargs):
        if not self.calls:
            self.calls.append(kwargs)
            return AdmissionDecision.reject(
                status_code=429,
                error="rate_limited",
                reason=self.reason,
                message="denied",
                retry_after_seconds=1,
            )
        return await super().acquire_provider(**kwargs)


def _router(models=("gpt-5", "claude-sonnet-4-6", "mistral-7b"), policy=None):
    """A real router; models are names from _MODELS or a name -> config map."""
    from src.llm_router_part1_router import ModelRouter

    if not isinstance(models, dict):
        models = {name: _MODELS[name] for name in models}
    router = ModelRouter(
        {
            "default_model": "gpt-5",
            "routing_strategy": "intelligent",
            "models": models,
            "routing_rules": [],
        },
        policy_cache=policy,
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
    return router


def _chain_engine(
    router, primary_error, *, chains=_CHAINS, scheduler=None, admission=None
):
    config = {"fallback": {"chains": chains}}
    if scheduler is not None:
        config["scheduler"] = scheduler
    engine = InferenceEngine(
        config, router, event_producer=AsyncMock(), admission_controller=admission
    )
    primary = (
        primary_error
        if callable(primary_error)
        else AsyncMock(side_effect=primary_error)
    )
    engine.providers = {
        "openai": SimpleNamespace(generate_response=primary),
        "anthropic": SimpleNamespace(
            generate_response=AsyncMock(
                side_effect=lambda _request, model: _response(model, "anthropic")
            )
        ),
        "vllm": SimpleNamespace(
            generate_response=AsyncMock(return_value=_response("mistral-7b", "vllm"))
        ),
    }
    return engine


def _shipped_scheduler(**overrides):
    """The scheduler section of config/config.yaml, as deployed.

    failure_mode becomes "open" so circuits fall back to process-local state:
    the fake admission controller has no Redis eval.
    """
    path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config = PlatformConfig.model_validate(yaml.safe_load(path.read_text()))
    scheduler = config.inference.scheduler.model_dump(mode="json")
    scheduler["failure_mode"] = "open"
    scheduler.update(overrides)
    return scheduler


def _published_event(engine, request):
    from src.llm_router_part3_pipeline import build_inference_completed_event

    producer = engine.event_producer.produce_inference_completed
    producer.assert_awaited_once()
    _request_arg, response, decision = producer.await_args.args
    return build_inference_completed_event(request, response, decision)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "primary_error",
    [
        pytest.param(_StatusError(503), id="5xx"),
        pytest.param(_StatusError(429), id="429"),
        pytest.param(_APIConnectionError("connection reset"), id="connection"),
    ],
)
async def test_unavailable_gpt5_is_answered_by_claude(primary_error):
    engine = _chain_engine(_router(), primary_error)
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.error is None
    assert response.model_name == "claude-sonnet-4-6"
    assert response.fallback_from_model == "gpt-5"
    assert response.attempts == 2
    event = _published_event(engine, request)
    assert event["selected_model"] == "claude-sonnet-4-6"
    assert event["fallback_from_model"] == "gpt-5"
    assert event["attempts"] == 2
    assert event["fallback_models"] == ["claude-sonnet-4-6", "mistral-7b"]
    engine.providers["vllm"].generate_response.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "primary_error",
    [
        pytest.param(_StatusError(503), id="5xx"),
        pytest.param(_StatusError(429), id="429"),
        pytest.param(_APIConnectionError("connection reset"), id="connection"),
    ],
)
async def test_failing_gpt5_leaves_its_chain_an_attempt_under_shipped_scheduler(
    primary_error,
):
    scheduler = _shipped_scheduler()
    assert scheduler["enabled"] is True
    assert scheduler["retry"]["max_attempts_per_request"] == 3
    engine = _chain_engine(
        _router(), primary_error, scheduler=scheduler, admission=_Admission()
    )

    response = await engine.process_query(_request(**_ENTERPRISE))

    # The shared budget allows 3 transport attempts; the primary retries
    # once and leaves the last one to its fallback.
    assert engine.providers["openai"].generate_response.await_count == 2
    assert response.error is None
    assert response.model_name == "claude-sonnet-4-6"
    assert response.fallback_from_model == "gpt-5"


@pytest.mark.asyncio
async def test_fallback_that_never_ran_is_not_marked_failed():
    # One attempt per request: the primary uses it and the chain cannot run.
    scheduler = _shipped_scheduler(retry={"max_attempts_per_request": 1})
    router = _router()
    engine = _chain_engine(
        router, _StatusError(503), scheduler=scheduler, admission=_Admission()
    )

    response = await engine.process_query(_request(**_ENTERPRISE))

    assert response.error == "inference_failed"
    engine.providers["anthropic"].generate_response.assert_not_awaited()
    engine.providers["vllm"].generate_response.assert_not_awaited()
    failed = [
        call.kwargs["model_name"]
        for call in router.update_model_stats.call_args_list
        if call.kwargs["success"] is False
    ]
    assert failed == ["gpt-5"]
    assert "claude-sonnet-4-6" not in engine.inference_stats


@pytest.mark.asyncio
async def test_circuit_open_gpt5_is_answered_by_claude():
    engine = _chain_engine(
        _router(),
        _StatusError(500),
        scheduler={"enabled": True},
        admission=_RejectFirstAdmission("circuit_open"),
    )

    response = await engine.process_query(_request(**_ENTERPRISE))

    assert response.model_name == "claude-sonnet-4-6"
    assert response.fallback_from_model == "gpt-5"
    engine.providers["openai"].generate_response.assert_not_awaited()
    # The open circuit kept gpt-5's provider from being called.
    assert response.attempts == 1


class _RejectModelsAdmission(_Admission):
    """Rejects provider admission for some models, as an open circuit would."""

    def __init__(self, *models):
        super().__init__()
        self.rejected = set(models)

    async def acquire_provider(self, **kwargs):
        if kwargs["model_name"] in self.rejected:
            return AdmissionDecision.reject(
                status_code=429,
                error="rate_limited",
                reason="circuit_open",
                message="denied",
                retry_after_seconds=1,
            )
        return await super().acquire_provider(**kwargs)


@pytest.mark.asyncio
async def test_attempts_count_only_models_whose_provider_was_called():
    engine = _chain_engine(
        _router(("gpt-5", "claude-sonnet-4-6", "claude-haiku-4-5")),
        _StatusError(503),
        chains={"gpt-5": ["claude-sonnet-4-6", "claude-haiku-4-5"]},
        scheduler=_shipped_scheduler(),
        admission=_RejectModelsAdmission("claude-sonnet-4-6"),
    )
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.model_name == "claude-haiku-4-5"
    calls = engine.providers["anthropic"].generate_response.await_args_list
    assert [call.args[1] for call in calls] == ["claude-haiku-4-5"]
    # gpt-5 and claude-haiku-4-5 were called; claude-sonnet-4-6 was skipped.
    assert response.attempts == 2
    assert _published_event(engine, request)["attempts"] == 2


@pytest.mark.asyncio
async def test_single_flight_follower_answered_by_a_fallback_reports_an_attempt():
    from src.llm_router_part2_inference import SingleFlightCoordinator

    engine = _chain_engine(
        _router(),
        None,
        scheduler=_shipped_scheduler(),
        admission=_RejectModelsAdmission("gpt-5"),
    )
    engine.single_flight = SingleFlightCoordinator({"enabled": True})

    async def slow_claude(_request, model):
        await asyncio.sleep(0.05)
        return _response(model, "anthropic")

    engine.providers["anthropic"].generate_response = AsyncMock(side_effect=slow_claude)
    decision = SimpleNamespace(selected_model="gpt-5", token_count=64)

    results = await asyncio.gather(
        *(
            engine._generate_response_with_fallback(_request(**_ENTERPRISE), decision)
            for _ in range(2)
        )
    )

    # One claude call answered both requests; gpt-5 was never called.
    engine.providers["anthropic"].generate_response.assert_awaited_once()
    engine.providers["openai"].generate_response.assert_not_awaited()
    for response, model_name, _decision in results:
        assert model_name == "claude-sonnet-4-6"
        assert response.fallback_from_model == "gpt-5"
        assert response.attempts == 1


@pytest.mark.asyncio
async def test_hung_gpt5_attempt_is_answered_by_claude():
    engine = _chain_engine(
        _router(),
        _hang,
        scheduler={
            "request_deadline_seconds": 0.5,
            "fallback_reserve_ratio": 0.3,
            **_RETRY,
        },
    )

    started = time.monotonic()
    response = await asyncio.wait_for(
        engine.process_query(_request(**_ENTERPRISE)), timeout=3
    )

    assert time.monotonic() - started < 0.5
    assert response.model_name == "claude-sonnet-4-6"
    assert response.fallback_from_model == "gpt-5"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "primary_error",
    [
        pytest.param(_StatusError(400), id="400"),
        pytest.param(_StatusError(413), id="413"),
        pytest.param(_StatusError(422), id="422"),
        pytest.param(ValueError("unsupported parameter"), id="value-error"),
    ],
)
async def test_request_errors_never_cross_providers(primary_error):
    engine = _chain_engine(_router(), primary_error)

    response = await engine.process_query(_request(**_ENTERPRISE))

    engine.providers["anthropic"].generate_response.assert_not_awaited()
    # The local vLLM fallback keeps its existing triggers.
    assert response.model_name == "mistral-7b"
    assert response.fallback_from_model == "gpt-5"


@pytest.mark.asyncio
async def test_chain_fallback_does_not_upgrade_the_user_tier():
    engine = _chain_engine(_router(("gpt-5", "claude-sonnet-4-6")), _StatusError(503))

    response = await engine.process_query(_request(user_tier=UserTier.PREMIUM))

    assert response.error == "inference_failed"
    engine.providers["anthropic"].generate_response.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "expected_model"),
    [
        pytest.param(
            {"blocked_models": ["claude-sonnet-4-6"]},
            "claude-haiku-4-5",
            id="blocked-model-skipped",
        ),
        pytest.param({"blocked_providers": ["anthropic"]}, None, id="blocked-provider"),
    ],
)
async def test_chain_skips_models_blocked_by_guardrails(policy, expected_model):
    router = _router(
        ("gpt-5", "claude-sonnet-4-6", "claude-haiku-4-5"),
        policy=_PolicyCache(**policy),
    )
    engine = _chain_engine(
        router,
        _StatusError(503),
        chains={"gpt-5": ["claude-sonnet-4-6", "claude-haiku-4-5"]},
    )

    response = await engine.process_query(_request(**_ENTERPRISE))

    calls = engine.providers["anthropic"].generate_response.await_args_list
    expected_calls = [expected_model] if expected_model else []
    assert [call.args[1] for call in calls] == expected_calls
    if expected_model:
        assert response.model_name == expected_model
    else:
        assert response.error == "inference_failed"


@pytest.mark.asyncio
async def test_vision_request_skips_the_chain_and_retries_as_text():
    engine = _chain_engine(_router(), _StatusError(503))
    request = _request(
        **_ENTERPRISE,
        metadata={
            "rag_vision_fallback_models": [],
            "rag_text_fallback_models": ["mistral-7b"],
        },
    )
    request._rag_images = [object()]

    response, model_name, _ = await engine._generate_response_with_fallback(
        request, SimpleNamespace(selected_model="gpt-5", routing_reason="vision")
    )

    # Chains do not check capability, so they never take image requests.
    engine.providers["anthropic"].generate_response.assert_not_awaited()
    assert model_name == "mistral-7b"
    assert response.fallback_from_model == "gpt-5"
    assert response.attempts == 2


@pytest.mark.asyncio
async def test_primary_success_reports_one_attempt_and_no_fallback():
    engine = _chain_engine(_router(), None)
    engine.providers["openai"].generate_response = AsyncMock(
        return_value=_response("gpt-5")
    )
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.fallback_from_model is None
    assert response.attempts == 1
    event = _published_event(engine, request)
    assert event["fallback_from_model"] is None
    assert event["attempts"] == 1
    assert event["fallback_models"] == []


@pytest.mark.asyncio
async def test_response_cache_hit_does_not_replay_an_old_fallback():
    engine = _chain_engine(_router(), None)
    cached = _response("claude-sonnet-4-6", "anthropic").model_dump(mode="json")
    cached.update(fallback_from_model="gpt-5", attempts=2)
    engine.cache.get_cached_response = AsyncMock(return_value=cached)

    response = await engine._get_cached_inference_response(
        _request(), "claude-sonnet-4-6"
    )

    assert response.cached is True
    assert response.fallback_from_model is None
    assert response.attempts == 0


def test_inference_config_parses_fallback_chains():
    config = InferenceConfig(fallback={"chains": {"gpt-5": ["claude-sonnet-4-6"]}})

    assert config.fallback.chains == {"gpt-5": ["claude-sonnet-4-6"]}
    assert InferenceConfig().fallback.chains == {}


@pytest.mark.parametrize("config_name", ["config.yaml", "config.compose.yaml"])
def test_shipped_configs_chain_gpt5_and_claude_both_ways(config_name):
    path = Path(__file__).resolve().parents[1] / "config" / config_name
    config = PlatformConfig.model_validate(yaml.safe_load(path.read_text()))

    assert config.inference.fallback.chains == _CHAINS
    assert config.inference.scheduler.fallback_reserve_ratio == 0.3
    assert config.inference.scheduler.min_attempt_window_seconds == 1.0
    raw = yaml.safe_load(path.read_text())["inference"]["scheduler"]
    assert raw["circuit_timeout_min_window_ratio"] == 0.5
    for model, chain in config.inference.fallback.chains.items():
        assert {model, *chain} <= set(config.router.models)


# --- R11b: local and vision fallbacks obey routing guardrails -------------------

_VISION_MODELS = {
    "gpt-5": {**_MODELS["gpt-5"], "capabilities": ["general", "vision"]},
    "claude-sonnet-4-6": {
        **_MODELS["claude-sonnet-4-6"],
        "capabilities": ["general", "vision"],
    },
    "mistral-7b": _MODELS["mistral-7b"],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param({"blocked_models": ["mistral-7b"]}, id="blocked-model"),
        pytest.param({"blocked_providers": ["vllm"]}, id="blocked-provider"),
    ],
)
async def test_local_fallback_skips_a_guardrail_blocked_mistral(policy):
    engine = _chain_engine(
        _router(("gpt-5", "mistral-7b"), policy=_PolicyCache(**policy)),
        _StatusError(503),
        chains={},
    )

    response = await engine.process_query(_request())

    assert response.error == "inference_failed"
    engine.providers["vllm"].generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_configured_vllm_fallback_skips_a_guardrail_blocked_mistral():
    models = {
        "qwen3.6-27b-fast": {**_MODELS["mistral-7b"], "model_path": "/models/qwen"},
        "mistral-7b": _MODELS["mistral-7b"],
    }
    router = _router(models, policy=_PolicyCache(blocked_models=["mistral-7b"]))
    engine = InferenceEngine(
        {
            "vllm": {
                "model_fallback": {
                    "enabled": True,
                    "fallbacks": {"qwen3.6-27b-fast": ["mistral-7b"]},
                }
            }
        },
        router,
    )
    engine.providers = {
        "vllm": SimpleNamespace(
            generate_response=AsyncMock(side_effect=_StatusError(503))
        )
    }
    decision = SimpleNamespace(
        selected_model="qwen3.6-27b-fast",
        query_type=QueryType.GENERAL,
        token_count=64,
        routing_reason="Fast lane",
    )

    with pytest.raises(_StatusError):
        await engine._generate_response_with_fallback(_request(), decision)

    called = [
        call.args[1]
        for call in engine.providers["vllm"].generate_response.await_args_list
    ]
    assert called == ["qwen3.6-27b-fast"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "policy"),
    [
        pytest.param(
            {}, {"blocked_providers": ["anthropic"]}, id="ranked-blocked-provider"
        ),
        pytest.param(
            {"rag_vision_fallback_models": ["claude-sonnet-4-6"]},
            {"blocked_models": ["claude-sonnet-4-6"]},
            id="routed-candidates-blocked-model",
        ),
    ],
)
@pytest.mark.parametrize("path", ["response", "pre-token-stream"])
async def test_vision_fallback_respects_guardrails(metadata, policy, path):
    router = _router(_VISION_MODELS, policy=_PolicyCache(**policy))
    engine = _chain_engine(router, _StatusError(503), chains={})
    request = _request(
        **_ENTERPRISE,
        metadata={**metadata, "rag_text_fallback_models": ["mistral-7b"]},
    )
    request._rag_images = [
        SimpleNamespace(
            figure_id="fig-1",
            content_checksum_sha256="abc",
            asset_ref=SimpleNamespace(index_version="v1", checksum_sha256="abc"),
        )
    ]
    decision = SimpleNamespace(selected_model="gpt-5", routing_reason="vision")

    if path == "response":
        _, model_name, _ = await engine._generate_response_with_fallback(
            request, decision
        )
    else:
        _, model_name, _ = await engine._execute_pre_token_vision_fallback(
            request, decision, "gpt-5"
        )

    engine.providers["anthropic"].generate_response.assert_not_awaited()
    assert model_name == "mistral-7b"


def test_fallback_context_estimates_tokens_when_routing_reported_none():
    engine = InferenceEngine({}, _router())
    request = _request(context="word " * 5000)

    context = engine._build_local_fallback_context(
        request,
        SimpleNamespace(selected_model="gpt-5", token_count=0),
        {"blocked_models": ["mistral-7b"]},
    )

    assert context["token_count"] > 4096
    assert context["blocked_models"] == ["mistral-7b"]
    assert (
        engine._get_local_fallback_models(
            request, SimpleNamespace(selected_model="gpt-5", token_count=0)
        )
        == []
    )


# --- I4: the text-only vision retry respects the user's tier ---------------------


def _vision_request(user_tier, **metadata):
    request = _request(user_tier=user_tier, metadata=metadata)
    request._rag_images = [object()]
    return request


@pytest.mark.asyncio
async def test_text_only_retry_skips_a_model_the_tier_may_not_use():
    engine = _chain_engine(_router(_VISION_MODELS), None, chains={})
    # claude-sonnet-4-6 is enterprise-only; metadata must not grant it.
    request = _vision_request(
        UserTier.PREMIUM,
        rag_original_model="claude-sonnet-4-6",
        rag_text_fallback_models=["mistral-7b"],
    )

    _, model_name, _ = await engine._execute_text_only_after_vision_failure(
        request, SimpleNamespace(selected_model="gpt-5"), execution_budget=None
    )

    engine.providers["anthropic"].generate_response.assert_not_awaited()
    assert model_name == "mistral-7b"
    assert request._rag_images == []


@pytest.mark.asyncio
async def test_text_only_retry_fails_when_no_candidate_fits_the_tier():
    engine = _chain_engine(_router(_VISION_MODELS), None, chains={})
    request = _vision_request(UserTier.FREE, rag_original_model="claude-sonnet-4-6")

    with pytest.raises(ValueError, match="tier"):
        await engine._execute_text_only_after_vision_failure(
            request, SimpleNamespace(selected_model="gpt-5"), execution_budget=None
        )

    engine.providers["anthropic"].generate_response.assert_not_awaited()
    engine.providers["openai"].generate_response.assert_not_awaited()


def _failed_vision_engine(router, text_only_error=None):
    """gpt-5 fails a vision request that has no vision fallback left."""
    engine = _chain_engine(router, _StatusError(503), chains={})
    if text_only_error is not None:
        engine.providers["vllm"].generate_response = AsyncMock(
            side_effect=text_only_error
        )

    async def attach_images(request, routing_decision, _rag_result):
        request._rag_images = [
            SimpleNamespace(
                figure_id="fig-1",
                content_checksum_sha256="abc",
                asset_ref=SimpleNamespace(index_version="v1", checksum_sha256="abc"),
            )
        ]
        return routing_decision

    engine._prepare_rag_images = attach_images
    return engine


@pytest.mark.asyncio
async def test_text_only_retry_with_no_tier_model_does_not_blame_the_primary_twice():
    router = _router(_VISION_MODELS)
    engine = _failed_vision_engine(router)
    request = _request(
        user_tier=UserTier.FREE, metadata={"rag_original_model": "claude-sonnet-4-6"}
    )

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    failures, errors, _event = _blame(router, engine, request)
    assert failures == ["gpt-5"]
    assert errors == {"gpt-5": 1}


@pytest.mark.asyncio
async def test_failed_text_only_retry_is_blamed_on_its_own_model():
    # With anthropic blocked no vision fallback is left, only the text retry.
    router = _router(
        _VISION_MODELS, policy=_PolicyCache(blocked_providers=["anthropic"])
    )
    engine = _failed_vision_engine(router, text_only_error=_StatusError(503))
    request = _request(
        **_ENTERPRISE, metadata={"rag_text_fallback_models": ["mistral-7b"]}
    )

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    engine.providers["vllm"].generate_response.assert_awaited_once()
    failures, errors, _event = _blame(router, engine, request)
    assert sorted(failures) == ["gpt-5", "mistral-7b"]
    assert errors == {"gpt-5": 1, "mistral-7b": 1}
