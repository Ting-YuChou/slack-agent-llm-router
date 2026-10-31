"""T2 (inference part): queue_wait_ms = ingress wait + provider scheduler wait."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api_security import CLIENT_METADATA_KEYS, filter_external_metadata
from src.llm_router_part2_inference import InferenceEngine
from src.llm_router_part3_pipeline import build_inference_completed_event
from src.provider_scheduler import ProviderCapacityScheduler, SchedulerLease
from src.utils.schema import InferenceResponse, QueryRequest, UserTier


class _Admission:
    async def acquire_provider(self, **_kwargs):  # pragma: no cover - unused
        raise AssertionError("acquire is mocked")


def _scheduler(**config):
    scheduler = ProviderCapacityScheduler(config, _Admission())
    scheduler.record_success = AsyncMock()
    scheduler.record_failure = AsyncMock()
    scheduler.release = AsyncMock(return_value=True)
    return scheduler


def _leases(*waits):
    return [
        SchedulerLease(provider="openai", model_name="gpt-5", wait_seconds=wait)
        for wait in waits
    ]


def _request(**metadata):
    return QueryRequest(
        query="hello", user_id="u1", user_tier=UserTier.PREMIUM, metadata=metadata
    )


@pytest.mark.asyncio
async def test_scheduler_reports_the_lease_wait_of_every_attempt():
    scheduler = _scheduler(retry={"max_attempts_per_request": 2})
    scheduler.acquire = AsyncMock(side_effect=_leases(0.1, 0.25))
    scheduler._consume_retry_budget = AsyncMock(return_value=True)
    scheduler._retry_backoff_seconds = lambda _attempt: 0.0
    operation = AsyncMock(side_effect=[RuntimeError("503 overloaded"), "ok"])
    waits = []

    result = await scheduler.run_provider_call(
        request=_request(),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        operation=operation,
        on_queue_wait=waits.append,
    )

    assert result == "ok"
    assert waits == [0.1, 0.25]


@pytest.mark.asyncio
async def test_scheduler_without_a_wait_sink_still_runs():
    scheduler = _scheduler()
    scheduler.acquire = AsyncMock(side_effect=_leases(0.1))

    result = await scheduler.run_provider_call(
        request=_request(),
        model_name="gpt-5",
        provider="openai",
        estimated_input_tokens=1,
        operation=AsyncMock(return_value="ok"),
    )

    assert result == "ok"


def _engine(inference_response_factory, *, cache=None, leases=(0.04,)):
    router = MagicMock()
    router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    event_producer = AsyncMock()
    engine = InferenceEngine(
        {"cache": {"enabled": True}, "scheduler": {}},
        router,
        event_producer=event_producer,
    )
    provider = AsyncMock()
    provider.generate_response = AsyncMock(return_value=inference_response_factory())
    engine.providers = {"openai": provider}
    engine.scheduler.acquire = AsyncMock(side_effect=_leases(*leases))
    engine.scheduler.record_success = AsyncMock()
    engine.scheduler.release = AsyncMock(return_value=True)
    engine.cache.get_cached_response = AsyncMock(return_value=cache)
    engine.cache.cache_response = AsyncMock()
    return engine, event_producer


@pytest.mark.asyncio
async def test_response_sums_ingress_and_scheduler_waits(inference_response_factory):
    engine, producer = _engine(inference_response_factory)
    request = _request(ingress_queue_wait_ms=15)

    response = await engine.process_query(request)

    assert response.queue_wait_ms == 55
    published = producer.produce_inference_completed.await_args.args[1]
    event = build_inference_completed_event(request, published)
    assert event["queue_wait_ms"] == 55
    assert event["ttft_ms"] is None
    cached = engine.cache.cache_response.await_args.args[1]
    assert "queue_wait_ms" not in cached


@pytest.mark.asyncio
async def test_scheduler_wait_alone_is_reported(inference_response_factory):
    engine, _producer = _engine(inference_response_factory, leases=(0.0,))

    response = await engine.process_query(_request())

    assert response.queue_wait_ms == 0


@pytest.mark.asyncio
async def test_no_source_leaves_queue_wait_unset(inference_response_factory):
    engine, _producer = _engine(inference_response_factory)
    engine.scheduler.enabled = False

    response = await engine.process_query(_request())

    assert response.queue_wait_ms is None


@pytest.mark.asyncio
async def test_cache_hit_reports_this_requests_wait(inference_response_factory):
    stale = inference_response_factory().model_dump(mode="json")
    stale["queue_wait_ms"] = 999
    engine, producer = _engine(inference_response_factory, cache=stale)

    hit = await engine.process_query(_request(ingress_queue_wait_ms=7))
    miss = await engine.process_query(_request())

    assert hit.cached is True
    assert hit.queue_wait_ms == 7
    assert miss.queue_wait_ms is None
    engine.scheduler.acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_error_response_carries_the_wait(inference_response_factory):
    engine, producer = _engine(inference_response_factory)
    engine.providers["openai"].generate_response = AsyncMock(
        side_effect=ValueError("bad request")
    )

    response = await engine.process_query(_request(ingress_queue_wait_ms=3))

    assert response.error == "inference_failed"
    assert response.queue_wait_ms == 43


def test_ingress_wait_is_server_only_metadata():
    request = _request(ingress_queue_wait_ms=5, preferred_models=["gpt-5"])

    dropped = filter_external_metadata(request)

    assert "ingress_queue_wait_ms" not in CLIENT_METADATA_KEYS
    assert dropped == ["ingress_queue_wait_ms"]
    assert request.metadata == {"preferred_models": ["gpt-5"]}


def test_inference_response_queue_wait_defaults_to_none():
    response = InferenceResponse(
        response_text="hi",
        model_name="gpt-5",
        provider="openai",
        token_count_input=1,
        token_count_output=1,
        total_tokens=2,
        latency_ms=1,
        tokens_per_second=1.0,
        cost_usd=0.0,
    )

    assert response.queue_wait_ms is None
    with pytest.raises(ValueError):
        response.model_validate({**response.model_dump(), "queue_wait_ms": -1})
