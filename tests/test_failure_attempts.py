"""I12: a failed request reports how many model calls it actually started."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part3_pipeline import build_inference_completed_event

# The resilience tests' engines: a real router with gpt-5 <-> claude chains.
from test_inference_resilience import (
    _ENTERPRISE,
    _StatusError,
    _chain_engine,
    _hang,
    _mock_engine,
    _request,
    _router,
)


def _event(engine, request):
    published = engine.event_producer.produce_inference_completed.await_args.args[1]
    return build_inference_completed_event(request, published)


@pytest.mark.asyncio
async def test_failure_before_any_model_call_reports_zero_attempts():
    router = _router()
    router.route_query = AsyncMock(side_effect=ValueError("no model fits"))
    engine = _chain_engine(router, None)
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    assert response.attempts == 0
    assert _event(engine, request)["attempts"] == 0


@pytest.mark.asyncio
async def test_failed_primary_and_fallback_report_two_attempts():
    engine = _chain_engine(
        _router(models=("gpt-5", "claude-sonnet-4-6")), _StatusError(503)
    )
    engine.providers["anthropic"].generate_response = AsyncMock(
        side_effect=_StatusError(503)
    )
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    engine.providers["openai"].generate_response.assert_awaited_once()
    engine.providers["anthropic"].generate_response.assert_awaited_once()
    assert response.attempts == 2
    assert _event(engine, request)["attempts"] == 2


@pytest.mark.asyncio
async def test_failed_primary_without_fallback_reports_one_attempt():
    engine = _chain_engine(_router(models=("gpt-5",)), _StatusError(503), chains={})
    request = _request(**_ENTERPRISE)

    response = await engine.process_query(request)

    assert response.error == "inference_failed"
    assert response.attempts == 1


@pytest.mark.asyncio
async def test_primary_cut_off_by_the_deadline_still_counts_as_started():
    engine = _mock_engine(
        {"gpt-5": "openai"}, scheduler={"request_deadline_seconds": 0.3}
    )
    engine.providers = {"openai": SimpleNamespace(generate_response=_hang)}
    engine._get_local_fallback_models = MagicMock(return_value=[])
    engine.event_producer = AsyncMock()

    response = await asyncio.wait_for(engine.process_query(_request()), timeout=3)

    assert response.error == "inference_failed"
    assert response.attempts == 1
