"""An exhausted billing account is an outage of that provider, not a bad request.

Anthropic says so with a 400 ("credit balance is too low"), OpenAI with a
429 (code insufficient_quota). Neither clears by retrying the same provider,
but both should open its circuit and let the fallback chain cross providers.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.provider_scheduler import ProviderCapacityScheduler

# The resilience tests' fallback-chain engine: a real router, gpt-5 <-> claude.
from test_inference_resilience import (
    _ENTERPRISE,
    _Admission,
    _chain_engine,
    _request,
    _response,
    _router,
    _shipped_scheduler,
)
from src.utils.schema import QueryType, UserTier


class _ProviderStatusError(Exception):
    def __init__(self, status_code, message, *, code=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.response = SimpleNamespace(status_code=status_code, headers={})


def _anthropic_low_credit():
    return _ProviderStatusError(
        400,
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'Your credit balance is too low "
        "to access the Anthropic API. Please go to Plans & Billing to upgrade "
        "or purchase credits.'}}",
    )


def _anthropic_bad_request():
    return _ProviderStatusError(
        400,
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'messages: roles must alternate'}}",
    )


def _openai_insufficient_quota():
    return _ProviderStatusError(
        429, "You exceeded your current quota", code="insufficient_quota"
    )


def _openai_rate_limited():
    return _ProviderStatusError(429, "Rate limit reached", code="rate_limit_exceeded")


def _scheduler():
    return ProviderCapacityScheduler(
        {
            "failure_mode": "open",
            "circuit_breaker": {"enabled": True, "failure_threshold": 1},
        },
        _Admission(),
    )


@pytest.mark.parametrize(
    ("make_error", "availability", "retryable"),
    [
        pytest.param(_anthropic_low_credit, True, False, id="anthropic-low-credit"),
        pytest.param(
            _openai_insufficient_quota, True, False, id="openai-insufficient-quota"
        ),
        pytest.param(_anthropic_bad_request, False, False, id="other-400"),
        pytest.param(_openai_rate_limited, True, True, id="plain-429"),
    ],
)
def test_billing_exhaustion_is_an_outage_that_is_not_retried(
    make_error, availability, retryable
):
    scheduler = _scheduler()
    error = make_error()

    assert scheduler.is_availability_failure(error) is availability
    assert scheduler._is_retryable_exception(error) is retryable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("make_error", "counted"),
    [
        pytest.param(_anthropic_low_credit, True, id="anthropic-low-credit"),
        pytest.param(_openai_insufficient_quota, True, id="openai-quota"),
        pytest.param(_anthropic_bad_request, False, id="other-400"),
    ],
)
async def test_billing_exhaustion_counts_toward_the_circuit(make_error, counted):
    scheduler = _scheduler()

    await scheduler.record_failure(
        provider="anthropic", model_name="claude-sonnet-4-6", error=make_error()
    )

    state = await scheduler._read_circuit_state("anthropic", "claude-sonnet-4-6")
    assert (state.get("failure_count", 0) > 0) is counted


def _claude_first_engine(anthropic_error):
    router = _router()
    router.route_query = AsyncMock(
        return_value=SimpleNamespace(
            selected_model="claude-sonnet-4-6",
            query_type=QueryType.GENERAL,
            token_count=64,
            routing_reason="Capability-based selection",
        )
    )
    engine = _chain_engine(
        router, None, scheduler=_shipped_scheduler(), admission=_Admission()
    )
    engine.providers["openai"].generate_response = AsyncMock(
        side_effect=lambda _request, model: _response(model, "openai")
    )
    engine.providers["anthropic"].generate_response = AsyncMock(
        side_effect=anthropic_error
    )
    return engine


@pytest.mark.asyncio
async def test_anthropic_out_of_credit_is_answered_by_gpt5():
    engine = _claude_first_engine(_anthropic_low_credit())

    response = await engine.process_query(_request(**_ENTERPRISE))

    assert response.error is None
    assert response.model_name == "gpt-5"
    assert response.fallback_from_model == "claude-sonnet-4-6"
    # Retrying an empty account would only fail again.
    engine.providers["anthropic"].generate_response.assert_awaited_once()
    engine.providers["openai"].generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_other_anthropic_400s_still_do_not_cross_providers():
    engine = _claude_first_engine(_anthropic_bad_request())

    await engine.process_query(_request(**_ENTERPRISE))

    # The local vLLM fallback keeps its own triggers; the chain stays shut.
    engine.providers["openai"].generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_insufficient_quota_is_not_retried_on_openai():
    engine = _chain_engine(
        _router(),
        _openai_insufficient_quota(),
        scheduler=_shipped_scheduler(),
        admission=_Admission(),
    )

    await engine.process_query(_request(user_tier=UserTier.PREMIUM))

    engine.providers["openai"].generate_response.assert_awaited_once()
