"""CH3: cloud provider clients keep idle connections alive."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm_router_part2_inference import (
    AnthropicProvider,
    InferenceEngine,
    OpenAIProvider,
)

MODULE = "src.llm_router_part2_inference"
PROVIDERS = [
    (OpenAIProvider, "openai", "AsyncOpenAI"),
    (AnthropicProvider, "anthropic", "AsyncAnthropic"),
]


def _sdk(client_name):
    """An SDK double with the two names the provider reads."""
    http_client = MagicMock(name="http_client")
    http_client.aclose = AsyncMock()
    sdk = SimpleNamespace(DefaultAsyncHttpxClient=MagicMock(return_value=http_client))
    setattr(sdk, client_name, MagicMock(name=client_name))
    return sdk, http_client


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_cls", "sdk_name", "client_name"), PROVIDERS)
@pytest.mark.parametrize(("configured", "expected"), [(None, 60.0), (15, 15.0)])
async def test_cloud_clients_keep_idle_connections_for_the_configured_time(
    provider_cls, sdk_name, client_name, configured, expected
):
    sdk, http_client = _sdk(client_name)
    config = {"api_key": "test"}
    if configured is not None:
        config["keepalive_expiry_seconds"] = configured

    with patch(f"{MODULE}.{sdk_name}", sdk):
        await provider_cls(config).initialize()

    limits = sdk.DefaultAsyncHttpxClient.call_args.kwargs["limits"]
    assert limits.keepalive_expiry == expected
    # The pool keeps the SDK's size: a smaller one would hit the 2 s pool timeout.
    assert limits.max_connections == 1000
    assert limits.max_keepalive_connections == 100
    assert getattr(sdk, client_name).call_args.kwargs["http_client"] is http_client


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_cls", "sdk_name", "client_name"), PROVIDERS)
@pytest.mark.parametrize("bad", [0, -1, "soon", True])
async def test_an_invalid_keepalive_setting_falls_back_to_the_default(
    provider_cls, sdk_name, client_name, bad
):
    sdk, _ = _sdk(client_name)

    with patch(f"{MODULE}.{sdk_name}", sdk):
        await provider_cls(
            {"api_key": "test", "keepalive_expiry_seconds": bad}
        ).initialize()

    limits = sdk.DefaultAsyncHttpxClient.call_args.kwargs["limits"]
    assert limits.keepalive_expiry == 60.0


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_cls", "sdk_name", "client_name"), PROVIDERS)
async def test_the_provider_timeout_is_still_passed(
    provider_cls, sdk_name, client_name
):
    sdk, _ = _sdk(client_name)

    with patch(f"{MODULE}.{sdk_name}", sdk):
        await provider_cls({"api_key": "test", "timeout": 45}).initialize()

    expected = httpx.Timeout(connect=3, read=45, write=10, pool=2)
    kwargs = getattr(sdk, client_name).call_args.kwargs
    assert kwargs["timeout"] == expected
    assert kwargs["max_retries"] == 0
    assert sdk.DefaultAsyncHttpxClient.call_args.kwargs["timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_cls", "sdk_name", "client_name"), PROVIDERS)
async def test_aclose_closes_the_providers_http_client_once(
    provider_cls, sdk_name, client_name
):
    sdk, http_client = _sdk(client_name)
    provider = provider_cls({"api_key": "test"})
    with patch(f"{MODULE}.{sdk_name}", sdk):
        await provider.initialize()

    await provider.aclose()
    await provider.aclose()

    http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_cls", "sdk_name", "client_name"), PROVIDERS)
async def test_aclose_tolerates_an_uninitialized_provider(
    provider_cls, sdk_name, client_name
):
    await provider_cls({"api_key": "test"}).aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_cls", "sdk_name", "client_name"), PROVIDERS)
async def test_an_sdk_without_a_default_http_client_gets_none(
    provider_cls, sdk_name, client_name
):
    sdk = SimpleNamespace()
    setattr(sdk, client_name, MagicMock(name=client_name))
    provider = provider_cls({"api_key": "test"})

    with patch(f"{MODULE}.{sdk_name}", sdk):
        await provider.initialize()

    assert "http_client" not in getattr(sdk, client_name).call_args.kwargs
    await provider.aclose()


@pytest.mark.asyncio
async def test_engine_shutdown_closes_cloud_provider_http_clients():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.batch_processor = SimpleNamespace(shutdown=AsyncMock())
    engine.tool_registry = SimpleNamespace(close=AsyncMock())
    engine.cache = SimpleNamespace(redis_client=None)
    closed = []
    for provider_cls, sdk_name, client_name in PROVIDERS:
        sdk, http_client = _sdk(client_name)
        provider = provider_cls({"api_key": "test"})
        with patch(f"{MODULE}.{sdk_name}", sdk):
            await provider.initialize()
        engine.providers = {**getattr(engine, "providers", {}), sdk_name: provider}
        closed.append(http_client)

    await engine.shutdown()

    for http_client in closed:
        http_client.aclose.assert_awaited_once()
