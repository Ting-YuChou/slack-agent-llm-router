"""I13b: context compression and the stream and batch paths are gone."""

import builtins
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import src.llm_router_part2_inference as inference_module
from src.llm_router_part2_inference import (
    AnthropicProvider,
    BaseInferenceProvider,
    InferenceEngine,
    OpenAIProvider,
    vLLMProvider,
)
from src.utils.metrics import INFERENCE_METRICS
from src.utils.schema import (
    FeatureFlagsConfig,
    InferenceConfig,
    PlatformConfig,
    QueryRequest,
    UserTier,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_CONFIGS = ("config.yaml", "config.compose.yaml", "config.demo.yaml")
DEAD_FEATURE_FLAGS = ("batch_processing", "streaming_responses")


def _load(filename):
    return yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8"))


def test_stream_and_batch_paths_are_gone():
    for provider_class in (
        BaseInferenceProvider,
        OpenAIProvider,
        AnthropicProvider,
        vLLMProvider,
    ):
        assert not hasattr(provider_class, "stream_response"), provider_class
        assert not hasattr(provider_class, "generate_batch_responses")
    assert not hasattr(vLLMProvider, "_extract_stream_delta")
    assert (
        "stream"
        not in inspect.signature(vLLMProvider._build_request_payload).parameters
    )
    assert not hasattr(InferenceEngine, "stream_query")
    assert not hasattr(InferenceEngine, "_stream_cached_response")


@pytest.mark.parametrize("api_mode", ["completions", "chat_completions"])
def test_vllm_payload_never_asks_for_a_stream(api_mode):
    provider = vLLMProvider({"base_url": "http://127.0.0.1:8001", "api_mode": api_mode})
    request = QueryRequest(query="hi", user_id="u1")

    _path, payload = provider._build_request_payload("hi", "mistral-7b", request)

    assert payload["stream"] is False


def test_dead_feature_flags_are_gone():
    for flag in DEAD_FEATURE_FLAGS:
        assert flag not in FeatureFlagsConfig.model_fields


@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
def test_checked_in_configs_drop_dead_stream_keys_and_still_load(filename):
    payload = _load(filename)

    assert "stream_cache_chunk_chars" not in (payload.get("inference") or {})
    for flag in DEAD_FEATURE_FLAGS:
        assert flag not in (payload.get("features") or {})
    PlatformConfig.model_validate(payload)


def test_context_compressor_is_gone():
    assert not hasattr(inference_module, "ContextCompressor")
    assert not hasattr(InferenceEngine({}, MagicMock()), "context_compressor")
    assert not hasattr(InferenceEngine, "_compress_request_context_if_needed")
    assert not hasattr(INFERENCE_METRICS, "context_compressions")
    assert not hasattr(INFERENCE_METRICS, "compression_ratio")
    assert "compression" not in InferenceConfig.model_fields
    assert "context_compression" not in FeatureFlagsConfig.model_fields


@pytest.mark.asyncio
async def test_initialize_never_imports_transformers(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("transformers must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    router = MagicMock()
    router.models = {}
    engine = InferenceEngine(
        {"compression": {"enabled": True}, "cache": {"enabled": False}}, router
    )

    await engine.initialize()

    assert "compression_enabled" not in engine.get_health_status()


@pytest.mark.asyncio
async def test_long_context_reaches_the_provider_unchanged(
    inference_response_factory,
):
    router = MagicMock()
    router.route_query = AsyncMock(return_value=SimpleNamespace(selected_model="gpt-5"))
    router.get_model_info.return_value = {"config": {"provider": "openai"}}
    router.update_model_stats = MagicMock()
    engine = InferenceEngine(
        {
            "compression": {"enabled": True, "max_context_tokens": 10},
            "cache": {"enabled": False},
        },
        router,
    )
    provider = AsyncMock()
    provider.generate_response = AsyncMock(return_value=inference_response_factory())
    engine.providers = {"openai": provider}
    context = "x" * 100_000
    request = QueryRequest(
        query="Summarize", user_id="u1", user_tier=UserTier.PREMIUM, context=context
    )

    response = await engine.process_query(request)

    assert provider.generate_response.await_args.args[0].context == context
    assert response.compressed_context is False


@pytest.mark.parametrize("filename", CHECKED_IN_CONFIGS)
def test_checked_in_configs_drop_compression(filename):
    payload = _load(filename)

    assert "compression" not in (payload.get("inference") or {})
    assert "context_compression" not in (payload.get("features") or {})
    PlatformConfig.model_validate(payload)
