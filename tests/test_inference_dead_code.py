"""I13b: the stream and batch inference paths are gone."""

import inspect
from pathlib import Path

import pytest
import yaml

from src.llm_router_part2_inference import (
    AnthropicProvider,
    BaseInferenceProvider,
    InferenceEngine,
    OpenAIProvider,
    vLLMProvider,
)
from src.utils.schema import FeatureFlagsConfig, PlatformConfig, QueryRequest

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
