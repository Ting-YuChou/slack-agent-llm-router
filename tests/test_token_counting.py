"""Token counting: the offline fallback, per-model encodings and tokenizer status."""

import builtins
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.llm_router_part1_router import ModelRouter, TokenCounter
from src.llm_router_part2_inference import InferenceEngine
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import QueryRequest


class _TaggedEncoder:
    """An encoder that records which encoding counted a text."""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def encode(self, text):
        self.calls.append(self.name)
        return text.split()


@pytest.fixture
def encoding_spy(monkeypatch):
    """Replace the tiktoken stub's loaders; returns (loaded names, encode calls)."""
    tiktoken = sys.modules["tiktoken"]
    loaded = []
    calls = []

    def get_encoding(name):
        loaded.append(name)
        return _TaggedEncoder(name, calls)

    def encoding_for_model(model_name):
        raise AssertionError(f"encodings are loaded by name, not for {model_name}")

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    monkeypatch.setattr(tiktoken, "encoding_for_model", encoding_for_model)
    return loaded, calls


def _block_tiktoken_import(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("No module named 'tiktoken'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class TestOfflineFallback:
    def test_disabled_tokenizer_counts_each_cjk_character(self):
        counter = TokenCounter(enabled=False)

        assert counter.count_tokens("中" * 1000) >= 500
        assert counter.count_tokens("中" * 1000) == 1000

    def test_fallback_adds_cjk_characters_to_scaled_words(self):
        counter = TokenCounter(enabled=False)

        # 4 CJK characters plus int(2 words * 1.3).
        assert counter.count_tokens("請幫我看 hello world") == 4 + 2

    def test_fallback_keeps_the_word_approximation_for_latin_text(self):
        counter = TokenCounter(enabled=False)

        assert counter.count_tokens("one two three") == 3

    def test_encode_failure_falls_back_to_the_cjk_aware_estimate(self, monkeypatch):
        counter = TokenCounter()

        class BrokenEncoder:
            def encode(self, _text):
                raise RuntimeError("encoder broke")

        monkeypatch.setattr(
            counter,
            "encoders",
            {name: BrokenEncoder() for name in counter.encoders},
        )

        assert counter.count_tokens("中" * 1000) == 1000

    def test_router_fallback_estimate_counts_cjk(self, router_config, monkeypatch):
        router = ModelRouter(router_config)

        def broken(_request, *_args, **_kwargs):
            raise RuntimeError("estimate failed")

        monkeypatch.setattr(router, "_estimate_request_token_count", broken)
        request = QueryRequest(query="中" * 1000, user_id="u1")

        assert router._fallback_token_estimate(request) >= 1000

    def test_text_without_spaces_counts_four_chars_per_token(self):
        counter = TokenCounter(enabled=False)

        # base64 or minified JSON is one "word" however long it is.
        assert counter.count_tokens("QUJD" * 15_000) >= 15_000
        assert counter.count_tokens('{"a":1,"b":[2,3]}' * 1000) >= 4000

    def test_cjk_and_unspaced_text_both_count(self):
        counter = TokenCounter(enabled=False)

        assert counter.count_tokens("中" * 1000 + "x" * 4000) == 1000 + 1000


class TestAdmissionEstimateOffTheLoop:
    @pytest.mark.asyncio
    async def test_long_gpt5_admission_estimate_runs_in_a_thread(
        self, router_config, monkeypatch, inference_response_factory
    ):
        router = ModelRouter(router_config)
        engine = InferenceEngine({}, router)
        provider = SimpleNamespace(
            generate_response=AsyncMock(return_value=inference_response_factory())
        )
        engine.providers = {"openai": provider}
        request = QueryRequest(query="Summarize", user_id="u1", context="word " * 5000)
        # Routing counted the prompt under cl100k; gpt-5 admission needs o200k.
        await router._build_query_context(request)
        threaded = []

        async def to_thread(func, *args, **kwargs):
            threaded.append(getattr(func, "__name__", repr(func)))
            return func(*args, **kwargs)

        monkeypatch.setattr(
            "src.llm_router_part2_inference.asyncio.to_thread", to_thread
        )

        await engine._execute_model_request(request, "gpt-5")

        assert "_estimate_provider_admission_input_tokens" in threaded

    @pytest.mark.asyncio
    async def test_short_or_memoized_admission_estimates_stay_inline(
        self, router_config, monkeypatch, inference_response_factory
    ):
        router = ModelRouter(router_config)
        engine = InferenceEngine({}, router)
        provider = SimpleNamespace(
            generate_response=AsyncMock(return_value=inference_response_factory())
        )
        engine.providers = {"openai": provider}
        threaded = []

        async def to_thread(func, *args, **kwargs):
            threaded.append(func)
            return func(*args, **kwargs)

        monkeypatch.setattr(
            "src.llm_router_part2_inference.asyncio.to_thread", to_thread
        )

        await engine._execute_model_request(
            QueryRequest(query="Short question", user_id="u1"), "gpt-5"
        )

        assert threaded == []


class TestModelEncodings:
    def test_default_encoding_stays_cl100k(self, encoding_spy):
        _loaded, calls = encoding_spy
        counter = TokenCounter()

        counter.count_tokens("one two")
        counter.count_tokens("one two", "unknown-model")
        counter.count_tokens("one two", "claude-sonnet-4-6")
        counter.count_tokens("one two", "qwen3.6-27b-fast")

        assert calls == ["cl100k_base"] * 4

    def test_gpt5_uses_o200k_when_the_model_is_known(self, encoding_spy):
        loaded, calls = encoding_spy
        counter = TokenCounter()

        counter.count_tokens("one two", "gpt-5")

        assert "o200k_base" in loaded
        assert calls == ["o200k_base"]
        assert counter.encoding_name("gpt-5") == "o200k_base"
        assert counter.encoding_name("default") == "cl100k_base"

    def test_gpt5_uses_cl100k_when_o200k_is_missing(self, monkeypatch, caplog):
        tiktoken = sys.modules["tiktoken"]
        calls = []

        def get_encoding(name):
            if name == "o200k_base":
                raise ValueError("o200k_base is not cached")
            return _TaggedEncoder(name, calls)

        monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
        with caplog.at_level(logging.ERROR, logger="src.llm_router_part1_router"):
            counter = TokenCounter()

        counter.count_tokens("one two", "gpt-5")

        assert calls == ["cl100k_base"]
        assert counter.encoding_name("gpt-5") == "cl100k_base"
        assert any("o200k_base" in record.getMessage() for record in caplog.records)

    def test_router_estimate_forwards_the_model(self, router_config):
        router = ModelRouter(router_config)
        seen = []

        def count_tokens(prompt, model="default"):
            seen.append(model)
            return len(prompt.split())

        router.token_counter.count_tokens = count_tokens
        request = QueryRequest(query="hello there", user_id="u1")

        router._estimate_request_token_count(request)
        router._estimate_request_token_count(request, "gpt-5")

        assert seen == ["default", "gpt-5"]

    def test_admission_estimate_counts_with_the_called_model(self, router_config):
        router = ModelRouter(router_config)
        seen = []

        def count_tokens(prompt, model="default"):
            seen.append(model)
            return 7

        router.token_counter.count_tokens = count_tokens
        engine = InferenceEngine({}, router)
        request = QueryRequest(query="hello there", user_id="u1")

        assert engine._estimate_provider_admission_input_tokens(request, "gpt-5") == 7
        assert seen == ["gpt-5"]


class TestTokenizerStatus:
    def test_missing_tokenizer_logs_one_error_and_reports_unavailable(
        self, monkeypatch, caplog
    ):
        _block_tiktoken_import(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="src.llm_router_part1_router"):
            counter = TokenCounter()
            for _ in range(3):
                counter.count_tokens("中" * 10)

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert counter.count_tokens("中" * 10) == 10
        status = counter.status()
        assert status["enabled"] is True
        assert status["available"] is False
        assert status["encodings"] == {"cl100k_base": False, "o200k_base": False}
        assert (
            ROUTER_METRICS.tokenizer_available.labels(
                encoding="cl100k_base"
            )._value.get()
            == 0
        )

    def test_available_tokenizer_reports_each_encoding(self):
        counter = TokenCounter()

        status = counter.status()

        assert status["available"] is True
        assert status["encodings"] == {"cl100k_base": True, "o200k_base": True}
        assert (
            ROUTER_METRICS.tokenizer_available.labels(
                encoding="o200k_base"
            )._value.get()
            == 1
        )

    def test_encode_failure_logs_one_error(self, monkeypatch, caplog):
        counter = TokenCounter()

        class BrokenEncoder:
            def encode(self, _text):
                raise RuntimeError("encoder broke")

        monkeypatch.setattr(
            counter,
            "encoders",
            {name: BrokenEncoder() for name in counter.encoders},
        )
        with caplog.at_level(logging.ERROR, logger="src.llm_router_part1_router"):
            for _ in range(3):
                counter.count_tokens("one two")

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1

    def test_missing_tokenizer_does_not_fail_router_health(
        self, router_config, monkeypatch
    ):
        _block_tiktoken_import(monkeypatch)
        router = ModelRouter(router_config)
        router._initialized = True
        router.classifier._is_initialized = True

        assert router.is_healthy() is True
        assert router.tokenizer_status()["available"] is False

    def test_readiness_reports_tokenizer_details_without_failing(
        self, router_config, monkeypatch
    ):
        import main

        _block_tiktoken_import(monkeypatch)
        router = ModelRouter(router_config)
        router._initialized = True
        router.classifier._is_initialized = True
        platform = SimpleNamespace(
            services={
                "router": router,
                "inference": SimpleNamespace(is_healthy=lambda: True),
            },
            _api_key_auth_configured=lambda: True,
            _api_key_auth_enabled=lambda: False,
            _api_key_header_name=lambda: "X-API-Key",
        )
        platform._get_service_status = (
            lambda: main.LLMRouterPlatform._get_service_status(platform)
        )

        payload = main.LLMRouterPlatform._build_readiness_payload(platform)

        assert payload["status"] == "ready"
        assert payload["tokenizer"]["available"] is False
