"""Context fit: input plus output against each model's context window (R7)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from src.llm_router_part1_router import ModelRouter, PromptTooLargeError
from src.llm_router_part2_inference import InferenceEngine
from src.utils.metrics import ROUTER_METRICS
from src.utils.schema import (
    ModelConfig,
    ModelSelection,
    PlatformConfig,
    QueryRequest,
    QueryType,
    UserTier,
)

ROOT = Path(__file__).resolve().parents[1]
ALL_TIERS = ["free", "premium", "enterprise"]


def _model(provider, window, **extra):
    return {
        "provider": provider,
        "max_tokens": window,
        "cost_per_token": 0.0,
        "priority": 3,
        "allowed_tiers": ALL_TIERS,
        "capabilities": ["general", "coding"],
        **extra,
    }


def _router(models, *, token_count=1000, enriched_tokens=0, default_model=None):
    router = ModelRouter(
        {
            "default_model": default_model or next(iter(models)),
            "routing_strategy": "intelligent",
            "tokenizer_enabled": False,
            "classifier": {"semantic_enabled": False},
            "models": models,
            "routing_rules": [],
        }
    )
    router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.9)

    def count_tokens(prompt, _model="default"):
        extra = enriched_tokens if "ENRICHED" in prompt else 0
        return token_count + extra

    router.token_counter.count_tokens = count_tokens
    return router


def _request(**overrides):
    payload = {"query": "Explain this", "user_id": "u1", "user_tier": UserTier.FREE}
    payload.update(overrides)
    return QueryRequest(**payload)


class TestModelConfigWindow:
    def test_context_window_defaults_to_max_tokens(self):
        config = ModelConfig(**_model("vllm", 32768), name="qwen")

        assert config.context_window == 32768
        assert config.max_output_tokens is None

    def test_explicit_window_and_output_cap(self):
        config = ModelConfig(
            **_model("anthropic", 1_000_000),
            name="sonnet",
            context_window=1_000_000,
            max_output_tokens=128_000,
        )

        assert config.context_window == 1_000_000
        assert config.max_output_tokens == 128_000

    def test_output_cap_must_fit_the_window(self):
        with pytest.raises(ValueError):
            ModelConfig(
                **_model("vllm", 8192),
                name="small",
                context_window=8192,
                max_output_tokens=9000,
            )

    @pytest.mark.parametrize(
        "filename", ["config.yaml", "config.compose.yaml", "config.demo.yaml"]
    )
    def test_checked_in_windows(self, filename):
        payload = yaml.safe_load((ROOT / "config" / filename).read_text())
        models = PlatformConfig.model_validate({"router": payload["router"]}).router
        expected = {
            "qwen3.6-27b-fast": (32768, None),
            "mistral-7b": (8192, None),
            "llama-3.1-70b": (128000, None),
            "gpt-5": (128000, None),
            "claude-sonnet-4-6": (1_000_000, 128_000),
        }
        raw_models = payload["router"]["models"]
        for name, model in models.models.items():
            assert "context_window" in raw_models[name], name
            assert (model.context_window, model.max_output_tokens) == expected[name]

    def test_qwen_window_matches_vllm_max_model_len(self):
        payload = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
        qwen = payload["router"]["models"]["qwen3.6-27b-fast"]

        assert qwen["context_window"] == payload["inference"]["vllm"]["max_model_len"]


class TestRouterFit:
    @pytest.mark.asyncio
    async def test_30k_prompt_with_4096_output_excludes_qwen(self):
        router = _router(
            {
                "qwen3.6-27b-fast": _model("vllm", 32768),
                "llama-3.1-70b": _model("vllm", 128000),
            },
            token_count=30_000,
        )

        decision = await router.route_query(
            _request(
                max_tokens=4096, metadata={"preferred_models": ["qwen3.6-27b-fast"]}
            )
        )

        assert decision.selected_model == "llama-3.1-70b"

    @pytest.mark.asyncio
    async def test_prompt_that_fits_only_without_output_is_too_large(self):
        router = _router(
            {"qwen3.6-27b-fast": _model("vllm", 32768)}, token_count=30_000
        )

        with pytest.raises(PromptTooLargeError) as exc_info:
            await router.route_query(_request(max_tokens=4096))

        assert exc_info.value.max_context_tokens == 32768
        assert exc_info.value.token_count == 30_000

    @pytest.mark.asyncio
    async def test_output_cap_bounds_the_reserved_output(self):
        router = _router(
            {"capped": _model("vllm", 32768, max_output_tokens=1024)},
            token_count=30_000,
        )

        decision = await router.route_query(_request(max_tokens=4096))

        assert decision.selected_model == "capped"

    def test_large_context_models_count_input_and_output(self):
        router = _router(
            {
                "qwen3.6-27b-fast": _model("vllm", 32768),
                "llama-3.1-70b": _model("vllm", 128000),
            }
        )

        models = router._models_with_large_context(
            {"token_count": 30_000, "max_tokens": 4096}
        )

        assert models == ["llama-3.1-70b"]

    def test_score_counts_output_in_context_fit(self):
        router = _router({"qwen3.6-27b-fast": _model("vllm", 32768)})
        base = {"query_type": "general", "user_tier": "free", "max_tokens": 0}

        # 26,000 input tokens fit fully within 80% of the window; with 4,096
        # output tokens reserved they no longer do.
        roomy = router._score_model("qwen3.6-27b-fast", {**base, "token_count": 26_000})
        tight = router._score_model(
            "qwen3.6-27b-fast", {**base, "token_count": 26_000, "max_tokens": 4096}
        )

        assert roomy > tight

    @pytest.mark.asyncio
    async def test_refit_reroutes_an_enriched_prompt_within_the_tier(self):
        router = _router(
            {
                "mistral-7b": _model("vllm", 8192),
                "llama-3.1-70b": _model("vllm", 128000),
            },
            enriched_tokens=20_000,
        )
        request = _request(metadata={"preferred_models": ["mistral-7b"]})
        decision = await router.route_query(request)
        assert decision.selected_model == "mistral-7b"

        request.metadata["web_search_context"] = "ENRICHED web results"
        refit = await router.refit_routing_decision(request, decision)

        assert refit.selected_model == "llama-3.1-70b"
        assert refit.token_count == 21_000
        assert "context window" in refit.routing_reason

    @pytest.mark.asyncio
    async def test_refit_keeps_a_model_that_still_fits(self):
        router = _router(
            {
                "mistral-7b": _model("vllm", 8192),
                "llama-3.1-70b": _model("vllm", 128000),
            },
            enriched_tokens=2_000,
        )
        request = _request(metadata={"preferred_models": ["mistral-7b"]})
        decision = await router.route_query(request)

        request.metadata["rag_context"] = "ENRICHED passages"
        refit = await router.refit_routing_decision(request, decision)

        assert refit.selected_model == "mistral-7b"
        assert refit.token_count == 3_000

    @pytest.mark.asyncio
    async def test_refit_never_upgrades_past_the_tier(self):
        router = _router(
            {
                "mistral-7b": _model("vllm", 8192),
                "gpt-5": {
                    **_model("openai", 128000),
                    "allowed_tiers": ["enterprise"],
                },
            },
            enriched_tokens=20_000,
        )
        request = _request()
        decision = await router.route_query(request)

        request.metadata["web_search_context"] = "ENRICHED web results"
        with pytest.raises(PromptTooLargeError):
            await router.refit_routing_decision(request, decision)


def _engine(router, providers):
    engine = InferenceEngine({}, router)
    engine.providers = providers
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()
    return engine


def _enrich_with_web_context(engine):
    async def run_web_search(request, _decision, _limit):
        request.metadata = dict(request.metadata or {})
        request.metadata["web_search_context"] = "ENRICHED web results"
        return None

    engine._run_web_search_if_needed = run_web_search


class TestInferenceFit:
    @pytest.mark.asyncio
    async def test_enriched_prompt_is_rerouted_before_the_provider_call(
        self, inference_response_factory
    ):
        router = _router(
            {
                "mistral-7b": _model("vllm", 8192),
                "gpt-5": _model("openai", 128000),
            },
            enriched_tokens=20_000,
        )
        vllm = AsyncMock()
        openai = AsyncMock()
        openai.generate_response.return_value = inference_response_factory(
            model_name="gpt-5"
        )
        engine = _engine(router, {"vllm": vllm, "openai": openai})
        _enrich_with_web_context(engine)

        response = await engine.process_query(
            _request(metadata={"preferred_models": ["mistral-7b"]})
        )

        assert response.error is None
        vllm.generate_response.assert_not_called()
        assert openai.generate_response.await_args.args[1] == "gpt-5"

    @pytest.mark.asyncio
    async def test_enriched_prompt_that_fits_no_model_is_413(self):
        router = _router({"mistral-7b": _model("vllm", 8192)}, enriched_tokens=20_000)
        vllm = AsyncMock()
        engine = _engine(router, {"vllm": vllm})
        _enrich_with_web_context(engine)

        response = await engine.process_query(_request())

        assert response.error_code == "prompt_too_large"
        vllm.generate_response.assert_not_called()
        # The prompt was too large; the model did not fail.
        assert router.model_stats.get("mistral-7b", {}).get("total_requests", 0) == 0

    @pytest.mark.asyncio
    async def test_fallback_candidates_use_the_enriched_token_count(
        self, inference_response_factory
    ):
        router = _router(
            {
                "gpt-5": _model("openai", 128000),
                "mistral-7b": _model("vllm", 8192),
            },
            enriched_tokens=20_000,
        )
        vllm = AsyncMock()
        vllm.generate_response.return_value = inference_response_factory(
            model_name="mistral-7b"
        )
        openai = AsyncMock()
        openai.generate_response.side_effect = ConnectionError("openai down")
        engine = _engine(router, {"vllm": vllm, "openai": openai})
        _enrich_with_web_context(engine)

        response = await engine.process_query(
            _request(metadata={"preferred_models": ["gpt-5"]})
        )

        assert response.error is not None
        vllm.generate_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_admission_estimate_includes_enrichment(
        self, inference_response_factory
    ):
        router = _router({"gpt-5": _model("openai", 128000)}, enriched_tokens=20_000)
        openai = AsyncMock()
        openai.generate_response.return_value = inference_response_factory()
        engine = _engine(router, {"openai": openai})
        _enrich_with_web_context(engine)
        estimates = []
        estimate = engine._estimate_provider_admission_input_tokens

        def recording_estimate(request, model_name=None):
            value = estimate(request, model_name)
            estimates.append(value)
            return value

        engine._estimate_provider_admission_input_tokens = recording_estimate

        await engine.process_query(_request())

        assert estimates == [21_000]

    @pytest.mark.asyncio
    async def test_max_tokens_is_clamped_to_the_model_output_cap(
        self, inference_response_factory
    ):
        router = _router({"capped": _model("vllm", 32768, max_output_tokens=1024)})
        vllm = AsyncMock()
        vllm.generate_response.return_value = inference_response_factory(
            model_name="capped"
        )
        engine = _engine(router, {"vllm": vllm})
        request = _request(max_tokens=4096)

        await engine.process_query(request)

        sent_request = vllm.generate_response.await_args.args[0]
        assert sent_request.max_tokens == 1024
        assert request.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_vision_request_no_vision_model_fits_uses_text_only(self):
        router = _router(
            {
                "small-vision": _model(
                    "openai", 8192, capabilities=["general", "vision"]
                ),
                "llama-3.1-70b": _model("vllm", 128000),
            },
            enriched_tokens=20_000,
        )
        engine = _engine(router, {})
        request = _request(metadata={"preferred_models": ["small-vision"]})
        decision = await router.route_query(request)
        request._rag_images = ["image"]
        request.metadata["rag_context"] = "ENRICHED passages"

        refit = await engine._fit_enriched_prompt(request, decision, None)

        assert refit.selected_model == "llama-3.1-70b"
        assert request._rag_images == []
        assert any(
            "no vision model fits" in warning
            for warning in request.metadata["rag_warnings"]
        )

    @pytest.mark.asyncio
    async def test_vision_request_reroutes_to_a_vision_model_that_fits(self):
        router = _router(
            {
                "small-vision": _model(
                    "openai", 8192, capabilities=["general", "vision"]
                ),
                "big-vision": _model(
                    "anthropic", 128000, capabilities=["general", "vision"]
                ),
                "llama-3.1-70b": _model("vllm", 128000),
            },
            enriched_tokens=20_000,
        )
        engine = _engine(router, {})
        request = _request(metadata={"preferred_models": ["small-vision"]})
        decision = await router.route_query(request)
        request._rag_images = ["image"]
        request.metadata["rag_context"] = "ENRICHED passages"

        refit = await engine._fit_enriched_prompt(request, decision, None)

        assert refit.selected_model == "big-vision"
        assert request._rag_images == ["image"]


def _metric(metric, **labels):
    return metric.labels(**labels)._value.get()


class TestRefitReview:
    @pytest.mark.asyncio
    async def test_refit_error_keeps_the_routing_error_decision(
        self, inference_response_factory
    ):
        router = _router({"mistral-7b": _model("vllm", 8192)}, enriched_tokens=100)

        def broken(_query):
            raise RuntimeError("classifier exploded")

        router.classifier.classify_query = broken
        vllm = AsyncMock()
        vllm.generate_response.return_value = inference_response_factory(
            model_name="mistral-7b"
        )
        engine = _engine(router, {"vllm": vllm})
        _enrich_with_web_context(engine)
        errors_before = _metric(ROUTER_METRICS.refit_outcomes, outcome="error")

        response = await engine.process_query(_request())

        assert response.error is None
        assert vllm.generate_response.await_args.args[1] == "mistral-7b"
        assert _metric(ROUTER_METRICS.refit_outcomes, outcome="error") == (
            errors_before + 1
        )
        assert router.model_stats["mistral-7b"]["success_rate"] > 0.95

    @pytest.mark.asyncio
    async def test_refit_still_rejects_a_prompt_no_model_fits(self):
        router = _router({"mistral-7b": _model("vllm", 8192)}, enriched_tokens=20_000)
        request = _request()
        decision = await router.route_query(request)
        request.metadata["web_search_context"] = "ENRICHED web results"

        with pytest.raises(PromptTooLargeError):
            await router.refit_routing_decision(request, decision)

    @pytest.mark.asyncio
    async def test_reroute_recomputes_cost_fast_lane_and_metrics(self):
        router = _router(
            {
                "fast": _model("vllm", 8192),
                "big": {**_model("openai", 128000), "cost_per_token": 0.001},
            },
            enriched_tokens=20_000,
        )
        router.fast_lane_models = ["fast"]
        request = _request(metadata={"preferred_models": ["fast"]})
        decision = (await router.route_query(request)).model_copy(
            update={"route_to_fast_lane": True, "actual_fast_lane_hit": True}
        )
        assert decision.estimated_cost == 0.0
        decisions_before = _metric(
            ROUTER_METRICS.routing_decisions, model="big", query_type="general"
        )
        rerouted_before = _metric(ROUTER_METRICS.refit_outcomes, outcome="rerouted")

        request.metadata["web_search_context"] = "ENRICHED web results"
        refit = await router.refit_routing_decision(request, decision)

        assert refit.selected_model == "big"
        assert refit.estimated_cost == pytest.approx(
            router._estimate_cost(
                ModelSelection(model_name="big", confidence=0.5, reason="r"),
                {"token_count": 21_000, "max_tokens": 512},
            )
        )
        assert refit.estimated_cost > 0
        assert refit.actual_fast_lane_hit is False
        assert refit.confidence <= decision.confidence
        # route_query already counted this request's routing decision.
        assert _metric(
            ROUTER_METRICS.routing_decisions, model="big", query_type="general"
        ) == (decisions_before)
        assert _metric(ROUTER_METRICS.refit_outcomes, outcome="rerouted") == (
            rerouted_before + 1
        )

    @pytest.mark.asyncio
    async def test_reroute_errors_keep_the_decision_with_the_enriched_count(
        self, monkeypatch
    ):
        router = _router(
            {
                "mistral-7b": _model("vllm", 8192),
                "llama-3.1-70b": _model("vllm", 128000),
            },
            enriched_tokens=20_000,
        )
        request = _request(metadata={"preferred_models": ["mistral-7b"]})
        decision = await router.route_query(request)
        errors_before = _metric(ROUTER_METRICS.refit_outcomes, outcome="error")

        def broken(_candidates, _context):
            raise RuntimeError("scoring exploded")

        monkeypatch.setattr(router, "_select_best_model", broken)
        request.metadata["web_search_context"] = "ENRICHED web results"
        refit = await router.refit_routing_decision(request, decision)

        assert refit.selected_model == "mistral-7b"
        assert refit.token_count == 21_000
        assert _metric(ROUTER_METRICS.refit_outcomes, outcome="error") == (
            errors_before + 1
        )

    @pytest.mark.asyncio
    async def test_context_errors_keep_the_decision_with_the_enriched_count(self):
        router = _router({"mistral-7b": _model("vllm", 8192)}, enriched_tokens=100)

        def broken(_query):
            raise RuntimeError("classifier exploded")

        router.classifier.classify_query = broken
        request = _request()
        decision = await router.route_query(request)
        assert decision.token_count == 1000

        request.metadata["web_search_context"] = "ENRICHED web results"
        refit = await router.refit_routing_decision(request, decision)

        assert refit.selected_model == "mistral-7b"
        assert refit.token_count == 1100

    @pytest.mark.asyncio
    async def test_no_enrichment_skips_the_refit(self, inference_response_factory):
        router = _router({"mistral-7b": _model("vllm", 8192)})
        vllm = AsyncMock()
        vllm.generate_response.return_value = inference_response_factory(
            model_name="mistral-7b"
        )
        engine = _engine(router, {"vllm": vllm})
        refit = AsyncMock(side_effect=router.refit_routing_decision)
        router.refit_routing_decision = refit

        response = await engine.process_query(_request())

        assert response.error is None
        refit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enrichment_runs_the_refit(self, inference_response_factory):
        router = _router({"mistral-7b": _model("vllm", 8192)}, enriched_tokens=10)
        vllm = AsyncMock()
        vllm.generate_response.return_value = inference_response_factory(
            model_name="mistral-7b"
        )
        engine = _engine(router, {"vllm": vllm})
        _enrich_with_web_context(engine)
        refit = AsyncMock(side_effect=router.refit_routing_decision)
        router.refit_routing_decision = refit

        await engine.process_query(_request())

        refit.assert_awaited_once()


def _text_only_router():
    return _router(
        {
            "vision": _model("openai", 128000, capabilities=["general", "vision"]),
            "small-text": _model("vllm", 8192),
            "big-text": _model("vllm", 128000),
        }
    )


class TestTextOnlyDowngradeFit:
    def _request(self, text_fallbacks):
        request = _request(
            metadata={
                "rag_original_model": "small-text",
                "rag_text_fallback_models": text_fallbacks,
            }
        )
        request._rag_images = ["image"]
        return request

    def _decision(self):
        return SimpleNamespace(
            selected_model="vision",
            token_count=21_000,
            query_type="general",
            routing_reason="RAG route",
        )

    @pytest.mark.asyncio
    async def test_downgrade_skips_text_models_the_enriched_prompt_overflows(
        self, inference_response_factory
    ):
        router = _text_only_router()
        vllm = AsyncMock()
        vllm.generate_response.return_value = inference_response_factory(
            model_name="big-text"
        )
        engine = _engine(router, {"vllm": vllm, "openai": AsyncMock()})

        (
            _response,
            model_name,
            _decision,
        ) = await engine._execute_text_only_after_vision_failure(
            self._request(["big-text"]), self._decision(), None
        )

        assert model_name == "big-text"
        assert vllm.generate_response.await_args.args[1] == "big-text"

    @pytest.mark.asyncio
    async def test_downgrade_with_no_fitting_text_model_is_413(self):
        router = _text_only_router()
        vllm = AsyncMock()
        engine = _engine(router, {"vllm": vllm, "openai": AsyncMock()})
        decision = self._decision()
        decision.selected_model = "small-text"

        with pytest.raises(PromptTooLargeError):
            await engine._execute_text_only_after_vision_failure(
                self._request([]), decision, None
            )
        vllm.generate_response.assert_not_called()
