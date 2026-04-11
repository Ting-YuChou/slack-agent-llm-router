from unittest.mock import AsyncMock

import pytest

from src.llm_router_part1_router import (
    ModelRouter,
    QueryClassifier,
    RoutingRule,
    TokenCounter,
)
from src.utils.schema import QueryRequest, QueryType, UserTier


class TestQueryClassifier:
    def test_classify_code_generation_query(self):
        classifier = QueryClassifier()
        classifier._is_initialized = True

        query_type, confidence = classifier.classify_query(
            "Write a Python function to sort a list"
        )

        assert query_type == QueryType.CODE_GENERATION
        assert confidence > 0.1


class TestTokenCounter:
    def test_count_tokens_uses_default_encoder(self):
        counter = TokenCounter()
        token_count = counter.count_tokens("one two three", "unknown-model")

        assert token_count == 3


class TestRoutingRule:
    def test_matches_returns_true_for_matching_context(self):
        rule = RoutingRule(
            condition="query_type == 'code_generation' and token_count < 1000",
            models=["gpt-5"],
            fallback="gpt-5",
        )

        assert (
            rule.matches({"query_type": "code_generation", "token_count": 128}) is True
        )
        assert rule.matches({"query_type": "analysis", "token_count": 128}) is False

    def test_matches_supports_uppercase_logical_operators(self):
        rule = RoutingRule(
            condition="query_type == 'analysis' AND token_count > 500",
            models=["gpt-5"],
            fallback="gpt-5",
        )

        assert rule.matches({"query_type": "analysis", "token_count": 800}) is True


class TestModelRouter:
    @pytest.mark.asyncio
    async def test_route_query_prefers_rule_based_selection(self, router_config):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (
            QueryType.CODE_GENERATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 500}
        router.model_stats["mistral-7b"] = {"success_rate": 0.90, "avg_latency": 300}

        request = QueryRequest(
            query="Write a Python helper", user_id="u1", user_tier=UserTier.PREMIUM
        )
        decision = await router.route_query(request)

        assert decision.selected_model in {"gpt-5", "mistral-7b"}
        assert decision.query_type == QueryType.CODE_GENERATION
        assert "Rule-based selection" in decision.routing_reason

    def test_capability_access_and_stats(self, router_config):
        router = ModelRouter(router_config)
        model = router.models["mistral-7b"]

        assert router._has_capability(model, "code_generation") is True
        assert router._check_user_access(model, "free") is True
        assert router._check_user_access(model, "enterprise") is True

        router.update_model_stats("mistral-7b", success=True, latency_ms=200)
        router.update_model_stats("mistral-7b", success=False, latency_ms=400)

        info = router.get_model_info("mistral-7b")

        assert info["config"]["provider"] == "vllm"
        assert info["stats"]["total_requests"] == 2
        assert info["stats"]["successful_requests"] == 1
        assert info["stats"]["avg_latency"] == 300

    def test_fallback_when_no_model_matches(self, router_config):
        router = ModelRouter(router_config)
        selection = router._capability_based_routing(
            {
                "query_type": "translation",
                "token_count": 999999,
                "user_tier": "free",
            }
        )

        assert selection.model_name == "mistral-7b"
        assert "default" in selection.reason.lower()

    @pytest.mark.asyncio
    async def test_route_query_uses_policy_cache_fast_lane_bias(self, router_config):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id):
                return {
                    "policy_source": "user",
                    "route_to_fast_lane": True,
                    "preferred_models": [],
                    "hint_reason": "priority=critical",
                }

        router = ModelRouter(router_config, policy_cache=PolicyCacheStub())
        router.classifier.classify_query = lambda _query: (
            QueryType.CODE_GENERATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 500}
        router.model_stats["mistral-7b"] = {"success_rate": 0.90, "avg_latency": 100}

        request = QueryRequest(
            query="Urgent fix for production issue",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "mistral-7b"
        assert "Policy-cache selection" in decision.routing_reason

    @pytest.mark.asyncio
    async def test_route_query_prefers_explicit_fast_lane_model(self):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id):
                return {
                    "policy_source": "request",
                    "route_to_fast_lane": True,
                    "preferred_models": ["claude-3.5-sonnet"],
                    "hint_reason": "priority=high",
                }

        router = ModelRouter(
            {
                "default_model": "mistral-7b",
                "routing_strategy": "intelligent",
                "fast_lane_models": ["claude-3.5-sonnet"],
                "models": {
                    "claude-3.5-sonnet": {
                        "provider": "anthropic",
                        "max_tokens": 200000,
                        "cost_per_token": 0.000015,
                        "priority": 2,
                        "capabilities": ["reasoning", "writing", "analysis"],
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                    "mistral-7b": {
                        "provider": "vllm",
                        "model_path": "/models/mistral",
                        "max_tokens": 4096,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "coding"],
                        "gpu_memory_gb": 16,
                    },
                },
                "routing_rules": [],
            },
            policy_cache=PolicyCacheStub(),
        )
        router.classifier.classify_query = lambda _query: (QueryType.ANALYSIS, 0.95)
        router.token_counter.count_tokens = lambda _query, _model="default": 256
        router.model_stats["claude-3.5-sonnet"] = {
            "success_rate": 0.99,
            "avg_latency": 700,
        }

        request = QueryRequest(
            query="Please analyze this customer escalation",
            user_id="enterprise-user",
            user_tier=UserTier.ENTERPRISE,
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "claude-3.5-sonnet"
        assert "Policy-cache selection" in decision.routing_reason
