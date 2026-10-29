"""Routing context is built once per request, and policy reads are batched (R9)."""

import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine
from src.llm_router_part3_policy import RoutingPolicyCache
from src.rag.service import RagImageInput
from src.rag.storage import RagAssetRef
from src.utils.schema import QueryRequest, QueryType, UserTier

ALL_TIERS = ["free", "premium", "enterprise"]
IMAGE = RagImageInput(
    source_rank=1,
    figure_id="fig-1",
    page=7,
    caption="Revenue by year",
    asset_ref=RagAssetRef(
        backend="local",
        uri="rag-asset://kb/doc/v1/fig-1",
        path="/tmp/private.png",
        media_type="image/png",
        checksum_sha256="checksum-a",
        size_bytes=9,
        width=100,
        height=80,
        knowledge_base_id="kb",
        document_id="doc",
        index_version="v1",
        figure_id="fig-1",
    ),
    content=b"png-bytes",
)


def _model(provider, capabilities, window=128000):
    return {
        "provider": provider,
        "max_tokens": window,
        "cost_per_token": 0.0,
        "priority": 3,
        "allowed_tiers": ALL_TIERS,
        "capabilities": capabilities,
    }


class CountingPolicyCache:
    def __init__(self):
        self.calls = 0

    async def get_effective_policy(self, request_id, user_id, session_id=None):
        self.calls += 1
        return {"policy_source": "none", "blocked_models": []}


class TwoArgPolicyCache:
    """Older stubs take (request_id, user_id) only."""

    def __init__(self):
        self.calls = []

    async def get_effective_policy(self, request_id, user_id):
        self.calls.append((request_id, user_id))
        return {"policy_source": "user", "preferred_models": ["text-model"]}


def _router(policy_cache=None):
    router = ModelRouter(
        {
            "default_model": "vision-a",
            "routing_strategy": "intelligent",
            "classifier": {"semantic_enabled": False},
            "models": {
                "vision-a": _model("openai", ["general", "vision"]),
                "vision-b": _model("anthropic", ["general", "vision"]),
                "text-model": _model("vllm", ["general", "coding"]),
            },
            "routing_rules": [],
        },
        policy_cache=policy_cache,
    )
    classify_calls = []

    def classify(query):
        classify_calls.append(query)
        return QueryType.GENERAL, 0.9

    router.classifier.classify_query = classify
    counted = []
    real_count = router.token_counter.count_tokens

    def count_tokens(prompt, model="default"):
        counted.append((prompt, router.token_counter.encoding_name(model)))
        return real_count(prompt, model)

    router.token_counter.count_tokens = count_tokens
    return router, classify_calls, counted


def _request(**overrides):
    payload = {
        "query": "What does the chart show?",
        "user_id": "u1",
        "user_tier": UserTier.PREMIUM,
        "metadata": {"preferred_models": ["vision-a"]},
    }
    payload.update(overrides)
    return QueryRequest(**payload)


def _engine_with_rag_images(router, inference_response_factory):
    rag_service = SimpleNamespace(
        enabled=True,
        visual_config={},
        materialize_image_inputs=AsyncMock(return_value=([IMAGE], [])),
    )
    engine = InferenceEngine({}, router, rag_service=rag_service)
    openai = AsyncMock()
    openai.generate_response.return_value = inference_response_factory(
        model_name="vision-a"
    )
    engine.providers = {"openai": openai, "anthropic": AsyncMock(), "vllm": AsyncMock()}
    engine.cache.get_cached_response = AsyncMock(return_value=None)
    engine.cache.cache_response = AsyncMock()

    async def run_rag(request, _decision, _limit):
        request.metadata = dict(request.metadata or {})
        request.metadata["rag_context"] = "Retrieved passages about the chart"
        return SimpleNamespace(
            results=["hit"],
            required_no_hits=False,
            warnings=[],
            sources=[],
            tool_call=None,
            latency_ms=0,
        )

    engine._run_rag_if_needed = run_rag
    return engine, openai


class TestRoutingContextMemo:
    @pytest.mark.asyncio
    async def test_rag_image_request_classifies_counts_and_reads_policy_once(
        self, inference_response_factory
    ):
        policy_cache = CountingPolicyCache()
        router, classify_calls, counted = _router(policy_cache)
        engine, openai = _engine_with_rag_images(router, inference_response_factory)

        response = await engine.process_query(_request())

        assert response.error is None
        openai.generate_response.assert_awaited_once()
        sent = openai.generate_response.await_args.args[0]
        assert getattr(sent, "_rag_images", []) == [IMAGE]
        assert len(classify_calls) == 1
        assert policy_cache.calls == 1
        # Each distinct prompt is counted once: before RAG for routing, after
        # RAG for the vision candidates, the enriched re-check and admission.
        assert Counter(counted).most_common(1)[0][1] == 1
        assert len(counted) == 2

    @pytest.mark.asyncio
    async def test_enrichment_invalidates_the_token_count_only(self):
        policy_cache = CountingPolicyCache()
        router, classify_calls, counted = _router(policy_cache)
        request = _request()

        first = await router._build_query_context(request)
        again = await router._build_query_context(request)
        request.metadata["web_search_context"] = "fresh web results " * 50
        enriched = await router._build_query_context(request)

        assert again["token_count"] == first["token_count"]
        assert enriched["token_count"] > first["token_count"]
        assert len(counted) == 2
        assert len(classify_calls) == 1
        assert policy_cache.calls == 1

    @pytest.mark.asyncio
    async def test_a_changed_query_is_classified_again(self):
        router, classify_calls, _counted = _router(CountingPolicyCache())
        request = _request()

        await router._build_query_context(request)
        request.query = "A different question"
        await router._build_query_context(request)

        assert classify_calls == ["What does the chart show?", "A different question"]

    @pytest.mark.asyncio
    async def test_two_argument_policy_stubs_still_work_and_are_memoized(self):
        policy_cache = TwoArgPolicyCache()
        router, _classify, _counted = _router(policy_cache)
        request = _request(metadata={})

        first = await router._build_query_context(request)
        await router._build_query_context(request)

        assert first["preferred_models"] == ["text-model"]
        assert policy_cache.calls == [(request.request_id, "u1")]

    @pytest.mark.asyncio
    async def test_fallback_policy_reads_stay_fresh(self):
        policy_cache = CountingPolicyCache()
        router, _classify, _counted = _router(policy_cache)
        request = _request()

        await router._build_query_context(request)
        await router._get_policy_context(request)

        assert policy_cache.calls == 2

    @pytest.mark.asyncio
    async def test_long_prompts_are_tokenized_off_the_event_loop(self, monkeypatch):
        router, _classify, _counted = _router(CountingPolicyCache())
        threaded = []

        async def to_thread(func, *args, **kwargs):
            threaded.append(func)
            return func(*args, **kwargs)

        monkeypatch.setattr("src.llm_router_part1_router.asyncio.to_thread", to_thread)

        await router._build_query_context(_request(context="word " * 5000))
        await router._build_query_context(_request(query="short"))

        assert len(threaded) == 1


class CountingRedis:
    """A tiny async Redis that counts round trips (one per await)."""

    def __init__(self):
        self.store = {}
        self.sets = {}
        self.round_trips = 0

    def pipeline(self, transaction=True):
        return _Pipeline(self)

    def _get(self, key):
        return self.store.get(key)

    def _smembers(self, key):
        return set(self.sets.get(key, set()))

    async def get(self, key):
        self.round_trips += 1
        return self._get(key)

    async def mget(self, keys):
        self.round_trips += 1
        return [self._get(key) for key in keys]

    async def smembers(self, key):
        self.round_trips += 1
        return self._smembers(key)

    async def srem(self, key, *members):
        self.round_trips += 1
        for member in members:
            self.sets.get(key, set()).discard(member)

    async def setex(self, key, _ttl, value):
        self.round_trips += 1
        self.store[key] = value


class _Pipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def get(self, key):
        self.commands.append(("get", key))
        return self

    def smembers(self, key):
        self.commands.append(("smembers", key))
        return self

    def srem(self, key, *members):
        self.commands.append(("srem", key, members))
        return self

    def setex(self, key, _ttl, value):
        self.commands.append(("setex", key, value))
        return self

    def sadd(self, key, member):
        self.commands.append(("sadd", key, member))
        return self

    async def execute(self):
        self.redis.round_trips += 1
        results = []
        for command in self.commands:
            if command[0] == "get":
                results.append(self.redis._get(command[1]))
            elif command[0] == "smembers":
                results.append(self.redis._smembers(command[1]))
            elif command[0] == "srem":
                for member in command[2]:
                    self.redis.sets.get(command[1], set()).discard(member)
                results.append(len(command[2]))
            elif command[0] == "setex":
                self.redis.store[command[1]] = command[2]
                results.append(True)
            elif command[0] == "sadd":
                self.redis.sets.setdefault(command[1], set()).add(command[2])
                results.append(1)
        self.commands = []
        return results


def _redis_policy_cache():
    cache = RoutingPolicyCache({"enabled": True, "local_cache_ttl_seconds": 30})
    cache.redis_client = CountingRedis()
    cache._initialized = True
    return cache


class TestPolicyReads:
    @pytest.mark.asyncio
    async def test_effective_policy_is_one_round_trip(self):
        cache = _redis_policy_cache()
        redis = cache.redis_client
        redis.store[cache._user_state_cache_key("u1")] = json.dumps(
            {"preferred_models": ["gpt-5"], "requires_low_latency": True}
        )
        redis.store[cache._cache_key("session", "s1")] = json.dumps(
            {"avoid_models": ["mistral-7b"]}
        )

        policy = await cache.get_effective_policy("r1", "u1", "s1")

        assert redis.round_trips <= 1
        assert policy["preferred_models"] == ["gpt-5"]
        assert policy["avoid_models"] == ["mistral-7b"]
        assert policy["policy_source"] == "session+user"

    @pytest.mark.asyncio
    async def test_active_guardrails_cost_one_more_round_trip_then_hit_l1(self):
        cache = _redis_policy_cache()
        redis = cache.redis_client
        redis.sets[cache._guardrail_index_key("model")] = {"gpt-5", "stale-model"}
        redis.store[cache._guardrail_cache_key("model", "gpt-5")] = json.dumps(
            {"guardrail_action": "avoid", "description": "error spike"}
        )

        policy = await cache.get_effective_policy("r1", "u1")

        assert policy["blocked_models"] == ["gpt-5"]
        # The pipeline, one MGET for the payloads and one prune of the
        # expired member.
        assert redis.round_trips <= 3
        assert redis.sets[cache._guardrail_index_key("model")] == {"gpt-5"}

        redis.round_trips = 0
        again = await cache.get_effective_policy("r2", "u1")

        assert again["blocked_models"] == ["gpt-5"]
        assert redis.round_trips <= 1

    @pytest.mark.asyncio
    async def test_empty_results_are_cached_in_l1(self):
        cache = _redis_policy_cache()
        redis = cache.redis_client

        assert await cache.get_effective_policy("r1", "u1", "s1") == {
            "blocked_models": [],
            "warn_models": [],
            "blocked_providers": [],
            "warn_providers": [],
            "guardrail_reasons": {},
        }
        redis.round_trips = 0
        await cache.get_effective_policy("r2", "u1", "s1")

        # Only the new request's own key is read; user, session and the
        # guardrail indexes come from L1.
        assert redis.round_trips <= 1
        redis.round_trips = 0
        assert await cache.get_user_state_policy("u1") is None
        assert redis.round_trips == 0

    @pytest.mark.asyncio
    async def test_user_hint_key_is_never_read(self):
        cache = _redis_policy_cache()
        redis = cache.redis_client
        read_keys = []
        original_get = redis._get

        def recording_get(key):
            read_keys.append(key)
            return original_get(key)

        redis._get = recording_get

        await cache.get_effective_policy("r1", "u1", "s1")

        assert cache._user_hint_cache_key("u1") not in read_keys
        assert cache._cache_key("request", "r1") in read_keys

    @pytest.mark.asyncio
    async def test_materialized_guardrail_is_visible_immediately(self):
        cache = _redis_policy_cache()
        await cache.get_effective_policy("r1", "u1")

        await cache.materialize_routing_guardrail(
            {"scope_type": "model", "scope_key": "gpt-5", "guardrail_action": "avoid"}
        )
        policy = await cache.get_effective_policy("r2", "u1")

        assert policy["blocked_models"] == ["gpt-5"]
