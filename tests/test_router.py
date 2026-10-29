import builtins
import time

import pytest

from src.llm_router_part1_router import (
    ModelRouter,
    PromptTooLargeError,
    QueryClassifier,
    RoutingRule,
    TokenCounter,
)
from src.utils.schema import (
    Attachment,
    AttachmentType,
    QueryRequest,
    QueryType,
    ToolPolicy,
    UserTier,
)

QUERY_MAX_CHARS = 50000

# A long single-paragraph paste, as a user would send a transcript or report.
MIXED_REALISTIC_QUERY = (
    "Why is that the p99 latency spikes every night after the deploy? "
    "How can we conclude it is the cache and not the database? "
    "Write a Python function that parses the gateway logs and build a script "
    "to compare the error trends per region. "
    "What does this code do: def handler(event): return event['body']. "
    "The latest release announced today includes a price update for the stock "
    "dashboard. Summarize the key points, brainstorm some ideas, and help me "
    "plan the rollout roadmap with milestones. "
) * 120

# Each keyword run lacks the pattern's final keyword (dense_keywords has no "do"
# within 80 chars of a "code"), so every start backtracks through its gaps.
ADVERSARIAL_QUERIES = {
    "why_is": "why is " * 7200,
    "how_can_we": "how can we " * 4600,
    "write": "write " * 8400,
    "what_does_this": "what does this " * 3400,
    "dense_keywords": ("what " * 16 + "does " * 16 + "this " * 16 + "code " * 16) * 160,
}

CLASSIFIER_STRESS_QUERIES = {
    **ADVERSARIAL_QUERIES,
    "mixed_realistic": MIXED_REALISTIC_QUERY,
}

# (query, type, confidence) recorded from classify_query before the regex rewrite.
CLASSIFIER_SNAPSHOT = [
    (
        "Write a Python function to sort a list",
        QueryType.CODE_GENERATION,
        0.5642857142857143,
    ),
    (
        "Can you implement a binary search algorithm in Rust?",
        QueryType.CODE_GENERATION,
        0.20714285714285713,
    ),
    (
        "Please review this code and fix the bug in my parser function",
        QueryType.CODE_ANALYSIS,
        0.43333333333333335,
    ),
    (
        "What does this code do? def add(a, b): return a + b",
        QueryType.CODE_GENERATION,
        0.20714285714285713,
    ),
    (
        "Analyze the sales data and compare the trends across regions",
        QueryType.ANALYSIS,
        0.5166666666666666,
    ),
    (
        "Summarize the key points of this meeting transcript",
        QueryType.SUMMARIZATION,
        0.46,
    ),
    (
        "Write a short story about a robot who learns to paint",
        QueryType.CREATIVE_WRITING,
        0.43333333333333335,
    ),
    (
        "Compose a poem about autumn leaves",
        QueryType.CREATIVE_WRITING,
        0.43333333333333335,
    ),
    ("What are some good ideas for a team offsite?", QueryType.BRAINSTORMING, 0.38),
    ("Help me think of names for my new coffee shop", QueryType.BRAINSTORMING, 0.23),
    (
        "How to plan a product launch roadmap with clear milestones",
        QueryType.PLANNING,
        0.7333333333333333,
    ),
    ("Solve the equation 3x + 5 = 20", QueryType.MATH, 0.43333333333333335),
    ("Calculate the derivative of x^2 * sin(x)", QueryType.MATH, 0.36666666666666664),
    (
        "Why is that the case, and how can we conclude the argument is valid?",
        QueryType.REASONING,
        0.5166666666666666,
    ),
    (
        "What is the latest news about the NVIDIA stock price today?",
        QueryType.WEB_RESEARCH,
        0.5466666666666666,
    ),
    ("最新的 AI 新聞是什麼？", QueryType.WEB_RESEARCH, 0.28444444444444444),
    ("Hi there, how are you doing?", QueryType.GENERAL, 0.5),
    ("Thanks, that was really helpful", QueryType.GENERAL, 0.5),
    ("Translate this sentence into French", QueryType.GENERAL, 0.5),
    # Keyword gaps that cross a dot inside a sentence.
    (
        "write a python 3.12 script to parse logs",
        QueryType.CODE_GENERATION,
        0.35714285714285715,
    ),
    (
        "fix the bug in utils.py function",
        QueryType.CODE_ANALYSIS,
        0.36666666666666664,
    ),
    ("Write a node.js function", QueryType.CODE_GENERATION, 0.2642857142857143),
]


class TestQueryClassifier:
    @pytest.mark.asyncio
    async def test_initialize_skips_sentence_transformer_when_semantic_disabled(
        self, monkeypatch
    ):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise AssertionError("sentence-transformers must not be imported")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        classifier = QueryClassifier(semantic_enabled=False)

        await classifier.initialize()

        assert classifier._is_initialized is True
        assert classifier.model is None

    def test_classify_code_generation_query(self):
        classifier = QueryClassifier()
        classifier._is_initialized = True

        query_type, confidence = classifier.classify_query(
            "Write a Python function to sort a list"
        )

        assert query_type == QueryType.CODE_GENERATION
        assert confidence > 0.1

    def test_classify_web_research_query(self):
        classifier = QueryClassifier()
        classifier._is_initialized = True

        query_type, confidence = classifier.classify_query(
            "What is the latest AI news today?"
        )

        assert query_type == QueryType.WEB_RESEARCH
        assert confidence > 0.1

    @pytest.mark.parametrize(
        "query",
        list(CLASSIFIER_STRESS_QUERIES.values()),
        ids=list(CLASSIFIER_STRESS_QUERIES),
    )
    def test_classify_query_is_fast_on_max_length_input(self, query):
        classifier = QueryClassifier()
        classifier._is_initialized = True
        query = query[:QUERY_MAX_CHARS]

        started = time.perf_counter()
        classifier.classify_query(query)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.25

    @pytest.mark.parametrize(
        "query", list(ADVERSARIAL_QUERIES.values()), ids=list(ADVERSARIAL_QUERIES)
    )
    def test_classifier_patterns_are_fast_on_untruncated_input(self, query):
        # Runs the regexes on the full input, bypassing the classifier's head and
        # tail cut, so the patterns themselves must not backtrack catastrophically.
        classifier = QueryClassifier()
        query = query[:QUERY_MAX_CHARS]
        patterns = [
            pattern
            for type_patterns in classifier._compiled_patterns.values()
            for pattern in type_patterns
        ]
        assert len(query) == QUERY_MAX_CHARS

        started = time.perf_counter()
        for pattern in patterns:
            pattern.findall(query)
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0

    @pytest.mark.parametrize(
        ("query", "expected_type", "expected_confidence"), CLASSIFIER_SNAPSHOT
    )
    def test_classify_query_matches_recorded_snapshot(
        self, query, expected_type, expected_confidence
    ):
        classifier = QueryClassifier()
        classifier._is_initialized = True

        query_type, confidence = classifier.classify_query(query)

        assert query_type == expected_type
        assert confidence == pytest.approx(expected_confidence)

    def test_classify_query_reads_request_after_long_paste(self):
        classifier = QueryClassifier()
        classifier._is_initialized = True
        # 10,000 chars of pasted text, then the actual request.
        long_paste = ("lorem ipsum dolor sit amet " * 371)[:9999] + "\n"
        query = (
            long_paste
            + "summarize the key points and search the web for the latest news"
        )

        query_type, confidence = classifier.classify_query(query)

        # Same result as origin/main, which classified the whole query.
        assert query_type == QueryType.WEB_RESEARCH
        assert confidence == pytest.approx(0.5688888888888889)


class TestTokenCounter:
    def test_disabled_tokenizer_uses_local_approximation_without_import(
        self, monkeypatch
    ):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise AssertionError("tiktoken must not be imported")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        counter = TokenCounter(enabled=False)

        assert counter.count_tokens("one two three") == 3

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
    async def test_nested_classifier_config_disables_semantic_model(
        self, router_config, monkeypatch
    ):
        real_import = builtins.__import__
        import_attempted = False

        def guarded_import(name, *args, **kwargs):
            nonlocal import_attempted
            if name == "sentence_transformers":
                import_attempted = True
                raise AssertionError("sentence-transformers must not be imported")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        router = ModelRouter(
            {
                **router_config,
                "classifier": {"semantic_enabled": False},
                "tokenizer_enabled": False,
            }
        )

        await router.initialize()

        assert router.classifier.model is None
        assert import_attempted is False

    @pytest.mark.asyncio
    async def test_missing_required_default_model_credential_fails_startup(
        self, router_config, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {**router_config, "credential_gating": True}

        router = ModelRouter(config)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await router.initialize()

    @pytest.mark.asyncio
    async def test_optional_provider_without_credential_is_skipped(
        self, router_config, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = {
            **router_config,
            "credential_gating": True,
            "classifier": {"semantic_enabled": False},
            "models": {
                "gpt-5": router_config["models"]["gpt-5"],
                "claude-sonnet-4-6": {
                    "provider": "anthropic",
                    "max_tokens": 8192,
                    "cost_per_token": 0.000015,
                    "priority": 2,
                    "capabilities": ["general", "reasoning", "analysis"],
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            },
        }

        router = ModelRouter(config)
        await router.initialize()

        assert set(router.models) == {"gpt-5"}

    @pytest.mark.asyncio
    async def test_optional_provider_with_credential_is_loaded(
        self, router_config, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
        config = {
            **router_config,
            "credential_gating": True,
            "classifier": {"semantic_enabled": False},
            "models": {
                "gpt-5": router_config["models"]["gpt-5"],
                "claude-sonnet-4-6": {
                    "provider": "anthropic",
                    "max_tokens": 8192,
                    "cost_per_token": 0.000015,
                    "priority": 2,
                    "capabilities": ["general", "reasoning", "analysis"],
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            },
        }

        router = ModelRouter(config)
        await router.initialize()

        assert set(router.models) == {"gpt-5", "claude-sonnet-4-6"}

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

    @pytest.mark.asyncio
    async def test_build_query_context_counts_context_tokens(self, router_config):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (QueryType.ANALYSIS, 0.95)
        counted_prompts = []

        def count_tokens(prompt, _model="default"):
            counted_prompts.append(prompt)
            return len(prompt.split())

        router.token_counter.count_tokens = count_tokens
        request = QueryRequest(
            query="Analyze",
            user_id="u1",
            context="one two three four five",
        )

        context = await router._build_query_context(request)

        assert context["token_count"] > len(request.query.split())
        assert "Context: one two three four five" in counted_prompts[0]

    @pytest.mark.asyncio
    async def test_capability_filter_counts_rag_context_near_model_limit(
        self, router_config
    ):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.9)
        router.models["mistral-7b"].capabilities.append("vision")
        router.models["gpt-5"].capabilities.append("vision")
        router.token_counter.count_tokens = (
            lambda prompt, _model="default": 5000 if "RAG context" in prompt else 1000
        )
        request = QueryRequest(
            query="Explain the chart",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
            metadata={"rag_context": "RAG context"},
        )

        candidates = await router.eligible_models_with_capability(request, "vision")

        assert "mistral-7b" not in candidates
        assert "gpt-5" in candidates

    @pytest.mark.asyncio
    async def test_build_query_context_marks_required_web_search(self, router_config):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.5)
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        request = QueryRequest(
            query="Check the web",
            user_id="u1",
            tool_policy=ToolPolicy.REQUIRED,
            allowed_tools=["web_search"],
        )

        context = await router._build_query_context(request)

        assert context["query_type"] == "web_research"
        assert context["needs_web_search"] is True

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

    def test_load_models_accepts_already_named_config(self, router_config):
        named_config = {
            **router_config,
            "models": {
                model_name: {"name": model_name, **model_config}
                for model_name, model_config in router_config["models"].items()
            },
        }

        router = ModelRouter(named_config)

        assert router.models["gpt-5"].name == "gpt-5"

    def test_fallback_when_no_model_matches(self, router_config):
        router = ModelRouter(router_config)
        with pytest.raises(PromptTooLargeError):
            router._capability_based_routing(
                {
                    "query_type": "translation",
                    "token_count": 999999,
                    "user_tier": "free",
                }
            )

    @pytest.mark.asyncio
    async def test_route_query_ignores_policy_fast_lane_without_explicit_sla(
        self, router_config
    ):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id):
                return {
                    "policy_source": "user",
                    "route_to_fast_lane": True,
                    "preferred_models": [],
                    "hint_reason": "stale_fast_lane_hint",
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

        assert decision.route_to_fast_lane is False
        assert decision.actual_fast_lane_hit is False
        assert decision.policy_source == "user"

    @pytest.mark.asyncio
    async def test_route_query_uses_explicit_metadata_low_latency_fast_lane(
        self, router_config
    ):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (
            QueryType.SUMMARIZATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 500}
        router.model_stats["qwen3.6-27b-fast"] = {
            "success_rate": 0.94,
            "avg_latency": 90,
        }
        router.model_stats["mistral-7b"] = {"success_rate": 0.92, "avg_latency": 100}

        request = QueryRequest(
            query="Summarize the latest thread",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
            metadata={"requires_low_latency": True},
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "qwen3.6-27b-fast"
        assert decision.route_to_fast_lane is True
        assert decision.actual_fast_lane_hit is True
        assert decision.policy_source == "request_features"
        assert "Request-feature selection" in decision.routing_reason

    @pytest.mark.asyncio
    async def test_route_query_uses_api_priority_for_fast_lane(self, router_config):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (
            QueryType.SUMMARIZATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 500}
        router.model_stats["qwen3.6-27b-fast"] = {
            "success_rate": 0.94,
            "avg_latency": 90,
        }
        router.model_stats["mistral-7b"] = {"success_rate": 0.92, "avg_latency": 100}

        request = QueryRequest(
            query="Summarize the latest thread",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
            priority=4,
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "qwen3.6-27b-fast"
        assert decision.route_to_fast_lane is True
        assert decision.actual_fast_lane_hit is True
        assert decision.policy_source == "request_features"

    @pytest.mark.asyncio
    async def test_route_query_does_not_fast_lane_keyword_urgency(self, router_config):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (
            QueryType.SUMMARIZATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128

        request = QueryRequest(
            query="Urgent production outage: summarize the alerts",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
        )

        decision = await router.route_query(request)

        assert decision.route_to_fast_lane is False
        assert decision.actual_fast_lane_hit is False

    @pytest.mark.asyncio
    async def test_route_query_can_use_qwen_for_free_simple_request_without_fast_lane(
        self, router_config
    ):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (
            QueryType.SUMMARIZATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["qwen3.6-27b-fast"] = {
            "success_rate": 0.94,
            "avg_latency": 90,
        }
        router.model_stats["mistral-7b"] = {"success_rate": 0.92, "avg_latency": 140}

        request = QueryRequest(
            query="Summarize this short note",
            user_id="free-user",
            user_tier=UserTier.FREE,
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "qwen3.6-27b-fast"
        assert decision.route_to_fast_lane is False
        assert decision.actual_fast_lane_hit is False

    @pytest.mark.asyncio
    async def test_route_query_does_not_fast_lane_complex_low_latency_request(
        self, router_config
    ):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (QueryType.ANALYSIS, 0.95)
        router.token_counter.count_tokens = lambda _query, _model="default": 4096

        request = QueryRequest(
            query="Analyze the root cause and tradeoffs for this incident",
            user_id="u1",
            user_tier=UserTier.ENTERPRISE,
            metadata={"requires_low_latency": True},
        )

        decision = await router.route_query(request)

        assert decision.route_to_fast_lane is False
        assert decision.actual_fast_lane_hit is False
        assert decision.policy_source == "request_features"

    @pytest.mark.asyncio
    async def test_route_query_does_not_fast_lane_long_context_request(
        self, router_config
    ):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (
            QueryType.SUMMARIZATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128

        request = QueryRequest(
            query="Summarize this context",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
            max_tokens=4096,
            metadata={"requires_low_latency": True},
        )

        decision = await router.route_query(request)

        assert decision.route_to_fast_lane is False
        assert decision.actual_fast_lane_hit is False

    @pytest.mark.asyncio
    async def test_route_query_does_not_fast_lane_attachment_heavy_request(
        self, router_config
    ):
        router = ModelRouter(router_config)
        router.classifier.classify_query = lambda _query: (
            QueryType.SUMMARIZATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": 128

        attachments = [
            Attachment(
                name=f"context-{index}.txt",
                type=AttachmentType.FILE,
                size_bytes=1024,
                mime_type="text/plain",
            )
            for index in range(3)
        ]
        request = QueryRequest(
            query="Summarize these files",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
            attachments=attachments,
            metadata={"requires_low_latency": True},
        )

        decision = await router.route_query(request)

        assert decision.route_to_fast_lane is False
        assert decision.actual_fast_lane_hit is False

    @pytest.mark.asyncio
    async def test_route_query_uses_session_policy_model_pinning(self, router_config):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id, session_id=None):
                assert session_id == "session-42"
                return {
                    "policy_source": "session+user",
                    "preferred_models": ["gpt-5"],
                    "query_type": "analysis",
                    "query_complexity": "complex",
                    "requires_high_reasoning": True,
                    "hint_reason": "session_model_pinning",
                }

        router = ModelRouter(router_config, policy_cache=PolicyCacheStub())
        router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.6)
        router.token_counter.count_tokens = lambda _query, _model="default": 256
        router.model_stats["gpt-5"] = {"success_rate": 0.97, "avg_latency": 650}
        router.model_stats["mistral-7b"] = {"success_rate": 0.95, "avg_latency": 90}

        request = QueryRequest(
            query="Continue the same architecture analysis as before",
            user_id="u1",
            user_tier=UserTier.ENTERPRISE,
            session_id="session-42",
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "gpt-5"
        assert decision.policy_source == "session+user"
        assert "session_model_pinning" in decision.routing_reason

    @pytest.mark.asyncio
    async def test_route_query_normalizes_enum_user_tier_for_rule_matching(
        self, router_config
    ):
        premium_router_config = dict(router_config)
        premium_router_config["routing_rules"] = [
            {
                "condition": "user_tier == 'premium'",
                "models": ["gpt-5"],
                "fallback": "mistral-7b",
            }
        ]

        router = ModelRouter(premium_router_config)
        router.classifier.classify_query = lambda _query: (QueryType.ANALYSIS, 0.95)
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 500}

        request = QueryRequest(
            query="Analyze this architecture",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "gpt-5"
        assert "Rule-based selection" in decision.routing_reason

    @pytest.mark.asyncio
    async def test_route_query_respects_request_metadata_preferred_models(self):
        router = ModelRouter(
            {
                "default_model": "mistral-7b",
                "routing_strategy": "intelligent",
                "models": {
                    "claude-3.5-sonnet": {
                        "provider": "anthropic",
                        "max_tokens": 200000,
                        "cost_per_token": 0.000015,
                        "priority": 2,
                        "capabilities": ["reasoning", "analysis"],
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                    "mistral-7b": {
                        "provider": "vllm",
                        "model_path": "/models/mistral",
                        "max_tokens": 4096,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "analysis"],
                        "gpu_memory_gb": 16,
                    },
                },
                "routing_rules": [],
            }
        )
        router.classifier.classify_query = lambda _query: (QueryType.ANALYSIS, 0.95)
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["claude-3.5-sonnet"] = {
            "success_rate": 0.99,
            "avg_latency": 700,
        }
        router.model_stats["mistral-7b"] = {"success_rate": 0.90, "avg_latency": 100}

        request = QueryRequest(
            query="Analyze the attached incidents",
            user_id="premium-user",
            user_tier=UserTier.PREMIUM,
            metadata={"preferred_models": ["claude-3.5-sonnet"]},
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "claude-3.5-sonnet"
        assert "Policy-cache selection" in decision.routing_reason

    @pytest.mark.asyncio
    async def test_route_query_policy_low_latency_does_not_create_fast_lane(self):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id):
                return {
                    "policy_source": "user",
                    "cost_sensitivity": "high",
                    "requires_low_latency": True,
                    "session_hotness": "hot",
                    "burst_protection_active": True,
                    "hint_reason": "burst_protection",
                }

        router = ModelRouter(
            {
                "default_model": "mistral-7b",
                "routing_strategy": "intelligent",
                "models": {
                    "gpt-5": {
                        "provider": "openai",
                        "max_tokens": 8192,
                        "cost_per_token": 0.00003,
                        "priority": 1,
                        "capabilities": ["general", "analysis", "reasoning"],
                    },
                    "mistral-7b": {
                        "provider": "vllm",
                        "model_path": "/models/mistral",
                        "max_tokens": 4096,
                        "cost_per_token": 0.0,
                        "priority": 3,
                        "capabilities": ["general", "analysis", "coding"],
                    },
                },
                "routing_rules": [],
            },
            policy_cache=PolicyCacheStub(),
        )
        router.classifier.classify_query = lambda _query: (QueryType.ANALYSIS, 0.9)
        router.token_counter.count_tokens = lambda _query, _model="default": 128
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 600}
        router.model_stats["mistral-7b"] = {"success_rate": 0.93, "avg_latency": 110}

        request = QueryRequest(
            query="Summarize the alerts from the last few minutes",
            user_id="u2",
            user_tier=UserTier.FREE,
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "mistral-7b"
        assert decision.policy_source == "user"
        assert decision.route_to_fast_lane is False
        assert decision.actual_fast_lane_hit is False

    @pytest.mark.asyncio
    async def test_route_query_policy_shape_does_not_override_current_request(
        self, router_config
    ):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id):
                return {
                    "policy_source": "request",
                    "query_type": "analysis",
                    "query_complexity": "complex",
                    "requires_high_reasoning": True,
                    "route_to_fast_lane": False,
                    "preferred_models": [],
                }

        router = ModelRouter(router_config, policy_cache=PolicyCacheStub())
        router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.75)
        router.token_counter.count_tokens = lambda _query, _model="default": 256
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 600}
        router.model_stats["mistral-7b"] = {"success_rate": 0.92, "avg_latency": 120}

        request = QueryRequest(
            query="Need a root cause analysis for this production incident",
            user_id="u1",
            user_tier=UserTier.ENTERPRISE,
        )

        decision = await router.route_query(request)

        assert decision.selected_model in {"gpt-5", "mistral-7b"}
        assert decision.query_type == QueryType.GENERAL

    @pytest.mark.asyncio
    async def test_route_query_filters_blocked_provider_guardrails(self, router_config):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id):
                return {
                    "policy_source": "request",
                    "preferred_models": ["gpt-5"],
                    "blocked_providers": ["openai"],
                    "guardrail_reasons": {
                        "provider:openai": "Provider latency exceeded threshold"
                    },
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
            query="Write a Python helper",
            user_id="u1",
            user_tier=UserTier.PREMIUM,
        )

        decision = await router.route_query(request)

        assert decision.selected_model == "mistral-7b"


class TestRuleBasedRoutingFilters:
    """Rule candidates go through the same guardrail, tier and window filter."""

    def _build_router(self, config, policy, token_count=128):
        class PolicyCacheStub:
            async def get_effective_policy(self, request_id, user_id):
                return dict(policy)

        router = ModelRouter(config, policy_cache=PolicyCacheStub())
        router.classifier.classify_query = lambda _query: (
            QueryType.CODE_GENERATION,
            0.95,
        )
        router.token_counter.count_tokens = lambda _query, _model="default": token_count
        # gpt-5 scores best, then qwen3.6-27b-fast, then mistral-7b.
        router.model_stats["gpt-5"] = {"success_rate": 0.99, "avg_latency": 100}
        router.model_stats["qwen3.6-27b-fast"] = {
            "success_rate": 0.95,
            "avg_latency": 1200,
        }
        router.model_stats["mistral-7b"] = {"success_rate": 0.80, "avg_latency": 1500}
        return router

    async def _route(self, router, user_tier=UserTier.PREMIUM):
        request = QueryRequest(
            query="Write a Python helper", user_id="u1", user_tier=user_tier
        )
        decision = await router.route_query(request)
        context = await router._build_query_context(request)
        eligible = router._filter_models_for_context(list(router.models), context)
        return decision, eligible

    def _build_enterprise_router(self, router_config, policy, query_type):
        config = dict(router_config)
        config["models"] = {
            **router_config["models"],
            # As shipped in config/config.yaml: no "general" or "coding".
            "claude-sonnet-4-6": {
                "provider": "anthropic",
                "max_tokens": 1000000,
                "cost_per_token": 0.000015,
                "priority": 1,
                "capabilities": [
                    "reasoning",
                    "writing",
                    "analysis",
                    "tool_use",
                    "web_search",
                    "vision",
                ],
                "api_key_env": "ANTHROPIC_API_KEY",
            },
        }
        config["routing_rules"] = [
            {
                "condition": "user_tier == 'enterprise'",
                "models": ["claude-sonnet-4-6", "gpt-5"],
                "fallback": "claude-sonnet-4-6",
            },
            *router_config["routing_rules"],
        ]
        router = self._build_router(config, policy)
        router.classifier.classify_query = lambda _query: (query_type, 0.95)
        router.model_stats["claude-sonnet-4-6"] = {
            "success_rate": 0.99,
            "avg_latency": 100,
        }
        return router

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_type", [QueryType.GENERAL, QueryType.CODE_GENERATION]
    )
    async def test_enterprise_rule_selects_sonnet_without_capability_check(
        self, router_config, query_type
    ):
        router = self._build_enterprise_router(router_config, {}, query_type)

        decision, _eligible = await self._route(router, user_tier=UserTier.ENTERPRISE)

        assert decision.selected_model == "claude-sonnet-4-6"
        assert "user_tier == 'enterprise'" in decision.routing_reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_type", [QueryType.GENERAL, QueryType.CODE_GENERATION]
    )
    async def test_enterprise_rule_skips_blocked_sonnet(
        self, router_config, query_type
    ):
        router = self._build_enterprise_router(
            router_config, {"blocked_models": ["claude-sonnet-4-6"]}, query_type
        )

        decision, _eligible = await self._route(router, user_tier=UserTier.ENTERPRISE)

        assert decision.selected_model != "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_rule_routing_skips_blocked_model(self, router_config):
        router = self._build_router(router_config, {"blocked_models": ["gpt-5"]})

        decision, eligible = await self._route(router)

        assert decision.selected_model != "gpt-5"
        assert decision.selected_model in eligible

    @pytest.mark.asyncio
    async def test_rule_routing_skips_blocked_provider(self, router_config):
        router = self._build_router(router_config, {"blocked_providers": ["openai"]})

        decision, eligible = await self._route(router)

        assert router.models[decision.selected_model].provider != "openai"
        assert decision.selected_model in eligible

    @pytest.mark.asyncio
    async def test_rule_routing_respects_free_tier_access(self, router_config):
        router = self._build_router(router_config, {})

        decision, eligible = await self._route(router, user_tier=UserTier.FREE)

        assert router.models[decision.selected_model].priority >= 3
        assert decision.selected_model in eligible

    @pytest.mark.asyncio
    async def test_rule_routing_respects_context_window(self, router_config):
        router = self._build_router(router_config, {}, token_count=6000)
        # mistral-7b (4096-token window) now outscores gpt-5 among rule models.
        router.model_stats["mistral-7b"] = {"success_rate": 0.99, "avg_latency": 50}

        decision, eligible = await self._route(router, user_tier=UserTier.FREE)

        assert router.models[decision.selected_model].max_tokens >= 6000
        assert decision.selected_model in eligible

    @pytest.mark.asyncio
    async def test_rule_routing_uses_fallback_when_rule_models_filtered(
        self, router_config
    ):
        config = dict(router_config)
        config["routing_rules"] = [
            {
                "condition": "query_type == 'code_generation'",
                "models": ["gpt-5"],
                "fallback": "mistral-7b",
            }
        ]
        router = self._build_router(config, {"blocked_models": ["gpt-5"]})

        decision, eligible = await self._route(router)

        # Capability routing would pick qwen3.6-27b-fast; the rule fallback wins.
        assert decision.selected_model == "mistral-7b"
        assert "query_type == 'code_generation'" in decision.routing_reason
        assert "fallback" in decision.routing_reason.lower()
        assert decision.selected_model in eligible

    @pytest.mark.asyncio
    async def test_rule_routing_moves_to_next_rule_when_fallback_filtered(
        self, router_config
    ):
        config = dict(router_config)
        config["routing_rules"] = [
            {
                "condition": "query_type == 'code_generation'",
                "models": ["gpt-5"],
                "fallback": "gpt-5",
            },
            {
                "condition": "query_type == 'code_generation' and token_count < 1000",
                "models": ["mistral-7b"],
                "fallback": "mistral-7b",
            },
        ]
        router = self._build_router(config, {"blocked_models": ["gpt-5"]})

        decision, eligible = await self._route(router)

        assert decision.selected_model == "mistral-7b"
        assert "token_count < 1000" in decision.routing_reason
        assert decision.selected_model in eligible

    @pytest.mark.asyncio
    async def test_rule_routing_falls_through_to_capability_routing(
        self, router_config
    ):
        config = dict(router_config)
        config["routing_rules"] = [
            {
                "condition": "query_type == 'code_generation'",
                "models": ["gpt-5"],
                "fallback": "gpt-5",
            }
        ]
        router = self._build_router(config, {"blocked_providers": ["openai"]})

        decision, eligible = await self._route(router)

        assert decision.selected_model != "gpt-5"
        assert "Capability-based selection" in decision.routing_reason
        assert decision.selected_model in eligible
