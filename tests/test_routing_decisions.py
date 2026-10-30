"""R12: routing decisions explain themselves and reach routing.decisions."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part3_pipeline import (
    KafkaConsumerManager,
    KafkaProducerManager,
    build_routing_decision_event,
)
from src.utils.schema import (
    InferenceResponse,
    QueryRequest,
    QueryType,
    RoutingDecision,
    UserTier,
)

SECRET_QUERY = "Please summarise the secret merger memo for ACME"


@pytest.fixture
def pipeline_config():
    return {
        "bootstrap_servers": ["localhost:9092"],
        "topics": {
            "queries": "test-queries",
            "alerts": "alerts",
            "routing_decisions": "routing.decisions",
        },
        "consumer": {"group_id": "test-group", "auto_offset_reset": "latest"},
        "clickhouse": {
            "host": "localhost",
            "port": 8123,
            "database": "test_db",
            "username": "default",
            "password": "",
        },
    }


class _PolicyStub:
    def __init__(self, policy):
        self.policy = policy

    async def get_effective_policy(self, request_id, user_id):
        return dict(self.policy)


def _router(config, policy=None, query_type=QueryType.GENERAL, tokens=128):
    router = ModelRouter(
        config, policy_cache=_PolicyStub(policy) if policy is not None else None
    )
    router.classifier.classify_query = lambda _query: (query_type, 0.9)
    router.token_counter.count_tokens = lambda _text, _model="default": tokens
    return router


def _request(tier=UserTier.FREE, **kwargs):
    return QueryRequest(query=SECRET_QUERY, user_id="u1", user_tier=tier, **kwargs)


def _candidates(decision):
    return {candidate.model: candidate for candidate in decision.candidates}


@pytest.mark.asyncio
async def test_free_user_decision_rejects_gpt5_for_tier(router_config):
    router = _router(router_config)

    decision = await router.route_query(_request(UserTier.FREE))

    candidates = _candidates(decision)
    assert set(candidates) == set(router.models)
    assert candidates["gpt-5"].eligible is False
    assert candidates["gpt-5"].reject_reason == "tier"
    assert candidates[decision.selected_model].eligible is True
    assert candidates[decision.selected_model].selected is True
    assert decision.reason_code == "capability"
    # The free text keeps its old wording.
    assert decision.routing_reason == "Capability-based selection"


@pytest.mark.asyncio
async def test_context_window_capability_and_guardrail_rejections(router_config):
    config = dict(router_config)
    config["models"] = {
        **router_config["models"],
        "vision-only": {
            "provider": "vllm",
            "max_tokens": 32768,
            "cost_per_token": 0.0,
            "priority": 3,
            "capabilities": ["vision"],
        },
        "big-general": {
            "provider": "vllm",
            "max_tokens": 128000,
            "cost_per_token": 0.0,
            "priority": 3,
            "capabilities": ["general"],
        },
    }
    router = _router(
        config,
        policy={"blocked_providers": ["openai"], "blocked_models": ["qwen-x"]},
        tokens=5000,
    )

    decision = await router.route_query(
        _request(UserTier.PREMIUM, metadata={"avoid_models": ["qwen3.6-27b-fast"]})
    )

    candidates = _candidates(decision)
    assert candidates["gpt-5"].reject_reason == "blocked_provider"
    assert candidates["qwen3.6-27b-fast"].reject_reason == "avoid_model"
    assert candidates["mistral-7b"].reject_reason == "context_window"
    assert candidates["vision-only"].reject_reason == "capability"
    assert decision.selected_model == "big-general"
    assert candidates["big-general"].reject_reason is None


@pytest.mark.asyncio
async def test_blocked_model_reason(router_config):
    router = _router(router_config, policy={"blocked_models": ["mistral-7b"]})

    decision = await router.route_query(_request(UserTier.PREMIUM))

    assert _candidates(decision)["mistral-7b"].reject_reason == "blocked_model"


@pytest.mark.asyncio
async def test_selected_model_carries_score_breakdown(router_config):
    router = _router(router_config)

    decision = await router.route_query(_request(UserTier.PREMIUM))

    selected = _candidates(decision)[decision.selected_model]
    assert selected.score is not None
    assert selected.score_breakdown
    assert {"reliability", "latency", "cost", "priority"} <= set(
        selected.score_breakdown
    )
    assert sum(selected.score_breakdown.values()) == pytest.approx(selected.score)
    eligible_scores = [
        candidate.score for candidate in decision.candidates if candidate.eligible
    ]
    assert selected.score == max(eligible_scores)


@pytest.mark.asyncio
async def test_rule_selection_reason_code(router_config):
    router = _router(router_config, query_type=QueryType.CODE_GENERATION)

    decision = await router.route_query(_request(UserTier.PREMIUM))

    assert decision.reason_code == "rule"
    assert "Rule-based selection" in decision.routing_reason


@pytest.mark.asyncio
async def test_candidate_table_failure_never_breaks_routing(router_config):
    router = _router(router_config)

    def broken(*_args, **_kwargs):
        raise RuntimeError("table exploded")

    router._score_components = broken
    router._candidate_table = broken

    decision = await router.route_query(_request(UserTier.FREE))

    assert decision.selected_model in router.models
    assert decision.candidates == []


@pytest.mark.asyncio
async def test_routing_error_decision_has_reason_code(router_config):
    router = _router(router_config)

    def broken_classify(_query):
        raise RuntimeError("classifier exploded")

    router.classifier.classify_query = broken_classify

    decision = await router.route_query(_request(UserTier.FREE))

    assert decision.reason_code == "routing_error"
    assert "routing error" in decision.routing_reason.lower()


def _decision(**updates):
    payload = dict(
        selected_model="gpt-5",
        query_type=QueryType.GENERAL,
        routing_reason="Capability-based selection",
        token_count=10,
        estimated_cost=0.001,
        routing_time_ms=2,
        confidence=0.8,
        reason_code="capability",
        candidates=[
            {
                "model": "gpt-5",
                "provider": "openai",
                "eligible": True,
                "score": 90.0,
                "score_breakdown": {"reliability": 38.0, "cost": 52.0},
                "selected": True,
            },
            {
                "model": "mistral-7b",
                "provider": "vllm",
                "eligible": False,
                "reject_reason": "context_window",
            },
        ],
    )
    payload.update(updates)
    return RoutingDecision(**payload)


def test_routing_decision_event_has_explicit_fields_and_no_query():
    request = _request(
        UserTier.PREMIUM, context="Private context about the merger", request_id="r-1"
    )
    response = InferenceResponse(
        response_text="answer",
        model_name="claude-sonnet-4-6",
        token_count_input=1,
        token_count_output=1,
        total_tokens=2,
        latency_ms=1,
        tokens_per_second=1.0,
        cost_usd=0.0,
        provider="anthropic",
    )

    event = build_routing_decision_event(request, _decision(), response)

    serialized = json.dumps(event, default=str)
    assert SECRET_QUERY not in serialized
    assert "merger" not in serialized
    assert "query" not in event and "context" not in event
    assert event["event_type"] == "routing.decision"
    assert event["request_id"] == "r-1"
    assert event["selected_model"] == "gpt-5"
    assert event["served_model"] == "claude-sonnet-4-6"
    assert event["reason_code"] == "capability"
    assert event["candidates"][1] == {
        "model": "mistral-7b",
        "provider": "vllm",
        "eligible": False,
        "reject_reason": "context_window",
        "score": None,
        "score_breakdown": {},
        "selected": False,
    }


def test_routing_decision_event_tolerates_minimal_decisions():
    request = _request(request_id="r-2")

    event = build_routing_decision_event(
        request, SimpleNamespace(selected_model="mistral-7b")
    )

    assert event["selected_model"] == "mistral-7b"
    assert event["candidates"] == []
    assert event["reason_code"] is None
    assert event["served_model"] is None


@pytest.mark.asyncio
async def test_produce_routing_decision_keys_by_request_id(pipeline_config):
    manager = KafkaProducerManager(pipeline_config)
    manager._enqueue_message = MagicMock(return_value=True)

    await manager.produce_routing_decision(_request(request_id="r-3"), _decision())

    args, kwargs = manager._enqueue_message.call_args
    assert args[:2] == ("routing_decisions", "routing.decisions")
    assert kwargs["key"] == "r-3"
    assert kwargs["value"]["request_id"] == "r-3"


@pytest.mark.asyncio
async def test_inference_publishes_routing_decision(sample_query_request):
    from src.llm_router_part2_inference import InferenceEngine

    producer = SimpleNamespace(
        produce_inference_completed=AsyncMock(),
        produce_query_log=AsyncMock(),
        produce_routing_decision=AsyncMock(),
    )
    engine = InferenceEngine({}, router=MagicMock(), event_producer=producer)
    decision = _decision()
    response = MagicMock()

    await engine._publish_analytics_events(
        sample_query_request, response, routing_decision=decision
    )

    producer.produce_routing_decision.assert_awaited_once_with(
        sample_query_request, decision, response
    )


@pytest.mark.asyncio
async def test_inference_skips_producers_without_routing_decisions(
    sample_query_request,
):
    from src.llm_router_part2_inference import InferenceEngine

    producer = SimpleNamespace(produce_routing_decision=AsyncMock())
    engine = InferenceEngine({}, router=MagicMock(), event_producer=producer)

    await engine._publish_analytics_events(
        sample_query_request, MagicMock(), routing_decision=None
    )

    producer.produce_routing_decision.assert_not_awaited()


class _Messages:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


@pytest.mark.asyncio
async def test_routing_decisions_are_consumed_into_clickhouse(pipeline_config):
    config = {
        **pipeline_config,
        "topics": {
            **pipeline_config.get("topics", {}),
            "routing_decisions": "routing.decisions",
        },
    }
    consumer = KafkaConsumerManager(config, MagicMock())
    consumer.clickhouse.batch_insert_routing_decisions = AsyncMock()
    event = build_routing_decision_event(_request(request_id="r-4"), _decision())
    message = SimpleNamespace(
        value=json.dumps(event).encode("utf-8"),
        key=b"r-4",
        topic="routing.decisions",
        partition=0,
        offset=7,
    )
    kafka_consumer = _Messages([message])
    consumer.consumers["routing_decisions"] = kafka_consumer

    await consumer._run_consumer("routing_decisions", kafka_consumer)
    await consumer.flush_pending_batches()

    rows = consumer.clickhouse.batch_insert_routing_decisions.await_args.args[0]
    row = rows[0].to_dict()
    assert row["event_id"] == "r-4"
    assert row["request_id"] == "r-4"
    assert row["selected_model"] == "gpt-5"
    assert row["reason_code"] == "capability"
    assert json.loads(row["candidates_json"])[1]["reject_reason"] == "context_window"
    assert SECRET_QUERY not in json.dumps(row, default=str)


@pytest.mark.asyncio
async def test_routing_decisions_table_is_created_last(pipeline_config):
    from src.llm_router_part3_pipeline import ClickHouseManager

    manager = ClickHouseManager(pipeline_config["clickhouse"])
    manager.client = MagicMock()
    manager._run_blocking = AsyncMock(return_value=None)

    await manager._create_tables()

    ddl = [call.args[1] for call in manager._run_blocking.await_args_list]
    creates = [sql for sql in ddl if "CREATE TABLE" in sql]
    assert "routing_decisions" in creates[-1]
    assert "query_text" not in creates[-1]


def test_router_config_topics_include_routing_decisions():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("config.yaml", "config.compose.yaml"):
        config = yaml.safe_load((root / "config" / name).read_text())
        assert config["kafka"]["topics"]["routing_decisions"] == "routing.decisions"
