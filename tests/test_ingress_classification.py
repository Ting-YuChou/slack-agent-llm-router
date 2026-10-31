"""requests.raw carries the router's query type and confidence (R13)."""

import pytest

import main
from flink.logic import build_request_enriched_event
from src import query_classifier
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part3_pipeline import build_request_raw_event
from src.utils.schema import QueryRequest, QueryType, ToolPolicy

from test_main import (  # noqa: F401
    DummyEventProducer,
    DummyInferenceEngine,
    _admission_platform,
    patched_platform_deps,
)


class RawEventProducer(DummyEventProducer):
    """Builds the requests.raw event the real producer would send."""

    def __init__(self, config):
        super().__init__(config)
        self.raw_events = []

    async def produce_request_raw(self, request, classification=None):
        await super().produce_request_raw(request)
        self.raw_events.append(
            build_request_raw_event(request, classification=classification)
        )


def _platform(tmp_path, router):
    platform = _admission_platform(tmp_path)
    producer = RawEventProducer({})
    platform.services["event_producer"] = producer
    platform.services["inference"] = DummyInferenceEngine({}, router)
    return platform, producer


def _router(router_config):
    return ModelRouter({**router_config, "tokenizer_enabled": False})


@pytest.mark.asyncio
async def test_requests_raw_carries_the_router_type_and_confidence(
    tmp_path, router_config, patched_platform_deps
):
    router = _router(router_config)
    platform, producer = _platform(tmp_path, router)
    query = "Translate this sentence into French"

    await main.ChatIngress(platform).process(QueryRequest(query=query, user_id="u1"))

    (event,) = producer.raw_events
    expected_type, expected_confidence = query_classifier.classify(query)
    assert event["query_type"] == expected_type == "translation"
    assert event["classifier_confidence"] == pytest.approx(expected_confidence)
    # Flink adopts the router's type instead of inferring its own.
    assert build_request_enriched_event(event)["query_type"] == "translation"


@pytest.mark.asyncio
async def test_ingress_classification_is_reused_by_routing(
    tmp_path, router_config, patched_platform_deps
):
    router = _router(router_config)
    calls = []

    def classify(query):
        calls.append(query)
        return QueryType.MATH, 0.8

    router.classifier.classify_query = classify
    platform, producer = _platform(tmp_path, router)
    request = QueryRequest(query="Solve 3x + 5 = 20", user_id="u1")

    await main.ChatIngress(platform).process(request)
    decision = await router.route_query(request)

    assert calls == ["Solve 3x + 5 = 20"]
    assert decision.query_type == QueryType.MATH
    assert producer.raw_events[0]["query_type"] == "math"
    assert producer.raw_events[0]["classifier_confidence"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_a_classifier_failure_publishes_requests_raw_without_a_type(
    tmp_path, router_config, patched_platform_deps
):
    router = _router(router_config)

    def broken(_query):
        raise RuntimeError("classifier exploded")

    router.classifier.classify_query = broken
    platform, producer = _platform(tmp_path, router)

    result, status_code, _headers = await main.ChatIngress(platform).process(
        QueryRequest(query="hello", user_id="u1")
    )

    assert status_code == 200
    assert result.response_text == "ok"
    (event,) = producer.raw_events
    assert event["query_type"] is None
    assert event["classifier_confidence"] is None
    assert build_request_enriched_event(event)["query_type"] == "general"


@pytest.mark.asyncio
async def test_ingress_without_a_classifying_router_still_publishes(
    tmp_path, patched_platform_deps
):
    platform, producer = _platform(tmp_path, router=object())

    await main.ChatIngress(platform).process(QueryRequest(query="hi", user_id="u1"))

    (event,) = producer.raw_events
    assert event["query_type"] is None


def test_requests_raw_built_without_ingress_has_no_type():
    event = build_request_raw_event(QueryRequest(query="Summarize this", user_id="u1"))

    assert event["query_type"] is None
    assert event["classifier_confidence"] is None


def test_classify_request_is_public_and_memoized(router_config):
    router = _router(router_config)
    calls = []

    def classify(query):
        calls.append(query)
        return QueryType.PLANNING, 0.7

    router.classifier.classify_query = classify
    request = QueryRequest(query="Plan the rollout", user_id="u1")

    assert router.classify_request(request) == (QueryType.PLANNING, 0.7)
    assert router.classify_request(request) == (QueryType.PLANNING, 0.7)
    assert calls == ["Plan the rollout"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"tool_policy": ToolPolicy.REQUIRED, "allowed_tools": ["web_search"]},
        {"metadata": {"enable_web_search": True}},
    ],
    ids=["required_tool", "enable_web_search"],
)
async def test_requests_raw_carries_the_routed_type_after_a_web_search_override(
    tmp_path, router_config, patched_platform_deps, overrides
):
    router = _router(router_config)
    router.classifier.classify_query = lambda _query: (QueryType.GENERAL, 0.2)
    platform, producer = _platform(tmp_path, router)
    request = QueryRequest(query="Check this", user_id="u1", **overrides)

    await main.ChatIngress(platform).process(request)
    decision = await router.route_query(request)

    assert decision.query_type == QueryType.WEB_RESEARCH
    assert producer.raw_events[0]["query_type"] == "web_research"
    assert producer.raw_events[0]["classifier_confidence"] == pytest.approx(0.2)


def test_classify_request_applies_the_web_search_override(router_config):
    router = _router(router_config)
    router.classifier.classify_query = lambda _query: (QueryType.MATH, 0.6)
    request = QueryRequest(
        query="Solve 2 + 2",
        user_id="u1",
        tool_policy=ToolPolicy.REQUIRED,
        allowed_tools=["web_search"],
    )

    assert router.classify_request(request) == (QueryType.WEB_RESEARCH, 0.6)


def test_build_request_raw_event_takes_the_classification_explicitly():
    request = QueryRequest(query="Summarize this", user_id="u1")

    event = build_request_raw_event(request, classification=("summarization", 0.8))

    assert event["query_type"] == "summarization"
    assert event["classifier_confidence"] == pytest.approx(0.8)
    assert not hasattr(request, "_query_classification")
