"""One query classifier for the router and Flink (R13)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import query_classifier
from src.llm_router_part1_router import QueryClassifier
from src.routing_features import build_routing_features, infer_query_type
from src.utils.schema import QueryType

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ROWS = [
    row
    for row in (
        json.loads(line)
        for line in (REPO_ROOT / "eval" / "routing" / "labeled.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if row["split"] == "dev"
]


def _router_classifier():
    classifier = QueryClassifier()
    classifier._is_initialized = True
    return classifier


def test_shared_classifier_imports_only_the_standard_library():
    # Flink mounts src/ read-only: the fallback must not pull in pydantic,
    # redis or the router.
    code = (
        "import sys; import src.query_classifier, src.routing_features; "
        "heavy = [m for m in ('pydantic', 'redis', 'src.llm_router_part1_router', "
        "'src.utils.schema') if m in sys.modules]; print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "[]"


def test_classify_returns_a_query_type_value_and_a_confidence():
    query_type, confidence = query_classifier.classify(
        "Write a Python function to sort a list"
    )

    assert query_type == "code_generation"
    assert 0.0 < confidence <= 1.0
    assert {rule_type for rule_type in query_classifier.RULES} <= {
        member.value for member in QueryType
    }


@pytest.mark.parametrize(
    "query",
    [
        "explain how recursion works now",
        "What is the current status of the migration ticket?",
        "Can you update me on what you can do?",
        "我現在想問一個問題",
    ],
)
def test_now_and_current_alone_are_not_web_research(query):
    assert query_classifier.classify(query)[0] == "general"


@pytest.mark.parametrize(
    "query",
    [
        "Translate this sentence into French",
        "translate 'good morning' to Japanese",
        "把這段話翻譯成英文",
    ],
)
def test_translation_is_detected(query):
    assert query_classifier.classify(query)[0] == "translation"


def test_general_fallback_is_less_confident_than_any_detection():
    fallback = query_classifier.classify("Hi there, how are you doing?")
    detections = [
        query_classifier.classify(row["query"])
        for row in DEV_ROWS
        if query_classifier.classify(row["query"])[0] != "general"
    ]

    assert fallback == ("general", query_classifier.GENERAL_CONFIDENCE)
    assert detections
    assert all(
        confidence > query_classifier.GENERAL_CONFIDENCE
        for _type, confidence in detections
    )


def test_router_classifier_wraps_the_shared_classifier():
    classifier = _router_classifier()

    for row in DEV_ROWS:
        query_type, confidence = classifier.classify_query(row["query"])
        assert isinstance(query_type, QueryType)
        assert (query_type.value, confidence) == query_classifier.classify(row["query"])


def test_long_input_is_classified_on_its_head_and_tail_by_both_sides():
    query = "lorem ipsum " * 2000 + "\ntranslate this into German"

    assert query_classifier.classify(query)[0] == "translation"
    assert infer_query_type(query) == "translation"


def test_flink_fallback_uses_the_shared_classifier():
    query = "Summarize the key points of this meeting transcript"

    assert infer_query_type(query) == query_classifier.classify(query)[0]
    assert infer_query_type(query) == "summarization"
    assert infer_query_type("") == "general"


def test_flink_keeps_the_payload_query_type():
    features = build_routing_features(
        {"query_text": "Summarize this thread", "query_type": "math"}
    )

    assert features["query_type"] == "math"


def test_the_old_first_match_rules_are_gone():
    import flink.logic
    import src.routing_features

    assert not hasattr(src.routing_features, "QUERY_TYPE_RULES")
    assert not hasattr(flink.logic, "QUERY_TYPE_RULES")
