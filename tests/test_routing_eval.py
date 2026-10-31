"""Routing classifier quality on the labeled set (X9b, R13).

eval/routing/labeled.jsonl is labeled from eval/routing/labels.json. The
classifier is tuned on the dev split only; the thresholds below are the F1 it
reached on the held-out test split, minus 0.05, rounded down.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.llm_router_part1_router import QueryClassifier
from src.query_classifier import classify
from src.redaction import redact_text
from src.routing_features import infer_query_type
from src.utils.schema import QueryType

ROUTING_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval" / "routing"
ROWS = [
    json.loads(line)
    for line in (ROUTING_EVAL_DIR / "labeled.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
LABEL_DEFINITIONS = json.loads(
    (ROUTING_EVAL_DIR / "labels.json").read_text(encoding="utf-8")
)["labels"]
LABELS = sorted({row["label"] for row in ROWS})

# Test-split F1 after the R13 changes (1.0 for every label), minus 0.05,
# rounded down to 2 decimals. With 5 test rows per label, one miss fails a
# label. The split was written together with dev, so 1.0 overstates quality on
# new wording; the thresholds guard against regressions, not generalization.
TEST_F1_THRESHOLDS = {
    "analysis": 0.95,
    "brainstorming": 0.95,
    "code_analysis": 0.95,
    "code_generation": 0.95,
    "creative_writing": 0.95,
    "general": 0.95,
    "math": 0.95,
    "planning": 0.95,
    "reasoning": 0.95,
    "summarization": 0.95,
    "translation": 0.95,
    "web_research": 0.95,
}
TEST_MACRO_F1_THRESHOLD = 0.95


def _f1_by_label(pairs):
    scores = {}
    for label in LABELS:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total = precision + recall
        scores[label] = 2 * precision * recall / total if total else 0.0
    return scores


def _test_split_f1():
    pairs = [
        (row["label"], classify(row["query"])[0])
        for row in ROWS
        if row["split"] == "test"
    ]
    return _f1_by_label(pairs)


def test_labeled_set_covers_every_label_in_both_languages_and_splits():
    labels = Counter(row["label"] for row in ROWS)
    test_labels = Counter(row["label"] for row in ROWS if row["split"] == "test")
    chinese = sum(1 for row in ROWS if row["lang"] == "zh")

    assert len(ROWS) >= 200
    assert chinese / len(ROWS) >= 0.4
    assert {row["split"] for row in ROWS} == {"dev", "test"}
    assert {row["lang"] for row in ROWS} == {"en", "zh"}
    assert set(LABELS) <= {query_type.value for query_type in QueryType}
    assert set(LABELS) == set(LABEL_DEFINITIONS)
    assert set(TEST_F1_THRESHOLDS) == set(LABELS)
    assert min(labels.values()) >= 8
    assert set(test_labels) == set(LABELS)
    assert min(test_labels.values()) >= 3
    assert len({row["query"] for row in ROWS}) == len(ROWS)


@pytest.mark.parametrize("label", sorted(TEST_F1_THRESHOLDS))
def test_test_split_f1_meets_threshold(label):
    assert _test_split_f1()[label] >= TEST_F1_THRESHOLDS[label]


def test_test_split_macro_f1_meets_threshold():
    scores = _test_split_f1()

    assert sum(scores.values()) / len(scores) >= TEST_MACRO_F1_THRESHOLD


def test_every_label_can_be_produced_by_the_classifier():
    predicted = {classify(row["query"])[0] for row in ROWS}

    assert set(LABELS) <= predicted


def test_classifier_gives_the_same_type_for_raw_and_redacted_text():
    mismatches = [
        row["query"]
        for row in ROWS
        if classify(row["query"]) != classify(redact_text(row["query"]))
    ]

    assert mismatches == []


def test_router_and_flink_fallback_agree_on_the_whole_set():
    classifier = QueryClassifier()
    classifier._is_initialized = True

    mismatches = [
        (row["query"], router_type.value, flink_type)
        for row in ROWS
        for router_type, flink_type in [
            (
                classifier.classify_query(row["query"])[0],
                infer_query_type(row["query"]),
            )
        ]
        if router_type.value != flink_type
    ]

    assert mismatches == []
