"""Routing classifier quality on the labeled set (X9b, R13).

eval/routing/labeled.jsonl is labeled from eval/routing/labels.json. The
classifier rules were written and tuned against the dev split. The test split
("source": "blind") was written from labels.json by a separate author who never
saw the classifier, and no rule was changed after measuring it: it is an
independent held-out measurement.
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

# Measured on the blind test split (120 rows, 10 per label), minus 0.05,
# floored to 2 decimals. They are regression guards, not quality targets:
# recall on unfamiliar wording is low and misses fall back to general.
# Measured: accuracy 85/120 (0.708), macro F1 0.738; per label below + 0.05.
# fc64233 baseline on the same blind split: router 35/120 (macro F1 0.32),
# Flink 28/120 (macro F1 0.26).
TEST_F1_THRESHOLDS = {
    "analysis": 0.77,  # 0.824
    "brainstorming": 0.89,  # 0.947
    "code_analysis": 0.25,  # 0.308
    "code_generation": 0.61,  # 0.667
    "creative_writing": 0.83,  # 0.889
    "general": 0.32,  # 0.375 (precision 0.24: misses land here)
    "math": 0.58,  # 0.632
    "planning": 0.95,  # 1.0
    "reasoning": 0.65,  # 0.706
    "summarization": 0.83,  # 0.889
    "translation": 0.95,  # 1.0
    "web_research": 0.57,  # 0.625
}
TEST_MACRO_F1_THRESHOLD = 0.68  # 0.738
TEST_ACCURACY_THRESHOLD = 0.65  # 0.708


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


def _test_split_pairs():
    return [
        (row["label"], classify(row["query"])[0])
        for row in ROWS
        if row["split"] == "test"
    ]


def _test_split_f1():
    return _f1_by_label(_test_split_pairs())


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
    # The test split is the independent blind set, and only it.
    assert all(
        (row["split"] == "test") == (row.get("source") == "blind") for row in ROWS
    )


@pytest.mark.parametrize("label", sorted(TEST_F1_THRESHOLDS))
def test_test_split_f1_meets_threshold(label):
    assert _test_split_f1()[label] >= TEST_F1_THRESHOLDS[label]


def test_test_split_macro_f1_meets_threshold():
    scores = _test_split_f1()

    assert sum(scores.values()) / len(scores) >= TEST_MACRO_F1_THRESHOLD


def test_test_split_accuracy_meets_threshold():
    pairs = _test_split_pairs()

    accuracy = sum(1 for gold, pred in pairs if gold == pred) / len(pairs)

    assert accuracy >= TEST_ACCURACY_THRESHOLD


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
