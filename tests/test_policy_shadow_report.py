"""A1: the shadow-mode report reads routing_decisions with parameterized
queries and prints counts only."""

import io
import sys
import types
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from scripts import policy_shadow_report as report


class FakeClient:
    def __init__(self, rows_by_dimension):
        self.rows_by_dimension = rows_by_dimension
        self.calls = []
        self.closed = False

    def query(self, sql, parameters=None):
        self.calls.append((sql, dict(parameters or {})))
        for marker, rows in self.rows_by_dimension.items():
            if f"{marker} AS bucket" in sql:
                return SimpleNamespace(result_rows=rows)
        raise AssertionError(f"unexpected query: {sql}")

    def close(self):
        self.closed = True


# bucket, decisions, agree, differ, skipped, not_replayable
ROWS = {
    "'all'": [("all", 12, 6, 2, 2, 2)],
    "selected_model": [("gpt-5", 9, 5, 1, 1, 2), ("qwen", 3, 1, 1, 1, 0)],
    "user_tier": [("free", 12, 6, 2, 2, 2)],
}


def test_queries_are_parameterized():
    client = FakeClient(ROWS)

    report.build_report(client, hours=6, database="llm_router")

    assert len(client.calls) == 3
    for sql, parameters in client.calls:
        assert "{hours:UInt32}" in sql
        assert "{database:Identifier}.routing_decisions" in sql
        assert parameters == {"hours": 6, "database": "llm_router"}
        assert "6 HOUR" not in sql
        for private in ("query_text", "routing_reason", "candidates_json"):
            assert private not in sql


def test_report_counts_and_rates():
    result = report.build_report(FakeClient(ROWS), hours=6, database="llm_router")

    (overall,) = result["overall"]
    assert overall == {
        "bucket": "all",
        "decisions": 12,
        "agree": 6,
        "differ": 2,
        "skipped": 2,
        "not_replayable": 2,
        "agree_rate": 0.75,
        "differ_rate": 0.25,
        # Of the 10 decisions whose strategy shadow mode can replay.
        "skipped_rate": 0.2,
    }
    assert [row["bucket"] for row in result["by_model"]] == ["gpt-5", "qwen"]
    assert result["by_tier"][0]["bucket"] == "free"


def test_empty_window_has_zero_rates():
    empty = {key: [] for key in ROWS}
    empty["'all'"] = [("all", 0, 0, 0, 0, 0)]

    result = report.build_report(FakeClient(empty), hours=1, database="db")

    assert result["overall"][0]["agree_rate"] == 0.0
    assert result["overall"][0]["skipped_rate"] == 0.0


def test_output_has_counts_and_no_text_beyond_models_and_tiers():
    result = report.build_report(FakeClient(ROWS), hours=6, database="llm_router")

    text = report.format_report(result, hours=6)

    assert "last 6 hours" in text
    assert "gpt-5" in text and "free" in text
    assert "75.0%" in text
    # skipped_empty and unavailable are one column in routing_decisions.
    assert "skipped_empty" in text and "unavailable" in text
    assert "outside shadow mode" in text
    assert "round_robin" in text and "not_replayable" in text


def test_round_robin_and_weighted_are_counted_apart_from_skipped():
    client = FakeClient(ROWS)

    report.build_report(client, hours=6, database="llm_router")

    sql = " ".join(client.calls[0][0].split())
    assert "routing_strategy IN ('round_robin', 'weighted')" in sql
    assert (
        "countIf(shadow_selected_model = '' AND routing_strategy NOT IN "
        "('round_robin', 'weighted')) AS skipped"
    ) in sql


def test_main_takes_the_password_from_the_environment(monkeypatch):
    captured = {}
    client = FakeClient(ROWS)

    def get_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setitem(
        sys.modules,
        "clickhouse_connect",
        types.SimpleNamespace(get_client=get_client),
    )
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "from-env")

    out = io.StringIO()
    with redirect_stdout(out):
        assert report.main(["--hours", "6"]) == 0

    assert captured["password"] == "from-env"
    assert client.closed
    assert "gpt-5" in out.getvalue()
    with pytest.raises(SystemExit):
        report.main(["--password", "x"])


def test_database_name_must_be_an_identifier():
    with pytest.raises(SystemExit):
        report.main(["--database", "x; DROP TABLE y"])
