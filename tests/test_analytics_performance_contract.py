import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analytics_performance_contract.py"


def test_flink_python_harness_can_import_workspace_package():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())

    for service in ("flink-jobmanager", "flink-taskmanager"):
        assert compose["services"][service]["environment"]["PYTHONPATH"] == "/workspace"


def test_contract_evaluation_requires_every_clickhouse_and_flink_gate():
    from scripts.analytics_performance_contract import evaluate_contracts

    metrics = {
        "clickhouse": {
            "rows": 1_000_000,
            "read_rows_reduction_ratio": 0.70,
            "dashboard_cold_p95_improvement_ratio": 0.40,
            "dashboard_query_count": 4,
            "model_performance_seed_rows": 1,
            "replay_final_count": 1,
            "migration_verified": True,
            "errors": [],
        },
        "flink": {
            "events": 10_000,
            "state_bytes_reduction_ratio": 0.80,
            "throughput_improvement_ratio": 5.0,
            "emits_per_scope": 2,
            "errors": [],
        },
    }

    passed, gates = evaluate_contracts(metrics)

    assert passed is True
    assert all(gate["passed"] for gate in gates)

    metrics["clickhouse"]["read_rows_reduction_ratio"] = 0.699
    passed, gates = evaluate_contracts(metrics)
    assert passed is False
    assert (
        next(
            gate for gate in gates if gate["name"] == "clickhouse.read_rows_reduction"
        )["passed"]
        is False
    )


def test_flink_benchmark_is_deterministic_in_shape_and_meets_hot_key_contract():
    from scripts.analytics_performance_contract import benchmark_flink_hot_key

    metrics = benchmark_flink_hot_key(events=10_000, repeats=2)

    assert metrics["events"] == 10_000
    assert metrics["legacy_state_bytes"] > metrics["incremental_state_bytes"]
    assert metrics["state_bytes_reduction_ratio"] >= 0.80
    assert metrics["throughput_improvement_ratio"] >= 5.0
    assert metrics["emits_per_scope"] <= 2
    assert metrics["measurement_mode"] == "rolling_scope_policy_emitter_harness"
    assert metrics["actual_operator_events"] == 10_000
    assert metrics["legacy_measured_events"] == 1_000
    assert metrics["legacy_extrapolation_factor"] > 1
    assert metrics["errors"] == []


def test_legacy_flink_benchmark_models_list_state_and_policy_work(monkeypatch):
    import flink.logic
    import scripts.analytics_performance_contract as contract

    original_loads = contract.json.loads
    original_dumps = contract.json.dumps
    json_work = {"loads": 0, "dumps": 0}
    policy_event_counts = []

    def measured_loads(*args, **kwargs):
        json_work["loads"] += 1
        return original_loads(*args, **kwargs)

    def measured_dumps(*args, **kwargs):
        json_work["dumps"] += 1
        return original_dumps(*args, **kwargs)

    def measured_policy(*, events, **kwargs):
        policy_event_counts.append(len(events))
        return {"request_count": len(events)}

    monkeypatch.setattr(contract.json, "loads", measured_loads)
    monkeypatch.setattr(contract.json, "dumps", measured_dumps)
    monkeypatch.setattr(
        flink.logic,
        "build_routing_policy_state_event",
        measured_policy,
    )

    checksum, state_bytes = contract._legacy_hot_key(3)

    assert checksum == 600
    assert state_bytes > 0
    assert policy_event_counts == [1, 2, 3]
    assert json_work["loads"] >= 3  # 0 + 1 + 2 prior ListState entries
    assert json_work["dumps"] >= 6  # 1 + 2 + 3 full ListState rewrites


def test_clickhouse_dashboard_count_is_derived_from_issued_queries():
    from scripts.analytics_performance_contract import _MeasuredDashboardQueries

    issued = []
    measurement = _MeasuredDashboardQueries(lambda sql: issued.append(sql))
    measurement.query("SELECT 1")
    measurement.query("SELECT 2")

    assert measurement.count == 2
    assert issued == ["SELECT 1", "SELECT 2"]


def test_dashboard_query_gate_executes_the_production_pipeline_bundle():
    from scripts.analytics_performance_contract import (
        _run_production_dashboard_bundle,
    )

    issued = []

    class _Result:
        result_rows = []

    class _Client:
        def query(self, sql, **kwargs):
            issued.append(sql)
            return _Result()

    count, bundle = _run_production_dashboard_bundle(
        _Client(), "analytics_contract_test"
    )

    assert count == 4
    assert bundle["errors"] == {}
    assert any("query_logs FINAL" in sql for sql in issued)
    assert any("model_performance FINAL" in sql for sql in issued)
    assert any("alert_events FINAL" in sql for sql in issued)
    assert any("routing_policy_state_events FINAL" in sql for sql in issued)


def test_legacy_dashboard_baseline_includes_all_independent_sections():
    from scripts.analytics_performance_contract import legacy_dashboard_queries

    queries = legacy_dashboard_queries("analytics_contract_test", hours=1)

    assert list(queries) == [
        "analytics_overall",
        "analytics_model_breakdown",
        "analytics_query_type_breakdown",
        "model_performance",
        "routing_guardrails",
        "routing_policy_state",
    ]
    assert sum("query_logs_legacy" in sql for sql in queries.values()) == 3
    assert "countIf(status = 'success')" in queries["analytics_overall"]
    assert "sum(cost_usd)" in queries["analytics_model_breakdown"]
    assert "FINAL" not in "\n".join(queries.values())
    assert "model_performance" in queries["model_performance"]
    assert "alert_events" in queries["routing_guardrails"]
    assert "routing_policy_state_events" in queries["routing_policy_state"]


def test_benchmark_legacy_query_logs_ddl_matches_origin_main_partitioning():
    from scripts.analytics_performance_contract import legacy_query_logs_ddl

    ddl = legacy_query_logs_ddl("analytics_contract_test", "event_id String")

    assert ddl == (
        "CREATE TABLE analytics_contract_test.query_logs_legacy (event_id String) "
        "ENGINE=ReplacingMergeTree PARTITION BY toYYYYMM(timestamp) "
        "ORDER BY event_id"
    )


def test_dashboard_p95_times_production_bundle_and_full_legacy_flow(monkeypatch):
    import scripts.analytics_performance_contract as contract

    production_durations = iter(index / 10 for index in range(1, 21))
    legacy_durations = iter(float(index) for index in range(1, 21))
    calls = []

    def time_production(client, database):
        calls.append("production")
        return 4, {"errors": {}}, next(production_durations)

    def time_legacy(client, database):
        calls.append("legacy")
        return 6, next(legacy_durations)

    monkeypatch.setattr(contract, "_time_production_dashboard_bundle", time_production)
    monkeypatch.setattr(contract, "_time_legacy_dashboard_bundle", time_legacy)

    metrics = contract.benchmark_dashboard_bundle_latency(
        object(), "analytics_contract_test", repeats=1
    )

    expected_calls = []
    for index in range(20):
        expected_calls.extend(
            ["legacy", "production"] if index % 2 == 0 else ["production", "legacy"]
        )
    assert calls == expected_calls
    assert metrics["dashboard_latency_samples"] == 20
    assert metrics["legacy_dashboard_cold_p95_seconds"] == 19.0
    assert metrics["v2_dashboard_cold_p95_seconds"] == 1.9
    assert metrics["dashboard_cold_p95_improvement_ratio"] == pytest.approx(0.90)
    assert metrics["dashboard_query_count"] == 4


def test_dashboard_p95_rejects_production_bundle_errors(monkeypatch):
    import scripts.analytics_performance_contract as contract

    monkeypatch.setattr(
        contract,
        "_time_legacy_dashboard_bundle",
        lambda client, database: (6, 1.0),
    )
    monkeypatch.setattr(
        contract,
        "_time_production_dashboard_bundle",
        lambda client, database: (4, {"errors": {"analytics": "failed"}}, 0.1),
    )

    with pytest.raises(RuntimeError, match="production dashboard bundle errors"):
        contract.benchmark_dashboard_bundle_latency(
            object(), "analytics_contract_test", repeats=5
        )


def test_clickhouse_benchmark_seeds_nonempty_model_performance_section():
    from scripts.analytics_performance_contract import seed_model_performance

    issued = []

    class _Result:
        first_row = (1,)

    class _Client:
        def command(self, sql):
            issued.append(sql)

        def query(self, sql):
            issued.append(sql)
            return _Result()

    count = seed_model_performance(_Client(), "analytics_contract_test")

    assert count == 1
    assert "INSERT INTO analytics_contract_test.model_performance" in issued[0]
    assert "requests_count" in issued[0]
    assert "avg_tokens_per_second" in issued[0]
    assert "count()" in issued[1]


def test_clickhouse_benchmark_disables_generated_sessions_for_concurrent_bundle(
    monkeypatch,
):
    from scripts.analytics_performance_contract import benchmark_clickhouse

    captured = {}

    def get_client(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after client construction")

    monkeypatch.setitem(
        sys.modules,
        "clickhouse_connect",
        SimpleNamespace(get_client=get_client),
    )

    benchmark_clickhouse(rows=1, repeats=1)

    assert captured["autogenerate_session_id"] is False


def test_clickhouse_event_id_expression_uses_a_24_8_supported_hash():
    from scripts.analytics_performance_contract import clickhouse_event_id_expression

    expression = clickhouse_event_id_expression("number")

    assert expression == "hex(sipHash128(number))"
    assert "cityHash128" not in expression


def test_clickhouse_scalar_reads_the_first_row_instead_of_mapping():
    from scripts.analytics_performance_contract import clickhouse_scalar

    class _Result:
        first_row = (1,)
        first_item = {"count()": 1}

    assert clickhouse_scalar(_Result()) == 1


def test_clickhouse_measurement_tag_is_recorded_in_query_log_comment():
    from scripts.analytics_performance_contract import clickhouse_query_options

    assert clickhouse_query_options("contract-123") == {
        "settings": {"log_queries": 1, "log_comment": "contract-123"},
    }


def test_read_rows_gate_compares_same_query_and_exposes_bundle_amplification():
    from scripts.analytics_performance_contract import clickhouse_read_rows_metrics

    metrics = clickhouse_read_rows_metrics(
        legacy_same_query_rows=1_000_000,
        v2_same_query_rows=20_000,
        legacy_bundle_rows=3_000_000,
    )

    assert metrics == {
        "legacy_same_query_read_rows": 1_000_000,
        "legacy_bundle_read_rows": 3_000_000,
        "v2_read_rows": 20_000,
        "read_rows_reduction_ratio": 0.98,
    }


def test_flink_only_cli_outputs_machine_readable_json_and_fails_bad_contract():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--flink-only",
            "--flink-events",
            "10",
            "--minimum-flink-events",
            "10000",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["passed"] is False
    assert payload["metrics"]["flink"]["events"] == 10
    assert any(
        gate["name"] == "flink.event_count" and not gate["passed"]
        for gate in payload["gates"]
    )


def test_clickhouse_contract_rejects_error_even_when_numeric_gates_pass():
    from scripts.analytics_performance_contract import evaluate_contracts

    metrics = {
        "clickhouse": {
            "rows": 1_000_000,
            "read_rows_reduction_ratio": 0.99,
            "dashboard_cold_p95_improvement_ratio": 0.99,
            "dashboard_query_count": 1,
            "replay_final_count": 1,
            "errors": ["query log missing"],
        }
    }

    passed, gates = evaluate_contracts(metrics, include_flink=False)

    assert passed is False
    assert (
        next(gate for gate in gates if gate["name"] == "clickhouse.success")["passed"]
        is False
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_contract_rejects_missing_or_non_finite_measurements(value):
    from scripts.analytics_performance_contract import evaluate_contracts

    metrics = {
        "flink": {
            "events": 10_000,
            "state_bytes_reduction_ratio": value,
            "throughput_improvement_ratio": 5.0,
            "emits_per_scope": 2,
            "errors": [],
        }
    }

    passed, _ = evaluate_contracts(metrics, include_clickhouse=False)
    assert passed is False
