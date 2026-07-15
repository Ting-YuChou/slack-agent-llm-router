import json
import subprocess
import sys
from pathlib import Path

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
            "cold_p95_improvement_ratio": 0.40,
            "dashboard_query_count": 4,
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
    assert metrics["errors"] == []


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
            "cold_p95_improvement_ratio": 0.99,
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
