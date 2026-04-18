import runpy
from pathlib import Path


MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "loadtest_api_baseline.py")
)


def test_build_payload_hot_mode_reuses_same_request():
    build_payload = MODULE["build_payload"]

    first = build_payload(0, "hot")
    second = build_payload(1, "hot")

    assert first["query"] == second["query"]
    assert first["user_id"] == second["user_id"]


def test_percentile_interpolates_between_points():
    percentile = MODULE["percentile"]

    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0.50) == 25.0
