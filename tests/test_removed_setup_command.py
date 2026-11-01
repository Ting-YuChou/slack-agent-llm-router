"""A12b: `main.py setup` wrote a stale requirements file, CI workflow and DDL
and pip-installed into ./venv; it is gone."""

from pathlib import Path

import main


def test_cli_has_no_setup_command():
    assert "setup" not in main.cli.commands


def test_the_setup_module_is_gone():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "llm_router_part0_setup.py").exists()
