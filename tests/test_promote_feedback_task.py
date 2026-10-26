import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import unicodedata

import pytest

from scripts import agent_eval
from scripts import promote_feedback_task as promote
from scripts.export_agent_feedback import export

NAME = "cart-total-quantity"
BROKEN = "def total(items):\n    return sum(price for price, _ in items)\n"
FIXED = "def total(items):\n    return sum(price * qty for price, qty in items)\n"
ORACLE = (
    "ns = {}\n"
    "exec(open('src/total.py').read(), ns)\n"
    "assert ns['total']([(2, 3), (1, 4)]) == 10\n"
)
SPEC = {
    "description": "total() must weigh each price by its quantity.",
    "prompt": "total() in src/total.py ignores quantities: total([(2, 3)]) returns 2"
    " instead of 6. Fix it and keep tests/test_total.py unchanged.",
    "routing_text": "Fix the cart total.",
    "profile": {
        "rubric": ["total() sums price times quantity"],
        "allowedPaths": ["src/"],
        "baselinePaths": ["tests/"],
        "checks": [
            {
                "name": "unit",
                "argv": ["python3", "tests/test_total.py"],
                "cwd": ".",
                "timeoutMs": 30000,
            }
        ],
    },
    "oracle": {"argv": [sys.executable, ".eval-oracle/total_oracle.py"]},
}
ROW = {
    "run_id": "run-needs-changes",
    "user_id": "U0OWNER123",
    "channel_id": "C0CHANNEL99",
    "thread_ts": "1712345678.000100",
    "run_status": "completed",
    "is_ci_repair": False,
    "feedback_verdict": "needs_changes",
    "reason_category": "wrong_fix",
    "comment": "Quantities are still ignored",
    "grader_status": "passed",
    "verification_status": "passed",
    "prompts_version": "pv-1",
    "prompt": "Fix the cart total",
    "changed_files": ["src/total.py"],
    "created_at": "2026-10-01T12:00:00Z",
}


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """A project root with no eval suites yet; nothing may appear here on refusal."""
    root = tmp_path / "project"
    monkeypatch.setattr(agent_eval, "PROJECT", root)
    return root


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _inputs(tmp_path, spec=SPEC, oracle=ORACLE, rows=(ROW,)):
    """What the operator hands the script: drafts, a spec, fixture, oracle and fix."""
    root = tmp_path / "inputs"
    _write(root / "fixture/src/total.py", BROKEN)
    _write(root / "fixture/tests/test_total.py", "print('visible test')\n")
    _write(root / "oracle/total_oracle.py", oracle)
    _write(root / "solution/src/total.py", FIXED)
    _write(root / "spec.json", json.dumps(spec))
    export(list(rows), root / "drafts.jsonl")
    return root


def _argv(inputs, *extra, name=NAME, oracle=True):
    argv = [
        str(inputs / "drafts.jsonl"),
        f"--name={name}",
        "--spec",
        str(inputs / "spec.json"),
        "--fixture",
        str(inputs / "fixture"),
        "--solution",
        str(inputs / "solution"),
    ]
    if oracle:
        argv += ["--oracle", str(inputs / "oracle")]
    return argv + list(extra)


def _promote(argv):
    """Run the script; return its exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = promote.main(argv)
    return code, out.getvalue(), err.getvalue()


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _oracle_returncode(task, tree, fix=None):
    """Lay the task out as the eval does (fixture plus .eval-oracle/) and run its oracle."""
    shutil.copytree(task["fixture_dir"], tree)
    if fix is not None:
        (tree / "src/total.py").write_text(fix)
    shutil.copytree(task["oracle_dir"], tree / agent_eval.ORACLE_DIR)
    return subprocess.run(
        task["oracle"]["argv"], cwd=tree, capture_output=True
    ).returncode


# --- rule 1: a human-written oracle is required --------------------------------


@pytest.mark.parametrize("oracle", [None, "missing", "empty"])
def test_refuses_a_draft_without_a_human_written_oracle(tmp_path, project_root, oracle):
    inputs = _inputs(tmp_path)
    argv = _argv(inputs, oracle=False)
    if oracle == "missing":
        argv += ["--oracle", str(tmp_path / "nowhere")]
    if oracle == "empty":
        (tmp_path / "empty").mkdir()
        argv += ["--oracle", str(tmp_path / "empty")]

    code, _, err = _promote(argv)

    assert code != 0 and "oracle" in err
    if oracle is None:
        assert "human-written oracle" in err and "--oracle" in err
    assert not project_root.exists()


# --- rule 2: never overwrite ---------------------------------------------------


@pytest.mark.parametrize("suite", ["tasks", "repair"])
@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_refuses_a_task_name_either_suite_already_holds(
    tmp_path, project_root, suite, extra
):
    existing = project_root / agent_eval.SUITES[suite][0] / NAME
    _write(existing / "task.json", '{"id": "kept"}\n')
    _write(existing / "oracle/kept.test.cjs", "// hand-written\n")
    before = _snapshot(project_root)
    inputs = _inputs(tmp_path)

    code, _, err = _promote(_argv(inputs, *extra))

    assert code != 0 and "already exists" in err
    assert _snapshot(project_root) == before


# --- rule 3: loads in its suite; the oracle tells the fix from the start --------


@pytest.mark.parametrize(
    "suite, fields", [("tasks", {}), ("repair", {"requires_repair": True})]
)
def test_the_promoted_task_loads_in_its_suite_and_its_oracle_needs_the_fix(
    tmp_path, project_root, suite, fields
):
    inputs = _inputs(tmp_path, spec={**SPEC, **fields})

    assert _promote(_argv(inputs, "--suite", suite))[0] == 0

    # The eval's own suite resolution and loader.
    args = agent_eval.parse_args(["--live-openai", "--suite", suite])
    [task] = agent_eval.load_tasks(args.tasks_dir)
    other = "repair" if suite == "tasks" else "tasks"
    assert not (project_root / agent_eval.SUITES[other][0]).exists()
    assert task["id"] == NAME
    assert (task["task_mode"], task["workflow_mode"], task["expect"]) == (
        "implement",
        "single",
        "pass",
    )
    assert bool(task.get("requires_repair")) is (suite == "repair")
    assert task["prompt"] == SPEC["prompt"]
    assert task["routing_text"] == SPEC["routing_text"]
    assert task["profile"] == SPEC["profile"]
    assert task["oracle"] == SPEC["oracle"]
    assert task["oracle_files"] == ["total_oracle.py"]
    assert _snapshot(task["fixture_dir"]) == _snapshot(inputs / "fixture")
    assert _snapshot(task["oracle_dir"]) == _snapshot(inputs / "oracle")
    assert sorted(path.name for path in (args.tasks_dir / NAME).iterdir()) == [
        "fixture",
        "oracle",
        "task.json",
    ]
    assert _oracle_returncode(task, tmp_path / "start") != 0
    assert _oracle_returncode(task, tmp_path / "fixed", fix=FIXED) == 0


@pytest.mark.parametrize(
    "oracle, message",
    [
        ("pass\n", "unmodified fixture"),
        ("raise SystemExit(1)\n", "reference fix"),
        ("import time\ntime.sleep(5)\n", "timed out"),
    ],
)
def test_refuses_an_oracle_that_cannot_tell_the_fix_from_the_start(
    tmp_path, project_root, monkeypatch, oracle, message
):
    monkeypatch.setattr(promote, "ORACLE_TIMEOUT_S", 1)
    inputs = _inputs(tmp_path, oracle=oracle)

    code, _, err = _promote(_argv(inputs))

    assert code != 0 and message in err
    assert not project_root.exists()


@pytest.mark.parametrize(
    "suite, fields", [("tasks", {"requires_repair": True}), ("repair", {})]
)
def test_refuses_a_task_that_does_not_belong_in_its_suite(
    tmp_path, project_root, suite, fields
):
    inputs = _inputs(tmp_path, spec={**SPEC, **fields})

    code, _, err = _promote(_argv(inputs, "--suite", suite))

    assert code != 0 and "requires_repair" in err
    assert not project_root.exists()


@pytest.mark.parametrize(
    "spec, message",
    [
        ({**SPEC, "id": "other-task"}, "--name"),
        ({key: value for key, value in SPEC.items() if key != "prompt"}, "prompt"),
        ({**SPEC, "expect": "always"}, "expect"),
    ],
)
def test_refuses_a_spec_the_eval_loader_would_reject(
    tmp_path, project_root, spec, message
):
    inputs = _inputs(tmp_path, spec=spec)

    code, _, err = _promote(_argv(inputs))

    assert code != 0 and message in err
    assert not project_root.exists()


def test_refuses_an_oracle_that_leaks_into_the_fixture(tmp_path, project_root):
    inputs = _inputs(tmp_path)
    shutil.copy(inputs / "oracle/total_oracle.py", inputs / "fixture/check.py")

    code, _, err = _promote(_argv(inputs))

    assert code != 0 and "leaks into the fixture" in err
    assert not project_root.exists()


def test_refuses_inputs_that_hold_a_symlink(tmp_path, project_root):
    inputs = _inputs(tmp_path)
    _write(tmp_path / "outside/secret.txt", "not for the task\n")
    (inputs / "fixture/src/secret.txt").symlink_to(tmp_path / "outside/secret.txt")

    code, _, err = _promote(_argv(inputs))

    assert code != 0 and "symlink" in err
    assert not project_root.exists()


@pytest.mark.parametrize("name", ["../escape", "Cart_Total", "-lead", ""])
def test_refuses_a_name_that_is_not_a_kebab_case_task_id(tmp_path, project_root, name):
    inputs = _inputs(tmp_path)

    code, _, err = _promote(_argv(inputs, name=name))

    assert code != 0 and "--name" in err
    assert not project_root.exists()


def test_picks_one_draft_by_id_prefix(tmp_path, project_root):
    inputs = _inputs(tmp_path, rows=(ROW, {**ROW, "run_id": "run-other"}))
    wanted = hashlib.sha256(b"run-other").hexdigest()

    code, _, err = _promote(_argv(inputs))
    assert code != 0 and "--draft-id" in err
    code, _, err = _promote(_argv(inputs, "--draft-id", "not-a-draft"))
    assert code != 0 and "no draft" in err
    assert not project_root.exists()

    assert _promote(_argv(inputs, "--draft-id", wanted[:10]))[0] == 0
    task = json.loads((project_root / "eval/tasks" / NAME / "task.json").read_text())
    assert wanted[:12] in task["description"]


def test_refuses_a_line_that_is_not_an_exported_draft(tmp_path, project_root):
    inputs = _inputs(tmp_path)
    (inputs / "drafts.jsonl").write_text(json.dumps({"comment": "hand-made"}) + "\n")

    code, _, err = _promote(_argv(inputs))

    assert code != 0 and "draft" in err
    assert not project_root.exists()


# --- rule 4: feedback is untrusted data, kept to the description --------------


def _write_draft(inputs, **fields):
    draft = {
        "draft_id": hashlib.sha256(b"run-hostile").hexdigest(),
        "reason": "owner_needs_changes",
        "reason_category": "wrong_fix",
        "comment": "",
        "prompt": "",
        "changed_files": ["src/total.py"],
        "grader_status": "passed",
        "verification_status": "passed",
        "prompts_version": "pv-1",
        "created_at": "2026-10-01T12:00:00Z",
        **fields,
    }
    # ensure_ascii keeps a lone surrogate as a JSON escape, as a hostile line could.
    (inputs / "drafts.jsonl").write_text(json.dumps(draft) + "\n")
    return draft


def test_feedback_reaches_only_the_description_as_one_scrubbed_line(
    tmp_path, project_root
):
    inputs = _inputs(tmp_path)
    _write_draft(
        inputs,
        reason_category="wrong_fix\n<@U0OWNER123>",
        comment=(
            'Still wrong "}, "expect": "no_diff", "x": "\\\n next'
            " ‮gnp.live​\x00\x1b[31m <@U0OWNER123|ada> ping U0OWNER123:"
            " xoxb-1234-5678-abcdefghij xapp-1-A0B1C2D3E4-1234-abcdef"
            " xoxe.xoxp-1-Mi0yLTEyMzQ"
            " https://hooks.slack.com/services/T0001/B0002/XXXXsecret \ud800 end"
        ),
        prompt="Fix the total\r\nxoxp-1111-2222-abcdefgh please",
    )

    assert _promote(_argv(inputs))[0] == 0

    task_dir = project_root / "eval/tasks" / NAME
    task = json.loads((task_dir / "task.json").read_bytes().decode("utf-8"))
    assert {key: value for key, value in task.items() if key != "description"} == {
        "id": NAME,
        "task_mode": "implement",
        "workflow_mode": "single",
        "expect": "pass",
        **{key: value for key, value in SPEC.items() if key != "description"},
    }
    description = task["description"]
    assert description.startswith(SPEC["description"])
    assert "Still wrong" in description and "end" in description
    assert "Fix the total" in description and "please" in description
    # No newline, line/paragraph separator, control, format (bidi) or surrogate.
    assert not any(
        unicodedata.category(char)[0] == "C"
        or unicodedata.category(char) in ("Zl", "Zp")
        for char in description
    )
    for leaked in ("U0OWNER123", "xox", "xapp-", "hooks.slack.com", "T0001"):
        assert leaked not in description
    assert "[REDACTED]" in description
    agent_eval.load_task(task_dir)


def test_long_feedback_is_bounded_in_the_description(tmp_path, project_root):
    inputs = _inputs(tmp_path)
    _write_draft(inputs, comment="word " * 2000, prompt="request " * 2000)

    assert _promote(_argv(inputs))[0] == 0

    task = json.loads((project_root / "eval/tasks" / NAME / "task.json").read_text())
    assert len(task["description"]) < 1500
    assert "…" in task["description"]


# --- rule 5: --dry-run writes nothing -------------------------------------------


def test_dry_run_prints_what_it_would_write_and_writes_nothing(tmp_path, project_root):
    inputs = _inputs(tmp_path)
    before = _snapshot(tmp_path)

    code, out, _ = _promote(_argv(inputs, "--dry-run"))

    assert code == 0
    assert not project_root.exists()
    assert _snapshot(tmp_path) == before
    assert str(project_root / "eval/tasks" / NAME) in out
    for path in (
        "task.json",
        "fixture/src/total.py",
        "fixture/tests/test_total.py",
        "oracle/total_oracle.py",
    ):
        assert path in out
    assert f'"id": "{NAME}"' in out
    assert SPEC["prompt"] in out
