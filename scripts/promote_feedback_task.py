"""Promote one owner-feedback draft into a hidden-oracle eval task.

``scripts/export_agent_feedback.py`` writes de-identified drafts. This script
turns one of them into ``eval/tasks/<name>/`` (``eval/repair-tasks/<name>/``
with ``--suite repair``). The draft holds no repository, so the operator
supplies what it lacks: a spec with the task fields (prompt, profile,
oracle argv), a fixture, a hidden oracle written by hand and a reference fix.
The script never writes or infers an oracle.

Nothing is written unless the task loads with ``agent_eval.load_task``, belongs
in its suite (repair tasks, and only they, set ``requires_repair``), and its
oracle fails on the unmodified fixture and passes once the reference fix is laid
over it. The oracle runs locally, not in the verification image. An existing
task, in either suite, is never overwritten. The draft's comment and request are
untrusted Slack text: they reach only the description, each as one bounded line
without Slack ids, tokens or control characters. ``--dry-run`` runs every check
and prints what it would write.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

try:
    from scripts import agent_eval
    from scripts.export_agent_feedback import REASONS, _clean
except ImportError:  # executed as ``python scripts/promote_feedback_task.py``
    import agent_eval
    from export_agent_feedback import REASONS, _clean

TASK_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DRAFT_ID = re.compile(r"[0-9a-f]{64}")
# Task fields a spec may set, in task.json order; the id comes from --name.
SPEC_FIELDS = (
    "description",
    "task_mode",
    "workflow_mode",
    "expect",
    "requires_repair",
    "requires_run_checks",
    "requires_degrade",
    "delivery_decision",
    "mcp_servers",
    "prompt",
    "routing_text",
    "profile",
    "oracle",
)
SPEC_DEFAULTS = {"task_mode": "implement", "workflow_mode": "single", "expect": "pass"}
# Slack tokens and webhook URLs the export's redaction does not cover
# (app-level xapp-, rotating xoxe.); a token always carries a digit.
_SLACK_SECRET = re.compile(
    r"\bx(?:app|ox[a-z])[-.](?=[A-Za-z0-9.-]*\d)[A-Za-z0-9.-]+"
    r"|https?://hooks\.slack(?:-gov)?\.com/\S*"
)
QUOTE_LIMIT = 500
ORACLE_TIMEOUT_S = 120


class Refusal(Exception):
    pass


def quote(text):
    """Untrusted ``text`` as one bounded line without Slack ids, tokens or control characters."""
    if not isinstance(text, str):
        return ""
    # Drop control, format (bidi, zero-width), surrogate and unassigned characters
    # first, so none can split a token past the redaction below.
    text = "".join(
        char for char in text if char.isspace() or unicodedata.category(char)[0] != "C"
    )
    text = _clean(_SLACK_SECRET.sub("[REDACTED]", text), ())
    line = " ".join(text.split())
    if len(line) <= QUOTE_LIMIT:
        return line
    return line[: QUOTE_LIMIT - 1] + "\u2026"


def describe(summary, draft):
    """The operator's summary, then where the task came from and the quoted feedback."""
    category = quote(draft.get("reason_category"))
    origin = draft["reason"] + (f": {category}" if category else "")
    parts = [" ".join(summary.split())] if summary else []
    parts.append(
        f"Promoted from feedback draft {draft['draft_id'][:12]} ({origin});"
        " the quotes are untrusted Slack text, kept as data."
    )
    for label, field in (("Owner comment", "comment"), ("Original request", "prompt")):
        text = quote(draft.get(field))
        if text:
            parts.append(f'{label}: "{text}".')
    return " ".join(parts)


def _read_json(path, label):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refusal(f"unreadable {label} {path}: {error}")


def load_draft(path, draft_id=None):
    """The one draft in the JSONL file ``path`` whose id starts with ``draft_id``."""
    try:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        drafts = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as error:
        raise Refusal(f"unreadable drafts file {path}: {error}")
    matches = [
        draft
        for draft in drafts
        if draft_id is None
        or (isinstance(draft, dict) and str(draft.get("draft_id")).startswith(draft_id))
    ]
    if not matches:
        wanted = f"matches --draft-id {draft_id}" if draft_id else "found"
        raise Refusal(f"no draft {wanted} in {path}")
    if len(matches) > 1:
        raise Refusal(f"{len(matches)} drafts in {path}; pick one with --draft-id")
    [draft] = matches
    if (
        not isinstance(draft, dict)
        or not DRAFT_ID.fullmatch(str(draft.get("draft_id")))
        or draft.get("reason") not in REASONS
    ):
        raise Refusal(
            f"{path} does not hold an export_agent_feedback draft "
            "(a sha256 draft_id and a known reason)"
        )
    return draft


def load_spec(path):
    spec = _read_json(path, "spec")
    if not isinstance(spec, dict):
        raise Refusal(f"spec {path} must be a JSON object of task fields")
    unknown = sorted(set(spec) - set(SPEC_FIELDS))
    if unknown:
        raise Refusal(
            f"spec {path} sets fields a spec may not: {', '.join(unknown)} "
            "(the task id comes from --name)"
        )
    if not isinstance(spec.get("description", ""), str):
        raise Refusal(f"spec {path}: description must be a string")
    return spec


def build_task(name, spec, draft):
    """task.json for ``name``: the spec's fields; the draft reaches only the description."""
    fields = {
        **SPEC_DEFAULTS,
        **spec,
        "description": describe(spec.get("description"), draft),
    }
    return {"id": name, **{key: fields[key] for key in SPEC_FIELDS if key in fields}}


def _inputs(path, label, allow_file=False):
    """``{relative path: source}`` of the regular files at ``path``, which holds no symlink."""
    path = pathlib.Path(path)
    if path.is_symlink():
        raise Refusal(f"{label} {path} is a symlink")
    if allow_file and path.is_file():
        return {path.name: path}
    if not path.is_dir():
        kind = "directory or file" if allow_file else "directory"
        raise Refusal(f"{label} {path} is not a {kind}")
    files = {}
    for root, dirs, names in os.walk(path):
        for name in sorted(dirs + names):
            entry = pathlib.Path(root, name)
            if entry.is_symlink():
                raise Refusal(
                    f"{label} {path} holds a symlink, {entry.relative_to(path)}"
                )
            if entry.is_file():
                files[entry.relative_to(path).as_posix()] = entry
    if not files:
        raise Refusal(f"{label} {path} is empty")
    return dict(sorted(files.items()))


def _place(files, destination):
    for relative, source in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


def _verdict(argv, tree):
    try:
        result = subprocess.run(
            argv, cwd=tree, capture_output=True, timeout=ORACLE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {ORACLE_TIMEOUT_S}s"
    except OSError as error:
        return f"could not run ({error})"
    return "passed" if result.returncode == 0 else "failed"


def check_oracle(task, solution, scratch):
    """The oracle must fail on the unmodified fixture and pass with the reference fix."""
    fixture = _inputs(task["fixture_dir"], "fixture")
    oracle = _inputs(task["oracle_dir"], "oracle")
    for label, overlay, wanted in (
        ("the unmodified fixture", {}, "failed"),
        ("the reference fix (--solution)", solution, "passed"),
    ):
        tree = scratch / ("fixed" if overlay else "start")
        _place({**fixture, **overlay}, tree)
        _place(oracle, tree / agent_eval.ORACLE_DIR)
        verdict = _verdict(task["oracle"]["argv"], tree)
        if verdict == wanted:
            continue
        if verdict == "passed":
            raise Refusal(
                f"the oracle passed on {label}, so it cannot tell a fix from no change"
            )
        must = "fail" if wanted == "failed" else "pass"
        raise Refusal(f"the oracle {verdict} on {label}; it must {must} there")


def check_suite(task, suite):
    if bool(task.get("requires_repair")) == (suite == "repair"):
        return
    if suite == "repair":
        raise Refusal("the repair suite holds only tasks that set requires_repair")
    raise Refusal(
        "a task with requires_repair belongs in --suite repair; "
        "the main suite measures first attempts"
    )


def suite_dir(suite):
    return agent_eval.PROJECT / agent_eval.SUITES[suite][0]


def promote(args):
    """Check the promotion in a scratch directory; write the task unless ``--dry-run``."""
    if not TASK_NAME.fullmatch(args.name):
        raise Refusal(f"--name {args.name!r} must be a kebab-case task id")
    if args.oracle is None:
        raise Refusal(
            "no human-written oracle: pass --oracle DIR|FILE with the hidden "
            "checks you wrote for this task; this script never writes one"
        )
    target = suite_dir(args.suite) / args.name
    for suite in agent_eval.SUITES:
        taken = suite_dir(suite) / args.name
        if os.path.lexists(taken):
            raise Refusal(f"{taken} already exists; promotion never overwrites a task")
    draft = load_draft(args.drafts, args.draft_id)
    spec = load_spec(args.spec)
    oracle = _inputs(args.oracle, "oracle", allow_file=True)
    fixture = _inputs(args.fixture, "fixture")
    solution = _inputs(args.solution, "solution")
    text = json.dumps(build_task(args.name, spec, draft), indent=2, ensure_ascii=False)
    with tempfile.TemporaryDirectory(prefix="promote-feedback-task-") as scratch:
        stage = pathlib.Path(scratch) / "task" / args.name
        _place(fixture, stage / "fixture")
        _place(oracle, stage / "oracle")
        (stage / "task.json").write_text(text + "\n", encoding="utf-8")
        try:
            task = agent_eval.load_task(stage)
        except agent_eval.TaskError as error:
            raise Refusal(f"invalid eval task: {error}")
        check_suite(task, args.suite)
        check_oracle(task, solution, pathlib.Path(scratch))
        written = ["task.json"]
        written += [f"fixture/{path}" for path in fixture]
        written += [f"oracle/{path}" for path in oracle]
        if args.dry_run:
            print(f"dry run, nothing written; would create {target}/ with:")
            print("\n".join(f"  {path}" for path in written))
            print(f"--- task.json\n{text}")
            return
        try:
            shutil.copytree(stage, target)
        except FileExistsError:
            raise Refusal(f"{target} already exists; promotion never overwrites a task")
    print(
        f"promoted draft {draft['draft_id'][:12]} to {target}; review and commit "
        f"it, and the next live eval of the {args.suite} suite records it",
        file=sys.stderr,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", type=pathlib.Path, help="export_agent_feedback JSONL")
    parser.add_argument(
        "--draft-id", help="Prefix of the draft to promote; needed for several drafts"
    )
    parser.add_argument("--name", required=True, help="The new task id (kebab-case)")
    parser.add_argument(
        "--suite",
        choices=agent_eval.SUITES,
        default="tasks",
        help="Suite the task joins (default tasks)",
    )
    parser.add_argument(
        "--spec",
        type=pathlib.Path,
        required=True,
        help="JSON task fields: prompt, profile, oracle.argv and optional others",
    )
    parser.add_argument(
        "--fixture", type=pathlib.Path, required=True, help="Starting repository"
    )
    parser.add_argument(
        "--oracle",
        type=pathlib.Path,
        help="Hidden oracle written by hand (directory or file); required",
    )
    parser.add_argument(
        "--solution",
        type=pathlib.Path,
        required=True,
        help="Reference fix: files laid over the fixture; the oracle must pass on it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and print what would be written; write nothing",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        promote(args)
    except Refusal as error:
        print(f"refusing to promote: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
