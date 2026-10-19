import json, os, pathlib, re, secrets, shutil, socket, subprocess, sys, tempfile, time, urllib.request, uuid, argparse, sqlite3

PROJECT = pathlib.Path(__file__).resolve().parents[1]
AGENT_IMAGE = "slack-pi-agent:structured-test"
VERIFICATION_IMAGE = "slack-pi-verification:structured-test"
GATEWAY_IMAGE = "slack-pi-model-gateway:structured-test"
OWNER = "fixture-owner"
ACTIVE_STATUSES = ("running", "starting", "awaiting_approval")
MODEL_PRICES = {
    "openai/gpt-5.6-luna": {"input": 1, "output": 8},
    "openai/gpt-5.6-sol": {"input": 8, "output": 40},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Live, budgeted structured workflow acceptance on a synthetic local repository; no Slack or GitHub publication."
    )
    parser.add_argument(
        "--live-openai",
        action="store_true",
        help="Explicitly allow model calls using OPENAI_API_KEY from .env.demo (up to $5 admission budget per run)",
    )
    parser.add_argument(
        "--exercise-resume",
        action="store_true",
        help="Interrupt read-only discovery after one worker completes and resume the same run",
    )
    args = parser.parse_args(argv)
    if not args.live_openai:
        parser.error("--live-openai is required; this command incurs model usage")
    return args


def run(*args, env=None, input=None):
    return subprocess.run(
        args, env=env, input=input, text=True, capture_output=True, check=True
    ).stdout.strip()


def docker(*args, env=None):
    return run("docker", *args, env=env)


def git(repo, *args):
    return run("git", "-C", str(repo), *args)


def make_root(prefix):
    root = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    root.chmod(0o700)
    return root


def read_openai_key(project=PROJECT):
    values = {}
    for line in (project / ".env.demo").read_text().splitlines():
        match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if match:
            values[match[1]] = match[2].strip().strip("\"'")
    key = values.get("OPENAI_API_KEY")
    assert key, "OpenAI credential missing"
    return key


def verification_image_id():
    return docker("image", "inspect", "--format={{.Id}}", VERIFICATION_IMAGE)


def write_two_bug_fixture(repo):
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src/counter.cjs").write_text("exports.increment = n => n - 1;\n")
    (repo / "src/label.cjs").write_text("exports.label = n => 'total:' + n;\n")
    (repo / "tests/runtime.test.cjs").write_text(
        "const {test}=require('node:test');const a=require('node:assert/strict');test('increment',()=>a.equal(require('../src/counter.cjs').increment(2),3));test('label',()=>a.equal(require('../src/label.cjs').label(2),'count:2'));\n"
    )


def init_fixture_repo(
    repo, remote="https://github.com/fixture/structured-acceptance.git"
):
    """Commit the fixture files already in ``repo``; returns the baseline sha."""
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=t@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    git(repo, "remote", "add", "origin", remote)
    return git(repo, "rev-parse", "HEAD")


def collect_spans(observer):
    spans = []
    otel = pathlib.Path(observer) / "otel.jsonl"
    if not otel.exists():
        return spans
    for line in otel.read_text().splitlines():
        try:
            value = json.loads(line)
            spans.extend(
                span
                for resource in value.get("resourceSpans", [])
                for scope in resource.get("scopeSpans", [])
                for span in scope.get("spans", [])
            )
        except ValueError:
            pass
    return spans


class Topology:
    """Isolated docker networks, OTEL receiver, model gateway and host runtime."""

    def __init__(self, project, root, repo, key, profile_path):
        self.project = project
        self.root = root
        self.repo = repo
        self.key = key
        self.profile_path = profile_path
        suffix = uuid.uuid4().hex[:8]
        self.suffix = suffix
        self.internal = "pi-structured-test-" + suffix
        self.external = self.internal + "-egress"
        self.names = []
        self.networks = []
        self.runtime = None
        self.log = None
        self.observer = root / "observer"
        self.locks = root / "locks"
        self.api_file = root / "api.json"
        self.entry = root / "host.mjs"
        self.token = secrets.token_hex(32)
        self.base = None
        self.env = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _prepare_locks(self):
        self.locks.mkdir()
        plugin = json.loads(
            (self.project / "agent-runtime/plugins.lock.json").read_text()
        )
        plugin["image_digest"] = docker(
            "image", "inspect", "--format={{.Id}}", AGENT_IMAGE
        )
        (self.locks / "plugins.lock.json").write_text(json.dumps(plugin))
        shutil.copytree(self.project / "agent-runtime/plugins", self.locks / "plugins")
        shutil.copy(
            self.project / "agent-runtime/skills.lock.json",
            self.locks / "skills.lock.json",
        )
        shutil.copytree(self.project / "agent-runtime/skills", self.locks / "skills")

    def start(self):
        root = self.root
        (root / "state/budget").mkdir(parents=True)
        self.observer.mkdir()
        self._prepare_locks()
        (self.observer / "receiver.cjs").write_text(
            "const h=require('node:http'),f=require('node:fs');h.createServer((q,s)=>{let a=[];q.on('data',b=>a.push(b));q.on('end',()=>{if(q.method==='POST')f.appendFileSync('/evidence/otel.jsonl',Buffer.concat(a).toString()+'\\n');s.writeHead(200,{'content-type':'application/json'}).end('{}')})}).listen(4318,'0.0.0.0');"
        )
        docker("network", "create", "--internal", self.internal)
        self.networks.append(self.internal)
        docker("network", "create", self.external)
        self.networks.append(self.external)
        receiver = "pi-structured-otel-" + self.suffix
        docker(
            "run",
            "-d",
            "--rm",
            "--name",
            receiver,
            "--network",
            self.external,
            "--network-alias",
            "otel-receiver",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--memory",
            "128m",
            "-p",
            "127.0.0.1::4318",
            "--mount",
            f"type=bind,src={self.observer},dst=/evidence",
            "node:22.22.0-bookworm-slim",
            "node",
            "/evidence/receiver.cjs",
        )
        self.names.append(receiver)
        docker(
            "network", "connect", "--alias", "otel-receiver", self.internal, receiver
        )
        otel_port = docker("port", receiver, "4318/tcp").split(":")[-1]
        signing = secrets.token_hex(32)
        with socket.socket() as chosen:
            chosen.bind(("127.0.0.1", 0))
            host_port = chosen.getsockname()[1]
        prices = json.dumps(MODEL_PRICES)
        gateway_env = {
            **os.environ,
            "OPENAI_API_KEY": self.key,
            "MODEL_GATEWAY_SIGNING_SECRET": signing,
            "PI_AGENT_BUDGET_RPC_URL": f"http://host.docker.internal:{host_port}",
            "PI_AGENT_MODEL_PRICES_JSON": prices,
            "PI_AGENT_OTEL_ENABLED": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-receiver:4318",
        }
        gateway = "pi-structured-models-" + self.suffix
        command = [
            "run",
            "-d",
            "--rm",
            "--name",
            gateway,
            "--network",
            self.internal,
            "--network-alias",
            "model-gateway",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--read-only",
        ]
        for variable in (
            "OPENAI_API_KEY",
            "MODEL_GATEWAY_SIGNING_SECRET",
            "PI_AGENT_BUDGET_RPC_URL",
            "PI_AGENT_MODEL_PRICES_JSON",
            "PI_AGENT_OTEL_ENABLED",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
        ):
            command.extend(["--env", variable])
        docker(*command, GATEWAY_IMAGE, env=gateway_env)
        self.names.append(gateway)
        docker("network", "connect", self.external, gateway)
        self.entry.write_text(
            'import {writeFileSync} from "node:fs";\nconst {createProductionServer}=await import('
            + json.dumps((self.project / "agent-runtime/dist/src/server.js").as_uri())
            + ");\nconst server=await createProductionServer();server.listen("
            + str(host_port)
            + ',"127.0.0.1",()=>writeFileSync('
            + json.dumps(str(self.api_file))
            + ',JSON.stringify({port:server.address().port})));process.on("SIGTERM",()=>server.close());\n'
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if not any(
                part in key
                for part in (
                    "API_KEY",
                    "GITHUB_",
                    "SLACK_",
                    "PI_AGENT_",
                    "MODEL_GATEWAY",
                    "AGENT_RUNTIME",
                )
            )
        }
        env.update(
            {
                "OPENAI_API_KEY": self.key,
                "AGENT_RUNTIME_TOKEN": self.token,
                "MODEL_GATEWAY_SIGNING_SECRET": signing,
                "PI_AGENT_REPO_PATH": str(self.repo),
                "PI_AGENT_STATE_ROOT": str(root / "state"),
                "PI_AGENT_WORKTREE_ROOT": str(root / "worktrees"),
                "PI_AGENT_IMAGE": AGENT_IMAGE,
                "PI_AGENT_NETWORK": self.internal,
                "PI_MODEL_GATEWAY_URL": "http://model-gateway:8080",
                "PI_AGENT_PLUGIN_LOCK": str(self.locks / "plugins.lock.json"),
                "PI_AGENT_SKILL_LOCK": str(self.locks / "skills.lock.json"),
                "PI_AGENT_VERIFICATION_PROFILE": str(self.profile_path),
                "PI_AGENT_MODEL_PRICES_JSON": prices,
                "PI_AGENT_OTEL_ENABLED": "true",
                "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{otel_port}",
                "PI_AGENT_JEV_MODE": "off",
                "PI_AGENT_JEV_CLASSIFIER_MODE": "off",
                "PI_AGENT_MCP_MODE": "off",
                "PI_AGENT_CI_REPAIR_ENABLED": "false",
                "PI_AGENT_TELEMETRY_ENABLED": "true",
                "AGENT_KAFKA_BROKERS": "127.0.0.1:1",
            }
        )
        self.env = env
        self.log = open(root / "runtime.log", "w")
        self.start_runtime()

    def start_runtime(self):
        self.runtime = subprocess.Popen(
            ["node", str(self.entry)], env=self.env, stdout=self.log, stderr=self.log
        )
        for _ in range(200):
            if self.api_file.exists():
                break
            if self.runtime.poll() is not None:
                raise RuntimeError(
                    "runtime startup failed; inspect private fixture log"
                )
            time.sleep(0.1)
        port = json.loads(self.api_file.read_text())["port"]
        self.base = "http://127.0.0.1:" + str(port)

    def request(self, path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path,
            data=payload,
            headers={
                "authorization": "Bearer " + self.token,
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.load(response)

    def close(self):
        if self.runtime:
            self.runtime.terminate()
            try:
                self.runtime.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.runtime.kill()
                self.runtime.wait()
            self.runtime = None
        if self.log:
            self.log.close()
            self.log = None
        for name in reversed(self.names):
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.names.clear()
        for name in reversed(self.networks):
            subprocess.run(
                ["docker", "network", "rm", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.networks.clear()
        return collect_spans(self.observer)


def drive_run(topology, run_id, timeout_s=360, poll_s=0.5):
    """Approve each pending approval as the fixture owner until the run ends.

    A run still active at the deadline is cancelled; returns (state, approvals).
    """
    approvals = set()
    deadline = time.monotonic() + timeout_s
    while True:
        state = topology.request("/v1/runs/" + run_id)
        if state["status"] == "awaiting_approval":
            approval = next(
                (
                    event
                    for event in reversed(state["events"])
                    if event.get("type") == "approval"
                    and event["approval_id"] not in approvals
                ),
                None,
            )
            if approval:
                topology.request(
                    "/v1/runs/" + run_id + "/decisions",
                    {
                        "user_id": OWNER,
                        "approval_id": approval["approval_id"],
                        "decision": "approve",
                    },
                )
                approvals.add(approval["approval_id"])
        if state["status"] not in ACTIVE_STATUSES:
            return state, approvals
        if time.monotonic() >= deadline:
            topology.request("/v1/runs/" + run_id + "/cancel", {"user_id": OWNER})
            return topology.request("/v1/runs/" + run_id), approvals
        time.sleep(poll_s)


def exercise_resume(topology, root, run_id):
    interrupted_nodes = []
    until = time.monotonic() + 240
    while time.monotonic() < until:
        with sqlite3.connect(root / "state/control/state.sqlite") as control:
            nodes = [
                json.loads(row[0])
                for row in control.execute(
                    "SELECT payload FROM workflow_nodes WHERE workflow=?",
                    (run_id,),
                )
            ]
        current = topology.request("/v1/runs/" + run_id)
        interrupted_nodes = [
            node["id"]
            for node in nodes
            if node["role"] != "planner" and node["state"] == "succeeded"
        ]
        if interrupted_nodes and current.get("phase") in ("planning", "investigating"):
            break
        if current["status"] not in ACTIVE_STATUSES:
            raise RuntimeError("Discovery ended before resume could be exercised")
        time.sleep(0.1)
    else:
        raise RuntimeError("No completed worker available to exercise resume")
    topology.runtime.terminate()
    topology.runtime.wait(timeout=25)
    topology.api_file.unlink(missing_ok=True)
    interrupted = json.loads((root / "state/runtime-state.json").read_text())["runs"][0]
    assert interrupted["status"] == "interrupted", interrupted["status"]
    topology.start_runtime()
    topology.request("/v1/runs/" + run_id + "/resume", {"user_id": OWNER})
    return {
        "reused_completed_workers": interrupted_nodes,
        "same_run_id": True,
        "known_cost_before_resume": interrupted.get("estimated_cost_usd"),
    }


def run_acceptance(args):
    project = PROJECT
    root = make_root("structured-acceptance-")
    repo = root / "repo"
    repo.mkdir()
    report = {
        "fixture": str(root),
        "external_publication": False,
        "approvals": "simulated fixture owner",
        "models": "live OpenAI",
        "status": "starting",
    }
    topology = None
    traces = []
    try:
        key = read_openai_key(project)
        verification_image = verification_image_id()
        write_two_bug_fixture(repo)
        init_fixture_repo(repo)
        profile = {
            "id": "structured-acceptance",
            "repository": "fixture/structured-acceptance",
            "image": verification_image,
            "rubric": [
                "The increment function adds exactly one and the label uses count:",
                "Both original regression tests remain intact",
                "Changes stay within the two source modules",
            ],
            "allowedPaths": ["src/"],
            "baselinePaths": ["tests/"],
            "checks": [
                {
                    "name": "unit",
                    "argv": ["node", "--test", "tests/runtime.test.cjs"],
                    "cwd": ".",
                    "timeoutMs": 30000,
                }
            ],
            "structuredWorkflow": {
                "enabled": True,
                "concurrency": 2,
                "maxCostUsd": 5,
                "activeTimeoutMs": 1800000,
                "discoveryCostUsd": 1.5,
            },
        }
        (root / "profile.json").write_text(json.dumps(profile))
        topology = Topology(project, root, repo, key, root / "profile.json")
        topology.start()
        created = topology.request(
            "/v1/sessions",
            {
                "team_id": "fixture",
                "channel_id": "fixture",
                "thread_ts": "1",
                "user_id": OWNER,
                "workflow_mode": "structured",
                "prompt": "Fix two independent bugs in src/counter.cjs and src/label.cjs. increment(2) must return 3 and label(2) must return count:2. Preserve both original tests. Investigate the two modules independently before writing.",
                "routing_text": "Fix counter increment and count label in separate modules, preserving tests.",
            },
        )
        run_id = created["run_id"]
        report["run_id"] = run_id
        if args.exercise_resume:
            report["resume"] = exercise_resume(topology, root, run_id)
        state, approvals = drive_run(topology, run_id)
        tasks = topology.request("/v1/runs/" + run_id + "/tasks")
        (root / "result.json").write_text(
            json.dumps({"run": state, "tasks": tasks}, indent=2)
        )
        report.update(
            {
                "status": state["status"],
                "phase": state.get("phase"),
                "verification": state.get("verification_status"),
                "grader": state.get("grader_status"),
                "nodes": [
                    {
                        "id": node["id"],
                        "role": node["role"],
                        "state": node["state"],
                        "attempt": node["attempt"],
                    }
                    for node in tasks
                ],
                "cost_usd": state.get("estimated_cost_usd"),
                "timing": state.get("timing"),
                "commit": state.get("commit"),
                "approval_count": len(approvals),
            }
        )
    finally:
        if topology:
            traces = topology.close()
        report["exported_spans"] = len(traces)
        report["span_names"] = sorted(set(span.get("name") for span in traces))
        report["trace_ids"] = sorted(set(span.get("traceId") for span in traces))
        (root / "acceptance.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    return report


def exit_code(report):
    """Fail the acceptance unless the run completed and every gate passed."""
    passed = (
        report.get("status") == "completed"
        and report.get("verification") == "passed"
        and report.get("grader") == "passed"
    )
    return 0 if passed else 1


def main(argv=None):
    return exit_code(run_acceptance(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
