import http.client, json, os, pathlib, re, secrets, shutil, socket, subprocess, tempfile, time, urllib.request, uuid, argparse, sqlite3


def main():
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
    args = parser.parse_args()
    if not args.live_openai:
        parser.error("--live-openai is required; this command incurs model usage")
    project = pathlib.Path(__file__).resolve().parents[1]
    root = pathlib.Path(tempfile.mkdtemp(prefix="structured-acceptance-"))
    root.chmod(0o700)
    suffix = uuid.uuid4().hex[:8]
    internal = "pi-structured-test-" + suffix
    external = internal + "-egress"
    names = []
    networks = []
    runtime = None
    ledger = root / "state/budget"
    ledger.mkdir(parents=True)
    observer = root / "observer"
    observer.mkdir()
    repo = root / "repo"
    repo.mkdir()
    locks = root / "locks"
    locks.mkdir()

    def run(*args, env=None, input=None):
        return subprocess.run(
            args, env=env, input=input, text=True, capture_output=True, check=True
        ).stdout.strip()

    def docker(*args, env=None):
        return run("docker", *args, env=env)

    def git(*args):
        return run("git", "-C", str(repo), *args)

    def request(base, path, token, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            base + path,
            data=payload,
            headers={
                "authorization": "Bearer " + token,
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.load(response)

    report = {
        "fixture": str(root),
        "external_publication": False,
        "approvals": "simulated fixture owner",
        "models": "live OpenAI",
        "status": "starting",
    }
    try:
        values = {}
        for line in (project / ".env.demo").read_text().splitlines():
            match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
            if match:
                values[match[1]] = match[2].strip().strip("\"'")
        key = values.get("OPENAI_API_KEY")
        assert key, "OpenAI credential missing"
        agent_image = "slack-pi-agent:structured-test"
        verification_image = docker(
            "image",
            "inspect",
            "--format={{.Id}}",
            "slack-pi-verification:structured-test",
        )
        plugin = json.loads((project / "agent-runtime/plugins.lock.json").read_text())
        plugin["image_digest"] = docker(
            "image", "inspect", "--format={{.Id}}", agent_image
        )
        (locks / "plugins.lock.json").write_text(json.dumps(plugin))
        shutil.copytree(project / "agent-runtime/plugins", locks / "plugins")
        shutil.copy(
            project / "agent-runtime/skills.lock.json", locks / "skills.lock.json"
        )
        shutil.copytree(project / "agent-runtime/skills", locks / "skills")
        (repo / "src").mkdir()
        (repo / "tests").mkdir()
        (repo / "src/counter.cjs").write_text("exports.increment = n => n - 1;\n")
        (repo / "src/label.cjs").write_text("exports.label = n => 'total:' + n;\n")
        (repo / "tests/runtime.test.cjs").write_text(
            "const {test}=require('node:test');const a=require('node:assert/strict');test('increment',()=>a.equal(require('../src/counter.cjs').increment(2),3));test('label',()=>a.equal(require('../src/label.cjs').label(2),'count:2'));\n"
        )
        git("init", "-q")
        git("add", ".")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "fixture",
        )
        git(
            "remote",
            "add",
            "origin",
            "https://github.com/fixture/structured-acceptance.git",
        )
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
        (observer / "receiver.cjs").write_text(
            "const h=require('node:http'),f=require('node:fs');h.createServer((q,s)=>{let a=[];q.on('data',b=>a.push(b));q.on('end',()=>{if(q.method==='POST')f.appendFileSync('/evidence/otel.jsonl',Buffer.concat(a).toString()+'\\n');s.writeHead(200,{'content-type':'application/json'}).end('{}')})}).listen(4318,'0.0.0.0');"
        )
        docker("network", "create", "--internal", internal)
        networks.append(internal)
        docker("network", "create", external)
        networks.append(external)
        receiver = "pi-structured-otel-" + suffix
        docker(
            "run",
            "-d",
            "--rm",
            "--name",
            receiver,
            "--network",
            external,
            "--network-alias",
            "otel-receiver",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--memory",
            "128m",
            "-p",
            "127.0.0.1::4318",
            "--mount",
            f"type=bind,src={observer},dst=/evidence",
            "node:22.22.0-bookworm-slim",
            "node",
            "/evidence/receiver.cjs",
        )
        names.append(receiver)
        docker("network", "connect", "--alias", "otel-receiver", internal, receiver)
        otel_port = docker("port", receiver, "4318/tcp").split(":")[-1]
        signing = secrets.token_hex(32)
        token = secrets.token_hex(32)
        with socket.socket() as chosen:
            chosen.bind(("127.0.0.1", 0))
            host_port = chosen.getsockname()[1]
        prices = json.dumps(
            {
                "openai/gpt-5.6-luna": {"input": 1, "output": 8},
                "openai/gpt-5.6-sol": {"input": 8, "output": 40},
            }
        )
        gateway_env = {
            **os.environ,
            "OPENAI_API_KEY": key,
            "MODEL_GATEWAY_SIGNING_SECRET": signing,
            "PI_AGENT_BUDGET_RPC_URL": f"http://host.docker.internal:{host_port}",
            "PI_AGENT_MODEL_PRICES_JSON": prices,
            "PI_AGENT_OTEL_ENABLED": "true",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-receiver:4318",
        }
        gateway = "pi-structured-models-" + suffix
        command = [
            "run",
            "-d",
            "--rm",
            "--name",
            gateway,
            "--network",
            internal,
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
        docker(*command, "slack-pi-model-gateway:structured-test", env=gateway_env)
        names.append(gateway)
        docker("network", "connect", external, gateway)
        api_file = root / "api.json"
        entry = root / "host.mjs"
        entry.write_text(
            'import {writeFileSync} from "node:fs";\nconst {createProductionServer}=await import('
            + json.dumps((project / "agent-runtime/dist/src/server.js").as_uri())
            + ");\nconst server=await createProductionServer();server.listen("
            + str(host_port)
            + ',"127.0.0.1",()=>writeFileSync('
            + json.dumps(str(api_file))
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
                "OPENAI_API_KEY": key,
                "AGENT_RUNTIME_TOKEN": token,
                "MODEL_GATEWAY_SIGNING_SECRET": signing,
                "PI_AGENT_REPO_PATH": str(repo),
                "PI_AGENT_STATE_ROOT": str(root / "state"),
                "PI_AGENT_WORKTREE_ROOT": str(root / "worktrees"),
                "PI_AGENT_IMAGE": agent_image,
                "PI_AGENT_NETWORK": internal,
                "PI_MODEL_GATEWAY_URL": "http://model-gateway:8080",
                "PI_AGENT_PLUGIN_LOCK": str(locks / "plugins.lock.json"),
                "PI_AGENT_SKILL_LOCK": str(locks / "skills.lock.json"),
                "PI_AGENT_VERIFICATION_PROFILE": str(root / "profile.json"),
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
        log = open(root / "runtime.log", "w")
        runtime = subprocess.Popen(
            ["node", str(entry)], env=env, stdout=log, stderr=log
        )
        for _ in range(200):
            if api_file.exists():
                break
            if runtime.poll() is not None:
                raise RuntimeError(
                    "runtime startup failed; inspect private fixture log"
                )
            time.sleep(0.1)
        base = "http://127.0.0.1:" + str(json.loads(api_file.read_text())["port"])
        created = request(
            base,
            "/v1/sessions",
            token,
            {
                "team_id": "fixture",
                "channel_id": "fixture",
                "thread_ts": "1",
                "user_id": "fixture-owner",
                "workflow_mode": "structured",
                "prompt": "Fix two independent bugs in src/counter.cjs and src/label.cjs. increment(2) must return 3 and label(2) must return count:2. Preserve both original tests. Investigate the two modules independently before writing.",
                "routing_text": "Fix counter increment and count label in separate modules, preserving tests.",
            },
        )
        run_id = created["run_id"]
        report["run_id"] = run_id
        if args.exercise_resume:
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
                current = request(base, "/v1/runs/" + run_id, token)
                interrupted_nodes = [
                    node["id"]
                    for node in nodes
                    if node["role"] != "planner" and node["state"] == "succeeded"
                ]
                if interrupted_nodes and current.get("phase") in (
                    "planning",
                    "investigating",
                ):
                    break
                if current["status"] not in (
                    "running",
                    "starting",
                    "awaiting_approval",
                ):
                    raise RuntimeError(
                        "Discovery ended before resume could be exercised"
                    )
                time.sleep(0.1)
            else:
                raise RuntimeError("No completed worker available to exercise resume")
            runtime.terminate()
            runtime.wait(timeout=25)
            api_file.unlink(missing_ok=True)
            interrupted = json.loads((root / "state/runtime-state.json").read_text())[
                "runs"
            ][0]
            assert interrupted["status"] == "interrupted", interrupted["status"]
            runtime = subprocess.Popen(
                ["node", str(entry)], env=env, stdout=log, stderr=log
            )
            for _ in range(200):
                if api_file.exists():
                    break
                if runtime.poll() is not None:
                    raise RuntimeError("Resume runtime startup failed")
                time.sleep(0.1)
            request(
                base,
                "/v1/runs/" + run_id + "/resume",
                token,
                {"user_id": "fixture-owner"},
            )
            report["resume"] = {
                "reused_completed_workers": interrupted_nodes,
                "same_run_id": True,
                "known_cost_before_resume": interrupted.get("estimated_cost_usd"),
            }
        approvals = set()
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            state = request(base, "/v1/runs/" + run_id, token)
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
                    request(
                        base,
                        "/v1/runs/" + run_id + "/decisions",
                        token,
                        {
                            "user_id": "fixture-owner",
                            "approval_id": approval["approval_id"],
                            "decision": "approve",
                        },
                    )
                    approvals.add(approval["approval_id"])
            if state["status"] not in ("running", "starting", "awaiting_approval"):
                break
            time.sleep(0.5)
        else:
            request(
                base,
                "/v1/runs/" + run_id + "/cancel",
                token,
                {"user_id": "fixture-owner"},
            )
            state = request(base, "/v1/runs/" + run_id, token)
        tasks = request(base, "/v1/runs/" + run_id + "/tasks", token)
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
        if runtime:
            runtime.terminate()
            try:
                runtime.wait(timeout=20)
            except subprocess.TimeoutExpired:
                runtime.kill()
                runtime.wait()
        for name in reversed(names):
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for name in reversed(networks):
            subprocess.run(
                ["docker", "network", "rm", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        traces = []
        if (observer / "otel.jsonl").exists():
            for line in (observer / "otel.jsonl").read_text().splitlines():
                try:
                    value = json.loads(line)
                    traces.extend(
                        span
                        for resource in value.get("resourceSpans", [])
                        for scope in resource.get("scopeSpans", [])
                        for span in scope.get("spans", [])
                    )
                except ValueError:
                    pass
        report["exported_spans"] = len(traces)
        report["span_names"] = sorted(set(span.get("name") for span in traces))
        report["trace_ids"] = sorted(set(span.get("traceId") for span in traces))
        (root / "acceptance.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
