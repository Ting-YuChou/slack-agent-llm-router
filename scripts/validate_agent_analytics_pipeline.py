import argparse, asyncio, base64, json, os, pathlib, secrets, shutil, socket, sqlite3, subprocess, sys, tempfile, time, uuid
import clickhouse_connect
from aiokafka.admin import AIOKafkaAdminClient, NewTopic


def main():
    parser = argparse.ArgumentParser(
        description="Replay a private outbox copy through temporary Kafka and ClickHouse using production publisher/consumer; the original outbox is preserved."
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Explicitly allow creating the temporary service containers",
    )
    parser.add_argument("--outbox", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if not args.docker:
        parser.error("--docker is required")
    project = pathlib.Path(__file__).resolve().parents[1]
    source = args.outbox.resolve()
    root = pathlib.Path(tempfile.mkdtemp(prefix="structured-pipeline-"))
    root.chmod(0o700)
    suffix = uuid.uuid4().hex[:8]
    names = []
    network = "pi-structured-data-" + suffix
    consumer = None
    report = {"source_events": 0, "status": "starting"}

    def port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def docker(*args, env=None):
        return subprocess.run(
            ["docker", *args], env=env, text=True, capture_output=True, check=True
        ).stdout.strip()

    async def topics(broker):
        client = AIOKafkaAdminClient(bootstrap_servers=broker, request_timeout_ms=3000)
        try:
            await client.start()
            await client.create_topics(
                [
                    NewTopic(name, 1, 1)
                    for name in ("agent.content.v1", "agent.events.v1", "agent.dlq.v1")
                ]
            )
        finally:
            await client.close()

    try:
        with sqlite3.connect(source) as original, sqlite3.connect(
            root / "outbox.sqlite"
        ) as clone:
            original.backup(clone)
            report["source_events"] = original.execute(
                "SELECT count() FROM outbox"
            ).fetchone()[0]
        assert report["source_events"] > 0
        docker("network", "create", network)
        kafka = "pi-structured-kafka-" + suffix
        ch = "pi-structured-ch-" + suffix
        kp, cp = port(), port()
        pw = secrets.token_hex(20)
        kafka_env = {
            **os.environ,
            "KAFKA_NODE_ID": "1",
            "KAFKA_PROCESS_ROLES": "broker,controller",
            "KAFKA_LISTENERS": "PLAINTEXT://0.0.0.0:29092,PLAINTEXT_HOST://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093",
            "KAFKA_ADVERTISED_LISTENERS": f"PLAINTEXT://kafka:29092,PLAINTEXT_HOST://127.0.0.1:{kp}",
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": "PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT,CONTROLLER:PLAINTEXT",
            "KAFKA_CONTROLLER_LISTENER_NAMES": "CONTROLLER",
            "KAFKA_CONTROLLER_QUORUM_VOTERS": "1@localhost:9093",
            "KAFKA_INTER_BROKER_LISTENER_NAME": "PLAINTEXT",
            "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR": "1",
            "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS": "0",
            "KAFKA_AUTO_CREATE_TOPICS_ENABLE": "true",
            "CLUSTER_ID": base64.urlsafe_b64encode(uuid.uuid4().bytes)
            .decode()
            .rstrip("="),
            "KAFKA_HEAP_OPTS": "-Xms256m -Xmx256m",
        }
        cmd = [
            "run",
            "-d",
            "--rm",
            "--name",
            kafka,
            "--network",
            network,
            "--network-alias",
            "kafka",
            "--memory",
            "1g",
            "--cpus",
            "1",
            "-p",
            f"127.0.0.1:{kp}:9092",
        ]
        for key in kafka_env:
            if key.startswith("KAFKA_") or key == "CLUSTER_ID":
                cmd.extend(["--env", key])
        docker(*cmd, "confluentinc/cp-kafka:7.8.7", env=kafka_env)
        names.append(kafka)
        env = {
            **os.environ,
            "CLICKHOUSE_USER": "acceptance",
            "CLICKHOUSE_PASSWORD": pw,
            "CLICKHOUSE_DB": "default",
        }
        docker(
            "run",
            "-d",
            "--rm",
            "--name",
            ch,
            "--network",
            network,
            "--memory",
            "1g",
            "--cpus",
            "1",
            "-p",
            f"127.0.0.1:{cp}:8123",
            "--env",
            "CLICKHOUSE_USER",
            "--env",
            "CLICKHOUSE_PASSWORD",
            "--env",
            "CLICKHOUSE_DB",
            "clickhouse/clickhouse-server:24.8",
            env=env,
        )
        names.append(ch)
        for _ in range(90):
            try:
                client = clickhouse_connect.get_client(
                    host="127.0.0.1",
                    port=cp,
                    username="acceptance",
                    password=pw,
                    autogenerate_session_id=False,
                )
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("ClickHouse not ready")
        for attempt in range(60):
            try:
                asyncio.run(topics(f"127.0.0.1:{kp}"))
                break
            except Exception:
                if attempt == 59:
                    raise
                time.sleep(0.5)
        env.update(
            {
                "CLICKHOUSE_HOST": "127.0.0.1",
                "CLICKHOUSE_PORT": str(cp),
                "AGENT_KAFKA_BROKERS": f"127.0.0.1:{kp}",
            }
        )
        log = open(root / "consumer.log", "w")
        consumer = subprocess.Popen(
            [str(project / ".venv/bin/python"), "-m", "src.agent_analytics"],
            env=env,
            stdout=log,
            stderr=log,
        )
        node = root / "publish.mjs"
        node.write_text(
            "const {Kafka,logLevel}=await import("
            + json.dumps(
                (project / "agent-runtime/node_modules/kafkajs/index.js").as_uri()
            )
            + ");const {TelemetryStore,publishOutbox}=await import("
            + json.dumps((project / "agent-runtime/dist/src/telemetry.js").as_uri())
            + ");const store=new TelemetryStore("
            + json.dumps(str(root / "outbox.sqlite"))
            + ",200000000);const producer=new Kafka({brokers:["
            + json.dumps(f"127.0.0.1:{kp}")
            + "],logLevel:logLevel.NOTHING}).producer({idempotent:true,maxInFlightRequests:1});try{await producer.connect();while(store.health().depth){await publishOutbox(store,producer)}}finally{await producer.disconnect();store.close()}"
        )
        subprocess.run(["node", str(node)], check=True, capture_output=True, text=True)
        for _ in range(180):
            if consumer.poll() is not None:
                raise RuntimeError(
                    "Analytics consumer exited; inspect private test log"
                )
            try:
                count = client.query(
                    "SELECT count() FROM agent_events_latest"
                ).first_row[0]
                if count >= report["source_events"]:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Timed out waiting for events in ClickHouse")
        first = client.query("SELECT count() FROM agent_usage_latest").first_row[0]
        # Replay exact source event IDs through the same producer and real consumer.
        with sqlite3.connect(source) as original, sqlite3.connect(
            root / "outbox.sqlite"
        ) as clone:
            original.backup(clone)
        subprocess.run(["node", str(node)], check=True, capture_output=True, text=True)
        time.sleep(2)
        second = client.query("SELECT count() FROM agent_usage_latest").first_row[0]
        assert first == second, (first, second)
        report.update(
            {
                "status": "passed",
                "events_in_clickhouse": count,
                "canonical_usage_rows": first,
                "usage_after_replay": second,
                "roles": client.query(
                    "SELECT DISTINCT role FROM agent_executions_latest ORDER BY role"
                ).result_rows,
                "node_states": client.query(
                    "SELECT node_id,argMax(state,version),max(attempt) FROM agent_workflow_nodes_latest GROUP BY node_id ORDER BY node_id"
                ).result_rows,
                "accounting": client.query("SELECT * FROM agent_workflow_usage")
                .named_results()
                .__next__(),
            }
        )
    finally:
        if consumer:
            consumer.terminate()
            try:
                consumer.wait(timeout=12)
            except subprocess.TimeoutExpired:
                consumer.kill()
                consumer.wait()
        for name in reversed(names):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "network", "rm", network], capture_output=True)
        (root / "result.json").write_text(json.dumps(report, default=str, indent=2))
        print(json.dumps({**report, "evidence": str(root)}, default=str, indent=2))


if __name__ == "__main__":
    main()
