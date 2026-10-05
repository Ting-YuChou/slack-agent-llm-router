"""Opt-in real SQLite → Kafka → consumer → ClickHouse test using synthetic Pi entries.

Requires a running disposable broker/ClickHouse and a built agent-runtime. This
creates and removes its own database and uses its own Kafka consumer group.
"""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid


@unittest.skipUnless(
    os.environ.get("AGENT_OBSERVABILITY_INTEGRATION") == "1",
    "requires explicit disposable Kafka/ClickHouse integration environment",
)
class AgentObservabilityIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_outbox_replay_large_content_and_canonical_usage(self):
        import clickhouse_connect
        from aiokafka import AIOKafkaConsumer
        from src.agent_analytics import AgentAnalyticsWorker

        root = Path(__file__).resolve().parents[1]
        suffix = uuid.uuid4().hex
        database = "agent_observability_test_" + suffix
        run_id = "integration-" + suffix
        brokers = os.environ["AGENT_TEST_KAFKA_BROKERS"].split(",")
        connection = {
            "host": os.environ["AGENT_TEST_CLICKHOUSE_HOST"],
            "port": int(os.environ.get("AGENT_TEST_CLICKHOUSE_PORT", "8123")),
            "username": os.environ["AGENT_TEST_CLICKHOUSE_USER"],
            "password": os.environ["AGENT_TEST_CLICKHOUSE_PASSWORD"],
            "autogenerate_session_id": False,
        }
        admin = clickhouse_connect.get_client(**connection)
        admin.command(f"CREATE DATABASE {database}")
        client = clickhouse_connect.get_client(**connection, database=database)

        def consumer_factory(*args, **kwargs):
            kwargs["group_id"] = "agent-observability-test-" + suffix
            return AIOKafkaConsumer(*args, **kwargs)

        worker = AgentAnalyticsWorker(
            client, brokers, consumer_factory=consumer_factory
        )
        task = None
        try:
            await worker.migrate()
            task = asyncio.create_task(worker.run())
            with tempfile.TemporaryDirectory() as directory:
                # The real writer thread owns SQLite and publishes through KafkaJS.
                program = """
import { AgentTelemetry } from './agent-runtime/dist/src/telemetry.js';
const t = new AgentTelemetry({file: process.env.TEST_OUTBOX,
  brokers: JSON.parse(process.env.TEST_BROKERS), secrets: ['integration-private-key']});
const run = process.env.TEST_RUN;
try {
  if(!await t.record(run, 's', 'session_stats', {phase:'start', pi_session_id:'p',
    stats:{tokens:{total:0},cost:0}}, run+':stats:start')) throw new Error('snapshot append failed');
  const ok = await t.record(run, 's', 'session_entry', {pi_session_id:'p', entry:{
    id:'entry', type:'message', message:{role:'assistant', model:'synthetic',
      content:[{type:'text', text:'中文'.repeat(150000)+' integration-private-key'}],
      usage:{input:2,output:3,cacheRead:1,cacheWrite:0,totalTokens:6,reasoning:1,
        futureMetric:17,cost:{input:.1,output:.2,cacheRead:.01,cacheWrite:0,total:.31}}}
  }}, run+':p:entry');
  if(!ok) throw new Error('outbox append failed');
  if(!await t.record(run, 's', 'session_stats', {phase:'end', pi_session_id:'p',
    stats:{tokens:{total:6},cost:.31}}, run+':stats:end')) throw new Error('snapshot append failed');
  if(!await t.record(run, 's', 'run', {status:'completed', model:'synthetic',
    capture_complete:true})) throw new Error('run append failed');
  if(!await t.record(run, 's', 'integration_marker', {})) throw new Error('marker append failed');
  const deadline = Date.now()+60000;
  while(!t.health().kafka_connected || t.health().depth !== 0) {
    if(Date.now()>deadline) throw new Error('publisher timed out');
    await new Promise(resolve=>setTimeout(resolve,200));
  }
} finally { await t.close(); }
"""
                environment = {
                    **os.environ,
                    "TEST_RUN": run_id,
                    "TEST_OUTBOX": str(Path(directory) / "outbox.sqlite"),
                    "TEST_BROKERS": json.dumps(brokers),
                }
                for _ in range(2):
                    process = await asyncio.create_subprocess_exec(
                        "node",
                        "--input-type=module",
                        "-e",
                        program,
                        cwd=root,
                        env=environment,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        _, stderr = await asyncio.wait_for(process.communicate(), 75)
                    except BaseException:
                        process.kill()
                        await process.wait()
                        raise
                    self.assertEqual(process.returncode, 0, stderr.decode())

                async def await_projection():
                    while True:
                        if task.done():
                            await task  # Surface worker failures immediately.
                            self.fail("analytics worker stopped unexpectedly")
                        result = await asyncio.to_thread(
                            client.query,
                            "SELECT count() FROM agent_events_latest "
                            "WHERE run_id={run:String} AND kind='integration_marker'",
                            parameters={"run": run_id},
                        )
                        if result.result_rows[0][0] >= 2:
                            return await asyncio.to_thread(
                                client.query,
                                "SELECT count(), sum(total_tokens), any(payload_json) "
                                "FROM agent_usage_latest WHERE run_id={run:String}",
                                parameters={"run": run_id},
                            )
                        await asyncio.sleep(0.2)

                projection = await asyncio.wait_for(await_projection(), 60)
                count, tokens, payload = projection.result_rows[0]
                self.assertEqual((count, tokens), (1, 6))
                self.assertEqual(json.loads(payload)["usage"]["futureMetric"], 17)
                reconciliation = await asyncio.to_thread(
                    client.query,
                    "SELECT reconciliation FROM agent_usage_reconciliation WHERE run_id={run:String}",
                    parameters={"run": run_id},
                )
                self.assertEqual(reconciliation.result_rows, [("matched",)])
                result = await asyncio.to_thread(
                    client.query,
                    "SELECT payload_json FROM agent_events_latest "
                    "WHERE run_id={run:String} AND kind='session_entry'",
                    parameters={"run": run_id},
                )
                self.assertEqual(len(result.result_rows), 1)
                raw = json.loads(result.result_rows[0][0])
                text = raw["entry"]["message"]["content"][0]["text"]
                self.assertEqual(text, "中文" * 150000 + " [REDACTED]")
                self.assertTrue(
                    await asyncio.to_thread(
                        lambda: client.query(
                            "SELECT count() FROM agent_content_chunks_latest WHERE run_id={run:String}",
                            parameters={"run": run_id},
                        ).result_rows[0][0]
                        >= 2
                    )
                )
        finally:
            worker.stop.set()
            try:
                if task:
                    try:
                        await asyncio.wait_for(asyncio.shield(task), 10)
                    except (TimeoutError, asyncio.TimeoutError):
                        task.cancel()
                    finally:
                        await asyncio.gather(task, return_exceptions=True)
            finally:
                client.close()
                try:
                    admin.command(f"DROP DATABASE IF EXISTS {database}")
                finally:
                    admin.close()
