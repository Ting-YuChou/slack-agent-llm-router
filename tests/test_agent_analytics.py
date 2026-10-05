import unittest
import importlib


class AgentAnalyticsTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module("src.agent_analytics")
        except ImportError:
            self.fail("agent analytics module must exist")

    def test_ledger_only_uses_session_entries_and_preserves_missing(self):
        module = self.module()
        envelope = {
            "event_id": "s:p:e",
            "run_id": "r",
            "session_id": "s",
            "collected_at": "2026-10-04T00:00:00Z",
            "sequence": 1,
            "kind": "session_entry",
            "payload": {
                "pi_session_id": "p",
                "entry": {
                    "id": "e",
                    "type": "message",
                    "parentId": "a",
                    "message": {
                        "role": "assistant",
                        "usage": {
                            "input": 2,
                            "output": 3,
                            "reasoning": 1,
                            "future": 8,
                            "cost": {"total": 0.5},
                        },
                    },
                },
            },
        }
        rows = module.project(envelope)
        usage = rows["agent_usage"][0]
        self.assertEqual(usage["output"], 3)
        self.assertEqual(usage["reasoning"], 1)
        self.assertIsNone(usage["cache_read"])
        self.assertIn("future", usage["payload_json"])
        envelope["kind"] = "rpc"
        self.assertNotIn("agent_usage", module.project(envelope))

    def test_usage_entries_compaction_and_tool_usage(self):
        module = self.module()
        for entry in [
            {
                "id": "e",
                "type": "usage",
                "kind": "classification",
                "model": "typesafe/jev-1.13",
                "provider": "openrouter",
                "usage": {"totalTokens": 4},
            },
            {"id": "e", "type": "compaction", "usage": {"totalTokens": 4}},
            {
                "id": "e",
                "type": "message",
                "message": {"role": "toolResult", "usage": {"totalTokens": 4}},
            },
        ]:
            rows = module.project(
                {
                    "event_id": "e",
                    "run_id": "r",
                    "session_id": "s",
                    "kind": "session_entry",
                    "payload": {"entry": entry},
                }
            )
            self.assertEqual(rows["agent_usage"][0]["total_tokens"], 4)
            if entry.get("model"):
                import json

                metadata = json.loads(rows["agent_usage"][0]["payload_json"])
                self.assertEqual(metadata["model"], entry["model"])
                self.assertEqual(metadata["provider"], entry["provider"])
                self.assertEqual(metadata["usage_kind"], entry["kind"])

    def test_arbitrary_shell_success_is_unknown_and_tap_failures_are_not_pass(self):
        module = self.module()
        self.assertEqual(
            module.parse_test_result("echo success", "done", 0)["status"], "unknown"
        )
        parsed = module.parse_test_result(
            "node --test", "TAP version 13\n1..2\nok 1 - good\nnot ok 2 - bad\n", 0
        )
        self.assertEqual(parsed["status"], "failed")
        self.assertEqual(parsed["failed"], 1)

    def test_run_completeness_boolean_projects_to_nullable_uint(self):
        module = self.module()
        for value in [True, False, None]:
            rows = module.project(
                {"kind": "run", "payload": {"capture_complete": value}}
            )
            module.validate_projections(rows)
            self.assertEqual(
                rows["agent_runs"][0]["capture_complete"],
                None if value is None else int(value),
            )

    def test_metric_tables_keep_ninety_days_without_long_term_tool_content(self):
        module = self.module()
        sql = module.schema_sql()
        for table in ["agent_tool_calls", "agent_session_stats"]:
            statement = next(
                s for s in sql if s.startswith(f"CREATE TABLE IF NOT EXISTS {table} (")
            )
            self.assertIn("INTERVAL 90 DAY", statement)
        rows = module.project(
            {
                "kind": "rpc",
                "payload": {
                    "type": "tool_execution_start",
                    "toolName": "bash",
                    "toolCallId": "t",
                    "parentToolCallId": "parent",
                    "args": {"command": "private source content"},
                },
            }
        )
        self.assertIn("private source content", rows["agent_events"][0]["payload_json"])
        self.assertNotIn(
            "private source content", rows["agent_tool_calls"][0]["payload_json"]
        )
        self.assertIn("parent", rows["agent_tool_calls"][0]["payload_json"])
        rows = module.project(
            {
                "kind": "session_entry",
                "payload": {
                    "command": "pytest private source content",
                    "entry": {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolName": "bash",
                            "toolCallId": "t",
                            "content": [],
                        },
                    },
                },
            }
        )
        self.assertNotIn(
            "private source content", rows["agent_test_results"][0]["payload_json"]
        )


if __name__ == "__main__":
    unittest.main()


class WorkerDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_consumer_failure_stops_the_other_before_producer_closes(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from src.agent_analytics import AgentAnalyticsWorker

        other_started = asyncio.Event()
        calls = []

        class Producer:
            async def start(self):
                pass

            async def stop(self):
                calls.append("producer_stopped")

        class Worker(AgentAnalyticsWorker):
            async def migrate(self):
                pass

            async def consume(self, topic, producer):
                if topic == "agent.content.v1":
                    await other_started.wait()
                    raise RuntimeError("insert failed")
                try:
                    other_started.set()
                    await asyncio.Event().wait()
                finally:
                    calls.append("other_consumer_stopped")

        with patch.dict(
            "sys.modules",
            {"aiokafka": SimpleNamespace(AIOKafkaProducer=lambda **_: Producer())},
        ):
            worker = Worker(None, [])
            with self.assertRaisesRegex(RuntimeError, "insert failed"):
                await worker.run()
        self.assertEqual(calls, ["other_consumer_stopped", "producer_stopped"])

    async def test_malformed_records_are_dead_lettered_before_valid_batch_commits(self):
        from src.agent_analytics import AgentAnalyticsWorker
        from types import SimpleNamespace
        import json

        calls = []
        base = {
            "schema_version": 1,
            "topic": "agent.events.v1",
            "kind": "session_entry",
            "event_id": "e",
            "run_id": "r",
            "session_id": "s",
            "version": "4",
            "payload": {
                "entry": {"id": "e", "type": "usage", "usage": {"totalTokens": 3}}
            },
        }
        invalid_version = {**base, "version": "-1"}
        invalid_entry = {**base, "payload": {"entry": "not-an-entry"}}
        overflow = {
            **base,
            "payload": {"entry": {"type": "usage", "usage": {"totalTokens": 2**64}}},
        }
        messages = [
            SimpleNamespace(offset=i, value=json.dumps(value).encode())
            for i, value in enumerate(
                [
                    invalid_version,
                    invalid_entry,
                    overflow,
                    {**base, "kind": "feedback", "payload": {}},
                    base,
                ]
            )
        ]

        class Client:
            def insert(self, table, rows, column_names):
                calls.append(("insert", table, len(rows)))

        worker = AgentAnalyticsWorker(Client(), ["broker"])

        class Consumer:
            def __init__(self, *args, **kwargs):
                pass

            async def start(self):
                pass

            async def stop(self):
                pass

            async def getmany(self, **kwargs):
                worker.stop.set()
                return {SimpleNamespaceKey: messages}

            async def commit(self, offsets):
                calls.append(("commit", offsets))

        class Partition:
            partition = 0

        SimpleNamespaceKey = Partition()
        worker.pending_since[("agent.events.v1", SimpleNamespaceKey, 4)] = 0

        class Producer:
            async def send_and_wait(self, topic, value, key):
                calls.append(("dlq", json.loads(value)["offset"]))

        worker.consumer_factory = Consumer
        await worker.consume("agent.events.v1", Producer())
        self.assertEqual(calls[:4], [("dlq", 0), ("dlq", 1), ("dlq", 2), ("dlq", 3)])
        self.assertEqual(
            calls[4:6], [("insert", "agent_events", 1), ("insert", "agent_usage", 1)]
        )
        self.assertEqual(calls[-1], ("commit", {SimpleNamespaceKey: 5}))
        self.assertEqual(worker.pending_since, {})

    async def test_insert_failure_does_not_commit_kafka_offset(self):
        from src.agent_analytics import AgentAnalyticsWorker
        from types import SimpleNamespace

        calls = []

        class Client:
            def insert(self, table, rows, column_names):
                calls.append(("insert", table))
                if table == "agent_usage":
                    raise RuntimeError("ClickHouse unavailable")

        worker = AgentAnalyticsWorker(Client(), ["broker"])
        message = SimpleNamespace(
            offset=7,
            value=b'{"schema_version":1,"topic":"agent.events.v1","kind":"session_entry","event_id":"e","run_id":"r","session_id":"s","payload":{"entry":{"id":"e","type":"usage","usage":{"totalTokens":3}}}}',
        )

        class Consumer:
            def __init__(self, *args, **kwargs):
                self.options = kwargs

            async def start(self):
                pass

            async def stop(self):
                calls.append(("stop",))

            async def getmany(self, **kwargs):
                worker.stop.set()
                return {"partition": [message]}

            async def commit(self, offsets):
                calls.append(("commit", offsets))

        worker.consumer_factory = Consumer
        with self.assertRaisesRegex(RuntimeError, "ClickHouse unavailable"):
            await worker.consume("agent.events.v1", None)
        self.assertFalse(any(call[0] == "commit" for call in calls))
        self.assertEqual(calls[-1], ("stop",))

    async def test_commit_follows_all_successful_table_inserts(self):
        from src.agent_analytics import AgentAnalyticsWorker
        from types import SimpleNamespace

        calls = []

        class Client:
            def insert(self, table, rows, column_names):
                calls.append(("insert", table))

        worker = AgentAnalyticsWorker(Client(), ["broker"])
        message = SimpleNamespace(
            offset=7,
            value=b'{"schema_version":1,"topic":"agent.events.v1","kind":"session_entry","event_id":"e","run_id":"r","session_id":"s","payload":{"entry":{"id":"e","type":"usage","usage":{"totalTokens":3}}}}',
        )

        class Consumer:
            def __init__(self, *args, **kwargs):
                pass

            async def start(self):
                pass

            async def stop(self):
                pass

            async def getmany(self, **kwargs):
                worker.stop.set()
                return {"partition": [message]}

            async def commit(self, offsets):
                calls.append(("commit", offsets))

        worker.consumer_factory = Consumer
        await worker.consume("agent.events.v1", None)
        self.assertEqual(
            calls,
            [
                ("insert", "agent_events"),
                ("insert", "agent_usage"),
                ("commit", {"partition": 8}),
            ],
        )

    async def test_full_content_reassembly_checks_checksum(self):
        from src.agent_analytics import AgentAnalyticsWorker
        from types import SimpleNamespace
        import hashlib, base64, json

        content = json.dumps({"text": "中文", "future": 4}, ensure_ascii=False).encode()
        checksum = hashlib.sha256(content).hexdigest()

        class Client:
            def query(self, *args, **kwargs):
                return SimpleNamespace(
                    result_rows=[
                        (0, 2, checksum, base64.b64encode(content[:5]).decode()),
                        (1, 2, checksum, base64.b64encode(content[5:]).decode()),
                    ]
                )

        worker = AgentAnalyticsWorker(Client(), [])
        self.assertEqual(
            await worker.resolve({"content_ref": "id"}), {"text": "中文", "future": 4}
        )


class ExistingTestParserTests(unittest.TestCase):
    def test_complete_archive_result_is_not_overwritten_by_later_recovered_entry(self):
        from src.agent_analytics import project

        entry = project(
            {
                "run_id": "r",
                "kind": "session_entry",
                "version": "200",
                "payload": {
                    "entry": {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolName": "bash",
                            "toolCallId": "t",
                            "content": [],
                        },
                    },
                },
            }
        )["agent_test_results"][0]
        archive = project(
            {
                "run_id": "r",
                "kind": "test_result",
                "version": "100",
                "payload": {
                    "tool_call_id": "t",
                    "source_manifest": "manifest",
                    "result": {
                        "status": "failed",
                        "parser": "node-tap",
                        "passed": 0,
                        "failed": 1,
                    },
                },
            }
        )["agent_test_results"][0]
        self.assertEqual(archive["event_id"], entry["event_id"])
        self.assertGreater(archive["version"], entry["version"])

    def test_pytest_quiet_output_and_junit_skip_are_parsed(self):
        from src.agent_analytics import parse_test_result

        self.assertEqual(
            parse_test_result("python -m pytest -q", "....\n4 passed in 0.20s\n", 0)[
                "status"
            ],
            "passed",
        )
        self.assertEqual(
            parse_test_result(
                "cat results.xml",
                "<testsuite><testcase><skipped/></testcase></testsuite>",
                0,
            )["status"],
            "unknown",
        )
