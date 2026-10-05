"""Pi analytics projections and standalone batched Kafka → ClickHouse worker.

Raw payloads are lossless JSON after runtime credential redaction. Usage is projected
only from unique official session entries, never streaming RPC or session totals.
"""
import asyncio
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time
import re
import xml.etree.ElementTree as ET


class ContentPendingError(LookupError):
    """A referenced content stream has not arrived yet."""


TOKEN_FIELDS = {
    "input": "input",
    "output": "output",
    "cache_read": "cacheRead",
    "cache_write": "cacheWrite",
    "cache_write_1h": "cacheWrite1h",
    "reasoning": "reasoning",
    "total_tokens": "totalTokens",
}
COST_FIELDS = {
    "cost_input": "input",
    "cost_output": "output",
    "cost_cache_read": "cacheRead",
    "cost_cache_write": "cacheWrite",
    "cost_total": "total",
}
BASE = {
    "sequence": "UInt64",
    "schema_version": "UInt16",
    "pi_version": "String",
    "trace_id": "Nullable(String)",
    "source_time": "Nullable(DateTime64(6, 'UTC'))",
    "event_id": "String",
    "run_id": "String",
    "session_id": "String",
    "kind": "String",
    "collected_at": "DateTime64(6, 'UTC')",
    "version": "UInt64",
    "payload_json": "String CODEC(ZSTD(3))",
}
TABLES = {
    "agent_events": {},
    "agent_runs": {
        "status": "String",
        "model": "String",
        "reasoning_effort": "String",
        "routing_json": "String",
        "capture_complete": "Nullable(UInt8)",
    },
    "agent_usage": {
        "pi_session_id": "String",
        "entry_id": "String",
        "parent_id": "Nullable(String)",
        "source": "String",
        **{k: "Nullable(UInt64)" for k in TOKEN_FIELDS},
        **{k: "Nullable(Float64)" for k in COST_FIELDS},
    },
    "agent_tool_calls": {
        "tool_call_id": "String",
        "tool": "String",
        "phase": "String",
        "is_error": "Nullable(UInt8)",
    },
    "agent_session_stats": {"phase": "String", "pi_session_id": "String"},
    "agent_content_chunks": {
        "content_id": "String",
        "chunk_index": "UInt32",
        "chunk_count": "UInt32",
        "sha256": "String",
        "data": "String CODEC(ZSTD(3))",
    },
    "agent_test_results": {
        "tool_call_id": "String",
        "status": "String",
        "parser": "String",
        "passed": "Nullable(UInt64)",
        "failed": "Nullable(UInt64)",
    },
    "agent_feedback": {
        "user_id": "String",
        "verdict": "String",
        "feedback_id": "String",
    },
}


def schema_sql():
    statements = []
    for table, extra in TABLES.items():
        columns = ",\n  ".join(
            f"{key} {kind}" for key, kind in {**BASE, **extra}.items()
        )
        key = "run_id" if table == "agent_runs" else "event_id"
        days = (
            90
            if table
            in {
                "agent_runs",
                "agent_usage",
                "agent_tool_calls",
                "agent_session_stats",
                "agent_test_results",
                "agent_feedback",
            }
            else 30
        )
        statements.append(
            f"CREATE TABLE IF NOT EXISTS {table} (\n  {columns}\n) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY {key} TTL collected_at + INTERVAL {days} DAY"
        )
        statements.append(
            f"CREATE VIEW IF NOT EXISTS {table}_latest AS SELECT * FROM {table} FINAL"
        )
    statements.append(
        "CREATE VIEW IF NOT EXISTS agent_feedback_current AS SELECT run_id, user_id, argMax(verdict, version) AS verdict FROM agent_feedback_latest GROUP BY run_id, user_id"
    )
    statements.append(
        """CREATE VIEW IF NOT EXISTS agent_usage_reconciliation AS
SELECT r.run_id, r.capture_complete, s.start_snapshots, s.end_snapshots,
  if(s.start_snapshots>0 AND s.end_snapshots>0, s.end_tokens-s.start_tokens, NULL) AS session_token_delta,
  u.ledger_tokens, u.ledger_cost,
  if(s.start_snapshots>0 AND s.end_snapshots>0, s.end_cost-s.start_cost, NULL) AS session_cost_delta,
  multiIf(r.capture_complete IS NULL OR r.capture_complete=0 OR s.start_snapshots=0 OR s.end_snapshots=0 OR u.reported_costs<u.ledger_rows OR u.reported_tokens<u.ledger_rows, 'unknown',
    abs((s.end_cost-s.start_cost)-u.ledger_cost)<0.00000001 AND s.end_tokens-s.start_tokens=u.ledger_tokens, 'matched', 'mismatch') AS reconciliation
FROM agent_runs_latest r LEFT JOIN
  (SELECT run_id, countIf(phase='start') AS start_snapshots, countIf(phase='end') AS end_snapshots,
    argMaxIf(JSONExtractInt(payload_json,'stats','tokens','total'),version,phase='start') AS start_tokens,
    argMaxIf(JSONExtractInt(payload_json,'stats','tokens','total'),version,phase='end') AS end_tokens,
    argMaxIf(JSONExtractFloat(payload_json,'stats','cost'),version,phase='start') AS start_cost,
    argMaxIf(JSONExtractFloat(payload_json,'stats','cost'),version,phase='end') AS end_cost
  FROM agent_session_stats_latest GROUP BY run_id) s ON r.run_id=s.run_id LEFT JOIN
  (SELECT run_id, count() AS ledger_rows, count(total_tokens) AS reported_tokens, count(cost_total) AS reported_costs, sum(total_tokens) AS ledger_tokens, sum(cost_total) AS ledger_cost
   FROM agent_usage_latest GROUP BY run_id) u ON r.run_id=u.run_id"""
    )
    return statements


def _number(value):
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def validate_projections(projections):
    """Reject malformed rows before a single poisoned value reaches a batched insert."""
    for table, rows in projections.items():
        for row in rows:
            for column, declaration in {**BASE, **TABLES[table]}.items():
                value = row[column]
                kind = declaration.split(" CODEC", 1)[0]
                if kind.startswith("Nullable("):
                    if value is None:
                        continue
                    kind = kind[len("Nullable(") : -1]
                if kind.startswith("UInt"):
                    valid = (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and 0 <= value < 2 ** int(kind[4:])
                    )
                elif kind == "Float64":
                    valid = _number(value) is not None
                elif kind == "String":
                    valid = isinstance(value, str)
                elif kind.startswith("DateTime64"):
                    valid = isinstance(value, datetime) and value.tzinfo is not None
                else:
                    raise ValueError("unsupported_projection_type")
                if not valid:
                    raise ValueError(f"invalid_projection:{table}.{column}")


def project(envelope, resolved_payload=None):
    payload = (
        envelope.get("payload", {}) if resolved_payload is None else resolved_payload
    )
    collected = datetime.fromisoformat(
        envelope.get("collected_at", datetime.now(timezone.utc).isoformat()).replace(
            "Z", "+00:00"
        )
    ).astimezone(timezone.utc)
    base = {
        key: str(envelope.get(key, ""))
        for key in ["event_id", "run_id", "session_id", "kind"]
    }
    source_time = envelope.get("source_time")
    base.update(
        sequence=envelope.get("sequence", 0),
        schema_version=envelope.get("schema_version", 1),
        pi_version=envelope.get("pi_version", "unknown"),
        trace_id=envelope.get("trace_id"),
        source_time=datetime.fromisoformat(source_time.replace("Z", "+00:00"))
        if source_time
        else None,
        collected_at=collected,
        version=int(envelope.get("version", int(collected.timestamp() * 1_000_000))),
        payload_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    rows = {"agent_events": [base.copy()]}
    kind = envelope.get("kind")
    if kind == "content":
        rows = {
            "agent_content_chunks": [
                {
                    **base,
                    **{key: payload[key] for key in TABLES["agent_content_chunks"]},
                }
            ]
        }
    elif kind == "session_entry":
        entry = payload.get("entry", {})
        message = entry.get("message", {})
        usage = (
            entry.get("usage")
            if entry.get("type") in {"usage", "compaction", "branch_summary"}
            else message.get("usage")
            if entry.get("type") == "message"
            and message.get("role") in {"assistant", "toolResult"}
            else None
        )
        if isinstance(usage, dict):
            costs = usage.get("cost") or {}
            rows["agent_usage"] = [
                {
                    **base,
                    "pi_session_id": payload.get("pi_session_id", "unknown"),
                    "entry_id": entry.get("id", ""),
                    "parent_id": entry.get("parentId"),
                    "source": message.get("role", entry.get("type", "unknown")),
                    "payload_json": json.dumps(
                        {
                            "usage": usage,
                            "entry_type": entry.get("type"),
                            "role": message.get("role"),
                            "model": message.get("model", entry.get("model")),
                            "provider": message.get("provider", entry.get("provider")),
                            "usage_kind": entry.get("kind"),
                            "stopReason": message.get("stopReason"),
                        },
                        ensure_ascii=False,
                    ),
                    **{
                        key: (
                            _number(usage.get(source))
                            if isinstance(usage.get(source), int)
                            else None
                        )
                        for key, source in TOKEN_FIELDS.items()
                    },
                    **{
                        key: _number(costs.get(source))
                        for key, source in COST_FIELDS.items()
                    },
                }
            ]
        if entry.get("type") == "message" and message.get("role") == "toolResult":
            text = "\n".join(
                block.get("text", "")
                for block in message.get("content", [])
                if block.get("type") == "text"
            )
            command = payload.get("command", "")
            result = parse_test_result(command, text, None)
            if result["parser"] != "unknown" or message.get("toolName") == "bash":
                rows["agent_test_results"] = [
                    {
                        **base,
                        "event_id": f"{base['run_id']}:test:{message.get('toolCallId','unknown')}",
                        "tool_call_id": message.get("toolCallId", ""),
                        **{
                            key: result[key]
                            for key in ["status", "parser", "passed", "failed"]
                        },
                        "payload_json": json.dumps(
                            {
                                "result": result,
                                "command_sha256": hashlib.sha256(
                                    command.encode()
                                ).hexdigest(),
                                "source_entry_id": entry.get("id"),
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]
    elif kind == "rpc" and payload.get("type") in {
        "tool_execution_start",
        "tool_execution_end",
    }:
        rows["agent_tool_calls"] = [
            {
                **base,
                "tool_call_id": payload.get("toolCallId", ""),
                "tool": payload.get("toolName", ""),
                "phase": "start" if payload["type"].endswith("start") else "end",
                "is_error": int(payload["isError"]) if "isError" in payload else None,
                "payload_json": json.dumps(
                    {
                        "type": payload["type"],
                        "tool_call_id": payload.get("toolCallId"),
                        "parent_tool_call_id": payload.get("parentToolCallId"),
                        "tool": payload.get("toolName"),
                        "is_error": payload.get("isError"),
                    }
                ),
            }
        ]
    elif kind == "session_stats":
        rows["agent_session_stats"] = [
            {
                **base,
                "phase": payload.get("phase", "unknown"),
                "pi_session_id": payload.get("pi_session_id", "unknown"),
            }
        ]
    elif kind == "run":
        rows["agent_runs"] = [
            {
                **base,
                "status": payload.get("status", "unknown"),
                "model": payload.get("model", "unknown"),
                "reasoning_effort": payload.get("reasoning_effort", "unknown"),
                "routing_json": json.dumps(payload.get("routing", {})),
                "capture_complete": int(payload["capture_complete"])
                if isinstance(payload.get("capture_complete"), bool)
                else payload.get("capture_complete"),
                "payload_json": json.dumps(
                    {k: v for k, v in payload.items() if k not in {"answer", "events"}},
                    ensure_ascii=False,
                ),
            }
        ]
    elif kind == "test_result":
        result = payload["result"]
        rows["agent_test_results"] = [
            {
                **base,
                "event_id": f"{base['run_id']}:test:{payload['tool_call_id']}",
                # A complete archive outranks a later replay of the truncated toolResult.
                # Runtime versions are microsecond timestamps, well below the UInt64 high bit.
                "version": (1 << 63) + base["version"],
                "tool_call_id": payload["tool_call_id"],
                **{
                    key: result[key] for key in ["status", "parser", "passed", "failed"]
                },
                "payload_json": json.dumps(
                    {
                        **{k: v for k, v in payload.items() if k != "command"},
                        "source_version": base["version"],
                        "command_sha256": hashlib.sha256(
                            payload.get("command", "").encode()
                        ).hexdigest(),
                    }
                ),
            }
        ]
    elif kind == "feedback":
        rows["agent_feedback"] = [
            {**base, **{key: payload[key] for key in TABLES["agent_feedback"]}}
        ]
    return rows


def parse_test_result(command, output, exit_code):
    result = {
        "status": "unknown",
        "parser": "unknown",
        "parser_version": 1,
        "passed": None,
        "failed": None,
        "exit_code": exit_code,
    }
    # Parse only recognizable complete summaries. An arbitrary exit code is not a test verdict.
    if "TAP version 13" in output or re.search(r"^# tests \d+", output, re.M):
        plan = re.search(r"^1\.\.(\d+)\s*$", output, re.M)
        passed = len(re.findall(r"^ok \d+\b", output, re.M))
        failed = len(re.findall(r"^not ok \d+\b", output, re.M))
        summary = re.search(r"^# pass (\d+)\n# fail (\d+)", output, re.M)
        if summary:
            passed, failed = map(int, summary.groups())
        result.update(parser="node-tap", passed=passed, failed=failed)
        complete = bool(summary or (plan and passed + failed == int(plan[1])))
        result["status"] = (
            "failed" if failed else "passed" if complete and passed > 0 else "unknown"
        )
    elif re.search(r"\bpytest\b", command):
        summary = re.search(
            r"(?:^|\n)(?:=+\s*)?((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?),?\s*)+)in\s+[\d.]+s",
            output,
        )
        if summary:
            counts = dict(
                (label, int(count))
                for count, label in re.findall(
                    r"(\d+) (passed|failed|error|errors)", summary[1]
                )
            )
            passed = counts.get("passed", 0)
            failed = (
                counts.get("failed", 0)
                + counts.get("error", 0)
                + counts.get("errors", 0)
            )
            result.update(
                parser="pytest",
                passed=passed,
                failed=failed,
                status="failed" if failed else "passed" if passed else "unknown",
            )
    elif output.lstrip().startswith(("<?xml", "<testsuite", "<testsuites")):
        try:
            root = ET.fromstring(output)
            cases = list(root.iter("testcase"))
            failed = sum(
                bool(case.find("failure") is not None or case.find("error") is not None)
                for case in cases
            )
            skipped = sum(case.find("skipped") is not None for case in cases)
            result.update(
                parser="junit",
                passed=len(cases) - failed - skipped,
                failed=failed,
                status="failed"
                if failed
                else "passed"
                if len(cases) > skipped
                else "unknown",
            )
        except ET.ParseError:
            pass
    return result


class AgentAnalyticsWorker:
    """Separate consumers allow chunk arrival to unblock a referenced event partition."""

    def __init__(self, client, brokers, consumer_factory=None, producer_factory=None):
        self.client = client
        self.brokers = brokers
        self.consumer_factory = consumer_factory
        self.producer_factory = producer_factory
        self.stop = asyncio.Event()
        self.pending_since = {}

    async def migrate(self):
        for statement in schema_sql():
            await asyncio.to_thread(self.client.command, statement)

    async def resolve(self, payload):
        if not isinstance(payload, dict) or "content_ref" not in payload:
            return payload
        result = await asyncio.to_thread(
            self.client.query,
            "SELECT chunk_index, chunk_count, sha256, data FROM agent_content_chunks FINAL WHERE content_id={id:String} ORDER BY chunk_index",
            parameters={"id": payload["content_ref"]},
        )
        rows = result.result_rows
        if (
            not rows
            or len(rows) != rows[0][1]
            or [row[0] for row in rows] != list(range(len(rows)))
        ):
            raise ContentPendingError("content_pending")
        content = b"".join(base64.b64decode(row[3], validate=True) for row in rows)
        if hashlib.sha256(content).hexdigest() != rows[0][2]:
            raise ValueError("content_checksum_mismatch")
        return json.loads(content)

    def _read_bash_test_result(self, envelope, payload):
        tail = b""
        checksum = hashlib.sha256()
        expected_index = 0
        byte_count = 0
        with self.client.query_rows_stream(
            "SELECT payload_json FROM agent_events FINAL WHERE run_id={run:String} AND kind='bash_output_chunk' "
            "AND JSONExtractString(payload_json,'tool_call_id')={tool:String} AND JSONExtractString(payload_json,'path')={path:String} "
            "ORDER BY JSONExtractUInt(payload_json,'chunk_index')",
            parameters={
                "run": envelope["run_id"],
                "tool": payload["tool_call_id"],
                "path": payload["path"],
            },
            settings={"max_block_size": 64},
        ) as stream:
            for row in stream:
                chunk = json.loads(row[0])
                if chunk["chunk_index"] != expected_index:
                    raise ContentPendingError("bash_chunks_pending")
                data = base64.b64decode(chunk["data"], validate=True)
                checksum.update(data)
                byte_count += len(data)
                expected_index += 1
                tail = (tail + data)[-1048576:]
        if expected_index != payload["chunks"]:
            raise ContentPendingError("bash_chunks_pending")
        if byte_count != payload["bytes"] or checksum.hexdigest() != payload["sha256"]:
            raise ValueError("bash_checksum_mismatch")
        parsed = parse_test_result(
            payload.get("command", ""), tail.decode("utf8", errors="replace"), None
        )
        parsed["parse_scope"] = "full" if byte_count <= 1048576 else "last_1MiB"
        return {
            **envelope,
            "event_id": envelope["event_id"] + ":test",
            "kind": "test_result",
            "payload": {
                "tool_call_id": payload["tool_call_id"],
                "command": payload.get("command", ""),
                "source_manifest": envelope["event_id"],
                "result": parsed,
            },
        }

    async def full_test_result(self, envelope, payload):
        return await asyncio.to_thread(self._read_bash_test_result, envelope, payload)

    async def consume(self, topic, producer):
        if self.consumer_factory is None:
            from aiokafka import AIOKafkaConsumer
        else:
            AIOKafkaConsumer = self.consumer_factory

        consumer = (self.consumer_factory or AIOKafkaConsumer)(
            topic,
            bootstrap_servers=self.brokers,
            group_id="pi-agent-analytics-v1",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            max_poll_interval_ms=600000,
        )
        await consumer.start()
        try:
            while not self.stop.is_set():
                batches = await consumer.getmany(timeout_ms=500, max_records=200)
                for partition, messages in batches.items():
                    tables = defaultdict(list)
                    last_offset = None
                    for message in messages:
                        try:
                            envelope = json.loads(message.value)
                            if not isinstance(envelope, dict) or not isinstance(
                                envelope.get("payload"), dict
                            ):
                                raise ValueError("invalid_envelope")
                            if (
                                envelope.get("schema_version") != 1
                                or envelope.get("topic") != topic
                            ):
                                raise ValueError("unsupported_envelope")
                            if not all(
                                isinstance(envelope.get(k), str) and envelope[k]
                                for k in ["event_id", "run_id", "session_id"]
                            ):
                                raise ValueError("missing_event_identity")
                            payload = await self.resolve(envelope.get("payload"))
                            if not isinstance(payload, dict):
                                raise ValueError("invalid_resolved_payload")
                            projections = project(envelope, payload)
                            if envelope.get("kind") == "bash_output_manifest":
                                test_result = await self.full_test_result(
                                    envelope, payload
                                )
                                for table, rows in project(test_result).items():
                                    projections.setdefault(table, []).extend(rows)
                            validate_projections(projections)
                            for table, rows in projections.items():
                                tables[table].extend(rows)
                        except ContentPendingError:
                            identity = (topic, partition, message.offset)
                            first_seen = self.pending_since.setdefault(
                                identity, time.monotonic()
                            )
                            if time.monotonic() - first_seen < 300:
                                break  # persist prefix; retry this offset after chunk consumer inserts it
                            await producer.send_and_wait(
                                "agent.dlq.v1",
                                json.dumps(
                                    {
                                        "topic": topic,
                                        "partition": partition.partition,
                                        "offset": message.offset,
                                        "error": "capture_incomplete_missing_chunks",
                                        "sha256": hashlib.sha256(
                                            message.value
                                        ).hexdigest(),
                                    }
                                ).encode(),
                                key=topic.encode(),
                            )
                            self.pending_since.pop(identity, None)
                        except (
                            ValueError,
                            KeyError,
                            TypeError,
                            OverflowError,
                            AttributeError,
                        ) as error:
                            await producer.send_and_wait(
                                "agent.dlq.v1",
                                json.dumps(
                                    {
                                        "topic": topic,
                                        "partition": partition.partition,
                                        "offset": message.offset,
                                        "error": type(error).__name__,
                                        "sha256": hashlib.sha256(
                                            message.value
                                        ).hexdigest(),
                                    }
                                ).encode(),
                                key=topic.encode(),
                            )
                        self.pending_since.pop((topic, partition, message.offset), None)
                        last_offset = message.offset
                    for table, rows in tables.items():
                        columns = list(rows[0])
                        await asyncio.to_thread(
                            self.client.insert,
                            table,
                            [[row[c] for c in columns] for row in rows],
                            column_names=columns,
                        )
                    if last_offset is not None:
                        await consumer.commit({partition: last_offset + 1})
                    # getmany advanced the position beyond a missing chunk: explicitly seek back.
                    if messages and last_offset != messages[-1].offset:
                        consumer.seek(
                            partition,
                            messages[0].offset
                            if last_offset is None
                            else last_offset + 1,
                        )
                        await asyncio.sleep(0.25)
        finally:
            await consumer.stop()

    async def run(self):
        from aiokafka import AIOKafkaProducer

        producer = (self.producer_factory or AIOKafkaProducer)(
            bootstrap_servers=self.brokers, acks="all", enable_idempotence=True
        )
        await self.migrate()
        await producer.start()
        tasks = [
            asyncio.create_task(self.consume(topic, producer))
            for topic in ["agent.content.v1", "agent.events.v1"]
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await producer.stop()


async def main(migrate_only=False):
    import clickhouse_connect
    import signal

    client = clickhouse_connect.get_client(
        host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
        port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        autogenerate_session_id=False,
        username=os.environ.get("CLICKHOUSE_USER", "llm_router"),
        password=os.environ["CLICKHOUSE_PASSWORD"],
        database=os.environ.get("CLICKHOUSE_DATABASE", "default"),
    )
    worker = AgentAnalyticsWorker(
        client, os.environ.get("AGENT_KAFKA_BROKERS", "localhost:9092").split(",")
    )
    for signum in [signal.SIGTERM, signal.SIGINT]:
        asyncio.get_running_loop().add_signal_handler(signum, worker.stop.set)
    try:
        if migrate_only:
            await worker.migrate()
        else:
            await worker.run()
    finally:
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print idempotent schema SQL without opening connections",
    )
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        help="Create tables and views without consuming Kafka",
    )
    args = parser.parse_args()
    if args.dry_run:
        print(";\n\n".join(schema_sql()) + ";")
    else:
        asyncio.run(main(args.migrate_only))
