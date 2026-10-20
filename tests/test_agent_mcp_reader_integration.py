"""Opt-in check that the ClickHouse MCP reader is confined to the agent_mcp views.

Requires the observability ClickHouse with the agent MCP overlay
(agent-mcp-users.xml mounted and agent-mcp-views.sql applied).
"""

import os
import unittest


@unittest.skipUnless(
    os.environ.get("AGENT_OBSERVABILITY_INTEGRATION") == "1",
    "requires explicit disposable Kafka/ClickHouse integration environment",
)
class AgentMcpReaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        import clickhouse_connect

        connection = {
            "host": os.environ["AGENT_TEST_CLICKHOUSE_HOST"],
            "port": int(os.environ.get("AGENT_TEST_CLICKHOUSE_PORT", "8123")),
            "autogenerate_session_id": False,
        }
        self.admin = clickhouse_connect.get_client(
            **connection,
            username=os.environ["AGENT_TEST_CLICKHOUSE_USER"],
            password=os.environ["AGENT_TEST_CLICKHOUSE_PASSWORD"],
        )
        self.addCleanup(self.admin.close)
        self.reader = clickhouse_connect.get_client(
            **connection,
            username="agent_mcp_reader",
            password=os.environ["AGENT_MCP_CLICKHOUSE_PASSWORD"],
            database="agent_mcp",
        )
        self.addCleanup(self.reader.close)

    def test_reader_cannot_select_raw_analytics_tables(self):
        from clickhouse_connect.driver.exceptions import DatabaseError

        # The table exists, so a failure below is a privilege failure, not a missing table.
        self.assertEqual(
            self.admin.command("EXISTS TABLE default.agent_runs"),
            1,
            "default.agent_runs must exist for this check to mean anything",
        )
        for query in (
            "SELECT * FROM default.agent_runs",
            "SELECT * FROM agent_mcp.run_metrics_hourly, default.agent_runs",
        ):
            with self.subTest(query=query):
                with self.assertRaisesRegex(
                    DatabaseError, r"Code: 497|ACCESS_DENIED|Not enough privileges"
                ):
                    self.reader.query(query)

    def test_reader_can_select_aggregate_views(self):
        for view in (
            "run_metrics_hourly",
            "model_metrics_daily",
            "tool_metrics_hourly",
            "test_metrics_hourly",
            "trace_metrics_hourly",
        ):
            with self.subTest(view=view):
                result = self.reader.query(f"SELECT count() FROM agent_mcp.{view}")
                self.assertEqual(len(result.result_rows), 1)


if __name__ == "__main__":
    unittest.main()
