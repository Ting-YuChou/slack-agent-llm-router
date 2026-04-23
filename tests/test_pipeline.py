import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.structs import TopicPartition

from src.llm_router_part3_pipeline import (
    AlertEventEntry,
    ClickHouseManager,
    KafkaConsumerManager,
    KafkaIngestionPipeline,
    KafkaProducerManager,
    MetricEntry,
    ModelPerformanceEntry,
    QueryLogEntry,
)
from src.utils.schema import InferenceResponse, QueryRequest, UserTier


@pytest.fixture
def pipeline_config():
    return {
        "bootstrap_servers": ["localhost:9092"],
        "topics": {
            "queries": "test-queries",
            "responses": "test-responses",
            "metrics": "test-metrics",
            "errors": "test-errors",
            "requests_raw": "requests.raw",
            "inference_completed": "inference.completed",
            "requests_enriched": "requests.enriched",
            "fast_lane_hints": "fast_lane_hints",
            "analytics_model_metrics_1m": "analytics.model_metrics_1m",
            "routing_guardrails": "routing.guardrails",
            "routing_policy_state": "routing.policy_state",
            "alerts": "alerts",
        },
        "consumer": {
            "group_id": "test-group",
            "auto_offset_reset": "latest",
        },
        "clickhouse": {
            "host": "localhost",
            "port": 8123,
            "database": "test_db",
            "username": "default",
            "password": "",
        },
    }


@pytest.fixture
def sample_query_log():
    return QueryLogEntry(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        user_id="user-1",
        user_tier=UserTier.PREMIUM,
        query_text="Analyze this data",
        query_type="analysis",
        selected_model="gpt-5",
        token_count_input=100,
        token_count_output=50,
        latency_ms=850,
        cost_usd=0.02,
        status="success",
    )


class AsyncMessageConsumer:
    def __init__(self, messages):
        self._messages = list(messages)
        self.commit = AsyncMock()

    def __aiter__(self):
        async def iterator():
            for message in self._messages:
                yield message

        return iterator()


class TestQueryLogEntry:
    def test_to_dict_serializes_enums(self, sample_query_log):
        payload = sample_query_log.to_dict()

        assert payload["user_tier"] == "premium"
        assert payload["query_type"] == "analysis"
        assert payload["query_text"] == "Analyze this data"


class TestKafkaProducerManager:
    @pytest.mark.asyncio
    async def test_initialize_creates_producer(self, pipeline_config):
        with patch(
            "src.llm_router_part3_pipeline.AIOKafkaProducer"
        ) as mock_producer_cls:
            producer = AsyncMock()
            mock_producer_cls.return_value = producer

            manager = KafkaProducerManager(pipeline_config)
            await manager.initialize()

            producer.start.assert_awaited_once()
            assert manager.producer is producer

    @pytest.mark.asyncio
    async def test_initialize_maps_reliability_producer_config(self, pipeline_config):
        config = {
            **pipeline_config,
            "producer": {
                "acks": "1",
                "enable_idempotence": True,
                "linger_ms": 25,
                "batch_size": 32768,
                "request_timeout_ms": 12000,
                "retry_backoff_ms": 250,
            },
        }
        with patch(
            "src.llm_router_part3_pipeline.AIOKafkaProducer"
        ) as mock_producer_cls:
            producer = AsyncMock()
            mock_producer_cls.return_value = producer

            manager = KafkaProducerManager(config)
            await manager.initialize()

            kwargs = mock_producer_cls.call_args.kwargs
            assert kwargs["acks"] == "all"
            assert kwargs["enable_idempotence"] is True
            assert kwargs["linger_ms"] == 25
            assert kwargs["max_batch_size"] == 32768
            assert kwargs["request_timeout_ms"] == 12000
            assert kwargs["retry_backoff_ms"] == 250

    @pytest.mark.asyncio
    async def test_send_message_waits_for_ack_and_retries(self, pipeline_config):
        config = {
            **pipeline_config,
            "producer": {
                "retries": 1,
                "retry_backoff_ms": 0,
                "send_timeout_seconds": 1,
            },
        }
        manager = KafkaProducerManager(config)
        manager.producer = AsyncMock()
        manager.producer.send_and_wait = AsyncMock(
            side_effect=[RuntimeError("temporary"), object()]
        )

        sent = await manager._send_message(
            "queries",
            "llm-queries",
            key="u1",
            value={"hello": "world"},
        )

        assert sent is True
        assert manager.producer.send_and_wait.await_count == 2
        assert manager.consecutive_failures == 0

    def test_serialize_message_handles_datetime(self, pipeline_config):
        manager = KafkaProducerManager(pipeline_config)
        payload = {
            "timestamp": datetime(2026, 3, 12, tzinfo=timezone.utc),
            "value": 42,
        }

        encoded = manager._serialize_message(payload)
        decoded = json.loads(encoded.decode("utf-8"))

        assert decoded["value"] == 42
        assert "timestamp" in decoded

    @pytest.mark.asyncio
    async def test_produce_request_raw_emits_requests_raw_event(self, pipeline_config):
        manager = KafkaProducerManager(pipeline_config)
        manager.producer = AsyncMock()
        request = QueryRequest(query="hello", user_id="u1", user_tier=UserTier.FREE)

        await manager.produce_request_raw(request)

        send_kwargs = manager.producer.send_and_wait.await_args.kwargs
        assert send_kwargs["topic"] == "requests.raw"
        assert send_kwargs["key"] == "u1"
        assert send_kwargs["value"]["event_type"] == "requests.raw"
        assert send_kwargs["value"]["request_id"] == request.request_id
        assert send_kwargs["value"]["query_text"] == "hello"

    @pytest.mark.asyncio
    async def test_produce_inference_completed_emits_completion_event(
        self, pipeline_config
    ):
        manager = KafkaProducerManager(pipeline_config)
        manager.producer = AsyncMock()
        request = QueryRequest(
            query="hello",
            user_id="u1",
            user_tier=UserTier.FREE,
            session_id="session-1",
            conversation_id="u1:c1:main",
        )
        response = InferenceResponse(
            response_text="world",
            model_name="gpt-5",
            provider="openai",
            token_count_input=3,
            token_count_output=5,
            total_tokens=8,
            latency_ms=12,
            tokens_per_second=120.0,
            cost_usd=0.01,
        )
        routing_decision = SimpleNamespace(
            query_type=SimpleNamespace(value="analysis"),
            routing_reason="Rule-based selection",
            routing_strategy="intelligent",
            route_to_fast_lane=True,
            actual_fast_lane_hit=True,
            policy_source="session+user",
            hint_reason="fast_lane_affinity",
            fallback_models=["mistral-7b"],
        )

        await manager.produce_inference_completed(request, response, routing_decision)

        send_kwargs = manager.producer.send_and_wait.await_args.kwargs
        assert send_kwargs["topic"] == "inference.completed"
        assert send_kwargs["key"] == request.request_id
        assert send_kwargs["value"]["event_type"] == "inference.completed"
        assert send_kwargs["value"]["selected_model"] == "gpt-5"
        assert send_kwargs["value"]["query_type"] == "analysis"
        assert send_kwargs["value"]["fallback_models"] == ["mistral-7b"]
        assert send_kwargs["value"]["session_id"] == "session-1"
        assert send_kwargs["value"]["conversation_id"] == "u1:c1:main"
        assert send_kwargs["value"]["route_to_fast_lane"] is True
        assert send_kwargs["value"]["actual_fast_lane_hit"] is True
        assert send_kwargs["value"]["policy_source"] == "session+user"

    @pytest.mark.asyncio
    async def test_produce_query_log_uses_request_id_and_fallback_query_type(
        self, pipeline_config
    ):
        manager = KafkaProducerManager(pipeline_config)
        manager.producer = AsyncMock()
        request = QueryRequest(query="hello", user_id="u1", user_tier=UserTier.FREE)
        response = InferenceResponse(
            response_text="world",
            model_name="gpt-5",
            provider="openai",
            token_count_input=3,
            token_count_output=5,
            total_tokens=8,
            latency_ms=12,
            tokens_per_second=120.0,
            cost_usd=0.01,
        )

        await manager.produce_query_log(request, response, SimpleNamespace())

        send_kwargs = manager.producer.send_and_wait.await_args.kwargs
        assert send_kwargs["topic"] == "test-queries"
        assert send_kwargs["value"]["query_id"] == request.request_id
        assert send_kwargs["value"]["query_type"] == "general"


class TestClickHouseManager:
    @pytest.mark.asyncio
    async def test_insert_query_log_calls_insert(
        self, pipeline_config, sample_query_log
    ):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()
            await manager.insert_query_log(sample_query_log)

            expected_payload = sample_query_log.to_dict()
            expected_columns = list(expected_payload.keys())
            expected_rows = [[expected_payload[column] for column in expected_columns]]

            client.insert.assert_called_once_with(
                "query_logs",
                expected_rows,
                column_names=expected_columns,
            )

    @pytest.mark.asyncio
    async def test_insert_metric_calls_insert(self, pipeline_config):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()

            metric = MetricEntry(
                timestamp=datetime.now(timezone.utc),
                service="router",
                metric_name="latency_ms",
                metric_value=12.5,
                labels={"env": "test"},
            )
            await manager.insert_metric(metric)

            expected_payload = metric.to_dict()
            expected_columns = list(expected_payload.keys())
            expected_rows = [[expected_payload[column] for column in expected_columns]]

            client.insert.assert_called_once_with(
                "system_metrics",
                expected_rows,
                column_names=expected_columns,
            )

    @pytest.mark.asyncio
    async def test_batch_insert_model_performance_calls_insert(self, pipeline_config):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()

            metric = ModelPerformanceEntry(
                timestamp=datetime.now(timezone.utc),
                model_name="mistral-7b",
                provider="vllm",
                window_start_ms=1_710_000_000_000,
                window_end_ms=1_710_000_060_000,
                window_size_seconds=60,
                requests_count=6,
                success_count=5,
                success_rate=5 / 6,
                avg_latency_ms=240.0,
                avg_tokens_per_second=110.0,
                error_count=1,
                total_tokens=360,
                total_cost_usd=0.0,
                queries_per_second=0.1,
                cache_hit_rate=0.5,
                cached_count=3,
            )

            await manager.batch_insert_model_performance([metric])

            expected_payload = metric.to_dict()
            expected_columns = list(expected_payload.keys())
            expected_rows = [[expected_payload[column] for column in expected_columns]]

            client.insert.assert_called_with(
                "model_performance",
                expected_rows,
                column_names=expected_columns,
            )

    @pytest.mark.asyncio
    async def test_batch_insert_alert_events_calls_insert(self, pipeline_config):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()

            alert = AlertEventEntry(
                timestamp=datetime.now(timezone.utc),
                alert_type="anomaly_detected",
                severity="warning",
                description="Latency spiked",
                anomaly_type="high_latency",
                source_event_type="analytics.model_metrics_1m",
                model_name="mistral-7b",
                provider="vllm",
                window_start_ms=1_710_000_000_000,
                window_end_ms=1_710_000_060_000,
                payload_json='{"severity":"warning"}',
            )

            await manager.batch_insert_alert_events([alert])

            expected_payload = alert.to_dict()
            expected_columns = list(expected_payload.keys())
            expected_rows = [[expected_payload[column] for column in expected_columns]]

            client.insert.assert_called_with(
                "alert_events",
                expected_rows,
                column_names=expected_columns,
            )

    @pytest.mark.asyncio
    async def test_get_query_analytics_uses_weighted_overall_aggregation(
        self, pipeline_config
    ):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()
            manager._run_blocking = AsyncMock(
                side_effect=[
                    SimpleNamespace(result_rows=[[10, 200, 3.5, 140.0, 80.0]]),
                    SimpleNamespace(
                        result_rows=[
                            ["gpt-5", 8, 2.5],
                            ["mistral-7b", 2, 1.0],
                        ]
                    ),
                    SimpleNamespace(
                        result_rows=[
                            ["analysis", 6],
                            ["general", 4],
                        ]
                    ),
                ]
            )

            analytics = await manager.get_query_analytics(hours=6)

            assert manager._run_blocking.await_count == 3
            overall_query = manager._run_blocking.await_args_list[0].args[1]
            assert "GROUP BY selected_model, query_type" not in overall_query
            assert analytics == {
                "total_queries": 10,
                "total_tokens": 200,
                "total_cost": 3.5,
                "avg_latency": 140.0,
                "success_rate": 80.0,
                "model_breakdown": {
                    "gpt-5": {"queries": 8, "cost": 2.5},
                    "mistral-7b": {"queries": 2, "cost": 1.0},
                },
                "query_type_breakdown": {"analysis": 6, "general": 4},
            }

    @pytest.mark.asyncio
    async def test_get_model_performance_groups_by_provider(self, pipeline_config):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()
            manager._run_blocking = AsyncMock(
                return_value=SimpleNamespace(
                    result_rows=[
                        ["gpt-5", "openai", 7, 99.5, 412.0, 155.0, 0, 1.25],
                        ["gpt-5", "azure", 3, 97.0, 430.0, 150.0, 1, 0.55],
                    ]
                )
            )

            performance = await manager.get_model_performance(hours=6)

            query_arg = manager._run_blocking.await_args.args[1]
            assert "GROUP BY model_name, provider" in query_arg
            assert performance == [
                {
                    "model_name": "gpt-5",
                    "provider": "openai",
                    "requests": 7,
                    "success_rate": 99.5,
                    "avg_latency_ms": 412.0,
                    "tokens_per_second": 155.0,
                    "error_count": 0,
                    "total_cost": 1.25,
                },
                {
                    "model_name": "gpt-5",
                    "provider": "azure",
                    "requests": 3,
                    "success_rate": 97.0,
                    "avg_latency_ms": 430.0,
                    "tokens_per_second": 150.0,
                    "error_count": 1,
                    "total_cost": 0.55,
                },
            ]

    @pytest.mark.asyncio
    async def test_initialize_raises_when_required_table_creation_fails(
        self, pipeline_config
    ):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]

            def fail_model_performance_table(sql):
                if "CREATE TABLE IF NOT EXISTS test_db.model_performance" in sql:
                    raise RuntimeError("boom")
                return None

            client.command.side_effect = fail_model_performance_table
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])

            with pytest.raises(
                RuntimeError,
                match="Failed to create required ClickHouse table model_performance",
            ):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_get_routing_guardrails_uses_run_blocking(self, pipeline_config):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()
            manager._run_blocking = AsyncMock(
                return_value=SimpleNamespace(
                    result_rows=[
                        [
                            datetime(2026, 4, 19, tzinfo=timezone.utc),
                            "high_latency",
                            "warning",
                            "Latency regressed",
                            "gpt-5",
                            "openai",
                            json.dumps(
                                {
                                    "scope_type": "model",
                                    "scope_key": "gpt-5",
                                    "guardrail_action": "warn",
                                }
                            ),
                        ]
                    ]
                )
            )

            guardrails = await manager.get_routing_guardrails(hours=6)

            manager._run_blocking.assert_awaited_once()
            func_arg, query_arg = manager._run_blocking.await_args.args
            assert func_arg == client.query
            assert "FROM test_db.alert_events" in query_arg
            assert "routing.guardrails" in query_arg
            assert guardrails[0]["guardrail_action"] == "warn"
            assert guardrails[0]["source"] == "clickhouse"

    @pytest.mark.asyncio
    async def test_get_routing_policy_state_events_uses_run_blocking(
        self, pipeline_config
    ):
        with patch(
            "src.llm_router_part3_pipeline.clickhouse_connect.get_client"
        ) as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()
            manager._run_blocking = AsyncMock(
                return_value=SimpleNamespace(
                    result_rows=[
                        [
                            datetime(2026, 4, 19, tzinfo=timezone.utc),
                            "user",
                            "user-1",
                            "user-1",
                            "session-1",
                            "premium",
                            "burst_protection",
                            5,
                            0.2,
                            420.0,
                            0.75,
                            "analysis",
                            "complex",
                            True,
                            False,
                            True,
                            True,
                            False,
                            ["gpt-5"],
                            ["mistral-7b"],
                            ["vllm"],
                            json.dumps({"policy_source": "rolling_state"}),
                        ]
                    ]
                )
            )

            state_events = await manager.get_routing_policy_state_events(hours=6)

            manager._run_blocking.assert_awaited_once()
            func_arg, query_arg = manager._run_blocking.await_args.args
            assert func_arg == client.query
            assert "FROM test_db.routing_policy_state_events" in query_arg
            assert state_events[0]["preferred_models"] == ["gpt-5"]
            assert state_events[0]["payload"]["policy_source"] == "rolling_state"


class TestKafkaConsumerManager:
    def test_deserialize_message_returns_json_payload(self, pipeline_config):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        raw = json.dumps({"hello": "world"}).encode("utf-8")

        assert consumer._deserialize_message(raw) == {"hello": "world"}

    @pytest.mark.asyncio
    async def test_notify_observers_invokes_async_handler(self, pipeline_config):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        seen = []

        async def handler(payload):
            seen.append(payload["event_type"])

        consumer.register_observer("alerts", handler)
        await consumer._notify_observers("alerts", {"event_type": "alerts"})

        assert seen == ["alerts"]

    @pytest.mark.asyncio
    async def test_consume_requests_enriched_commits_offsets_after_observer_delivery(
        self, pipeline_config
    ):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        seen = []

        async def handler(payload):
            seen.append(payload["request_id"])

        message = SimpleNamespace(
            value={"event_type": "requests.enriched", "request_id": "req-1"},
            topic="requests.enriched",
            partition=0,
            offset=4,
        )
        kafka_consumer = AsyncMessageConsumer([message])
        consumer.consumers["requests_enriched"] = kafka_consumer
        consumer.register_observer("requests_enriched", handler)

        await consumer._consume_requests_enriched(kafka_consumer)

        kafka_consumer.commit.assert_awaited_once_with(
            {TopicPartition("requests.enriched", 0): 5}
        )
        assert seen == ["req-1"]

    @pytest.mark.asyncio
    async def test_consume_query_poison_message_goes_to_dlq_and_commits(
        self, pipeline_config
    ):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        consumer.dlq_producer = AsyncMock()
        message = SimpleNamespace(
            value=b"{not-json",
            key=b"bad-key",
            topic="test-queries",
            partition=0,
            offset=12,
        )
        kafka_consumer = AsyncMessageConsumer([message])
        consumer.consumers["queries"] = kafka_consumer

        await consumer._consume_queries(kafka_consumer)

        dlq_kwargs = consumer.dlq_producer.send_and_wait.await_args.kwargs
        assert dlq_kwargs["topic"] == "test-queries.dlq"
        assert dlq_kwargs["key"] == "test-queries:0:12"
        assert dlq_kwargs["value"]["source_offset"] == 12
        assert dlq_kwargs["value"]["source_key"] == "bad-key"
        assert dlq_kwargs["value"]["error_type"] == "JSONDecodeError"
        kafka_consumer.commit.assert_awaited_once_with(
            {TopicPartition("test-queries", 0): 13}
        )

    @pytest.mark.asyncio
    async def test_consume_query_records_kafka_metadata_for_idempotency(
        self, pipeline_config
    ):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        payload = {
            "query_id": "query-1",
            "timestamp": "2026-04-19T00:00:00+00:00",
            "user_id": "user-1",
            "user_tier": "premium",
            "query_text": "hello",
            "query_type": "general",
            "selected_model": "gpt-5",
            "token_count_input": 1,
            "token_count_output": 2,
            "latency_ms": 10,
            "cost_usd": 0.01,
            "status": "success",
        }
        message = SimpleNamespace(
            value=json.dumps(payload).encode("utf-8"),
            topic="test-queries",
            partition=2,
            offset=8,
        )
        kafka_consumer = AsyncMessageConsumer([message])

        await consumer._consume_queries(kafka_consumer)

        entry = consumer.batch_processors["queries"][0]
        assert entry.event_id == "test-queries:2:8"
        assert entry.kafka_topic == "test-queries"
        assert entry.kafka_partition == 2
        assert entry.kafka_offset == 8

    @pytest.mark.asyncio
    async def test_flush_pending_batches_commits_offsets_after_clickhouse_write(
        self, pipeline_config, sample_query_log
    ):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        consumer.clickhouse.batch_insert_query_logs = AsyncMock()
        consumer.consumers["queries"] = AsyncMock()
        tp = TopicPartition("test-queries", 0)
        consumer.batch_processors["queries"].append(sample_query_log)
        consumer.pending_commit_offsets["queries"][tp] = 8

        await consumer.flush_pending_batches()

        consumer.clickhouse.batch_insert_query_logs.assert_awaited_once_with(
            [sample_query_log]
        )
        consumer.consumers["queries"].commit.assert_awaited_once_with({tp: 8})
        assert consumer.batch_processors["queries"] == []
        assert consumer.pending_commit_offsets["queries"] == {}
        assert consumer.awaiting_commit_offsets["queries"] == {}

    @pytest.mark.asyncio
    async def test_flush_pending_batches_retries_commit_without_duplicate_writes(
        self, pipeline_config, sample_query_log
    ):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        consumer.clickhouse.batch_insert_query_logs = AsyncMock()
        consumer.consumers["queries"] = AsyncMock()
        consumer.consumers["queries"].commit = AsyncMock(
            side_effect=[RuntimeError("commit failed"), None]
        )
        tp = TopicPartition("test-queries", 0)
        consumer.batch_processors["queries"].append(sample_query_log)
        consumer.pending_commit_offsets["queries"][tp] = 9

        with pytest.raises(RuntimeError, match="commit failed"):
            await consumer.flush_pending_batches()

        assert consumer.clickhouse.batch_insert_query_logs.await_count == 1
        assert consumer.batch_processors["queries"] == []
        assert consumer.awaiting_commit_offsets["queries"] == {tp: 9}

        await consumer.flush_pending_batches()

        assert consumer.clickhouse.batch_insert_query_logs.await_count == 1
        assert consumer.awaiting_commit_offsets["queries"] == {}

    @pytest.mark.asyncio
    async def test_flush_pending_batches_commits_routing_guardrail_offsets(
        self, pipeline_config
    ):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        consumer.clickhouse.batch_insert_alert_events = AsyncMock()
        consumer.consumers["routing_guardrails"] = AsyncMock()
        tp = TopicPartition("routing.guardrails", 0)
        consumer.batch_processors["routing_guardrails"].append(
            AlertEventEntry(
                timestamp=datetime.now(timezone.utc),
                alert_type="provider_high_latency",
                severity="error",
                description="Provider latency exceeded threshold",
                source_event_type="routing.guardrails",
            )
        )
        consumer.pending_commit_offsets["routing_guardrails"][tp] = 5

        await consumer.flush_pending_batches()

        consumer.clickhouse.batch_insert_alert_events.assert_awaited_once()
        consumer.consumers["routing_guardrails"].commit.assert_awaited_once_with(
            {tp: 5}
        )
        assert consumer.batch_processors["routing_guardrails"] == []
        assert consumer.pending_commit_offsets["routing_guardrails"] == {}
        assert consumer.awaiting_commit_offsets["routing_guardrails"] == {}


class TestKafkaIngestionPipeline:
    @pytest.mark.asyncio
    async def test_health_status_reflects_component_state(self, pipeline_config):
        pipeline = KafkaIngestionPipeline(pipeline_config)
        pipeline.running = True
        pipeline.producer_manager.producer = object()
        pipeline.clickhouse_manager.client = object()
        pipeline.consumer_manager = SimpleNamespace(
            consumers={"queries": object(), "metrics": object()}
        )

        health = pipeline.get_health_status()

        assert health["pipeline_running"] is True
        assert health["producer_healthy"] is True
        assert health["consumer_healthy"] is True
        assert health["clickhouse_healthy"] is True
        assert health["total_consumers"] == 2

    @pytest.mark.asyncio
    async def test_log_query_delegates_to_producer(self, pipeline_config):
        pipeline = KafkaIngestionPipeline(pipeline_config)
        pipeline.producer_manager.produce_query_log = AsyncMock()

        request = QueryRequest(query="hello", user_id="u1", user_tier=UserTier.FREE)
        response = InferenceResponse(
            response_text="world",
            model_name="gpt-5",
            provider="openai",
            token_count_input=1,
            token_count_output=1,
            total_tokens=2,
            latency_ms=10,
            tokens_per_second=100.0,
            cost_usd=0.01,
        )
        decision = MagicMock()

        await pipeline.log_query(request, response, decision)

        pipeline.producer_manager.produce_query_log.assert_awaited_once_with(
            request, response, decision
        )

    @pytest.mark.asyncio
    async def test_get_query_analytics_alias_delegates_to_clickhouse(self, pipeline_config):
        pipeline = KafkaIngestionPipeline(pipeline_config)
        pipeline.clickhouse_manager.get_query_analytics = AsyncMock(
            return_value={"total_queries": 3}
        )

        analytics = await pipeline.get_query_analytics(hours=6)

        pipeline.clickhouse_manager.get_query_analytics.assert_awaited_once_with(
            None, 6
        )
        assert analytics == {"total_queries": 3}

    def test_attach_monitoring_service_registers_stream_handlers(self, pipeline_config):
        pipeline = KafkaIngestionPipeline(pipeline_config)

        class MonitoringStub:
            async def ingest_stream_request_enriched(self, routing_event):
                return routing_event

            async def ingest_stream_model_metrics(self, metric_event):
                return metric_event

            async def ingest_stream_routing_guardrail(self, guardrail_event):
                return guardrail_event

            async def ingest_stream_routing_policy_state(self, state_event):
                return state_event

            async def ingest_stream_alert(self, alert_event):
                return alert_event

        pipeline.attach_monitoring_service(MonitoringStub())

        assert len(pipeline._stream_handlers["requests_enriched"]) == 1
        assert len(pipeline._stream_handlers["analytics_model_metrics_1m"]) == 1
        assert len(pipeline._stream_handlers["routing_guardrails"]) == 1
        assert len(pipeline._stream_handlers["routing_policy_state"]) == 1
        assert len(pipeline._stream_handlers["alerts"]) == 1
