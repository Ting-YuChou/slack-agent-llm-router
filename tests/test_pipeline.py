import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm_router_part3_pipeline import (
    ClickHouseManager,
    KafkaConsumerManager,
    KafkaIngestionPipeline,
    KafkaProducerManager,
    MetricEntry,
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
        selected_model="gpt-4-turbo",
        token_count_input=100,
        token_count_output=50,
        latency_ms=850,
        cost_usd=0.02,
        status="success",
    )


class TestQueryLogEntry:
    def test_to_dict_serializes_enums(self, sample_query_log):
        payload = sample_query_log.to_dict()

        assert payload["user_tier"] == "premium"
        assert payload["query_type"] == "analysis"
        assert payload["query_text"] == "Analyze this data"


class TestKafkaProducerManager:
    @pytest.mark.asyncio
    async def test_initialize_creates_producer(self, pipeline_config):
        with patch("src.llm_router_part3_pipeline.AIOKafkaProducer") as mock_producer_cls:
            producer = AsyncMock()
            mock_producer_cls.return_value = producer

            manager = KafkaProducerManager(pipeline_config)
            await manager.initialize()

            producer.start.assert_awaited_once()
            assert manager.producer is producer

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


class TestClickHouseManager:
    @pytest.mark.asyncio
    async def test_insert_query_log_calls_insert(self, pipeline_config, sample_query_log):
        with patch("src.llm_router_part3_pipeline.clickhouse_connect.get_client") as mock_get_client:
            client = MagicMock()
            client.query.return_value.result_rows = [[1]]
            mock_get_client.return_value = client

            manager = ClickHouseManager(pipeline_config["clickhouse"])
            await manager.initialize()
            await manager.insert_query_log(sample_query_log)

            client.insert.assert_called_once_with("query_logs", [sample_query_log.to_dict()])

    @pytest.mark.asyncio
    async def test_insert_metric_calls_insert(self, pipeline_config):
        with patch("src.llm_router_part3_pipeline.clickhouse_connect.get_client") as mock_get_client:
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

            client.insert.assert_called_once_with("system_metrics", [metric.to_dict()])


class TestKafkaConsumerManager:
    def test_deserialize_message_returns_json_payload(self, pipeline_config):
        consumer = KafkaConsumerManager(pipeline_config, MagicMock())
        raw = json.dumps({"hello": "world"}).encode("utf-8")

        assert consumer._deserialize_message(raw) == {"hello": "world"}


class TestKafkaIngestionPipeline:
    @pytest.mark.asyncio
    async def test_health_status_reflects_component_state(self, pipeline_config):
        pipeline = KafkaIngestionPipeline(pipeline_config)
        pipeline.running = True
        pipeline.producer_manager.producer = object()
        pipeline.clickhouse_manager.client = object()
        pipeline.consumer_manager = SimpleNamespace(consumers={"queries": object(), "metrics": object()})

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
            model_name="gpt-4-turbo",
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

        pipeline.producer_manager.produce_query_log.assert_awaited_once_with(request, response, decision)
