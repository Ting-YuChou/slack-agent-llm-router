"""
Unit tests for Kafka pipeline components
Tests for stream processing, data ingestion, and ClickHouse integration
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm_router_part3_pipeline import (
    KafkaProducerManager,
    KafkaConsumerManager,
    ClickHouseManager,
    KafkaIngestionPipeline,
    QueryLogEntry,
    MetricEntry
)
from src.utils.schema import QueryRequest, InferenceResponse


@pytest.fixture
def mock_config():
    """Mock configuration for pipeline components"""
    return {
        'bootstrap_servers': ['localhost:9092'],
        'topics': {
            'queries': 'test-queries',
            'responses': 'test-responses',
            'metrics': 'test-metrics',
            'errors': 'test-errors'
        },
        'producer': {
            'acks': 'all',
            'retries': 3,
            'batch_size': 16384
        },
        'consumer': {
            'group_id': 'test-consumer-group',
            'auto_offset_reset': 'latest'
        },
        'clickhouse': {
            'host': 'localhost',
            'port': 8123,
            'database': 'test_llm_router',
            'username': 'default',
            'password': ''
        }
    }


@pytest.fixture
def sample_query_request():
    """Sample query request for testing"""
    return QueryRequest(
        query="What is machine learning?",
        user_id="test_user_123",
        user_tier="premium",
        max_tokens=500,
        temperature=0.7,
        context="This is test context"
    )


@pytest.fixture
def sample_inference_response():
    """Sample inference response for testing"""
    return InferenceResponse(
        response_text="Machine learning is a subset of artificial intelligence...",
        model_name="gpt-3.5-turbo",
        token_count_input=100,
        token_count_output=200,
        latency_ms=1500,
        cost_usd=0.05,
        provider="openai",
        cached=False
    )


@pytest.fixture
def sample_routing_decision():
    """Sample routing decision for testing"""
    class MockRoutingDecision:
        def __init__(self):
            self.selected_model = "gpt-3.5-turbo"
            self.query_type = "general"
    
    return MockRoutingDecision()


class TestKafkaProducerManager:
    """Test Kafka producer functionality"""
    
    @pytest.mark.asyncio
    async def test_producer_initialization(self, mock_config):
        """Test producer initialization"""
        with patch('src.llm_router_part3_pipeline.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            producer_manager = KafkaProducerManager(mock_config)
            await producer_manager.initialize()
            
            assert producer_manager.producer == mock_producer
            mock_producer.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_produce_query_log(self, mock_config, sample_query_request, 
                                   sample_inference_response, sample_routing_decision):
        """Test query log production"""
        with patch('src.llm_router_part3_pipeline.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            producer_manager = KafkaProducerManager(mock_config)
            await producer_manager.initialize()
            
            # Test producing query log
            await producer_manager.produce_query_log(
                sample_query_request,
                sample_inference_response,
                sample_routing_decision
            )
            
            # Verify producer.send was called
            mock_producer.send.assert_called_once()
            call_args = mock_producer.send.call_args
            
            assert call_args.kwargs['topic'] == 'test-queries'
            assert call_args.kwargs['key'] == sample_query_request.user_id
            
            # Verify message content
            message_data = call_args.kwargs['value']
            assert message_data['user_id'] == sample_query_request.user_id
            assert message_data['selected_model'] == sample_inference_response.model_name
            assert message_data['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_produce_metric(self, mock_config):
        """Test metric production"""
        with patch('src.llm_router_part3_pipeline.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            producer_manager = KafkaProducerManager(mock_config)
            await producer_manager.initialize()
            
            # Test producing metric
            await producer_manager.produce_metric(
                service="test_service",
                metric_name="test_metric",
                metric_value=42.5,
                labels={"env": "test"}
            )
            
            mock_producer.send.assert_called_once()
            call_args = mock_producer.send.call_args
            
            assert call_args.kwargs['topic'] == 'test-metrics'
            assert call_args.kwargs['key'] == "test_service:test_metric"
            
            message_data = call_args.kwargs['value']
            assert message_data['service'] == "test_service"
            assert message_data['metric_name'] == "test_metric"
            assert message_data['metric_value'] == 42.5
            assert message_data['labels'] == {"env": "test"}
    
    @pytest.mark.asyncio
    async def test_producer_error_handling(self, mock_config):
        """Test producer error handling"""
        with patch('src.llm_router_part3_pipeline.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.send.side_effect = Exception("Kafka error")
            mock_producer_class.return_value = mock_producer
            
            producer_manager = KafkaProducerManager(mock_config)
            await producer_manager.initialize()
            
            # Should not raise exception, should handle gracefully
            await producer_manager.produce_metric("test", "test", 1.0)
            
            # Verify error was handled (logged, not raised)
            mock_producer.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_serialization(self, mock_config):
        """Test message serialization"""
        producer_manager = KafkaProducerManager(mock_config)
        
        # Test datetime serialization
        test_data = {
            'timestamp': datetime.now(timezone.utc),
            'value': 42,
            'name': 'test'
        }
        
        serialized = producer_manager._serialize_message(test_data)
        
        # Should be valid JSON bytes
        assert isinstance(serialized, bytes)
        deserialized = json.loads(serialized.decode('utf-8'))
        assert deserialized['value'] == 42
        assert deserialized['name'] == 'test'
        assert 'timestamp' in deserialized


class TestClickHouseManager:
    """Test ClickHouse database operations"""
    
    @pytest.mark.asyncio
    async def test_clickhouse_initialization(self, mock_config):
        """Test ClickHouse initialization"""
        with patch('src.llm_router_part3_pipeline.clickhouse_connect.get_client') as mock_client:
            mock_client.return_value.query.return_value.result_rows = [[1]]
            
            ch_manager = ClickHouseManager(mock_config['clickhouse'])
            await ch_manager.initialize()
            
            assert ch_manager.client is not None
            mock_client.assert_called_once_with(
                host='localhost',
                port=8123,
                username='default',
                password='',
                database='test_llm_router'
            )
    
    @pytest.mark.asyncio
    async def test_insert_query_log(self, mock_config):
        """Test query log insertion"""
        with patch('src.llm_router_part3_pipeline.clickhouse_connect.get_client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.query.return_value.result_rows = [[1]]
            
            ch_manager = ClickHouseManager(mock_config['clickhouse'])
            await ch_manager.initialize()
            
            # Create test query log
            query_log = QueryLogEntry(
                query_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                user_id="test_user",
                user_tier="premium",
                query_text="Test query",
                query_type="general",
                selected_model="gpt-3.5-turbo",
                token_count_input=50,
                token_count_output=100,
                latency_ms=1000,
                cost_usd=0.02,
                status="success"
            )
            
            await ch_manager.insert_query_log(query_log)
            
            # Verify insert was called
            mock_client_instance.insert.assert_called_once_with('query_logs', [query_log.to_dict()])
    
    @pytest.mark.asyncio
    async def test_batch_insert_query_logs(self, mock_config):
        """Test batch insertion of query logs"""
        with patch('src.llm_router_part3_pipeline.clickhouse_connect.get_client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.query.return_value.result_rows = [[1]]
            
            ch_manager = ClickHouseManager(mock_config['clickhouse'])
            await ch_manager.initialize()
            
            # Create test query logs
            query_logs = []
            for i in range(5):
                query_logs.append(QueryLogEntry(
                    query_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    user_id=f"test_user_{i}",
                    user_tier="premium",
                    query_text=f"Test query {i}",
                    query_type="general",
                    selected_model="gpt-3.5-turbo",
                    token_count_input=50,
                    token_count_output=100,
                    latency_ms=1000,
                    cost_usd=0.02,
                    status="success"
                ))
            
            await ch_manager.batch_insert_query_logs(query_logs)
            
            # Verify batch insert was called
            mock_client_instance.insert.assert_called_once()
            call_args = mock_client_instance.insert.call_args
            assert call_args[0][0] == 'query_logs'  # table name
            assert len(call_args[0][1]) == 5  # 5 records
    
    @pytest.mark.asyncio
    async def test_get_query_analytics(self, mock_config):
        """Test query analytics retrieval"""
        with patch('src.llm_router_part3_pipeline.clickhouse_connect.get_client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.query.return_value.result_rows = [[1]]
            
            # Mock analytics query result
            mock_client_instance.query.return_value.result_rows = [
                [100, 5000, 2.5, 1500, 95.0, "gpt-3.5-turbo", "general"],
                [50, 2500, 1.2, 800, 98.0, "claude-3", "analysis"]
            ]
            
            ch_manager = ClickHouseManager(mock_config['clickhouse'])
            await ch_manager.initialize()
            
            analytics = await ch_manager.get_query_analytics(user_id="test_user", hours=24)
            
            assert analytics['total_queries'] == 150  # Sum of queries
            assert analytics['total_tokens'] == 7500  # Sum of tokens
            assert analytics['total_cost'] == 3.7  # Sum of costs
            assert 'model_breakdown' in analytics
            assert 'query_type_breakdown' in analytics
    
    @pytest.mark.asyncio
    async def test_get_model_performance(self, mock_config):
        """Test model performance retrieval"""
        with patch('src.llm_router_part3_pipeline.clickhouse_connect.get_client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.query.return_value.result_rows = [[1]]
            
            # Mock performance query result
            mock_client_instance.query.return_value.result_rows = [
                ["gpt-3.5-turbo", 1000, 95.5, 1200, 50.2, 45, 12.5],
                ["claude-3", 500, 98.1, 800, 75.3, 10, 8.2]
            ]
            
            ch_manager = ClickHouseManager(mock_config['clickhouse'])
            await ch_manager.initialize()
            
            performance = await ch_manager.get_model_performance(hours=24)
            
            assert len(performance) == 2
            assert performance[0]['model_name'] == 'gpt-3.5-turbo'
            assert performance[0]['requests'] == 1000
            assert performance[0]['success_rate'] == 95.5
            assert performance[1]['model_name'] == 'claude-3'


class TestQueryLogEntry:
    """Test QueryLogEntry data structure"""
    
    def test_query_log_entry_creation(self):
        """Test creating QueryLogEntry"""
        entry = QueryLogEntry(
            query_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            user_id="test_user",
            user_tier="premium",
            query_text="Test query",
            query_type="general",
            selected_model="gpt-3.5-turbo",
            token_count_input=50,
            token_count_output=100,
            latency_ms=1000,
            cost_usd=0.02,
            status="success"
        )
        
        assert entry.user_id == "test_user"
        assert entry.status == "success"
        assert entry.cost_usd == 0.02
    
    def test_query_log_entry_to_dict(self):
        """Test converting QueryLogEntry to dict"""
        entry = QueryLogEntry(
            query_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            user_id="test_user",
            user_tier="premium",
            query_text="Test query",
            query_type="general",
            selected_model="gpt-3.5-turbo",
            token_count_input=50,
            token_count_output=100,
            latency_ms=1000,
            cost_usd=0.02,
            status="success"
        )
        
        data = entry.to_dict()
        
        assert