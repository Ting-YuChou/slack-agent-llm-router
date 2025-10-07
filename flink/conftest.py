"""
Test configuration and shared fixtures for LLM Router tests
Provides common test utilities, fixtures, and configuration
"""

import asyncio
import json
import os
import pytest
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

# Test configuration
TEST_CONFIG = {
    'api': {
        'host': '127.0.0.1',
        'port': 8080,
        'debug': True
    },
    'router': {
        'models': {
            'gpt-3.5-turbo': {
                'provider': 'openai',
                'priority': 1,
                'cost_per_token': 0.000002,
                'max_tokens': 4096,
                'capabilities': ['text']
            },
            'claude-3.5-sonnet': {
                'provider': 'anthropic',
                'priority': 2,
                'cost_per_token': 0.000003,
                'max_tokens': 200000,
                'capabilities': ['text', 'analysis']
            },
            'mistral-7b': {
                'provider': 'vllm',
                'priority': 3,
                'cost_per_token': 0.0,
                'max_tokens': 8192,
                'capabilities': ['text', 'code']
            }
        },
        'routing_rules': {
            'general': 'gpt-3.5-turbo',
            'analysis': 'claude-3.5-sonnet',
            'code_generation': 'mistral-7b',
            'summarization': 'gpt-3.5-turbo'
        }
    },
    'inference': {
        'openai': {
            'api_key': 'test-openai-key',
            'base_url': 'https://api.openai.com/v1',
            'timeout': 60
        },
        'anthropic': {
            'api_key': 'test-anthropic-key',
            'base_url': 'https://api.anthropic.com',
            'timeout': 60
        },
        'vllm': {
            'host': 'localhost',
            'port': 8000,
            'timeout': 300
        },
        'compression': {
            'enabled': True,
            'compression_ratio': 0.3,
            'max_context_tokens': 100000,
            'method': 'semantic_graph'
        },
        'cache': {
            'enabled': True,
            'ttl': 3600,
            'max_size': '1GB'
        }
    },
    'kafka': {
        'bootstrap_servers': ['localhost:9092'],
        'topics': {
            'queries': 'test-llm-queries',
            'responses': 'test-llm-responses',
            'metrics': 'test-llm-metrics',
            'errors': 'test-llm-errors'
        },
        'producer': {
            'acks': 'all',
            'retries': 3,
            'batch_size': 16384
        },
        'consumer': {
            'group_id': 'test-consumer-group',
            'auto_offset_reset': 'latest'
        }
    },
    'clickhouse': {
        'host': 'localhost',
        'port': 8123,
        'database': 'test_llm_router',
        'username': 'default',
        'password': ''
    },
    'monitoring': {
        'prometheus_port': 8000,
        'health_checks': {
            'interval_seconds': 30
        },
        'alerts': {
            'error_rate_threshold': 0.05,
            'latency_threshold_ms': 5000,
            'memory_usage_threshold': 0.9
        },
        'slack': {
            'enabled': False,
            'webhook_url': 'https://hooks.slack.com/test',
            'channel': '#test-alerts'
        }
    }
}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return TEST_CONFIG.copy()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_query_requests():
    """Generate sample query requests for testing"""
    from src.utils.schema import QueryRequest
    
    return [
        QueryRequest(
            query="What is machine learning?",
            user_id="user_001",
            user_tier="premium",
            max_tokens=500,
            temperature=0.7,
            context="Educational context about AI"
        ),
        QueryRequest(
            query="Write a Python function to calculate fibonacci numbers",
            user_id="user_002",
            user_tier="enterprise",
            max_tokens=1000,
            temperature=0.3,
            context="Code generation task"
        ),
        QueryRequest(
            query="Analyze the quarterly sales data and provide insights",
            user_id="user_003",
            user_tier="enterprise",
            max_tokens=2000,
            temperature=0.5,
            context="Sales data analysis for Q3 2024"
        ),
        QueryRequest(
            query="Summarize this technical document",
            user_id="user_004",
            user_tier="free",
            max_tokens=300,
            temperature=0.4,
            context="Document summarization task"
        )
    ]


@pytest.fixture
def sample_inference_responses():
    """Generate sample inference responses for testing"""
    from src.utils.schema import InferenceResponse
    
    return [
        InferenceResponse(
            response_text="Machine learning is a subset of artificial intelligence...",
            model_name="gpt-3.5-turbo",
            token_count_input=100,
            token_count_output=200,
            latency_ms=1500,
            cost_usd=0.0006,
            provider="openai",
            cached=False
        ),
        InferenceResponse(
            response_text="def fibonacci(n):\n    if n <= 1:\n        return n...",
            model_name="mistral-7b",
            token_count_input=150,
            token_count_output=300,
            latency_ms=2000,
            cost_usd=0.0,
            provider="vllm",
            cached=False
        ),
        InferenceResponse(
            response_text="Based on the Q3 sales data analysis...",
            model_name="claude-3.5-sonnet",
            token_count_input=500,
            token_count_output=800,
            latency_ms=3000,
            cost_usd=0.0039,
            provider="anthropic",
            cached=True
        )
    ]


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing"""
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send = AsyncMock()
    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer for testing"""
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    
    # Mock async iteration
    async def mock_aiter():
        # Return some mock messages
        messages = [
            create_mock_kafka_message({"query_id": "test1", "user_id": "user1"}),
            create_mock_kafka_message({"query_id": "test2", "user_id": "user2"})
        ]
        for msg in messages:
            yield msg
    
    consumer.__aiter__ = mock_aiter
    return consumer


def create_mock_kafka_message(data: Dict[str, Any]):
    """Create mock Kafka message"""
    message = MagicMock()
    message.value = data
    message.key = data.get('user_id', 'test_key')
    message.timestamp = int(datetime.now().timestamp() * 1000)
    message.topic = 'test-topic'
    message.partition = 0
    message.offset = 12345
    return message


@pytest.fixture
def mock_clickhouse_client():
    """Mock ClickHouse client for testing"""
    client = MagicMock()
    
    # Mock query results
    client.query.return_value.result_rows = [
        [100, 5000, 2.5, 1500, 95.0, "gpt-3.5-turbo", "general"],
        [50, 2500, 1.2, 800, 98.0, "claude-3", "analysis"]
    ]
    
    # Mock insert operation
    client.insert = MagicMock()
    
    # Mock command operation
    client.command = MagicMock()
    
    # Mock close operation
    client.close = MagicMock()
    
    return client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    client = AsyncMock()
    client.ping = AsyncMock()
    client.get = AsyncMock()