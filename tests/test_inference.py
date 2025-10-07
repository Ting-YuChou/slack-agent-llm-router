"""
Unit tests for inference engine components
Tests for model providers, context compression, caching, and routing
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.llm_router_part2_inference import (
    InferenceEngine,
    OpenAIProvider,
    AnthropicProvider,
    vLLMProvider,
    ContextCompressor,
    ResponseCache,
    BatchProcessor
)
from src.llm_router_part1_router import ModelRouter
from src.utils.schema import QueryRequest, InferenceResponse


@pytest.fixture
def mock_config():
    """Mock configuration for inference components"""
    return {
        'openai': {
            'api_key': 'test-key',
            'timeout': 60
        },
        'anthropic': {
            'api_key': 'test-key',
            'timeout': 60
        },
        'vllm': {
            'host': 'localhost',
            'port': 8000,
            'timeout': 300
        },
        'compression': {
            'compression_ratio': 0.3,
            'max_context_tokens': 100000,
            'method': 'semantic_graph'
        },
        'cache': {
            'enabled': True,
            'ttl': 3600,
            'max_size': '1GB'
        },
        'batching': {
            'enabled': True,
            'max_batch_size': 32,
            'max_wait_time_ms': 50
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
        context="This is test context for the query."
    )


@pytest.fixture
def mock_router():
    """Mock router for testing"""
    router = MagicMock(spec=ModelRouter)
    
    # Mock routing decision
    class MockRoutingDecision:
        def __init__(self):
            self.selected_model = "gpt-3.5-turbo"
            self.query_type = "general"
            self.confidence = 0.95
    
    router.route_query.return_value = MockRoutingDecision()
    router.get_model_info.return_value = {
        'config': {'provider': 'openai'},
        'status': 'healthy'
    }
    router.update_model_stats = MagicMock()
    
    return router


class TestOpenAIProvider:
    """Test OpenAI provider implementation"""
    
    @pytest.mark.asyncio
    async def test_provider_initialization(self, mock_config):
        """Test OpenAI provider initialization"""
        with patch('src.llm_router_part2_inference.openai.AsyncOpenAI') as mock_openai:
            provider = OpenAIProvider(mock_config['openai'])
            await provider.initialize()
            
            mock_openai.assert_called_once_with(
                api_key='test-key',
                base_url='https://api.openai.com/v1',
                timeout=60
            )
            assert provider.client is not None
    
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_config, sample_query_request):
        """Test response generation"""
        with patch('src.llm_router_part2_inference.openai.AsyncOpenAI') as mock_openai:
            # Mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Machine learning is a subset of AI..."
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 100
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            provider = OpenAIProvider(mock_config['openai'])
            await provider.initialize()
            
            response = await provider.generate_response(sample_query_request, "gpt-3.5-turbo")
            
            assert isinstance(response, InferenceResponse)
            assert response.response_text == "Machine learning is a subset of AI..."
            assert response.model_name == "gpt-3.5-turbo"
            assert response.token_count_input == 50
            assert response.token_count_output == 100
            assert response.provider == "openai"
            assert not response.cached
    
    @pytest.mark.asyncio
    async def test_stream_response(self, mock_config, sample_query_request):
        """Test streaming response"""
        with patch('src.llm_router_part2_inference.openai.AsyncOpenAI') as mock_openai:
            # Mock streaming response
            async def mock_stream():
                chunks = ["Hello", " world", "!"]
                for chunk in chunks:
                    mock_chunk = MagicMock()
                    mock_chunk.choices = [MagicMock()]
                    mock_chunk.choices[0].delta.content = chunk
                    yield mock_chunk
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_stream()
            mock_openai.return_value = mock_client
            
            provider = OpenAIProvider(mock_config['openai'])
            await provider.initialize()
            
            chunks = []
            async for chunk in provider.stream_response(sample_query_request, "gpt-3.5-turbo"):
                chunks.append(chunk)
            
            assert chunks == ["Hello", " world", "!"]
    
    def test_cost_calculation(self, mock_config):
        """Test cost calculation"""
        provider = OpenAIProvider(mock_config['openai'])
        
        # Mock usage object
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 150
        
        cost = provider._calculate_cost(usage, "gpt-3.5-turbo")
        
        assert isinstance(cost, float)
        assert cost > 0
    
    def test_health_status(self, mock_config):
        """Test health status"""
        provider = OpenAIProvider(mock_config['openai'])
        
        health = provider.get_health_status()
        
        assert health['provider'] == 'openai'
        assert 'status' in health
        assert 'models_available' in health


class TestAnthropicProvider:
    """Test Anthropic provider implementation"""
    
    @pytest.mark.asyncio
    async def test_provider_initialization(self, mock_config):
        """Test Anthropic provider initialization"""
        with patch('src.llm_router_part2_inference.anthropic.AsyncAnthropic') as mock_anthropic:
            provider = AnthropicProvider(mock_config['anthropic'])
            await provider.initialize()
            
            mock_anthropic.assert_called_once_with(
                api_key='test-key',
                base_url='https://api.anthropic.com',
                timeout=60
            )
            assert provider.client is not None
    
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_config, sample_query_request):
        """Test response generation"""
        with patch('src.llm_router_part2_inference.anthropic.AsyncAnthropic') as mock_anthropic:
            # Mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Machine learning is a field of study..."
            mock_response.usage.input_tokens = 60
            mock_response.usage.output_tokens = 120
            
            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client
            
            provider = AnthropicProvider(mock_config['anthropic'])
            await provider.initialize()
            
            response = await provider.generate_response(sample_query_request, "claude-3.5-sonnet")
            
            assert isinstance(response, InferenceResponse)
            assert response.response_text == "Machine learning is a field of study..."
            assert response.model_name == "claude-3.5-sonnet"
            assert response.token_count_input == 60
            assert response.token_count_output == 120
            assert response.provider == "anthropic"
    
    @pytest.mark.asyncio
    async def test_stream_response(self, mock_config, sample_query_request):
        """Test streaming response"""
        with patch('src.llm_router_part2_inference.anthropic.AsyncAnthropic') as mock_anthropic:
            # Mock streaming context manager
            class MockStreamContext:
                def __init__(self):
                    self.text_stream = self._mock_stream()
                
                async def __aenter__(self):
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
                
                async def _mock_stream(self):
                    chunks = ["Machine", " learning", " is", " fascinating"]
                    for chunk in chunks:
                        yield chunk
            
            mock_client = AsyncMock()
            mock_client.messages.stream.return_value = MockStreamContext()
            mock_anthropic.return_value = mock_client
            
            provider = AnthropicProvider(mock_config['anthropic'])
            await provider.initialize()
            
            chunks = []
            async for chunk in provider.stream_response(sample_query_request, "claude-3.5-sonnet"):
                chunks.append(chunk)
            
            assert chunks == ["Machine", " learning", " is", " fascinating"]


class TestvLLMProvider:
    """Test vLLM provider implementation"""
    
    @pytest.mark.asyncio
    async def test_provider_initialization(self, mock_config):
        """Test vLLM provider initialization"""
        with patch('src.llm_router_part2_inference.httpx.AsyncClient') as mock_httpx:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_httpx.return_value = mock_client
            
            provider = vLLMProvider(mock_config['vllm'])
            await provider.initialize()
            
            assert provider.http_client == mock_client
            assert provider.base_url == "http://localhost:8000"
    
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_config, sample_query_request):
        """Test response generation"""
        with patch('src.llm_router_part2_inference.httpx.AsyncClient') as mock_httpx:
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"text": "Machine learning algorithms learn patterns..."}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 80}
            }
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.get.return_value = MagicMock(status_code=200)  # Health check
            mock_client.post.return_value = mock_response
            mock_httpx.return_value = mock_client
            
            provider = vLLMProvider(mock_config['vllm'])
            await provider.initialize()
            
            response = await provider.generate_response(sample_query_request, "mistral-7b")
            
            assert isinstance(response, InferenceResponse)
            assert response.response_text == "Machine learning algorithms learn patterns..."
            assert response.model_name == "mistral-7b"
            assert response.token_count_input == 40
            assert response.token_count_output == 80
            assert response.provider == "vllm"
            assert response.cost_usd == 0.0  # Self-hosted models have no API cost
    
    @pytest.mark.asyncio
    async def test_stream_response(self, mock_config, sample_query_request):
        """Test streaming response"""
        with patch('src.llm_router_part2_inference.httpx.AsyncClient') as mock_httpx:
            # Mock streaming response
            async def mock_aiter_lines():
                lines = [
                    'data: {"choices": [{"text": "Hello"}]}',
                    'data: {"choices": [{"text": " world"}]}',
                    'data: {"choices": [{"text": "!"}]}'
                ]
                for line in lines:
                    yield line
            
            mock_response = MagicMock()
            mock_response.aiter_lines = mock_aiter_lines
            
            mock_client = AsyncMock()
            mock_client.get.return_value = MagicMock(status_code=200)
            mock_client.stream.return_value.__aenter__.return_value = mock_response
            mock_httpx.return_value = mock_client
            
            provider = vLLMProvider(mock_config['vllm'])
            await provider.initialize()
            
            chunks = []
            async for chunk in provider.stream_response(sample_query_request, "mistral-7b"):
                chunks.append(chunk)
            
            assert chunks == ["Hello", " world", "!"]


class TestContextCompressor:
    """Test context compression functionality"""
    
    @pytest.mark.asyncio
    async def test_compressor_initialization(self, mock_config):
        """Test context compressor initialization"""
        with patch('src.llm_router_part2_inference.AutoTokenizer.from_pretrained') as mock_tokenizer:
            compressor = ContextCompressor(mock_config['compression'])
            await compressor.initialize()
            
            mock_tokenizer.assert_called_once()
            assert compressor.compression_ratio == 0.3
            assert compressor.max_context_tokens == 100000
    
    @pytest.mark.asyncio
    async def test_semantic_graph_compression(self, mock_config):
        """Test semantic graph compression method"""
        compressor = ContextCompressor(mock_config['compression'])
        
        long_context = """
        This is the first sentence with important information.
        This is a less important sentence.
        This contains crucial data that should be preserved.
        Another mundane sentence here.
        Final sentence with key insights.
        """
        
        compressed = await compressor.compress_context(long_context, 100)
        
        assert isinstance(compressed, str)
        assert len(compressed) <= 100
        # Should preserve important sentences
        assert "important" in compressed or "crucial" in compressed or "key" in compressed
    
    @pytest.mark.asyncio
    async def test_sliding_window_compression(self, mock_config):
        """Test sliding window compression"""
        config = mock_config['compression'].copy()
        config['method'] = 'sliding_window'
        
        compressor = ContextCompressor(config)
        
        long_text = "A" * 1000 + "B" * 1000 + "C" * 1000
        target_length = 500
        
        compressed = compressor._sliding_window_compression(long_text, target_length)
        
        assert len(compressed) <= target_length + 50  # Allow for separator
        assert compressed.startswith("A")
        assert compressed.endswith("C")
        assert "[COMPRESSED]" in compressed
    
    def test_sentence_splitting(self, mock_config):
        """Test sentence splitting"""
        compressor = ContextCompressor(mock_config['compression'])
        
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = compressor._split_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0] == "First sentence"
        assert sentences[1] == "Second sentence"
        assert sentences[2] == "Third sentence"
        assert sentences[3] == "Fourth sentence"
    
    def test_key_phrase_extraction(self, mock_config):
        """Test key phrase extraction"""
        compressor = ContextCompressor(mock_config['compression'])