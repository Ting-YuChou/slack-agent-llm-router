"""
LLM Router Part 2: Inference Engine - Multi-Provider Model Inference
Handles inference requests across OpenAI, Anthropic, and vLLM providers
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncIterator, Union, Tuple
import hashlib

import httpx
import openai
import anthropic
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer

from src.llm_router_part1_router import ModelRouter
from src.utils.logger import setup_logging
from src.utils.metrics import INFERENCE_METRICS
from src.utils.schema import (
    AttachmentType,
    QueryRequest,
    InferenceResponse,
    ModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceContext:
    """Context information for inference requests"""

    request_id: str
    user_id: str
    model_name: str
    start_time: float
    token_count_input: int
    compressed_context: bool = False
    cached_response: bool = False


class ContextCompressor:
    """Advanced context compression for long sequences"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_ratio = config.get("compression_ratio", 0.3)
        self.max_context_tokens = config.get("max_context_tokens", 100000)
        self.method = config.get("method", "semantic_graph")
        self.tokenizer = None

    async def initialize(self):
        """Initialize compression components"""
        try:
            # Load tokenizer for token counting
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            logger.info("Context compressor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize context compressor: {e}")

    async def compress_context(self, context: str, target_length: int) -> str:
        """Compress context to target length while preserving key information"""
        if not context or len(context) < target_length:
            return context

        if self.method == "semantic_graph":
            return await self._semantic_graph_compression(context, target_length)
        elif self.method == "sketch_based":
            return await self._sketch_based_compression(context, target_length)
        elif self.method == "attention_based":
            return await self._attention_based_compression(context, target_length)
        else:
            return self._sliding_window_compression(context, target_length)

    async def _semantic_graph_compression(
        self, context: str, target_length: int
    ) -> str:
        """Semantic graph-based compression preserving key relationships"""
        # Split into sentences
        sentences = self._split_sentences(context)
        if len(sentences) <= 3:
            return context

        # Score sentences by importance
        sentence_scores = await self._score_sentences(sentences)

        # Select top sentences maintaining order
        selected_indices = sorted(
            sorted(
                range(len(sentences)), key=lambda i: sentence_scores[i], reverse=True
            )[: max(3, int(len(sentences) * self.compression_ratio))]
        )

        compressed = " ".join(sentences[i] for i in selected_indices)

        # Ensure target length
        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."

        return compressed

    async def _sketch_based_compression(self, context: str, target_length: int) -> str:
        """Sketch-based compression using key phrase extraction"""
        # Extract key phrases and entities
        key_phrases = self._extract_key_phrases(context)

        # Build compressed version focusing on key information
        sections = context.split("\n\n")
        important_sections = []

        for section in sections:
            if any(phrase.lower() in section.lower() for phrase in key_phrases):
                important_sections.append(section)

        compressed = "\n\n".join(important_sections)

        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."

        return compressed or context[:target_length]

    async def _attention_based_compression(
        self, context: str, target_length: int
    ) -> str:
        """Attention-based compression (simplified implementation)"""
        # This would ideally use a trained attention model
        # For now, implement a heuristic approach

        paragraphs = context.split("\n\n")
        if len(paragraphs) <= 2:
            return context[:target_length]

        # Score paragraphs by length and keyword density
        paragraph_scores = []
        important_keywords = self._extract_keywords(context)

        for para in paragraphs:
            score = len(para)  # Length score
            score += sum(para.lower().count(kw) for kw in important_keywords) * 50
            paragraph_scores.append(score)

        # Select top paragraphs
        num_select = max(2, int(len(paragraphs) * self.compression_ratio))
        top_indices = sorted(
            range(len(paragraphs)), key=lambda i: paragraph_scores[i], reverse=True
        )[:num_select]

        # Maintain original order
        selected_paragraphs = [paragraphs[i] for i in sorted(top_indices)]
        compressed = "\n\n".join(selected_paragraphs)

        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."

        return compressed

    def _sliding_window_compression(self, context: str, target_length: int) -> str:
        """Simple sliding window compression"""
        if len(context) <= target_length:
            return context

        # Take beginning and end, skip middle
        keep_start = target_length // 2
        keep_end = target_length - keep_start - 20  # Reserve space for separator

        start_part = context[:keep_start]
        end_part = context[-keep_end:]

        return f"{start_part}\n...[COMPRESSED]...\n{end_part}"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    async def _score_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences by importance"""
        scores = []

        # Simple heuristic scoring
        for sentence in sentences:
            score = 0

            # Length factor (prefer medium-length sentences)
            length = len(sentence.split())
            if 10 <= length <= 30:
                score += 2
            elif length > 30:
                score += 1

            # Keyword density
            keywords = ["important", "key", "main", "crucial", "significant", "primary"]
            score += sum(sentence.lower().count(kw) for kw in keywords)

            # Position factor (first and last sentences often important)
            if sentences.index(sentence) in [0, len(sentences) - 1]:
                score += 1

            scores.append(score)

        return scores

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple implementation - in production, use NLP libraries
        words = text.lower().split()

        # Find repeated phrases
        phrases = {}
        for i in range(len(words) - 1):
            phrase = " ".join(words[i : i + 2])
            phrases[phrase] = phrases.get(phrase, 0) + 1

        # Return top phrases
        return sorted(phrases.keys(), key=phrases.get, reverse=True)[:10]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        words = text.lower().split()

        # Filter out common words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]

        # Count frequency
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1

        # Return top keywords
        return sorted(word_count.keys(), key=word_count.get, reverse=True)[:15]


class ResponseCache:
    """Intelligent response caching system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.ttl = config.get("ttl", 3600)  # 1 hour
        self.max_size = config.get("max_size", "1GB")
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection"""
        if not self.enabled:
            return

        try:
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Response cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}")
            self.enabled = False

    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        if not self.enabled or not self.redis_client:
            return None

        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                INFERENCE_METRICS.cache_hits.inc()
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        INFERENCE_METRICS.cache_misses.inc()
        return None

    async def cache_response(self, cache_key: str, response_data: Dict[str, Any]):
        """Cache response data"""
        if not self.enabled or not self.redis_client:
            return

        try:
            await self.redis_client.setex(
                cache_key, self.ttl, json.dumps(response_data)
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def generate_cache_key(self, request: QueryRequest, model_name: str) -> str:
        """Generate cache key for request"""
        attachment_signatures = []
        for attachment in request.attachments:
            attachment_signatures.append(
                {
                    "name": attachment.name,
                    "type": attachment.type.value,
                    "size_bytes": attachment.size_bytes,
                    "mime_type": attachment.mime_type,
                    "url": attachment.url,
                    "content_sha1": hashlib.sha1(attachment.content).hexdigest()
                    if attachment.content
                    else None,
                }
            )

        key_payload = {
            "model_name": model_name,
            "user_id": request.user_id,
            "user_tier": request.user_tier.value,
            "query": request.query,
            "context": request.context,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "priority": request.priority,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "metadata": request.metadata,
            "attachments": attachment_signatures,
        }
        return hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


class BaseInferenceProvider(ABC):
    """Abstract base class for inference providers"""

    ATTACHMENT_PROMPT_CHAR_LIMIT = 12_000
    ATTACHMENT_PREVIEW_CHAR_LIMIT = 4_000
    TEXT_MIME_MARKERS = (
        "text/",
        "json",
        "csv",
        "xml",
        "yaml",
        "yml",
        "markdown",
        "javascript",
    )

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None

    @abstractmethod
    async def initialize(self):
        """Initialize the provider"""
        pass

    @abstractmethod
    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response for the given request"""
        pass

    @abstractmethod
    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response for the given request"""
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get provider health status"""
        pass

    def _build_prompt(self, request: QueryRequest) -> str:
        """Build a provider prompt that includes context plus extracted attachment content."""
        prompt_sections = []

        response_style_instructions = (request.metadata or {}).get(
            "response_style_instructions"
        )
        if response_style_instructions:
            prompt_sections.append(f"Response style instructions:\n{response_style_instructions}")

        prompt = request.query
        attachment_prompt = self._build_attachment_prompt(request)
        if attachment_prompt:
            prompt = f"{prompt}\n\n{attachment_prompt}"
        if request.context:
            prompt_sections.append(f"Context: {request.context}")
        prompt_sections.append(f"Query: {prompt}")
        return "\n\n".join(prompt_sections)

    def _build_attachment_prompt(self, request: QueryRequest) -> str:
        """Render attachment metadata and extracted text into the provider prompt."""
        if not request.attachments:
            return ""

        sections = [
            "Use the following attachment data when it is relevant to the user's request:"
        ]
        remaining_chars = self.ATTACHMENT_PROMPT_CHAR_LIMIT

        for index, attachment in enumerate(request.attachments, start=1):
            header = (
                f"[Attachment {index}] {attachment.name} "
                f"({attachment.mime_type}, {attachment.size_bytes} bytes)"
            )
            sections.append(header)

            extracted_text = self._extract_attachment_text(attachment)
            if not extracted_text:
                sections.append(
                    "Binary attachment metadata only. No text could be extracted automatically."
                )
                continue

            excerpt_limit = min(self.ATTACHMENT_PREVIEW_CHAR_LIMIT, remaining_chars)
            if excerpt_limit <= 0:
                sections.append("Additional extracted text omitted because the prompt limit was reached.")
                break

            excerpt = extracted_text[:excerpt_limit]
            if len(extracted_text) > excerpt_limit:
                excerpt = f"{excerpt}\n...[truncated]"
            sections.append(f"Content excerpt:\n{excerpt}")
            remaining_chars -= len(excerpt)

        return "\n\n".join(sections)

    def _extract_attachment_text(self, attachment) -> Optional[str]:
        """Decode text-like attachment content into a prompt-safe string."""
        if not attachment.content or not self._is_text_attachment(attachment):
            return None

        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                decoded = attachment.content.decode(encoding)
            except UnicodeDecodeError:
                continue

            normalized = decoded.strip()
            if not normalized:
                return None

            printable = sum(
                1 for character in normalized if character.isprintable() or character in "\n\r\t"
            )
            if printable / max(len(normalized), 1) < 0.85:
                continue
            return normalized

        return None

    def _is_text_attachment(self, attachment) -> bool:
        """Return whether an attachment is safe to inline as text."""
        mime_type = (attachment.mime_type or "").lower()
        if attachment.type != AttachmentType.DOCUMENT:
            return False
        return mime_type.startswith("text/") or any(
            marker in mime_type for marker in self.TEXT_MIME_MARKERS
        )

    async def generate_batch_responses(
        self, requests: List[QueryRequest], model_name: str
    ) -> List[InferenceResponse]:
        """Generate responses for a batch of requests."""
        return await asyncio.gather(
            *(self.generate_response(request, model_name) for request in requests)
        )


class OpenAIProvider(BaseInferenceProvider):
    """OpenAI API provider implementation"""

    async def initialize(self):
        """Initialize OpenAI client"""
        self.client = openai.AsyncOpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
            timeout=self.config.get("timeout", 60),
        )
        logger.info("OpenAI provider initialized")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)
            response = await self.client.responses.create(
                model=model_name,
                input=prompt,
                instructions=None,
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                user=request.user_id,
            )

            usage = response.usage
            generated_text = response.output_text

            return InferenceResponse(
                response_text=generated_text,
                model_name=model_name,
                token_count_input=usage.input_tokens,
                token_count_output=usage.output_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_per_second=(
                    usage.output_tokens / max(time.time() - start_time, 1e-6)
                ),
                cost_usd=self._calculate_cost(usage, model_name),
                provider="openai",
                cached=False,
            )

        except Exception as e:
            logger.error(f"OpenAI inference error: {e}")
            raise

    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response using OpenAI API"""
        try:
            prompt = self._build_prompt(request)
            async with self.client.responses.stream(
                model=model_name,
                input=prompt,
                instructions=None,
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                user=request.user_id,
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"

    def _calculate_cost(self, usage, model_name: str) -> float:
        """Calculate cost based on usage"""
        # OpenAI pricing (example rates - update with actual rates)
        pricing = {
            "gpt-5": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
        }

        model_pricing = pricing.get(
            model_name, {"input": 0.001 / 1000, "output": 0.002 / 1000}
        )

        input_cost = usage.input_tokens * model_pricing["input"]
        output_cost = usage.output_tokens * model_pricing["output"]

        return input_cost + output_cost

    def get_health_status(self) -> Dict[str, Any]:
        """Get OpenAI provider health status"""
        return {
            "provider": "openai",
            "status": "healthy" if self.client else "unhealthy",
            "models_available": ["gpt-5", "gpt-5.4", "gpt-3.5-turbo"],
        }


class AnthropicProvider(BaseInferenceProvider):
    """Anthropic API provider implementation"""

    async def initialize(self):
        """Initialize Anthropic client"""
        self.client = anthropic.AsyncAnthropic(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url", "https://api.anthropic.com"),
            timeout=self.config.get("timeout", 60),
        )
        logger.info("Anthropic provider initialized")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response using Anthropic API"""
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)

            # Make API call
            response = await self.client.messages.create(
                model=model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract response data
            generated_text = response.content[0].text

            return InferenceResponse(
                response_text=generated_text,
                model_name=model_name,
                token_count_input=response.usage.input_tokens,
                token_count_output=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_per_second=(
                    response.usage.output_tokens / max(time.time() - start_time, 1e-6)
                ),
                cost_usd=self._calculate_cost(response.usage, model_name),
                provider="anthropic",
                cached=False,
            )

        except Exception as e:
            logger.error(f"Anthropic inference error: {e}")
            raise

    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response using Anthropic API"""
        try:
            prompt = self._build_prompt(request)

            async with self.client.messages.stream(
                model=model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            yield f"Error: {str(e)}"

    def _calculate_cost(self, usage, model_name: str) -> float:
        """Calculate cost based on usage"""
        # Anthropic pricing (example rates)
        pricing = {
            "claude-3.5-sonnet": {"input": 0.003 / 1000, "output": 0.015 / 1000},
            "claude-3-opus": {"input": 0.015 / 1000, "output": 0.075 / 1000},
        }

        model_pricing = pricing.get(
            model_name, {"input": 0.003 / 1000, "output": 0.015 / 1000}
        )

        input_cost = usage.input_tokens * model_pricing["input"]
        output_cost = usage.output_tokens * model_pricing["output"]

        return input_cost + output_cost

    def get_health_status(self) -> Dict[str, Any]:
        """Get Anthropic provider health status"""
        return {
            "provider": "anthropic",
            "status": "healthy" if self.client else "unhealthy",
            "models_available": ["claude-3.5-sonnet", "claude-3-opus"],
        }


class vLLMProvider(BaseInferenceProvider):
    """vLLM self-hosted provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = (
            f"http://{config.get('host', 'localhost')}:{config.get('port', 8000)}"
        )
        self.http_client = None

    async def initialize(self):
        """Initialize vLLM client"""
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.config.get(
                "timeout", 300
            ),  # Longer timeout for local inference
        )

        # Test connection
        try:
            response = await self.http_client.get("/health")
            if response.status_code == 200:
                logger.info("vLLM provider initialized successfully")
            else:
                logger.warning(f"vLLM health check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"vLLM initialization error: {e}")

    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def generate_response(
        self, request: QueryRequest, model_name: str
    ) -> InferenceResponse:
        """Generate response using vLLM"""
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)

            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False,
                "logprobs": None,
            }

            # Make request to vLLM server
            response = await self.http_client.post("/v1/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            generated_text = data["choices"][0]["text"]
            usage = data.get("usage", {})

            return InferenceResponse(
                response_text=generated_text,
                model_name=model_name,
                token_count_input=usage.get("prompt_tokens", 0),
                token_count_output=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
                or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_per_second=(
                    usage.get("completion_tokens", 0)
                    / max(time.time() - start_time, 1e-6)
                ),
                cost_usd=0.0,  # Self-hosted models have no API cost
                provider="vllm",
                cached=False,
            )

        except Exception as e:
            logger.error(f"vLLM inference error: {e}")
            raise

    async def stream_response(
        self, request: QueryRequest, model_name: str
    ) -> AsyncIterator[str]:
        """Stream response using vLLM"""
        try:
            prompt = self._build_prompt(request)

            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True,
            }

            async with self.http_client.stream(
                "POST", "/v1/completions", json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("text", "")
                            if delta:
                                yield delta

        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            yield f"Error: {str(e)}"

    def get_health_status(self) -> Dict[str, Any]:
        """Get vLLM provider health status"""
        return {
            "provider": "vllm",
            "status": "healthy" if self.http_client else "unhealthy",
            "base_url": self.base_url,
            "models_available": ["mistral-7b", "llama-3.1-70b"],  # Example models
        }


class BatchProcessor:
    """Batch processing for multiple inference requests"""

    @dataclass
    class CoalescedRequest:
        request: QueryRequest
        shared_future: asyncio.Future

    @dataclass
    class BatchBucket:
        provider: BaseInferenceProvider
        model_name: str
        entries: Dict[str, "BatchProcessor.CoalescedRequest"] = field(
            default_factory=dict
        )
        flush_task: Optional[asyncio.Task] = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.max_batch_size = config.get("max_batch_size", 32)
        self.max_wait_time = (
            config.get("max_wait_time_ms", 50) / 1000
        )  # Convert to seconds
        self.pending_requests: Dict[Tuple[int, str], BatchProcessor.BatchBucket] = {}
        self.inflight_requests: Dict[Tuple[int, str, str], asyncio.Future] = {}
        self.batch_lock = asyncio.Lock()

    async def add_request(
        self, request: QueryRequest, provider: BaseInferenceProvider, model_name: str
    ) -> InferenceResponse:
        """Add request to batch or process immediately"""
        if not self.enabled:
            return await provider.generate_response(request, model_name)

        loop = asyncio.get_running_loop()
        bucket_key = (id(provider), model_name)
        coalesce_key = self._build_coalesce_key(request, model_name)
        inflight_key = (id(provider), model_name, coalesce_key)

        async with self.batch_lock:
            shared_future = self.inflight_requests.get(inflight_key)
            if shared_future is None:
                bucket = self.pending_requests.get(bucket_key)
                if bucket is None:
                    bucket = self.BatchBucket(provider=provider, model_name=model_name)
                    self.pending_requests[bucket_key] = bucket

                entry = bucket.entries.get(coalesce_key)
                if entry is None:
                    entry = self.CoalescedRequest(
                        request=request.model_copy(deep=True),
                        shared_future=loop.create_future(),
                    )
                    bucket.entries[coalesce_key] = entry
                shared_future = entry.shared_future

                if len(bucket.entries) >= self.max_batch_size:
                    self._schedule_flush_locked(bucket_key, bucket, immediate=True)
                elif bucket.flush_task is None or bucket.flush_task.done():
                    self._schedule_flush_locked(bucket_key, bucket, immediate=False)

        response = await shared_future
        return response.model_copy(deep=True)

    def _schedule_flush_locked(
        self, bucket_key: Tuple[int, str], bucket: "BatchProcessor.BatchBucket",
        immediate: bool
    ):
        """Schedule a batch flush for a bucket."""
        if bucket.flush_task and not bucket.flush_task.done():
            if immediate:
                bucket.flush_task.cancel()
            else:
                return

        if immediate:
            bucket.flush_task = asyncio.create_task(self._flush_bucket(bucket_key))
        else:
            bucket.flush_task = asyncio.create_task(self._delayed_flush(bucket_key))

    async def _delayed_flush(self, bucket_key: Tuple[int, str]):
        """Flush a bucket after the configured wait time."""
        try:
            await asyncio.sleep(self.max_wait_time)
            await self._flush_bucket(bucket_key)
        except asyncio.CancelledError:
            return

    async def _flush_bucket(self, bucket_key: Tuple[int, str]):
        """Flush a coalesced batch to the provider."""
        async with self.batch_lock:
            bucket = self.pending_requests.pop(bucket_key, None)
            if bucket is None or not bucket.entries:
                return

            current_task = asyncio.current_task()
            if (
                bucket.flush_task
                and bucket.flush_task is not current_task
                and not bucket.flush_task.done()
            ):
                bucket.flush_task.cancel()
            bucket.flush_task = None
            entry_items = list(bucket.entries.items())
            inflight_keys = []
            for coalesce_key, entry in entry_items:
                inflight_key = (id(bucket.provider), bucket.model_name, coalesce_key)
                self.inflight_requests[inflight_key] = entry.shared_future
                inflight_keys.append(inflight_key)

        try:
            entries = [entry for _, entry in entry_items]
            requests = [entry.request for entry in entries]
            if len(requests) == 1:
                responses = [
                    await bucket.provider.generate_response(
                        requests[0], bucket.model_name
                    )
                ]
            else:
                responses = await bucket.provider.generate_batch_responses(
                    requests, bucket.model_name
                )

            if len(responses) != len(entries):
                raise ValueError("Batch provider returned an unexpected response count")

            for entry, response in zip(entries, responses):
                if entry.shared_future.done():
                    continue
                entry.shared_future.set_result(response)
        except Exception as exc:
            for entry in entries:
                if entry.shared_future.done():
                    continue
                entry.shared_future.set_exception(exc)
        finally:
            async with self.batch_lock:
                for inflight_key in inflight_keys:
                    self.inflight_requests.pop(inflight_key, None)

    def _build_coalesce_key(self, request: QueryRequest, model_name: str) -> str:
        """Build a stable signature for in-flight request coalescing."""
        attachment_signatures = []
        for attachment in request.attachments:
            attachment_signatures.append(
                {
                    "name": attachment.name,
                    "type": attachment.type.value,
                    "size_bytes": attachment.size_bytes,
                    "mime_type": attachment.mime_type,
                    "url": attachment.url,
                    "content_sha1": hashlib.sha1(attachment.content).hexdigest()
                    if attachment.content
                    else None,
                }
            )

        payload = {
            "model_name": model_name,
            "user_id": request.user_id,
            "user_tier": request.user_tier.value,
            "query": request.query,
            "context": request.context,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "priority": request.priority,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "metadata": request.metadata,
            "attachments": attachment_signatures,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    async def shutdown(self):
        """Flush pending coalesced requests during shutdown."""
        async with self.batch_lock:
            flush_tasks = []
            for bucket_key, bucket in list(self.pending_requests.items()):
                if bucket.flush_task and not bucket.flush_task.done():
                    bucket.flush_task.cancel()
                flush_tasks.append(asyncio.create_task(self._flush_bucket(bucket_key)))

        if flush_tasks:
            await asyncio.gather(*flush_tasks, return_exceptions=True)


class InferenceEngine:
    """Main inference engine coordinating all providers"""

    def __init__(
        self,
        config: Dict[str, Any],
        router: ModelRouter,
        event_producer: Optional[Any] = None,
    ):
        self.config = config
        self.router = router
        self.event_producer = event_producer
        self.providers = {}
        self.context_compressor = ContextCompressor(config.get("compression", {}))
        self.cache = ResponseCache(config.get("cache", {}))
        self.batch_processor = BatchProcessor(config.get("batching", {}))

        # Performance tracking
        self.inference_stats = {}

    async def initialize(self):
        """Initialize all providers and components"""
        logger.info("Initializing inference engine...")

        # Initialize context compressor
        await self.context_compressor.initialize()

        # Initialize cache
        await self.cache.initialize()

        # Initialize providers based on configuration
        if "openai" in self.config:
            self.providers["openai"] = OpenAIProvider(self.config["openai"])
            await self.providers["openai"].initialize()

        if "anthropic" in self.config:
            self.providers["anthropic"] = AnthropicProvider(self.config["anthropic"])
            await self.providers["anthropic"].initialize()

        if "vllm" in self.config:
            self.providers["vllm"] = vLLMProvider(self.config["vllm"])
            await self.providers["vllm"].initialize()

        logger.info(
            f"Inference engine initialized with {len(self.providers)} providers"
        )

    async def process_query(self, request: QueryRequest) -> InferenceResponse:
        """Process query through the complete inference pipeline"""
        start_time = time.time()

        # Generate request ID for tracking
        request_id = hashlib.md5(
            f"{request.user_id}:{request.query}:{time.time()}".encode()
        ).hexdigest()

        try:
            # Step 1: Route query to appropriate model
            routing_decision = await self.router.route_query(request)
            model_name = routing_decision.selected_model

            # Step 2: Check cache
            cache_key = self.cache.generate_cache_key(request, model_name)
            cached_response = await self.cache.get_cached_response(cache_key)

            if cached_response:
                logger.info(f"Cache hit for request {request_id}")
                cached_response["cached"] = True
                response = InferenceResponse(**cached_response)
                await self._publish_inference_completed_event(
                    request,
                    response,
                    routing_decision=routing_decision,
                )
                return response

            # Step 3: Context compression if needed
            compressed_context = False
            if (
                request.context
                and len(request.context) > self.context_compressor.max_context_tokens
            ):
                original_length = len(request.context)
                target_length = int(
                    original_length * self.context_compressor.compression_ratio
                )
                request.context = await self.context_compressor.compress_context(
                    request.context, target_length
                )
                compressed_context = True
                logger.info(
                    f"Compressed context from {original_length} to {len(request.context)} chars"
                )

            # Step 4: Get appropriate provider
            provider = self._get_provider_for_model(model_name)
            if not provider:
                raise ValueError(f"No provider available for model: {model_name}")

            # Step 5: Generate response
            response = await self.batch_processor.add_request(
                request, provider, model_name
            )
            response.compressed_context = compressed_context

            # Step 6: Cache response
            await self.cache.cache_response(
                cache_key,
                response.model_dump(mode="json"),
            )

            # Step 7: Update metrics and stats
            inference_time = time.time() - start_time
            await self._update_stats(model_name, response, inference_time)

            # Step 8: Update router stats
            self.router.update_model_stats(
                model_name=model_name, success=True, latency_ms=response.latency_ms
            )

            INFERENCE_METRICS.requests_total.labels(
                model=model_name, provider=response.provider
            ).inc()

            INFERENCE_METRICS.request_duration.labels(
                model=model_name, provider=response.provider
            ).observe(inference_time)

            await self._publish_inference_completed_event(
                request,
                response,
                routing_decision=routing_decision,
            )
            return response

        except Exception as e:
            # Update error metrics
            INFERENCE_METRICS.errors_total.labels(
                model=routing_decision.selected_model
                if "routing_decision" in locals()
                else "unknown",
                error_type=type(e).__name__,
            ).inc()

            # Update router stats
            if "routing_decision" in locals():
                self.router.update_model_stats(
                    model_name=routing_decision.selected_model,
                    success=False,
                    latency_ms=int((time.time() - start_time) * 1000),
                )

            logger.error(f"Inference failed for request {request_id}: {e}")

            # Return error response
            error_response = InferenceResponse(
                response_text=f"Error processing request: {str(e)}",
                model_name=routing_decision.selected_model
                if "routing_decision" in locals()
                else "unknown",
                token_count_input=0,
                token_count_output=0,
                total_tokens=0,
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_per_second=0.0,
                cost_usd=0.0,
                provider="error",
                cached=False,
                error=str(e),
            )
            await self._publish_inference_completed_event(
                request,
                error_response,
                routing_decision=routing_decision
                if "routing_decision" in locals()
                else None,
            )
            return error_response

    async def stream_query(self, request: QueryRequest) -> AsyncIterator[str]:
        """Stream query response"""
        try:
            # Route query
            routing_decision = await self.router.route_query(request)
            model_name = routing_decision.selected_model

            # Get provider
            provider = self._get_provider_for_model(model_name)
            if not provider:
                yield f"Error: No provider available for model: {model_name}"
                return

            # Stream response
            async for chunk in provider.stream_response(request, model_name):
                yield chunk

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {str(e)}"

    def _get_provider_for_model(
        self, model_name: str
    ) -> Optional[BaseInferenceProvider]:
        """Get the appropriate provider for a model"""
        # Get model configuration from router
        model_info = self.router.get_model_info(model_name)
        if not model_info:
            return None

        provider_name = model_info["config"]["provider"]
        return self.providers.get(provider_name)

    async def _publish_inference_completed_event(
        self,
        request: QueryRequest,
        response: InferenceResponse,
        routing_decision: Optional[Any] = None,
    ):
        """Best-effort publication of post-inference completion events."""
        if not self.event_producer:
            return

        publish_method = getattr(self.event_producer, "produce_inference_completed", None)
        if publish_method is None:
            return

        try:
            await publish_method(request, response, routing_decision)
        except Exception as exc:
            logger.warning(f"Failed to publish inference completion event: {exc}")

    async def _update_stats(
        self, model_name: str, response: InferenceResponse, inference_time: float
    ):
        """Update inference statistics"""
        if model_name not in self.inference_stats:
            self.inference_stats[model_name] = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_time": 0.0,
                "error_count": 0,
            }

        stats = self.inference_stats[model_name]
        stats["total_requests"] += 1
        stats["total_tokens"] += (
            response.token_count_input + response.token_count_output
        )
        stats["total_cost"] += response.cost_usd
        stats["total_time"] += inference_time

        if response.error:
            stats["error_count"] += 1

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        provider_health = {}
        for name, provider in self.providers.items():
            provider_health[name] = provider.get_health_status()

        return {
            "inference_engine": "healthy",
            "providers": provider_health,
            "cache_enabled": self.cache.enabled,
            "compression_enabled": self.context_compressor.config.get("enabled", True),
            "stats": self.inference_stats,
        }

    async def shutdown(self):
        """Shutdown the inference engine"""
        logger.info("Shutting down inference engine...")

        await self.batch_processor.shutdown()

        # Close HTTP clients
        for provider in self.providers.values():
            if hasattr(provider, "http_client") and provider.http_client:
                await provider.http_client.aclose()

        # Close Redis connection
        if self.cache.redis_client:
            await self.cache.redis_client.close()

        logger.info("Inference engine shutdown complete")
