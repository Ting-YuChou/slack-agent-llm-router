"""
Model wrapper abstraction for vLLM and self-hosted models
Production-grade model engine with GPU optimization and batching
"""

import asyncio
import json
import logging
import time
import torch
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncIterator, Union, Tuple
import aiohttp
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading and inference"""

    model_name: str
    model_path: str
    device: str = "auto"
    dtype: str = "auto"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None
    trust_remote_code: bool = False
    enable_lora: bool = False
    lora_modules: Optional[Dict[str, str]] = None


@dataclass
class GenerationRequest:
    """Request for text generation"""

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    logprobs: Optional[int] = None


@dataclass
class GenerationResponse:
    """Response from text generation"""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    logprobs: Optional[List[Dict]] = None
    generation_time: float = 0.0
    tokens_per_second: float = 0.0


class BaseModelEngine(ABC):
    """Abstract base class for model engines"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    @abstractmethod
    async def load_model(self):
        """Load the model into memory"""
        pass

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt"""
        pass

    @abstractmethod
    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream text generation"""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload model from memory"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status"""
        pass


class vLLMEngine(BaseModelEngine):
    """vLLM-based model engine for high-performance inference"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.engine: Optional[AsyncLLMEngine] = None
        self.llm: Optional[LLM] = None
        self.generation_config = None

    async def load_model(self):
        """Load model using vLLM for optimized inference"""
        try:
            logger.info(f"Loading model {self.config.model_name} with vLLM...")

            # Check if model path exists
            model_path = Path(self.config.model_path)
            if not model_path.exists() and not self.config.model_path.startswith(
                "huggingface.co"
            ):
                raise FileNotFoundError(
                    f"Model path not found: {self.config.model_path}"
                )

            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=self.config.trust_remote_code
            )

            # Configure vLLM engine parameters
            engine_args = {
                "model": self.config.model_path,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "dtype": self.config.dtype,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "trust_remote_code": self.config.trust_remote_code,
                "disable_log_stats": False,
                "enforce_eager": False,  # Use CUDA graphs for better performance
            }

            # Add quantization if specified
            if self.config.quantization:
                engine_args["quantization"] = self.config.quantization

            # Add LoRA configuration if enabled
            if self.config.enable_lora and self.config.lora_modules:
                engine_args["enable_lora"] = True
                engine_args["lora_modules"] = self.config.lora_modules

            # Initialize vLLM engine
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            engine_args_obj = AsyncEngineArgs(**engine_args)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args_obj)

            # Also create synchronous LLM for non-streaming requests
            llm_args = engine_args.copy()
            llm_args.pop("disable_log_stats", None)
            self.llm = LLM(**llm_args)

            self.is_loaded = True
            logger.info(f"Model {self.config.model_name} loaded successfully with vLLM")

        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            self.is_loaded = False
            raise

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using vLLM engine"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop_sequences,
                logprobs=request.logprobs,
            )

            # Count input tokens
            input_tokens = len(self.tokenizer.encode(request.prompt))

            # Generate using vLLM
            if self.engine:
                # Use async engine for better concurrency
                results = await self.engine.generate(
                    request.prompt,
                    sampling_params,
                    request_id=f"req_{int(time.time() * 1000000)}",
                )
                result = results[0] if results else None
            else:
                # Fallback to synchronous LLM
                results = self.llm.generate([request.prompt], sampling_params)
                result = results[0] if results else None

            if not result:
                raise RuntimeError("No output generated")

            # Extract generated text
            output = result.outputs[0]
            generated_text = output.text

            # Calculate metrics
            generation_time = time.time() - start_time
            output_tokens = (
                len(output.token_ids)
                if output.token_ids
                else len(self.tokenizer.encode(generated_text))
            )
            total_tokens = input_tokens + output_tokens
            tokens_per_second = (
                output_tokens / generation_time if generation_time > 0 else 0
            )

            # Extract logprobs if requested
            logprobs_data = None
            if request.logprobs and output.logprobs:
                logprobs_data = [
                    {token_id: logprob for token_id, logprob in logprob_dict.items()}
                    for logprob_dict in output.logprobs
                ]

            return GenerationResponse(
                text=generated_text,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                finish_reason=output.finish_reason,
                logprobs=logprobs_data,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream text generation using vLLM engine"""
        if not self.is_loaded or not self.engine:
            raise RuntimeError("Model not loaded or async engine not available")

        try:
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop_sequences,
            )

            request_id = f"stream_{int(time.time() * 1000000)}"

            # Start streaming generation
            results_generator = self.engine.generate(
                request.prompt, sampling_params, request_id=request_id
            )

            async for request_output in results_generator:
                if request_output.outputs:
                    output = request_output.outputs[0]
                    if output.text:
                        yield output.text

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"

    def unload_model(self):
        """Unload model from memory"""
        try:
            if self.engine:
                # vLLM doesn't have explicit unload, but we can clear references
                self.engine = None

            if self.llm:
                self.llm = None

            self.tokenizer = None
            self.is_loaded = False

            # Force garbage collection
            import gc

            gc.collect()

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Model {self.config.model_name} unloaded")

        except Exception as e:
            logger.error(f"Error unloading model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status"""
        info = {
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "engine_type": "vllm",
            "is_loaded": self.is_loaded,
            "max_model_len": self.config.max_model_len,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
        }

        if self.is_loaded and torch.cuda.is_available():
            info.update(
                {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_allocated": torch.cuda.memory_allocated(),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(),
                }
            )

        return info


class HuggingFaceEngine(BaseModelEngine):
    """HuggingFace Transformers-based model engine"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = None
        self.generation_config = None

    async def load_model(self):
        """Load model using HuggingFace Transformers"""
        try:
            logger.info(f"Loading model {self.config.model_name} with HuggingFace...")

            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=self.config.trust_remote_code
            )

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": self._get_torch_dtype(),
                "device_map": "auto" if self.device == "cuda" else None,
            }

            # Add quantization if specified
            if self.config.quantization == "4bit":
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, **model_kwargs
            )

            # Move to device if not using device_map
            if model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.device)

            # Set up generation config
            self.generation_config = GenerationConfig.from_pretrained(
                self.config.model_path,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            self.is_loaded = True
            logger.info(
                f"Model {self.config.model_name} loaded successfully with HuggingFace"
            )

        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            self.is_loaded = False
            raise

    def _get_torch_dtype(self):
        """Get appropriate torch dtype"""
        if self.config.dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        elif self.config.dtype == "float16":
            return torch.float16
        elif self.config.dtype == "bfloat16":
            return torch.bfloat16
        elif self.config.dtype == "float32":
            return torch.float32
        else:
            return torch.float16

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using HuggingFace model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_model_len - request.max_tokens,
            ).to(self.device)

            input_length = inputs.input_ids.shape[1]

            # Set generation parameters
            generation_kwargs = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if request.stop_sequences:
                # Convert stop sequences to token IDs
                stop_token_ids = []
                for stop_seq in request.stop_sequences:
                    stop_tokens = self.tokenizer.encode(
                        stop_seq, add_special_tokens=False
                    )
                    stop_token_ids.extend(stop_tokens)
                generation_kwargs["eos_token_id"] = stop_token_ids

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            # Decode generated text
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Calculate metrics
            generation_time = time.time() - start_time
            prompt_tokens = input_length
            completion_tokens = len(generated_tokens)
            total_tokens = prompt_tokens + completion_tokens
            tokens_per_second = (
                completion_tokens / generation_time if generation_time > 0 else 0
            )

            # Determine finish reason
            finish_reason = "stop"
            if len(generated_tokens) >= request.max_tokens:
                finish_reason = "length"
            elif self.tokenizer.eos_token_id in generated_tokens:
                finish_reason = "stop"

            return GenerationResponse(
                text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream text generation using HuggingFace model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            # This is a simplified streaming implementation
            # For true streaming, you'd need to implement token-by-token generation
            response = await self.generate(request)

            # Simulate streaming by yielding chunks
            text = response.text
            chunk_size = max(1, len(text) // 10)  # 10 chunks

            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay to simulate streaming

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"

    def unload_model(self):
        """Unload model from memory"""
        try:
            if self.model:
                del self.model
                self.model = None

            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None

            self.generation_config = None
            self.is_loaded = False

            # Force garbage collection
            import gc

            gc.collect()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Model {self.config.model_name} unloaded")

        except Exception as e:
            logger.error(f"Error unloading model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status"""
        info = {
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "engine_type": "huggingface",
            "is_loaded": self.is_loaded,
            "device": self.device,
            "max_model_len": self.config.max_model_len,
        }

        if self.is_loaded and torch.cuda.is_available():
            info.update(
                {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_allocated": torch.cuda.memory_allocated(),
                    "gpu_memory_reserved": torch.cuda.memory_reserved(),
                }
            )

        return info


class RemoteModelEngine(BaseModelEngine):
    """Remote model engine for API-based models"""

    def __init__(
        self, config: ModelConfig, api_url: str, api_key: Optional[str] = None
    ):
        super().__init__(config)
        self.api_url = api_url
        self.api_key = api_key
        self.session = None

    async def load_model(self):
        """Initialize remote model connection"""
        try:
            logger.info(f"Connecting to remote model {self.config.model_name}...")

            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
                headers={"Authorization": f"Bearer {self.api_key}"}
                if self.api_key
                else {},
            )

            # Test connection
            async with self.session.get(f"{self.api_url}/health") as response:
                if response.status == 200:
                    self.is_loaded = True
                    logger.info(f"Connected to remote model {self.config.model_name}")
                else:
                    raise ConnectionError(f"Health check failed: {response.status}")

        except Exception as e:
            logger.error(
                f"Failed to connect to remote model {self.config.model_name}: {e}"
            )
            self.is_loaded = False
            raise

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using remote API"""
        if not self.is_loaded or not self.session:
            raise RuntimeError("Remote model not connected")

        start_time = time.time()

        try:
            payload = {
                "model": self.config.model_name,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "stop": request.stop_sequences,
                "stream": False,
            }

            async with self.session.post(
                f"{self.api_url}/v1/completions", json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()

            # Parse response
            choice = data["choices"][0]
            generated_text = choice["text"]

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            generation_time = time.time() - start_time
            tokens_per_second = (
                completion_tokens / generation_time if generation_time > 0 else 0
            )

            return GenerationResponse(
                text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=choice.get("finish_reason", "stop"),
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
            )

        except Exception as e:
            logger.error(f"Remote generation failed: {e}")
            raise

    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream text generation using remote API"""
        if not self.is_loaded or not self.session:
            raise RuntimeError("Remote model not connected")

        try:
            payload = {
                "model": self.config.model_name,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": True,
            }

            async with self.session.post(
                f"{self.api_url}/v1/completions", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("text", "")
                                if delta:
                                    yield delta
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Remote streaming failed: {e}")
            yield f"Error: {str(e)}"

    def unload_model(self):
        """Close remote connection"""
        try:
            if self.session:
                asyncio.create_task(self.session.close())
                self.session = None

            self.is_loaded = False
            logger.info(f"Disconnected from remote model {self.config.model_name}")

        except Exception as e:
            logger.error(f"Error disconnecting from remote model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get remote model information"""
        return {
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "engine_type": "remote",
            "is_loaded": self.is_loaded,
            "api_url": self.api_url,
        }


class ModelManager:
    """Manages multiple model engines and provides unified interface"""

    def __init__(self):
        self.engines: Dict[str, BaseModelEngine] = {}
        self.model_configs: Dict[str, ModelConfig] = {}

    def register_model(
        self, model_name: str, config: ModelConfig, engine_type: str = "vllm", **kwargs
    ):
        """Register a new model with the manager"""
        try:
            if engine_type == "vllm":
                engine = vLLMEngine(config)
            elif engine_type == "huggingface":
                engine = HuggingFaceEngine(config)
            elif engine_type == "remote":
                api_url = kwargs.get("api_url")
                api_key = kwargs.get("api_key")
                if not api_url:
                    raise ValueError("api_url required for remote engine")
                engine = RemoteModelEngine(config, api_url, api_key)
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")

            self.engines[model_name] = engine
            self.model_configs[model_name] = config

            logger.info(f"Registered model {model_name} with {engine_type} engine")

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise

    async def load_model(self, model_name: str):
        """Load a specific model"""
        if model_name not in self.engines:
            raise ValueError(f"Model {model_name} not registered")

        engine = self.engines[model_name]
        if not engine.is_loaded:
            await engine.load_model()

    async def unload_model(self, model_name: str):
        """Unload a specific model"""
        if model_name in self.engines:
            engine = self.engines[model_name]
            engine.unload_model()

    async def generate(
        self, model_name: str, request: GenerationRequest
    ) -> GenerationResponse:
        """Generate text using specified model"""
        if model_name not in self.engines:
            raise ValueError(f"Model {model_name} not registered")

        engine = self.engines[model_name]
        if not engine.is_loaded:
            await engine.load_model()

        return await engine.generate(request)

    async def stream_generate(
        self, model_name: str, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Stream text generation using specified model"""
        if model_name not in self.engines:
            raise ValueError(f"Model {model_name} not registered")

        engine = self.engines[model_name]
        if not engine.is_loaded:
            await engine.load_model()

        async for chunk in engine.stream_generate(request):
            yield chunk

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return [name for name, engine in self.engines.items() if engine.is_loaded]

    def get_all_models(self) -> List[str]:
        """Get list of all registered models"""
        return list(self.engines.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.engines:
            return self.engines[model_name].get_model_info()
        return None

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "total_models": len(self.engines),
            "loaded_models": len(self.get_loaded_models()),
            "models": {
                name: engine.get_model_info() for name, engine in self.engines.items()
            },
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "total_gpu_memory": [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    ],
                    "gpu_memory_allocated": [
                        torch.cuda.memory_allocated(i)
                        for i in range(torch.cuda.device_count())
                    ],
                    "gpu_memory_reserved": [
                        torch.cuda.memory_reserved(i)
                        for i in range(torch.cuda.device_count())
                    ],
                }
            )
        else:
            info["cuda_available"] = False

        return info

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models"""
        health_status = {"overall_healthy": True, "models": {}}

        for model_name, engine in self.engines.items():
            try:
                model_info = engine.get_model_info()
                is_healthy = engine.is_loaded

                health_status["models"][model_name] = {
                    "healthy": is_healthy,
                    "loaded": engine.is_loaded,
                    "engine_type": model_info.get("engine_type", "unknown"),
                    "last_check": time.time(),
                }

                if not is_healthy:
                    health_status["overall_healthy"] = False

            except Exception as e:
                health_status["models"][model_name] = {
                    "healthy": False,
                    "error": str(e),
                    "last_check": time.time(),
                }
                health_status["overall_healthy"] = False

        return health_status

    async def shutdown(self):
        """Shutdown all models and cleanup"""
        logger.info("Shutting down model manager...")

        for model_name, engine in self.engines.items():
            try:
                engine.unload_model()
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")

        self.engines.clear()
        self.model_configs.clear()

        logger.info("Model manager shutdown complete")


# Factory functions for easy model creation
def create_mistral_7b_config(
    model_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> ModelConfig:
    """Create configuration for Mistral 7B model"""
    return ModelConfig(
        model_name="mistral-7b",
        model_path=model_path,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        quantization=None,
        trust_remote_code=False,
    )


def create_llama_7b_config(
    model_path: str = "meta-llama/Llama-2-7b-chat-hf",
) -> ModelConfig:
    """Create configuration for LLaMA 7B model"""
    return ModelConfig(
        model_name="llama-7b",
        model_path=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        quantization=None,
        trust_remote_code=False,
    )


def create_codellama_config(
    model_path: str = "codellama/CodeLlama-7b-Instruct-hf",
) -> ModelConfig:
    """Create configuration for CodeLlama model"""
    return ModelConfig(
        model_name="codellama-7b",
        model_path=model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        quantization=None,
        trust_remote_code=False,
    )


# Example usage
async def main():
    """Example usage of the model manager"""
    # Create model manager
    manager = ModelManager()

    # Register models
    mistral_config = create_mistral_7b_config()
    manager.register_model("mistral-7b", mistral_config, engine_type="vllm")

    # Load model
    await manager.load_model("mistral-7b")

    # Create generation request
    request = GenerationRequest(
        prompt="What is machine learning?", max_tokens=512, temperature=0.7, top_p=0.95
    )

    # Generate response
    response = await manager.generate("mistral-7b", request)
    print(f"Generated: {response.text}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Speed: {response.tokens_per_second:.2f} tokens/sec")

    # Stream generation
    print("\nStreaming generation:")
    async for chunk in manager.stream_generate("mistral-7b", request):
        print(chunk, end="", flush=True)

    # Get system info
    system_info = manager.get_system_info()
    print(f"\nSystem info: {system_info}")

    # Cleanup
    await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
