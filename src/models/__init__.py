"""Models package for LLM Router Platform."""

from .mistral_engine import (
    BaseModelEngine,
    GenerationRequest,
    GenerationResponse,
    HuggingFaceEngine,
    ModelConfig as EngineModelConfig,
    ModelManager,
    RemoteModelEngine,
    create_codellama_config,
    create_llama_7b_config,
    create_mistral_7b_config,
    vLLMEngine,
)

__version__ = "1.0.0"
__description__ = "Model engines and inference abstractions"

__all__ = [
    "BaseModelEngine",
    "ModelManager",
    "EngineModelConfig",
    "GenerationRequest",
    "GenerationResponse",
    "vLLMEngine",
    "HuggingFaceEngine",
    "RemoteModelEngine",
    "create_mistral_7b_config",
    "create_llama_7b_config",
    "create_codellama_config",
]
