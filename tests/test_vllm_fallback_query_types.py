"""vLLM fallback query types follow the classifier's labels (R13 review).

The classifier never outputs question_answering; it now outputs translation,
which used to be classified general and so was fallback-eligible.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine
from src.utils.schema import ProviderModelFallbackConfig, QueryType

from test_routing_entitlement import (
    LOCAL_VLLM_ROUTER_CONFIG,
    VLLM_MODEL_FALLBACK_CONFIG,
    _request,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEFAULT = ["general", "code_generation", "code_analysis", "translation"]


def test_schema_default_matches_the_classifier_labels():
    assert ProviderModelFallbackConfig().allowed_query_types == EXPECTED_DEFAULT


def test_checked_in_config_allows_translation_not_question_answering():
    config = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    allowed = config["inference"]["vllm"]["model_fallback"]["allowed_query_types"]

    assert "translation" in allowed
    assert "question_answering" not in allowed


@pytest.mark.parametrize(
    ("query_type", "allowed"),
    [
        (QueryType.TRANSLATION, True),
        (QueryType.GENERAL, True),
        (QueryType.QUESTION_ANSWERING, False),
    ],
)
def test_inference_default_allowed_query_types_match_the_schema(query_type, allowed):
    router = ModelRouter(LOCAL_VLLM_ROUTER_CONFIG)
    engine = InferenceEngine(VLLM_MODEL_FALLBACK_CONFIG, router)
    request = _request("premium", query="Explain this", max_tokens=256)
    decision = SimpleNamespace(
        selected_model="qwen3.6-27b-fast", query_type=query_type, token_count=64
    )
    context = engine._build_local_fallback_context(request, decision)
    context["query_type"] = query_type.value
    # A policy that leaves allowed_query_types out gets the default.
    policy = {
        key: value
        for key, value in engine._vllm_model_fallback_config().items()
        if key != "allowed_query_types"
    }

    assert (
        engine._vllm_model_fallback_allowed(policy, request, decision, context)
        is allowed
    )
