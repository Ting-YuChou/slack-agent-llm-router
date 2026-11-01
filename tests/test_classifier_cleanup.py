"""The classifier loads no models and starts no tasks (R16c)."""

import asyncio
import builtins

import pytest

from src.llm_router_part1_router import ModelRouter, QueryClassifier
from src.utils.schema import QueryType, RouterConfig

HEAVY_MODULES = ("sentence_transformers", "sklearn")


@pytest.fixture
def no_heavy_imports(monkeypatch):
    real_import = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in HEAVY_MODULES:
            attempted.append(name)
            raise AssertionError(f"{name} must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return attempted


@pytest.mark.asyncio
async def test_initialize_loads_no_semantic_model_even_when_enabled(no_heavy_imports):
    classifier = QueryClassifier(semantic_enabled=True)

    await classifier.initialize()

    assert classifier._is_initialized is True
    assert classifier.model is None
    assert classifier.tokenizer is None
    assert no_heavy_imports == []


def test_classifying_before_initialize_needs_no_event_loop():
    classifier = QueryClassifier()

    assert classifier.classify_query("Translate this into German") == (
        QueryType.TRANSLATION,
        pytest.approx(0.775),
    )
    assert classifier._is_initialized is False


@pytest.mark.asyncio
async def test_classifying_before_initialize_starts_no_task(monkeypatch):
    def no_tasks(*_args, **_kwargs):
        raise AssertionError("classify_query must not start a task")

    monkeypatch.setattr(asyncio, "create_task", no_tasks)
    classifier = QueryClassifier()

    query_type, _confidence = classifier.classify_query("Summarize this thread")

    assert query_type == QueryType.SUMMARIZATION


@pytest.mark.asyncio
async def test_router_with_semantic_enabled_still_initializes_without_models(
    router_config, no_heavy_imports
):
    router = ModelRouter(
        {
            **router_config,
            "classifier": {"semantic_enabled": True},
            "tokenizer_enabled": False,
        }
    )

    assert router.is_healthy() is False
    await router.initialize()

    assert router.classifier._is_initialized is True
    assert router.classifier.model is None
    assert router.is_healthy() is True
    assert no_heavy_imports == []


@pytest.mark.parametrize("semantic_enabled", [True, False])
def test_configs_that_set_semantic_enabled_still_load(semantic_enabled):
    config = RouterConfig(classifier={"semantic_enabled": semantic_enabled})

    assert config.classifier.semantic_enabled is semantic_enabled
