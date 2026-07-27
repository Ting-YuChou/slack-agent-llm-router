import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.llm_router_part2_inference import (
    AnthropicProvider,
    InferenceEngine,
    OpenAIProvider,
    ResponseCache,
    SingleFlightCoordinator,
)
from src.rag.chunker import DocumentChunk
from src.rag.service import RagImageInput, RagService
from src.rag.storage import RagAssetRef
from src.rag.vector_store import InMemoryRagVectorStore, RagSearchResult
from src.utils.schema import QueryRequest


class AssetLoader:
    def __init__(self, payload=b"png-bytes"):
        self.payload = payload

    async def load(self, _ref):
        return self.payload


def _request():
    return QueryRequest(
        query="What does the chart show?",
        user_id="user-1",
        metadata={},
    )


def _image(checksum="checksum-a"):
    return RagImageInput(
        source_rank=1,
        figure_id="fig-1",
        page=7,
        caption="Revenue by year",
        asset_ref=RagAssetRef(
            backend="local",
            uri="rag-asset://kb/doc/v1/fig-1",
            path="/tmp/private.png",
            media_type="image/png",
            checksum_sha256=checksum,
            size_bytes=9,
            width=100,
            height=80,
            knowledge_base_id="kb",
            document_id="doc",
            index_version="v1",
            figure_id="fig-1",
        ),
        content=b"png-bytes",
    )


def test_openai_uses_responses_image_blocks_without_exposing_asset_uri():
    request = _request()
    request._rag_images = [_image()]
    kwargs = OpenAIProvider({"api_key": "test"})._build_response_request_kwargs(
        request, "gpt-5", "Answer with sources."
    )

    content = kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert "[S1]" in content[0]["text"]
    assert content[1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,"
        + base64.b64encode(b"png-bytes").decode("ascii"),
    }
    assert "rag-asset://" not in str(kwargs)


def test_anthropic_uses_labeled_base64_image_blocks():
    request = _request()
    request._rag_images = [_image()]
    content = AnthropicProvider({"api_key": "test"})._build_messages(request)[0][
        "content"
    ]

    assert content[0]["type"] == "text"
    assert "[S1]" in content[0]["text"]
    assert content[1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(b"png-bytes").decode("ascii"),
        },
    }


@pytest.mark.asyncio
async def test_direct_and_related_figures_are_ranked_deduplicated_and_materialized():
    service = RagService(
        {"enabled": True, "backend": "memory"},
        vector_store=InMemoryRagVectorStore(),
        asset_store=AssetLoader(),
    )
    figure = _image().asset_ref.to_dict()
    direct = DocumentChunk(
        chunk_id="figure",
        document_id="doc",
        text="Revenue chart",
        page_start=7,
        page_end=7,
        block_ids=["figure"],
        block_types=["figure"],
        metadata={
            "is_figure": True,
            "figure_id": "fig-1",
            "asset_ref": figure,
            "figure_caption": "Revenue by year",
        },
    )
    text = DocumentChunk(
        chunk_id="text",
        document_id="doc",
        text="Revenue improved.",
        page_start=6,
        page_end=6,
        block_ids=["text"],
        block_types=["text"],
        metadata={
            "related_figures": [
                {
                    "figure_id": "fig-1",
                    "page": 7,
                    "asset_ref": figure,
                    "visual": {"caption": "Revenue by year"},
                },
                {
                    "figure_id": "fig-2",
                    "page": 6,
                    "asset_ref": {
                        **figure,
                        "figure_id": "fig-2",
                        "uri": "rag-asset://kb/doc/v1/fig-2",
                    },
                },
            ]
        },
    )
    results = [
        RagSearchResult(direct, 0.9, "hybrid", "kb", "v1"),
        RagSearchResult(text, 0.8, "vector", "kb", "v1"),
    ]

    images, warnings = await service.materialize_image_inputs(results)

    assert [image.figure_id for image in images] == ["fig-1", "fig-2"]
    assert [image.source_rank for image in images] == [1, 2]
    assert warnings == []


def test_cache_key_changes_with_figure_checksum_and_never_contains_bytes():
    request_a = _request()
    request_b = _request()
    request_a._rag_images = [_image("checksum-a")]
    request_b._rag_images = [_image("checksum-b")]
    cache = ResponseCache({})

    key_a = cache.generate_cache_key(request_a, "gpt-5")
    key_b = cache.generate_cache_key(request_b, "gpt-5")

    assert key_a != key_b
    assert "png-bytes" not in key_a


def test_single_flight_key_changes_with_figure_checksum():
    request_a = _request()
    request_b = _request()
    request_a._rag_images = [_image("checksum-a")]
    request_b._rag_images = [_image("checksum-b")]
    coordinator = SingleFlightCoordinator({"enabled": True})

    assert coordinator._build_coalesce_key(
        request_a, "gpt-5"
    ) != coordinator._build_coalesce_key(request_b, "gpt-5")


def test_serialized_direct_and_related_figures_only_expose_opaque_refs():
    service = RagService(
        {"enabled": True, "backend": "memory"},
        vector_store=InMemoryRagVectorStore(),
    )
    private_ref = {
        **_image().asset_ref.to_dict(),
        "bucket": "private-bucket",
        "key": "rag/staging/assets/kb/doc/v1/figure.png",
        "path": "/private/figure.png",
    }
    chunk = DocumentChunk(
        chunk_id="figure",
        document_id="doc",
        text="Revenue chart",
        page_start=7,
        page_end=7,
        block_ids=["figure"],
        block_types=["figure"],
        metadata={
            "asset_ref": private_ref,
            "related_figures": [{"figure_id": "fig-2", "asset_ref": private_ref}],
        },
    )

    serialized = service.serialize_results(
        [RagSearchResult(chunk, 0.9, "hybrid", "kb", "v1")]
    )[0]
    metadata = serialized["metadata"]

    assert metadata["image_ref"].startswith("rag-asset://")
    assert metadata["related_figures"][0]["image_ref"].startswith("rag-asset://")
    assert "private-bucket" not in str(serialized)
    assert "/private/figure.png" not in str(serialized)


class VisionRag:
    visual_config = {"answer_images": {"enabled": True, "max_images": 3}}

    async def materialize_image_inputs(self, *_args, **_kwargs):
        return [_image()], []


class VisionRouter:
    def __init__(self, replacement):
        self.replacement = replacement

    def get_model_info(self, name):
        capabilities = ["general"] if name == "local" else ["general", "vision"]
        return {"config": {"capabilities": capabilities}}

    async def select_model_with_capability(self, *_args, **_kwargs):
        return self.replacement


@pytest.mark.asyncio
async def test_text_model_is_rerouted_to_policy_eligible_vision_model():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.rag_service = VisionRag()
    engine.router = VisionRouter("gpt-5")
    request = _request()
    decision = SimpleNamespace(selected_model="local", routing_reason="fast lane")

    rerouted = await engine._prepare_rag_images(
        request, decision, SimpleNamespace(results=[object()])
    )

    assert rerouted.selected_model == "gpt-5"
    assert request._rag_images[0].figure_id == "fig-1"


@pytest.mark.asyncio
async def test_no_vision_model_keeps_text_route_and_returns_warning():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.rag_service = VisionRag()
    engine.router = VisionRouter(None)
    request = _request()
    result = SimpleNamespace(results=[object()])
    decision = SimpleNamespace(selected_model="local", routing_reason="fast lane")

    rerouted = await engine._prepare_rag_images(request, decision, result)

    assert rerouted.selected_model == "local"
    assert request._rag_images == []
    assert result.warnings == ["rag_vision_fallback: no policy-eligible vision model"]


@pytest.mark.asyncio
async def test_failed_primary_vision_model_downgrades_to_available_text_model():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine._execute_model_request = AsyncMock(return_value=SimpleNamespace(warnings=[]))
    request = _request()
    request._rag_images = [_image()]
    request.metadata["rag_text_fallback_models"] = ["local-text"]
    decision = SimpleNamespace(selected_model="gpt-5", routing_reason="vision route")

    response, model_name, _ = await engine._execute_text_only_after_vision_failure(
        request, decision, execution_budget=None
    )

    assert model_name == "local-text"
    engine._execute_model_request.assert_awaited_once_with(
        request, "local-text", execution_budget=None
    )
    assert response.warnings == [
        "rag_vision_fallback: vision providers unavailable; used text only"
    ]
