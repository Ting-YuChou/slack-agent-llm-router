import base64
import hashlib

import pytest

from src.memory import HashEmbeddingProvider
from src.rag.chunker import DocumentChunk
from src.rag.parser import DocumentBlock, ParsedDocument
from src.rag.service import IngestionJob, RagService
from src.rag.storage import RagAssetRef
from src.rag.vector_store import InMemoryRagVectorStore
from src.rag.visual import VisualFigureResult


class FigureParser:
    def parse_bytes(self, **kwargs):
        document_id = kwargs["document_id"]
        return ParsedDocument(
            document_id=document_id,
            filename=kwargs["filename"],
            content_hash="hash",
            blocks=[
                DocumentBlock(
                    doc_id=document_id,
                    page=4,
                    bbox=[10, 20, 300, 220],
                    block_type="figure",
                    reading_order=1,
                    metadata={
                        "section_path": ["Capital"],
                        "figure": {"figure_id": "figure-1"},
                    },
                )
            ],
        )


class BytesCropper:
    def crop_figures(self, *, parsed_document, **_kwargs):
        figure = parsed_document.blocks[0].metadata["figure"]
        figure.update(
            {
                "_image_bytes": b"\x89PNG\r\n\x1a\nchart",
                "media_type": "image/png",
                "width": 640,
                "height": 480,
                "crop_status": "completed",
            }
        )
        return []


class CaptionProcessor:
    async def process_figure(self, figure):
        assert figure["_image_bytes"].startswith(b"\x89PNG")
        return VisualFigureResult(
            ocr_text="FY24 14.6 FY25 15.0",
            caption="CET1 ratio increased in FY25",
            ocr_provider="ocr",
            caption_provider="vlm",
        )


class RecordingAssetStore:
    def __init__(self):
        self.refs = []
        self.activated = []
        self.deleted = []
        self.cleaned = []

    async def put(self, **kwargs):
        content = kwargs["content"]
        ref = RagAssetRef(
            backend="s3",
            uri=("rag-asset://bank/doc-figure/generation-1/" "figures/figure-1.png"),
            bucket="bucket",
            key="rag/test/assets/bank/doc-figure/generation-1/figures/figure-1.png",
            version_id="version-1",
            media_type=kwargs["media_type"],
            checksum_sha256=base64.b64encode(hashlib.sha256(content).digest()).decode(
                "ascii"
            ),
            size_bytes=len(content),
            width=kwargs["width"],
            height=kwargs["height"],
            knowledge_base_id=kwargs["knowledge_base_id"],
            document_id=kwargs["document_id"],
            index_version=kwargs["index_version"],
            figure_id=kwargs["figure_id"],
        )
        self.refs.append(ref)
        return ref

    async def activate(self, ref):
        self.activated.append(ref)

    async def delete_other_generations(
        self, document_id, knowledge_base_id, active_version
    ):
        self.cleaned.append((document_id, knowledge_base_id, active_version))

    async def delete_document(self, document_id, knowledge_base_id):
        self.deleted.append((document_id, knowledge_base_id))


@pytest.mark.asyncio
async def test_ingestion_stores_captioned_figure_asset_then_activates_after_index():
    store = InMemoryRagVectorStore()
    assets = RecordingAssetStore()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"provider": "hash", "dimensions": 8},
            "visual": {
                "enabled": True,
                "ocr": {"enabled": True},
                "caption": {"enabled": True},
            },
        },
        parser=FigureParser(),
        vector_store=store,
        embedding_provider=HashEmbeddingProvider(dimensions=8),
        figure_cropper=BytesCropper(),
        visual_processor=CaptionProcessor(),
        asset_store=assets,
    )

    job = await service.process_ingestion_job(
        "job-figure",
        content=b"%PDF",
        filename="report.pdf",
        knowledge_base_id="bank",
        document_id="doc-figure",
    )

    assert job.status == "completed"
    assert len(assets.refs) == 1
    assert assets.activated == assets.refs
    chunk = next(iter(store.items.values())).chunk
    assert chunk.metadata["is_figure"] is True
    assert chunk.metadata["image_ref"].startswith("rag-asset://")
    assert chunk.metadata["asset_ref"]["version_id"] == "version-1"
    assert "visual_embedding" not in chunk.metadata
    assert "_image_bytes" not in chunk.metadata["figure_sidecar"]


@pytest.mark.asyncio
async def test_delete_document_removes_durable_figure_assets():
    assets = RecordingAssetStore()
    service = RagService(
        {"enabled": True, "backend": "memory"},
        vector_store=InMemoryRagVectorStore(),
        asset_store=assets,
    )

    await service.delete_document("doc-figure", "bank")

    assert assets.deleted == [("doc-figure", "bank")]


@pytest.mark.asyncio
async def test_delete_document_without_kb_removes_assets_from_every_indexed_kb():
    assets = RecordingAssetStore()
    vector_store = InMemoryRagVectorStore()
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-figure",
        text="figure",
        page_start=1,
        page_end=1,
        block_ids=["block"],
        block_types=["text"],
    )
    for kb_id in ("bank-a", "bank-b"):
        await vector_store.upsert_chunks(
            [chunk],
            [[0.1]],
            knowledge_base_id=kb_id,
            index_version=f"version-{kb_id}",
        )
    service = RagService(
        {"enabled": True, "backend": "memory"},
        vector_store=vector_store,
        asset_store=assets,
    )

    await service.delete_document("doc-figure")

    assert assets.deleted == [
        ("doc-figure", "bank-a"),
        ("doc-figure", "bank-b"),
    ]


@pytest.mark.asyncio
async def test_unscoped_delete_can_retry_asset_failure_before_vector_mapping_is_lost():
    class FlakyDeleteStore(RecordingAssetStore):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        async def delete_document(self, document_id, knowledge_base_id):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary S3 failure")
            await super().delete_document(document_id, knowledge_base_id)

    assets = FlakyDeleteStore()
    vector_store = InMemoryRagVectorStore()
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-figure",
        text="figure",
        page_start=1,
        page_end=1,
        block_ids=["block"],
        block_types=["text"],
    )
    await vector_store.upsert_chunks(
        [chunk], [[0.1]], knowledge_base_id="bank", index_version="v1"
    )
    service = RagService(
        {"enabled": True, "backend": "memory"},
        vector_store=vector_store,
        asset_store=assets,
    )

    with pytest.raises(RuntimeError, match="temporary S3 failure"):
        await service.delete_document("doc-figure")
    assert await vector_store.knowledge_base_ids_for_document("doc-figure") == ["bank"]

    deleted = await service.delete_document("doc-figure")

    assert deleted == 1
    assert assets.deleted == [("doc-figure", "bank")]


@pytest.mark.asyncio
async def test_activation_retry_reuses_committed_versioned_asset_reference():
    class FlakyActivationStore(RecordingAssetStore):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        async def activate(self, ref):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary tag failure")
            await super().activate(ref)

    assets = FlakyActivationStore()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"provider": "hash", "dimensions": 8},
            "visual": {"enabled": True},
        },
        parser=FigureParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=8),
        figure_cropper=BytesCropper(),
        visual_processor=CaptionProcessor(),
        asset_store=assets,
    )

    failed = await service.process_ingestion_job(
        "job-retry",
        content=b"%PDF",
        filename="report.pdf",
        knowledge_base_id="bank",
        document_id="doc-figure",
        raise_on_error=False,
    )
    failed_status = failed.status
    recovered = await service.process_ingestion_job(
        "job-retry",
        content=b"%PDF",
        filename="report.pdf",
        knowledge_base_id="bank",
        document_id="doc-figure",
    )

    assert failed_status == "failed"
    assert recovered.status == "completed"
    assert len(assets.refs) == 1
    assert assets.activated == assets.refs
    assert assets.cleaned == [("doc-figure", "bank", recovered.index_version)]


@pytest.mark.asyncio
async def test_detected_figure_labels_are_indexed_for_text_retrieval():
    class LabelOnlyProcessor:
        async def process_figure(self, figure):
            assert figure["_image_bytes"]
            return VisualFigureResult(
                detected_elements=[
                    {
                        "label": "axis",
                        "text": "FY25 CET1 15.0",
                        "bbox": [1, 2, 3, 4],
                    }
                ]
            )

    vector_store = InMemoryRagVectorStore()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"provider": "hash", "dimensions": 8},
            "retrieval": {
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
            "visual": {"enabled": True},
        },
        parser=FigureParser(),
        vector_store=vector_store,
        embedding_provider=HashEmbeddingProvider(dimensions=8),
        figure_cropper=BytesCropper(),
        visual_processor=LabelOnlyProcessor(),
        asset_store=RecordingAssetStore(),
    )
    await service.process_ingestion_job(
        "job-labels",
        content=b"%PDF",
        filename="report.pdf",
        knowledge_base_id="bank",
        document_id="doc-labels",
    )

    results = await service.retrieve("FY25 CET1", knowledge_base_ids=["bank"])

    assert results
    assert "Detected labels: axis | FY25 CET1 15.0" in results[0].chunk.text
    assert "bbox" not in results[0].chunk.text


def test_public_job_serialization_omits_internal_asset_refs():
    job = IngestionJob(
        job_id="job",
        document_id="doc",
        filename="report.pdf",
        knowledge_base_id="bank",
        asset_refs=[
            {
                "uri": "rag-asset://bank/doc/v1/figures/figure.png",
                "bucket": "private-bucket",
                "key": "rag/staging/assets/bank/doc/v1/figure.png",
                "version_id": "secret-version",
            }
        ],
    )

    assert "asset_refs" not in job.to_public_dict()
    assert job.to_dict()["asset_refs"][0]["bucket"] == "private-bucket"
