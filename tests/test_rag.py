import base64

import pytest

from src.memory import HashEmbeddingProvider
from src.rag.chunker import DocumentChunk, StructureAwareChunker
from src.rag.parser import (
    DoclingParser,
    DocumentBlock,
    ParsedDocument,
    TextDocumentParser,
)
from src.rag.service import RagService
from src.rag.vector_store import (
    InMemoryRagVectorStore,
    RagSearchResult,
    RedisStackRagVectorStore,
)
from src.rag.visual import VisualFigureResult


class FailingEmbeddingProvider:
    async def embed(self, _text):
        raise RuntimeError("embedding endpoint down")


class FailingParser:
    def parse_bytes(self, **_kwargs):
        raise RuntimeError("parser exploded")


class ReverseReranker:
    async def rerank(self, _query, results):
        return list(reversed(results))


class FailingReranker:
    async def rerank(self, _query, _results):
        raise RuntimeError("reranker unavailable")


class FigureParser:
    def parse_bytes(self, **kwargs):
        document_id = kwargs.get("document_id") or "doc-figure"
        return ParsedDocument(
            document_id=document_id,
            filename=kwargs.get("filename", "figures.pdf"),
            content_hash="hash",
            blocks=[
                DocumentBlock(
                    doc_id=document_id,
                    page=2,
                    bbox=[10, 20, 110, 120],
                    block_type="figure",
                    text="",
                    reading_order=1,
                    asset_ref="#/pictures/0",
                    metadata={
                        "figure": {
                            "figure_id": "pictures-0",
                            "page": 2,
                            "bbox": [10, 20, 110, 120],
                            "asset_ref": "#/pictures/0",
                        }
                    },
                )
            ],
        )


class FakeFigureCropper:
    def __init__(self):
        self.deleted = []

    def crop_figures(self, *, parsed_document, knowledge_base_id, **_kwargs):
        for block in parsed_document.blocks:
            if block.block_type != "figure":
                continue
            figure = block.metadata.setdefault("figure", {})
            figure.update(
                {
                    "figure_id": figure.get("figure_id") or "pictures-0",
                    "image_ref": (
                        "data/rag/assets/"
                        f"{knowledge_base_id}/{parsed_document.document_id}/figures/pictures-0.png"
                    ),
                    "content_hash": "image-hash",
                    "width": 320,
                    "height": 240,
                    "crop_scope": "bbox",
                    "crop_status": "completed",
                }
            )
        return []

    def delete_document_assets(self, document_id, knowledge_base_id):
        self.deleted.append((document_id, knowledge_base_id))


class FakeVisualProcessor:
    def __init__(self, *, include_text=True):
        self.include_text = include_text

    async def process_figure(self, _figure):
        if not self.include_text:
            return VisualFigureResult(
                visual_embedding=[1.0, 0.0],
                visual_embedding_provider="fake_visual",
            )
        return VisualFigureResult(
            ocr_text="Registration flow text inside the diagram.",
            caption="A diagram explaining registration steps.",
            diagram_summary="Students select classes, confirm enrollment, then pay fees.",
            chart_summary="",
            visual_embedding=[1.0, 0.0],
            ocr_provider="fake_ocr",
            caption_provider="fake_caption",
            visual_embedding_provider="fake_visual",
        )

    async def embed_query(self, _query):
        return [1.0, 0.0]


class FakeStreamRedis:
    def __init__(self):
        self.values = {}
        self.sets = {}
        self.streams = {}
        self.groups = set()
        self.acked = []
        self.zsets = {}
        self.expirations = {}
        self.xadd_kwargs = []

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, _ttl, value):
        self.values[key] = value

    async def delete(self, key):
        self.values.pop(key, None)
        self.sets.pop(key, None)

    async def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    async def expire(self, key, ttl):
        self.expirations[key] = ttl
        return True

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def xgroup_create(self, stream, group, id="0", mkstream=True):
        self.groups.add((stream, group))
        self.streams.setdefault(stream, [])

    async def xadd(self, stream, fields, **kwargs):
        self.xadd_kwargs.append((stream, kwargs))
        entries = self.streams.setdefault(stream, [])
        message_id = f"{len(entries) + 1}-0"
        entries.append((message_id, dict(fields)))
        return message_id

    async def xack(self, stream, group, message_id):
        self.acked.append((stream, group, message_id))
        return 1

    async def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)

    async def zrangebyscore(self, key, minimum, maximum):
        values = self.zsets.get(key, {})
        return [
            member
            for member, score in values.items()
            if float(minimum) <= float(score) <= float(maximum)
        ]

    async def zrem(self, key, member):
        self.zsets.setdefault(key, {}).pop(member, None)

    async def xreadgroup(self, group, consumer, streams, count=1, block=0):
        stream = next(iter(streams.keys()))
        entries = self.streams.get(stream, [])[:count]
        return [(stream, entries)] if entries else []

    async def xautoclaim(
        self, stream, group, consumer, min_idle_time, start_id, count=1
    ):
        return ("0-0", self.streams.get(stream, [])[:count], [])


class FakeSearchRedis:
    def __init__(self):
        self.commands = []

    async def execute_command(self, *command):
        self.commands.append(command)
        query = command[2]
        if "@text:" in query:
            return [0]
        return [
            1,
            b"rag:chunk:school:chunk-1",
            [
                b"knowledge_base_id",
                b"school",
                b"document_id",
                b"doc-1",
                b"chunk_id",
                b"chunk-1",
                b"index_version",
                b"v1",
                b"text",
                b"Tuition fallback keyword",
                b"metadata",
                b"{}",
                b"section_path",
                b"[]",
                b"block_ids",
                b"[]",
                b"block_types",
                b"text",
                b"bboxes",
                b"[]",
                b"page_start",
                b"1",
                b"page_end",
                b"1",
                b"created_at",
                b"1",
                b"updated_at",
                b"1",
                b"embedding",
                b"",
            ],
        ]


def _queued_rag_service(tmp_path, *, parser=None, max_attempts=3):
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "ingestion_queue": {
                "enabled": True,
                "max_attempts": max_attempts,
                "retry_backoff_seconds": 0,
            },
            "storage": {"staging_dir": str(tmp_path / "rag_uploads")},
            "retrieval": {
                "top_k": 5,
                "candidate_count": 10,
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
        },
        parser=parser or TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    service.vector_store.client = FakeStreamRedis()
    return service


def _raw_redis_chunk_fields(
    *,
    chunk_id=b"chunk-1",
    document_id=b"doc-1",
    text=b"Tuition deadline",
    index_version=b"v1",
):
    return [
        b"knowledge_base_id",
        b"school",
        b"document_id",
        document_id,
        b"chunk_id",
        chunk_id,
        b"index_version",
        index_version,
        b"text",
        text,
        b"metadata",
        b'{"filename": "handbook.pdf"}',
        b"section_path",
        b"[]",
        b"block_ids",
        b"[]",
        b"block_types",
        b"text",
        b"bboxes",
        b"[]",
        b"page_start",
        b"1",
        b"page_end",
        b"1",
        b"created_at",
        b"1",
        b"updated_at",
        b"1",
        b"embedding",
        b"",
    ]


def test_chunker_preserves_layout_and_section_metadata():
    document = ParsedDocument(
        document_id="doc-1",
        filename="handbook.pdf",
        content_hash="hash",
        metadata={"title": "School Handbook"},
        blocks=[
            DocumentBlock(
                doc_id="doc-1",
                page=2,
                bbox=[0, 0, 100, 20],
                block_type="section_header",
                text="Tuition Deadlines",
                reading_order=1,
            ),
            DocumentBlock(
                doc_id="doc-1",
                page=2,
                bbox=[0, 30, 100, 80],
                block_type="text",
                text="Tuition payment is due on May 1.",
                reading_order=2,
            ),
        ],
    )

    chunks = StructureAwareChunker({"chunk_size_tokens": 50}).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].page_start == 2
    assert chunks[0].page_end == 2
    assert chunks[0].section_path == ["Tuition Deadlines"]
    assert chunks[0].bboxes == [[0, 0, 100, 20], [0, 30, 100, 80]]
    assert chunks[0].metadata["title"] == "School Handbook"


def test_chunker_emits_table_summary_and_row_chunks():
    table = {
        "table_id": "table-1",
        "columns": ["Item", "Fall '25", "Winter '26", "Spring '26"],
        "header_rows": [0],
        "rows": [
            {
                "row_id": "r1",
                "row_index": 1,
                "label": "UNDERGRAD/GRAD GRADE CHANGE OPTION",
                "values": {
                    "Fall '25": "Nov 30 Sun",
                    "Winter '26": "Mar 6 Fri",
                    "Spring '26": "May 29 Fri",
                },
                "cells": [],
                "semantic_text": (
                    "UNDERGRAD/GRAD GRADE CHANGE OPTION: "
                    "Fall '25 = Nov 30 Sun; Winter '26 = Mar 6 Fri; "
                    "Spring '26 = May 29 Fri"
                ),
            }
        ],
        "markdown": (
            "Item | Fall '25 | Winter '26 | Spring '26\n"
            "UNDERGRAD/GRAD GRADE CHANGE OPTION | Nov 30 Sun | "
            "Mar 6 Fri | May 29 Fri"
        ),
    }
    chunks = StructureAwareChunker({"chunk_size_tokens": 20}).chunk(
        ParsedDocument(
            document_id="doc-1",
            filename="calendar.pdf",
            content_hash="hash",
            blocks=[
                DocumentBlock(
                    doc_id="doc-1",
                    page=1,
                    bbox=None,
                    block_type="section_header",
                    text="Academic Calendar",
                    reading_order=1,
                ),
                DocumentBlock(
                    doc_id="doc-1",
                    page=1,
                    bbox=[0, 0, 100, 100],
                    block_type="table",
                    text=table["markdown"],
                    reading_order=2,
                    metadata={"table": table},
                ),
            ],
        )
    )

    table_chunks = [
        chunk for chunk in chunks if chunk.metadata.get("table_id") == "table-1"
    ]
    assert len(table_chunks) == 2
    assert table_chunks[0].metadata["is_table_summary"] is True
    assert table_chunks[1].metadata["is_table_row"] is True
    assert table_chunks[1].metadata["table_row_values"]["Fall '25"] == "Nov 30 Sun"
    assert "Winter '26 = Mar 6 Fri" in table_chunks[1].text
    assert table_chunks[1].section_path == ["Academic Calendar"]


def test_large_table_chunks_preserve_row_column_relationships():
    rows = []
    for index in range(1, 8):
        rows.append(
            {
                "row_id": f"r{index}",
                "row_index": index,
                "label": f"Deadline {index}",
                "values": {"Fall '25": f"Nov {index}", "Winter '26": f"Mar {index}"},
                "cells": [],
                "semantic_text": (
                    f"Deadline {index}: Fall '25 = Nov {index}; "
                    f"Winter '26 = Mar {index}"
                ),
            }
        )
    table = {
        "table_id": "table-large",
        "columns": ["Item", "Fall '25", "Winter '26"],
        "header_rows": [0],
        "rows": rows,
        "markdown": "Item | Fall '25 | Winter '26",
    }

    chunks = StructureAwareChunker({"chunk_size_tokens": 5}).chunk(
        ParsedDocument(
            document_id="doc-1",
            filename="calendar.pdf",
            content_hash="hash",
            blocks=[
                DocumentBlock(
                    doc_id="doc-1",
                    page=1,
                    bbox=None,
                    block_type="table",
                    text=table["markdown"],
                    reading_order=1,
                    metadata={"table": table},
                )
            ],
        )
    )

    row_chunks = [chunk for chunk in chunks if chunk.metadata.get("is_table_row")]
    assert len(row_chunks) == 7
    assert all("Fall '25 =" in chunk.text for chunk in row_chunks)
    assert all("Winter '26 =" in chunk.text for chunk in row_chunks)


def test_chunker_emits_image_chunk_from_figure_metadata():
    chunks = StructureAwareChunker({"chunk_size_tokens": 20}).chunk(
        ParsedDocument(
            document_id="doc-figure",
            filename="figures.pdf",
            content_hash="hash",
            blocks=[
                DocumentBlock(
                    doc_id="doc-figure",
                    page=2,
                    bbox=[10, 20, 110, 120],
                    block_type="figure",
                    text=(
                        "Caption: A diagram explaining registration steps.\n"
                        "OCR text: Registration flow text inside the diagram."
                    ),
                    reading_order=1,
                    asset_ref="#/pictures/0",
                    metadata={
                        "figure": {
                            "figure_id": "pictures-0",
                            "page": 2,
                            "bbox": [10, 20, 110, 120],
                            "image_ref": "data/rag/assets/school/doc/figures/pictures-0.png",
                            "visual": {
                                "caption": "A diagram explaining registration steps.",
                                "ocr_text": "Registration flow text inside the diagram.",
                                "visual_embedding": [1.0, 0.0],
                                "visual_embedding_provider": "fake_visual",
                            },
                        }
                    },
                )
            ],
        )
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["is_figure"] is True
    assert chunks[0].metadata["figure_id"] == "pictures-0"
    assert chunks[0].metadata["image_ref"].endswith("pictures-0.png")
    assert chunks[0].metadata["visual_embedding"] == [1.0, 0.0]
    assert "registration steps" in chunks[0].text


@pytest.mark.asyncio
async def test_rag_service_ingests_retrieves_formats_and_deletes_document():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 3,
                "candidate_count": 10,
                "keyword_weight": 0.5,
                "vector_weight": 0.5,
                "recency_weight": 0.0,
                "max_context_chars": 2000,
            },
        },
        parser=TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    await service.initialize()

    job = await service.ingest_document(
        content=b"# Tuition\n\nTuition payment is due on May 1.",
        filename="handbook.md",
        knowledge_base_id="school",
        metadata={"title": "School Handbook"},
        document_id="doc-1",
    )
    results = await service.retrieve(
        "When is tuition due?",
        knowledge_base_ids=["school"],
    )
    context = service.build_context(results)
    sources = service.sources_from_results(results)
    deleted = await service.delete_document("doc-1", knowledge_base_id="school")
    after_delete = await service.retrieve(
        "When is tuition due?",
        knowledge_base_ids=["school"],
    )

    assert job.status == "completed"
    assert job.chunks_indexed >= 1
    assert results
    assert "[S1]" in context
    assert "May 1" in context
    assert sources[0].source_type == "rag"
    assert sources[0].document_id == "doc-1"
    assert deleted == job.chunks_indexed
    assert after_delete == []


@pytest.mark.asyncio
async def test_rag_reingest_same_document_replaces_stale_chunks():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 5,
                "candidate_count": 10,
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
        },
        parser=TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    await service.initialize()

    await service.ingest_document(
        content=b"Old tuition payment deadline is May 1.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    await service.ingest_document(
        content=b"New tuition payment deadline is June 1.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )

    old_results = await service.retrieve("May tuition", knowledge_base_ids=["school"])
    new_results = await service.retrieve("June tuition", knowledge_base_ids=["school"])

    assert all("Old tuition" not in result.chunk.text for result in old_results)
    assert any("June 1" in result.chunk.text for result in new_results)
    assert len({result.index_version for result in new_results}) == 1


@pytest.mark.asyncio
async def test_rag_delete_document_is_knowledge_base_scoped():
    store = InMemoryRagVectorStore()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 5,
                "candidate_count": 10,
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
        },
        parser=TextDocumentParser(),
        vector_store=store,
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    await service.initialize()

    await service.ingest_document(
        content=b"Shared document in school knowledge base.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    await service.ingest_document(
        content=b"Shared document in athletics knowledge base.",
        filename="handbook.md",
        knowledge_base_id="athletics",
        document_id="doc-1",
    )

    deleted = await service.delete_document("doc-1", knowledge_base_id="school")
    school_results = await service.retrieve("Shared", knowledge_base_ids=["school"])
    athletics_results = await service.retrieve(
        "Shared", knowledge_base_ids=["athletics"]
    )

    assert deleted >= 1
    assert school_results == []
    assert athletics_results
    assert athletics_results[0].knowledge_base_id == "athletics"


def test_rag_context_truncates_oversized_first_source():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "retrieval": {"max_context_chars": 500},
        },
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    chunk = StructureAwareChunker({"chunk_size_tokens": 80}).chunk(
        ParsedDocument(
            document_id="doc-1",
            filename="handbook.md",
            content_hash="hash",
            blocks=[
                DocumentBlock(
                    doc_id="doc-1",
                    page=1,
                    bbox=None,
                    block_type="text",
                    text="tuition " * 1000,
                    reading_order=1,
                )
            ],
        )
    )[0]
    context = service.build_context(
        [
            RagSearchResult(
                chunk=chunk,
                score=0.9,
                match_source="keyword",
                knowledge_base_id="school",
                index_version="v1",
            )
        ]
    )

    assert "[S1]" in context
    assert "Content: tuition" in context
    assert len(context) <= 520


def test_rag_context_formats_table_row_column_values():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "retrieval": {"max_context_chars": 2000},
        },
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    chunk = DocumentChunk(
        chunk_id="table-1-r1",
        document_id="calendar",
        text=(
            "Section: Academic Calendar\n"
            "Table columns: Item, Fall '25, Winter '26, Spring '26\n"
            "UNDERGRAD/GRAD GRADE CHANGE OPTION: Fall '25 = Nov 30 Sun; "
            "Winter '26 = Mar 6 Fri; Spring '26 = May 29 Fri"
        ),
        page_start=1,
        page_end=1,
        block_ids=["table-1:r1"],
        block_types=["table"],
        section_path=["Academic Calendar"],
        metadata={
            "filename": "calendar.pdf",
            "table_id": "table-1",
            "table_columns": ["Item", "Fall '25", "Winter '26", "Spring '26"],
            "is_table_row": True,
            "table_row_id": "r1",
            "table_row_label": "UNDERGRAD/GRAD GRADE CHANGE OPTION",
            "table_row_values": {
                "Fall '25": "Nov 30 Sun",
                "Winter '26": "Mar 6 Fri",
                "Spring '26": "May 29 Fri",
            },
        },
    )

    context = service.build_context(
        [
            RagSearchResult(
                chunk=chunk,
                score=0.95,
                match_source="hybrid",
                knowledge_base_id="school",
                index_version="v1",
            )
        ]
    )

    assert "[S1]" in context
    assert "Table row: UNDERGRAD/GRAD GRADE CHANGE OPTION" in context
    assert "Fall '25 = Nov 30 Sun" in context
    assert "Winter '26 = Mar 6 Fri" in context
    assert "Spring '26 = May 29 Fri" in context


@pytest.mark.asyncio
async def test_image_aware_rag_ingests_and_retrieves_figure_chunk():
    cropper = FakeFigureCropper()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "docling"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 3,
                "candidate_count": 5,
                "keyword_weight": 0.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
            "visual": {
                "enabled": True,
                "embedding": {"enabled": True, "dimensions": 2},
                "retrieval": {"weight": 1.0, "min_score": 0.0},
                "storage": {"assets_dir": "data/rag/assets"},
            },
        },
        parser=FigureParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
        visual_processor=FakeVisualProcessor(),
        figure_cropper=cropper,
    )
    await service.initialize()

    job = await service.ingest_document(
        content=b"%PDF fake",
        filename="figures.pdf",
        knowledge_base_id="school",
        document_id="doc-figure",
    )
    results = await service.retrieve(
        "find the registration diagram",
        knowledge_base_ids=["school"],
    )
    context = service.build_context(results)
    sources = service.sources_from_results(results)
    deleted = await service.delete_document("doc-figure", knowledge_base_id="school")

    assert job.status == "completed"
    assert job.chunks_indexed == 1
    assert results
    assert results[0].match_source == "visual"
    assert results[0].chunk.metadata["is_figure"] is True
    assert "Caption: A diagram explaining registration steps." in context
    assert "OCR text: Registration flow text inside the diagram." in context
    assert sources[0].figure_id == "pictures-0"
    assert sources[0].image_ref.endswith("pictures-0.png")
    assert deleted == 1
    assert cropper.deleted == [("doc-figure", "school")]


@pytest.mark.asyncio
async def test_image_only_visual_chunk_is_indexable_without_caption_or_ocr():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "docling"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 1,
                "candidate_count": 5,
                "keyword_weight": 0.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
            "visual": {
                "enabled": True,
                "embedding": {"enabled": True, "dimensions": 2},
                "retrieval": {"weight": 1.0, "min_score": 0.0},
            },
        },
        parser=FigureParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
        visual_processor=FakeVisualProcessor(include_text=False),
        figure_cropper=FakeFigureCropper(),
    )
    await service.initialize()

    job = await service.ingest_document(
        content=b"%PDF fake",
        filename="figures.pdf",
        knowledge_base_id="school",
        document_id="doc-figure",
    )
    results = await service.retrieve(
        "visual-only figure", knowledge_base_ids=["school"]
    )

    assert job.status == "completed"
    assert job.chunks_indexed == 1
    assert results
    assert results[0].chunk.metadata["is_figure"] is True
    assert results[0].match_source == "visual"


@pytest.mark.asyncio
async def test_embedding_failure_completes_with_warning_and_keyword_index():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "retrieval": {
                "top_k": 5,
                "candidate_count": 10,
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
        },
        parser=TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=FailingEmbeddingProvider(),
    )
    await service.initialize()

    job = await service.ingest_document(
        content=b"Tuition deadline is May 1.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    results = await service.retrieve("tuition", knowledge_base_ids=["school"])

    assert job.status == "completed_with_warnings"
    assert job.warnings
    assert results


@pytest.mark.asyncio
async def test_in_memory_keyword_search_uses_bm25_term_frequency():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 2,
                "candidate_count": 2,
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
        },
        parser=TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
    )
    await service.initialize()
    await service.ingest_document(
        content=(
            b"Tuition deadline appears once in a longer administrative paragraph "
            b"with registration billing residency transcript calendar policy text."
        ),
        filename="one.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    await service.ingest_document(
        content=b"Tuition deadline deadline deadline.",
        filename="two.md",
        knowledge_base_id="school",
        document_id="doc-2",
    )

    results = await service.retrieve("deadline", knowledge_base_ids=["school"])

    assert [result.chunk.document_id for result in results] == ["doc-2", "doc-1"]
    assert results[0].score == pytest.approx(1.0)
    assert 0 < results[1].score < 1.0


def test_redis_withscores_results_parse_and_normalize_bm25_scores():
    store = RedisStackRagVectorStore(
        {
            "backend": "redis_stack",
            "embedding": {"dimensions": 4},
            "retrieval": {"keyword_scorer": "BM25STD"},
        }
    )
    raw_results = [
        2,
        b"rag:chunk:school:chunk-1",
        b"4.0",
        _raw_redis_chunk_fields(
            chunk_id=b"chunk-1",
            document_id=b"doc-1",
            text=b"Tuition deadline deadline",
            index_version=b"v1",
        ),
        b"rag:chunk:school:chunk-2",
        b"2.0",
        _raw_redis_chunk_fields(
            chunk_id=b"chunk-2",
            document_id=b"doc-2",
            text=b"Tuition deadline",
            index_version=b"v2",
        ),
    ]

    parsed = store._parse_results(
        raw_results,
        "keyword",
        with_scores=True,
        normalize_scores=True,
    )

    assert [result.chunk.chunk_id for result in parsed] == ["chunk-1", "chunk-2"]
    assert parsed[0].score == pytest.approx(1.0)
    assert parsed[1].score == pytest.approx(0.5)
    assert parsed[0].index_version == "v1"
    assert parsed[1].chunk.document_id == "doc-2"


@pytest.mark.asyncio
async def test_redis_keyword_search_falls_back_to_tag_keywords():
    store = RedisStackRagVectorStore(
        {
            "backend": "redis_stack",
            "embedding": {"dimensions": 4},
            "retrieval": {"keyword_scorer": "BM25STD"},
        }
    )
    store.client = FakeSearchRedis()

    results = await store._keyword_search("tuition", ["school"], 5)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"
    assert any("WITHSCORES" in command for command in store.client.commands)
    assert any("@keywords" in command[2] for command in store.client.commands)


@pytest.mark.asyncio
async def test_rag_retrieve_applies_configured_reranker_to_candidates():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 1,
                "candidate_count": 3,
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
            "rerank": {"enabled": True, "provider": "test", "top_n": 3},
        },
        parser=TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
        reranker=ReverseReranker(),
    )
    await service.initialize()
    for index in range(1, 4):
        await service.ingest_document(
            content=f"Tuition policy alpha document {index}.".encode(),
            filename=f"handbook-{index}.md",
            knowledge_base_id="school",
            document_id=f"doc-{index}",
        )

    results = await service.retrieve(
        "alpha",
        knowledge_base_ids=["school"],
        max_results=1,
        candidate_count=3,
    )

    assert [result.chunk.document_id for result in results] == ["doc-3"]


@pytest.mark.asyncio
async def test_rag_retrieve_falls_back_when_reranker_fails():
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "parser": {"provider": "text"},
            "chunking": {"chunk_size_tokens": 80},
            "embedding": {"provider": "hash", "dimensions": 16},
            "retrieval": {
                "top_k": 1,
                "candidate_count": 2,
                "keyword_weight": 1.0,
                "vector_weight": 0.0,
                "recency_weight": 0.0,
            },
            "rerank": {"enabled": True, "provider": "test", "top_n": 2},
        },
        parser=TextDocumentParser(),
        vector_store=InMemoryRagVectorStore(),
        embedding_provider=HashEmbeddingProvider(dimensions=16),
        reranker=FailingReranker(),
    )
    await service.initialize()
    await service.ingest_document(
        content=b"Tuition policy alpha first.",
        filename="one.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    await service.ingest_document(
        content=b"Tuition policy alpha second.",
        filename="two.md",
        knowledge_base_id="school",
        document_id="doc-2",
    )

    results = await service.retrieve(
        "alpha",
        knowledge_base_ids=["school"],
        max_results=1,
        candidate_count=2,
    )

    assert [result.chunk.document_id for result in results] == ["doc-1"]


@pytest.mark.asyncio
async def test_queue_ingestion_stages_file_creates_job_and_xadds_stream(tmp_path):
    service = _queued_rag_service(tmp_path)
    await service.initialize()

    job = await service.queue_document_ingestion(
        content=b"Tuition deadline is May 1.",
        filename="../../handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )

    staged_path = service.load_staged_content(job.storage_ref)
    messages = service.vector_store.client.streams[service.stream_key]

    assert staged_path == b"Tuition deadline is May 1."
    assert ".." not in job.storage_ref
    assert job.status == "queued"
    assert messages[0][1]["job_id"] == job.job_id
    assert messages[0][1]["storage_ref"] == job.storage_ref


@pytest.mark.asyncio
async def test_worker_success_marks_completed_and_acks_stream_message(tmp_path):
    service = _queued_rag_service(tmp_path)
    await service.initialize()
    job = await service.queue_document_ingestion(
        content=b"Tuition deadline is May 1.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    message_id, fields = service.vector_store.client.streams[service.stream_key][0]

    completed = await service.process_stream_message(message_id, fields, "worker-1")

    assert completed.status == "completed"
    assert completed.attempts == 1
    assert service.vector_store.client.acked == [
        (service.stream_key, service.group_name, message_id)
    ]


@pytest.mark.asyncio
async def test_terminal_pending_job_is_acked_without_reprocessing(tmp_path):
    service = _queued_rag_service(tmp_path)
    await service.initialize()
    job = await service.queue_document_ingestion(
        content=b"Tuition deadline is May 1.",
        filename="handbook.md",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    job.status = "completed"
    await service._save_job(job)
    message_id, fields = service.vector_store.client.streams[service.stream_key][0]

    returned = await service.process_stream_message(message_id, fields, "worker-1")

    assert returned.status == "completed"
    assert service.vector_store.client.acked[-1] == (
        service.stream_key,
        service.group_name,
        message_id,
    )


@pytest.mark.asyncio
async def test_worker_failure_under_max_attempts_moves_job_to_retry_zset(tmp_path):
    service = _queued_rag_service(tmp_path, parser=FailingParser(), max_attempts=2)
    await service.initialize()
    job = await service.queue_document_ingestion(
        content=b"broken pdf",
        filename="handbook.pdf",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    message_id, fields = service.vector_store.client.streams[service.stream_key][0]

    failed = await service.process_stream_message(message_id, fields, "worker-1")

    assert failed.status == "retrying"
    assert failed.attempts == 1
    assert job.job_id in service.vector_store.client.zsets[service.retry_zset_key]
    assert service.vector_store.client.acked[-1] == (
        service.stream_key,
        service.group_name,
        message_id,
    )


@pytest.mark.asyncio
async def test_retry_promoter_requeues_due_jobs(tmp_path):
    service = _queued_rag_service(tmp_path, parser=FailingParser(), max_attempts=2)
    await service.initialize()
    job = await service.queue_document_ingestion(
        content=b"broken pdf",
        filename="handbook.pdf",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    message_id, fields = service.vector_store.client.streams[service.stream_key][0]
    await service.process_stream_message(message_id, fields, "worker-1")

    promoted = await service.promote_due_retries()
    requeued = await service.get_job(job.job_id)

    assert promoted == 1
    assert requeued.status == "queued"
    assert len(service.vector_store.client.streams[service.stream_key]) == 2


@pytest.mark.asyncio
async def test_worker_exhausts_retries_to_dead_letter_stream(tmp_path):
    service = _queued_rag_service(tmp_path, parser=FailingParser(), max_attempts=1)
    await service.initialize()
    job = await service.queue_document_ingestion(
        content=b"broken pdf",
        filename="handbook.pdf",
        knowledge_base_id="school",
        document_id="doc-1",
    )
    message_id, fields = service.vector_store.client.streams[service.stream_key][0]

    dead = await service.process_stream_message(message_id, fields, "worker-1")

    assert dead.status == "dead_lettered"
    assert dead.attempts == 1
    dlq = service.vector_store.client.streams[service.dead_letter_stream_key]
    assert dlq[0][1]["job_id"] == job.job_id
    assert "parser exploded" in dlq[0][1]["error"]
    assert service.vector_store.client.xadd_kwargs[-1] == (
        service.dead_letter_stream_key,
        {"maxlen": 10000, "approximate": True},
    )


@pytest.mark.asyncio
async def test_batch_progress_tracks_queued_and_completed_jobs(tmp_path):
    service = _queued_rag_service(tmp_path)
    await service.initialize()

    batch = await service.create_batch(
        knowledge_base_id="school",
        documents=[
            {
                "filename": "one.md",
                "document_id": "doc-1",
                "content_base64": base64.b64encode(b"One tuition rule").decode(),
            },
            {
                "filename": "two.md",
                "document_id": "doc-2",
                "content_base64": base64.b64encode(b"Two tuition rule").decode(),
            },
        ],
    )
    before = await service.get_batch(batch.batch_id)
    message_id, fields = service.vector_store.client.streams[service.stream_key][0]
    await service.process_stream_message(message_id, fields, "worker-1")
    after = await service.get_batch(batch.batch_id)

    assert before["total"] == 2
    assert before["status_counts"]["queued"] == 2
    assert after["status_counts"]["completed"] == 1
    assert after["status_counts"]["queued"] == 1


@pytest.mark.asyncio
async def test_queue_rejects_storage_ref_outside_staging_dir(tmp_path):
    service = _queued_rag_service(tmp_path)
    await service.initialize()

    with pytest.raises(ValueError, match="storage_ref"):
        await service.queue_document_ingestion(
            storage_ref="/tmp/not-in-rag-staging.pdf",
            filename="handbook.pdf",
            knowledge_base_id="school",
            document_id="doc-1",
        )


def test_docling_parser_extracts_table_text_and_layout_metadata():
    parser = DoclingParser()
    blocks = parser._extract_blocks(
        {
            "texts": [
                {
                    "label": "section_header",
                    "text": "Tuition",
                    "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
                }
            ],
            "tables": [
                {
                    "label": "table",
                    "prov": [{"page_no": 2, "bbox": [10, 20, 30, 40]}],
                    "data": {
                        "table_cells": [
                            {
                                "text": "Item",
                                "start_row_offset_idx": 0,
                                "start_col_offset_idx": 0,
                            },
                            {
                                "text": "Fall '25",
                                "start_row_offset_idx": 0,
                                "start_col_offset_idx": 1,
                            },
                            {
                                "text": "Winter '26",
                                "start_row_offset_idx": 0,
                                "start_col_offset_idx": 2,
                            },
                            {
                                "text": "UNDERGRAD/GRAD GRADE CHANGE OPTION",
                                "start_row_offset_idx": 1,
                                "start_col_offset_idx": 0,
                            },
                            {
                                "text": "Nov 30 Sun",
                                "start_row_offset_idx": 1,
                                "start_col_offset_idx": 1,
                            },
                            {
                                "text": "Mar 6 Fri",
                                "start_row_offset_idx": 1,
                                "start_col_offset_idx": 2,
                            },
                        ]
                    },
                }
            ],
            "pictures": [
                {
                    "label": "picture",
                    "self_ref": "#/pictures/0",
                    "prov": [
                        {"page_no": 3, "bbox": {"x0": 5, "y0": 6, "x1": 7, "y1": 8}}
                    ],
                }
            ],
        },
        "doc-1",
    )

    assert [block.block_type for block in blocks] == [
        "section_header",
        "table",
        "figure",
    ]
    assert "Item | Fall '25 | Winter '26" in blocks[1].text
    assert (
        "UNDERGRAD/GRAD GRADE CHANGE OPTION | Nov 30 Sun | Mar 6 Fri" in blocks[1].text
    )
    assert "Fall '25 = Nov 30 Sun" in blocks[1].text
    table = blocks[1].metadata["table"]
    assert table["columns"] == ["Item", "Fall '25", "Winter '26"]
    assert table["header_rows"] == [0]
    assert table["rows"][0]["label"] == "UNDERGRAD/GRAD GRADE CHANGE OPTION"
    assert table["rows"][0]["values"]["Winter '26"] == "Mar 6 Fri"
    assert table["rows"][0]["cells"][1]["col_index"] == 1
    assert blocks[0].bbox == [1.0, 2.0, 3.0, 4.0]
    assert blocks[1].bbox == [10.0, 20.0, 30.0, 40.0]
    assert blocks[2].asset_ref == "#/pictures/0"
    assert blocks[2].metadata["figure"]["figure_id"] == "pictures-0"
    assert blocks[2].metadata["figure"]["bbox"] == [5.0, 6.0, 7.0, 8.0]
