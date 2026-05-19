import pytest

from src.memory import HashEmbeddingProvider
from src.rag.chunker import StructureAwareChunker
from src.rag.parser import (
    DoclingParser,
    DocumentBlock,
    ParsedDocument,
    TextDocumentParser,
)
from src.rag.service import RagService
from src.rag.vector_store import InMemoryRagVectorStore, RagSearchResult


class FailingEmbeddingProvider:
    async def embed(self, _text):
        raise RuntimeError("embedding endpoint down")


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
                                "text": "Fee",
                                "start_row_offset_idx": 0,
                                "start_col_offset_idx": 0,
                            },
                            {
                                "text": "Amount",
                                "start_row_offset_idx": 0,
                                "start_col_offset_idx": 1,
                            },
                            {
                                "text": "Tuition",
                                "start_row_offset_idx": 1,
                                "start_col_offset_idx": 0,
                            },
                            {
                                "text": "$100",
                                "start_row_offset_idx": 1,
                                "start_col_offset_idx": 1,
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
    assert "Fee | Amount" in blocks[1].text
    assert "Tuition | $100" in blocks[1].text
    assert blocks[0].bbox == [1.0, 2.0, 3.0, 4.0]
    assert blocks[2].asset_ref == "#/pictures/0"
