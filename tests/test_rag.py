import pytest

from src.memory import HashEmbeddingProvider
from src.rag.chunker import StructureAwareChunker
from src.rag.parser import DocumentBlock, ParsedDocument, TextDocumentParser
from src.rag.service import RagService
from src.rag.vector_store import InMemoryRagVectorStore


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
