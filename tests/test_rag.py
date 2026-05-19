import base64

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


class FailingParser:
    def parse_bytes(self, **_kwargs):
        raise RuntimeError("parser exploded")


class FakeStreamRedis:
    def __init__(self):
        self.values = {}
        self.sets = {}
        self.streams = {}
        self.groups = set()
        self.acked = []
        self.zsets = {}

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, _ttl, value):
        self.values[key] = value

    async def delete(self, key):
        self.values.pop(key, None)
        self.sets.pop(key, None)

    async def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def xgroup_create(self, stream, group, id="0", mkstream=True):
        self.groups.add((stream, group))
        self.streams.setdefault(stream, [])

    async def xadd(self, stream, fields, **_kwargs):
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
