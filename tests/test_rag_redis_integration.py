import os
import uuid

import pytest

from src.rag.chunker import DocumentChunk
from src.rag.vector_store import RedisStackRagVectorStore


async def _redis_stack_client_or_skip():
    redis_asyncio = pytest.importorskip("redis.asyncio")
    client = redis_asyncio.Redis(
        host=os.getenv("RAG_REDIS_HOST", "localhost"),
        port=int(os.getenv("RAG_REDIS_PORT", "6380")),
        db=int(os.getenv("RAG_REDIS_DB", "0")),
        decode_responses=False,
    )
    try:
        await client.ping()
        await client.execute_command("FT._LIST")
    except Exception as exc:
        if hasattr(client, "aclose"):
            await client.aclose()
        elif hasattr(client, "close"):
            await client.close()
        pytest.skip(f"Redis Stack is not available: {exc}")
    return client


def _chunk(chunk_id: str, document_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        page_start=1,
        page_end=1,
        block_ids=[chunk_id],
        block_types=["text"],
        metadata={"filename": "handbook.pdf"},
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_rag_store_reindex_delete_keyword_and_vector_search():
    probe = await _redis_stack_client_or_skip()
    if hasattr(probe, "aclose"):
        await probe.aclose()
    elif hasattr(probe, "close"):
        await probe.close()

    key_prefix = f"ragtest:{uuid.uuid4().hex}"
    store = RedisStackRagVectorStore(
        {
            "backend": "redis_stack",
            "embedding": {"dimensions": 4},
            "redis": {
                "host": os.getenv("RAG_REDIS_HOST", "localhost"),
                "port": int(os.getenv("RAG_REDIS_PORT", "6380")),
                "db": int(os.getenv("RAG_REDIS_DB", "0")),
                "key_prefix": key_prefix,
            },
        }
    )

    try:
        await store.initialize()
        await store.upsert_chunks(
            [_chunk("old", "doc-1", "old tuition deadline is May")],
            [[1.0, 0.0, 0.0, 0.0]],
            knowledge_base_id="school",
        )
        await store.upsert_chunks(
            [_chunk("new", "doc-1", "new tuition deadline is June")],
            [[0.0, 1.0, 0.0, 0.0]],
            knowledge_base_id="school",
        )
        await store.upsert_chunks(
            [_chunk("new", "doc-1", "athletics handbook deadline")],
            [[1.0, 0.0, 0.0, 0.0]],
            knowledge_base_id="athletics",
        )

        stale = await store.search(
            "old",
            None,
            knowledge_base_ids=["school"],
            limit=5,
            candidate_count=10,
            keyword_weight=1.0,
            vector_weight=0.0,
            recency_weight=0.0,
            min_score=0.01,
        )
        fresh = await store.search(
            "June",
            None,
            knowledge_base_ids=["school"],
            limit=5,
            candidate_count=10,
            keyword_weight=1.0,
            vector_weight=0.0,
            recency_weight=0.0,
            min_score=0.01,
        )
        vector = await store.search(
            "semantic",
            [1.0, 0.0, 0.0, 0.0],
            knowledge_base_ids=["athletics"],
            limit=5,
            candidate_count=10,
            keyword_weight=0.0,
            vector_weight=1.0,
            recency_weight=0.0,
            min_score=0.01,
        )
        deleted = await store.delete_document("doc-1", knowledge_base_id="school")
        remaining = await store.search(
            "athletics",
            None,
            knowledge_base_ids=["athletics"],
            limit=5,
            candidate_count=10,
            keyword_weight=1.0,
            vector_weight=0.0,
            recency_weight=0.0,
            min_score=0.01,
        )

        assert stale == []
        assert fresh and fresh[0].chunk.text == "new tuition deadline is June"
        assert fresh[0].index_version
        assert vector and vector[0].knowledge_base_id == "athletics"
        assert deleted == 1
        assert remaining and remaining[0].knowledge_base_id == "athletics"
    finally:
        if store.client:
            try:
                await store.client.execute_command(
                    "FT.DROPINDEX", store.index_name, "DD"
                )
            except Exception:
                pass
        await store.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_stream_group_ack_and_pending_autoclaim():
    client = await _redis_stack_client_or_skip()
    stream = f"ragtest:{uuid.uuid4().hex}:stream"
    group = "rag-workers"
    try:
        await client.xgroup_create(stream, group, id="0", mkstream=True)
        message_id = await client.xadd(stream, {"job_id": "job-1"})
        first_read = await client.xreadgroup(
            group,
            "worker-1",
            {stream: ">"},
            count=1,
            block=100,
        )
        claimed = await client.xautoclaim(
            stream,
            group,
            "worker-2",
            0,
            "0-0",
            count=1,
        )
        acked = await client.xack(stream, group, message_id)
        pending = await client.xpending(stream, group)

        assert first_read
        assert claimed[1][0][0] == message_id
        assert acked == 1
        pending_count = pending["pending"] if isinstance(pending, dict) else pending[0]
        assert pending_count == 0
    finally:
        await client.delete(stream)
        if hasattr(client, "aclose"):
            await client.aclose()
        elif hasattr(client, "close"):
            await client.close()
