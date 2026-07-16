import json
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


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    metadata=None,
    block_types=None,
) -> DocumentChunk:
    chunk_metadata = {"filename": "handbook.pdf"}
    if metadata:
        chunk_metadata.update(metadata)
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        page_start=1,
        page_end=1,
        block_ids=[chunk_id],
        block_types=block_types or ["text"],
        metadata=chunk_metadata,
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
            [
                _chunk(
                    "bm25",
                    "doc-2",
                    "tuition deadline deadline deadline deadline policy",
                )
            ],
            [[0.0, 0.8, 0.0, 0.0]],
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
        bm25 = await store.search(
            "deadline",
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
        assert len(bm25) >= 2
        assert bm25[0].score == pytest.approx(1.0)
        assert any(0.0 < result.score < 1.0 for result in bm25[1:])
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
async def test_redis_rag_store_persists_and_cleans_table_sidecars():
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

    try:
        await store.initialize()
        await store.upsert_chunks(
            [
                _chunk(
                    "table-summary",
                    "calendar",
                    table["markdown"],
                    block_types=["table"],
                    metadata={
                        "table_id": "table-1",
                        "table_columns": table["columns"],
                        "is_table_summary": True,
                        "is_table_row": False,
                        "table_sidecar": table,
                    },
                ),
                _chunk(
                    "table-row-r1",
                    "calendar",
                    table["rows"][0]["semantic_text"],
                    block_types=["table"],
                    metadata={
                        "table_id": "table-1",
                        "table_columns": table["columns"],
                        "is_table_summary": False,
                        "is_table_row": True,
                        "table_row_id": "r1",
                        "table_row_label": table["rows"][0]["label"],
                        "table_row_values": table["rows"][0]["values"],
                        "table_row_sidecar": table["rows"][0],
                    },
                ),
            ],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            knowledge_base_id="school",
        )

        table_key = store._table_key("school", "calendar", "table-1")
        row_key = store._table_row_key("school", "calendar", "table-1", "r1")
        table_payload = await store.client.hgetall(table_key)
        row_payload = await store.client.hgetall(row_key)
        results = await store.search(
            "grade change option Fall Winter Spring",
            [1.0, 0.0, 0.0, 0.0],
            knowledge_base_ids=["school"],
            limit=3,
            candidate_count=10,
            keyword_weight=0.3,
            vector_weight=0.7,
            recency_weight=0.0,
            min_score=0.01,
        )

        await store.upsert_chunks(
            [_chunk("replacement", "calendar", "calendar replacement text")],
            [[0.0, 1.0, 0.0, 0.0]],
            knowledge_base_id="school",
        )
        table_exists_after_reindex = await store.client.exists(table_key)
        row_exists_after_reindex = await store.client.exists(row_key)

        assert table_payload
        assert row_payload
        assert json.loads(table_payload[b"table_json"])["columns"][1] == "Fall '25"
        assert (
            json.loads(row_payload[b"row_json"])["values"]["Spring '26"] == "May 29 Fri"
        )
        assert results
        assert results[0].chunk.metadata.get("is_table_row") is True
        assert (
            results[0].chunk.metadata["table_row_values"]["Winter '26"] == "Mar 6 Fri"
        )
        assert table_exists_after_reindex == 0
        assert row_exists_after_reindex == 0
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
async def test_redis_rag_store_indexes_visual_chunks_and_cleans_sidecars():
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
            "visual": {
                "enabled": True,
                "embedding": {"enabled": True, "dimensions": 4},
            },
            "redis": {
                "host": os.getenv("RAG_REDIS_HOST", "localhost"),
                "port": int(os.getenv("RAG_REDIS_PORT", "6380")),
                "db": int(os.getenv("RAG_REDIS_DB", "0")),
                "key_prefix": key_prefix,
            },
        }
    )
    figure = {
        "figure_id": "figure-1",
        "page": 2,
        "bbox": [10, 20, 110, 120],
        "image_ref": "data/rag/assets/school/doc-fig/figures/figure-1.png",
        "visual": {
            "caption": "A diagram explaining registration steps.",
            "ocr_text": "Registration flow text inside the diagram.",
            "visual_embedding_provider": "fake_visual",
            "visual_embedding": [1.0, 0.0, 0.0, 0.0],
        },
    }

    try:
        await store.initialize()
        await store.upsert_chunks(
            [
                _chunk(
                    "figure-chunk",
                    "doc-fig",
                    "Caption: A diagram explaining registration steps.",
                    block_types=["figure"],
                    metadata={
                        "is_figure": True,
                        "figure_id": "figure-1",
                        "image_ref": figure["image_ref"],
                        "figure_caption": figure["visual"]["caption"],
                        "figure_ocr_text": figure["visual"]["ocr_text"],
                        "visual_embedding_provider": "fake_visual",
                        "figure_sidecar": figure,
                    },
                )
            ],
            [[0.0, 1.0, 0.0, 0.0]],
            visual_embeddings=[[1.0, 0.0, 0.0, 0.0]],
            knowledge_base_id="school",
        )

        figure_key = store._figure_key("school", "doc-fig", "figure-1")
        visual_key = store._visual_chunk_key(
            "figure-chunk", "school", document_id="doc-fig"
        )
        figure_payload = await store.client.hgetall(figure_key)
        visual_payload = await store.client.hgetall(visual_key)
        results = await store.search(
            "registration diagram",
            None,
            knowledge_base_ids=["school"],
            limit=3,
            candidate_count=10,
            keyword_weight=0.0,
            vector_weight=0.0,
            recency_weight=0.0,
            min_score=0.0,
            visual_embedding=[1.0, 0.0, 0.0, 0.0],
            visual_weight=1.0,
            visual_min_score=0.0,
        )

        await store.upsert_chunks(
            [_chunk("replacement", "doc-fig", "replacement text")],
            [[0.0, 1.0, 0.0, 0.0]],
            visual_embeddings=[None],
            knowledge_base_id="school",
        )
        figure_exists_after_reindex = await store.client.exists(figure_key)
        visual_exists_after_reindex = await store.client.exists(visual_key)

        assert figure_payload
        assert visual_payload
        assert json.loads(figure_payload[b"figure_json"])["figure_id"] == "figure-1"
        assert results
        assert results[0].chunk.metadata["is_figure"] is True
        assert results[0].match_source == "visual"
        assert figure_exists_after_reindex == 0
        assert visual_exists_after_reindex == 0
    finally:
        if store.client:
            for index_name in (store.index_name, store.visual_index_name):
                try:
                    await store.client.execute_command("FT.DROPINDEX", index_name, "DD")
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stale_generation_cannot_overwrite_newer_sidecars(monkeypatch):
    client = await _redis_stack_client_or_skip()
    key_prefix = f"ragtest:{uuid.uuid4().hex}"
    store = RedisStackRagVectorStore(
        {
            "embedding": {"dimensions": 4},
            "redis": {
                "host": os.getenv("RAG_REDIS_HOST", "localhost"),
                "port": int(os.getenv("RAG_REDIS_PORT", "6380")),
                "key_prefix": key_prefix,
            },
        }
    )
    try:
        await store.initialize()
        newer = _chunk(
            "newer",
            "doc-race",
            "new generation",
            metadata={
                "figure_id": "figure-new",
                "figure_sidecar": {"caption": "new"},
            },
        )
        await store.upsert_chunks(
            [newer], [[1.0, 0.0, 0.0, 0.0]], knowledge_base_id="school"
        )
        await store.client.set(
            store._active_generation_key("doc-race", "school"),
            "99999999999999999999-99999999",
        )
        monkeypatch.setattr(
            "src.rag.vector_store._new_index_version",
            lambda: "00000000000000000001-00000001",
        )
        stale = _chunk(
            "stale",
            "doc-race",
            "stale generation",
            metadata={
                "figure_id": "figure-stale",
                "figure_sidecar": {"caption": "stale"},
            },
        )

        await store.upsert_chunks(
            [stale], [[0.0, 1.0, 0.0, 0.0]], knowledge_base_id="school"
        )

        assert await store.client.exists(
            store._figure_key("school", "doc-race", "figure-new")
        )
        assert not await store.client.exists(
            store._figure_key("school", "doc-race", "figure-stale")
        )
        assert await store.client.smembers(
            store._document_key("doc-race", "school")
        ) == {b"newer"}
    finally:
        keys = [key async for key in client.scan_iter(match=f"{key_prefix}:*")]
        if keys:
            await client.delete(*keys)
        await store.shutdown()
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_v1_and_v2_search_indexes_have_disjoint_key_roots():
    client = await _redis_stack_client_or_skip()
    prefix = f"ragisolation:{uuid.uuid4().hex}"
    v1_index = f"{prefix}:idx:v1"
    v2_index = f"{prefix}:idx:v2"
    try:
        for index_name, key_root in (
            (v1_index, f"{prefix}:chunk:"),
            (v2_index, f"{prefix}:v2:chunk:"),
        ):
            await client.execute_command(
                "FT.CREATE",
                index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                key_root,
                "SCHEMA",
                "chunk_id",
                "TAG",
                "SORTABLE",
                "text",
                "TEXT",
            )
        for chunk_id in ("a", "b"):
            mapping = {"chunk_id": chunk_id, "text": f"policy {chunk_id}"}
            await client.hset(f"{prefix}:chunk:school:{chunk_id}", mapping=mapping)
            await client.hset(
                f"{prefix}:v2:chunk:{{school:doc}}:{chunk_id}", mapping=mapping
            )

        v1 = await client.execute_command(
            "FT.SEARCH", v1_index, "*", "SORTBY", "chunk_id", "ASC", "LIMIT", "0", "10"
        )
        v2 = await client.execute_command(
            "FT.SEARCH", v2_index, "*", "SORTBY", "chunk_id", "ASC", "LIMIT", "0", "10"
        )

        assert v1[0] == 2
        assert v2[0] == 2
        assert [v1[2][1], v1[4][1]] == [b"a", b"b"]
        assert [v2[2][1], v2[4][1]] == [b"a", b"b"]
    finally:
        for index_name in (v1_index, v2_index):
            try:
                await client.execute_command("FT.DROPINDEX", index_name, "DD")
            except Exception:
                pass
        await client.aclose()
