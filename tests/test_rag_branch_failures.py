"""I10: a hung or broken retrieval branch is visible, and its partner's hits survive."""

import asyncio
import logging
import time

import pytest

from src.llm_router_part2_inference import OpenAIProvider, route_response_status
from src.memory import NoOpEmbeddingProvider
from src.rag.service import RagService
from src.rag.vector_store import RedisStackRagVectorStore
from src.utils.schema import RagPolicy

# The time-cap tests' engine double: a 1 s request deadline, an instant model.
from test_retrieval_time_cap import (
    _engine,
    _model_failures,
    _process,
    _published_event,
    _request,
)

_FIELDS = [
    b"knowledge_base_id",
    b"school",
    b"document_id",
    b"doc-1",
    b"chunk_id",
    b"chunk-1",
    b"text",
    b"Tuition payment is due on May 1.",
    b"metadata",
    b"{}",
    b"page_start",
    b"1",
    b"page_end",
    b"1",
]
_RAG_NOTICE = "not grounded in the knowledge base"


def _keyword_reply(command, *, hit):
    if not hit:
        return [0]
    if "WITHSCORES" in command:
        return [1, b"rag:chunk:school:chunk-1", b"1.0", _FIELDS]
    return [1, b"rag:chunk:school:chunk-1", _FIELDS]


class _VectorHangsRedis:
    """The keyword query answers at once; the vector (KNN) query hangs."""

    def __init__(self, *, keyword_hit=True):
        self.keyword_hit = keyword_hit

    async def execute_command(self, *command):
        if "KNN" in str(command[2]):
            await asyncio.sleep(10)
        return _keyword_reply(command, hit=self.keyword_hit)


class _Redis:
    """Every query answers; the keyword query finds the chunk when hit=True."""

    def __init__(self, *, keyword_hit):
        self.keyword_hit = keyword_hit
        self.vector_queries = 0

    async def execute_command(self, *command):
        if "KNN" in str(command[2]):
            self.vector_queries += 1
            return [0]
        return _keyword_reply(command, hit=self.keyword_hit)


class _Embedder:
    def __init__(self, embedding=(1.0,)):
        self.embedding = list(embedding)

    async def embed(self, _text):
        return list(self.embedding)


class _DownEmbedder:
    async def embed(self, _text):
        raise ConnectionError("embedding API down")


def _service(client, *, deadline_seconds=0.3, embedder=None):
    store = RedisStackRagVectorStore(
        {
            "retrieval": {"deadline_seconds": deadline_seconds},
            "embedding": {"dimensions": 1},
        }
    )
    store.client = client
    return RagService(
        {
            "enabled": True,
            "backend": "memory",
            "retrieval": {"deadline_seconds": deadline_seconds},
        },
        embedding_provider=embedder or _Embedder(),
        vector_store=store,
    )


# --- A hung branch: the answered branch's hits come back, degraded ----------


@pytest.mark.asyncio
async def test_hung_vector_branch_keeps_keyword_hits_as_degraded():
    service = _service(_VectorHangsRedis(), deadline_seconds=0.3)

    started = time.monotonic()
    outcome = await service.retrieve_with_outcome("tuition payment")

    assert outcome.status == "hits"
    assert [hit.chunk.chunk_id for hit in outcome.results] == ["chunk-1"]
    assert outcome.degraded is True
    assert "vector" in outcome.detail
    assert time.monotonic() - started < 0.3


@pytest.mark.asyncio
async def test_hung_vector_branch_without_keyword_hits_is_a_timeout():
    service = _service(_VectorHangsRedis(keyword_hit=False), deadline_seconds=0.3)

    outcome = await service.retrieve_with_outcome("tuition payment")

    assert outcome.status == "timeout"


@pytest.mark.asyncio
async def test_a_shorter_request_share_is_the_deadline_the_store_works_to():
    # Configured for 30 s; the request can only spare 0.3 s.
    service = _service(_VectorHangsRedis(), deadline_seconds=30)

    started = time.monotonic()
    outcome = await service.retrieve_with_outcome(
        "tuition payment", deadline_seconds=0.3
    )

    assert outcome.status == "hits"
    assert outcome.degraded is True
    assert time.monotonic() - started < 0.3


@pytest.mark.asyncio
async def test_plain_retrieve_also_keeps_the_answered_branch_hits():
    service = _service(_VectorHangsRedis(), deadline_seconds=0.3)

    results = await service.retrieve("tuition payment")

    assert [hit.chunk.chunk_id for hit in results] == ["chunk-1"]


@pytest.mark.asyncio
async def test_engine_answers_with_degraded_hits_when_the_vector_branch_hangs():
    # The 30 s store deadline cannot fire; the request's 0.5 s share does.
    engine = _engine(
        rag_service=_service(_VectorHangsRedis(), deadline_seconds=30),
        scheduler={"request_deadline_seconds": 1.0, "min_attempt_window_seconds": 0},
    )
    request = _request(knowledge_base_ids=["school"])

    response, elapsed = await _process(engine, request)

    assert response.error is None
    assert "rag_degraded" in response.warnings
    assert "rag_unavailable" not in response.warnings
    assert "Tuition payment is due on May 1." in request.metadata["rag_context"]
    assert elapsed < 0.9
    engine.cache.cache_response.assert_not_awaited()
    assert _model_failures(engine) == []


# --- An embedding outage or a wrong-sized embedding fails the vector branch ---


@pytest.mark.asyncio
async def test_embedding_outage_with_keyword_hits_is_degraded():
    service = _service(_Redis(keyword_hit=True), embedder=_DownEmbedder())

    outcome = await service.retrieve_with_outcome("tuition payment")

    assert outcome.status == "hits"
    assert outcome.degraded is True


@pytest.mark.asyncio
async def test_embedding_outage_without_keyword_hits_is_an_error():
    service = _service(_Redis(keyword_hit=False), embedder=_DownEmbedder())

    outcome = await service.retrieve_with_outcome("tuition payment")

    assert outcome.status == "error"


@pytest.mark.asyncio
async def test_wrong_embedding_size_without_keyword_hits_is_an_error(caplog):
    client = _Redis(keyword_hit=False)
    service = _service(client, embedder=_Embedder((1.0, 0.0)))

    with caplog.at_level(logging.WARNING):
        outcome = await service.retrieve_with_outcome("tuition payment")

    assert outcome.status == "error"
    assert client.vector_queries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keyword_hit", "expected_status"), [(True, "hits"), (False, "no_hits")]
)
async def test_keyword_only_configuration_is_not_an_outage(
    keyword_hit, expected_status
):
    service = _service(
        _Redis(keyword_hit=keyword_hit), embedder=NoOpEmbeddingProvider()
    )

    outcome = await service.retrieve_with_outcome("tuition payment")

    assert outcome.status == expected_status
    assert outcome.degraded is False


@pytest.mark.asyncio
async def test_required_rag_with_embeddings_down_and_no_keyword_hits_is_503():
    engine = _engine(
        rag_service=_service(_Redis(keyword_hit=False), embedder=_DownEmbedder())
    )
    request = _request(rag_policy=RagPolicy.REQUIRED)

    response, _ = await _process(engine, request)

    assert response.error_code == "retrieval_unavailable"
    assert route_response_status(response) == (503, {})
    assert "文件庫沒有足夠資訊" not in response.response_text
    assert _model_failures(engine) == []
    assert _published_event(engine, request)["error_type"] == "internal"


@pytest.mark.asyncio
async def test_auto_rag_with_embeddings_down_and_no_keyword_hits_tells_the_model():
    engine = _engine(
        rag_service=_service(_Redis(keyword_hit=False), embedder=_DownEmbedder())
    )
    request = _request(knowledge_base_ids=["school"])

    response, _ = await _process(engine, request)

    assert response.error is None
    assert "rag_unavailable" in response.warnings
    sent = engine.providers["openai"].generate_response.await_args.args[0]
    assert _RAG_NOTICE in OpenAIProvider({})._build_prompt(sent)
