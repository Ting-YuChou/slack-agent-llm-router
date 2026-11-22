"""Keyword search for queries and chunks written without spaces."""

import pytest

from src.memory import tokenize
from src.rag.chunker import DocumentChunk
from src.rag.keywords import cjk_bigrams, keyword_terms
from src.rag.vector_store import InMemoryRagVectorStore, RedisStackRagVectorStore

SCHOOL = "@knowledge_base_id:{school}"


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        page_start=1,
        page_end=1,
        block_ids=[chunk_id],
        block_types=["text"],
        metadata={"filename": "handbook.pdf"},
    )


class _Redis:
    """Records commands; answers index and search commands from a script."""

    def __init__(self, *, info=None, failing=(), hits=0):
        self.commands = []
        self.info = info
        self.failing = set(failing)
        self.hits = hits

    async def execute_command(self, *command):
        self.commands.append(command)
        name = command[0]
        if name in self.failing:
            raise RuntimeError(f"{name} refused")
        if name == "FT.INFO":
            if self.info is None:
                raise RuntimeError("Unknown index name")
            return self.info
        if name == "FT.SEARCH":
            return [self.hits]
        return b"OK"

    def named(self, name):
        return [command for command in self.commands if command[0] == name]

    def queries(self):
        return [command[2] for command in self.named("FT.SEARCH")]


def _index_info(*fields):
    attributes = [
        [b"identifier", field.encode(), b"attribute", field.encode(), b"type", b"TEXT"]
        for field in fields
    ]
    return [b"index_name", b"rag:idx:v2:chunks", b"attributes", attributes]


def _store(client, *, ready=True) -> RedisStackRagVectorStore:
    store = RedisStackRagVectorStore(
        {"backend": "redis_stack", "embedding": {"dimensions": 4}}
    )
    store.client = client
    store._keyword_terms_ready = ready
    return store


def test_cjk_bigrams_are_the_overlapping_pairs_of_each_run():
    assert cjk_bigrams("加退選截止日") == sorted(["加退", "退選", "選截", "截止", "止日"])
    assert cjk_bigrams("カタカナ") == sorted(["カタ", "タカ", "カナ"])
    assert cjk_bigrams("도서관") == sorted(["도서", "서관"])


def test_cjk_bigrams_keep_a_lone_character_and_never_span_a_gap():
    assert cjk_bigrams("第 3 節") == sorted(["第", "節"])
    assert cjk_bigrams("課程CS3205教室，大樓") == sorted(["課程", "教室", "大樓"])


def test_cjk_bigrams_of_text_without_cjk_are_empty():
    assert cjk_bigrams("Tuition is due on May 1.") == []
    assert cjk_bigrams("") == []
    assert cjk_bigrams(None) == []


def test_keyword_terms_equal_tokenize_for_ascii_text():
    for text in (
        "The billing project uses ClickHouse",
        "Figure pictures-0 at 10:30, see v1.2",
        "",
    ):
        assert keyword_terms(text) == tokenize(text)


def test_keyword_terms_combine_ascii_tokens_and_cjk_pairs():
    assert keyword_terms("CS3205 作業系統 期末考") == sorted(
        ["cs3205", "作業", "業系", "系統", "期末", "末考"]
    )


@pytest.mark.asyncio
async def test_in_memory_store_matches_a_chinese_query_by_keyword():
    store = InMemoryRagVectorStore()
    for chunk in (
        _chunk("deadline", "網路加退選截止日為九月十九日下午五時。"),
        _chunk("dorm", "宿舍門禁時間為晚上十一時三十分。"),
        _chunk("english", "Tuition payment is due on May 1."),
    ):
        await store.upsert_chunks([chunk], [None], knowledge_base_id="school")

    results = await store.search(
        "加退選截止日",
        None,
        knowledge_base_ids=["school"],
        limit=5,
        candidate_count=10,
        keyword_weight=1.0,
        vector_weight=0.0,
        recency_weight=0.0,
        min_score=0.0,
    )

    assert [result.chunk.chunk_id for result in results] == ["deadline"]
    assert results[0].match_source == "keyword"
    assert results[0].score == pytest.approx(1.0)


def test_redis_chunk_stores_its_terms_in_their_own_field():
    store = _store(_Redis())
    values = dict(knowledge_base_id="school", index_version="v1", updated_at=1.0)

    chinese = store._chunk_mapping(_chunk("zh", "加退選 CS3205"), None, **values)
    english = store._chunk_mapping(
        _chunk("en", "Physics\n\nPHYS1150 meets at 10:30, see pictures-0"),
        None,
        **values,
    )
    empty = store._chunk_mapping(_chunk("none", "!!"), None, **values)

    assert chinese["keyword_terms"] == "cs3205 加退 退選"
    assert chinese["keywords"] == "cs3205"
    # Punctuation inside a term is escaped so the engine keeps it whole.
    assert english["keyword_terms"].split(" ") == [
        "10\\:30",
        "meets",
        "phys1150",
        "physics",
        "pictures\\-0",
        "see",
    ]
    assert english["text"] == "Physics\n\nPHYS1150 meets at 10:30, see pictures-0"
    assert set(empty) == {
        "knowledge_base_id",
        "document_id",
        "chunk_id",
        "index_version",
        "text",
        "keywords",
        "block_types",
        "metadata",
        "section_path",
        "block_ids",
        "bboxes",
        "page_start",
        "page_end",
        "created_at",
        "updated_at",
    }


@pytest.mark.asyncio
async def test_redis_ascii_query_also_searches_the_terms_field():
    client = _Redis(hits=0)
    store = _store(client)

    await store._keyword_search("tuition payment at 10:30", ["school"], 5)

    terms = "10\\:30|payment|tuition"
    assert client.queries() == [
        f"{SCHOOL} ((@text:({terms})) | (@keyword_terms:({terms})))",
        f"{SCHOOL} @keywords:{{{terms}}}",
    ]


@pytest.mark.asyncio
async def test_redis_chinese_query_searches_the_terms_field_only():
    client = _Redis(hits=0)
    store = _store(client)

    results = await store._keyword_search("加退選", ["school"], 5)

    assert results == []
    # One scored search; no tag fallback, which holds ASCII tokens only.
    assert client.queries() == [f"{SCHOOL} @keyword_terms:(加退|退選)"]
    assert "WITHSCORES" in client.named("FT.SEARCH")[0]


@pytest.mark.asyncio
async def test_redis_mixed_query_is_a_union_of_both_fields():
    client = _Redis(hits=0)
    store = _store(client)

    await store._keyword_search("CS3205 作業", ["school"], 5)
    await store._keyword_search("CS3205 作業", [], 5)

    queries = client.queries()
    union = "(@text:(cs3205)) | (@keyword_terms:(cs3205|作業))"
    assert queries[0] == f"{SCHOOL} ({union})"
    assert queries[1] == f"{SCHOOL} @keywords:{{cs3205}}"
    assert queries[2] == union


@pytest.mark.asyncio
async def test_redis_queries_are_unchanged_until_the_index_has_the_terms_field():
    client = _Redis(hits=0)
    store = _store(client, ready=False)

    assert await store._keyword_search("加退選", ["school"], 5) == []
    await store._keyword_search("CS3205 作業", ["school"], 5)
    await store._keyword_search("tuition payment", [], 5)

    assert client.queries() == [
        f"{SCHOOL} @text:(cs3205)",
        f"{SCHOOL} @keywords:{{cs3205}}",
        "@text:(payment|tuition)",
        "@keywords:{payment|tuition}",
    ]


@pytest.mark.asyncio
async def test_a_new_index_is_created_with_the_terms_field():
    client = _Redis(info=None)
    store = _store(client, ready=False)

    await store._ensure_index()

    (create,) = client.named("FT.CREATE")
    position = create.index("keyword_terms")
    assert create[position : position + 3] == ("keyword_terms", "TEXT", "NOSTEM")
    assert client.named("FT.ALTER") == []
    assert store._keyword_terms_ready is True


@pytest.mark.asyncio
async def test_an_existing_index_without_the_terms_field_gets_it_added():
    client = _Redis(info=_index_info("text"))
    store = _store(client, ready=False)

    await store._ensure_index()

    assert client.named("FT.CREATE") == []
    assert client.named("FT.ALTER") == [
        (
            "FT.ALTER",
            store.index_name,
            "SCHEMA",
            "ADD",
            "keyword_terms",
            "TEXT",
            "NOSTEM",
        )
    ]
    assert store._keyword_terms_ready is True


@pytest.mark.asyncio
async def test_an_index_that_already_has_the_terms_field_is_left_alone():
    client = _Redis(info=_index_info("text", "keyword_terms"))
    store = _store(client, ready=False)

    await store._ensure_index()

    assert [command[0] for command in client.commands] == ["FT.INFO"]
    assert store._keyword_terms_ready is True


@pytest.mark.asyncio
async def test_failing_to_add_the_terms_field_leaves_text_search_working():
    client = _Redis(info=_index_info("text"), failing={"FT.ALTER"})
    store = _store(client, ready=False)

    await store._ensure_index()
    await store._keyword_search("CS3205 作業", ["school"], 5)

    assert store._keyword_terms_ready is False
    assert client.queries()[0] == f"{SCHOOL} @text:(cs3205)"


@pytest.mark.asyncio
async def test_losing_the_race_to_add_the_terms_field_still_enables_it():
    class _RacedRedis(_Redis):
        async def execute_command(self, *command):
            if command[0] == "FT.ALTER":
                # Another worker added the field between our FT.INFO and now.
                self.info = _index_info("text", "keyword_terms")
                self.commands.append(command)
                raise RuntimeError("Duplicate field in schema - keyword_terms")
            return await super().execute_command(*command)

    client = _RacedRedis(info=_index_info("text"))
    store = _store(client, ready=False)

    await store._ensure_index()

    assert store._keyword_terms_ready is True
    assert [command[0] for command in client.commands] == [
        "FT.INFO",
        "FT.ALTER",
        "FT.INFO",
    ]
