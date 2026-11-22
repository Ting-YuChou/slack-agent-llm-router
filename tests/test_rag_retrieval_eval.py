"""The retrieval evaluation harness and its bilingual fixture."""

import json

import pytest

from scripts.rag_retrieval_eval import (
    FixtureParser,
    load_fixture,
    run_eval,
    summarize,
)

TOY_SETTINGS = {
    "chunking": {"chunk_size_tokens": 80, "chunk_overlap_tokens": 0},
    "retrieval": {"top_k": 2, "candidate_count": 10, "max_context_chars": 2000},
}
TOY_CORPUS = [
    {
        "document_id": "fees",
        "title": "Fees",
        "language": "en",
        "blocks": [
            {"type": "section_header", "text": "Fees", "page": 1},
            {"type": "text", "text": "The parking permit costs $120.", "page": 1},
        ],
    },
    {
        "document_id": "dining",
        "title": "Dining",
        "language": "en",
        "blocks": [
            {"type": "text", "text": "The cafeteria opens at 7:00 a.m.", "page": 1},
        ],
    },
    {
        "document_id": "sports",
        "title": "Sports",
        "language": "en",
        "blocks": [
            {
                "type": "text",
                "text": "The swimming pool closes at 9:00 p.m.",
                "page": 2,
            },
        ],
    },
]


def _query(query_id, query, document_id, span, kind="exact"):
    return {
        "id": query_id,
        "language": "en",
        "kind": kind,
        "query": query,
        "document_id": document_id,
        "answer_span": span,
    }


def test_fixture_is_balanced_and_every_answer_is_in_its_document():
    corpus, queries = load_fixture()
    texts = {
        document["document_id"]: "\n".join(
            block["text"] for block in document["blocks"]
        )
        for document in corpus
    }

    assert len({query["id"] for query in queries}) == len(queries)
    by_language = {
        language: [query for query in queries if query["language"] == language]
        for language in ("zh", "en")
    }
    assert len(by_language["zh"]) == len(by_language["en"]) == len(queries) // 2
    assert {document["language"] for document in corpus} == {"zh", "en"}
    for query in queries:
        assert query["kind"] in {"exact", "semantic"}
        assert query["answer_span"] in texts[query["document_id"]], query["id"]
        elsewhere = [
            document_id
            for document_id, text in texts.items()
            if document_id != query["document_id"] and query["answer_span"] in text
        ]
        assert elsewhere == [], query["id"]


def test_fixture_holds_the_long_sections_the_evaluation_is_about():
    corpus, queries = load_fixture()
    by_id = {document["document_id"]: document for document in corpus}
    spans = {query["id"]: query["answer_span"] for query in queries}

    # A Chinese section whose answer sits past the 6,000-character cap.
    course_text = "\n\n".join(
        block["text"] for block in by_id["zh-courses"]["blocks"][1:]
    )
    assert all(block["type"] == "text" for block in by_id["zh-courses"]["blocks"][1:])
    assert course_text.index(spans["zh-01"]) > 6000
    # One English paragraph longer than a default chunk.
    lab_block = by_id["en-labs"]["blocks"][1]["text"]
    assert len(lab_block.split()) > 900
    assert len(lab_block) > 6000


def test_fixture_parser_keeps_block_order_types_and_pages():
    document = TOY_CORPUS[0]

    parsed = FixtureParser().parse_bytes(
        content=json.dumps(document).encode("utf-8"),
        filename="fees.json",
        document_id="fees",
        metadata={"title": "Fees"},
    )

    assert parsed.document_id == "fees"
    assert parsed.metadata == {"title": "Fees"}
    assert [block.block_type for block in parsed.blocks] == ["section_header", "text"]
    assert [block.reading_order for block in parsed.blocks] == [1, 2]
    assert parsed.blocks[1].text == "The parking permit costs $120."
    assert parsed.blocks[1].doc_id == "fees"


def test_summarize_reports_each_stage_per_group():
    rows = [
        {
            "language": "en",
            "kind": "exact",
            "indexed": True,
            "rank": 1,
            "in_context": True,
            "results": 2,
            "sources_in_context": 2,
        },
        {
            "language": "en",
            "kind": "exact",
            "indexed": True,
            "rank": 4,
            "in_context": False,
            "results": 4,
            "sources_in_context": 1,
        },
        {
            "language": "zh",
            "kind": "semantic",
            "indexed": False,
            "rank": None,
            "in_context": False,
            "results": 0,
            "sources_in_context": 0,
        },
    ]

    summary = summarize(rows)

    assert set(summary) == {"all", "en", "en/exact", "zh", "zh/semantic"}
    assert summary["en"] == {
        "n": 2,
        "indexed": 1.0,
        "recall": 1.0,
        "mrr": pytest.approx((1 + 0.25) / 2),
        "in_context": 0.5,
        "coverage": 0.5,
    }
    assert summary["zh"]["recall"] == 0.0
    assert summary["zh"]["coverage"] == 0.0
    assert summary["all"]["n"] == 3
    assert summary["all"]["indexed"] == pytest.approx(2 / 3)
    assert summary["all"]["recall"] == pytest.approx(2 / 3)
    assert summarize([]) == {
        "all": {
            "n": 0,
            "indexed": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "in_context": 0.0,
            "coverage": 0.0,
        }
    }


@pytest.mark.asyncio
async def test_run_eval_scores_hits_misses_and_wrong_documents():
    queries = [
        _query("hit", "How much is the parking permit?", "fees", "permit costs $120"),
        # Nothing in the corpus shares a word with this query.
        _query("miss", "zzzunknown", "dining", "opens at 7:00 a.m."),
        # The span is indexed, but under another document than the expected one.
        _query("other", "swimming pool", "dining", "closes at 9:00 p.m."),
    ]

    report = await run_eval(TOY_CORPUS, queries, settings=TOY_SETTINGS)

    rows = {row["id"]: row for row in report["rows"]}
    assert report["backend"] == "memory"
    assert report["chunks"] == 3
    assert rows["hit"]["indexed"] is True
    assert rows["hit"]["rank"] == 1
    assert rows["hit"]["in_context"] is True
    assert rows["hit"]["sources_in_context"] == rows["hit"]["results"] >= 1
    assert rows["miss"]["indexed"] is True
    assert rows["miss"]["rank"] is None
    assert rows["miss"]["in_context"] is False
    assert rows["miss"]["status"] == "no_hits"
    assert rows["other"]["indexed"] is False
    assert rows["other"]["rank"] is None
    assert report["summary"]["all"]["recall"] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_run_eval_fails_loudly_when_a_document_cannot_be_ingested():
    broken = [
        {"document_id": "empty", "title": "Empty", "language": "en", "blocks": []}
    ]

    with pytest.raises(ValueError, match="no indexable text chunks"):
        await run_eval(broken, [], settings=TOY_SETTINGS)


@pytest.mark.asyncio
async def test_every_answer_in_the_fixture_reaches_the_index():
    corpus, queries = load_fixture()

    report = await run_eval(corpus, queries)

    lost = [row["id"] for row in report["rows"] if not row["indexed"]]
    assert lost == []


@pytest.mark.asyncio
async def test_exact_term_queries_find_their_answer_in_both_languages():
    corpus, queries = load_fixture()

    report = await run_eval(corpus, queries)

    summary = report["summary"]
    assert summary["zh/exact"]["recall"] >= 0.9
    assert summary["en/exact"]["recall"] >= 0.9
    assert summary["zh/exact"]["in_context"] >= 0.9
    assert summary["en/exact"]["in_context"] >= 0.9


@pytest.mark.asyncio
async def test_every_retrieved_source_reaches_the_context():
    corpus, queries = load_fixture()

    report = await run_eval(corpus, queries)

    dropped = [
        row["id"]
        for row in report["rows"]
        if row["sources_in_context"] != row["results"]
    ]
    assert dropped == []
