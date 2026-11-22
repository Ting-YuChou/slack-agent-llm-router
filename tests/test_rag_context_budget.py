"""How the context budget is shared between retrieved sources."""

from pathlib import Path

import yaml

from src.prompt_assembly import RAG_CONTEXT_HEADER, RAG_CONTEXT_INSTRUCTIONS
from src.rag.chunker import DocumentChunk, StructureAwareChunker
from src.rag.service import RagService
from src.rag.vector_store import RagSearchResult
from src.utils.schema import RagChunkingConfig, RagRetrievalConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = f"{RAG_CONTEXT_HEADER}\n\n{RAG_CONTEXT_INSTRUCTIONS}"


def _service(max_context_chars=None) -> RagService:
    service = RagService.__new__(RagService)
    service.retrieval_config = (
        {} if max_context_chars is None else {"max_context_chars": max_context_chars}
    )
    return service


def _result(index: int, text: str) -> RagSearchResult:
    return RagSearchResult(
        chunk=DocumentChunk(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            text=text,
            page_start=index,
            page_end=index,
            block_ids=[f"block-{index}"],
            block_types=["text"],
            metadata={"title": f"Title {index}"},
        ),
        score=1.0 / index,
        match_source="keyword",
        knowledge_base_id="school",
    )


def _entry(index: int, content: str) -> str:
    return (
        f"[S{index}] Title {index} (p. {index}, kb=school)\n"
        f"Chunk: chunk-{index}\n"
        f"Content: {content}"
    )


def _contents(context: str) -> dict:
    entries = context.split("\n\n")[2:]
    return {
        int(entry[2 : entry.index("]")]): entry.split("Content: ", 1)[1]
        for entry in entries
    }


def test_every_source_gets_a_share_when_they_cannot_all_fit():
    results = [_result(index, f"s{index} " + "x" * 5000) for index in range(1, 6)]

    context = _service(12000).build_context(results)

    contents = _contents(context)
    assert sorted(contents) == [1, 2, 3, 4, 5]
    assert len(context) <= 12000
    assert all(content.endswith("...") for content in contents.values())
    assert all(content.startswith(f"s{index} ") for index, content in contents.items())
    lengths = [len(content) for content in contents.values()]
    assert max(lengths) - min(lengths) <= 5
    # Nearly the whole budget is used.
    assert len(context) > 12000 - 50


def test_short_sources_stay_whole_and_long_ones_share_what_is_left():
    texts = ["short one", "y" * 9000, "short three", "z" * 9000, "w" * 9000]
    results = [_result(index, text) for index, text in enumerate(texts, start=1)]

    context = _service(12000).build_context(results)

    contents = _contents(context)
    assert contents[1] == "short one"
    assert contents[3] == "short three"
    long_lengths = [len(contents[index]) for index in (2, 4, 5)]
    assert all(length > 3000 for length in long_lengths)
    assert max(long_lengths) - min(long_lengths) <= 5
    assert len(context) <= 12000


def test_sources_that_all_fit_are_written_out_in_full():
    results = [_result(1, "First answer."), _result(2, "Second answer.")]

    context = _service(12000).build_context(results)

    assert context == "\n\n".join(
        [PREAMBLE, _entry(1, "First answer."), _entry(2, "Second answer.")]
    )


def test_a_single_oversized_source_fills_the_budget():
    context = _service(600).build_context([_result(1, "t" * 5000)])

    room = 600 - len(PREAMBLE) - 2 - len(_entry(1, "")) - 2
    assert context == f"{PREAMBLE}\n\n{_entry(1, 't' * (room - 3) + '...')}"


def test_a_budget_too_small_for_every_header_keeps_the_top_ranked_sources():
    results = [_result(index, "q" * 3000) for index in range(1, 6)]
    budget = len(PREAMBLE) + 2 + 2 * (len(_entry(1, "")) + 2) + 200

    context = _service(budget).build_context(results)

    contents = _contents(context)
    assert sorted(contents) == [1, 2]
    assert len(context) <= budget
    assert all(len(content) >= 80 for content in contents.values())


def test_no_results_and_no_room_behave_as_before():
    assert _service(12000).build_context([]) == ""
    assert _service(100).build_context([_result(1, "text")]) == PREAMBLE[:100]


def test_default_budget_holds_five_default_sized_chunks():
    chunker = StructureAwareChunker({})
    # Five letters and a space: about the average English word.
    words = " ".join(f"w{index:04d}" for index in range(chunker.chunk_size_tokens))
    results = [_result(index, words) for index in range(1, 6)]

    context = _service().build_context(results)

    contents = _contents(context)
    assert sorted(contents) == [1, 2, 3, 4, 5]
    assert all(content == words for content in contents.values())


def test_defaults_agree_between_schema_code_and_checked_in_configs():
    chunking = RagChunkingConfig()
    retrieval = RagRetrievalConfig()
    chunker = StructureAwareChunker({})

    assert (chunking.chunk_size_tokens, chunking.chunk_overlap_tokens) == (300, 40)
    assert chunking.max_chunk_chars == 6000
    assert retrieval.max_context_chars == 12000
    assert (chunker.chunk_size_tokens, chunker.chunk_overlap_tokens) == (300, 40)
    for name in ("config.yaml", "config.compose.yaml"):
        rag = yaml.safe_load((REPO_ROOT / "config" / name).read_text("utf-8"))["rag"]
        assert rag["chunking"]["chunk_size_tokens"] == 300, name
        assert rag["chunking"]["chunk_overlap_tokens"] == 40, name
        assert rag["chunking"]["max_chunk_chars"] == 6000, name
        assert rag["retrieval"]["max_context_chars"] == 12000, name


def test_short_sources_that_fit_are_kept_even_when_the_budget_is_tight():
    results = [_result(1, "a" * 40), _result(2, "b"), _result(3, "c" * 30)]
    whole = "\n\n".join(
        [PREAMBLE, _entry(1, "a" * 40), _entry(2, "b"), _entry(3, "c" * 30)]
    )

    context = _service(len(whole) + 2).build_context(results)

    assert context == whole
