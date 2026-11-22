"""Chunk sizes for Chinese text, and what happens to text over the limits."""

from typing import List, Sequence

from src.rag.chunker import DocumentChunk, StructureAwareChunker
from src.rag.parser import DocumentBlock, ParsedDocument

LIMITS = {
    "chunk_size_tokens": 900,
    "chunk_overlap_tokens": 120,
    "max_chunk_chars": 6000,
}


def _document(blocks: Sequence[tuple]) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        filename="doc.pdf",
        content_hash="hash",
        blocks=[
            DocumentBlock(
                doc_id="doc-1",
                page=1,
                bbox=None,
                block_type=block_type,
                text=text,
                reading_order=order,
            )
            for order, (block_type, text) in enumerate(blocks, start=1)
        ],
    )


def _chunk(config: dict, blocks: Sequence[tuple]) -> List[DocumentChunk]:
    return StructureAwareChunker(config).chunk(_document(blocks))


def _words(prefix: str, count: int, width: int = 0) -> List[str]:
    return [f"{prefix}{index:04d}".ljust(width, "x") for index in range(count)]


def _zh_sentences(label: str, count: int) -> List[str]:
    # 30 characters each, every one different.
    return [f"{label}第{index:03d}條規定學生應於期限內完成各項註冊與選課手續。" for index in range(count)]


def _within_limits(chunker: StructureAwareChunker, chunks: Sequence[DocumentChunk]):
    return all(
        len(chunk.text) <= chunker.max_chunk_chars
        and chunker._estimate_tokens(chunk.text) <= chunker.chunk_size_tokens
        for chunk in chunks
    )


def test_token_estimate_counts_chinese_characters_and_english_words():
    chunker = StructureAwareChunker({})

    assert chunker._estimate_tokens("加退選截止日") == 6
    assert chunker._estimate_tokens("CS3205 作業系統 final exam") == 7
    assert chunker._estimate_tokens("かな カナ 한국어") == 7
    for text in (
        "Tuition payment is due on May 1.",
        "  spaced   out\twords\nand lines  ",
        "e-mail: registrar@example.edu (ext. 3120)",
    ):
        assert chunker._estimate_tokens(text) == len(text.split())
    assert chunker._estimate_tokens("") == 1


def test_chinese_paragraphs_under_one_heading_are_all_indexed():
    paragraphs = [_zh_sentences(f"甲{number}", 30) for number in range(8)]
    blocks = [("section_header", "學則")] + [
        ("text", "".join(sentences)) for sentences in paragraphs
    ]
    chunker = StructureAwareChunker(LIMITS)

    chunks = chunker.chunk(_document(blocks))

    text = "\n".join(chunk.text for chunk in chunks)
    missing = [
        sentence
        for sentences in paragraphs
        for sentence in sentences
        if sentence not in text
    ]
    assert missing == []
    assert len(chunks) > 1
    assert _within_limits(chunker, chunks)
    assert all(chunk.section_path == ["學則"] for chunk in chunks)


def test_one_long_chinese_block_is_split_where_sentences_end():
    sentences = _zh_sentences("乙", 240)
    chunker = StructureAwareChunker(LIMITS)

    chunks = chunker.chunk(_document([("text", "".join(sentences))]))

    assert len(chunks) > 1
    assert _within_limits(chunker, chunks)
    assert all(chunk.text.endswith("。") for chunk in chunks)
    assert all(
        any(sentence in chunk.text for chunk in chunks) for sentence in sentences
    )
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chinese_text_without_punctuation_is_still_split():
    text = "".join(chr(0x4E00 + index % 3000) for index in range(5000))
    chunker = StructureAwareChunker(LIMITS)

    chunks = chunker.chunk(_document([("text", text)]))

    assert len(chunks) > 1
    assert _within_limits(chunker, chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_english_block_longer_than_the_character_cap_keeps_every_word():
    words = _words("w", 1000, width=9)
    chunker = StructureAwareChunker(LIMITS)

    chunks = chunker.chunk(_document([("text", " ".join(words))]))

    indexed = {word for chunk in chunks for word in chunk.text.split()}
    assert sorted(set(words) - indexed) == []
    assert _within_limits(chunker, chunks)


def test_english_blocks_that_pass_the_character_cap_together_start_a_new_chunk():
    blocks = [
        ("text", " ".join(_words(f"b{number}", 80, width=9))) for number in range(10)
    ]
    chunker = StructureAwareChunker(LIMITS)

    chunks = chunker.chunk(_document(blocks))

    indexed = {word for chunk in chunks for word in chunk.text.split()}
    expected = {word for _kind, text in blocks for word in text.split()}
    assert sorted(expected - indexed) == []
    assert len(chunks) > 1
    assert _within_limits(chunker, chunks)


def test_english_word_windows_are_unchanged_below_the_character_cap():
    words = _words("w", 200)
    config = {
        "chunk_size_tokens": 50,
        "chunk_overlap_tokens": 10,
        "max_chunk_chars": 6000,
    }

    chunks = _chunk(config, [("text", " ".join(words))])

    assert [chunk.text for chunk in chunks] == [
        " ".join(words[start : start + 50]) for start in range(0, 200, 40)
    ]


def test_overlap_between_chunks_stays_within_the_configured_tokens():
    blocks = []
    for number in range(6):
        sentences = [
            " ".join(_words(f"b{number}s{sentence}", 19)) + "." for sentence in range(5)
        ]
        blocks.append(("text", " ".join(sentences)))
    config = {
        "chunk_size_tokens": 250,
        "chunk_overlap_tokens": 30,
        "max_chunk_chars": 6000,
    }

    chunks = _chunk(config, blocks)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        shared = set(previous.text.split()) & set(current.text.split())
        assert 0 < len(shared) <= 30
    # The carried text is whole sentences from the end of the previous chunk.
    assert chunks[1].text.startswith("b1s4")


def test_a_chunk_with_overlap_stays_within_the_size_limit():
    blocks = [("text", " ".join(_words(f"b{number}", 69)) + ".") for number in range(4)]
    config = {
        "chunk_size_tokens": 100,
        "chunk_overlap_tokens": 40,
        "max_chunk_chars": 6000,
    }
    chunker = StructureAwareChunker(config)

    chunks = chunker.chunk(_document(blocks))

    assert len(chunks) == 4
    assert _within_limits(chunker, chunks)


def test_overlap_above_half_the_chunk_size_does_not_multiply_chunks():
    words = _words("w", 1000)
    config = {"chunk_size_tokens": 80, "chunk_overlap_tokens": 120}

    chunks = _chunk(config, [("text", " ".join(words))])

    indexed = {word for chunk in chunks for word in chunk.text.split()}
    assert sorted(set(words) - indexed) == []
    assert len(chunks) <= 30


def test_no_chunk_repeats_another_before_an_oversized_block():
    blocks = [
        ("text", " ".join(_words("a", 60))),
        ("text", " ".join(_words("b", 60))),
        ("text", " ".join(_words("c", 500))),
    ]
    config = {"chunk_size_tokens": 100, "chunk_overlap_tokens": 80}

    chunks = _chunk(config, blocks)

    texts = [chunk.text for chunk in chunks]
    assert len(texts) == len(set(texts))
    indexed = {word for text in texts for word in text.split()}
    assert {"a0000", "b0059", "c0499"} <= indexed
