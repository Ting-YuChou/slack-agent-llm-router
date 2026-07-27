import pytest

from src.memory import HashEmbeddingProvider
from src.rag.normalizer import DocumentLayoutNormalizer
from src.rag.chunker import StructureAwareChunker
from src.rag.parser import DoclingParser, DocumentBlock, ParsedDocument
from src.rag.service import RagService
from src.rag.vector_store import InMemoryRagVectorStore


def _document(*blocks: DocumentBlock) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-layout",
        filename="annual-report.pdf",
        content_hash="sha256",
        blocks=list(blocks),
    )


def _block(
    text: str,
    *,
    page: int,
    block_type: str = "text",
    bbox: list[float] | None = None,
    order: int = 1,
    metadata: dict | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        doc_id="doc-layout",
        page=page,
        bbox=bbox,
        block_type=block_type,
        text=text,
        reading_order=order,
        metadata=dict(metadata or {}),
    )


def test_normalizer_removes_explicit_and_repeated_page_boilerplate():
    blocks = []
    for page in range(1, 4):
        blocks.extend(
            [
                _block(
                    "DBS Group Holdings Ltd · Annual Report 2025",
                    page=page,
                    block_type="page_header" if page == 1 else "text",
                    bbox=[10, 0, 500, 40],
                    order=1,
                    metadata={"page_height": 800},
                ),
                _block(
                    f"Useful body {page}",
                    page=page,
                    bbox=[10, 100, 500, 140],
                    order=2,
                    metadata={"page_height": 800},
                ),
                _block(
                    str(page),
                    page=page,
                    block_type="page_footer" if page == 1 else "text",
                    bbox=[250, 770, 270, 795],
                    order=3,
                    metadata={"page_height": 800},
                ),
            ]
        )

    result = DocumentLayoutNormalizer({}).normalize(_document(*blocks))

    assert [block.text for block in result.document.blocks] == [
        "Useful body 1",
        "Useful body 2",
        "Useful body 3",
    ]
    assert result.stats["boilerplate_removed"] == 6


def test_normalizer_interprets_docling_bottom_left_bbox_origin():
    result = DocumentLayoutNormalizer({}).normalize(
        _document(
            _block(
                "1",
                page=1,
                bbox=[250, 5, 270, 25],
                metadata={
                    "page_height": 800,
                    "provenance": [
                        {
                            "bbox": {"coord_origin": "BOTTOMLEFT"},
                        }
                    ],
                },
            ),
            _block(
                "Useful body",
                page=1,
                bbox=[10, 300, 500, 330],
                order=2,
                metadata={
                    "page_height": 800,
                    "provenance": [
                        {
                            "bbox": {"coord_origin": "BOTTOMLEFT"},
                        }
                    ],
                },
            ),
        )
    )

    assert [block.text for block in result.document.blocks] == ["Useful body"]


def test_normalizer_links_matching_footnote_without_merging_unresolved_footnote():
    document = _document(
        _block(
            "Net interest income increased 14 bps (1).",
            page=99,
            bbox=[10, 100, 500, 130],
            order=1,
        ),
        _block(
            "(1) Excludes the one-time gain from disposal.",
            page=99,
            block_type="footnote",
            bbox=[10, 730, 500, 770],
            order=2,
        ),
        _block(
            "(2) Unresolved disclosure.",
            page=99,
            block_type="footnote",
            bbox=[10, 775, 500, 795],
            order=3,
        ),
    )

    result = DocumentLayoutNormalizer({}).normalize(document)

    assert result.document.blocks[0].text.endswith(
        "Footnote (1): Excludes the one-time gain from disposal."
    )
    unresolved = result.document.blocks[1]
    assert unresolved.text == "Footnote (2): Unresolved disclosure."
    assert unresolved.metadata["is_unresolved_footnote"] is True
    assert result.warnings == ["Unresolved footnote (2) on page 99"]


def test_explicit_footnote_reference_wins_over_later_ambiguous_marker():
    document = _document(
        _block(
            "Authoritative disclosure.",
            page=1,
            order=1,
            metadata={
                "source_block_id": "body-1",
                "child_refs": ["footnote-1"],
            },
        ),
        _block(
            "Unrelated later mention (1).",
            page=1,
            order=2,
            metadata={"source_block_id": "body-2"},
        ),
        _block(
            "(1) Excludes the one-time gain.",
            page=1,
            block_type="footnote",
            order=3,
            metadata={"source_block_id": "footnote-1"},
        ),
    )

    result = DocumentLayoutNormalizer({}).normalize(document)

    assert "Footnote (1)" in result.document.blocks[0].text
    assert "Footnote (1)" not in result.document.blocks[1].text


def test_normalizer_stitches_only_supported_cross_page_sentence_seams():
    document = _document(
        _block(
            "The CET1 ratio im-",
            page=99,
            bbox=[10, 720, 500, 790],
            order=1,
            metadata={"page_height": 800, "section_path": ["Capital"]},
        ),
        _block(
            "proved because retained earnings increased.",
            page=100,
            bbox=[10, 5, 500, 80],
            order=1,
            metadata={"page_height": 800, "section_path": ["Capital"]},
        ),
        _block(
            "A complete sentence.",
            page=100,
            bbox=[10, 700, 500, 790],
            order=2,
            metadata={"page_height": 800, "section_path": ["Capital"]},
        ),
        _block(
            "A new paragraph.",
            page=101,
            bbox=[10, 5, 500, 80],
            order=1,
            metadata={"page_height": 800, "section_path": ["Capital"]},
        ),
    )

    result = DocumentLayoutNormalizer({}).normalize(document)

    assert result.document.blocks[0].text == (
        "The CET1 ratio improved because retained earnings increased."
    )
    assert result.document.blocks[0].metadata["source_pages"] == [99, 100]
    assert result.document.blocks[1].text == "A complete sentence."
    assert result.document.blocks[2].text == "A new paragraph."


def test_normalizer_merges_matching_cross_page_tables_and_preserves_cell_pages():
    first_table = {
        "table_id": "table-1",
        "columns": ["Item", "FY24", "FY25"],
        "header_rows": [0],
        "rows": [
            {
                "row_id": "r1",
                "row_index": 1,
                "label": "CET1",
                "values": {"FY24": "14.6", "FY25": "15.0"},
                "cells": [{"text": "CET1", "page": 99}],
                "semantic_text": "CET1: FY24 = 14.6; FY25 = 15.0",
            }
        ],
        "markdown": "Item | FY24 | FY25\nCET1 | 14.6 | 15.0",
    }
    second_table = {
        "table_id": "table-2",
        "columns": ["Item", "FY24", "FY25"],
        "header_rows": [0],
        "rows": [
            {
                "row_id": "r1",
                "row_index": 1,
                "label": "ROE",
                "values": {"FY24": "18.0", "FY25": "18.6"},
                "cells": [{"text": "ROE", "page": 100}],
                "semantic_text": "ROE: FY24 = 18.0; FY25 = 18.6",
            }
        ],
        "markdown": "Item | FY24 | FY25\nROE | 18.0 | 18.6",
    }
    document = _document(
        _block(
            first_table["markdown"],
            page=99,
            block_type="table",
            bbox=[10, 600, 500, 795],
            metadata={
                "page_height": 800,
                "section_path": ["Capital"],
                "table": first_table,
            },
        ),
        _block(
            second_table["markdown"],
            page=100,
            block_type="table",
            bbox=[10, 5, 500, 200],
            metadata={
                "page_height": 800,
                "section_path": ["Capital"],
                "table": second_table,
            },
        ),
    )

    result = DocumentLayoutNormalizer({}).normalize(document)

    assert len(result.document.blocks) == 1
    merged = result.document.blocks[0]
    assert merged.metadata["source_pages"] == [99, 100]
    assert [row["label"] for row in merged.metadata["table"]["rows"]] == [
        "CET1",
        "ROE",
    ]
    assert merged.metadata["table"]["rows"][1]["cells"][0]["page"] == 100


def test_normalizer_does_not_merge_cross_page_tables_with_unknown_columns():
    document = _document(
        _block(
            "first fragment",
            page=1,
            block_type="table",
            bbox=[10, 650, 500, 795],
            metadata={
                "page_height": 800,
                "section_path": ["Capital"],
                "table": {"columns": [], "rows": []},
            },
        ),
        _block(
            "second fragment",
            page=2,
            block_type="table",
            bbox=[10, 5, 500, 160],
            metadata={
                "page_height": 800,
                "section_path": ["Capital"],
                "table": {"columns": [], "rows": []},
            },
        ),
    )

    result = DocumentLayoutNormalizer({}).normalize(document)

    assert len(result.document.blocks) == 2
    assert result.warnings == ["Ambiguous cross-page tables on pages 1-2"]


def test_chunker_flushes_previous_section_before_applying_new_heading():
    document = _document(
        _block("Capital", page=1, block_type="section_header", order=1),
        _block("Capital policy text.", page=1, order=2),
        _block("Liquidity", page=1, block_type="section_header", order=3),
        _block("Liquidity policy text.", page=1, order=4),
    )

    chunks = StructureAwareChunker(
        {"chunk_size_tokens": 900, "chunk_overlap_tokens": 0}
    ).chunk(document)

    assert len(chunks) == 2
    assert chunks[0].section_path == ["Capital"]
    assert "Capital policy text." in chunks[0].text
    assert "Liquidity policy text." not in chunks[0].text
    assert chunks[1].section_path == ["Liquidity"]


def test_chunker_uses_normalized_source_pages_for_cross_page_table():
    table = {
        "table_id": "table-1",
        "columns": ["Item", "FY25"],
        "rows": [
            {
                "row_id": "r1",
                "label": "CET1",
                "values": {"FY25": "15.0"},
                "semantic_text": "CET1: FY25 = 15.0",
            }
        ],
        "markdown": "Item | FY25\nCET1 | 15.0",
    }
    document = _document(
        _block(
            table["markdown"],
            page=99,
            block_type="table",
            metadata={"table": table, "source_pages": [99, 100]},
        )
    )

    chunks = StructureAwareChunker({}).chunk(document)

    assert all(chunk.page_start == 99 for chunk in chunks)
    assert all(chunk.page_end == 100 for chunk in chunks)


def test_docling_adapter_uses_body_reference_order_and_keeps_layout_provenance():
    payload = {
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/pictures/0"},
            ]
        },
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "text",
                "text": "Before table",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": [10, 10, 500, 30],
                        "page_height": 800,
                    }
                ],
            },
            {
                "self_ref": "#/texts/1",
                "label": "footnote",
                "text": "(1) Disclosure",
                "font_size": 8,
                "parent": {"$ref": "#/body"},
                "prov": [{"page_no": 1, "bbox": [10, 700, 500, 730]}],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": [{"page_no": 1, "bbox": [10, 100, 500, 300]}],
                "data": {"rows": [["Item", "FY25"], ["CET1", "15.0"]]},
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "prov": [{"page_no": 1, "bbox": [10, 400, 500, 650]}],
            }
        ],
    }

    blocks = DoclingParser()._extract_blocks(payload, "doc-layout")

    assert [block.block_type for block in blocks] == [
        "text",
        "table",
        "footnote",
        "figure",
    ]
    assert [block.metadata["document_order"] for block in blocks] == [1, 2, 3, 4]
    assert blocks[0].metadata["page_height"] == 800
    assert blocks[2].metadata["font_size"] == 8
    assert blocks[2].metadata["parent_ref"] == "#/body"


def test_docling_adapter_uses_document_level_page_dimensions():
    payload = {
        "body": {"children": [{"$ref": "#/texts/0"}]},
        "pages": {"1": {"size": {"width": 612, "height": 792}}},
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "text",
                "text": "Page body",
                "prov": [{"page_no": 1, "bbox": [10, 10, 500, 30]}],
            }
        ],
    }

    block = DoclingParser()._extract_blocks(payload, "doc-layout")[0]

    assert block.metadata["page_width"] == 612
    assert block.metadata["page_height"] == 792


def test_docling_adapter_fallback_order_interleaves_block_categories():
    payload = {
        "pages": {"1": {"size": {"width": 600, "height": 800}}},
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "text",
                "text": "Top text",
                "prov": [{"page_no": 1, "bbox": [10, 50, 500, 80]}],
            }
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": [{"page_no": 1, "bbox": [10, 500, 500, 700]}],
                "data": {"rows": [["Item", "Value"]]},
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "prov": [{"page_no": 1, "bbox": [10, 200, 500, 450]}],
            }
        ],
    }

    blocks = DoclingParser()._extract_blocks(payload, "doc-layout")

    assert [block.block_type for block in blocks] == ["text", "figure", "table"]


def test_normalizer_links_nearby_text_and_table_blocks_to_figure():
    document = _document(
        _block(
            "Capital ratio discussion",
            page=10,
            order=1,
            metadata={"section_path": ["Capital"]},
        ),
        _block(
            "",
            page=11,
            block_type="figure",
            order=1,
            metadata={
                "section_path": ["Capital"],
                "figure": {"figure_id": "figure-capital"},
            },
        ),
        _block(
            "Different section",
            page=11,
            order=2,
            metadata={"section_path": ["Liquidity"]},
        ),
    )

    result = DocumentLayoutNormalizer({}).normalize(document)

    assert result.document.blocks[0].metadata["related_figure_ids"] == [
        "figure-capital"
    ]
    assert "related_figure_ids" not in result.document.blocks[2].metadata


@pytest.mark.asyncio
async def test_rag_service_normalizes_layout_before_chunking_and_surfaces_warnings():
    class Parser:
        def parse_bytes(self, **_kwargs):
            return _document(
                _block("Useful disclosure.", page=1, order=1),
                _block(
                    "(7) Unresolved note.",
                    page=1,
                    block_type="footnote",
                    order=2,
                ),
            )

    store = InMemoryRagVectorStore()
    service = RagService(
        {
            "enabled": True,
            "backend": "memory",
            "layout_normalization": {"enabled": True},
            "embedding": {"provider": "hash", "dimensions": 8},
            "visual": {"enabled": False},
        },
        parser=Parser(),
        vector_store=store,
        embedding_provider=HashEmbeddingProvider(dimensions=8),
    )

    job = await service.process_ingestion_job(
        "job-layout",
        content=b"%PDF",
        filename="annual-report.pdf",
        knowledge_base_id="bank",
        document_id="doc-layout",
    )

    assert job.status == "completed_with_warnings"
    assert job.warnings == ["Unresolved footnote (7) on page 1"]
    assert any(
        item.chunk.metadata.get("is_unresolved_footnote")
        for item in store.items.values()
    )
