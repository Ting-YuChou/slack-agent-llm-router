"""Structure-aware chunking for parsed school documents."""

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List

from src.rag.parser import DocumentBlock, ParsedDocument
from src.rag.visual import compose_figure_text


_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"
_CJK_RE = re.compile(f"[{_CJK}]")
# One unit per CJK character and one per run of anything else between spaces,
# so text without spaces is not counted as a single word.
_UNIT_RE = re.compile(f"[{_CJK}]|[^\\s{_CJK}]+")
_SENTENCE_END_RE = re.compile(r"[。！？；!?;]+[」』”’）)\]]*\s*|\.[\"')\]]*\s+|\n+")


def estimate_tokens(text: str) -> int:
    """Approximate token count; equals ``len(text.split())`` without CJK."""
    return len(_UNIT_RE.findall(text or ""))


def split_sentences(text: str) -> List[str]:
    """Cut after sentence punctuation and line breaks; the pieces rejoin to text."""
    sentences = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        sentences.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        sentences.append(text[start:])
    return [sentence for sentence in sentences if sentence.strip()]


@dataclass
class DocumentChunk:
    """A retrievable unit stored in the RAG index."""

    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    block_ids: List[str]
    block_types: List[str]
    bboxes: List[List[float]] = field(default_factory=list)
    section_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    """Chunk text on parser block boundaries while preserving layout metadata."""

    HEADER_BLOCK_TYPES = {"section_header", "title", "heading"}
    STANDALONE_BLOCK_TYPES = {"figure", "table"}

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.chunk_size_tokens = int(self.config.get("chunk_size_tokens", 900))
        self.chunk_overlap_tokens = int(self.config.get("chunk_overlap_tokens", 120))
        self.max_chunk_chars = int(self.config.get("max_chunk_chars", 6000))
        # More overlap than half a chunk makes every chunk mostly repetition
        # (and a step of one word when it reaches the chunk size).
        self._overlap_tokens = max(
            0, min(self.chunk_overlap_tokens, self.chunk_size_tokens // 2)
        )

    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        current_blocks: List[DocumentBlock] = []
        current_section: List[str] = []
        current_tokens = 0
        current_chars = 0

        for block in sorted(
            document.blocks, key=lambda item: (item.page, item.reading_order)
        ):
            if not block.text.strip() and not self._has_indexable_metadata(block):
                continue

            if block.block_type in self.HEADER_BLOCK_TYPES:
                if current_blocks:
                    chunks.append(
                        self._build_chunk(document, current_blocks, current_section)
                    )
                    current_blocks = []
                    current_tokens = 0
                    current_chars = 0
                current_section = self._next_section_path(current_section, block.text)

            block_tokens = self._estimate_tokens(block.text)
            # Blocks are joined by a blank line.
            block_chars = len(block.text.strip()) + 2
            if block.block_type in self.STANDALONE_BLOCK_TYPES:
                if current_blocks:
                    chunks.append(
                        self._build_chunk(document, current_blocks, current_section)
                    )
                    current_blocks = []
                    current_tokens = 0
                    current_chars = 0
                table_chunks = self._build_table_chunks(
                    document, block, current_section
                )
                if table_chunks:
                    chunks.extend(table_chunks)
                    continue
                figure_chunks = self._build_figure_chunks(
                    document, block, current_section
                )
                if figure_chunks:
                    chunks.extend(figure_chunks)
                    continue
                chunks.extend(self._split_large_block(document, block, current_section))
                continue

            if (
                block_tokens > self.chunk_size_tokens
                or block_chars > self.max_chunk_chars
            ):
                if current_blocks:
                    chunks.append(
                        self._build_chunk(document, current_blocks, current_section)
                    )
                    current_blocks = []
                    current_tokens = 0
                    current_chars = 0
                chunks.extend(self._split_large_block(document, block, current_section))
                continue

            if current_blocks and (
                current_tokens + block_tokens > self.chunk_size_tokens
                or current_chars + block_chars > self.max_chunk_chars
            ):
                chunks.append(
                    self._build_chunk(document, current_blocks, current_section)
                )
                current_blocks = self._overlap_tail(
                    current_blocks,
                    self.chunk_size_tokens - block_tokens,
                    self.max_chunk_chars - block_chars,
                )
                current_tokens = sum(
                    self._estimate_tokens(item.text) for item in current_blocks
                )
                current_chars = sum(
                    len(item.text.strip()) + 2 for item in current_blocks
                )

            current_blocks.append(block)
            current_tokens += block_tokens
            current_chars += block_chars

        if current_blocks:
            chunks.append(self._build_chunk(document, current_blocks, current_section))

        return chunks

    def _build_figure_chunks(
        self,
        document: ParsedDocument,
        block: DocumentBlock,
        section_path: List[str],
    ) -> List[DocumentChunk]:
        figure = (
            block.metadata.get("figure") if isinstance(block.metadata, dict) else None
        )
        if block.block_type != "figure" or not isinstance(figure, dict):
            return []

        figure_id = str(figure.get("figure_id") or self._block_id(block))
        figure["section_path"] = list(section_path)
        visual = dict(figure.get("visual") or {})
        text = block.text.strip() or compose_figure_text(block)
        if not text:
            return []
        base_metadata = {
            **dict(document.metadata or {}),
            "filename": document.filename,
            "content_hash": document.content_hash,
            "is_figure": True,
            "figure_id": figure_id,
            "image_ref": figure.get("image_ref"),
            "asset_ref": figure.get("asset_ref"),
            "figure_index_status": figure.get("index_status"),
            "ocr_provider": visual.get("ocr_provider"),
            "caption_provider": visual.get("caption_provider"),
            "bbox": figure.get("bbox") or block.bbox,
            "page": figure.get("page") or block.page,
            "figure_caption": visual.get("caption"),
            "figure_ocr_text": visual.get("ocr_text"),
            "figure_chart_summary": visual.get("chart_summary"),
            "figure_diagram_summary": visual.get("diagram_summary"),
            "figure_sidecar": figure,
        }
        return [
            self._chunk_from_table_parts(
                document=document,
                block=block,
                block_ids=[f"{self._block_id(block)}:{figure_id}"],
                text=text[: self.max_chunk_chars],
                section_path=section_path,
                metadata=base_metadata,
            )
        ]

    def _build_table_chunks(
        self,
        document: ParsedDocument,
        block: DocumentBlock,
        section_path: List[str],
    ) -> List[DocumentChunk]:
        table = (
            block.metadata.get("table") if isinstance(block.metadata, dict) else None
        )
        if not isinstance(table, dict) or not table.get("rows"):
            return []

        table_id = str(table.get("table_id") or self._block_id(block))
        columns = [str(column) for column in table.get("columns") or []]
        base_metadata = {
            **dict(document.metadata or {}),
            "filename": document.filename,
            "content_hash": document.content_hash,
            "table_id": table_id,
            "table_columns": columns,
        }
        chunks = [
            self._table_summary_chunk(
                document,
                block,
                section_path,
                table=table,
                table_id=table_id,
                base_metadata=base_metadata,
            )
        ]
        for row in table.get("rows") or []:
            if not isinstance(row, dict):
                continue
            row_text = self._table_row_text(table, row, section_path)
            if not row_text.strip():
                continue
            row_id = str(row.get("row_id") or f"r{row.get('row_index', len(chunks))}")
            metadata = {
                **base_metadata,
                "is_table_row": True,
                "is_table_summary": False,
                "table_row_id": row_id,
                "table_row_label": str(row.get("label") or ""),
                "table_row_values": dict(row.get("values") or {}),
                "table_row_sidecar": row,
            }
            block_id = f"{self._block_id(block)}:{row_id}"
            chunks.append(
                self._chunk_from_table_parts(
                    document=document,
                    block=block,
                    block_ids=[block_id],
                    text=row_text,
                    section_path=section_path,
                    metadata=metadata,
                )
            )
        return chunks

    def _table_summary_chunk(
        self,
        document: ParsedDocument,
        block: DocumentBlock,
        section_path: List[str],
        *,
        table: Dict[str, Any],
        table_id: str,
        base_metadata: Dict[str, Any],
    ) -> DocumentChunk:
        text = str(table.get("markdown") or block.text).strip()
        if len(text) > self.max_chunk_chars:
            text = text[: self.max_chunk_chars]
        metadata = {
            **base_metadata,
            "is_table_summary": True,
            "is_table_row": False,
            "table_sidecar": table,
        }
        return self._chunk_from_table_parts(
            document=document,
            block=block,
            block_ids=[f"{self._block_id(block)}:{table_id}:summary"],
            text=text,
            section_path=section_path,
            metadata=metadata,
        )

    def _table_row_text(
        self,
        table: Dict[str, Any],
        row: Dict[str, Any],
        section_path: List[str],
    ) -> str:
        section = " > ".join(section_path)
        columns = ", ".join(str(column) for column in table.get("columns") or [])
        row_label = str(row.get("label") or "").strip()
        semantic = str(row.get("semantic_text") or "").strip()
        parts = []
        if section:
            parts.append(f"Section: {section}")
        if columns:
            parts.append(f"Table columns: {columns}")
        if row_label:
            parts.append(f"Row: {row_label}")
        if semantic:
            parts.append(semantic)
        return "\n".join(parts)

    def _chunk_from_table_parts(
        self,
        *,
        document: ParsedDocument,
        block: DocumentBlock,
        block_ids: List[str],
        text: str,
        section_path: List[str],
        metadata: Dict[str, Any],
    ) -> DocumentChunk:
        source_pages = self._source_pages(block)
        return DocumentChunk(
            chunk_id=self._chunk_id(document.document_id, block_ids, text),
            document_id=document.document_id,
            text=text[: self.max_chunk_chars],
            page_start=min(source_pages),
            page_end=max(source_pages),
            block_ids=block_ids,
            block_types=[block.block_type],
            bboxes=[block.bbox] if block.bbox else [],
            section_path=list(section_path),
            metadata=metadata,
        )

    def _build_chunk(
        self,
        document: ParsedDocument,
        blocks: List[DocumentBlock],
        section_path: List[str],
    ) -> DocumentChunk:
        text = "\n\n".join(block.text.strip() for block in blocks if block.text.strip())
        block_ids = [self._block_id(block) for block in blocks]
        chunk_id = self._chunk_id(document.document_id, block_ids, text)
        bboxes = [block.bbox for block in blocks if block.bbox]
        source_pages = [page for block in blocks for page in self._source_pages(block)]
        page_start = min(source_pages)
        page_end = max(source_pages)
        layout_metadata: Dict[str, Any] = {"source_pages": sorted(set(source_pages))}
        if any(bool(block.metadata.get("is_unresolved_footnote")) for block in blocks):
            layout_metadata["is_unresolved_footnote"] = True
        linked_footnotes = [
            value
            for block in blocks
            for value in block.metadata.get("linked_footnote_block_ids", [])
        ]
        if linked_footnotes:
            layout_metadata["linked_footnote_block_ids"] = linked_footnotes
        related_figures = [
            value
            for block in blocks
            for value in block.metadata.get("related_figure_ids", [])
        ]
        if related_figures:
            layout_metadata["related_figure_ids"] = list(
                dict.fromkeys(str(value) for value in related_figures)
            )
        related_figure_sidecars = [
            value
            for block in blocks
            for value in block.metadata.get("related_figures", [])
            if isinstance(value, dict)
        ]
        if related_figure_sidecars:
            layout_metadata["related_figures"] = related_figure_sidecars
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            text=text[: self.max_chunk_chars],
            page_start=page_start,
            page_end=page_end,
            block_ids=block_ids,
            block_types=sorted({block.block_type for block in blocks}),
            bboxes=bboxes,
            section_path=list(section_path),
            metadata={
                **dict(document.metadata or {}),
                "filename": document.filename,
                "content_hash": document.content_hash,
                **layout_metadata,
            },
        )

    def _split_large_block(
        self,
        document: ParsedDocument,
        block: DocumentBlock,
        section_path: List[str],
    ) -> List[DocumentChunk]:
        pieces = self._split_text(block.text)
        chunks = []
        for index, piece in enumerate(pieces):
            split_block = DocumentBlock(
                doc_id=block.doc_id,
                page=block.page,
                bbox=block.bbox,
                block_type=block.block_type,
                text=piece,
                reading_order=block.reading_order * 1000 + index,
                confidence=block.confidence,
                asset_ref=block.asset_ref,
                metadata=block.metadata,
            )
            chunks.append(self._build_chunk(document, [split_block], section_path))
        return chunks

    def _split_text(self, text: str) -> List[str]:
        if _CJK_RE.search(text):
            # Words are not separated by spaces here: pack whole sentences.
            return self._pack_sentences(text, self._overlap_tokens)
        words = text.split()
        if not words:
            return []
        step = max(self.chunk_size_tokens - self._overlap_tokens, 1)
        pieces = []
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + self.chunk_size_tokens])
            if len(piece) > self.max_chunk_chars:
                pieces.extend(self._pack_sentences(piece, 0))
            elif piece:
                pieces.append(piece)
        return pieces

    def _pack_sentences(self, text: str, overlap_tokens: int) -> List[str]:
        """Split text into pieces within both limits, cutting between sentences.

        Consecutive pieces share up to ``overlap_tokens`` of whole sentences.
        """
        parts: List[str] = []
        for sentence in split_sentences(text):
            if (
                estimate_tokens(sentence) <= self.chunk_size_tokens
                and len(sentence) <= self.max_chunk_chars
            ):
                parts.append(sentence)
            else:
                parts.extend(self._hard_split(sentence))
        pieces: List[str] = []
        current: List[str] = []
        tokens = 0
        chars = 0
        for part in parts:
            part_tokens = estimate_tokens(part)
            if current and (
                tokens + part_tokens > self.chunk_size_tokens
                or chars + len(part) > self.max_chunk_chars
            ):
                pieces.append("".join(current).strip())
                current = self._tail_within(
                    current,
                    min(overlap_tokens, self.chunk_size_tokens - part_tokens),
                    self.max_chunk_chars - len(part),
                )
                tokens = sum(estimate_tokens(item) for item in current)
                chars = sum(len(item) for item in current)
            current.append(part)
            tokens += part_tokens
            chars += len(part)
        if current:
            pieces.append("".join(current).strip())
        return [piece for piece in pieces if piece]

    def _hard_split(self, sentence: str) -> List[str]:
        """Cut one over-long sentence between units; the slices rejoin to it."""
        slices: List[str] = []
        start = None
        count = 0
        for match in _UNIT_RE.finditer(sentence):
            unit_start, unit_end = match.span()
            if unit_end - unit_start > self.max_chunk_chars:
                # A single unbroken run longer than a chunk, e.g. encoded data.
                if start is not None:
                    slices.append(sentence[start:unit_start])
                    start = None
                    count = 0
                slices.extend(
                    sentence[offset : min(offset + self.max_chunk_chars, unit_end)]
                    for offset in range(unit_start, unit_end, self.max_chunk_chars)
                )
                continue
            if start is not None and (
                count + 1 > self.chunk_size_tokens
                or unit_end - start > self.max_chunk_chars
            ):
                slices.append(sentence[start:unit_start])
                start = None
                count = 0
            if start is None:
                start = unit_start
            count += 1
        if start is not None:
            slices.append(sentence[start:])
        return slices

    def _tail_within(
        self, parts: List[str], token_budget: int, char_budget: int
    ) -> List[str]:
        """The longest run of trailing parts that fits both budgets."""
        selected: List[str] = []
        tokens = 0
        chars = 0
        for part in reversed(parts):
            tokens += estimate_tokens(part)
            chars += len(part)
            if tokens > token_budget or chars > char_budget:
                break
            selected.append(part)
        return list(reversed(selected))

    def _overlap_tail(
        self, blocks: List[DocumentBlock], room_tokens: int, room_chars: int
    ) -> List[DocumentBlock]:
        """What the next chunk repeats from the end of the one just closed.

        Whole trailing blocks while they fit the overlap; when even the last
        block is larger, only its final sentences. ``room_*`` is what the next
        block leaves free, so the new chunk stays within the limits.
        """
        token_budget = min(self._overlap_tokens, room_tokens)
        if token_budget <= 0 or room_chars <= 0:
            return []
        selected: List[DocumentBlock] = []
        tokens = 0
        chars = 0
        for block in reversed(blocks):
            tokens += self._estimate_tokens(block.text)
            chars += len(block.text.strip()) + 2
            if tokens > token_budget or chars > room_chars:
                if not selected:
                    tail = "".join(
                        self._tail_within(
                            split_sentences(block.text), token_budget, room_chars - 2
                        )
                    ).strip()
                    if tail:
                        selected.append(replace(block, text=tail))
                break
            selected.append(block)
        return list(reversed(selected))

    def _next_section_path(self, current: List[str], text: str) -> List[str]:
        heading = " ".join(text.split())[:160]
        if not heading:
            return current
        if not current:
            return [heading]
        return [*current[:-1], heading]

    def _block_id(self, block: DocumentBlock) -> str:
        if block.asset_ref:
            return block.asset_ref
        payload = (
            f"{block.doc_id}:{block.page}:{block.reading_order}:{block.block_type}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _chunk_id(self, document_id: str, block_ids: List[str], text: str) -> str:
        payload = "|".join([document_id, *block_ids, text[:200]])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        return max(1, estimate_tokens(text))

    def _has_indexable_metadata(self, block: DocumentBlock) -> bool:
        if block.block_type != "figure" or not isinstance(block.metadata, dict):
            return False
        figure = block.metadata.get("figure")
        if not isinstance(figure, dict):
            return False
        visual = figure.get("visual")
        return isinstance(visual, dict) and bool(
            visual.get("ocr_text")
            or visual.get("caption")
            or visual.get("chart_summary")
            or visual.get("diagram_summary")
        )

    def _source_pages(self, block: DocumentBlock) -> List[int]:
        values = block.metadata.get("source_pages") if block.metadata else None
        pages = []
        if isinstance(values, list):
            for value in values:
                try:
                    pages.append(max(1, int(value)))
                except (TypeError, ValueError):
                    continue
        return sorted(set(pages)) or [max(1, int(block.page))]


def build_chunker(config: Dict[str, Any]) -> StructureAwareChunker:
    return StructureAwareChunker(config)
