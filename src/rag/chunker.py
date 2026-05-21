"""Structure-aware chunking for parsed school documents."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.parser import DocumentBlock, ParsedDocument


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
    STANDALONE_BLOCK_TYPES = {"table"}

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.chunk_size_tokens = int(self.config.get("chunk_size_tokens", 900))
        self.chunk_overlap_tokens = int(self.config.get("chunk_overlap_tokens", 120))
        self.max_chunk_chars = int(self.config.get("max_chunk_chars", 6000))

    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        current_blocks: List[DocumentBlock] = []
        current_section: List[str] = []
        current_tokens = 0

        for block in sorted(
            document.blocks, key=lambda item: (item.page, item.reading_order)
        ):
            if not block.text.strip():
                continue

            if block.block_type in self.HEADER_BLOCK_TYPES:
                current_section = self._next_section_path(current_section, block.text)

            block_tokens = self._estimate_tokens(block.text)
            if block.block_type in self.STANDALONE_BLOCK_TYPES:
                if current_blocks:
                    chunks.append(
                        self._build_chunk(document, current_blocks, current_section)
                    )
                    current_blocks = []
                    current_tokens = 0
                table_chunks = self._build_table_chunks(
                    document, block, current_section
                )
                if table_chunks:
                    chunks.extend(table_chunks)
                    continue
                chunks.extend(self._split_large_block(document, block, current_section))
                continue

            if (
                current_blocks
                and current_tokens + block_tokens > self.chunk_size_tokens
            ):
                chunks.append(
                    self._build_chunk(document, current_blocks, current_section)
                )
                current_blocks = self._overlap_tail(current_blocks)
                current_tokens = sum(
                    self._estimate_tokens(item.text) for item in current_blocks
                )

            if block_tokens > self.chunk_size_tokens:
                if current_blocks:
                    chunks.append(
                        self._build_chunk(document, current_blocks, current_section)
                    )
                    current_blocks = []
                    current_tokens = 0
                chunks.extend(self._split_large_block(document, block, current_section))
                continue

            current_blocks.append(block)
            current_tokens += block_tokens

        if current_blocks:
            chunks.append(self._build_chunk(document, current_blocks, current_section))

        return chunks

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
        return DocumentChunk(
            chunk_id=self._chunk_id(document.document_id, block_ids, text),
            document_id=document.document_id,
            text=text[: self.max_chunk_chars],
            page_start=block.page,
            page_end=block.page,
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
        page_start = min(block.page for block in blocks)
        page_end = max(block.page for block in blocks)
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
        words = text.split()
        if not words:
            return []
        step = max(self.chunk_size_tokens - self.chunk_overlap_tokens, 1)
        pieces = []
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + self.chunk_size_tokens])
            if piece:
                pieces.append(piece)
        return pieces

    def _overlap_tail(self, blocks: List[DocumentBlock]) -> List[DocumentBlock]:
        if self.chunk_overlap_tokens <= 0:
            return []
        total = 0
        selected: List[DocumentBlock] = []
        for block in reversed(blocks):
            total += self._estimate_tokens(block.text)
            selected.append(block)
            if total >= self.chunk_overlap_tokens:
                break
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
        return max(1, len(text.split()))


def build_chunker(config: Dict[str, Any]) -> StructureAwareChunker:
    return StructureAwareChunker(config)
