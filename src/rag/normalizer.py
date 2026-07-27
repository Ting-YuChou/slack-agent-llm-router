"""Conservative layout normalization between document parsing and chunking."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.rag.parser import DocumentBlock, ParsedDocument


_FOOTNOTE_PREFIX = re.compile(r"^\s*(?:\(([^)]+)\)|\[([^\]]+)\])\s*")
_TERMINAL_PUNCTUATION = re.compile(r'[.!?。！？]["\')\]]?\s*$')
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?(?:\d+|[ivxlcdm]+)\s*$", re.IGNORECASE)


@dataclass
class LayoutNormalizationResult:
    document: ParsedDocument
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


class DocumentLayoutNormalizer:
    """Remove boilerplate and reconcile layout only when evidence is strong."""

    HEADER_TYPES = {"page_header", "page_footer"}
    HEADING_TYPES = {"section_header", "title", "heading"}
    SEAM_TYPES = {"text", "paragraph", "list_item"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.repeated_block_min_pages = max(
            3, int(self.config.get("repeated_block_min_pages", 3))
        )
        self.repeated_block_min_ratio = float(
            self.config.get("repeated_block_min_ratio", 0.60)
        )
        self.top_margin_ratio = float(self.config.get("top_margin_ratio", 0.10))
        self.bottom_margin_ratio = float(self.config.get("bottom_margin_ratio", 0.10))
        self.page_seam_margin_ratio = float(
            self.config.get("page_seam_margin_ratio", 0.20)
        )

    def normalize(self, document: ParsedDocument) -> LayoutNormalizationResult:
        if not self.enabled:
            return LayoutNormalizationResult(document=document)

        normalized = ParsedDocument(
            document_id=document.document_id,
            filename=document.filename,
            content_hash=document.content_hash,
            blocks=[copy.deepcopy(block) for block in document.blocks],
            metadata=copy.deepcopy(document.metadata),
        )
        normalized.blocks.sort(key=self._sort_key)
        self._assign_sections(normalized.blocks)
        blocks, excluded = self._remove_boilerplate(normalized.blocks)
        blocks, footnote_warnings, linked, unresolved = self._link_footnotes(blocks)
        blocks, stitched = self._stitch_page_seams(blocks)
        blocks, table_warnings, merged_tables = self._merge_cross_page_tables(blocks)
        self._link_related_figures(blocks)
        normalized.blocks = blocks
        normalized.metadata["layout_normalization"] = {
            "boilerplate_removed": len(excluded),
            "footnotes_linked": linked,
            "footnotes_unresolved": unresolved,
            "page_seams_stitched": stitched,
            "cross_page_tables_merged": merged_tables,
        }
        normalized.metadata["layout_normalization_audit"] = {
            "excluded_blocks": excluded
        }
        return LayoutNormalizationResult(
            document=normalized,
            warnings=[*footnote_warnings, *table_warnings],
            stats=dict(normalized.metadata["layout_normalization"]),
        )

    def _link_related_figures(self, blocks: Sequence[DocumentBlock]) -> None:
        figures = []
        for index, block in enumerate(blocks):
            if block.block_type != "figure":
                continue
            figure = block.metadata.get("figure")
            if not isinstance(figure, dict) or not figure.get("figure_id"):
                continue
            figures.append(
                (
                    str(figure["figure_id"]),
                    block.page,
                    list(block.metadata.get("section_path") or []),
                    index,
                )
            )
        for figure_id, page, figure_section, figure_index in figures:
            candidates = [
                (abs(block.page - page), abs(index - figure_index), index, block)
                for index, block in enumerate(blocks)
                if block.block_type in {"text", "paragraph", "list_item", "table"}
                and list(block.metadata.get("section_path") or []) == figure_section
                and abs(block.page - page) <= 1
            ]
            if not candidates:
                continue
            _page_distance, _order_distance, _index, target = min(
                candidates, key=lambda item: item[:3]
            )
            target.metadata.setdefault("related_figure_ids", []).append(figure_id)

    def _sort_key(self, block: DocumentBlock) -> tuple[Any, ...]:
        explicit = block.metadata.get("document_order")
        if explicit is not None:
            return (0, int(explicit), block.page, block.reading_order)
        bbox = block.bbox or [0.0, 0.0, 0.0, 0.0]
        return (1, block.page, block.reading_order, float(bbox[1]), float(bbox[0]))

    def _assign_sections(self, blocks: Sequence[DocumentBlock]) -> None:
        current: List[str] = []
        for block in blocks:
            existing = block.metadata.get("section_path")
            if isinstance(existing, list) and existing:
                current = [str(value) for value in existing]
            elif block.block_type in self.HEADING_TYPES and block.text.strip():
                current = [" ".join(block.text.split())[:160]]
            block.metadata.setdefault("section_path", list(current))
            block.metadata.setdefault("source_pages", [block.page])

    def _remove_boilerplate(
        self, blocks: Sequence[DocumentBlock]
    ) -> tuple[List[DocumentBlock], List[Dict[str, Any]]]:
        page_count = len({block.page for block in blocks})
        repeated: Dict[tuple[str, str], set[int]] = {}
        for block in blocks:
            zone = self._margin_zone(block)
            text_key = self._normalized_text(block.text)
            if (
                page_count >= self.repeated_block_min_pages
                and zone
                and text_key
                and not _PAGE_NUMBER.fullmatch(text_key)
            ):
                repeated.setdefault((zone, text_key), set()).add(block.page)

        repeated_keys = {
            key
            for key, pages in repeated.items()
            if len(pages) >= self.repeated_block_min_pages
            and len(pages) / max(page_count, 1) >= self.repeated_block_min_ratio
        }
        kept: List[DocumentBlock] = []
        excluded: List[Dict[str, Any]] = []
        for block in blocks:
            zone = self._margin_zone(block)
            text_key = self._normalized_text(block.text)
            explicit = block.block_type in self.HEADER_TYPES
            page_number = zone == "bottom" and bool(_PAGE_NUMBER.fullmatch(text_key))
            is_repeated = bool(zone and (zone, text_key) in repeated_keys)
            if explicit or page_number or is_repeated:
                excluded.append(
                    {
                        "block_id": self._block_id(block),
                        "page": block.page,
                        "block_type": block.block_type,
                        "reason": (
                            "explicit_header_footer"
                            if explicit
                            else "page_number"
                            if page_number
                            else "repeated_margin_block"
                        ),
                        "text": block.text,
                    }
                )
                continue
            kept.append(block)
        return kept, excluded

    def _margin_zone(self, block: DocumentBlock) -> Optional[str]:
        if block.block_type == "page_header":
            return "top"
        if block.block_type == "page_footer":
            return "bottom"
        if not block.bbox:
            return None
        page_height = self._page_height(block)
        if not page_height:
            return None
        top, bottom = self._vertical_bounds(block, page_height)
        if top <= page_height * self.top_margin_ratio:
            return "top"
        if bottom >= page_height * (1.0 - self.bottom_margin_ratio):
            return "bottom"
        return None

    def _link_footnotes(
        self, blocks: Sequence[DocumentBlock]
    ) -> tuple[List[DocumentBlock], List[str], int, int]:
        output: List[DocumentBlock] = []
        warnings: List[str] = []
        linked = 0
        unresolved = 0
        for block in blocks:
            if block.block_type != "footnote":
                output.append(block)
                continue
            marker, body = self._footnote_parts(block.text)
            target = self._referenced_footnote_target(block, output)
            if target is None and marker:
                needle_variants = (f"({marker})", f"[{marker}]")
                for candidate in reversed(output):
                    if (
                        candidate.page != block.page
                        or candidate.block_type == "footnote"
                    ):
                        continue
                    if any(needle in candidate.text for needle in needle_variants):
                        target = candidate
                        break
            if target is not None:
                label = f"Footnote ({marker})" if marker else "Footnote"
                target.text = f"{target.text.rstrip()}\n\n{label}: {body}".strip()
                target.metadata.setdefault("linked_footnote_block_ids", []).append(
                    self._block_id(block)
                )
                target.metadata["source_pages"] = sorted(
                    set(target.metadata.get("source_pages") or [target.page])
                    | set(block.metadata.get("source_pages") or [block.page])
                )
                linked += 1
                continue
            label = f"Footnote ({marker})" if marker else "Footnote"
            block.text = f"{label}: {body}".strip()
            block.metadata["is_unresolved_footnote"] = True
            output.append(block)
            unresolved += 1
            warning_marker = f" ({marker})" if marker else ""
            warnings.append(f"Unresolved footnote{warning_marker} on page {block.page}")
        return output, warnings, linked, unresolved

    def _referenced_footnote_target(
        self, footnote: DocumentBlock, candidates: Sequence[DocumentBlock]
    ) -> Optional[DocumentBlock]:
        parent_ref = str(footnote.metadata.get("parent_ref") or "")
        source_id = self._block_id(footnote)
        for candidate in reversed(candidates):
            candidate_id = self._block_id(candidate)
            child_refs = {
                str(value) for value in candidate.metadata.get("child_refs", [])
            }
            if parent_ref and parent_ref == candidate_id:
                return candidate
            if source_id in child_refs:
                return candidate
        return None

    def _stitch_page_seams(
        self, blocks: Sequence[DocumentBlock]
    ) -> tuple[List[DocumentBlock], int]:
        output: List[DocumentBlock] = []
        stitched = 0
        index = 0
        while index < len(blocks):
            current = blocks[index]
            if index + 1 >= len(blocks):
                output.append(current)
                break
            following = blocks[index + 1]
            if self._can_stitch(current, following):
                current.text = self._join_seam_text(current.text, following.text)
                current.metadata["source_pages"] = sorted(
                    set(current.metadata.get("source_pages") or [current.page])
                    | set(following.metadata.get("source_pages") or [following.page])
                )
                current.metadata.setdefault("stitched_block_ids", []).append(
                    self._block_id(following)
                )
                output.append(current)
                stitched += 1
                index += 2
                continue
            output.append(current)
            index += 1
        return output, stitched

    def _can_stitch(self, left: DocumentBlock, right: DocumentBlock) -> bool:
        if (
            left.block_type not in self.SEAM_TYPES
            or right.block_type not in self.SEAM_TYPES
        ):
            return False
        if right.page != left.page + 1:
            return False
        if left.metadata.get("section_path") != right.metadata.get("section_path"):
            return False
        if _TERMINAL_PUNCTUATION.search(left.text):
            return False
        return self._near_bottom(left) and self._near_top(right)

    def _merge_cross_page_tables(
        self, blocks: Sequence[DocumentBlock]
    ) -> tuple[List[DocumentBlock], List[str], int]:
        output: List[DocumentBlock] = []
        warnings: List[str] = []
        merged_count = 0
        index = 0
        while index < len(blocks):
            current = blocks[index]
            if index + 1 >= len(blocks):
                output.append(current)
                break
            following = blocks[index + 1]
            if self._can_merge_tables(current, following):
                self._merge_table_blocks(current, following)
                output.append(current)
                merged_count += 1
                index += 2
                continue
            if (
                current.block_type == "table"
                and following.block_type == "table"
                and following.page == current.page + 1
                and self._near_bottom(current)
                and self._near_top(following)
            ):
                warnings.append(
                    f"Ambiguous cross-page tables on pages {current.page}-{following.page}"
                )
            output.append(current)
            index += 1
        return output, warnings, merged_count

    def _can_merge_tables(self, left: DocumentBlock, right: DocumentBlock) -> bool:
        if left.block_type != "table" or right.block_type != "table":
            return False
        if right.page != left.page + 1:
            return False
        if left.metadata.get("section_path") != right.metadata.get("section_path"):
            return False
        if not (self._near_bottom(left) and self._near_top(right)):
            return False
        left_table = left.metadata.get("table")
        right_table = right.metadata.get("table")
        if not isinstance(left_table, dict) or not isinstance(right_table, dict):
            return False
        left_signature = self._column_signature(left_table)
        right_signature = self._column_signature(right_table)
        return bool(left_signature) and left_signature == right_signature

    def _merge_table_blocks(
        self, target: DocumentBlock, continuation: DocumentBlock
    ) -> None:
        table = target.metadata["table"]
        continuation_table = continuation.metadata["table"]
        existing_rows = list(table.get("rows") or [])
        appended_rows = copy.deepcopy(list(continuation_table.get("rows") or []))
        if appended_rows and self._row_matches_columns(
            appended_rows[0], table.get("columns") or []
        ):
            appended_rows = appended_rows[1:]
        for offset, row in enumerate(appended_rows, start=len(existing_rows) + 1):
            row["row_id"] = f"r{offset}"
            row["row_index"] = offset
            for cell in row.get("cells") or []:
                cell.setdefault("page", continuation.page)
        table["rows"] = [*existing_rows, *appended_rows]
        table["page_end"] = max(
            continuation.page,
            int(continuation_table.get("page_end") or continuation.page),
        )
        table["source_pages"] = sorted(
            set(table.get("source_pages") or [target.page])
            | set(continuation_table.get("source_pages") or [continuation.page])
        )
        markdown_rows = [
            str(table.get("markdown") or "").strip(),
            self._without_repeated_header(
                str(continuation_table.get("markdown") or ""),
                table.get("columns") or [],
            ),
        ]
        table["markdown"] = "\n".join(value for value in markdown_rows if value)
        table["rowwise_text"] = "\n".join(
            str(row.get("semantic_text") or "").strip()
            for row in table["rows"]
            if str(row.get("semantic_text") or "").strip()
        )
        target.text = "\n\n".join(
            value for value in (table["markdown"], table["rowwise_text"]) if value
        )
        target.metadata["source_pages"] = table["source_pages"]
        target.metadata.setdefault("merged_table_block_ids", []).append(
            self._block_id(continuation)
        )

    def _near_bottom(self, block: DocumentBlock) -> bool:
        if not block.bbox:
            return bool(block.metadata.get("continues_on_next_page"))
        height = self._page_height(block)
        if not height:
            return bool(block.metadata.get("continues_on_next_page"))
        _top, bottom = self._vertical_bounds(block, height)
        return bottom >= height * (1.0 - self.page_seam_margin_ratio)

    def _near_top(self, block: DocumentBlock) -> bool:
        if not block.bbox:
            return bool(block.metadata.get("continued_from_previous_page"))
        height = self._page_height(block)
        if not height:
            return bool(block.metadata.get("continued_from_previous_page"))
        top, _bottom = self._vertical_bounds(block, height)
        return top <= height * self.page_seam_margin_ratio

    def _vertical_bounds(
        self, block: DocumentBlock, page_height: float
    ) -> tuple[float, float]:
        low = min(float(block.bbox[1]), float(block.bbox[3]))
        high = max(float(block.bbox[1]), float(block.bbox[3]))
        provenance = block.metadata.get("provenance") or []
        first = provenance[0] if provenance and isinstance(provenance[0], dict) else {}
        bbox = first.get("bbox") if isinstance(first.get("bbox"), dict) else {}
        origin = str(
            first.get("coord_origin")
            or bbox.get("coord_origin")
            or block.metadata.get("coord_origin")
            or ""
        ).lower()
        if "bottom" in origin:
            return page_height - high, page_height - low
        return low, high

    def _page_height(self, block: DocumentBlock) -> Optional[float]:
        value = block.metadata.get("page_height")
        try:
            height = float(value)
        except (TypeError, ValueError):
            return None
        return height if height > 0 else None

    def _footnote_parts(self, text: str) -> tuple[str, str]:
        match = _FOOTNOTE_PREFIX.match(text)
        if not match:
            return "", " ".join(text.split())
        marker = str(match.group(1) or match.group(2) or "").strip()
        return marker, " ".join(text[match.end() :].split())

    def _column_signature(self, table: Dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            self._normalized_text(value) for value in table.get("columns") or []
        )

    def _row_matches_columns(self, row: Dict[str, Any], columns: Sequence[Any]) -> bool:
        cells = row.get("cells") or []
        values = [
            cell.get("value") or cell.get("text") or ""
            for cell in cells
            if isinstance(cell, dict)
        ]
        return bool(columns) and tuple(
            self._normalized_text(value) for value in values
        ) == tuple(self._normalized_text(value) for value in columns)

    def _without_repeated_header(self, markdown: str, columns: Sequence[Any]) -> str:
        lines = [line for line in markdown.splitlines() if line.strip()]
        if not lines:
            return ""
        expected = self._normalized_text(" | ".join(str(value) for value in columns))
        if self._normalized_text(lines[0]) == expected:
            lines = lines[1:]
        return "\n".join(lines)

    def _join_seam_text(self, left: str, right: str) -> str:
        left = left.rstrip()
        right = right.lstrip()
        if left.endswith("-") and right and right[0].islower():
            return f"{left[:-1]}{right}"
        return f"{left} {right}".strip()

    def _normalized_text(self, value: Any) -> str:
        return " ".join(str(value or "").lower().split())

    def _block_id(self, block: DocumentBlock) -> str:
        return str(
            block.asset_ref
            or block.metadata.get("source_block_id")
            or f"{block.page}:{block.reading_order}:{block.block_type}"
        )
