"""Document parsing contracts for school-document RAG ingestion."""

import hashlib
import logging
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class DocumentBlock:
    """One parsed block from a source document."""

    doc_id: str
    page: int
    bbox: Optional[List[float]]
    block_type: str
    text: str = ""
    reading_order: int = 0
    confidence: Optional[float] = None
    asset_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Normalized parser output used by chunking and indexing."""

    document_id: str
    filename: str
    content_hash: str
    blocks: List[DocumentBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser(ABC):
    """Base interface for parser backends."""

    @abstractmethod
    def parse_bytes(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        """Parse raw document bytes into normalized blocks."""


class TextDocumentParser(DocumentParser):
    """Small deterministic parser for plain text and tests."""

    def parse_bytes(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        resolved_doc_id = document_id or str(uuid.uuid4())
        content_hash = hashlib.sha256(content).hexdigest()
        text = _decode_text(content)
        blocks = []
        for index, paragraph in enumerate(_split_paragraphs(text), start=1):
            blocks.append(
                DocumentBlock(
                    doc_id=resolved_doc_id,
                    page=1,
                    bbox=None,
                    block_type=_guess_block_type(paragraph),
                    text=paragraph,
                    reading_order=index,
                )
            )
        return ParsedDocument(
            document_id=resolved_doc_id,
            filename=filename,
            content_hash=content_hash,
            blocks=blocks,
            metadata=dict(metadata or {}),
        )


class DoclingParser(DocumentParser):
    """Docling-backed PDF parser with a text fallback for non-PDF files."""

    TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.text_parser = TextDocumentParser()

    def parse_bytes(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        if suffix in self.TEXT_EXTENSIONS:
            return self.text_parser.parse_bytes(
                content=content,
                filename=filename,
                document_id=document_id,
                metadata=metadata,
            )

        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:
            raise RuntimeError(
                "Docling is required to parse PDFs. Install the runtime dependency "
                "or switch rag.parser.provider to 'text' for tests."
            ) from exc

        resolved_doc_id = document_id or str(uuid.uuid4())
        content_hash = hashlib.sha256(content).hexdigest()
        with tempfile.NamedTemporaryFile(suffix=suffix or ".pdf") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            result = self._build_converter(DocumentConverter).convert(
                source=temp_file.name
            )

        document = getattr(result, "document", result)
        blocks = self._extract_blocks(document, resolved_doc_id)
        if not blocks:
            markdown = _safe_export_markdown(document)
            blocks = (
                [
                    DocumentBlock(
                        doc_id=resolved_doc_id,
                        page=1,
                        bbox=None,
                        block_type="text",
                        text=markdown,
                        reading_order=1,
                    )
                ]
                if markdown
                else []
            )

        return ParsedDocument(
            document_id=resolved_doc_id,
            filename=filename,
            content_hash=content_hash,
            blocks=blocks,
            metadata=dict(metadata or {}),
        )

    def _extract_blocks(self, document: Any, document_id: str) -> List[DocumentBlock]:
        payload = _safe_export_dict(document)
        if not payload:
            return []

        raw_blocks: List[Dict[str, Any]] = []
        for key in ("texts", "tables", "pictures", "groups"):
            values = payload.get(key)
            if isinstance(values, list):
                raw_blocks.extend(value for value in values if isinstance(value, dict))

        blocks = []
        for index, raw in enumerate(raw_blocks, start=1):
            block_type = _normalize_block_type(
                raw.get("label") or raw.get("type") or raw.get("name")
            )
            page, bbox = _extract_page_bbox(raw)
            table_metadata = (
                _extract_table_metadata(raw, page=page, bbox=bbox, fallback_index=index)
                if block_type == "table"
                else None
            )
            text = (
                _table_text_from_metadata(table_metadata)
                if table_metadata
                else _extract_text(raw)
            )
            if not text and block_type == "text":
                continue
            block_metadata = {
                "raw_label": str(raw.get("label") or ""),
            }
            if table_metadata:
                block_metadata["table"] = table_metadata
            if block_type == "figure":
                block_metadata["figure"] = _extract_figure_metadata(
                    raw,
                    page=page,
                    bbox=bbox,
                    fallback_index=index,
                )
            blocks.append(
                DocumentBlock(
                    doc_id=document_id,
                    page=page,
                    bbox=bbox,
                    block_type=block_type,
                    text=text,
                    reading_order=int(raw.get("reading_order") or index),
                    confidence=_safe_float(raw.get("confidence")),
                    asset_ref=str(raw.get("self_ref") or raw.get("id") or "") or None,
                    metadata=block_metadata,
                )
            )
        blocks.sort(key=lambda block: (block.page, block.reading_order))
        return blocks

    def _build_converter(self, converter_cls: Any) -> Any:
        kwargs = dict(self.config.get("converter_kwargs") or {})
        if kwargs:
            return converter_cls(**kwargs)
        return converter_cls()


def build_document_parser(config: Dict[str, Any]) -> DocumentParser:
    provider = str(config.get("provider", "docling")).lower()
    if provider == "text":
        return TextDocumentParser()
    return DoclingParser(config)


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [" ".join(part.split()) for part in text.split("\n\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if paragraphs:
        return paragraphs
    compact = " ".join(text.split())
    return [compact] if compact else []


def _guess_block_type(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("#"):
        return "section_header"
    if len(stripped) <= 120 and not stripped.endswith((".", "?", "!")):
        return "section_header"
    return "text"


def _safe_export_dict(document: Any) -> Dict[str, Any]:
    if isinstance(document, dict):
        return document
    for method_name in ("export_to_dict", "model_dump", "dict"):
        method = getattr(document, method_name, None)
        if not callable(method):
            continue
        try:
            payload = method()
            if isinstance(payload, dict):
                return payload
        except Exception:
            logger.debug("Docling %s export failed", method_name, exc_info=True)
    return {}


def _safe_export_markdown(document: Any) -> str:
    method = getattr(document, "export_to_markdown", None)
    if callable(method):
        try:
            return str(method()).strip()
        except Exception:
            logger.debug("Docling markdown export failed", exc_info=True)
    return ""


def _extract_text(raw: Dict[str, Any]) -> str:
    table_text = _extract_table_text(raw)
    if table_text:
        return table_text
    for key in ("text", "orig", "content"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    if isinstance(raw.get("data"), dict):
        for key in ("text", "content", "markdown"):
            value = raw["data"].get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        collected = _collect_text_values(raw["data"])
        if collected:
            return " ".join(collected)
    return ""


def _extract_table_text(raw: Dict[str, Any]) -> str:
    table = _extract_table_metadata(raw)
    if table:
        return _table_text_from_metadata(table)
    return ""


def _extract_table_metadata(
    raw: Dict[str, Any],
    *,
    page: int = 1,
    bbox: Optional[List[float]] = None,
    fallback_index: int = 0,
) -> Dict[str, Any]:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    cells = data.get("table_cells") or data.get("cells")
    table_id = str(
        raw.get("self_ref") or raw.get("id") or data.get("self_ref") or ""
    ).strip()
    if not table_id:
        table_id = f"table-{fallback_index or 1}"
    if isinstance(cells, list):
        rows: Dict[int, Dict[int, Dict[str, Any]]] = {}
        header_rows = set()
        for index, cell in enumerate(cells):
            cell_payload = _as_dict(cell)
            if not cell_payload:
                continue
            text = _first_text_value(cell_payload)
            if not text:
                continue
            row = _safe_int(
                _first_present(
                    cell_payload,
                    "start_row_offset_idx",
                    "row",
                    "row_index",
                ),
                index,
            )
            col = _safe_int(
                _first_present(
                    cell_payload,
                    "start_col_offset_idx",
                    "col",
                    "column",
                    "column_index",
                ),
                0,
            )
            cell_bbox = _normalize_bbox(
                cell_payload.get("bbox") or cell_payload.get("box")
            )
            if bool(
                _first_present(
                    cell_payload,
                    "column_header",
                    "is_column_header",
                    "is_header",
                    "header",
                )
            ):
                header_rows.add(row)
            rows.setdefault(row, {})[col] = {
                "row_index": row,
                "col_index": col,
                "text": text,
                "bbox": cell_bbox,
                "page": page,
            }
        if rows:
            if not header_rows:
                header_rows.add(min(rows))
            return _build_table_metadata(
                table_id=table_id,
                rows=rows,
                header_rows=sorted(header_rows),
                page=page,
                bbox=bbox,
            )

    rows_value = data.get("rows") or data.get("grid")
    if isinstance(rows_value, list):
        rows: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for row_index, row in enumerate(rows_value):
            if isinstance(row, list):
                for col_index, cell in enumerate(row):
                    text = (
                        cell
                        if isinstance(cell, str)
                        else _first_text_value(_as_dict(cell))
                    )
                    text = " ".join(str(text).split())
                    if not text:
                        continue
                    rows.setdefault(row_index, {})[col_index] = {
                        "row_index": row_index,
                        "col_index": col_index,
                        "text": text,
                        "bbox": None,
                        "page": page,
                    }
            elif isinstance(row, dict):
                for col_index, value in enumerate(_collect_text_values(row)):
                    rows.setdefault(row_index, {})[col_index] = {
                        "row_index": row_index,
                        "col_index": col_index,
                        "text": value,
                        "bbox": None,
                        "page": page,
                    }
        if rows:
            return _build_table_metadata(
                table_id=table_id,
                rows=rows,
                header_rows=[min(rows)],
                page=page,
                bbox=bbox,
            )

    return {}


def _build_table_metadata(
    *,
    table_id: str,
    rows: Dict[int, Dict[int, Dict[str, Any]]],
    header_rows: List[int],
    page: int,
    bbox: Optional[List[float]],
) -> Dict[str, Any]:
    sorted_header_rows = sorted(set(header_rows))
    header_map: Dict[int, str] = {}
    for header_row in sorted_header_rows:
        for col_index, cell in rows.get(header_row, {}).items():
            text = str(cell.get("text") or "").strip()
            if text:
                header_map[col_index] = text
    first_data_row = next(
        (row for row in sorted(rows) if row not in sorted_header_rows),
        None,
    )
    label_col = _infer_label_column(header_map, rows.get(first_data_row, {}))
    data_columns = _infer_data_columns(
        header_map, rows.get(first_data_row, {}), label_col
    )

    rendered_rows = []
    structured_rows = []
    for row_index in sorted(rows):
        row_cells = rows[row_index]
        rendered_rows.append(
            " | ".join(
                str(row_cells[col].get("text") or "").strip()
                for col in sorted(row_cells)
                if str(row_cells[col].get("text") or "").strip()
            )
        )
        if row_index in sorted_header_rows:
            continue
        row_label = _row_label(row_cells, label_col)
        values = {}
        for col_index, column in data_columns:
            text = str(row_cells.get(col_index, {}).get("text") or "").strip()
            if text:
                values[column] = text
        semantic_text = _row_semantic_text(row_label, values)
        structured_rows.append(
            {
                "row_id": f"r{len(structured_rows) + 1}",
                "row_index": row_index,
                "label": row_label,
                "values": values,
                "cells": [row_cells[col] for col in sorted(row_cells)],
                "semantic_text": semantic_text,
            }
        )

    columns = [column for _col, column in data_columns]
    if label_col is not None:
        label_name = header_map.get(label_col) or "Item"
        columns = [label_name, *columns]
    markdown = "\n".join(row for row in rendered_rows if row)
    rowwise_text = "\n".join(
        row["semantic_text"] for row in structured_rows if row.get("semantic_text")
    )
    return {
        "table_id": table_id,
        "page": page,
        "bbox": bbox,
        "columns": columns,
        "header_rows": sorted_header_rows,
        "rows": structured_rows,
        "markdown": markdown,
        "rowwise_text": rowwise_text,
    }


def _infer_label_column(
    header_map: Dict[int, str], first_data_row: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    if not first_data_row:
        return None
    data_cols = sorted(first_data_row)
    if not data_cols:
        return None
    if data_cols[0] not in header_map:
        return data_cols[0]
    if len(header_map) == max(len(data_cols) - 1, 0):
        return data_cols[0]
    first_header = header_map.get(data_cols[0], "").strip().lower()
    if first_header in {"", "item", "description", "deadline", "event"}:
        return data_cols[0]
    return None


def _infer_data_columns(
    header_map: Dict[int, str],
    first_data_row: Dict[int, Dict[str, Any]],
    label_col: Optional[int],
) -> List[tuple[int, str]]:
    data_cols = [col for col in sorted(first_data_row) if col != label_col]
    header_items = [
        (col, value) for col, value in sorted(header_map.items()) if col != label_col
    ]
    if len(header_items) == len(data_cols):
        return [
            (data_col, header_items[index][1])
            for index, data_col in enumerate(data_cols)
        ]
    columns = []
    for col in data_cols:
        columns.append((col, header_map.get(col) or f"Column {col + 1}"))
    return columns


def _row_label(row_cells: Dict[int, Dict[str, Any]], label_col: Optional[int]) -> str:
    if label_col is not None:
        return str(row_cells.get(label_col, {}).get("text") or "").strip()
    first_col = min(row_cells) if row_cells else None
    if first_col is None:
        return ""
    return str(row_cells.get(first_col, {}).get("text") or "").strip()


def _row_semantic_text(row_label: str, values: Dict[str, str]) -> str:
    assignments = "; ".join(f"{column} = {value}" for column, value in values.items())
    if row_label and assignments:
        return f"{row_label}: {assignments}"
    return assignments or row_label


def _table_text_from_metadata(table: Optional[Dict[str, Any]]) -> str:
    if not table:
        return ""
    parts = [
        str(table.get("markdown") or "").strip(),
        str(table.get("rowwise_text") or "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def _extract_figure_metadata(
    raw: Dict[str, Any],
    *,
    page: int,
    bbox: Optional[List[float]],
    fallback_index: int,
) -> Dict[str, Any]:
    asset_ref = str(raw.get("self_ref") or raw.get("id") or "").strip()
    figure_id = asset_ref.strip("#/").replace("/", "-") if asset_ref else ""
    if not figure_id:
        figure_id = f"figure-{fallback_index or 1}"
    return {
        "figure_id": figure_id,
        "page": page,
        "bbox": bbox,
        "asset_ref": asset_ref or None,
        "raw_label": str(raw.get("label") or "figure"),
    }


def _normalize_block_type(value: Any) -> str:
    normalized = str(value or "text").strip().lower()
    mapping = {
        "paragraph": "text",
        "text": "text",
        "section_header": "section_header",
        "title": "section_header",
        "document_index": "section_header",
        "table": "table",
        "picture": "figure",
        "image": "figure",
        "figure": "figure",
        "list_item": "list_item",
    }
    return mapping.get(normalized, normalized or "text")


def _extract_page_bbox(raw: Dict[str, Any]) -> tuple[int, Optional[List[float]]]:
    provenance = raw.get("prov")
    first_prov = None
    if isinstance(provenance, list) and provenance:
        first_prov = _as_dict(provenance[0])
    elif isinstance(provenance, dict):
        first_prov = provenance

    source = first_prov if isinstance(first_prov, dict) else raw
    page = _safe_int(source.get("page_no") or source.get("page"), 1)
    bbox = _normalize_bbox(source.get("bbox") or source.get("box"))
    return max(page, 1), bbox


def _normalize_bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        candidates = (
            ("l", "t", "r", "b"),
            ("left", "top", "right", "bottom"),
            ("x0", "y0", "x1", "y1"),
        )
        for keys in candidates:
            if all(key in value for key in keys):
                return [_safe_float(value[key]) or 0.0 for key in keys]
    object_payload = _as_dict(value)
    if object_payload and object_payload is not value:
        return _normalize_bbox(object_payload)
    for method_name in ("as_tuple", "to_tuple"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _normalize_bbox(method())
            except Exception:
                logger.debug("Docling bbox %s failed", method_name, exc_info=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = [_safe_float(item) for item in list(value)[:4]]
        if len(numbers) == 4 and all(number is not None for number in numbers):
            return [float(number) for number in numbers if number is not None]
    return None


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                payload = method()
                if isinstance(payload, dict):
                    return payload
            except Exception:
                logger.debug(
                    "Docling object %s export failed", method_name, exc_info=True
                )
    payload = {
        key: getattr(value, key)
        for key in (
            "l",
            "t",
            "r",
            "b",
            "left",
            "top",
            "right",
            "bottom",
            "x0",
            "y0",
            "x1",
            "y1",
            "page_no",
            "page",
            "bbox",
            "text",
        )
        if hasattr(value, key)
    }
    return payload


def _first_text_value(payload: Dict[str, Any]) -> str:
    for key in ("text", "orig", "content", "markdown"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _collect_text_values(value: Any) -> List[str]:
    if isinstance(value, str):
        stripped = " ".join(value.split())
        return [stripped] if stripped else []
    if isinstance(value, dict):
        collected: List[str] = []
        for key, nested in value.items():
            if key in {"self_ref", "parent", "children", "prov", "bbox", "label"}:
                continue
            collected.extend(_collect_text_values(nested))
        return collected
    if isinstance(value, list):
        collected = []
        for nested in value:
            collected.extend(_collect_text_values(nested))
        return collected
    return []


def _first_present(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
