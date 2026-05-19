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
            text = _extract_text(raw)
            block_type = _normalize_block_type(
                raw.get("label") or raw.get("type") or raw.get("name")
            )
            if not text and block_type == "text":
                continue
            page, bbox = _extract_page_bbox(raw)
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
                    metadata={
                        "raw_label": str(raw.get("label") or ""),
                    },
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
    data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
    cells = data.get("table_cells") or data.get("cells")
    if isinstance(cells, list):
        rows: Dict[int, Dict[int, str]] = {}
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
            rows.setdefault(row, {})[col] = text
        if rows:
            rendered = []
            for row_index in sorted(rows):
                row = rows[row_index]
                rendered.append(" | ".join(row[col] for col in sorted(row)))
            return "\n".join(rendered)

    rows_value = data.get("rows") or data.get("grid")
    if isinstance(rows_value, list):
        rendered_rows = []
        for row in rows_value:
            if isinstance(row, list):
                values = [
                    _first_text_value(_as_dict(cell))
                    if not isinstance(cell, str)
                    else cell
                    for cell in row
                ]
                values = [value for value in values if value]
                if values:
                    rendered_rows.append(" | ".join(values))
            elif isinstance(row, dict):
                values = _collect_text_values(row)
                if values:
                    rendered_rows.append(" | ".join(values))
        if rendered_rows:
            return "\n".join(rendered_rows)

    return ""


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
