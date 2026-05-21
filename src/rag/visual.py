"""Image-aware helpers for school-document RAG ingestion."""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class VisualFigureResult:
    """Normalized output from optional visual OCR/caption/embedding services."""

    ocr_text: str = ""
    markdown: str = ""
    detected_elements: List[Dict[str, Any]] = field(default_factory=list)
    chart_data: Any = None
    formula_text: str = ""
    caption: str = ""
    diagram_summary: str = ""
    chart_summary: str = ""
    structured_json: Any = None
    warnings: List[str] = field(default_factory=list)
    visual_embedding: Optional[List[float]] = None
    ocr_provider: str = ""
    caption_provider: str = ""
    visual_embedding_provider: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ocr_text": self.ocr_text,
            "markdown": self.markdown,
            "detected_elements": self.detected_elements,
            "chart_data": self.chart_data,
            "formula_text": self.formula_text,
            "caption": self.caption,
            "diagram_summary": self.diagram_summary,
            "chart_summary": self.chart_summary,
            "structured_json": self.structured_json,
            "warnings": self.warnings,
            "ocr_provider": self.ocr_provider,
            "caption_provider": self.caption_provider,
            "visual_embedding_provider": self.visual_embedding_provider,
        }
        if self.visual_embedding is not None:
            payload["visual_embedding"] = list(self.visual_embedding)
        return payload


class FigureCropper:
    """Crop Docling figure regions from a PDF into local image assets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        storage = dict(self.config.get("storage", {}) or {})
        self.assets_dir = Path(storage.get("assets_dir") or "data/rag/assets")
        self.crop_dpi = int(self.config.get("crop_dpi", 180))
        self.min_crop_pixels = int(self.config.get("min_crop_pixels", 64))
        self.max_crops_per_document = int(self.config.get("max_crops_per_document", 50))

    def crop_figures(
        self,
        *,
        content: bytes,
        filename: str,
        parsed_document: Any,
        knowledge_base_id: str,
    ) -> List[str]:
        figures = [
            block
            for block in getattr(parsed_document, "blocks", [])
            if getattr(block, "block_type", "") == "figure"
        ][: self.max_crops_per_document]
        if not figures:
            return []
        for block in figures:
            figure = _figure_metadata(block)
            block.metadata["figure"] = figure

        warnings: List[str] = []
        try:
            import fitz  # type: ignore
        except Exception as exc:
            warning = f"Figure crop skipped; PyMuPDF is unavailable: {exc}"
            for block in figures:
                block.metadata["figure"]["crop_status"] = "skipped"
                block.metadata["figure"]["crop_warning"] = warning
            return [warning]

        try:
            pdf = fitz.open(
                stream=content, filetype=Path(filename).suffix.lstrip(".") or "pdf"
            )
        except Exception as exc:
            warning = f"Figure crop skipped; PDF renderer failed: {exc}"
            for block in figures:
                block.metadata["figure"]["crop_status"] = "failed"
                block.metadata["figure"]["crop_warning"] = warning
            return [warning]

        try:
            target_dir = (
                self.assets_dir
                / _safe_path_part(knowledge_base_id)
                / _safe_path_part(parsed_document.document_id)
                / "figures"
            )
            _clear_directory(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            scale = max(self.crop_dpi, 72) / 72.0
            matrix = fitz.Matrix(scale, scale)
            for block in figures:
                figure = block.metadata["figure"]
                page_index = max(int(getattr(block, "page", 1)) - 1, 0)
                if page_index >= len(pdf):
                    figure["crop_status"] = "failed"
                    figure["crop_warning"] = "figure page is outside the PDF page range"
                    warnings.append(figure["crop_warning"])
                    continue
                page = pdf[page_index]
                crop_scope = "bbox" if getattr(block, "bbox", None) else "page"
                clip = _fitz_clip_rect(fitz, page.rect, getattr(block, "bbox", None))
                if clip is None:
                    crop_scope = "page"
                pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
                if (
                    pixmap.width < self.min_crop_pixels
                    or pixmap.height < self.min_crop_pixels
                ):
                    figure["crop_status"] = "skipped"
                    figure["crop_warning"] = "figure crop is below min_crop_pixels"
                    warnings.append(figure["crop_warning"])
                    continue
                image_path = target_dir / f"{_safe_path_part(figure['figure_id'])}.png"
                pixmap.save(str(image_path))
                image_bytes = image_path.read_bytes()
                figure.update(
                    {
                        "image_ref": str(image_path),
                        "content_hash": hashlib.sha256(image_bytes).hexdigest(),
                        "width": pixmap.width,
                        "height": pixmap.height,
                        "crop_scope": crop_scope,
                        "crop_status": "completed",
                    }
                )
        finally:
            pdf.close()
        return warnings

    def delete_document_assets(self, document_id: str, knowledge_base_id: str) -> None:
        target_dir = (
            self.assets_dir
            / _safe_path_part(knowledge_base_id)
            / _safe_path_part(document_id)
        )
        _clear_directory(target_dir)


class VisualProcessor:
    """Base visual processing interface."""

    async def process_figure(self, figure: Dict[str, Any]) -> VisualFigureResult:
        return VisualFigureResult()

    async def embed_query(self, query: str) -> Optional[List[float]]:
        return None


class LocalHttpVisualProcessor(VisualProcessor):
    """Calls optional local HTTP services for OCR/captioning/visual embeddings."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.ocr_config = dict(self.config.get("ocr", {}) or {})
        self.caption_config = dict(self.config.get("caption", {}) or {})
        self.embedding_config = dict(self.config.get("embedding", {}) or {})

    async def process_figure(self, figure: Dict[str, Any]) -> VisualFigureResult:
        result = VisualFigureResult(
            ocr_provider=str(self.ocr_config.get("provider") or ""),
            caption_provider=str(self.caption_config.get("provider") or ""),
            visual_embedding_provider=str(self.embedding_config.get("provider") or ""),
        )
        if self.ocr_config.get("enabled", False):
            ocr_payload = await self._call_provider(
                self.ocr_config,
                {
                    "task": "document_parse",
                    "model": self.ocr_config.get("model"),
                    "image_ref": figure.get("image_ref"),
                    "figure": _json_safe(figure),
                },
            )
            self._merge_ocr_payload(result, ocr_payload)
        if self.caption_config.get("enabled", False):
            caption_payload = await self._call_provider(
                self.caption_config,
                {
                    "task": "caption_chart_diagram",
                    "model": self.caption_config.get("model"),
                    "image_ref": figure.get("image_ref"),
                    "figure": _json_safe(figure),
                },
            )
            self._merge_caption_payload(result, caption_payload)
        if self.embedding_config.get("enabled", False):
            embedding_payload = await self._call_provider(
                self.embedding_config,
                {
                    "task": "retrieval.document",
                    "model": self.embedding_config.get("model"),
                    "input": [{"image_ref": figure.get("image_ref")}],
                    "image_ref": figure.get("image_ref"),
                },
            )
            result.visual_embedding = _extract_embedding(embedding_payload)
        return result

    async def embed_query(self, query: str) -> Optional[List[float]]:
        if not self.embedding_config.get("enabled", False):
            return None
        payload = await self._call_provider(
            self.embedding_config,
            {
                "task": "retrieval.query",
                "model": self.embedding_config.get("model"),
                "input": [{"text": query}],
                "query": query,
            },
        )
        return _extract_embedding(payload)

    async def _call_provider(
        self, provider_config: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        url = str(provider_config.get("url") or "").strip()
        if not url:
            return {}
        timeout = float(provider_config.get("timeout", 30))
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=timeout) as response:
                    if response.status >= 400:
                        text = await response.text()
                        raise RuntimeError(f"{response.status}: {text[:200]}")
                    data = await response.json()
                    return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.info("Visual provider unavailable at %s: %s", url, exc)
            return {"warnings": [str(exc)]}

    def _merge_ocr_payload(
        self, result: VisualFigureResult, payload: Dict[str, Any]
    ) -> None:
        result.ocr_text = _first_string(payload, "ocr_text", "text", "content")
        result.markdown = _first_string(payload, "markdown", "md")
        result.formula_text = _first_string(payload, "formula_text", "formula")
        result.detected_elements = _as_list_of_dicts(payload.get("detected_elements"))
        result.chart_data = payload.get("chart_data")
        result.warnings.extend(_as_string_list(payload.get("warnings")))

    def _merge_caption_payload(
        self, result: VisualFigureResult, payload: Dict[str, Any]
    ) -> None:
        result.caption = _first_string(payload, "caption", "description", "summary")
        result.diagram_summary = _first_string(payload, "diagram_summary", "diagram")
        result.chart_summary = _first_string(payload, "chart_summary", "chart")
        result.structured_json = payload.get("structured_json") or payload.get("json")
        result.warnings.extend(_as_string_list(payload.get("warnings")))


def build_visual_processor(config: Optional[Dict[str, Any]]) -> VisualProcessor:
    return LocalHttpVisualProcessor(config)


def compose_figure_text(block: Any) -> str:
    metadata = dict(getattr(block, "metadata", {}) or {})
    figure = dict(metadata.get("figure") or {})
    visual = dict(figure.get("visual") or {})
    parts: List[str] = []
    if figure.get("section_path"):
        parts.append(f"Section: {' > '.join(figure['section_path'])}")
    label = figure.get("label") or figure.get("raw_label")
    if label:
        parts.append(f"Figure label: {label}")
    for key, title in (
        ("caption", "Caption"),
        ("ocr_text", "OCR text"),
        ("markdown", "Markdown"),
        ("diagram_summary", "Diagram summary"),
        ("chart_summary", "Chart summary"),
        ("formula_text", "Formula"),
    ):
        value = visual.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{title}: {' '.join(value.split())}")
    for key in ("chart_data", "structured_json"):
        value = visual.get(key)
        if value:
            parts.append(f"{key}: {json.dumps(_json_safe(value), sort_keys=True)}")
    if not parts and visual.get("visual_embedding") is not None:
        parts.append(f"Figure on page {getattr(block, 'page', figure.get('page', 1))}")
    return "\n".join(parts)


def _figure_metadata(block: Any) -> Dict[str, Any]:
    metadata = dict(getattr(block, "metadata", {}) or {})
    figure = dict(metadata.get("figure") or {})
    asset_ref = str(getattr(block, "asset_ref", "") or figure.get("asset_ref") or "")
    figure_id = figure.get("figure_id") or _stable_figure_id(
        getattr(block, "doc_id", ""),
        getattr(block, "page", 1),
        getattr(block, "reading_order", 0),
        getattr(block, "bbox", None),
        asset_ref,
    )
    figure.update(
        {
            "figure_id": str(figure_id),
            "page": int(getattr(block, "page", figure.get("page", 1)) or 1),
            "bbox": getattr(block, "bbox", None),
            "asset_ref": asset_ref or None,
            "raw_label": metadata.get("raw_label") or "figure",
        }
    )
    return figure


def _stable_figure_id(
    doc_id: str,
    page: int,
    reading_order: int,
    bbox: Optional[Sequence[float]],
    asset_ref: str,
) -> str:
    if asset_ref:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", asset_ref.strip("#/")) or "figure"
    payload = json.dumps(
        {
            "doc_id": doc_id,
            "page": page,
            "reading_order": reading_order,
            "bbox": list(bbox or []),
        },
        sort_keys=True,
    )
    return f"figure-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _fitz_clip_rect(fitz: Any, page_rect: Any, bbox: Optional[Sequence[float]]) -> Any:
    if not bbox or len(bbox) != 4:
        return None
    rect = fitz.Rect([float(value) for value in bbox])
    rect = rect & page_rect
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        return
    for child in path.iterdir():
        if child.is_dir():
            _clear_directory(child)
            child.rmdir()
        else:
            child.unlink()


def _safe_path_part(value: Any) -> str:
    text = str(value or "default").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-") or "default"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


def _first_string(payload: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _as_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _extract_embedding(payload: Dict[str, Any]) -> Optional[List[float]]:
    candidates: List[Any] = [
        payload.get("embedding"),
        payload.get("vector"),
    ]
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            candidates.append(first.get("embedding"))
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        candidates.append(embeddings[0])
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            try:
                return [float(value) for value in candidate]
            except (TypeError, ValueError):
                continue
    return None
