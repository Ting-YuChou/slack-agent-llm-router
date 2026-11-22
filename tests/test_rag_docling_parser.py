"""Docling converter reuse and parser config wiring."""

import importlib.util
import logging
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag.parser import DoclingParser
from src.rag.service import RagService
from src.utils.schema import RagParserConfig


class _FakeResult:
    def __init__(self, text="Tuition is due in August."):
        self.document = {"texts": [{"label": "text", "text": text}]}


class _FakeErrorItem:
    def __init__(self, error_message):
        self.error_message = error_message


class _FakeDocling:
    """Stand-in for the docling modules that records converter usage."""

    def __init__(self):
        self.converters = []
        self.max_active_per_converter = 0
        self.max_active_total = 0
        self.convert_delay = 0.0
        self.results = []
        self._lock = threading.Lock()
        self._active_total = 0
        fake = self

        class DocumentConverter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.active = 0
                self.sources = []
                with fake._lock:
                    fake.converters.append(self)

            def convert(self, source):
                with fake._lock:
                    self.active += 1
                    fake._active_total += 1
                    fake.max_active_per_converter = max(
                        fake.max_active_per_converter, self.active
                    )
                    fake.max_active_total = max(
                        fake.max_active_total, fake._active_total
                    )
                    self.sources.append(source)
                    outcome = fake.results.pop(0) if fake.results else _FakeResult()
                try:
                    if fake.convert_delay:
                        time.sleep(fake.convert_delay)
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome
                finally:
                    with fake._lock:
                        self.active -= 1
                        fake._active_total -= 1

        class PdfFormatOption:
            def __init__(self, pipeline_options=None):
                self.pipeline_options = pipeline_options

        class _OcrOptions:
            def __init__(self):
                self.lang = []

        class PdfPipelineOptions:
            def __init__(self):
                self.do_ocr = True
                self.do_table_structure = True
                self.document_timeout = None
                self.ocr_options = _OcrOptions()

        class InputFormat:
            PDF = "pdf"

        class ConversionStatus:
            SUCCESS = "success"
            PARTIAL_SUCCESS = "partial_success"

        self.DocumentConverter = DocumentConverter
        self.PdfFormatOption = PdfFormatOption
        self.PdfPipelineOptions = PdfPipelineOptions
        self.InputFormat = InputFormat
        self.ConversionStatus = ConversionStatus

    def install(self, monkeypatch, *, with_options=True):
        package = types.ModuleType("docling")
        converter_module = types.ModuleType("docling.document_converter")
        converter_module.DocumentConverter = self.DocumentConverter
        datamodel = types.ModuleType("docling.datamodel")
        base_models = types.ModuleType("docling.datamodel.base_models")
        pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
        if with_options:
            converter_module.PdfFormatOption = self.PdfFormatOption
            base_models.InputFormat = self.InputFormat
            base_models.ConversionStatus = self.ConversionStatus
            pipeline_options.PdfPipelineOptions = self.PdfPipelineOptions
        for name, module in (
            ("docling", package),
            ("docling.document_converter", converter_module),
            ("docling.datamodel", datamodel),
            ("docling.datamodel.base_models", base_models),
            ("docling.datamodel.pipeline_options", pipeline_options),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return self


@pytest.fixture
def fake_docling(monkeypatch):
    return _FakeDocling().install(monkeypatch)


def _parse(parser, name="handbook.pdf"):
    return parser.parse_bytes(content=b"%PDF-1.7 fake", filename=name)


def test_converter_is_built_once_for_consecutive_documents(fake_docling):
    parser = DoclingParser({})

    documents = [_parse(parser, f"doc-{index}.pdf") for index in range(3)]

    assert len(fake_docling.converters) == 1
    assert len(fake_docling.converters[0].sources) == 3
    assert [doc.blocks[0].text for doc in documents] == [
        "Tuition is due in August."
    ] * 3


def test_pool_gives_each_thread_an_exclusive_converter(fake_docling):
    fake_docling.convert_delay = 0.05
    parser = DoclingParser({"converter_pool_size": 2})

    with ThreadPoolExecutor(max_workers=4) as executor:
        documents = list(
            executor.map(lambda index: _parse(parser, f"doc-{index}.pdf"), range(6))
        )

    assert len(documents) == 6
    assert len(fake_docling.converters) <= 2
    assert fake_docling.max_active_per_converter == 1
    assert fake_docling.max_active_total == 2


def test_default_pool_size_serializes_conversions(fake_docling):
    fake_docling.convert_delay = 0.02
    parser = DoclingParser({})

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda index: _parse(parser, f"doc-{index}.pdf"), range(3)))

    assert len(fake_docling.converters) == 1
    assert fake_docling.max_active_total == 1


def test_converter_is_returned_after_a_failed_conversion(fake_docling):
    fake_docling.results = [RuntimeError("corrupt pdf")]
    parser = DoclingParser({})

    with pytest.raises(RuntimeError, match="corrupt pdf"):
        _parse(parser)
    document = _parse(parser)

    assert document.blocks[0].text == "Tuition is due in August."
    assert len(fake_docling.converters) == 1


def test_failed_converter_construction_does_not_consume_a_pool_slot(
    fake_docling, monkeypatch
):
    parser = DoclingParser({})
    original = parser._build_converter
    attempts = []

    def flaky(converter_cls):
        attempts.append(converter_cls)
        if len(attempts) == 1:
            raise RuntimeError("model download failed")
        return original(converter_cls)

    monkeypatch.setattr(parser, "_build_converter", flaky)

    with pytest.raises(RuntimeError, match="model download failed"):
        _parse(parser)
    document = _parse(parser)
    _parse(parser)

    assert document.blocks
    assert len(attempts) == 2
    assert len(fake_docling.converters) == 1


def test_pipeline_options_carry_the_parser_config(fake_docling):
    parser = DoclingParser(
        {
            "ocr_enabled": False,
            "table_structure_enabled": False,
            "ocr_languages": ["ch_tra", "en"],
            "timeout_seconds": 120,
        }
    )

    _parse(parser)

    kwargs = fake_docling.converters[0].kwargs
    assert set(kwargs) == {"format_options"}
    assert set(kwargs["format_options"]) == {fake_docling.InputFormat.PDF}
    options = kwargs["format_options"][fake_docling.InputFormat.PDF].pipeline_options
    assert options.do_ocr is False
    assert options.do_table_structure is False
    assert options.ocr_options.lang == ["ch_tra", "en"]
    assert options.document_timeout == 120


def test_empty_ocr_languages_keep_the_docling_default(fake_docling):
    parser = DoclingParser({"ocr_languages": []})

    _parse(parser)

    options = (
        fake_docling.converters[0]
        .kwargs["format_options"][fake_docling.InputFormat.PDF]
        .pipeline_options
    )
    assert options.ocr_options.lang == []
    assert options.do_ocr is True
    assert options.do_table_structure is True
    assert options.document_timeout == 300


def test_converter_kwargs_are_passed_through_without_option_wiring(fake_docling):
    parser = DoclingParser(
        {
            "converter_kwargs": {"allowed_formats": ["pdf"]},
            "ocr_enabled": False,
            "timeout_seconds": 120,
        }
    )

    _parse(parser)

    assert fake_docling.converters[0].kwargs == {"allowed_formats": ["pdf"]}


def test_partial_success_from_a_timeout_raises_instead_of_returning_half(
    fake_docling,
):
    result = _FakeResult("Only the first page")
    result.status = fake_docling.ConversionStatus.PARTIAL_SUCCESS
    result.errors = [_FakeErrorItem("Page 7: document timeout exceeded")]
    fake_docling.results = [result]
    parser = DoclingParser({"timeout_seconds": 120})

    with pytest.raises(RuntimeError, match="(?i)timeout.*120"):
        _parse(parser)

    assert _parse(parser).blocks[0].text == "Tuition is due in August."
    assert len(fake_docling.converters) == 1


def test_partial_success_without_a_timeout_error_still_returns_the_document(
    fake_docling,
):
    result = _FakeResult("Readable pages")
    result.status = fake_docling.ConversionStatus.PARTIAL_SUCCESS
    result.errors = [_FakeErrorItem("Page 3 failed to parse.")]
    fake_docling.results = [result]

    document = _parse(DoclingParser({}))

    assert document.blocks[0].text == "Readable pages"


def test_missing_option_classes_fall_back_to_a_plain_converter(monkeypatch, caplog):
    fake = _FakeDocling().install(monkeypatch, with_options=False)
    parser = DoclingParser({"ocr_enabled": False, "converter_pool_size": 2})

    with caplog.at_level(logging.WARNING, logger="src.rag.parser"):
        _parse(parser)
        fake.convert_delay = 0.05
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: _parse(parser), range(2)))

    assert len(fake.converters) == 2
    assert all(converter.kwargs == {} for converter in fake.converters)
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "Docling" in record.getMessage()
    ]
    assert len(warnings) == 1


@pytest.mark.parametrize("filename", ["notes.txt", "README.md"])
def test_text_files_never_build_a_converter(fake_docling, filename):
    parser = DoclingParser({})

    document = parser.parse_bytes(content=b"Plain text body.", filename=filename)

    assert document.blocks
    assert fake_docling.converters == []


_REAL_DOCLING_CHECK = """
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter

from src.rag.parser import DoclingParser

parser = DoclingParser(
    {
        "ocr_enabled": False,
        "table_structure_enabled": False,
        "ocr_languages": ["ch_tra", "en"],
        "timeout_seconds": 120,
    }
)
format_options = parser._build_format_options()
assert format_options is not None, "options were not built"
options = format_options[InputFormat.PDF].pipeline_options
assert isinstance(options, PdfPipelineOptions)
assert options.do_ocr is False
assert options.do_table_structure is False
assert options.document_timeout == 120
assert list(options.ocr_options.lang) == ["ch_tra", "en"]
# Re-validating the dump runs Docling's own field validation.
PdfPipelineOptions.model_validate(options.model_dump())
# Pipelines, and with them the models, are only created on convert().
DocumentConverter(format_options=format_options)
print("ok")
"""


def test_real_docling_accepts_the_built_options():
    # conftest.py replaces torch with a stub that the real Docling cannot
    # import against, so the installed package is exercised in a clean process.
    if importlib.util.find_spec("docling.document_converter") is None:
        pytest.skip("docling is not installed")
    completed = subprocess.run(
        [sys.executable, "-c", _REAL_DOCLING_CHECK],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().endswith("ok")


def _service(config, **kwargs):
    return RagService(
        {
            "enabled": True,
            "backend": "memory",
            "embedding": {"provider": "hash", "dimensions": 16},
            **config,
        },
        **kwargs,
    )


def test_service_sizes_the_converter_pool_from_ingestion_concurrency():
    service = _service(
        {
            "parser": {"provider": "docling"},
            "ingestion_queue": {"concurrency": 3},
        }
    )

    assert service.parser.converter_pool_size == 3


def test_service_keeps_an_explicit_converter_pool_size():
    service = _service(
        {
            "parser": {"provider": "docling", "converter_pool_size": 2},
            "ingestion_queue": {"concurrency": 5},
        }
    )

    assert service.parser.converter_pool_size == 2


def test_service_leaves_an_injected_parser_untouched():
    parser = DoclingParser({})

    service = _service({"ingestion_queue": {"concurrency": 4}}, parser=parser)

    assert service.parser is parser
    assert parser.converter_pool_size == 1


def test_parser_config_validates_converter_pool_size():
    assert RagParserConfig().converter_pool_size is None
    assert RagParserConfig(converter_pool_size=4).converter_pool_size == 4
    with pytest.raises(ValidationError):
        RagParserConfig(converter_pool_size=0)
