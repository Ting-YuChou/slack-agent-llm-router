import base64
import hashlib
from io import BytesIO
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag.storage import LocalRagAssetStore, S3RagAssetStore
from src.rag.parser import DocumentBlock, ParsedDocument
from src.rag.visual import FigureCropper, LocalHttpVisualProcessor


@pytest.mark.asyncio
async def test_local_asset_store_versions_assets_and_returns_opaque_reference(tmp_path):
    store = LocalRagAssetStore({"assets_dir": str(tmp_path)})
    content = b"\x89PNG\r\n\x1a\nfigure"

    ref = await store.put(
        knowledge_base_id="bank",
        document_id="annual-report",
        index_version="generation-1",
        figure_id="figure-1",
        content=content,
        media_type="image/png",
        width=640,
        height=480,
    )

    assert ref.uri == (
        "rag-asset://bank/annual-report/generation-1/figures/figure-1.png"
    )
    assert ref.backend == "local"
    assert ref.checksum_sha256 == base64.b64encode(
        hashlib.sha256(content).digest()
    ).decode("ascii")
    assert await store.load(ref) == content
    await store.activate(ref)
    assert ref.path is not None
    assert ref.path.endswith("generation-1/figures/figure-1.png")


@pytest.mark.asyncio
async def test_local_asset_store_delete_document_removes_all_generations(tmp_path):
    store = LocalRagAssetStore({"assets_dir": str(tmp_path)})
    refs = []
    for generation in ("generation-1", "generation-2"):
        refs.append(
            await store.put(
                knowledge_base_id="bank",
                document_id="annual-report",
                index_version=generation,
                figure_id="figure-1",
                content=b"\x89PNG\r\n\x1a\nfigure",
                media_type="image/png",
                width=10,
                height=10,
            )
        )

    await store.delete_document("annual-report", "bank")

    for ref in refs:
        with pytest.raises(FileNotFoundError):
            await store.load(ref)


class FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.tags = {}
        self.deleted = []

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        self.objects[key] = bytes(kwargs["Body"])
        self.tags[key] = kwargs["Tagging"]
        return {"VersionId": "version-1", "ETag": '"etag-1"'}

    def get_object(self, **kwargs):
        key = kwargs["Key"]

        class Body:
            def __init__(self, value):
                self.value = value

            def read(self):
                return self.value

        return {"Body": Body(self.objects[key])}

    def put_object_tagging(self, **kwargs):
        self.tags[kwargs["Key"]] = kwargs["Tagging"]["TagSet"][0]["Value"]

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        return {
            "Contents": [
                {"Key": key} for key in self.objects if key.startswith(prefix)
            ],
            "IsTruncated": False,
        }

    def list_object_versions(self, **kwargs):
        prefix = kwargs["Prefix"]
        return {
            "Versions": [
                {"Key": key, "VersionId": "version-1"}
                for key in self.objects
                if key.startswith(prefix)
            ],
            "DeleteMarkers": [],
            "IsTruncated": False,
        }

    def delete_objects(self, **kwargs):
        for item in kwargs["Delete"]["Objects"]:
            key = item["Key"]
            self.deleted.append(key)
            self.objects.pop(key, None)
        return {}


@pytest.mark.asyncio
async def test_s3_asset_store_uses_generation_key_and_staged_then_active_tags():
    client = FakeS3Client()
    store = S3RagAssetStore(
        {
            "bucket": "private-rag",
            "environment": "staging",
            "prefix": "rag",
        },
        client=client,
    )

    ref = await store.put(
        knowledge_base_id="bank",
        document_id="annual-report",
        index_version="generation-1",
        figure_id="figure-1",
        content=b"\x89PNG\r\n\x1a\nfigure",
        media_type="image/png",
        width=640,
        height=480,
    )

    assert ref.key == (
        "rag/staging/assets/bank/annual-report/generation-1/figures/figure-1.png"
    )
    assert client.tags[ref.key] == "state=asset_staged"
    await store.activate(ref)
    assert client.tags[ref.key] == "active"
    assert await store.load(ref) == b"\x89PNG\r\n\x1a\nfigure"


@pytest.mark.asyncio
async def test_s3_asset_cleanup_deletes_version_ids_but_keeps_active_generation():
    client = FakeS3Client()
    store = S3RagAssetStore(
        {"bucket": "private-rag", "environment": "staging", "prefix": "rag"},
        client=client,
    )
    for generation in ("old", "active"):
        await store.put(
            knowledge_base_id="bank",
            document_id="report",
            index_version=generation,
            figure_id="figure-1",
            content=b"png",
            media_type="image/png",
            width=10,
            height=10,
        )

    await store.delete_other_generations("report", "bank", "active")

    assert not any("/old/" in key for key in client.objects)
    assert any("/active/" in key for key in client.objects)


@pytest.mark.asyncio
async def test_s3_asset_cleanup_raises_on_partial_delete_errors():
    client = FakeS3Client()

    def fail_delete(**_kwargs):
        return {
            "Errors": [
                {
                    "Key": "rag/staging/assets/bank/report/v1/figure.png",
                    "VersionId": "version-1",
                    "Code": "AccessDenied",
                }
            ]
        }

    client.delete_objects = fail_delete
    store = S3RagAssetStore(
        {"bucket": "private-rag", "environment": "staging", "prefix": "rag"},
        client=client,
    )
    await store.put(
        knowledge_base_id="bank",
        document_id="report",
        index_version="v1",
        figure_id="figure-1",
        content=b"png",
        media_type="image/png",
        width=10,
        height=10,
    )

    with pytest.raises(RuntimeError, match="per-object failures"):
        await store.delete_document("report", "bank")


@pytest.mark.asyncio
async def test_s3_asset_store_requires_bucket_versioning():
    client = FakeS3Client()
    client.put_object = lambda **_kwargs: {"ETag": '"etag-1"'}
    store = S3RagAssetStore(
        {"bucket": "private-rag", "environment": "staging", "prefix": "rag"},
        client=client,
    )

    with pytest.raises(RuntimeError, match="VersionId"):
        await store.put(
            knowledge_base_id="bank",
            document_id="report",
            index_version="generation",
            figure_id="figure-1",
            content=b"png",
            media_type="image/png",
            width=10,
            height=10,
        )


@pytest.mark.asyncio
async def test_visual_processor_only_calls_ocr_and_caption_not_embedding():
    class RecordingProcessor(LocalHttpVisualProcessor):
        def __init__(self):
            super().__init__(
                {
                    "ocr": {"enabled": True, "model": "ocr"},
                    "caption": {"enabled": True, "model": "caption"},
                    "embedding": {"enabled": True, "model": "must-not-run"},
                }
            )
            self.tasks = []

        async def _call_provider(self, _config, payload):
            self.tasks.append(payload["task"])
            if payload["task"] == "document_parse":
                return {"ocr_text": "FY25 15.0"}
            return {"caption": "CET1 ratio chart"}

    processor = RecordingProcessor()
    result = await processor.process_figure(
        {
            "image_ref": "rag-asset://bank/doc/generation/figures/figure.png",
            "_image_bytes": b"\x89PNG\r\n\x1a\nfigure",
            "media_type": "image/png",
        }
    )

    assert processor.tasks == ["document_parse", "caption_chart_diagram"]
    assert result.ocr_text == "FY25 15.0"
    assert result.caption == "CET1 ratio chart"
    assert "visual_embedding" not in result.to_metadata()


@pytest.mark.asyncio
async def test_visual_provider_echo_error_never_returns_submitted_base64(
    monkeypatch,
):
    secret_base64 = base64.b64encode(b"private-image-bytes").decode("ascii")

    class Response:
        status = 400

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def text(self):
            return f'{{"image_base64":"{secret_base64}"}}'

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientSession=lambda: Session()),
    )
    processor = LocalHttpVisualProcessor({})

    result = await processor._call_provider(
        {"url": "http://visual.invalid"},
        {"image_base64": secret_base64},
    )

    assert result == {"warnings": ["visual provider returned HTTP 400"]}
    assert secret_base64 not in str(result)


def test_figure_cropper_keeps_png_bytes_transient_instead_of_writing_asset(
    tmp_path, monkeypatch
):
    png = b"\x89PNG\r\n\x1a\nfigure"

    class Rect:
        def __init__(self, values):
            self.values = list(values)
            self.width = self.values[2] - self.values[0]
            self.height = self.values[3] - self.values[1]
            self.is_empty = False

        def __and__(self, _other):
            return self

    class Pixmap:
        width = 640
        height = 480

        def save(self, path):
            with open(path, "wb") as target:
                target.write(png)

        def tobytes(self, _format):
            return png

    class Page:
        rect = Rect([0, 0, 300, 300])

        def get_pixmap(self, **_kwargs):
            return Pixmap()

    class Pdf:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return Page()

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(
            open=lambda **_kwargs: Pdf(),
            Matrix=lambda *_args: object(),
            Rect=Rect,
        ),
    )
    content = b"%PDF"
    parsed = ParsedDocument(
        document_id="doc",
        filename="report.pdf",
        content_hash="hash",
        blocks=[
            DocumentBlock(
                doc_id="doc",
                page=1,
                bbox=[20, 20, 280, 280],
                block_type="figure",
                reading_order=1,
                metadata={"figure": {"figure_id": "figure-1"}},
            )
        ],
    )
    cropper = FigureCropper({"storage": {"assets_dir": str(tmp_path)}})

    warnings = cropper.crop_figures(
        content=content,
        filename="report.pdf",
        parsed_document=parsed,
        knowledge_base_id="bank",
    )

    figure = parsed.blocks[0].metadata["figure"]
    assert warnings == []
    assert figure["crop_status"] == "completed"
    assert figure["media_type"] == "image/png"
    assert figure["_image_bytes"].startswith(b"\x89PNG")
    assert "image_ref" not in figure
    assert not list(tmp_path.rglob("*.png"))


def test_figure_cropper_converts_docling_bottomleft_bbox_before_rendering():
    fitz = pytest.importorskip("fitz")
    image_module = pytest.importorskip("PIL.Image")
    pdf = fitz.open()
    page = pdf.new_page(width=200, height=200)
    page.draw_rect(fitz.Rect(0, 0, 100, 50), color=(1, 0, 0), fill=(1, 0, 0))
    page.draw_rect(fitz.Rect(0, 150, 100, 200), color=(0, 0, 1), fill=(0, 0, 1))
    content = pdf.tobytes()
    pdf.close()
    parsed = ParsedDocument(
        document_id="doc",
        filename="report.pdf",
        content_hash="hash",
        blocks=[
            DocumentBlock(
                doc_id="doc",
                page=1,
                bbox=[0, 0, 100, 50],
                block_type="figure",
                reading_order=1,
                metadata={
                    "provenance": [{"bbox": {"coord_origin": "BOTTOMLEFT"}}],
                    "figure": {"figure_id": "bottom-figure"},
                },
            )
        ],
    )

    warnings = FigureCropper({"crop_dpi": 72, "min_crop_pixels": 1}).crop_figures(
        content=content,
        filename="report.pdf",
        parsed_document=parsed,
        knowledge_base_id="bank",
    )

    image = image_module.open(
        BytesIO(parsed.blocks[0].metadata["figure"]["_image_bytes"])
    ).convert("RGB")
    red, green, blue = image.getpixel((image.width // 2, image.height // 2))
    assert warnings == []
    assert blue > red
    assert blue > green


def test_terraform_scopes_asset_write_and_inference_read_permissions():
    root = Path(__file__).resolve().parents[1]
    main_tf = (root / "infra" / "aws" / "rag" / "main.tf").read_text()
    outputs_tf = (root / "infra" / "aws" / "rag" / "outputs.tf").read_text()

    assert 'value = "asset_staged"' in main_tf
    assert '"${aws_s3_bucket.rag.arn}/rag/${var.environment}/assets/*"' in main_tf
    assert (
        'Resource = "${aws_s3_bucket.rag.arn}/rag/${var.environment}/*"' not in main_tf
    )
    assert (
        main_tf.count(
            'Resource = "${aws_s3_bucket.rag.arn}/rag/${var.environment}/*/source"'
        )
        >= 2
    )
    worker_policy = main_tf.split('resource "aws_iam_policy" "worker"', 1)[1].split(
        'resource "aws_iam_policy" "inference"', 1
    )[0]
    assert '"s3:PutObjectVersionTagging"' in worker_policy
    assert (
        'Resource = "${aws_s3_bucket.rag.arn}/rag/${var.environment}/*/source"'
        in worker_policy
    )
    assert 'resource "aws_iam_policy" "inference"' in main_tf
    assert 'Action   = ["kms:Decrypt"]' in main_tf
    assert '"s3:ListBucketVersions"' in main_tf
    assert main_tf.count('"s3:ListBucketVersions"') >= 2
    assert '"s3:prefix" = "rag/${var.environment}/assets/*"' in main_tf
    assert 'id     = "expire-staged-asset-versions"' in main_tf
    assert "noncurrent_version_expiration { noncurrent_days = 1 }" in main_tf
    assert "expired_object_delete_marker = true" in main_tf
    assert "inference_policy_arn" in outputs_tf
