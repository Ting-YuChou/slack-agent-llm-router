"""Object storage adapters for durable RAG source documents."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.rag.staging import RagStagingStore
from src.utils.metrics import RAG_METRICS


class RagObjectIntegrityError(ValueError):
    """Raised when a completed object differs from the upload intent."""


@dataclass(frozen=True)
class RagObjectRef:
    backend: str
    uri: str
    bucket: Optional[str] = None
    key: Optional[str] = None
    version_id: Optional[str] = None
    etag: Optional[str] = None
    checksum_sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RagObjectRef":
        return cls(**payload)


@dataclass(frozen=True)
class RagAssetRef:
    backend: str
    uri: str
    media_type: str
    checksum_sha256: str
    size_bytes: int
    width: int
    height: int
    knowledge_base_id: str
    document_id: str
    index_version: str
    figure_id: str
    bucket: Optional[str] = None
    key: Optional[str] = None
    version_id: Optional[str] = None
    etag: Optional[str] = None
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RagAssetRef":
        return cls(**payload)


class LocalRagAssetStore:
    """Generation-scoped local figure assets for development."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.root = Path(self.config.get("assets_dir") or "data/rag/assets")

    async def put(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
        index_version: str,
        figure_id: str,
        content: bytes,
        media_type: str,
        width: int,
        height: int,
    ) -> RagAssetRef:
        components = self._components(
            knowledge_base_id, document_id, index_version, figure_id
        )
        target = (
            self.root
            / components[0]
            / components[1]
            / components[2]
            / "figures"
            / f"{components[3]}.png"
        )
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, content)
        checksum = _sha256_base64(content)
        return RagAssetRef(
            backend="local",
            uri=_asset_uri(*components),
            path=str(target),
            media_type=media_type,
            checksum_sha256=checksum,
            size_bytes=len(content),
            width=int(width),
            height=int(height),
            knowledge_base_id=components[0],
            document_id=components[1],
            index_version=components[2],
            figure_id=components[3],
        )

    async def load(self, ref: RagAssetRef) -> bytes:
        path = self._validated_path(ref)
        payload = await asyncio.to_thread(path.read_bytes)
        _verify_asset_payload(ref, payload)
        return payload

    async def activate(self, ref: RagAssetRef) -> None:
        self._validated_path(ref)

    async def delete_document(self, document_id: str, knowledge_base_id: str) -> None:
        target = (
            self.root
            / _safe_asset_component(knowledge_base_id)
            / _safe_asset_component(document_id)
        )
        if target.exists():
            await asyncio.to_thread(shutil.rmtree, target)

    async def delete_other_generations(
        self, document_id: str, knowledge_base_id: str, active_version: str
    ) -> None:
        root = (
            self.root
            / _safe_asset_component(knowledge_base_id)
            / _safe_asset_component(document_id)
        )
        if not root.exists():
            return
        active = _safe_asset_component(active_version)
        for child in list(root.iterdir()):
            if child.is_dir() and child.name != active:
                await asyncio.to_thread(shutil.rmtree, child)

    def _components(
        self,
        knowledge_base_id: str,
        document_id: str,
        index_version: str,
        figure_id: str,
    ) -> tuple[str, str, str, str]:
        return tuple(
            _safe_asset_component(value)
            for value in (knowledge_base_id, document_id, index_version, figure_id)
        )  # type: ignore[return-value]

    def _validated_path(self, ref: RagAssetRef) -> Path:
        if ref.backend != "local" or not ref.path:
            raise ValueError("local asset store requires a local asset reference")
        root = self.root.resolve()
        path = Path(ref.path).resolve()
        if root != path and root not in path.parents:
            raise ValueError("local asset reference is outside assets_dir")
        return path


class S3RagAssetStore:
    """Private, versioned S3 figure assets used by workers and inference."""

    def __init__(self, config: Dict[str, Any], *, client: Optional[Any] = None):
        self.config = dict(config or {})
        self.bucket = str(self.config.get("bucket") or "").strip()
        if not self.bucket:
            raise ValueError("rag.storage.s3.bucket is required")
        self.environment = _safe_asset_component(
            str(self.config.get("environment") or "development")
        )
        self.prefix = str(self.config.get("prefix") or "rag").strip("/")
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("boto3 is required for S3 RAG assets") from exc
            kwargs = {}
            if self.config.get("region"):
                kwargs["region_name"] = self.config["region"]
            if self.config.get("endpoint_url"):
                kwargs["endpoint_url"] = self.config["endpoint_url"]
            self._client = boto3.client("s3", **kwargs)
        return self._client

    async def put(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
        index_version: str,
        figure_id: str,
        content: bytes,
        media_type: str,
        width: int,
        height: int,
    ) -> RagAssetRef:
        components = tuple(
            _safe_asset_component(value)
            for value in (knowledge_base_id, document_id, index_version, figure_id)
        )
        key = self._asset_key(*components)
        checksum = _sha256_base64(content)
        response = await asyncio.to_thread(
            self._get_client().put_object,
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentLength=len(content),
            ContentType=media_type,
            ChecksumSHA256=checksum,
            Tagging="state=asset_staged",
        )
        version_id = str(response.get("VersionId") or "").strip()
        if not version_id:
            raise RuntimeError(
                "S3 asset upload did not return VersionId; bucket versioning is required"
            )
        return RagAssetRef(
            backend="s3",
            uri=_asset_uri(*components),
            bucket=self.bucket,
            key=key,
            version_id=version_id,
            etag=str(response.get("ETag") or "").strip('"') or None,
            media_type=media_type,
            checksum_sha256=checksum,
            size_bytes=len(content),
            width=int(width),
            height=int(height),
            knowledge_base_id=components[0],
            document_id=components[1],
            index_version=components[2],
            figure_id=components[3],
        )

    async def load(self, ref: RagAssetRef) -> bytes:
        bucket, key = self._validated_location(ref)
        kwargs: Dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "ChecksumMode": "ENABLED",
        }
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        response = await asyncio.to_thread(self._get_client().get_object, **kwargs)
        payload = await asyncio.to_thread(response["Body"].read)
        _verify_asset_payload(ref, payload)
        return payload

    async def activate(self, ref: RagAssetRef) -> None:
        bucket, key = self._validated_location(ref)
        kwargs: Dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Tagging": {"TagSet": [{"Key": "state", "Value": "active"}]},
        }
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        await asyncio.to_thread(self._get_client().put_object_tagging, **kwargs)

    async def delete_document(self, document_id: str, knowledge_base_id: str) -> None:
        prefix = self._document_prefix(knowledge_base_id, document_id)
        await self._delete_prefix(prefix)

    async def delete_other_generations(
        self, document_id: str, knowledge_base_id: str, active_version: str
    ) -> None:
        prefix = self._document_prefix(knowledge_base_id, document_id)
        active_fragment = f"/{_safe_asset_component(active_version)}/"
        await self._delete_versioned_prefix(
            prefix,
            keep=lambda key: active_fragment in key,
        )

    async def _delete_prefix(self, prefix: str) -> None:
        await self._delete_versioned_prefix(prefix)

    async def _delete_versioned_prefix(
        self, prefix: str, *, keep: Optional[Any] = None
    ) -> None:
        client = self._get_client()
        key_marker = None
        version_marker = None
        while True:
            kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if key_marker:
                kwargs["KeyMarker"] = key_marker
            if version_marker:
                kwargs["VersionIdMarker"] = version_marker
            response = await asyncio.to_thread(client.list_object_versions, **kwargs)
            objects = []
            for item in [
                *(response.get("Versions") or []),
                *(response.get("DeleteMarkers") or []),
            ]:
                key = str(item.get("Key") or "")
                version_id = item.get("VersionId")
                if key and version_id and not (keep and keep(key)):
                    objects.append({"Key": key, "VersionId": version_id})
            await self._delete_objects(objects)
            if not response.get("IsTruncated"):
                return
            key_marker = response.get("NextKeyMarker")
            version_marker = response.get("NextVersionIdMarker")

    async def _delete_keys(self, keys: list[str]) -> None:
        await self._delete_objects([{"Key": key} for key in keys])

    async def _delete_objects(self, objects: list[Dict[str, Any]]) -> None:
        if not objects:
            return
        response = await asyncio.to_thread(
            self._get_client().delete_objects,
            Bucket=self.bucket,
            Delete={"Objects": objects, "Quiet": True},
        )
        errors = response.get("Errors") or []
        if errors:
            failed = [
                f"{item.get('Key')}@{item.get('VersionId') or 'current'}"
                for item in errors
            ]
            raise RuntimeError(
                "S3 asset deletion reported per-object failures: " + ", ".join(failed)
            )

    def _asset_key(
        self,
        knowledge_base_id: str,
        document_id: str,
        index_version: str,
        figure_id: str,
    ) -> str:
        return (
            f"{self.prefix}/{self.environment}/assets/{knowledge_base_id}/"
            f"{document_id}/{index_version}/figures/{figure_id}.png"
        )

    def _document_prefix(self, knowledge_base_id: str, document_id: str) -> str:
        return (
            f"{self.prefix}/{self.environment}/assets/"
            f"{_safe_asset_component(knowledge_base_id)}/"
            f"{_safe_asset_component(document_id)}/"
        )

    def _validated_location(self, ref: RagAssetRef) -> tuple[str, str]:
        required_prefix = f"{self.prefix}/{self.environment}/assets/"
        if (
            ref.backend != "s3"
            or ref.bucket != self.bucket
            or not ref.key
            or not ref.key.startswith(required_prefix)
        ):
            raise ValueError("S3 asset reference is outside the configured prefix")
        return self.bucket, ref.key


class LocalObjectStore:
    """Adapter exposing the existing governed staging store as an object store."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.root = Path(self.config.get("staging_dir") or "data/rag/uploads")
        self._store = RagStagingStore(self.root, self.config)

    async def put_bytes(self, *, job_id: str, content: bytes) -> RagObjectRef:
        path = await asyncio.to_thread(
            self._store.stage_bytes, str(job_id), "source", content
        )
        checksum = base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")
        RAG_METRICS.object_operations.labels("local", "put", "success").inc()
        return RagObjectRef(
            backend="local",
            uri=str(path),
            checksum_sha256=checksum,
            size_bytes=len(content),
        )

    async def load(self, ref: RagObjectRef) -> bytes:
        path = self._validated_path(ref)
        payload = await asyncio.to_thread(path.read_bytes)
        RAG_METRICS.object_operations.labels("local", "load", "success").inc()
        return payload

    async def tag(self, ref: RagObjectRef, status: str) -> None:
        await asyncio.to_thread(self._store.update_status, ref.uri, status)
        RAG_METRICS.object_operations.labels("local", "tag", "success").inc()

    async def delete(self, ref: RagObjectRef) -> None:
        await asyncio.to_thread(self._store.delete, ref.uri)
        RAG_METRICS.object_operations.labels("local", "delete", "success").inc()

    def _validated_path(self, ref: RagObjectRef) -> Path:
        if ref.backend != "local":
            raise ValueError("local object store requires a local reference")
        root = self.root.resolve()
        path = Path(ref.uri).resolve()
        if root != path and root not in path.parents:
            raise ValueError("local object reference is outside staging_dir")
        return path


class S3ObjectStore:
    """S3-backed source object store using the standard AWS credential chain."""

    def __init__(self, config: Dict[str, Any], *, client: Optional[Any] = None):
        self.config = dict(config or {})
        self.bucket = str(self.config.get("bucket") or "").strip()
        if not self.bucket:
            raise ValueError("rag.storage.s3.bucket is required")
        self.environment = str(self.config.get("environment") or "development")
        self.prefix = str(self.config.get("prefix") or "rag").strip("/")
        self.presign_ttl_seconds = int(self.config.get("presign_ttl_seconds", 900))
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("boto3 is required for the S3 RAG backend") from exc
            kwargs = {}
            region = self.config.get("region")
            endpoint_url = self.config.get("endpoint_url")
            if region:
                kwargs["region_name"] = region
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _key(self, job_id: str) -> str:
        safe_job_id = str(job_id).strip()
        if not safe_job_id or "/" in safe_job_id or safe_job_id in {".", ".."}:
            raise ValueError("job_id is not safe for an S3 object key")
        return f"{self.prefix}/{self.environment}/{safe_job_id}/source"

    async def create_upload(
        self, *, job_id: str, size_bytes: int, checksum_sha256: str
    ) -> tuple[RagObjectRef, Dict[str, Any]]:
        if size_bytes < 1:
            raise ValueError("size_bytes must be positive")
        if not checksum_sha256:
            raise ValueError("checksum_sha256 is required")
        key = self._key(job_id)
        params = {
            "Bucket": self.bucket,
            "Key": key,
            "ContentLength": int(size_bytes),
            "ChecksumSHA256": checksum_sha256,
            "Tagging": "state=pending",
        }
        client = self._get_client()
        url = await asyncio.to_thread(
            client.generate_presigned_url,
            "put_object",
            Params=params,
            ExpiresIn=self.presign_ttl_seconds,
        )
        RAG_METRICS.object_operations.labels("s3", "presign", "success").inc()
        ref = RagObjectRef(
            backend="s3",
            uri=f"s3://{self.bucket}/{key}",
            bucket=self.bucket,
            key=key,
            checksum_sha256=checksum_sha256,
            size_bytes=int(size_bytes),
        )
        return ref, {
            "method": "PUT",
            "url": url,
            "headers": {
                "content-length": str(size_bytes),
                "x-amz-checksum-sha256": checksum_sha256,
                "x-amz-tagging": "state=pending",
            },
            "expires_in": self.presign_ttl_seconds,
        }

    async def complete_upload(self, ref: RagObjectRef) -> RagObjectRef:
        bucket, key = self._validated_location(ref)
        response = await asyncio.to_thread(
            self._get_client().head_object,
            Bucket=bucket,
            Key=key,
            ChecksumMode="ENABLED",
        )
        RAG_METRICS.object_operations.labels("s3", "head", "success").inc()
        actual_size = int(response.get("ContentLength", -1))
        actual_checksum = response.get("ChecksumSHA256")
        if ref.size_bytes is not None and actual_size != ref.size_bytes:
            raise RagObjectIntegrityError("S3 object size does not match upload intent")
        if ref.checksum_sha256 and actual_checksum != ref.checksum_sha256:
            raise RagObjectIntegrityError(
                "S3 object checksum does not match upload intent"
            )
        return RagObjectRef(
            backend="s3",
            uri=ref.uri,
            bucket=bucket,
            key=key,
            version_id=response.get("VersionId"),
            etag=str(response.get("ETag") or "").strip('"') or None,
            checksum_sha256=actual_checksum or ref.checksum_sha256,
            size_bytes=actual_size,
        )

    async def put_bytes(self, *, job_id: str, content: bytes) -> RagObjectRef:
        key = self._key(job_id)
        checksum = base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")
        response = await asyncio.to_thread(
            self._get_client().put_object,
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentLength=len(content),
            ChecksumSHA256=checksum,
            Tagging="state=queued",
        )
        RAG_METRICS.object_operations.labels("s3", "put", "success").inc()
        return RagObjectRef(
            backend="s3",
            uri=f"s3://{self.bucket}/{key}",
            bucket=self.bucket,
            key=key,
            version_id=response.get("VersionId"),
            etag=str(response.get("ETag") or "").strip('"') or None,
            checksum_sha256=response.get("ChecksumSHA256") or checksum,
            size_bytes=len(content),
        )

    async def load(self, ref: RagObjectRef) -> bytes:
        bucket, key = self._validated_location(ref)
        kwargs = {"Bucket": bucket, "Key": key, "ChecksumMode": "ENABLED"}
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        response = await asyncio.to_thread(self._get_client().get_object, **kwargs)
        payload = await asyncio.to_thread(response["Body"].read)
        RAG_METRICS.object_operations.labels("s3", "load", "success").inc()
        return payload

    async def tag(self, ref: RagObjectRef, status: str) -> None:
        bucket, key = self._validated_location(ref)
        kwargs = {
            "Bucket": bucket,
            "Key": key,
            "Tagging": {"TagSet": [{"Key": "state", "Value": str(status)}]},
        }
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        await asyncio.to_thread(self._get_client().put_object_tagging, **kwargs)
        RAG_METRICS.object_operations.labels("s3", "tag", "success").inc()

    async def delete(self, ref: RagObjectRef) -> None:
        bucket, key = self._validated_location(ref)
        kwargs = {"Bucket": bucket, "Key": key}
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        await asyncio.to_thread(self._get_client().delete_object, **kwargs)
        RAG_METRICS.object_operations.labels("s3", "delete", "success").inc()

    def _validated_location(self, ref: RagObjectRef) -> tuple[str, str]:
        parsed = urlparse(ref.uri)
        bucket = ref.bucket or parsed.netloc
        key = ref.key or parsed.path.lstrip("/")
        required_prefix = f"{self.prefix}/{self.environment}/"
        if (
            ref.backend != "s3"
            or bucket != self.bucket
            or not key.startswith(required_prefix)
        ):
            raise ValueError(
                "S3 storage reference is outside the configured bucket/prefix"
            )
        return bucket, key


def _safe_asset_component(value: Any) -> str:
    normalized = str(value or "").strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ValueError("RAG asset identifier is not safe")
    return normalized


def _asset_uri(
    knowledge_base_id: str, document_id: str, index_version: str, figure_id: str
) -> str:
    return (
        f"rag-asset://{knowledge_base_id}/{document_id}/{index_version}/"
        f"figures/{figure_id}.png"
    )


def _sha256_base64(content: bytes) -> str:
    return base64.b64encode(hashlib.sha256(content).digest()).decode("ascii")


def _verify_asset_payload(ref: RagAssetRef, payload: bytes) -> None:
    if len(payload) != ref.size_bytes:
        raise RagObjectIntegrityError("RAG asset size does not match reference")
    if _sha256_base64(payload) != ref.checksum_sha256:
        raise RagObjectIntegrityError("RAG asset checksum does not match reference")
