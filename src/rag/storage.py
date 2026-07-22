"""Object storage adapters for durable RAG source documents."""

from __future__ import annotations

import asyncio
import base64
import hashlib
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
