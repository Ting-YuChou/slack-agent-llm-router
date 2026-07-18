"""Cross-process capacity and retention governance for staged RAG uploads."""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from src.utils.metrics import RAG_METRICS


class StagingCapacityError(RuntimeError):
    """Raised when shared staging capacity cannot accept more bytes."""


class RagStagingStore:
    def __init__(
        self,
        staging_dir: Path,
        config: Dict[str, Any],
        *,
        clock: Callable[[], float] = time.time,
    ):
        self.root = Path(staging_dir)
        self._clock = clock
        self._reconciled = False
        self.max_bytes = int(config.get("max_staging_bytes", 10_737_418_240))
        self.high_watermark_ratio = float(config.get("high_watermark_ratio", 0.9))
        self.completed_ttl = int(config.get("completed_file_ttl_seconds", 86_400))
        self.failed_ttl = int(config.get("failed_file_ttl_seconds", 604_800))
        self.orphan_ttl = int(config.get("orphan_file_ttl_seconds", 604_800))
        self._lock_path = self.root / ".staging.lock"
        self._ledger_path = self.root / ".usage.json"

    def ensure_reconciled(self) -> None:
        if not self._reconciled:
            self.reconcile()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _manifest_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.manifest.json"

    def _read_json(self, path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, ValueError, TypeError):
            return dict(default)

    def _atomic_json(self, path: Path, payload: Dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    def _usage_unlocked(self) -> int:
        return int(self._read_json(self._ledger_path, {"bytes": 0}).get("bytes", 0))

    def _set_usage_unlocked(self, value: int) -> None:
        self._atomic_json(
            self._ledger_path,
            {"bytes": max(0, int(value)), "updated_at": self._clock()},
        )
        RAG_METRICS.staged_bytes.set(max(0, int(value)))

    def _update_file_metric(self) -> None:
        RAG_METRICS.staged_files.set(len(list(self.root.glob("*.manifest.json"))))

    def usage_bytes(self) -> int:
        self.ensure_reconciled()
        with self._locked():
            return self._usage_unlocked()

    def begin(self, job_id: str, filename: str) -> tuple[Path, Path]:
        self.ensure_reconciled()
        data_path = self.root / f"{job_id}-{filename}"
        part_path = self.root / f".{job_id}-{filename}.part"
        with self._locked():
            usage = self._usage_unlocked()
            if usage >= int(self.max_bytes * self.high_watermark_ratio):
                RAG_METRICS.storage_rejections.labels(reason="high_watermark").inc()
                raise StagingCapacityError("RAG staging high watermark reached")
            manifest = {
                "job_id": job_id,
                "bytes": 0,
                "status": "queued",
                "updated_at": self._clock(),
                "expires_at": self._clock() + self.orphan_ttl,
                "data_path": str(data_path),
                "part_path": str(part_path),
            }
            self._atomic_json(self._manifest_path(job_id), manifest)
            self._update_file_metric()
        return part_path, data_path

    def reserve(self, job_id: str, amount: int) -> None:
        with self._locked():
            manifest_path = self._manifest_path(job_id)
            manifest = self._read_json(manifest_path, {})
            if not manifest:
                raise RuntimeError(f"missing staging manifest for {job_id}")
            usage = self._usage_unlocked()
            if usage + amount > self.max_bytes:
                RAG_METRICS.storage_rejections.labels(reason="hard_limit").inc()
                raise StagingCapacityError("RAG staging hard limit would be exceeded")
            manifest["bytes"] = int(manifest.get("bytes", 0)) + int(amount)
            manifest["updated_at"] = self._clock()
            manifest["expires_at"] = self._clock() + self.orphan_ttl
            self._atomic_json(manifest_path, manifest)
            self._set_usage_unlocked(usage + amount)

    def publish(self, job_id: str, part_path: Path, data_path: Path) -> str:
        os.replace(part_path, data_path)
        with self._locked():
            manifest_path = self._manifest_path(job_id)
            manifest = self._read_json(manifest_path, {})
            manifest["part_path"] = None
            manifest["data_path"] = str(data_path)
            manifest["updated_at"] = self._clock()
            self._atomic_json(manifest_path, manifest)
        return str(data_path)

    def rollback(self, job_id: str) -> None:
        with self._locked():
            manifest_path = self._manifest_path(job_id)
            manifest = self._read_json(manifest_path, {})
            for key in ("part_path", "data_path"):
                if manifest.get(key):
                    Path(manifest[key]).unlink(missing_ok=True)
            self._set_usage_unlocked(
                self._usage_unlocked() - int(manifest.get("bytes", 0))
            )
            manifest_path.unlink(missing_ok=True)
            self._update_file_metric()

    def stage_bytes(self, job_id: str, filename: str, content: bytes) -> str:
        part_path, data_path = self.begin(job_id, filename)
        try:
            self.reserve(job_id, len(content))
            with part_path.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            return self.publish(job_id, part_path, data_path)
        except BaseException:
            self.rollback(job_id)
            raise

    def _manifest_for_storage_ref(
        self, storage_ref: str
    ) -> tuple[Path, Dict[str, Any]]:
        resolved = str(Path(storage_ref).resolve())
        for manifest_path in self.root.glob("*.manifest.json"):
            manifest = self._read_json(manifest_path, {})
            if str(Path(manifest.get("data_path", "")).resolve()) == resolved:
                return manifest_path, manifest
        raise FileNotFoundError(f"No staging manifest for {storage_ref}")

    def update_status(self, storage_ref: str, status: str) -> None:
        self.ensure_reconciled()
        with self._locked():
            manifest_path, manifest = self._manifest_for_storage_ref(storage_ref)
            now = self._clock()
            if status in {"completed", "completed_with_warnings"}:
                ttl = self.completed_ttl
            elif status in {"failed", "dead_lettered"}:
                ttl = self.failed_ttl
            else:
                ttl = self.orphan_ttl
            manifest.update(
                {"status": status, "updated_at": now, "expires_at": now + ttl}
            )
            self._atomic_json(manifest_path, manifest)

    def refresh_lease(self, storage_ref: str, status: str) -> None:
        self.update_status(storage_ref, status)

    def delete(self, storage_ref: str) -> None:
        with self._locked():
            manifest_path, manifest = self._manifest_for_storage_ref(storage_ref)
            job_id = str(manifest.get("job_id") or "")
        if job_id:
            self.rollback(job_id)

    def run_janitor(self) -> Dict[str, int]:
        self.ensure_reconciled()
        deleted = 0
        with self._locked():
            now = self._clock()
            usage = self._usage_unlocked()
            for manifest_path in list(self.root.glob("*.manifest.json")):
                manifest = self._read_json(manifest_path, {})
                if float(manifest.get("expires_at", now + 1)) > now:
                    continue
                for key in ("part_path", "data_path"):
                    if manifest.get(key):
                        Path(manifest[key]).unlink(missing_ok=True)
                usage -= int(manifest.get("bytes", 0))
                manifest_path.unlink(missing_ok=True)
                deleted += 1
            self._set_usage_unlocked(usage)
            self._update_file_metric()
            RAG_METRICS.janitor_outcomes.labels(outcome="deleted").inc(deleted)
        return {"deleted": deleted, "errors": 0}

    def reconcile(self) -> Dict[str, int]:
        removed_parts = 0
        with self._locked():
            for part_path in self.root.rglob("*.part"):
                part_path.unlink(missing_ok=True)
                removed_parts += 1
            usage = 0
            for manifest_path in list(self.root.glob("*.manifest.json")):
                manifest = self._read_json(manifest_path, {})
                data_path = Path(manifest.get("data_path", ""))
                if not data_path.is_file():
                    manifest_path.unlink(missing_ok=True)
                    continue
                size = data_path.stat().st_size
                manifest["bytes"] = size
                manifest["part_path"] = None
                self._atomic_json(manifest_path, manifest)
                usage += size
            self._set_usage_unlocked(usage)
            self._update_file_metric()
            self._reconciled = True
        return {"removed_parts": removed_parts, "bytes": usage}
