#!/usr/bin/env python3
"""Resumable migration of legacy RAG hashes to document-tagged v2 keys.

The command is a dry-run unless ``--apply`` is supplied.  Cutover is deliberately
delegated to explicit operator hooks: the repository cannot truthfully pause or
deploy independently-running API/worker processes from a Redis script.
"""

import argparse
import asyncio
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PHASES = (
    "copy",
    "validate_initial",
    "pause",
    "delta_copy",
    "validate_final",
    "switch",
    "resume",
    "complete",
)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class MigrationSpec:
    key_prefix: str = "rag"
    batch_size: int = 64
    pause_command: Optional[str] = None
    resume_command: Optional[str] = None
    switch_command: Optional[str] = None
    rollback_command: Optional[str] = None

    def tag(self, knowledge_base_id: str, document_id: str) -> str:
        return f"{{{knowledge_base_id}:{document_id}}}"

    def target_chunk_key(self, kb: str, document: str, chunk: str) -> str:
        return f"{self.key_prefix}:v2:chunk:{self.tag(kb, document)}:{chunk}"

    def target_visual_key(self, kb: str, document: str, chunk: str) -> str:
        return f"{self.key_prefix}:v2:visual_chunk:{self.tag(kb, document)}:{chunk}"

    def target_document_key(self, kb: str, document: str, kind: str) -> str:
        if kind == "chunks":
            return f"{self.key_prefix}:document:v2:{self.tag(kb, document)}:chunks"
        if kind == "visual_chunks":
            return (
                f"{self.key_prefix}:document:v2:{self.tag(kb, document)}:visual_chunks"
            )
        if kind == "tables":
            return f"{self.key_prefix}:tables:v2:{self.tag(kb, document)}"
        if kind == "figures":
            return f"{self.key_prefix}:figures:v2:{self.tag(kb, document)}"
        raise ValueError(f"unknown document collection: {kind}")

    def target_sidecar_key(
        self, kb: str, document: str, kind: str, identity: str
    ) -> str:
        return f"{self.key_prefix}:{kind}:v2:{self.tag(kb, document)}:{identity}"

    def target_pattern(self, kind: str) -> str:
        if kind in {"chunk", "visual_chunk"}:
            return f"{self.key_prefix}:v2:{kind}:*"
        return f"{self.key_prefix}:{kind}:v2:*"


class MigrationRunner:
    """Small persisted state machine; every completed phase is restart-safe."""

    def __init__(self, client: Any, spec: MigrationSpec):
        self.client = client
        self.spec = spec
        self.state_key = f"{spec.key_prefix}:migration:rag-v2-tagged:phase"
        self.metrics: Dict[str, Any] = {}

    def plan(self) -> List[Dict[str, str]]:
        return [
            {"phase": "copy", "action": "copy live legacy hashes to tagged v2 keys"},
            {
                "phase": "validate_initial",
                "action": "compare count and content checksum",
            },
            {"phase": "pause", "action": "run the required ingestion pause/flush hook"},
            {
                "phase": "delta_copy",
                "action": "repeat idempotent copy while writers are paused",
            },
            {"phase": "validate_final", "action": "require an exact final checksum"},
            {
                "phase": "switch",
                "action": "run the explicit deployment/index switch hook",
            },
            {
                "phase": "resume",
                "action": "resume ingestion only after successful switch",
            },
        ]

    async def run(self, *, apply: bool) -> List[Dict[str, str]]:
        plan = self.plan()
        if not apply:
            return plan
        if not self.spec.pause_command:
            raise ValueError("--pause-command is required with --apply")
        if not self.spec.resume_command:
            raise ValueError("--resume-command is required with --apply")
        if not self.spec.switch_command:
            raise ValueError("--switch-command is required with --apply")
        current = _text(await self.client.get(self.state_key) or "copy")
        if current not in PHASES:
            raise RuntimeError(f"unknown persisted migration phase: {current}")
        for phase in PHASES[PHASES.index(current) :]:
            if phase == "complete":
                await self.client.set(self.state_key, phase)
                break
            await self._run_phase(phase)
            await self.client.set(self.state_key, PHASES[PHASES.index(phase) + 1])
        return plan

    async def rollback(self) -> None:
        if not (self.spec.pause_command and self.spec.resume_command):
            raise ValueError("rollback requires pause and resume commands")
        if not self.spec.rollback_command:
            raise ValueError("--rollback-command is required")
        current = _text(await self.client.get(self.state_key) or "")
        if current == "rolled_back":
            return
        if current == "rollback_switched":
            await self._run_hook(self.spec.resume_command)
            await self.client.set(self.state_key, "rolled_back")
            return
        if current not in {"complete", "rollback_paused", "rollback_copied"}:
            raise RuntimeError("rollback is only safe after a completed v2 cutover")
        await self._run_hook(self.spec.pause_command)
        await self.client.set(self.state_key, "rollback_paused")
        try:
            await self._copy_hashes_to_legacy()
            await self._validate()
            await self.client.set(self.state_key, "rollback_copied")
            await self._run_hook(self.spec.rollback_command)
            await self.client.set(self.state_key, "rollback_switched")
            await self._run_hook(self.spec.resume_command)
        except Exception:
            # Deliberately remain paused. Resuming after a failed rollback can
            # allow writes to diverge across layouts.
            raise
        await self.client.set(self.state_key, "rolled_back")

    async def _copy_hashes_to_legacy(self) -> Dict[str, int]:
        families = (
            ("chunk", "chunk_id", "chunks"),
            ("visual_chunk", "chunk_id", "visual_chunks"),
            ("table", "table_id", "tables"),
            ("figure", "figure_id", "figures"),
            ("table_row", "row_id", None),
        )
        counts: Dict[str, int] = {}
        expected_legacy_keys: set[str] = set()
        for kind, identity_field, collection in families:
            copied = 0
            async for key in self.client.scan_iter(
                match=self.spec.target_pattern(kind), count=self.spec.batch_size
            ):
                raw = await self.client.hgetall(key)
                fields = {_text(name): value for name, value in raw.items()}
                kb = _text(fields.get("knowledge_base_id", ""))
                document = _text(fields.get("document_id", ""))
                identity = _text(fields.get(identity_field, ""))
                table_id = _text(fields.get("table_id", ""))
                if not (kb and document and identity):
                    raise RuntimeError(f"v2 hash lacks identity fields: {_text(key)}")
                legacy_key = self._legacy_hash_key(
                    kind, kb, document, identity, table_id=table_id
                )
                expected_legacy_keys.add(legacy_key)
                pipe = self.client.pipeline(transaction=False)
                pipe.hset(legacy_key, mapping=raw)
                if collection:
                    pipe.sadd(
                        self._legacy_collection_key(kb, document, collection),
                        identity,
                    )
                if kind == "table_row":
                    pipe.sadd(
                        self._legacy_collection_key(
                            kb, document, "table_rows", table_id=table_id
                        ),
                        identity,
                    )
                    pipe.sadd(
                        self._legacy_collection_key(kb, document, "tables"),
                        table_id,
                    )
                pipe.sadd(
                    f"{self.spec.key_prefix}:document:{document}:knowledge_bases", kb
                )
                await pipe.execute()
                copied += 1

            stale = []
            async for key in self.client.scan_iter(
                match=f"{self.spec.key_prefix}:{kind}:*",
                count=self.spec.batch_size,
            ):
                decoded = _text(key)
                if ":v2:" not in decoded and decoded not in expected_legacy_keys:
                    stale.append(key)
            if stale:
                await self.client.delete(*stale)
            counts[f"copied_{kind}"] = copied
        expected_sets = await self._expected_collection_sets(v2=False)
        await self._replace_collection_sets(expected_sets, v2=False)
        counts["collection_sets"] = len(expected_sets)
        return counts

    def _legacy_hash_key(
        self,
        kind: str,
        kb: str,
        document: str,
        identity: str,
        *,
        table_id: str = "",
    ) -> str:
        if kind in {"chunk", "visual_chunk"}:
            return f"{self.spec.key_prefix}:{kind}:{kb}:{identity}"
        if kind == "table_row":
            return (
                f"{self.spec.key_prefix}:table_row:{kb}:{document}:"
                f"{table_id}:{identity}"
            )
        return f"{self.spec.key_prefix}:{kind}:{kb}:{document}:{identity}"

    def _legacy_collection_key(
        self,
        kb: str,
        document: str,
        kind: str,
        *,
        table_id: str = "",
    ) -> str:
        if kind == "chunks":
            return f"{self.spec.key_prefix}:document:{kb}:{document}:chunks"
        if kind == "visual_chunks":
            return f"{self.spec.key_prefix}:document:{kb}:{document}:visual_chunks"
        if kind == "table_rows":
            return f"{self.spec.key_prefix}:table_rows:{kb}:{document}:{table_id}"
        return f"{self.spec.key_prefix}:{kind}:{kb}:{document}"

    async def _expected_collection_sets(self, *, v2: bool) -> Dict[str, set[str]]:
        expected: Dict[str, set[str]] = {}
        families = (
            ("chunk", "chunk_id"),
            ("visual_chunk", "chunk_id"),
            ("table", "table_id"),
            ("figure", "figure_id"),
            ("table_row", "row_id"),
        )

        def add(key: str, member: str) -> None:
            expected.setdefault(key, set()).add(member)

        for kind, identity_field in families:
            pattern = (
                self.spec.target_pattern(kind)
                if v2
                else f"{self.spec.key_prefix}:{kind}:*"
            )
            async for key in self.client.scan_iter(
                match=pattern, count=self.spec.batch_size
            ):
                if not v2 and ":v2:" in _text(key):
                    continue
                raw = await self.client.hgetall(key)
                fields = {_text(name): value for name, value in raw.items()}
                kb = _text(fields.get("knowledge_base_id", ""))
                document = _text(fields.get("document_id", ""))
                identity = _text(fields.get(identity_field, ""))
                table_id = _text(fields.get("table_id", ""))
                if not (kb and document and identity):
                    raise RuntimeError(
                        f"RAG hash lacks collection identity fields: {_text(key)}"
                    )
                add(
                    f"{self.spec.key_prefix}:document:{document}:knowledge_bases",
                    kb,
                )
                if kind == "chunk":
                    collection = (
                        self.spec.target_document_key(kb, document, "chunks")
                        if v2
                        else self._legacy_collection_key(kb, document, "chunks")
                    )
                    add(collection, identity)
                elif kind == "visual_chunk":
                    collection = (
                        self.spec.target_document_key(kb, document, "visual_chunks")
                        if v2
                        else self._legacy_collection_key(kb, document, "visual_chunks")
                    )
                    add(collection, identity)
                elif kind == "table":
                    collection = (
                        self.spec.target_document_key(kb, document, "tables")
                        if v2
                        else self._legacy_collection_key(kb, document, "tables")
                    )
                    add(collection, identity)
                elif kind == "figure":
                    collection = (
                        self.spec.target_document_key(kb, document, "figures")
                        if v2
                        else self._legacy_collection_key(kb, document, "figures")
                    )
                    add(collection, identity)
                else:
                    tables = (
                        self.spec.target_document_key(kb, document, "tables")
                        if v2
                        else self._legacy_collection_key(kb, document, "tables")
                    )
                    rows = (
                        self.spec.target_sidecar_key(
                            kb, document, "table_rows", table_id
                        )
                        if v2
                        else self._legacy_collection_key(
                            kb, document, "table_rows", table_id=table_id
                        )
                    )
                    add(tables, table_id)
                    add(rows, identity)
        return expected

    async def _existing_collection_keys(self, *, v2: bool) -> set[str]:
        patterns = (
            f"{self.spec.key_prefix}:document:*",
            f"{self.spec.key_prefix}:tables:*",
            f"{self.spec.key_prefix}:figures:*",
            f"{self.spec.key_prefix}:table_rows:*",
        )
        keys: set[str] = set()
        for pattern in patterns:
            async for key in self.client.scan_iter(
                match=pattern, count=self.spec.batch_size
            ):
                decoded = _text(key)
                shared = decoded.endswith(":knowledge_bases")
                if shared or ((":v2:" in decoded) == v2):
                    keys.add(decoded)
        return keys

    async def _replace_collection_sets(
        self, expected: Dict[str, set[str]], *, v2: bool
    ) -> None:
        existing = await self._existing_collection_keys(v2=v2)
        operations = sorted(existing | set(expected))
        for start in range(0, len(operations), self.spec.batch_size):
            pipe = self.client.pipeline(transaction=False)
            for key in operations[start : start + self.spec.batch_size]:
                pipe.delete(key)
                members = expected.get(key)
                if members:
                    pipe.sadd(key, *sorted(members))
            await pipe.execute()

    async def _validate_collection_sets(self, *, v2: bool) -> int:
        expected = await self._expected_collection_sets(v2=v2)
        existing = await self._existing_collection_keys(v2=v2)
        if existing != set(expected):
            raise RuntimeError(
                "migration collection key validation failed: "
                f"expected={len(expected)} actual={len(existing)}"
            )
        for key, members in expected.items():
            actual = {_text(value) for value in await self.client.smembers(key)}
            if actual != members:
                raise RuntimeError(
                    f"migration collection membership validation failed: {key}"
                )
        return len(expected)

    async def _run_phase(self, phase: str) -> None:
        if phase in {"copy", "delta_copy"}:
            self.metrics[phase] = await self._copy_hashes()
        elif phase in {"validate_initial", "validate_final"}:
            self.metrics[phase] = await self._validate()
        elif phase == "pause":
            await self._run_hook(self.spec.pause_command)
        elif phase == "switch":
            await self._run_hook(self.spec.switch_command)
        elif phase == "resume":
            await self._run_hook(self.spec.resume_command)

    async def _run_hook(self, command: Optional[str]) -> None:
        if not command:
            raise ValueError("migration hook is not configured")
        process = await asyncio.create_subprocess_exec(*shlex.split(command))
        if await process.wait() != 0:
            raise RuntimeError(f"migration hook failed: {command}")

    async def _copy_hashes(self) -> Dict[str, int]:
        families = (
            ("chunk", "chunk_id", "chunk", "chunks"),
            ("visual_chunk", "chunk_id", "visual_chunk", "visual_chunks"),
            ("table", "table_id", "table", "tables"),
            ("figure", "figure_id", "figure", "figures"),
            ("table_row", "row_id", "table_row", None),
        )
        counts: Dict[str, int] = {}
        for source_kind, identity_field, target_kind, collection in families:
            copied = 0
            expected_target_keys: set[str] = set()
            async for key in self.client.scan_iter(
                match=f"{self.spec.key_prefix}:{source_kind}:*",
                count=self.spec.batch_size,
            ):
                if ":v2:" in _text(key):
                    continue
                raw = await self.client.hgetall(key)
                fields = {_text(name): value for name, value in raw.items()}
                kb = _text(fields.get("knowledge_base_id", ""))
                document = _text(fields.get("document_id", ""))
                identity = _text(fields.get(identity_field, ""))
                if source_kind == "table_row":
                    table_id = _text(fields.get("table_id", ""))
                    identity = f"{table_id}:{identity}" if table_id and identity else ""
                if not (kb and document and identity):
                    raise RuntimeError(
                        f"legacy sidecar lacks identity fields: {_text(key)}"
                    )
                if source_kind == "chunk":
                    target = self.spec.target_chunk_key(kb, document, identity)
                elif source_kind == "visual_chunk":
                    target = self.spec.target_visual_key(kb, document, identity)
                else:
                    target = self.spec.target_sidecar_key(
                        kb, document, target_kind, identity
                    )
                expected_target_keys.add(target)
                pipe = self.client.pipeline(transaction=False)
                pipe.hset(target, mapping=raw)
                if collection:
                    member = identity
                    pipe.sadd(
                        self.spec.target_document_key(kb, document, collection), member
                    )
                if source_kind == "table_row":
                    table_id, row_id = identity.split(":", 1)
                    pipe.sadd(
                        self.spec.target_sidecar_key(
                            kb, document, "table_rows", table_id
                        ),
                        row_id,
                    )
                    pipe.sadd(
                        self.spec.target_document_key(kb, document, "tables"), table_id
                    )
                await pipe.execute()
                copied += 1
            stale = [
                key
                async for key in self.client.scan_iter(
                    match=self.spec.target_pattern(source_kind),
                    count=self.spec.batch_size,
                )
                if _text(key) not in expected_target_keys
            ]
            if stale:
                await self.client.delete(*stale)
            counts[f"copied_{source_kind}"] = copied
        expected_sets = await self._expected_collection_sets(v2=True)
        await self._replace_collection_sets(expected_sets, v2=True)
        counts["collection_sets"] = len(expected_sets)
        return counts

    async def _snapshot(self, pattern: str) -> Dict[str, str]:
        values: Dict[str, str] = {}
        async for key in self.client.scan_iter(pattern, count=self.spec.batch_size):
            if ":v2:" in _text(key) and ":v2:" not in pattern:
                continue
            raw = await self.client.hgetall(key)
            fields = {_text(name): value for name, value in raw.items()}
            identity = ":".join(
                _text(fields.get(name, ""))
                for name in (
                    "knowledge_base_id",
                    "document_id",
                    "chunk_id",
                    "table_id",
                    "row_id",
                    "figure_id",
                )
            )
            canonical = {
                name: (
                    {"bytes_hex": value.hex()}
                    if isinstance(value, bytes)
                    else {"text": str(value)}
                )
                for name, value in fields.items()
            }
            digest = hashlib.sha256(
                json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            values[identity] = digest
        return values

    async def _validate(self) -> Dict[str, Any]:
        source: Dict[str, str] = {}
        target: Dict[str, str] = {}
        for kind in ("chunk", "visual_chunk", "table", "figure", "table_row"):
            source.update(
                {
                    f"{kind}:{key}": value
                    for key, value in (
                        await self._snapshot(f"{self.spec.key_prefix}:{kind}:*")
                    ).items()
                }
            )
            target.update(
                {
                    f"{kind}:{key}": value
                    for key, value in (
                        await self._snapshot(self.spec.target_pattern(kind))
                    ).items()
                }
            )
        if source != target:
            missing = sorted(set(source) - set(target))[:10]
            mismatched = sorted(
                key
                for key in source.keys() & target.keys()
                if source[key] != target[key]
            )[:10]
            raise RuntimeError(
                f"migration validation failed: source={len(source)} target={len(target)} "
                f"missing={missing} mismatched={mismatched}"
            )
        source_sets = await self._validate_collection_sets(v2=False)
        target_sets = await self._validate_collection_sets(v2=True)
        if source_sets != target_sets:
            raise RuntimeError(
                "migration collection count validation failed: "
                f"source={source_sets} target={target_sets}"
            )
        return {
            "count": len(source),
            "collection_sets": source_sets,
            "checksum": self._aggregate_checksum(source),
        }

    @staticmethod
    def _aggregate_checksum(values: Dict[str, str]) -> str:
        return hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


async def migrate(args: argparse.Namespace) -> int:
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    redis_config = dict((config.get("rag", {}) or {}).get("redis", {}) or {})
    import redis.asyncio as redis

    client = redis.Redis(
        host=redis_config.get("host", "localhost"),
        port=int(redis_config.get("port", 6380)),
        db=int(redis_config.get("db", 0)),
        decode_responses=False,
    )
    spec = MigrationSpec(
        key_prefix=redis_config.get("key_prefix", "rag"),
        batch_size=args.batch_size,
        pause_command=args.pause_command,
        resume_command=args.resume_command,
        switch_command=args.switch_command,
        rollback_command=args.rollback_command,
    )
    runner = MigrationRunner(client, spec)
    try:
        if args.rollback:
            await runner.rollback()
        else:
            plan = await runner.run(apply=args.apply)
            if not args.apply:
                print(json.dumps({"mode": "dry-run", "plan": plan}, indent=2))
            else:
                print(
                    json.dumps({"mode": "applied", "metrics": runner.metrics}, indent=2)
                )
    finally:
        await client.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--pause-command")
    parser.add_argument("--resume-command")
    parser.add_argument("--switch-command")
    parser.add_argument("--rollback-command")
    args = parser.parse_args()
    return asyncio.run(migrate(args))


if __name__ == "__main__":
    raise SystemExit(main())
