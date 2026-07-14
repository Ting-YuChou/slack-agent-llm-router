#!/usr/bin/env python3
"""Safely report or delete legacy provider scheduler queue sorted sets."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, List

import redis.asyncio as redis


LEGACY_PATTERN = "provider_scheduler:queue:*"
_DELETE_IF_LEGACY_STATE = """
local key_type = redis.call("TYPE", KEYS[1]).ok
local ttl = redis.call("PTTL", KEYS[1])
if key_type == "zset" and ttl == -1 then
    return {redis.call("DEL", KEYS[1]), key_type, ttl}
end
return {0, key_type, ttl}
"""


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _is_legacy_queue_key(key: str) -> bool:
    prefix = "provider_scheduler:queue:"
    if not key.startswith(prefix):
        return False
    suffix = key[len(prefix) :]
    parts = suffix.split(":")
    return len(parts) >= 2


async def cleanup_legacy_keys(
    client: Any, *, apply: bool = False, scan_count: int = 500
) -> Dict[str, Any]:
    """SCAN the narrow v1 pattern and optionally delete verified ZSETs."""
    candidates: List[str] = []
    skipped_wrong_type: List[str] = []
    skipped_versioned: List[str] = []
    skipped_with_ttl: List[str] = []

    async for raw_key in client.scan_iter(match=LEGACY_PATTERN, count=scan_count):
        key = _decode(raw_key)
        if not _is_legacy_queue_key(key):
            skipped_versioned.append(key)
            continue
        key_type = _decode(await client.type(raw_key)).lower()
        if key_type != "zset":
            skipped_wrong_type.append(key)
            continue
        if int(await client.pttl(raw_key)) != -1:
            skipped_with_ttl.append(key)
            continue
        candidates.append(key)

    candidates.sort()
    skipped_wrong_type.sort()
    skipped_versioned.sort()
    skipped_with_ttl.sort()
    deleted: List[str] = []
    if apply and candidates:
        for key in candidates:
            result = await client.eval(_DELETE_IF_LEGACY_STATE, 1, key)
            if int(result[0]) == 1:
                deleted.append(key)

    return {
        "apply": apply,
        "pattern": LEGACY_PATTERN,
        "candidates": candidates,
        "deleted": deleted,
        "skipped_wrong_type": skipped_wrong_type,
        "skipped_versioned": skipped_versioned,
        "skipped_with_ttl": skipped_with_ttl,
    }


def _text_report(report: Dict[str, Any]) -> str:
    mode = "APPLY" if report["apply"] else "DRY RUN"
    lines = [
        f"mode: {mode}",
        f"pattern: {report['pattern']}",
        f"candidates: {len(report['candidates'])}",
        f"deleted: {len(report['deleted'])}",
    ]
    lines.extend(f"candidate: {key}" for key in report["candidates"])
    lines.extend(f"deleted-key: {key}" for key in report["deleted"])
    lines.extend(f"skipped-wrong-type: {key}" for key in report["skipped_wrong_type"])
    lines.extend(f"skipped-versioned: {key}" for key in report["skipped_versioned"])
    lines.extend(f"skipped-with-ttl: {key}" for key in report["skipped_with_ttl"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://localhost:6379/4")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete verified candidates (default is a non-mutating dry run)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--scan-count", type=int, default=500)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    client = redis.Redis.from_url(args.redis_url, decode_responses=True)
    try:
        report = await cleanup_legacy_keys(
            client, apply=args.apply, scan_count=max(1, args.scan_count)
        )
    finally:
        await client.aclose()
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_text_report(report))


if __name__ == "__main__":
    asyncio.run(main())
