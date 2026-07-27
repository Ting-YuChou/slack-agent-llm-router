#!/usr/bin/env python3
"""Dry-run-first removal of legacy RAG visual-vector keys and index."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List


async def cleanup_legacy_visual_index(
    client: Any,
    *,
    key_prefix: str = "rag",
    control_plane_version: str = "v2",
    apply: bool = False,
) -> Dict[str, Any]:
    patterns = [
        f"{key_prefix}:{control_plane_version}:visual_chunk:*",
        f"{key_prefix}:visual_chunk:*",
        f"{key_prefix}:document:{control_plane_version}:*:visual_chunks",
        f"{key_prefix}:document:*:*:visual_chunks",
    ]
    keys: List[bytes] = []
    seen = set()
    for pattern in patterns:
        async for key in client.scan_iter(match=pattern):
            if key not in seen:
                seen.add(key)
                keys.append(key)

    versioned_index_name = (
        f"{key_prefix}:idx:{control_plane_version}:visual_chunks"
        if control_plane_version
        else f"{key_prefix}:idx:visual_chunks"
    )
    index_names = list(
        dict.fromkeys([versioned_index_name, f"{key_prefix}:idx:visual_chunks"])
    )
    existing_indexes = []
    for index_name in index_names:
        try:
            await client.execute_command("FT.INFO", index_name)
            existing_indexes.append(index_name)
        except Exception:
            pass

    deleted = 0
    if apply:
        for offset in range(0, len(keys), 500):
            deleted += int(await client.unlink(*keys[offset : offset + 500]))
        for index_name in existing_indexes:
            await client.execute_command("FT.DROPINDEX", index_name)

    return {
        "apply": apply,
        "key_count": len(keys),
        "deleted_keys": deleted,
        "visual_index": versioned_index_name,
        "visual_index_exists": versioned_index_name in existing_indexes,
        "visual_indexes": index_names,
        "existing_visual_indexes": existing_indexes,
    }


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default=os.getenv("RAG_REDIS_URL"))
    parser.add_argument("--key-prefix", default="rag")
    parser.add_argument("--control-plane-version", default="v2")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.redis_url:
        parser.error("--redis-url or RAG_REDIS_URL is required")

    import redis.asyncio as redis

    client = redis.Redis.from_url(args.redis_url, decode_responses=False)
    try:
        result = await cleanup_legacy_visual_index(
            client,
            key_prefix=args.key_prefix,
            control_plane_version=args.control_plane_version,
            apply=args.apply,
        )
        print(json.dumps(result, sort_keys=True))
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
