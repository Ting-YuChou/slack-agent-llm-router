#!/usr/bin/env python3
"""Create and validate versioned RAG RediSearch indexes; dry-run by default."""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.vector_store import RedisStackRagVectorStore


async def migrate(config_path: str, *, apply: bool) -> int:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    rag_config = dict(config.get("rag", {}) or {})
    store = RedisStackRagVectorStore(rag_config)
    if not apply:
        print(f"DRY RUN: ensure index {store.index_name}")
        if store._visual_enabled():
            print(f"DRY RUN: ensure index {store.visual_index_name}")
        print("Existing chunk hashes remain in place; no active data is rewritten.")
        return 0

    await store.initialize()
    try:
        info = await store.client.execute_command("FT.INFO", store.index_name)
        if not info:
            raise RuntimeError(f"index validation failed: {store.index_name}")
        print(f"READY: {store.index_name}")
    finally:
        await store.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return asyncio.run(migrate(args.config, apply=args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
