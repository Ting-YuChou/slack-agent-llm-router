#!/usr/bin/env python3
"""
Smoke test for Slack per-user memory on a real Redis Stack instance.
"""

import argparse
import asyncio
import sys
import uuid
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory import MemoryManager


def build_config(args, suffix: str):
    key_prefix = f"{args.key_prefix}_{suffix}"
    return {
        "enabled": True,
        "backend": "redis_stack",
        "key_prefix": key_prefix,
        "max_items_per_user": 50,
        "search": {
            "max_results": 3,
            "max_context_chars": 1000,
            "max_item_chars": 300,
            "keyword_weight": 0.45,
            "vector_weight": 0.45,
            "recency_weight": 0.05,
            "importance_weight": 0.05,
        },
        "embedding": {
            "provider": "hash",
            "dimensions": args.dimensions,
            "timeout": 5,
        },
        "redis": {
            "host": args.redis_host,
            "port": args.redis_port,
            "db": args.redis_db,
            "key_prefix": key_prefix,
            "password_env": args.redis_password_env,
            "dedicated_service_recommended": True,
        },
    }


async def run_smoke(args):
    suffix = uuid.uuid4().hex[:8]
    config = build_config(args, suffix)
    manager = MemoryManager(config)
    scope = "T-smoke:U-smoke"

    try:
        await manager.initialize()
        await manager.remember(
            scope,
            "Prefer concise Python examples when explaining API design.",
            metadata={"source": "slack", "channel_id": "C-smoke"},
            importance=0.8,
        )
        await manager.remember(
            scope,
            "The analytics pipeline uses ClickHouse for query logs.",
            metadata={"source": "slack", "channel_id": "C-smoke"},
            importance=0.7,
        )

        results = await manager.search(
            scope,
            "Give me a concise Python API example",
            metadata={"source": "slack"},
        )
        if not results:
            raise RuntimeError("Expected at least one Redis Stack memory search result")
        if not any(result.match_source in {"vector", "hybrid"} for result in results):
            raise RuntimeError("Expected Redis Stack vector retrieval to contribute")

        context = manager.build_context(results, "user: previous short context")
        if "Long-term user memory:" not in context:
            raise RuntimeError("Expected memory context injection")
        if "score" in context or "hybrid" in context:
            raise RuntimeError("Memory context leaked dynamic debug fields")

        deleted = await manager.forget_all(scope)
        if deleted < 1:
            raise RuntimeError("Expected Redis Stack memory cleanup to delete records")

        print(
            "Redis Stack memory smoke test passed: "
            f"host={args.redis_host} port={args.redis_port} db={args.redis_db} "
            f"key_prefix={config['key_prefix']}"
        )
    finally:
        store = getattr(manager, "store", None)
        client = getattr(store, "client", None)
        index_name = getattr(store, "index_name", None)
        if client is not None and index_name:
            with suppress(Exception):
                await client.execute_command("FT.DROPINDEX", index_name, "DD")
        await manager.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Redis Stack smoke test for Slack memory"
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6380)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-password-env", default="REDIS_PASSWORD")
    parser.add_argument("--key-prefix", default="slack_memory_smoke")
    parser.add_argument("--dimensions", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_smoke(parse_args()))
