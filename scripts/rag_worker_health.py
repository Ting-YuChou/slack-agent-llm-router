#!/usr/bin/env python3
"""Exit successfully when a live RAG worker heartbeat exists in Redis."""

import argparse
import asyncio
import os

import redis.asyncio as redis
import yaml


async def _check(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    rag_config = dict(config.get("rag", {}) or {})
    redis_config = dict(rag_config.get("redis", {}) or {})
    password_env = redis_config.get("password_env", "REDIS_PASSWORD")
    client = redis.Redis(
        host=redis_config.get("host", "localhost"),
        port=int(redis_config.get("port", 6379)),
        db=int(redis_config.get("db", 0)),
        password=os.getenv(str(password_env)) if password_env else None,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    prefix = str(redis_config.get("key_prefix") or "rag")
    try:
        async for _ in client.scan_iter(match=f"{prefix}:worker:heartbeat:*", count=1):
            return 0
        return 1
    finally:
        await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return asyncio.run(_check(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
