#!/usr/bin/env python3
"""Smoke test for the /route web_search tool path."""

import argparse
import json
import os
import sys
from typing import Any, Dict

import httpx


def _build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "query": args.query,
        "user_id": args.user_id,
        "user_tier": args.user_tier,
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
        "tool_policy": "required",
        "allowed_tools": ["web_search"],
        "web_search_options": {
            "max_results": args.max_results,
            "search_depth": args.search_depth,
        },
    }


def _validate_response(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    tool_calls = payload.get("tool_calls") or []
    sources = payload.get("sources") or []
    if not tool_calls:
        raise RuntimeError("Expected at least one tool_call in response")

    web_search_call = next(
        (
            tool_call
            for tool_call in tool_calls
            if tool_call.get("name") == "web_search"
        ),
        None,
    )
    if not web_search_call:
        raise RuntimeError("Expected a web_search tool_call in response")

    if args.expect == "sources" and not sources:
        raise RuntimeError(
            f"Expected web search sources, got tool error: {web_search_call.get('error')}"
        )
    if args.expect == "tool-error" and not web_search_call.get("error"):
        raise RuntimeError("Expected structured tool error but web_search had no error")


def run(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("LLM_ROUTER_API_KEY", "dev-api-key")
    endpoint = args.api_url.rstrip("/") + "/route"
    response = httpx.post(
        endpoint,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json=_build_payload(args),
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Route returned inference error: {payload['error']}")

    _validate_response(args, payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "model_name": payload.get("model_name"),
                "provider": payload.get("provider"),
                "sources": len(payload.get("sources") or []),
                "tool_calls": payload.get("tool_calls") or [],
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test /route web_search")
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--query", default="What is the latest OpenAI news today?")
    parser.add_argument("--user-id", default="web-search-smoke-user")
    parser.add_argument("--user-tier", default="premium")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument(
        "--search-depth", choices=["basic", "advanced"], default="basic"
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--expect",
        choices=["sources", "tool-error", "any"],
        default="sources",
        help="Expected web_search outcome.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"web_search smoke failed: {exc}", file=sys.stderr)
        raise
