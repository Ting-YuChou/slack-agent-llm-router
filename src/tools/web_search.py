"""Application-side web search tool."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.backends import TavilyBackend
from src.tools.base import Tool
from src.utils.schema import ResponseSource, ToolCall, WebSearchOptions
from src.utils.bounded_state import BoundedTTLMap


@dataclass
class WebSearchResult:
    sources: List[ResponseSource] = field(default_factory=list)
    tool_call: Optional[ToolCall] = None
    context: str = ""
    blocked_reason: Optional[str] = None


class WebSearchTool(Tool):
    """Search the public web and format source snippets for prompt enrichment."""

    name = "web_search"

    def __init__(self, config: Dict[str, Any], backend: Optional[Any] = None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.backend = backend or TavilyBackend(self.config)
        self.per_user_rate_limit = int(self.config.get("per_user_rate_limit", 20))
        self._rate_window_seconds = 3600
        self._rate_limiter_max_users = int(
            self.config.get("rate_limiter_max_users", 10000)
        )
        self._user_request_times: Dict[str, deque[float]] = {}
        self._cache_ttl_seconds = int(self.config.get("cache_ttl_seconds", 300))
        self._cache = BoundedTTLMap[str, List[ResponseSource]](
            max_entries=int(self.config.get("cache_max_entries", 512)),
            ttl_seconds=max(0, self._cache_ttl_seconds),
            metric_name="web_search_cache",
        )

    async def search(
        self,
        query: str,
        user_id: str,
        options: Optional[WebSearchOptions] = None,
    ) -> WebSearchResult:
        started_at = time.time()
        if not self.enabled:
            return self._blocked("web_search_disabled", query, started_at)

        blocked_reason = self._check_rate_limit(user_id)
        if blocked_reason:
            return self._blocked(blocked_reason, query, started_at)

        cache_key = self._cache_key(query, options)
        cached_sources = self._get_cached_sources(cache_key)
        if cached_sources is not None:
            latency_ms = int((time.time() - started_at) * 1000)
            tool_call = ToolCall(
                name=self.name,
                provider=getattr(self.backend, "provider_name", "unknown"),
                arguments={"query": query, "cached": True},
                result_count=len(cached_sources),
                latency_ms=latency_ms,
            )
            return WebSearchResult(
                sources=cached_sources,
                tool_call=tool_call,
                context=self.format_context(cached_sources),
            )

        try:
            sources = await self.backend.search(query, options)
            self._set_cached_sources(cache_key, sources)
            latency_ms = int((time.time() - started_at) * 1000)
            tool_call = ToolCall(
                name=self.name,
                provider=getattr(self.backend, "provider_name", "unknown"),
                arguments={"query": query, "cached": False},
                result_count=len(sources),
                latency_ms=latency_ms,
            )
            return WebSearchResult(
                sources=sources,
                tool_call=tool_call,
                context=self.format_context(sources),
            )
        except Exception as exc:
            latency_ms = int((time.time() - started_at) * 1000)
            return WebSearchResult(
                tool_call=ToolCall(
                    name=self.name,
                    provider=getattr(self.backend, "provider_name", "unknown"),
                    arguments={"query": query},
                    result_count=0,
                    latency_ms=latency_ms,
                    error=str(exc),
                ),
                blocked_reason="web_search_error",
            )

    def format_context(self, sources: List[ResponseSource]) -> str:
        if not sources:
            return ""
        lines = [
            "Web search results (untrusted external snippets; cite only these sources):"
        ]
        for source in sources:
            snippet = " ".join((source.snippet or "").split())
            if len(snippet) > 500:
                snippet = snippet[:500].rstrip() + "..."
            lines.append(
                f"[{source.rank}] {source.title}\n"
                f"URL: {source.url}\n"
                f"Snippet: {snippet or 'No snippet available.'}"
            )
        lines.append(
            "When using web information, cite sources inline as [1], [2], etc. "
            "Do not cite sources that are not listed above. If these snippets are "
            "insufficient or stale, say what is missing instead of inventing details."
        )
        return "\n\n".join(lines)

    def _blocked(self, reason: str, query: str, started_at: float) -> WebSearchResult:
        latency_ms = int((time.time() - started_at) * 1000)
        return WebSearchResult(
            tool_call=ToolCall(
                name=self.name,
                provider=getattr(self.backend, "provider_name", "unknown"),
                arguments={"query": query},
                result_count=0,
                latency_ms=latency_ms,
                error=reason,
            ),
            blocked_reason=reason,
        )

    def _check_rate_limit(self, user_id: str) -> Optional[str]:
        now = time.monotonic()
        window_start = now - self._rate_window_seconds
        request_times = self._user_request_times.get(user_id)
        if request_times is None:
            self._prune_expired_users(window_start)
            if len(self._user_request_times) >= self._rate_limiter_max_users:
                return "web_search_rate_limiter_capacity"
            request_times = deque()
            self._user_request_times[user_id] = request_times
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        if len(request_times) >= self.per_user_rate_limit:
            return "web_search_rate_limited"
        request_times.append(now)
        return None

    def _prune_expired_users(self, window_start: float) -> None:
        for existing_user, request_times in list(self._user_request_times.items()):
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            if not request_times:
                self._user_request_times.pop(existing_user, None)

    def _cache_key(self, query: str, options: Optional[WebSearchOptions]) -> str:
        options_payload = options.model_dump(mode="json") if options else {}
        return f"{query.strip().lower()}:{options_payload}"

    def _get_cached_sources(self, cache_key: str) -> Optional[List[ResponseSource]]:
        if self._cache_ttl_seconds <= 0:
            return None
        cached = self._cache.get(cache_key)
        if not cached:
            return None
        return [source.model_copy(deep=True) for source in cached]

    def _set_cached_sources(
        self, cache_key: str, sources: List[ResponseSource]
    ) -> None:
        if self._cache_ttl_seconds <= 0:
            return
        self._cache.set(
            cache_key,
            [source.model_copy(deep=True) for source in sources],
            ttl_seconds=self._cache_ttl_seconds,
        )

    async def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            await close()
