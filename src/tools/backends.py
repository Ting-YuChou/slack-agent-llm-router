"""Search backend adapters for the web_search tool."""

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx

from src.utils.schema import ResponseSource, WebSearchOptions


def _source_domain(url: str) -> Optional[str]:
    hostname = urlparse(url).hostname
    if not hostname:
        return None
    return hostname.lower().removeprefix("www.")


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    normalized_path = parsed.path.rstrip("/") or parsed.path
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            normalized_path,
            "",
            parsed.query,
            "",
        )
    )


class TavilyBackend:
    """Minimal Tavily Search API adapter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.api_key = self.config.get("api_key") or os.getenv(
            self.config.get("api_key_env", "TAVILY_API_KEY")
        )
        self.base_url = str(
            self.config.get("base_url", "https://api.tavily.com")
        ).rstrip("/")
        self.timeout_seconds = float(self.config.get("timeout_seconds", 5.0))
        self.max_results = int(self.config.get("max_results", 5))
        self.search_depth = str(self.config.get("search_depth", "basic"))
        self.include_answer = bool(self.config.get("include_answer", False))
        self.max_results_per_domain = int(self.config.get("max_results_per_domain", 2))
        self.blocked_domains = {
            str(domain).lower().removeprefix("www.")
            for domain in list(self.config.get("blocked_domains", []) or [])
            if str(domain).strip()
        }
        self.freshness_days = self.config.get("freshness_days", 14)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        return "tavily"

    async def search(
        self,
        query: str,
        options: Optional[WebSearchOptions] = None,
    ) -> List[ResponseSource]:
        if not self.api_key:
            raise RuntimeError("Tavily API key is not configured")

        max_results = (
            options.max_results
            if options and options.max_results is not None
            else self.max_results
        )
        search_query = self._rewrite_query(query)
        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "query": search_query,
            "search_depth": (
                options.search_depth
                if options and options.search_depth is not None
                else self.search_depth
            ),
            "include_answer": (
                options.include_answer
                if options and options.include_answer is not None
                else self.include_answer
            ),
            "max_results": max_results,
        }
        if options and options.topic:
            payload["topic"] = options.topic
        if options and options.days:
            payload["days"] = options.days
        elif self._looks_fresh(query):
            payload["topic"] = "news"
            if self.freshness_days:
                payload["days"] = int(self.freshness_days)

        response = await self.client.post("/search", json=payload)
        response.raise_for_status()
        data = response.json()
        return self._parse_sources(data.get("results", []), max_results)

    def _rewrite_query(self, query: str) -> str:
        compact = " ".join(query.split())
        compact = re.sub(r"^(please|can you|could you)\s+", "", compact, flags=re.I)
        compact = re.sub(r"\b(search|look up|查詢|搜尋)\b", "", compact, flags=re.I)
        compact = " ".join(compact.split())
        return compact[:300] or query.strip()[:300]

    def _looks_fresh(self, query: str) -> bool:
        lowered = query.lower()
        freshness_markers = {
            "latest",
            "today",
            "current",
            "recent",
            "news",
            "price",
            "release",
            "now",
            "最新",
            "今天",
            "現在",
            "新聞",
        }
        return any(marker in lowered for marker in freshness_markers)

    def _parse_sources(
        self, results: List[Dict[str, Any]], max_results: int
    ) -> List[ResponseSource]:
        sources: List[ResponseSource] = []
        seen_urls = set()
        domain_counts: Dict[str, int] = defaultdict(int)
        for result in results:
            url = str(result.get("url") or "").strip()
            if not url:
                continue
            normalized_url = _normalize_url(url)
            if normalized_url in seen_urls:
                continue
            domain = _source_domain(normalized_url)
            if domain in self.blocked_domains:
                continue
            if domain and domain_counts[domain] >= self.max_results_per_domain:
                continue
            snippet = str(
                result.get("content")
                or result.get("snippet")
                or result.get("raw_content")
                or ""
            ).strip()
            seen_urls.add(normalized_url)
            if domain:
                domain_counts[domain] += 1
            sources.append(
                ResponseSource(
                    title=str(result.get("title") or normalized_url),
                    url=normalized_url,
                    snippet=snippet[:1000],
                    source_domain=domain,
                    published_at=result.get("published_date"),
                    score=result.get("score"),
                    rank=len(sources) + 1,
                )
            )
            if len(sources) >= max_results:
                break
        return sources

    async def close(self) -> None:
        await self.client.aclose()
