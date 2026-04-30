import pytest

from src.tools.backends import TavilyBackend
from src.tools.web_search import WebSearchTool
from src.utils.schema import ResponseSource


class DummyBackend:
    provider_name = "tavily"

    def __init__(self):
        self.calls = 0

    async def search(self, query, options=None):
        self.calls += 1
        return [
            ResponseSource(
                title="Result",
                url="https://example.com/article",
                snippet=f"Snippet for {query}",
                source_domain="example.com",
                rank=1,
            )
        ]

    async def close(self):
        return None


class FailingBackend:
    provider_name = "tavily"

    async def search(self, query, options=None):
        raise RuntimeError("Tavily API key is not configured")

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_web_search_tool_returns_sources_and_context():
    backend = DummyBackend()
    tool = WebSearchTool(
        {"enabled": True, "cache_ttl_seconds": 300},
        backend=backend,
    )

    result = await tool.search("latest AI news", user_id="u1")

    assert result.sources[0].url == "https://example.com/article"
    assert result.tool_call.provider == "tavily"
    assert result.tool_call.result_count == 1
    assert "[1] Result" in result.context


@pytest.mark.asyncio
async def test_web_search_tool_uses_short_cache():
    backend = DummyBackend()
    tool = WebSearchTool(
        {"enabled": True, "cache_ttl_seconds": 300},
        backend=backend,
    )

    await tool.search("latest AI news", user_id="u1")
    cached = await tool.search("latest AI news", user_id="u1")

    assert backend.calls == 1
    assert cached.tool_call.arguments["cached"] is True


@pytest.mark.asyncio
async def test_web_search_tool_enforces_rate_limit():
    tool = WebSearchTool(
        {
            "enabled": True,
            "cache_ttl_seconds": 0,
            "per_user_rate_limit": 1,
        },
        backend=DummyBackend(),
    )

    await tool.search("first", user_id="u1")
    blocked = await tool.search("second", user_id="u1")

    assert blocked.blocked_reason == "web_search_rate_limited"
    assert blocked.tool_call.error == "web_search_rate_limited"


@pytest.mark.asyncio
async def test_web_search_tool_returns_structured_error_on_backend_failure():
    tool = WebSearchTool(
        {"enabled": True, "cache_ttl_seconds": 0},
        backend=FailingBackend(),
    )

    result = await tool.search("latest AI news", user_id="u1")

    assert result.sources == []
    assert result.context == ""
    assert result.blocked_reason == "web_search_error"
    assert result.tool_call.name == "web_search"
    assert "Tavily API key" in result.tool_call.error


@pytest.mark.asyncio
async def test_tavily_backend_filters_duplicates_domains_and_blocked_sources():
    backend = TavilyBackend(
        {
            "max_results": 5,
            "max_results_per_domain": 1,
            "blocked_domains": ["blocked.com"],
        }
    )
    try:
        sources = backend._parse_sources(
            [
                {
                    "title": "A",
                    "url": "https://example.com/article#fragment",
                    "content": "first",
                    "score": 0.9,
                },
                {
                    "title": "Duplicate",
                    "url": "https://example.com/article",
                    "content": "duplicate",
                    "score": 0.8,
                },
                {
                    "title": "Same domain",
                    "url": "https://example.com/other",
                    "content": "same domain",
                    "score": 0.7,
                },
                {
                    "title": "Blocked",
                    "url": "https://blocked.com/story",
                    "content": "blocked",
                    "score": 0.6,
                },
                {
                    "title": "B",
                    "url": "https://news.example.net/story",
                    "content": "fresh",
                    "published_date": "2026-04-30",
                    "score": 0.5,
                },
            ],
            max_results=5,
        )
    finally:
        await backend.close()

    assert [source.url for source in sources] == [
        "https://example.com/article",
        "https://news.example.net/story",
    ]
    assert sources[0].score == 0.9
    assert sources[1].published_at == "2026-04-30"


@pytest.mark.asyncio
async def test_tavily_backend_rewrites_queries_and_detects_freshness():
    backend = TavilyBackend({})
    try:
        assert backend._rewrite_query("please search latest OpenAI news") == (
            "latest OpenAI news"
        )
        assert backend._looks_fresh("current release news") is True
    finally:
        await backend.close()
