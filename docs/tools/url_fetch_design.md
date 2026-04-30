# URL Fetch Tool Design

## Goal

Add a constrained `url_fetch` tool after the Tavily `web_search` MVP so the router can read a small number of source pages when snippets are not enough. The tool should improve grounded answers without becoming a general crawler.

## Proposed Flow

1. `web_search` returns ranked `ResponseSource` entries.
2. `url_fetch` selects the top 2-3 HTTP(S) URLs after dedupe/domain filtering.
3. It fetches text-like content with short timeouts and strict size limits.
4. It extracts the main readable text, strips scripts/styles/navigation, and truncates per URL.
5. The provider prompt receives a separate `Fetched source excerpts:` section.
6. `InferenceResponse.sources` keeps the original URL metadata and adds fetch status in tool trace metadata.

## Safety Constraints

- Only allow `http` and `https` URLs.
- Block private IP ranges, localhost, link-local, and metadata-service addresses.
- Enforce `Content-Type` allowlist: `text/html`, `text/plain`, `application/xhtml+xml`.
- Set connect/read timeout to 3-5 seconds per URL.
- Limit response body bytes before parsing.
- Limit extracted text to a configurable `max_chars_per_url`.
- Treat fetched text as untrusted input and keep the existing prompt-injection warning.
- Do not send cookies, Slack tokens, API keys, or user credentials.

## Config Shape

```yaml
tools:
  url_fetch:
    enabled: false
    max_urls: 3
    timeout_seconds: 5
    max_response_bytes: 1048576
    max_chars_per_url: 4000
    allowed_content_types:
      - text/html
      - text/plain
      - application/xhtml+xml
    blocked_domains: []
```

## Tests

- Reject private and non-HTTP URLs.
- Skip binary or oversized responses.
- Extract deterministic text from simple HTML.
- Preserve source rank and URL in response trace.
- Degrade gracefully when one URL fails while others succeed.
