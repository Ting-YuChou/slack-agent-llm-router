# ToolRunner Roadmap

## Current State

The first web-search implementation uses `ToolRegistry` plus a direct `InferenceEngine` call into `WebSearchTool`. That is enough for one enrichment tool, but it will not scale cleanly once `url_fetch`, `calculator`, `time`, or `document_search` are added.

## Target Shape

Introduce a `ToolRunner` that owns execution policy and returns a structured trace:

```text
QueryRequest
  -> ModelRouter decides allowed/required tools
  -> ToolRunner evaluates policy, timeouts, and budgets
  -> Tool implementations run
  -> ToolResult traces enrich provider prompt and InferenceResponse
```

## Responsibilities

- Enforce `tool_policy`, `allowed_tools`, user tier, and per-tool config.
- Apply global and per-tool timeouts.
- Run independent tools concurrently when safe.
- Normalize all success/error outcomes into `ToolCall` and future `ToolResult`.
- Keep prompt formatting separate from tool execution.
- Attach trace data for analytics without leaking secrets.
- Centralize retry and rate-limit behavior.

## Initial Tool Set

- `web_search`: Tavily search results and citations.
- `url_fetch`: constrained reading of top source URLs.
- `calculator`: deterministic arithmetic and unit conversion.
- `time`: current date/time and timezone-sensitive answers.
- `document_search`: future internal knowledge-base retrieval.

## Migration Steps

1. Move `_run_web_search_if_needed()` from `InferenceEngine` into `ToolRunner.run()`.
2. Keep `WebSearchTool.format_context()` behavior but let `ToolRunner` assemble prompt sections.
3. Add `ToolResult` to schema when a second tool is implemented.
4. Extend analytics from web-search-specific fields to generic per-tool traces.
5. Keep provider-native tools optional adapters, not the primary cross-provider path.
