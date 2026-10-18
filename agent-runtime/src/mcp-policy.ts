import { parseGitHubRepositories } from "./github-repository.js";
import type { McpClaims } from "./mcp-token.js";

const CLICKHOUSE_VIEWS = new Set([
  "run_metrics_hourly", "model_metrics_daily", "tool_metrics_hourly", "test_metrics_hourly", "trace_metrics_hourly",
]);
const ACTIONS_GET = new Set(["get_workflow", "get_workflow_run", "get_workflow_run_usage", "get_job"]);
const ACTIONS_LIST = new Set(["list_workflows", "list_workflow_runs", "list_jobs", "list_artifacts"]);

export function authorizeGitHubTool(tool: string, args: Record<string, unknown>, claims: McpClaims): void {
  const repository = claims.scope.repository;
  if (tool === "search_code") {
    const qualifiers = typeof args.query === "string" ? searchRepositoryQualifiers(args.query) : null;
    if (!qualifiers || qualifiers.length !== 1 || qualifiers[0]?.toLowerCase() !== repository?.toLowerCase()) deny("Repository is not allowed");
  } else {
    const [owner, repo] = repository!.split("/");
    if (String(args.owner ?? "").toLowerCase() !== owner?.toLowerCase() || String(args.repo ?? "").toLowerCase() !== repo?.toLowerCase()) {
      deny("Repository is not allowed");
    }
  }
  const perPage = args.perPage ?? args.per_page;
  if (perPage !== undefined && (!Number.isInteger(perPage) || Number(perPage) < 1 || Number(perPage) > 50)) deny("Page size is not allowed");
  if (tool === "actions_get" && !ACTIONS_GET.has(String(args.method ?? ""))) deny("Actions method is not allowed");
  if (tool === "actions_list" && !ACTIONS_LIST.has(String(args.method ?? ""))) deny("Actions method is not allowed");
  if (tool === "get_job_logs") {
    const tail = args.tail_lines ?? args.tail ?? 500;
    if (!Number.isInteger(tail) || Number(tail) < 1 || Number(tail) > 500) deny("Job log line limit is not allowed");
    args.tail_lines = tail;
    delete args.tail;
  }
}

/** Read-only permissions covering the allowlisted GitHub MCP toolsets; metadata is implied. */
export const GITHUB_MCP_TOKEN_PERMISSIONS = {
  contents: "read", issues: "read", pull_requests: "read", actions: "read",
  security_events: "read", vulnerability_alerts: "read", secret_scanning_alerts: "read",
} as const;

/** Installation token scope for the GitHub MCP gateway: only the configured repositories, read-only. */
export function githubMcpTokenScope(configuredRepositories: string | undefined): { repositories: string[]; permissions: Record<string, "read"> } {
  const repositories = parseGitHubRepositories(configuredRepositories);
  if (repositories.length === 0) throw new Error("PI_AGENT_GITHUB_REPOSITORIES is required for the GitHub MCP gateway");
  const owners = new Set(repositories.map((repository) => repository.split("/")[0]!.toLowerCase()));
  if (owners.size !== 1) throw new Error("GitHub MCP repositories must belong to a single installation owner");
  return { repositories: repositories.map((repository) => repository.split("/")[1]!), permissions: { ...GITHUB_MCP_TOKEN_PERMISSIONS } };
}

const SCOPE_QUALIFIER = /(?<![A-Za-z0-9_])(repo|org|user|owner|enterprise):/gi;

/**
 * Returns the repo: qualifier values of a code search query, or null when the query could
 * widen its scope (OR, grouping, negated repo:, org:/user:/owner:/enterprise:, ambiguous quoting).
 */
function searchRepositoryQualifiers(query: string): string[] | null {
  if (query.includes("\\\"") || (query.match(/"/g)?.length ?? 0) % 2 !== 0) return null;
  // Quoted strings are literal text in GitHub code search.
  const tokens = query.replace(/"[^"]*"/g, " ").split(/\s+/).filter(Boolean);
  const repositories: string[] = [];
  for (const [index, token] of tokens.entries()) {
    if (/[()]/.test(token) || /^(or|\|\|)$/i.test(token)) return null;
    for (const match of token.matchAll(SCOPE_QUALIFIER)) {
      const negated = token.startsWith("-") || /^not$/i.test(tokens[index - 1] ?? "");
      if (match[1]!.toLowerCase() !== "repo" || negated || match.index !== 0) return null;
      repositories.push(token.slice(match.index + match[0].length));
    }
  }
  return repositories;
}

export function authorizeClickHouseTool(tool: string, args: Record<string, unknown>, _claims: McpClaims): void {
  if (tool === "list_databases") return;
  if (tool === "list_tables") {
    if (args.database !== "agent_mcp") deny("ClickHouse database is not allowed");
    return;
  }
  if (tool !== "run_query" || typeof args.query !== "string" || Buffer.byteLength(args.query) > 8192) deny("ClickHouse query is invalid");
  const query = args.query!.trim();
  if (!/^select\b/i.test(query) || /;\s*\S|\b(insert|update|delete|drop|alter|create|truncate|attach|detach|grant|revoke|system)\b/i.test(query) ||
      /\b(url|remote|file|s3)\s*\(/i.test(query) || /\bsettings\b/i.test(query)) deny("ClickHouse query is not read-only");
  const references = [...query.matchAll(/\b(?:from|join)\s+([\w.]+)/ig)].map((match) => match[1]!.toLowerCase());
  if (references.length === 0 || references.some((name) => {
    const [database, table] = name.includes(".") ? name.split(".", 2) : ["agent_mcp", name];
    return database !== "agent_mcp" || !CLICKHOUSE_VIEWS.has(table!);
  })) deny("ClickHouse query references a disallowed table");
  args.query = `${query} SETTINGS max_execution_time=5, max_result_rows=1000, result_overflow_mode='break', max_memory_usage=268435456`;
}

export function authorizeContext7Tool(tool: string, args: Record<string, unknown>, _claims: McpClaims): void {
  if (tool === "resolve-library-id") {
    if (typeof args.libraryName !== "string" || args.libraryName.length < 1 || args.libraryName.length > 200) deny("Context7 libraryName is invalid");
    if (typeof args.query !== "string" || args.query.length < 1 || args.query.length > 2000) deny("Context7 query is invalid");
    return;
  }
  if (tool === "query-docs") {
    if (typeof args.query !== "string" || args.query.length < 1 || args.query.length > 2000) deny("Context7 query is invalid");
    if (typeof args.libraryId !== "string" || args.libraryId.length > 300) deny("Context7 libraryId is invalid");
    return;
  }
  deny("Context7 tool is not allowed");
}

export function sanitizeClickHouseResponse(method: string, tool: string, value: Record<string, any>): void {
  if (method !== "tools/call" || tool !== "list_databases") return;
  for (const item of value.result?.content ?? []) {
    if (item?.type !== "text" || typeof item.text !== "string") continue;
    try {
      const parsed = JSON.parse(item.text);
      if (Array.isArray(parsed)) item.text = JSON.stringify(parsed.filter((entry) => entry === "agent_mcp"));
    } catch { /* upstream schema validation handles malformed JSON-RPC */ }
  }
  const structured = value.result?.structuredContent;
  if (structured && typeof structured.result === "string") structured.result = filterDatabaseList(structured.result);
}

export function validateContext7Response(method: string, _tool: string, value: Record<string, any>): void {
  if (method !== "tools/list") return;
  const tools = value.result?.tools;
  if (!Array.isArray(tools)) deny("Context7 contract is invalid");
  const names = tools.map((entry: any) => entry?.name).sort();
  const expected: Record<string, string[]> = {
    "resolve-library-id": ["libraryName", "query"],
    "query-docs": ["libraryId", "query"],
  };
  if (JSON.stringify(names) !== JSON.stringify(Object.keys(expected).sort()) || tools.some((entry: any) => {
    const fields = Object.keys(entry?.inputSchema?.properties ?? {}).sort();
    const required = [...(entry?.inputSchema?.required ?? [])].sort();
    const wanted = expected[entry?.name] ? [...expected[entry.name]].sort() : undefined;
    return entry?.inputSchema?.type !== "object" || !wanted || JSON.stringify(fields) !== JSON.stringify(wanted) || JSON.stringify(required) !== JSON.stringify(wanted);
  })) deny("Context7 contract has changed");
}

function filterDatabaseList(value: string): string {
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? JSON.stringify(parsed.filter((entry) => entry === "agent_mcp")) : value;
  } catch { return value; }
}

function deny(message: string): never { throw Object.assign(new Error(message), { code: "policy_denied" }); }
