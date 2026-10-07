# Pi Codemode read-only MCP

The Agent runtime accepts only the trusted MCP registry compiled into `agent-runtime/src/mcp-config.ts`.
Repository `.pi/mcp.json` files are not loaded.

## Enablement

MCP is disabled by default:

```text
PI_AGENT_MCP_MODE=off
PI_AGENT_MCP_SERVERS=github,clickhouse,context7,codegraph
```

Set `PI_AGENT_MCP_MODE=read_only` and enable one server at a time. The legacy
`github_read_only` mode and `PI_AGENT_MCP_GATEWAY_URL` remain available for GitHub-only deployments.
An unavailable optional gateway appears as degraded in `/health`; it does not block an Agent run.

HTTP gateways receive a server-specific derived signing key. Keep
`MCP_GATEWAY_SIGNING_SECRET` in the Agent runtime and derive deployment values with:

```bash
MCP_GATEWAY_SIGNING_SECRET=... node agent-runtime/scripts/derive-mcp-secret.mjs github
MCP_GATEWAY_SIGNING_SECRET=... node agent-runtime/scripts/derive-mcp-secret.mjs clickhouse
MCP_GATEWAY_SIGNING_SECRET=... node agent-runtime/scripts/derive-mcp-secret.mjs context7
```

## Boundaries

- GitHub and Context7 gateways join the internal Agent network and the MCP egress network.
- The ClickHouse gateway joins the internal Agent network and the internal observability network.
- GitHub App, Context7, ClickHouse database, and upstream bearer credentials stay in their gateway or upstream container.
- The Agent receives only run-bound tokens. Token claims bind the server, run, session, Slack user, scope, exact tools, call budget, and expiry.
- Codemode results are treated as untrusted data and reduced to an 8 KiB summary.

## ClickHouse

Apply `docker/observability/agent-mcp-views.sql` after the Agent analytics tables exist.
The `agent_mcp_reader` account defined by `agent-mcp-users.xml` can select only five aggregate views.
The Gateway additionally rejects non-SELECT queries, raw tables, multiple statements, external table
functions, custom SETTINGS, oversized queries, and results above 512 KiB.

The `docker-compose.agent-mcp.yaml` overlay contains the initializer, upstream MCP, and gateway.
It expects the Agent observability profile and its internal network to be running.

## CodeGraphContext

The local proxy starts with a fixed seven-tool catalog. The first tool call copies current tracked and
non-ignored source files into a bounded run-scoped mirror, indexes that mirror, and starts CGC.
The proxy runs CGC with a private HOME outside the worktree, `CGC_ALLOWED_ROOTS` set to the mirror,
and a 384 MiB Kuzu buffer. Container termination removes the tmpfs mirror and index.

## Rollout and teardown

Enable in this order: GitHub, ClickHouse, Context7, CodeGraph. Observe the first 25 runs for each
server and remove its name from `PI_AGENT_MCP_SERVERS` to roll it back. Use
`PI_AGENT_MCP_MODE=off` for full rollback.

`scripts/run_slack_demo.sh` tracks every container and network it creates and stops or removes them
on normal exit, failure, SIGINT, or SIGTERM. CodeGraph state lives only in the Agent container tmpfs.
