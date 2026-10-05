# Slack Agent LLM Router

Multi-model LLM router with a FastAPI API, Slack bot integration, Kafka/ClickHouse analytics, and local Docker Compose support.

## What It Does

- Routes requests across `gpt-5`, `claude-sonnet-4-6`, and optional local `vLLM` models
- Exposes a protected API for query routing and dashboard access
- Supports Slack bot interactions through `app_mention`, slash commands, and active reply threads
- Can enrich current-info answers with Tavily-backed `web_search` results and structured sources
- Persists analytics and request events through Kafka and ClickHouse when the pipeline is enabled
- Supports Redis-backed cache and Slack state for multi-process durability

## Current Routing Behavior

The default host-run config in [config/config.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.yaml) is tuned like this:

- `gpt-5` is the default model
- `claude-sonnet-4-6` gets extra weight for `enterprise` users and difficult `analysis` / `reasoning` requests
- host-run fast-lane traffic targets the external `vLLM` model `qwen3.6-27b-fast`
- simple general tasks, including free-tier traffic, can also select local `vLLM` models such as `qwen3.6-27b-fast` or `mistral-7b` through normal capability/scoring/rule routing
- if a cloud model fails on the non-streaming API path, inference can fall back to a local model when one is configured
- if `qwen3.6-27b-fast` is temporarily unavailable, host-run config allows fallback to `mistral-7b` only for simple small text requests that are not attachment-heavy, tool/RAG-required, or complex reasoning
- fast-lane routing is explicit-SLA only: set `metadata.requires_low_latency: true`, `metadata.latency_sla: "low"` or `"interactive"`, use API `priority >= 4`, or use Slack `/llm fast <query>`
- fast-lane requests still need to be simple enough for the configured fast-lane model; complex reasoning, long-context, and attachment-heavy requests stay on the normal capability-aware path

The compose runtime config in [config/config.compose.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.compose.yaml) is intentionally cloud-only:

- it does not start a local `vLLM` server
- it disables local-model scoring bias so the API does not route to a non-existent local model

## Single-GPU vLLM Fast Lane

The checked-in host config can route explicit low-latency text requests to
`qwen3.6-27b-fast`, but this repository does not manage the GPU model server.
Start vLLM separately before running the API:

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b-fast \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  --generation-config vllm \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

MTP speculative decoding is enabled by the vLLM `--speculative-config` flag.
The router only sends OpenAI-compatible chat completion requests to the endpoint
configured under `inference.vllm`.

Recommended single-card target:

- 40-48GB VRAM
- text-only coding/chat requests
- `16K-32K` context
- low concurrency; if startup OOMs, lower `--max-num-seqs` to `2`, then lower `--max-model-len` to `16384`

Smoke checklist:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/v1/models
curl -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-fast","messages":[{"role":"user","content":"Write a tiny Python add function."}],"max_tokens":128,"temperature":0.3}'
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{"query":"Write a tiny Python add function.","user_id":"test-user","metadata":{"requires_low_latency":true}}'
```

For benchmarking, run the same vLLM server once with
`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`
and once without it, then compare vLLM serving latency plus `/route` end-to-end
latency on the same prompt set.

## vLLM Serving Pool

The application can either point `inference.vllm.base_url` at an external
[vLLM Production Stack](https://docs.vllm.ai/projects/production-stack/en/latest/)
router, or use the repo-local provider pool by configuring multiple
OpenAI-compatible vLLM endpoints. The old single `base_url` / `host` / `port`
config remains valid and is normalized to one endpoint.

The repo-local pool improves health/failover/observability with one endpoint.
Tail-latency, queue spillover, prefix-cache locality, and model isolation only
become meaningful when two or more serving endpoints are reachable.

Example two-endpoint Qwen pool:

```yaml
inference:
  vllm:
    api_mode: chat_completions
    routing_strategy: least_outstanding_prefix_aware
    metrics_scrape_enabled: true
    health_check_interval_seconds: 15
    failure_cooldown_seconds: 30
    metrics_refresh_seconds: 5
    prefix_affinity_ttl_seconds: 300
    endpoints:
      - name: qwen-a
        base_url: http://127.0.0.1:8001
        models: [qwen3.6-27b-fast]
        weight: 1.0
        max_outstanding: 4
        health_path: /health
        metrics_path: /metrics
        prefix_cache_enabled: true
      - name: qwen-b
        base_url: http://127.0.0.1:8002
        models: [qwen3.6-27b-fast]
        weight: 1.0
        max_outstanding: 4
        health_path: /health
        metrics_path: /metrics
        prefix_cache_enabled: true
```

Pool behavior:

- endpoints with a non-empty `models` list only serve those model names
- unhealthy endpoints are skipped until their cooldown expires
- outstanding request count and optional vLLM `/metrics` signals influence endpoint choice
- repeated prompt prefixes prefer the same healthy, non-saturated endpoint for best-effort prefix-cache locality
- prefix-aware routing is process-local in this app; use an external vLLM Production Stack router if you need a dedicated serving gateway
- endpoint pool failover keeps the same selected model; cross-model fallback such as `qwen3.6-27b-fast` to `mistral-7b` is controlled separately by `inference.vllm.model_fallback`

Pool smoke checklist:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8002/v1/models
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{"query":"Write a tiny Python add function.","user_id":"test-user","metadata":{"requires_low_latency":true}}'
```

To check queue failover, stop one vLLM endpoint and resend the same `/route`
request. To check prefix affinity, send repeated requests with the same long
stable prefix and inspect provider health output for endpoint outstanding counts
and last errors. vLLM exports request and KV/cache metrics on `/metrics`; the
pool reads `vllm:num_requests_running`, `vllm:num_requests_waiting`, and
`vllm:gpu_cache_usage_perc` when `metrics_scrape_enabled` is true.

## Provider Capacity Scheduler

Provider guardrails and the provider scheduler handle different layers:

- routing guardrails influence which model/provider the router selects
- `inference.scheduler` controls whether a selected provider/model may be called now
- provider/model active request limits, request rate, and token budgets still come from `api.rate_limiting`
- scheduler queues are per provider/model and ordered by explicit low-latency intent, request priority, user tier, then age
- scheduler-managed retry and circuit-breaker state protect providers from retry storms and repeated unhealthy calls

The `/route` API remains synchronous. If provider capacity is unavailable and no
safe local fallback applies, scheduler rejection bubbles through the existing
admission response path, including `Retry-After` when present.

## API Surface

Public health endpoints:

- `GET /live`
- `GET /ready`
- `GET /health`

Protected endpoints:

- `POST /route`
- `GET /metrics`
- `GET /dashboard`
- `GET /dashboard/logs`

API key auth is enabled by default. The accepted keys come from `LLM_ROUTER_API_KEYS`.

Example:

```bash
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{"query": "Summarize this design", "user_id": "test-user"}'
```

Web search example, when `tools.web_search.enabled` is true and `TAVILY_API_KEY` is set:

```bash
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key" \
  -d '{
    "query": "What is the latest OpenAI news today?",
    "user_id": "test-user",
    "tool_policy": "required",
    "allowed_tools": ["web_search"],
    "web_search_options": {"max_results": 3}
  }'
```

Successful web-search responses include `sources[]` and a `web_search` entry in
`tool_calls[]`. If Tavily is not configured or a search fails, the request still
falls back to a normal model answer and records the structured tool error.
Search results are deduplicated by normalized URL, capped per source domain, and
can exclude configured `blocked_domains`. Current-info queries automatically ask
Tavily for fresher/news-oriented results.
The follow-up designs for constrained URL reading and multi-tool orchestration live
in [docs/tools/url_fetch_design.md](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/docs/tools/url_fetch_design.md)
and [docs/tools/tool_runner_roadmap.md](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/docs/tools/tool_runner_roadmap.md).

## Slack Behavior

The Slack bot no longer replies to every channel message.

It responds only to:

- `app_mention`
- slash commands such as `/llm help` and free-form queries such as `/llm explain this error`
- replies inside an active bot thread

Allowed Slack channels can be configured by channel name or channel ID.

Slack file attachments on message / mention events are converted into `QueryRequest.attachments`.
Files up to the configured size limit are downloaded with the bot token; text-like files are inlined into provider prompts, while larger or binary files keep metadata and private URLs only.

Slack tiering:

- `slack.user_tiers.overrides` maps Slack user IDs to `free` / `premium` / `enterprise`
- `slack.rate_limiting.by_tier` applies tier-specific hourly and burst limits
- model visibility respects the routed model's tier access rules

Slack state backends:

- `memory`: process-local only
- `file`: persists a JSON snapshot to the configured path
- `redis`: persists per-user, per-rate-limit, per-conversation, and per-thread keys for multi-process durability

Persisted Slack state includes:

- user tier and preferences
- rate-limit counters
- conversation history
- active bot thread tracking

Slack per-user memory:

- disabled by default; enable with `slack.memory.enabled: true`
- stores only explicit `/llm remember <text>` entries in the first version
- retrieves memories with hybrid keyword + Redis Stack vector search before a Slack query
- scopes memory by `team_id:user_id` when Slack provides the workspace ID
- stores memories as channel-scoped by default; use `--global` for cross-channel preferences
- injects only global memories plus memories saved in the current channel
- falls back to keyword-only search if embedding generation fails
- memory management commands should be used as Slack slash commands so responses stay ephemeral

Memory commands:

- `/llm remember <text>` saves a channel-scoped long-term memory for the current Slack user
- `/llm remember --global <text>` saves a cross-channel user preference memory
- `/llm memories [query]` lists or searches current-channel and global memories
- `/llm memories --global [query]` lists or searches global memories only
- `/llm memories --all [query]` lists or searches all memories for the current user
- `/llm forget <memory_id>` deletes one memory
- `/llm forget all` deletes all memories for the current user

Memory storage should not share the response-cache Redis DB. The default runtime
separates Redis usage as response cache DB 0, policy cache DB 1, Slack state DB 2,
and memory DB 3. For production, prefer a dedicated Redis Stack service/instance
for `slack.memory.redis` so vector/hash memory data cannot evict response cache
entries through Redis `maxmemory` policy. Host-run config uses `localhost:6380`
for Redis Stack; compose config uses the internal `redis-stack:6379` service.

Redis Stack smoke test:

```bash
make smoke-redis-stack-memory
```

This starts the dedicated `redis-stack` Docker service, creates a RediSearch vector
index through the real `RedisStackMemoryStore`, writes explicit memories, performs
hybrid retrieval, verifies deterministic context formatting, and deletes the smoke
test data/index. The service maps to host port `6380` so it does not collide with
the regular response-cache Redis on `6379`.

## Quick Start

### Single-Model Slack Demo

This is the smallest live demo with two execution modes in the same Slack bot:
normal queries use the existing Python ModelRouter, while `/llm agent <task>`
uses Pi's coding-agent runtime with isolated `read`, `write`, `edit`, `bash`,
`grep`, `find`, `ls`, and read-only `lsp` tools.
Chat mode keeps the single OpenAI demo route; Agent sessions can explicitly use
an allowlisted OpenAI, Anthropic, or OpenCode Go model. Redis, Kafka, ClickHouse, Flink, RAG, Tavily
web search, response cache, and the provider scheduler are disabled in
[config/config.demo.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.demo.yaml).

Agent monitoring can be enabled separately through the optional
[observability overlay](docker-compose.agent-observability.yaml). It captures
official Pi usage and session data through a SQLite outbox into Kafka/ClickHouse,
and adds OTEL traces, Grafana, and owner feedback. See the
[setup and verification guide](docs/plans/coding-agent-observability.md).

1. In Slack, create an app **from a manifest** and paste
   [slack/app-manifest.demo.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/slack/app-manifest.demo.yaml).
2. Install the app to the workspace. When updating an existing demo app,
   re-apply the manifest and reinstall it so the message shortcut and new
   `lists:read` scope are granted.
3. Under **Basic Information → App-Level Tokens**, generate an app token with
   `connections:write`. Keep the resulting `xapp-` token.
4. Invite the bot to `#ai-testing`. If you use another channel, update
   `slack.channels` in the demo config.
5. Install the pinned Pi runtime dependencies, build the three local images,
   and fetch the pinned official GitHub MCP image once:

```bash
make demo-agent-install
make demo-agent-images
```

The image target verifies the pinned plugin and Skill checksums, builds the hardened Agent
image, and records that exact local image digest in `agent-runtime/plugins.lock.json`.
Review unexpected lock changes before starting the demo. Run the launcher as a
regular host user, not root, so the container can write only to that user's
dedicated worktree and Pi session directory.

6. Create the local secret file:

```bash
cp config/demo.env.example .env.demo
```

Fill in `OPENAI_API_KEY`, `SLACK_BOT_TOKEN`, and `SLACK_APP_TOKEN`. Add
`ANTHROPIC_API_KEY` or `OPENCODE_API_KEY` only when you want those optional
Agent providers. Jev routing is off by default. To collect routing decisions
without changing the selected model, set `OPENROUTER_API_KEY` and
`PI_AGENT_JEV_MODE=shadow`; use `on` only after the evaluation below.
The OpenRouter key stays in the host Agent Runtime and is removed from the
Slack worker environment. Pi 1.0.1 can also use Jev during a run for typed
classification through its built-in `codemode` classifier API. Enable that
separately with:

```bash
PI_AGENT_JEV_CLASSIFIER_MODE=on
PI_AGENT_JEV_CLASSIFIER_MAX_CALLS=8
```

This enables `builtin:codemode` and the pinned
`openrouter/typesafe/jev-1.13` classifier. The Agent container receives a
short-lived token limited to that classifier, the current run, an expiry, and
the configured call budget. The real OpenRouter key stays in the host runtime
and model gateway. Classifier output may guide analysis, classification, and
ranking; write, shell, approval, deployment, and deletion policy continues to
be enforced outside the model. Then start the demo:

```bash
make demo-slack
```

### Optional read-only GitHub MCP

GitHub MCP is off by default. To enable it, create a GitHub App with read access
to **Contents**, **Issues**, and **Pull requests**, install it only on the
repositories Pi may inspect, and set these values in `.env.demo`:

```bash
PI_AGENT_MCP_MODE=github_read_only
PI_AGENT_GITHUB_REPOSITORIES=owner/repository
GITHUB_APP_ID=123456
GITHUB_APP_INSTALLATION_ID=12345678
GITHUB_APP_PRIVATE_KEY_PATH=/absolute/path/to/github-app.private-key.pem
```

The repository list is comma-separated. The current checkout's `origin` must
match one entry. The private key is mounted only into the MCP gateway; it is not
placed in a Docker environment variable or shared with Pi. The gateway mints
short-lived installation tokens and sends them to the pinned official GitHub
MCP server. Pi remains on its internal network and receives a separate token
bound to the run, session, Slack user, repository, mode, server, and exact tool
allowlist.

This release exposes only `get_file_contents`, `search_code`, `issue_read`,
`list_issues`, `pull_request_read`, and `list_pull_requests`. The official server
runs in read-only and lockdown mode, and the local gateway rejects every other
tool or repository. Repository `.pi/mcp.json` files are not loaded. Issue creation,
comments, PR creation or merge, workflow dispatch, and repository settings are
unavailable. Returning `PI_AGENT_MCP_MODE` to `off` removes MCP from new Agent
containers.

`make demo-slack` starts a loopback-only host orchestrator, a model-only gateway,
and the Slack worker. It creates an internal Docker network so Agent containers
cannot reach the general internet. The gateway owns the real provider keys; each
Agent prompt receives only a short-lived token bound to one allowlisted provider,
model, API protocol, run ID, and reasoning setting. All managed processes and
containers are stopped together. No public
webhook or tunnel is required because the app uses Socket Mode.

Try these two paths:

- `/llm explain Transformer` — normal Chat mode through ModelRouter.
- `/llm agent 找出一個測試缺口並修正` — starts a stateful Pi coding agent in
  a dedicated branch, host worktree, and hardened container. Read-only tools run
  automatically. The first edit and non-allowlisted shell commands require an
  owner-only Slack approval.
- From an existing Slack message or thread, open the message actions menu and
  choose **Run Pi Agent**. The modal lets the owner enter a task, choose a cached
  configured Agent model, and include a bounded snapshot of that thread.
- `/llm agent --no-thread-context <task>` explicitly starts without loading
  previous Slack thread messages.
- `/llm agent --model openai/gpt-5.6-sol 找出測試問題` — creates a session fixed
  to Sol at `max` effort. Other explicit choices are `openai/gpt-5.6-luna`,
  `anthropic/claude-sonnet-4-6`, and `opencode-go/deepseek-v4-pro`.
  Explicit-model sessions stay fixed. Automatic sessions can choose Luna or
  Sol again for each follow-up run.
- `/llm agent /skill:test-gap 找出一個測試缺口並修正` — explicitly invokes
  the bundled `test-gap` Skill. Pi may also select it from its description and
  read that exact immutable `SKILL.md` automatically.
- Reply normally inside that bot thread to continue the same Pi session.
- `/llm agent status`, `/llm agent stop`, and `/llm agent close` inspect or
  control your latest session without consuming query quota.

With Jev enabled, only the current task text (up to 4,000 characters) goes to
OpenRouter. High-confidence read-only work uses Luna `low`, a small patch uses
Luna `medium`, and complex work uses Sol `high`. Unclear, low-confidence, or
failed classifications use Luna `max`. The full Slack bootstrap context goes
only to Pi. Run records expose the decision, latency, selected model, and
estimated cost without retaining routing text. Before enabling `on`, collect
at least 50 `shadow` decisions and compare 20 tasks in each of the three
categories against Luna `max` on isolated worktrees. Enable automatic routing
only if eligible tasks finish at least 10% faster at the median with no quality
regression; require the complex-task Sol group to improve separately. Confirm
the OpenAI key can call Sol before that trial.

Each new coding prompt counts as one Slack query; internal model turns and tools
do not. A successful prompt is committed on its `pi-agent/...` branch and Slack
shows the changed files, diff stat, commit, and cherry-pick command. The runtime
never pushes, merges, or modifies the current checkout. Rejected, cancelled,
failed, and timed-out prompts roll their isolated worktree back to the previous
successful commit. Agent failures never silently fall back to Chat mode.

Agent bootstrap context is read-only reference data. The default limits are 20
messages, 10 resources, 4,000 message characters, 8,000 resource characters,
and 12,000 characters total. UTF-8 text/code files and up to 100 current Slack
List rows can be included. Binary files are represented only by skipped-resource
warnings. Canvas uses the title, metadata, and Slack-provided AI summary when
available; the public Canvas API does not provide a reliable full-body export,
so the body is reported as unavailable. Context fetch failures are visible in
Slack; with the demo's `fail_open: true`, the Agent starts task-only instead of
silently guessing what the missing thread contained.

The bundled tools are `read`, `write`, `edit`, `bash`, `grep`, `find`, and `ls`.
The locked `lsp` plugin adds diagnostics, definition, references, hover,
document-symbol, and workspace-symbol queries for TypeScript/JavaScript and
Python. It bundles exact versions of `typescript-language-server` and `pyright`,
rejects paths outside the worktree or protected files, caps responses, and does
not expose rename, code actions, execute-command, or workspace edits.
Extensions, plugins, and Skills are discovery-disabled. Only plugins pinned in
`agent-runtime/plugins.lock.json` and Skills pinned in
`agent-runtime/skills.lock.json` are loaded by explicit container arguments.
The runtime validates each Skill's name, exact version, path, frontmatter, and
SHA-256 checksum before accepting coding runs. The model may read only the exact
approved `SKILL.md` paths outside the worktree; write/edit access remains blocked.
Plugins and Skill instructions remain trusted content and are contained to reduce
host impact, not treated as sandbox boundaries by themselves.

To add another Skill, place its reviewed `SKILL.md` under
`agent-runtime/skills/<name>/`, add an exact-version entry and SHA-256 checksum
to `agent-runtime/skills.lock.json`, then rebuild with `make demo-agent-images`.
There is no Slack or runtime install/update command; a checksum mismatch makes
health unhealthy and blocks new Agent runs.

### Prerequisites

- Python 3.9+
- Node.js 22.19+
- Docker / Docker Compose
- OpenAI and/or Anthropic API key if you want live model responses
- optional GPU and local model server if you want `vLLM`

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Environment

Minimum useful env vars:

```bash
export LLM_ROUTER_API_KEYS=dev-api-key
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export TAVILY_API_KEY=your_tavily_key
export LLM_ROUTER_WEB_SEARCH_ENABLED=true
export SLACK_BOT_TOKEN=xoxb-your-slack-token
export SLACK_APP_TOKEN=xapp-your-slack-app-token
```

## Running Locally

### Option 1: Full Stack With Docker Compose

The root [docker-compose.yml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/docker-compose.yml) starts:

- Redis
- Kafka
- ClickHouse
- API container
- worker container
- Flink JobManager / TaskManager

Start it:

```bash
docker compose up -d --build
```

Useful commands:

```bash
docker compose ps
docker compose logs -f api workers
docker compose config
```

Compose uses [config/config.compose.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.compose.yaml).
Set `LLM_ROUTER_WEB_SEARCH_ENABLED=true` plus `TAVILY_API_KEY` in `.env` to enable
Tavily web search without editing the checked-in config.

After the API is healthy, smoke test the web-search path:

```bash
make smoke-web-search
```

To verify graceful fallback with web search disabled or without a Tavily key, run:

```bash
python scripts/web_search_smoke.py --expect tool-error
```

### Option 2: Host-Run API / Workers

Start infra only:

```bash
docker compose up -d redis kafka clickhouse
```

Start the API:

```bash
python main.py start-api --dev --config config/config.yaml
```

Start background workers:

```bash
python main.py start-workers --config config/config.yaml
```

Run everything in one process:

```bash
python main.py start --config config/config.yaml
```

## Configuration Files

- [config/config.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.yaml): host-run configuration, includes optional local `vLLM`
- [config/config.compose.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.compose.yaml): compose runtime configuration, uses service names like `redis`, `kafka`, `clickhouse`
- [config/config.demo.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.demo.yaml): single-provider Slack demo with external infrastructure disabled

Important defaults in `config/config.yaml`:

- API auth enabled with `X-API-Key`
- Redis cache enabled for inference responses
- Slack state backend defaults to `memory` and can be switched to `file` or `redis`
- pipeline disabled by default for host-run config
- Streamlit enabled in host config, but not started by root compose

## Repository Layout

```text
.
├── main.py
├── docker-compose.yml
├── config/
│   ├── config.yaml
│   └── config.compose.yaml
├── docker/
│   ├── Dockerfile
│   ├── requirements-runtime.txt
│   └── flink/
│       └── Dockerfile
├── flink/
│   └── analytics_job.py
├── slack/
│   ├── bot.py
│   └── bot_real.py
├── src/
│   ├── llm_router_part1_router.py
│   ├── llm_router_part2_inference.py
│   ├── llm_router_part3_pipeline.py
│   ├── llm_router_part3_policy.py
│   ├── llm_router_part4_monitor.py
│   └── utils/
└── tests/
```

## Testing

Run the main regression suite:

```bash
python -m pytest tests/test_router.py tests/test_inference.py tests/test_main.py tests/test_slack_helpers.py tests/test_schema.py tests/test_pipeline.py -q
```

Run all tests:

```bash
python -m pytest tests -q
```

Useful focused suites:

```bash
python -m pytest tests/test_main.py -q
python -m pytest tests/test_slack_helpers.py -q
python -m pytest tests/test_pipeline.py -q
```

`pytest.ini` disables built-in capture because the current local macOS / conda base combination can crash on `readline`.

## Notes And Limitations

- root compose does not start a local `vLLM` model server
- `stream_query` does not currently have the same cloud-to-local fallback path as the main non-streaming route
- monitoring in this repo is API/dashboard-oriented; root compose does not start Grafana or Prometheus UI services
- Slack `redis` backend is the right choice for durability, but true production rollout still needs secret management and runtime ops hardening

## Deployment Notes

`python main.py deploy --output .` can generate deployment artifacts, but the checked-in repo should be treated as the source of truth for local development, not the generated templates.

### RAG on S3 and SQS

The default RAG ingestion path remains `local + redis_stream`. Production can independently migrate source documents to S3 and then ingestion work to SQS:

```yaml
rag:
  enabled: true
  storage:
    backend: s3
    s3:
      bucket: slack-llm-router-production-rag-ACCOUNT_ID
      region: us-west-2
      environment: production
  ingestion_queue:
    enabled: true
    backend: sqs
    sqs:
      queue_url: https://sqs.us-west-2.amazonaws.com/ACCOUNT_ID/slack-llm-router-production-rag
      region: us-west-2
```

Create the AWS resources with the checked-in module under `infra/aws/rag`, and attach its API and worker IAM policy outputs to the corresponding workloads. AWS credentials are resolved only through boto3's standard workload credential chain.

For direct uploads, call `POST /rag/uploads` with the filename, byte length, and base64-encoded SHA-256 digest. Upload using the returned signed PUT request and headers, then call `POST /rag/uploads/{job_id}/complete`. Existing `/rag/documents` requests remain supported as an API-proxied compatibility path.

The live smoke test is intentionally disabled unless it is pointed at an isolated bucket and queue:

```bash
RAG_AWS_SMOKE_CONFIRM=1 \
RAG_AWS_BUCKET=... \
RAG_AWS_QUEUE_URL=... \
AWS_REGION=us-west-2 \
python -m pytest tests/test_rag_aws_live.py -q
```

The reproducible worker-scaling benchmark is documented in
[`docs/rag-s3-sqs-benchmark.md`](docs/rag-s3-sqs-benchmark.md). It drives the
full presign, S3 PUT, complete, SQS, worker, and terminal-poll workflow rather
than treating upload creation as a single endpoint load test.
