# Slack Agent LLM Router

Multi-model LLM router with a FastAPI API, Slack bot integration, Kafka/ClickHouse analytics, and local Docker Compose support.

## What It Does

- Routes requests across `gpt-5`, `claude-sonnet-4-6`, and optional local `vLLM` models
- Exposes a protected API for query routing and dashboard access
- Supports Slack bot interactions through `app_mention`, slash commands, and active reply threads
- Persists analytics and request events through Kafka and ClickHouse when the pipeline is enabled
- Supports Redis-backed cache and Slack state for multi-process durability

## Current Routing Behavior

The default host-run config in [config/config.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.yaml) is tuned like this:

- `gpt-5` is the default model
- `claude-sonnet-4-6` gets extra weight for `enterprise` users and difficult `analysis` / `reasoning` requests
- simple general tasks can prefer local `vLLM` models such as `mistral-7b`
- if a cloud model fails on the non-streaming API path, inference can fall back to a local model when one is configured

The compose runtime config in [config/config.compose.yaml](/Users/zhoutingyou/Desktop/Slack%20LLM%20Router/config/config.compose.yaml) is intentionally cloud-only:

- it does not start a local `vLLM` server
- it disables local-model scoring bias so the API does not route to a non-existent local model

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

## Quick Start

### Prerequisites

- Python 3.9+
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
