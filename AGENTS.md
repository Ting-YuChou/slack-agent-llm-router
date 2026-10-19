# AGENTS.md

Guidance for coding agents (Pi, Claude Code, Codex) working in this repository.

## Module map

| Path | What lives there |
|---|---|
| `main.py`, `src/llm_router_part*.py` | FastAPI router: routing, inference, pipeline, policy, monitoring, deploy |
| `src/agent_runtime.py`, `src/agent_analytics.py` | Python client for the Pi agent runtime and its analytics |
| `src/rag/`, `src/tools/`, `src/utils/`, `src/models/` | RAG, tool adapters, schemas and shared helpers |
| `slack/bot_real.py`, `slack/agent_*.py` | Slack bot: chat, agent shortcuts, approvals, run status |
| `agent-runtime/src/` | TypeScript Pi coding-agent runtime: `server.ts` (HTTP API), `orchestrator.ts` (run lifecycle), `structured-workflow.ts`, `worktree-manager.ts`, `verification.ts`, `quality-gate.ts`, `model-gateway.ts`, `policy.ts` |
| `agent-runtime/extensions/` | Pi extensions loaded inside the agent container (policy, grader, discussion, gateway) |
| `agent-runtime/test/` | Node test suite (`node:test`, compiled to `dist/test`) |
| `tests/` | Python test suite (pytest) |
| `scripts/` | Smoke, load, contract and live acceptance drivers (`agent_eval.py`, `validate_structured_workflow.py`) |
| `eval/` | Live agent eval tasks, results and baseline |
| `config/` | App config and verification profiles (`config/verification/*.json`) |
| `flink/`, `docker/`, `infra/` | Analytics jobs, images and infrastructure |

## Build, test, lint

```sh
pip install -r requirements-dev.txt
python -m pytest tests -q                         # needs Redis on localhost:6379
python -m pytest tests -q -k "not redis_state_store"   # without Redis
python -m black --check src tests slack
python -m flake8 src tests slack
python -m mypy
npm --prefix agent-runtime ci
npm --prefix agent-runtime run typecheck
npm --prefix agent-runtime test                   # builds, then runs dist/test/**/*.test.js
npm --prefix agent-runtime run lint               # after a build: no-floating-promises and friends
```

Docker-backed agent-runtime tests are opt-in: `PI_AGENT_DOCKER_INTEGRATION=1 npm --prefix agent-runtime test`.
Anything with `--live-openai` (for example `scripts/agent_eval.py`) spends real model budget; do not run it unless asked.

## Do not touch

- `agent-runtime/plugins.lock.json`, `agent-runtime/skills.lock.json`, `agent-runtime/plugins/`, `agent-runtime/skills/`: pinned by digest; regenerate only with `npm --prefix agent-runtime run lock-image`.
- `package-lock.json`, `requirements*.txt` and `*.lock` files, unless the task is a dependency change.
- `config/verification/*.json`, `pytest.ini`, `mypy.ini`, `.flake8`, `agent-runtime/tsconfig*.json`, `.github/workflows/`: these define how your change is judged.
- `eval/tasks/*/oracle/` and `eval/baseline.json` (the baseline changes only through `agent_eval.py --update-baseline`).
- `.env*` files, credentials and `docs/validation/` evidence.

## Rules

- **Never weaken assertions.** Do not delete, skip, loosen or rewrite an existing test or assertion to make it pass. Fix the code. If a test is genuinely wrong, stop and say so.
- Add or update a test with every behaviour change; prefer a failing test first.
- Keep changes inside the task's scope and the profile's `allowedPaths`.
- Match the surrounding style; run the formatters and the relevant test suite before finishing.
