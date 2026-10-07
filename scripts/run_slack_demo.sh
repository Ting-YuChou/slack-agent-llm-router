#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${DEMO_ENV_FILE:-${repo_root}/.env.demo}"
agent_runtime_dir="${repo_root}/agent-runtime"
agent_image="${PI_AGENT_IMAGE:-slack-pi-agent:1.0.1}"
gateway_image="${PI_MODEL_GATEWAY_IMAGE:-slack-pi-model-gateway:0.1.0}"
mcp_gateway_image="${PI_MCP_GATEWAY_IMAGE:-slack-pi-mcp-gateway:0.1.0}"
github_mcp_image="${GITHUB_MCP_IMAGE:-ghcr.io/github/github-mcp-server@sha256:7aaeeec9ae4fe9a736d100c1ff0798f3c219b5009e05f5d3945fcacb13cc196b}"
clickhouse_mcp_image="${PI_CLICKHOUSE_MCP_IMAGE:-slack-pi-clickhouse-mcp:0.7.0}"
agent_network="${PI_AGENT_NETWORK:-pi-model-only}"
mcp_egress_network="${PI_MCP_EGRESS_NETWORK:-pi-mcp-egress}"
gateway_container="${PI_MODEL_GATEWAY_CONTAINER:-slack-pi-model-gateway}"
mcp_gateway_container="${PI_MCP_GATEWAY_CONTAINER:-slack-pi-mcp-github}"
github_mcp_container="${GITHUB_MCP_CONTAINER:-slack-github-mcp}"
clickhouse_mcp_container="${CLICKHOUSE_MCP_CONTAINER:-slack-clickhouse-mcp}"
clickhouse_gateway_container="${CLICKHOUSE_MCP_GATEWAY_CONTAINER:-slack-pi-mcp-clickhouse}"
context7_gateway_container="${CONTEXT7_MCP_GATEWAY_CONTAINER:-slack-pi-mcp-context7}"
agent_pid=""
worker_pid=""
gateway_started=false
network_created=false
mcp_network_created=false
mcp_gateway_started=false
github_mcp_started=false
clickhouse_mcp_started=false
clickhouse_gateway_started=false
context7_gateway_started=false
cleanup_started=false

stop_child() {
  local child_pid="$1"
  if [[ -z "${child_pid}" ]]; then return; fi
  if kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
    for _attempt in {1..50}; do
      if ! kill -0 "${child_pid}" 2>/dev/null; then break; fi
      sleep 0.1
    done
    if kill -0 "${child_pid}" 2>/dev/null; then kill -KILL "${child_pid}" 2>/dev/null || true; fi
  fi
  wait "${child_pid}" 2>/dev/null || true
}

cleanup() {
  if [[ "${cleanup_started}" == true ]]; then return; fi
  cleanup_started=true
  stop_child "${worker_pid}"
  stop_child "${agent_pid}"
  if [[ "${DEMO_TEST_MODE:-0}" != 1 ]]; then
    while IFS= read -r container_id; do
      if [[ -n "${container_id}" ]]; then docker stop "${container_id}" >/dev/null 2>&1 || true; fi
    done < <(docker ps -q --filter label=slack-pi-agent-session=true 2>/dev/null || true)
    if [[ "${gateway_started}" == true ]]; then docker stop "${gateway_container}" >/dev/null 2>&1 || true; fi
    if [[ "${mcp_gateway_started}" == true ]]; then docker stop "${mcp_gateway_container}" >/dev/null 2>&1 || true; fi
    if [[ "${github_mcp_started}" == true ]]; then docker stop "${github_mcp_container}" >/dev/null 2>&1 || true; fi
    if [[ "${clickhouse_gateway_started}" == true ]]; then docker stop "${clickhouse_gateway_container}" >/dev/null 2>&1 || true; fi
    if [[ "${clickhouse_mcp_started}" == true ]]; then docker stop "${clickhouse_mcp_container}" >/dev/null 2>&1 || true; fi
    if [[ "${context7_gateway_started}" == true ]]; then docker stop "${context7_gateway_container}" >/dev/null 2>&1 || true; fi
    if [[ "${network_created}" == true ]]; then docker network rm "${agent_network}" >/dev/null 2>&1 || true; fi
    if [[ "${mcp_network_created}" == true ]]; then docker network rm "${mcp_egress_network}" >/dev/null 2>&1 || true; fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

missing=()
for name in OPENAI_API_KEY SLACK_BOT_TOKEN SLACK_APP_TOKEN; do
  if [[ -z "${!name:-}" ]]; then missing+=("${name}"); fi
done
if (( ${#missing[@]} > 0 )); then
  printf 'Missing required demo environment variables:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf 'Copy config/demo.env.example to .env.demo and fill in the values.\n' >&2
  exit 2
fi

for command_name in node npm curl git; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 2
  fi
done
if [[ "${DEMO_TEST_MODE:-0}" != 1 ]] && ! command -v docker >/dev/null 2>&1; then
  printf 'Missing required command: docker\n' >&2
  exit 2
fi

node_version="$(node --version)"
node_version="${node_version#v}"
IFS=. read -r node_major node_minor _node_patch <<<"${node_version}"
if (( node_major < 22 || (node_major == 22 && node_minor < 19) )); then
  printf 'Pi agent runtime requires Node >=22.19.0; found %s.\n' "${node_version}" >&2
  exit 2
fi
if [[ "${DEMO_TEST_MODE:-0}" != 1 ]] && [[ ! -d "${agent_runtime_dir}/node_modules" ]]; then
  printf 'Agent runtime dependencies are missing. Run: make demo-agent-install\n' >&2
  exit 2
fi

if [[ -z "${AGENT_RUNTIME_TOKEN:-}" ]]; then
  AGENT_RUNTIME_TOKEN="$(node -e 'process.stdout.write(require("node:crypto").randomBytes(32).toString("hex"))')"
fi
if [[ -z "${MODEL_GATEWAY_SIGNING_SECRET:-}" ]]; then
  MODEL_GATEWAY_SIGNING_SECRET="$(node -e 'process.stdout.write(require("node:crypto").randomBytes(32).toString("hex"))')"
fi
export PI_AGENT_MCP_MODE="${PI_AGENT_MCP_MODE:-off}"
mcp_servers="${PI_AGENT_MCP_SERVERS:-}"
if [[ "${PI_AGENT_MCP_MODE}" == github_read_only ]]; then mcp_servers=github; fi
if [[ "${PI_AGENT_MCP_MODE}" != off && "${PI_AGENT_MCP_MODE}" != read_only && "${PI_AGENT_MCP_MODE}" != github_read_only ]]; then
  printf 'PI_AGENT_MCP_MODE must be off, read_only, or github_read_only.\n' >&2
  exit 2
fi
github_mcp_enabled=false; clickhouse_mcp_enabled=false; context7_mcp_enabled=false
if [[ -n "${mcp_servers}" ]]; then
  IFS=, read -ra requested_mcp_servers <<<"${mcp_servers}"
  for requested_server in "${requested_mcp_servers[@]}"; do
    requested_server="${requested_server//[[:space:]]/}"
    case "${requested_server}" in
      github) github_mcp_enabled=true ;;
      clickhouse) clickhouse_mcp_enabled=true ;;
      context7) context7_mcp_enabled=true ;;
      codegraph) ;;
      *) printf 'Unsupported PI_AGENT_MCP_SERVERS entry: %s\n' "${requested_server}" >&2; exit 2 ;;
    esac
  done
fi
if [[ "${PI_AGENT_MCP_MODE}" != off ]] && [[ -z "${MCP_GATEWAY_SIGNING_SECRET:-}" ]] && { [[ "${github_mcp_enabled}" == true ]] || [[ "${clickhouse_mcp_enabled}" == true ]] || [[ "${context7_mcp_enabled}" == true ]]; }; then
  MCP_GATEWAY_SIGNING_SECRET="$(node -e 'process.stdout.write(require("node:crypto").randomBytes(32).toString("hex"))')"
fi
export AGENT_RUNTIME_TOKEN MODEL_GATEWAY_SIGNING_SECRET
if [[ "${PI_AGENT_MCP_MODE}" != off ]]; then export MCP_GATEWAY_SIGNING_SECRET; fi
derive_mcp_secret() {
  MCP_DERIVE_SERVER="$1" node -e 'const c=require("node:crypto"); process.stdout.write(Buffer.from(c.hkdfSync("sha256", process.env.MCP_GATEWAY_SIGNING_SECRET, "slack-pi-mcp-v2", process.env.MCP_DERIVE_SERVER, 32)).toString("base64url"))'
}
github_mcp_signing_secret=""; clickhouse_mcp_signing_secret=""; context7_mcp_signing_secret=""
if [[ "${github_mcp_enabled}" == true ]]; then github_mcp_signing_secret="$(derive_mcp_secret github)"; fi
if [[ "${clickhouse_mcp_enabled}" == true ]]; then clickhouse_mcp_signing_secret="$(derive_mcp_secret clickhouse)"; fi
if [[ "${context7_mcp_enabled}" == true ]]; then context7_mcp_signing_secret="$(derive_mcp_secret context7)"; fi
export PI_AGENT_REPO_PATH="${PI_AGENT_REPO_PATH:-${repo_root}}"
export PI_AGENT_MCP_SERVERS="${mcp_servers}"
export PI_AGENT_MCP_GITHUB_GATEWAY_URL="${PI_AGENT_MCP_GITHUB_GATEWAY_URL:-${PI_AGENT_MCP_GATEWAY_URL:-http://mcp-github:8090/mcp}}"
export PI_AGENT_MCP_CLICKHOUSE_GATEWAY_URL="${PI_AGENT_MCP_CLICKHOUSE_GATEWAY_URL:-http://mcp-clickhouse-gateway:8090/mcp}"
export PI_AGENT_MCP_CONTEXT7_GATEWAY_URL="${PI_AGENT_MCP_CONTEXT7_GATEWAY_URL:-http://mcp-context7:8090/mcp}"
if [[ "${github_mcp_enabled}" == true ]]; then
  for name in PI_AGENT_GITHUB_REPOSITORIES GITHUB_APP_ID GITHUB_APP_INSTALLATION_ID GITHUB_APP_PRIVATE_KEY_PATH; do
    if [[ -z "${!name:-}" ]]; then
      printf 'Missing required GitHub MCP setting: %s\n' "${name}" >&2
      exit 2
    fi
  done
  if [[ ! -f "${GITHUB_APP_PRIVATE_KEY_PATH}" ]]; then
    printf 'GitHub App private key does not exist: %s\n' "${GITHUB_APP_PRIVATE_KEY_PATH}" >&2
    exit 2
  fi
fi
if [[ "${context7_mcp_enabled}" == true && -z "${CONTEXT7_API_KEY:-}" ]]; then printf 'CONTEXT7_API_KEY is required for Context7 MCP.\n' >&2; exit 2; fi
if [[ "${clickhouse_mcp_enabled}" == true ]]; then
  for name in AGENT_MCP_CLICKHOUSE_PASSWORD CLICKHOUSE_MCP_BEARER_TOKEN; do
    if [[ -z "${!name:-}" ]]; then printf 'Missing required ClickHouse MCP setting: %s\n' "${name}" >&2; exit 2; fi
  done
fi
configured_providers="openai"
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then configured_providers="${configured_providers},anthropic"; fi
if [[ -n "${OPENCODE_API_KEY:-}" ]]; then configured_providers="${configured_providers},opencode-go"; fi
export PI_AGENT_CONFIGURED_PROVIDERS="${configured_providers}"

cd "${repo_root}"
npm --prefix "${agent_runtime_dir}" run build

if [[ "${DEMO_TEST_MODE:-0}" != 1 ]]; then
  if ! docker info >/dev/null 2>&1; then
    printf 'Docker is not running or is not accessible.\n' >&2
    exit 2
  fi
  images=("${agent_image}" "${gateway_image}")
  if [[ "${github_mcp_enabled}" == true ]]; then images+=("${mcp_gateway_image}" "${github_mcp_image}"); fi
  if [[ "${clickhouse_mcp_enabled}" == true ]]; then images+=("${mcp_gateway_image}" "${clickhouse_mcp_image}"); fi
  if [[ "${context7_mcp_enabled}" == true ]]; then images+=("${mcp_gateway_image}"); fi
  for image in "${images[@]}"; do
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
      printf 'Missing demo image %s. Run: make demo-agent-images\n' "${image}" >&2
      exit 2
    fi
  done
  export PI_AGENT_IMAGE="${agent_image}"
  PI_AGENT_IMAGE_DIGEST="$(docker image inspect --format='{{.Id}}' "${agent_image}")"
  export PI_AGENT_IMAGE_DIGEST

  if ! docker network inspect "${agent_network}" >/dev/null 2>&1; then
    docker network create --internal "${agent_network}" >/dev/null
    network_created=true
  fi
  docker run --detach --rm \
    --name "${gateway_container}" \
    --label slack-pi-model-gateway=true \
    --network "${agent_network}" \
    --network-alias model-gateway \
    --read-only \
    --user 10002:10002 \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --cpus 1 \
    --memory 512m \
    --pids-limit 128 \
    --tmpfs /tmp:rw,noexec,nosuid,size=67108864 \
    --env OPENAI_API_KEY \
    --env ANTHROPIC_API_KEY \
    --env OPENCODE_API_KEY \
    --env OPENROUTER_API_KEY \
    --env MODEL_GATEWAY_SIGNING_SECRET \
    --env PI_AGENT_OTEL_ENABLED \
    --env "OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318" \
    "${gateway_image}" >/dev/null
  gateway_started=true
  docker network connect bridge "${gateway_container}"
  if [[ "${PI_AGENT_OTEL_ENABLED:-false}" == true ]]; then
    docker network connect "${PI_AGENT_OBSERVABILITY_NETWORK:-slack-agent-observability}" "${gateway_container}"
  fi
  if [[ "${github_mcp_enabled}" == true || "${context7_mcp_enabled}" == true ]]; then
    if ! docker network inspect "${mcp_egress_network}" >/dev/null 2>&1; then
      docker network create "${mcp_egress_network}" >/dev/null
      mcp_network_created=true
    fi
  fi
  if [[ "${github_mcp_enabled}" == true ]]; then
    docker run --detach --rm \
      --name "${github_mcp_container}" \
      --label slack-github-mcp=true \
      --network "${mcp_egress_network}" \
      --network-alias github-mcp \
      --read-only \
      --cap-drop ALL \
      --security-opt no-new-privileges \
      --cpus 1 \
      --memory 512m \
      --pids-limit 128 \
      --tmpfs /tmp:rw,noexec,nosuid,size=67108864 \
      --env GITHUB_READ_ONLY=1 \
      --env GITHUB_LOCKDOWN_MODE=1 \
      --env GITHUB_TOOLSETS=repos,issues,pull_requests,actions,code_security,dependabot,secret_protection \
      --env GITHUB_TOOLS=get_file_contents,search_code,issue_read,pull_request_read,list_issues,list_pull_requests,actions_get,actions_list,get_job_logs,get_code_scanning_alert,list_code_scanning_alerts,get_dependabot_alert,list_dependabot_alerts,get_secret_scanning_alert,list_secret_scanning_alerts \
      "${github_mcp_image}" http --listen-host 0.0.0.0 --port 8082 >/dev/null
    github_mcp_started=true
    docker run --detach --rm \
      --name "${mcp_gateway_container}" \
      --label slack-pi-mcp-gateway=true \
      --network "${agent_network}" \
      --network-alias mcp-github \
      --publish 127.0.0.1:8090:8090 \
      --read-only \
      --user 10003:10003 \
      --cap-drop ALL \
      --security-opt no-new-privileges \
      --cpus 1 \
      --memory 512m \
      --pids-limit 128 \
      --tmpfs /tmp:rw,noexec,nosuid,size=67108864 \
      --mount "type=bind,src=${GITHUB_APP_PRIVATE_KEY_PATH},dst=/run/secrets/github-app.pem,readonly" \
      --env "MCP_GATEWAY_SERVER_SIGNING_SECRET=${github_mcp_signing_secret}" \
      --env GITHUB_APP_ID \
      --env GITHUB_APP_INSTALLATION_ID \
      --env GITHUB_APP_PRIVATE_KEY_PATH=/run/secrets/github-app.pem \
      --env MCP_SERVER_ID=github \
      --env MCP_UPSTREAM_URL=http://github-mcp:8082/mcp \
      "${mcp_gateway_image}" >/dev/null
    mcp_gateway_started=true
    docker network connect "${mcp_egress_network}" "${mcp_gateway_container}"
  fi
  if [[ "${clickhouse_mcp_enabled}" == true ]]; then
    observability_network="${PI_AGENT_OBSERVABILITY_NETWORK:-slack-agent-observability}"
    if ! docker network inspect "${observability_network}" >/dev/null 2>&1; then
      printf 'ClickHouse MCP requires the agent observability network: %s\n' "${observability_network}" >&2
      exit 2
    fi
    docker run --detach --rm \
      --name "${clickhouse_mcp_container}" --label slack-clickhouse-mcp=true \
      --network "${observability_network}" --network-alias mcp-clickhouse \
      --read-only --user 10004:10004 --cap-drop ALL --security-opt no-new-privileges \
      --cpus 1 --memory 256m --pids-limit 128 --tmpfs /tmp:rw,noexec,nosuid,size=67108864 \
      --env CLICKHOUSE_HOST=clickhouse --env CLICKHOUSE_PORT=8123 --env CLICKHOUSE_USER=agent_mcp_reader \
      --env "CLICKHOUSE_PASSWORD=${AGENT_MCP_CLICKHOUSE_PASSWORD}" --env CLICKHOUSE_DATABASE=agent_mcp --env CLICKHOUSE_SECURE=false \
      --env "CLICKHOUSE_MCP_AUTH_TOKEN=${CLICKHOUSE_MCP_BEARER_TOKEN}" --env CLICKHOUSE_MCP_ALLOWED_HOSTS=mcp-clickhouse:8000 \
      "${clickhouse_mcp_image}" >/dev/null
    clickhouse_mcp_started=true
    docker run --detach --rm \
      --name "${clickhouse_gateway_container}" --label slack-pi-mcp-gateway=true \
      --network "${agent_network}" --network-alias mcp-clickhouse-gateway --publish 127.0.0.1:8091:8090 \
      --read-only --user 10003:10003 --cap-drop ALL --security-opt no-new-privileges \
      --cpus 1 --memory 256m --pids-limit 128 --tmpfs /tmp:rw,noexec,nosuid,size=67108864 \
      --env "MCP_GATEWAY_SERVER_SIGNING_SECRET=${clickhouse_mcp_signing_secret}" --env MCP_SERVER_ID=clickhouse \
      --env MCP_UPSTREAM_URL=http://mcp-clickhouse:8000/mcp --env CLICKHOUSE_MCP_BEARER_TOKEN \
      "${mcp_gateway_image}" >/dev/null
    clickhouse_gateway_started=true
    docker network connect "${observability_network}" "${clickhouse_gateway_container}"
  fi
  if [[ "${context7_mcp_enabled}" == true ]]; then
    docker run --detach --rm \
      --name "${context7_gateway_container}" --label slack-pi-mcp-gateway=true \
      --network "${agent_network}" --network-alias mcp-context7 --publish 127.0.0.1:8092:8090 \
      --read-only --user 10003:10003 --cap-drop ALL --security-opt no-new-privileges \
      --cpus 1 --memory 256m --pids-limit 128 --tmpfs /tmp:rw,noexec,nosuid,size=67108864 \
      --env "MCP_GATEWAY_SERVER_SIGNING_SECRET=${context7_mcp_signing_secret}" --env MCP_SERVER_ID=context7 \
      --env MCP_UPSTREAM_URL=https://mcp.context7.com/mcp --env CONTEXT7_API_KEY \
      "${mcp_gateway_image}" >/dev/null
    context7_gateway_started=true
    docker network connect "${mcp_egress_network}" "${context7_gateway_container}"
  fi
fi

env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u OPENCODE_API_KEY \
  -u GITHUB_APP_ID -u GITHUB_APP_INSTALLATION_ID -u GITHUB_APP_PRIVATE_KEY_PATH \
  -u CONTEXT7_API_KEY -u AGENT_MCP_CLICKHOUSE_PASSWORD -u CLICKHOUSE_MCP_BEARER_TOKEN \
  node "${agent_runtime_dir}/dist/src/server.js" &
agent_pid=$!
agent_health_url="${AGENT_RUNTIME_HEALTH_URL:-http://127.0.0.1:3001/health}"
agent_ready=false
for _attempt in {1..50}; do
  health_payload="$(curl -fsS "${agent_health_url}" 2>/dev/null || true)"
  if [[ "${health_payload}" == *'"status":"healthy"'* ]]; then agent_ready=true; break; fi
  if ! kill -0 "${agent_pid}" 2>/dev/null; then
    printf 'Agent runtime exited before becoming healthy.\n' >&2
    exit 1
  fi
  sleep 0.2
done
if [[ "${agent_ready}" != true ]]; then
  printf 'Agent runtime did not become healthy at %s.\n' "${agent_health_url}" >&2
  exit 1
fi

env -u ANTHROPIC_API_KEY -u OPENCODE_API_KEY -u OPENROUTER_API_KEY \
  -u MCP_GATEWAY_SIGNING_SECRET -u GITHUB_APP_ID -u GITHUB_APP_INSTALLATION_ID \
  -u GITHUB_APP_PRIVATE_KEY_PATH -u PI_AGENT_GITHUB_REPOSITORIES \
  -u CONTEXT7_API_KEY -u AGENT_MCP_CLICKHOUSE_PASSWORD -u CLICKHOUSE_MCP_BEARER_TOKEN \
  "${PYTHON:-python}" main.py start-workers --config config/config.demo.yaml &
worker_pid=$!
if wait "${worker_pid}"; then worker_status=0; else worker_status=$?; fi
worker_pid=""
exit "${worker_status}"
