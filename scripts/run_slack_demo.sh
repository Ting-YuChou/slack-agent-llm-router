#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${DEMO_ENV_FILE:-${repo_root}/.env.demo}"
agent_runtime_dir="${repo_root}/agent-runtime"
agent_image="${PI_AGENT_IMAGE:-slack-pi-agent:0.83.0}"
gateway_image="${PI_MODEL_GATEWAY_IMAGE:-slack-pi-model-gateway:0.1.0}"
agent_network="${PI_AGENT_NETWORK:-pi-model-only}"
gateway_container="${PI_MODEL_GATEWAY_CONTAINER:-slack-pi-model-gateway}"
agent_pid=""
worker_pid=""
gateway_started=false
network_created=false
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
    if [[ "${network_created}" == true ]]; then docker network rm "${agent_network}" >/dev/null 2>&1 || true; fi
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
if [[ ! -d "${agent_runtime_dir}/node_modules" ]]; then
  printf 'Agent runtime dependencies are missing. Run: make demo-agent-install\n' >&2
  exit 2
fi

if [[ -z "${AGENT_RUNTIME_TOKEN:-}" ]]; then
  AGENT_RUNTIME_TOKEN="$(node -e 'process.stdout.write(require("node:crypto").randomBytes(32).toString("hex"))')"
fi
if [[ -z "${MODEL_GATEWAY_SIGNING_SECRET:-}" ]]; then
  MODEL_GATEWAY_SIGNING_SECRET="$(node -e 'process.stdout.write(require("node:crypto").randomBytes(32).toString("hex"))')"
fi
export AGENT_RUNTIME_TOKEN MODEL_GATEWAY_SIGNING_SECRET
export PI_AGENT_REPO_PATH="${PI_AGENT_REPO_PATH:-${repo_root}}"

cd "${repo_root}"
npm --prefix "${agent_runtime_dir}" run build

if [[ "${DEMO_TEST_MODE:-0}" != 1 ]]; then
  if ! docker info >/dev/null 2>&1; then
    printf 'Docker is not running or is not accessible.\n' >&2
    exit 2
  fi
  for image in "${agent_image}" "${gateway_image}"; do
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
    --env MODEL_GATEWAY_SIGNING_SECRET \
    "${gateway_image}" >/dev/null
  gateway_started=true
  docker network connect bridge "${gateway_container}"
fi

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

"${PYTHON:-python}" main.py start-workers --config config/config.demo.yaml &
worker_pid=$!
if wait "${worker_pid}"; then worker_status=0; else worker_status=$?; fi
worker_pid=""
exit "${worker_status}"
