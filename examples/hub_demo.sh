#!/usr/bin/env bash
# Demo script for the Tree-RAG Memory MCP Hub (`dashboard/server.js`).
# Seeds a temporary directory from `examples/sample_base`, launches the dashboard
# on an available local port, queries key HTTP endpoints with curl, and prints OK.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_BASE="$(mktemp -d "${TMPDIR:-/tmp}/treerag-hub-demo.XXXXXX")"
PORT="${TREE_RAG_DEMO_PORT:-18095}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    rm -rf "${TMP_BASE}"
}
trap cleanup EXIT

cp -R "${REPO_ROOT}/examples/sample_base/"* "${TMP_BASE}/"

node "${REPO_ROOT}/dashboard/server.js" --port "${PORT}" --base-dir "${TMP_BASE}" >/dev/null 2>&1 &
SERVER_PID=$!

for _ in {1..30}; do
    if curl -fsS "http://127.0.0.1:${PORT}/api/memory/status" >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done

STATUS_JSON="$(curl -fsS "http://127.0.0.1:${PORT}/api/memory/status")"
TREE_JSON="$(curl -fsS "http://127.0.0.1:${PORT}/api/memory/tree")"
SEARCH_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/memory/search" -H "Content-Type: application/json" -d '{"query":"rate limiting token bucket redis","scope":"global","top_k":3}')"
MERMAID_JSON="$(curl -fsS "http://127.0.0.1:${PORT}/api/memory/mermaid?workspace=sample_project")"

echo "${STATUS_JSON}" | grep -q '"ok":true'
echo "${TREE_JSON}" | grep -q '"totalWorkspaces":2'
echo "${SEARCH_JSON}" | grep -q '"ok":true'
echo "${MERMAID_JSON}" | grep -q 'flowchart LR'

echo "OK — Tree-RAG Memory MCP Hub demo verified (2 workspaces, status/tree/search/mermaid passed)."
