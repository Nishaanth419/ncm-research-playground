#!/usr/bin/env bash
# Start the backend (FastAPI on :8000) and the frontend (Vite on :5173) together.
# Ctrl+C stops both. Usage: ./run.sh   (env vars such as NCM_DEVICE are passed through)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

fail() { echo "error: $*" >&2; exit 1; }

[[ -x "$ROOT/.venv/bin/uvicorn" ]] || fail "no Python env at .venv. Run: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r backend/requirements.txt"
[[ -d "$ROOT/frontend/node_modules" ]] || fail "frontend dependencies missing. Run: (cd frontend && npm install)"
[[ -d "$ROOT/research-ncm" ]] || echo "warning: research-ncm/ not found; NCM models will show as unavailable (see README, Setup)" >&2

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "port $port is already in use (stop the other process or set BACKEND_PORT / FRONTEND_PORT)"
  fi
done

pids=()
# Kill a process and all its descendants (npm starts vite as a child).
kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child"; done
  kill "$1" 2>/dev/null || true
}
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "stopping..."
  for pid in "${pids[@]}"; do kill_tree "$pid"; done
  # Give them a few seconds to shut down cleanly.
  for _ in 1 2 3 4 5; do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    (( alive )) || break
    sleep 1
  done
}
trap cleanup EXIT
trap 'exit 0' INT TERM

# Each subshell execs into the server, so $! is the server itself; output is prefixed per process.
(cd "$ROOT/backend" && exec "$ROOT/.venv/bin/uvicorn" --factory app.main:create_app \
  --host 127.0.0.1 --port "$BACKEND_PORT") > >(sed -u 's/^/[backend]  /') 2>&1 &
pids+=($!); disown $!  # no job-status notices on shutdown

(cd "$ROOT/frontend" && NCM_BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" \
  exec npm run dev -- --port "$FRONTEND_PORT" --strictPort) > >(sed -u 's/^/[frontend] /') 2>&1 &
pids+=($!); disown $!  # no job-status notices on shutdown

echo "backend:  http://127.0.0.1:$BACKEND_PORT  (API docs at /docs)"
echo "frontend: http://localhost:$FRONTEND_PORT"
echo "press Ctrl+C to stop both"

# Exit (and stop the other) as soon as either process dies.
while :; do
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "a process exited; shutting down" >&2; exit 1; }
  done
  sleep 1
done
