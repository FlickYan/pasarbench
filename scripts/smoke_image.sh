#!/usr/bin/env bash
# The service's image, run the way docs/SERVICE.md tells a user to run it:
#
#   scripts/smoke_image.sh pasarbench-service:ci
#
# In a volume of its own: migrate a SQLite database, make a tenant and its key
# (the image's CLI), start the server and wait for the image's own HEALTHCHECK
# to call it healthy; check it runs as a user other than root and holds no
# .env; then scripts/smoke_api.py over real HTTP (a turn, its retry, a stream,
# the audit log); `replay` of every tool call; and a clean stop on SIGTERM. CI
# runs it on every push, and on the image it publishes, before it publishes it.
# What it makes is named for the run and removed at the end, pass or fail.
set -euo pipefail

IMAGE=${1:?usage: scripts/smoke_image.sh IMAGE}
RUN=pasar-smoke-$$
PORT=${PASAR_SMOKE_PORT:-18080}
HERE=$(cd "$(dirname "$0")" && pwd)

cleanup() {
  local status=$?
  if [ "$status" -ne 0 ] && docker container inspect "$RUN" >/dev/null 2>&1; then
    echo "--- the server's log, last 60 lines"
    docker logs --tail 60 "$RUN" 2>&1 || true
  fi
  docker rm -f "$RUN" >/dev/null 2>&1 || true
  docker volume rm "$RUN" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

fail() { echo "FAIL  $*"; exit 1; }
cli() { docker run --rm -v "$RUN:/app/data" "$IMAGE" "$@"; }

echo "--- $IMAGE"
cli migrate
made=$(cli create-tenant smoke --name "Smoke Test" --world demo)
KEY=$(printf '%s\n' "$made" | grep -m1 -E '^psk_[A-Za-z0-9_-]+$' || true)
[ -n "$KEY" ] || fail "create-tenant printed no key: $made"
[ -z "${GITHUB_ACTIONS:-}" ] || echo "::add-mask::$KEY"
echo "ok    migrated; a tenant and its key, from the image's CLI"

docker run -d --name "$RUN" -v "$RUN:/app/data" -p "127.0.0.1:$PORT:8000" "$IMAGE" >/dev/null
health=starting
for _ in $(seq 90); do
  health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$RUN")
  [ "$health" = healthy ] && break
  [ "$(docker inspect -f '{{.State.Running}}' "$RUN")" = true ] || fail "the server exited"
  [ "$health" = unhealthy ] && fail "its HEALTHCHECK calls it unhealthy"
  [ "$health" = none ] && fail "the image has no HEALTHCHECK"
  sleep 1
done
[ "$health" = healthy ] || fail "not healthy after 90 seconds: $health"
echo "ok    serving; the image's HEALTHCHECK calls it healthy"

uid=$(docker exec "$RUN" python -c 'import os; print(os.getuid())')
[ "$uid" != 0 ] || fail "the server runs as root"
found=$(docker exec "$RUN" python -c 'import pathlib
print(*sorted(str(p) for p in pathlib.Path("/app").rglob("*")
              if p.name.startswith(".env") or p.name.endswith(".env")))')
[ -z "$found" ] || fail "the image holds $found"
echo "ok    the server runs as uid $uid, not root; no .env in the image"

python3 "$HERE/smoke_api.py" "http://127.0.0.1:$PORT" "$KEY"

report=$(cli replay smoke) || fail "replay found a difference: $report"
printf '%s' "$report" | python3 -c 'import json, sys
r = json.load(sys.stdin)
sys.exit(0 if r["ok"] and r["calls"] > 0 and r["identical"] == r["calls"] else 1)' \
  || fail "replay: $report"
echo "ok    replay: every tool call re-run, identical, and the world with them"

# SIGTERM, as docker stop and Kubernetes send it: uvicorn drains and stops,
# then re-raises the signal it caught -- exit 143, or 0 where the kernel will
# not kill a container's PID 1 with a signal it has stopped handling. 137 is a
# server docker stop had to kill after 20 seconds.
docker stop -t 20 "$RUN" >/dev/null
code=$(docker inspect -f '{{.State.ExitCode}}' "$RUN")
case "$code" in
  0|143) echo "ok    stops by itself on SIGTERM (exit $code)" ;;
  *) fail "on SIGTERM it ended with exit code $code" ;;
esac
echo "smoke_image: all checks passed"
