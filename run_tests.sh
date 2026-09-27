#!/usr/bin/env bash
# Run every test suite. Works from ANY directory -- it resolves the repo root
# from its own location, so `cd` mistakes cannot produce a ModuleNotFoundError.
#
#   ./run_tests.sh
#   bash run_tests.sh          # if the execute bit was lost in transit
#
# The archive extracts to pasarbench/ which CONTAINS a pasarbench/ package.
# Running from one level too deep is the single most common first-run failure,
# and this script exists to make it impossible.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3 python; do
    command -v "$c" >/dev/null 2>&1 && PY="$c" && break
  done
fi
[ -z "$PY" ] && { echo "no python found; set PYTHON=/path/to/python"; exit 1; }

VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")
echo "repo root : $ROOT"
echo "python    : $PY ($VER)"

"$PY" - <<'EOF' || exit 1
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old; "
             "3.10+ is required (the code uses PEP 604 unions and match-free "
             "modern typing throughout)")
EOF

if [ ! -f "pasarbench/__init__.py" ]; then
  echo
  echo "pasarbench/__init__.py is missing. You are probably not in the repo root,"
  echo "or the archive was extracted incompletely. Expected layout:"
  echo "    <root>/pasarbench/   <root>/tests/   <root>/README.md"
  exit 1
fi

echo "importable: $("$PY" -c 'import pasarbench; print("yes")' 2>&1 | tail -1)"
echo

FAILED=()
run() {
  printf "  %-26s " "$1"
  if OUT=$("$PY" -m "$1" 2>&1); then
    echo "${OUT##*$'\n'}"
  else
    echo "FAILED"
    FAILED+=("$1")
    echo "$OUT" | tail -12 | sed 's/^/      /'
  fi
}

echo "=== suites ==="
run pasarbench.run
for t in traps harness reward generated context judge exposure serving report training; do
  run "tests.test_$t"
done

echo
echo "=== end-to-end sweep (scripted backend, no API calls, no GPU) ==="
if "$PY" -m pasarbench.sweep --backend scripted --suite all --strategies full \
      --run-id smoke >/dev/null 2>&1; then
  echo "  ok"
  rm -rf traces/smoke
else
  echo "  FAILED"
  FAILED+=("sweep")
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL GREEN. Phase 0 complete -- see docs/RUNBOOK.md for Phase 1."
else
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
