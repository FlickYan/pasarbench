#!/usr/bin/env bash
# Replace your GitHub repo's contents with this working tree.
#
#   ./scripts/push_to_github.sh git@github.com:you/pasarbench.git
#   ./scripts/push_to_github.sh https://github.com/you/pasarbench.git
#
# Use this when you have pushed an older snapshot and want to replace it. It
# creates ONE clean commit authored by you and force-pushes. Nothing depends on
# the old history, so nothing is lost -- but read the confirmation prompt, the
# remote branch is overwritten.
#
# It never touches your credentials. Authentication is whatever git already
# uses: your ssh-agent for git@ URLs, or your credential helper for https://.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE="${1:-}"
BRANCH="${BRANCH:-main}"
[ -z "$REMOTE" ] && { echo "usage: $0 <git-remote-url>"; exit 1; }

# A LOCAL identity beats a --global one. The shipped .git/config used to carry
# a placeholder, which silently overrode whatever the user had set globally and
# made the guard below fire no matter what they did. Clear it first, then read.
git config --local --unset user.name  2>/dev/null || true
git config --local --unset user.email 2>/dev/null || true

NAME="$(git config --get user.name  || true)"
EMAIL="$(git config --get user.email || true)"
PLACEHOLDERS="you@example.com your@email.com your.real@email.com"
if [ -z "$NAME" ] || [ -z "$EMAIL" ] || [[ " $PLACEHOLDERS " == *" $EMAIL "* ]]; then
  echo "No real git identity found (currently: ${NAME:-unset} <${EMAIL:-unset}>)."
  echo
  echo "Set it -- inside this folder, WITHOUT --global, is simplest:"
  echo "  git config user.name  \"Your Real Name\""
  echo "  git config user.email \"your.real@email.com\""
  echo
  echo "Then check it took:"
  echo "  git config --get user.email"
  exit 1
fi

echo "repo     : $ROOT"
echo "remote   : $REMOTE"
echo "branch   : $BRANCH"
echo "author   : $NAME <$EMAIL>"
echo
echo "This will DISCARD local git history, create one fresh commit, and"
echo "FORCE-PUSH over $BRANCH on the remote."
printf 'Type REPLACE to continue: '
read -r CONFIRM
[ "$CONFIRM" = "REPLACE" ] || { echo "aborted"; exit 1; }

# Never ship run artifacts. .gitignore already covers these; belt and braces
# because a committed traces/ directory is megabytes of noise in a repo whose
# whole point is that it is readable.
rm -rf traces data logs checkpoints
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Sanity gate: do not push a broken tree. This is the cheapest possible
# protection against pushing something that fails on someone else's machine.
echo
echo "running the suite before pushing..."
if ! ./run_tests.sh >/tmp/pasar_push_tests.log 2>&1; then
  echo "TESTS FAILED -- not pushing. See /tmp/pasar_push_tests.log"
  tail -20 /tmp/pasar_push_tests.log
  exit 1
fi
echo "all green"

rm -rf .git
git init -q -b "$BRANCH"
git add -A
git commit -q -m "PasarBench: verifiable SEA e-commerce agent environment

186 tasks across 6 markets and 6 language varieties. Verification on final
database state rather than text, which also makes it a verifiable reward.

- environment: 20 tools, policy doc with real branching (COD, livestream
  disputes, zero-minor-unit currencies), state-based verifier with pass^k
- generator: 3 orthogonal axes (trap x market x language); locale twins share
  a byte-identical checks object, so a language gap is attributable to language
- harness: agent runtime written from scratch -- budgets, structured error
  recovery, interrupt/resume, per-step tracing, 5 pluggable context strategies,
  4 tool-exposure arms with 300 distractors
- rl: verifier-as-reward with tested anti-hacking guarantees
- judge: 9-criterion binary rubric, kappa with prevalence/PABAK and bootstrap
  CIs, human test-retest ceiling, position and length bias probes
- serving: SGLang and vLLM metrics, cost per RESOLVED conversation, non-inferiority guard

9 test suites, ~300 assertions, zero dependencies outside the training track."

git remote add origin "$REMOTE"
git push --force -u origin "$BRANCH"

echo
echo "done. Check the Actions tab -- CI runs all 9 suites on push."
echo "Verify the update landed:"
echo "  grep -c extra-body pasarbench/sweep.py    # expect 5"
