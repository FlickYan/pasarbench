#!/usr/bin/env bash
# Replace your GitHub repo's history with ONE commit of this working tree.
#
#   ./scripts/push_to_github.sh git@github.com:you/pasarbench.git
#   ./scripts/push_to_github.sh https://github.com/you/pasarbench.git
#
# Only for when your local and GitHub histories have diverged and you want
# GitHub to match this folder exactly. For an ordinary update -- a new version
# unpacked over your working copy -- commit the changed files and push instead
# (docs/RUNBOOK.md, "Updating an already-pushed repo").
#
# What it will not do: delete anything in this folder, or commit a .env file or
# a run artifact. Before anything else it checks that .gitignore keeps .env,
# traces/, data/ and checkpoints/ out, and it checks the snapshot again before
# committing it. The commit is built in a scratch index, so your branch, index
# and files are untouched until that check has passed, and the old history
# stays in your local reflog (`git reflog main`).
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

[ -d .git ] || git init -q -b "$BRANCH"

# Keys and runs never go to GitHub. .gitignore is the first line of defence: a
# folder without it would commit everything, .env included.
MISSING=()
for p in .env traces/x data/x checkpoints/x; do
  git check-ignore -q --no-index "$p" || MISSING+=("${p%/x}")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "REFUSING: .gitignore does not keep these out of git: ${MISSING[*]}"
  echo "Add each to .gitignore, one per line (directories with a trailing /), and run again."
  exit 1
fi

echo "repo     : $ROOT"
echo "remote   : $REMOTE"
echo "branch   : $BRANCH"
echo "author   : $NAME <$EMAIL>"
echo
echo "This makes ONE commit of this folder, minus what .gitignore excludes, and"
echo "FORCE-PUSHES it over $BRANCH on the remote, replacing the history there."
echo "Nothing in this folder is deleted; the old local history stays in git reflog."
printf 'Type REPLACE to continue: '
read -r CONFIRM
[ "$CONFIRM" = "REPLACE" ] || { echo "aborted"; exit 1; }

# Sanity gate: do not push a broken tree. This is the cheapest possible
# protection against pushing something that fails on someone else's machine.
echo
echo "running the suite before pushing..."
LOG="${TMPDIR:-/tmp}/pasar_push_tests.log"
if ! bash ./run_tests.sh >"$LOG" 2>&1; then
  echo "TESTS FAILED -- not pushing. See $LOG"
  tail -20 "$LOG"
  exit 1
fi
echo "all green"

# The snapshot, built in a scratch index from the working tree as .gitignore
# sees it -- and checked once more, because a .gitignore can be edited between
# the check above and here, and a pattern can miss a name like .env.local.
IDX="$(mktemp "${TMPDIR:-/tmp}/pasar-index.XXXXXX")"
rm -f "$IDX"
trap 'rm -f "$IDX"' EXIT
GIT_INDEX_FILE="$IDX" git add -A
BAD="$(GIT_INDEX_FILE="$IDX" git ls-files | grep -E '(^|/)\.env($|\.)|^(traces|data|checkpoints)/' || true)"
if [ -n "$BAD" ]; then
  echo "REFUSING: the commit would include:"
  echo "$BAD" | head -20 | sed 's/^/  /'
  echo "Nothing was committed or pushed."
  exit 1
fi
FILES="$(GIT_INDEX_FILE="$IDX" git ls-files | wc -l | tr -d ' ')"
TREE="$(GIT_INDEX_FILE="$IDX" git write-tree)"
COMMIT="$(git commit-tree "$TREE" -m "PasarBench: a verifiable customer-service agent environment for SEA e-commerce

Agents are scored on the database they leave behind, not on what they say.
README.md has what was measured; docs/WHAT_FAILED.md what went wrong on the way.")"

git update-ref -m "push_to_github.sh: one commit replaces the history" "refs/heads/$BRANCH" "$COMMIT"
if [ "$(git symbolic-ref -q --short HEAD || true)" = "$BRANCH" ]; then
  git reset -q          # the index follows the new commit; your files are not touched
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi
git push --force -u origin "$BRANCH"

echo
echo "done: $FILES files in one commit, $(git rev-parse --short "$COMMIT")."
echo "GitHub runs run_tests.sh on every push (.github/workflows/tests.yml): see the Actions tab."
echo "The history it replaced is still here: git reflog $BRANCH"
