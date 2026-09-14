#!/usr/bin/env bash
# Phase 1a smoke test against the DeepSeek API.
#
#   export DEEPSEEK_API_KEY=sk-...
#   ./scripts/smoke_deepseek.sh
#
# Five tasks, one seed. Costs cents. Read the traces before spending more.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY first}"

MODEL="${MODEL:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-https://api.deepseek.com/v1}"

# THIS FLAG IS NOT OPTIONAL.
#
# DeepSeek V4 runs with thinking ENABLED by default and emits hundreds of
# reasoning tokens even for trivial turns. Reasoning counts toward max_tokens,
# so with thinking on the model can burn the whole budget before emitting a
# tool call -- you get empty assistant turns and blame the harness.
#
# It also destroys two measurements this project depends on:
#   * the token axis of the context ablation becomes reasoning verbosity
#   * tokens-per-char in the multilingual diagnosis becomes unreadable
#
# Turn it back on later as a deliberate, separately-labelled arm if you want to
# measure what thinking buys on this workload. That is a real experiment. What
# is not acceptable is running with it on by accident.
THINKING='{"thinking":{"type":"disabled"}}'

# The key is NOT passed as --api-key. The sweep reads DEEPSEEK_API_KEY from the
# environment, so it never appears in argv -- on a shared cluster,
# /proc/<pid>/cmdline is world-readable and `ps aux` would show it to everyone.
echo "model     : $MODEL"
echo "endpoint  : $BASE_URL"
echo "thinking  : disabled (explicit)"
echo "key       : \$DEEPSEEK_API_KEY (${#DEEPSEEK_API_KEY} chars, not shown)"
echo

python -m pasarbench.sweep \
  --backend openai --model "$MODEL" \
  --base-url "$BASE_URL" \
  --extra-body "$THINKING" \
  --simulator openai --sim-model "$MODEL" \
  --suite core --tasks T01,T02,T08,T13,T16 \
  --strategies full -k 1 \
  --max-steps 20 \
  --run-id smoke-deepseek

cat <<'EOF'

=== Read the traces before spending anything more ===

  traces/smoke-deepseek/full/*.jsonl

Four things to check, in order:

1. DID IT CALL TOOLS AT ALL?
     grep -c '"tool_calls": \[{' traces/smoke-deepseek/full/T01.jsonl
   Zero means tool calling is not working -- check the model supports it and
   that `tools` is in the request, before assuming the agent is bad.

2. ARE ASSISTANT TURNS EMPTY?
   Empty content with no tool call usually means max_tokens was exhausted.
   If you see it, thinking is probably still on somewhere, or raise
   --max-tokens.

3. DOES THE SIMULATOR WITHHOLD FACTS?
   The customer must NOT volunteer the order id in its opening turn. If it
   does, every later pass rate describes a single-turn benchmark. Run the
   simqa audit before trusting any number.

4. WHAT IS THE PROMPT CACHE HIT RATE?
   The ~2.2k-token policy prefix is identical on every call, so it should be
   high. A low rate means something is perturbing the prefix between calls.

NOTE ON THE SIMULATOR: this script uses the same model for the agent and the
customer, which is fine for a smoke test and NOT fine for reported numbers. A
simulator sharing the agent's weights is unusually easy for that agent to
satisfy. Before the real runs, point --sim-model and --sim-url at a different
model family (any cheap provider works) and say so in the writeup.
EOF
