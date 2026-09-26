#!/usr/bin/env bash
# Smoke test with the agent and the customer on DIFFERENT providers.
#
#   export DEEPSEEK_API_KEY=sk-...          # agent
#   export PASARBENCH_SIM_API_KEY=sk-...    # simulator (Alibaba Model Studio)
#   ./scripts/smoke_mixed.sh
#
# WHY TWO PROVIDERS
# A simulator sharing the agent's weights is unusually easy for that agent to
# satisfy: it phrases things the way the agent expects, concedes where the agent
# pushes, and inflates the pass rate in a way no number in the output reveals.
# Different family = independent evaluator. Say which model played the customer
# in the writeup.
#
# WHY A CHEAP TIER FOR THE CUSTOMER
# The simulator is roughly half the tokens in every episode and its job is to
# say "It's O1007" while withholding facts until asked. A flagship model is pure
# cost. qwen3.8-max is $2/$6 per Mtok; a flash/plus tier is a fraction of that
# and plays a persona just as well. Check Model Studio's current model list --
# these names churn every few months.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY (the agent)}"
: "${PASARBENCH_SIM_API_KEY:?set PASARBENCH_SIM_API_KEY (the simulator)}"

AGENT_MODEL="${AGENT_MODEL:-deepseek-flash}"
AGENT_URL="${AGENT_URL:-https://api.deepseek.com/v1}"

# Singapore region -- closest to you, and where the international free quota
# lives. Some accounts use a workspace-scoped host instead
# (https://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1);
# check which one your Model Studio console shows.
# qwen3.8-flash, not -max. Singapore list price is $0.15/$0.47 per Mtok against
# max's $2/$6 -- 13x cheaper in, 12.8x cheaper out, with an 89% cache discount.
# The customer's job is to play a persona and withhold facts until asked;
# frontier reasoning buys nothing here and costs more than the agent.
SIM_MODEL="${SIM_MODEL:-qwen3.8-flash}"
SIM_URL="${SIM_URL:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}"

# Both providers default their reasoning ON. Disable it on BOTH sides or your
# token measurements are measuring reasoning verbosity, not context strategy.
AGENT_BODY='{"thinking":{"type":"disabled"}}'
SIM_BODY='{"reasoning_effort":"low"}'

echo "agent     : $AGENT_MODEL  @ $AGENT_URL"
echo "customer  : $SIM_MODEL  @ $SIM_URL"
echo "reasoning : disabled on both sides (explicit)"
echo

python -m pasarbench.sweep \
  --backend openai --model "$AGENT_MODEL" --base-url "$AGENT_URL" \
  --extra-body "$AGENT_BODY" \
  --simulator openai --sim-model "$SIM_MODEL" --sim-url "$SIM_URL" \
  --sim-extra-body "$SIM_BODY" \
  --suite core --tasks T01,T02,T08,T13,T16 \
  --strategies full -k 3 \
  --max-steps 20 --run-id smoke-mixed

echo
echo "k=3 here, not 1 -- your first run passed T08 and the one before failed it."
echo "At k=1 you cannot tell capability from a lucky sample."
echo
echo "  python scripts/inspect_trace.py traces/smoke-mixed/full"
echo
echo "Check the leak rate first. A different-family simulator often follows the"
echo "'reveal only when asked' instruction differently -- verify, do not assume."
