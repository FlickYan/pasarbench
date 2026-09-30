#!/usr/bin/env bash
# Download the agent, the customer and the customer's tokenizer into HF_HOME
# (default ~/.cache/huggingface), about 90 GB. Run once per persistent disk:
# setup_node.sh calls it on Lambda, and `modal run scripts/modal_pipeline.py
# --stage download` on Modal, without a GPU attached.
#
# All three repos are Apache 2.0. HF_TOKEN is optional; set it if a download is
# refused or rate-limited.

set -euo pipefail
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.8-27B}"
SIM_DEFAULT="RedHatAI/gemma-4-31B-it-FP8-Dynamic"
SIM_MODEL="${SIM_MODEL:-$SIM_DEFAULT}"

if command -v hf >/dev/null 2>&1; then DL=(hf download); else DL=(huggingface-cli download); fi
echo "into ${HF_HOME:-$HOME/.cache/huggingface}:"

echo "  agent     $AGENT_MODEL (~56 GB)"
"${DL[@]}" "$AGENT_MODEL" >/dev/null
echo "  customer  $SIM_MODEL (~33 GB)"
"${DL[@]}" "$SIM_MODEL" >/dev/null
if [ "$SIM_MODEL" = "$SIM_DEFAULT" ]; then
  # The FP8 checkpoint was first published before Google's later chat-template
  # fixes, so its tokenizer and template come from Google's repo (no weights).
  echo "  tokenizer google/gemma-4-31B-it (no weights)"
  "${DL[@]}" google/gemma-4-31B-it --exclude "*.safetensors" >/dev/null
fi
du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || true
echo "weights ready"
