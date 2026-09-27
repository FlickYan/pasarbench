#!/usr/bin/env bash
# One-time setup on a freshly rented GPU node (Linux, NVIDIA driver installed).
#
#   git clone <your repo> pasarbench && cd pasarbench   (or scp the tarball)
#   bash scripts/setup_node.sh
#
# Keep the repo -- traces/, data/, checkpoints/ -- on the provider's PERSISTENT
# volume, not the instance disk: an evicted or stopped instance takes its disk
# with it, and every stage below is built to resume from what is on disk. Point
# HF_HOME at the volume too, or every restart downloads ~50 GB of weights again.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$HOME/venvs/pasar}"
# The version this repo was checked against: its /tokenize, its LoRA naming
# (<base>:<adapter>) and every flag in serve_sglang.sh. Change it on purpose.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ "$(nvidia-smi -L | wc -l)" -ge 2 ] || echo "!! fewer than 2 GPUs: agent and customer need one each"
# SGLang 0.5.20 and the torch it pins are built for CUDA 13, which needs driver
# 580 or newer. Find out now rather than after a 10 GB install.
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
if [ "${driver%%.*}" -lt 580 ]; then
  echo "!! NVIDIA driver $driver: SGLang $SGLANG_VERSION needs CUDA 13, i.e. driver 580 or newer."
  echo "   Choose a machine image with a newer driver (providers list them as CUDA 13)."
  exit 1
fi

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip
# SGLang pins torch, transformers and tokenizers that work together. Install it
# first and hold those three where it put them while peft and accelerate go in.
pip install "sglang==$SGLANG_VERSION"
pip freeze | grep -iE '^(torch|transformers|tokenizers)==' > "$VENV/constraints.txt"
pip install -c "$VENV/constraints.txt" peft accelerate
python - <<'EOF'
import torch, transformers, peft, sglang
assert torch.cuda.is_available(), "torch cannot see a GPU: check the driver and CUDA_VISIBLE_DEVICES"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), {torch.cuda.device_count()} x "
      f"{torch.cuda.get_device_name(0)}; transformers {transformers.__version__}, "
      f"peft {peft.__version__}, sglang {sglang.__version__}")
EOF

# Weights now, so the first server start is not a download. All three repos are
# Apache 2.0; HF_TOKEN is optional (set it if a download asks for one). The
# customer's weights are RedHat's FP8 checkpoint, first published before
# Google's later chat-template fixes, so its tokenizer and template come from
# Google's repo (see serve_sglang.sh).
if command -v hf >/dev/null 2>&1; then DL=(hf download); else DL=(huggingface-cli download); fi
"${DL[@]}" "${AGENT_MODEL:-Qwen/Qwen3-8B}" >/dev/null
"${DL[@]}" "${SIM_MODEL:-RedHatAI/gemma-4-31B-it-FP8-Dynamic}" >/dev/null
"${DL[@]}" google/gemma-4-31B-it --exclude "*.safetensors" >/dev/null
echo "weights downloaded"

bash run_tests.sh
echo
echo "Next, in tmux:  bash scripts/gpu_pipeline.sh smoke"
