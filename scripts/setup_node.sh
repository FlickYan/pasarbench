#!/usr/bin/env bash
# One-time setup on a rented GPU machine with SSH (e.g. Lambda), after
# docs/GPU_GUIDE.md part B. On Modal the image in scripts/modal_pipeline.py
# does the same.
#
#   export PASAR_HOME=/lambda/nfs/pasar          # the persistent filesystem
#   cd $PASAR_HOME/pasarbench && bash scripts/setup_node.sh
#
# Everything that must survive the machine goes on the persistent filesystem:
# the repo (traces/, data/, checkpoints/, logs/), the Python environment and
# the ~90 GB of weights. Then a terminated instance costs nothing to replace:
# launch a new one with the same filesystem and run this again -- it finds the
# environment and the weights already there and only checks them.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PASAR_HOME="${PASAR_HOME:-$HOME}"
VENV="${VENV:-$PASAR_HOME/venv}"
export HF_HOME="${HF_HOME:-$PASAR_HOME/hf}"
# The version this repo was checked against: its /tokenize, its LoRA naming
# (<base>:<adapter>), Qwen3.8's tool parser and every flag in serve_sglang.sh.
# scripts/modal_pipeline.py pins the same one. Change both on purpose.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Two 80 GB cards (one server each) or one B200/H200 (both on it): the plan the
# stages will use, and a warning now if this machine cannot hold both models.
python3 scripts/gpu_plan.py show || true
python3 scripts/gpu_plan.py layout >/dev/null || echo "!! (see above: pick another instance type)"
# SGLang 0.5.20 and the torch it pins are built for CUDA 13, which needs driver
# 580 or newer. Find out now rather than after a 10 GB install.
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
if [ "${driver%%.*}" -lt 580 ]; then
  echo "!! NVIDIA driver $driver: SGLang $SGLANG_VERSION needs CUDA 13, i.e. driver 580 or newer."
  echo "   Terminate this instance and use Modal (driver 580), or an image that lists CUDA 13."
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  # Ubuntu ships venv support as a separate package; install it if missing.
  python3 -m venv "$VENV" 2>/dev/null || {
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv "$VENV"; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip
# SGLang pins torch, transformers and tokenizers that work together. Install it
# first and hold those three where it put them while the training packages go
# in. flash-linear-attention supplies the Gated DeltaNet kernels transformers
# uses to train Qwen3.8; without it training falls back to a slow torch loop.
pip install -q "sglang==$SGLANG_VERSION"
pip freeze | grep -iE '^(torch|transformers|tokenizers|cuda-toolkit)==' > "$VENV/constraints.txt"
# ...and a CUDA compiler of the same version as torch's CUDA: SGLang builds some
# kernels when a server starts, and the machine's own toolkit may be too old.
pip install -q -c "$VENV/constraints.txt" peft accelerate flash-linear-attention \
  "cuda-toolkit[nvcc,cccl]"
python - <<'EOF'
import torch, transformers, peft, sglang
from transformers.utils.import_utils import is_flash_linear_attention_available
assert torch.cuda.is_available(), "torch cannot see a GPU: check the driver and CUDA_VISIBLE_DEVICES"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), {torch.cuda.device_count()} x "
      f"{torch.cuda.get_device_name(0)}; transformers {transformers.__version__}, "
      f"peft {peft.__version__}, sglang {sglang.__version__}, "
      f"fla kernels {'on' if is_flash_linear_attention_available() else 'OFF'}")
EOF

# gpu_pipeline.sh reads this, so the stages find the environment and the
# weights without anything typed into each new shell.
CUDA_HOME="$(python scripts/cuda_home.py --check)"
cat > .pasar_env <<ENV
export PASAR_HOME="$PASAR_HOME"
export HF_HOME="$HF_HOME"
source "$VENV/bin/activate"
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
ENV

bash scripts/download_weights.sh
# The tests run their toy models on the CPU. With GPUs visible, transformers
# would route Qwen3.8's linear-attention layers to the fla GPU kernels even for
# CPU tensors, and the toy test would fail for a reason that says nothing
# about the pipeline.
CUDA_VISIBLE_DEVICES= bash run_tests.sh
echo
echo "Setup done. Next, inside tmux:  bash scripts/gpu_pipeline.sh smoke"
