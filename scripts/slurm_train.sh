#!/bin/bash
#SBATCH --job-name=pasarbench-rft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4                 # drop to 1 for collection / serving
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00              # ask for LESS -- short jobs schedule sooner
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:USR1@300          # 5-minute warning before preemption

# On a busy shared queue, assume you WILL be preempted. Everything below is
# built around that: checkpoint often, always resume, keep state off the node.

set -euo pipefail
mkdir -p logs

# Persistent volume, NOT the node's local disk. Node-local data is gone the
# moment you are evicted.
export PASAR_DATA="${PASAR_DATA:-$HOME/scratch/pasarbench}"
mkdir -p "$PASAR_DATA"/{data,checkpoints,traces}
ln -sfn "$PASAR_DATA/data" data
ln -sfn "$PASAR_DATA/checkpoints" checkpoints
ln -sfn "$PASAR_DATA/traces" traces

module load cuda/12.4 2>/dev/null || true
source "${VENV:-$HOME/venvs/pasar}/bin/activate"

# Graceful shutdown: flush and exit cleanly on the preemption signal so the
# next job resumes from a consistent checkpoint rather than a torn one.
_on_term() { echo "preemption signal received; checkpointing and exiting"; wait; exit 0; }
trap _on_term USR1

echo "node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RESUME=""
LATEST=$(ls -dt checkpoints/rft-8b/checkpoint-* 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
  echo "resuming from $LATEST"
  RESUME="--resume_from_checkpoint $LATEST"
fi

accelerate launch \
  --num_processes 4 --mixed_precision bf16 --use_deepspeed \
  --zero_stage 3 --gradient_accumulation_steps 8 \
  scripts/train_rft.py train \
    --model Qwen/Qwen3-8B \
    --data data/rft/sft.jsonl \
    --out checkpoints/rft-8b \
    $RESUME

echo "done. Re-serve the checkpoint and re-run the sweep --"
echo "pass^k on HELD-OUT tasks is the only number that counts. Loss is not a result."
