#!/bin/bash
#SBATCH --job-name=pasarbench-stage2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00              # ask for LESS -- short jobs schedule sooner
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300          # 5-minute warning before preemption

# Stage 2 as a batch job: one LoRA per fold, then the held-out evaluation.
# On a busy shared queue, assume you WILL be preempted: training checkpoints
# every 50 steps and resumes from the latest, finished adapters are skipped,
# and the evaluation sweep keeps every finished episode.

set -euo pipefail
mkdir -p logs

# Persistent volume, NOT the node's local disk. Node-local data is gone the
# moment you are evicted.
export PASAR_DATA="${PASAR_DATA:-$HOME/scratch/pasarbench}"
mkdir -p "$PASAR_DATA"/{data,checkpoints,traces}
for d in data checkpoints traces; do
  [ -L "$d" ] || { [ -e "$d" ] && echo "!! ./$d exists and is not a link to $PASAR_DATA; move it" && exit 1; }
  ln -sfn "$PASAR_DATA/$d" "$d"
done

module load cuda 2>/dev/null || true
source "${VENV:-$HOME/venvs/pasar}/bin/activate"

_on_term() { echo "preemption signal: the next job resumes from the last checkpoint"; exit 0; }
trap _on_term USR1

echo "node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} job=${SLURM_JOB_ID:-?}"
bash scripts/gpu_pipeline.sh stage2
