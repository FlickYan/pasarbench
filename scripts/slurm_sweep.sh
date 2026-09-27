#!/bin/bash
#SBATCH --job-name=pasarbench-stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2                 # agent on one card, the customer on the other
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00              # ask for LESS -- short jobs schedule sooner
#SBATCH --output=logs/%x-%j.out

# The same stages as a rented node, as a batch job. Every stage resumes, so a
# preempted job is re-submitted, not restarted: finished episodes are kept.
#
#   sbatch scripts/slurm_sweep.sh              # stage 1: baseline, collection, examples
#   STAGE=smoke sbatch scripts/slurm_sweep.sh  # 16 episodes, to size the rest

set -euo pipefail
mkdir -p logs
source "${VENV:-$HOME/venvs/pasar}/bin/activate"
bash scripts/gpu_pipeline.sh "${STAGE:-stage1}"
