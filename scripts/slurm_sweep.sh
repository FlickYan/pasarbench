#!/bin/bash
#SBATCH --job-name=pasarbench-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # one GPU is enough to serve an 8B
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Serve a model and sweep against it in one job. A 4-hour single-GPU job
# schedules far sooner than a 24-hour 4-GPU one, and this is all Phase 3 needs.

set -euo pipefail
mkdir -p logs
source "${VENV:-$HOME/venvs/pasar}/bin/activate"

PORT=${PORT:-8000}
vllm serve Qwen/Qwen3-8B --port $PORT --api-key EMPTY \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 32768 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# Wait for readiness rather than sleeping a fixed amount -- an 8B cold start is
# 2-4 minutes and varies with filesystem contention on a shared cluster.
for i in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/health" >/dev/null && break
  sleep 5
done || { echo "vllm did not become ready"; exit 1; }

curl -s "http://localhost:$PORT/metrics" > metrics_before.txt

python -m pasarbench.sweep \
  --backend openai --model Qwen/Qwen3-8B \
  --base-url "http://localhost:$PORT/v1" --api-key EMPTY \
  --simulator openai --sim-model Qwen/Qwen3-8B \
  --suite all --sample 2 --strategies full,trim3 -k 3 \
  --run-id "slurm-$SLURM_JOB_ID"

curl -s "http://localhost:$PORT/metrics" > metrics_after.txt
echo "diff the counters, do NOT read the gauges:"
echo "  MetricsSnapshot.from_text(open('metrics_before.txt').read()).diff(...)"
