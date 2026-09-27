#!/usr/bin/env bash
# The post-training phase on one rented 2x H100 node, in resumable stages.
#
#   bash scripts/gpu_pipeline.sh smoke    # 16 episodes; projects the hours of the rest
#   bash scripts/gpu_pipeline.sh stage1   # split, baseline, collection, examples, checks
#     ...read logs/sim_audit.txt, logs/check-A.txt, data/rft/stats.json...
#   bash scripts/gpu_pipeline.sh stage2   # two LoRAs, held-out eval (+ reference if keyed)
#
# Run inside tmux. After a crash, an eviction or a dropped SSH session, run the
# same stage again: sweeps resume per episode, training resumes from its last
# checkpoint, and finished steps are skipped. The stage owns the servers and
# stops them when it exits, so the GPUs are free for the next one.
#
# The human step between the stages is deliberate. The customer is a new model
# and the agent a new family: whether the leak detector can read the agent's
# questions (WHAT_FAILED #26), and what the training examples actually contain,
# are things to read before paying for training, not after.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY="${PYTHON:-python}"
STAGE="${1:-}"
# One customer for every run: the server, the sweeps and the reference row all
# read this, so overriding it here changes all three together.
export SIM_MODEL="${SIM_MODEL:-RedHatAI/gemma-4-31B-it-FP8-Dynamic}"

# Each server runs in its own process group (setsid), because SGLang runs its
# scheduler and detokenizer as child processes: killing only the parent can
# leave a child holding a GPU's memory, and the next stage then runs out of it.
stop() {
  if [ -f logs/servers.pid ]; then
    for p in $(cat logs/servers.pid); do kill -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done
    for p in $(cat logs/servers.pid); do
      for _ in $(seq 1 60); do kill -0 -- "-$p" 2>/dev/null || break; sleep 2; done
      kill -9 -- "-$p" 2>/dev/null || true
    done
    rm -f logs/servers.pid
    sleep 5                          # let the driver reclaim the memory
  fi
}
trap stop EXIT
SETSID=""
command -v setsid >/dev/null 2>&1 && SETSID=setsid

wait_ready() {  # port name log pid
  # SGLang answers /health with 503 until the model is loaded and warmed up.
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1; then echo "  $2 ready"; return 0; fi
    if ! kill -0 "$4" 2>/dev/null; then
      echo "!! $2 exited while starting; last lines of $3:"; tail -25 "$3"
      if grep -qi "out of memory" "$3"; then
        echo "   (out of memory: retry with SGLANG_ARGS=\"--mem-fraction-static 0.8\")"
      fi
      exit 1
    fi
    sleep 5
  done
  echo "!! $2 not ready after 30 minutes; see $3"; exit 1
}

serve() {  # $1 = agent | rft
  stop
  echo "starting $1 on GPU ${AGENT_GPU:-0} and the customer on GPU ${SIM_GPU:-1}"
  $SETSID bash scripts/serve_sglang.sh "$1" > "logs/$1.log" 2>&1 & local a=$!
  $SETSID bash scripts/serve_sglang.sh sim > logs/sim.log 2>&1 & local s=$!
  echo "$a $s" > logs/servers.pid
  wait_ready 8000 "$1" "logs/$1.log" "$a"
  wait_ready 8001 customer logs/sim.log "$s"
}

case "$STAGE" in
  smoke)
    $PY -m pasarbench.rl.split
    serve agent
    t0=$(date +%s)
    $PY scripts/train_rft.py baseline --sample 1 -k 1 --run-id P-smoke --workers 16 \
      2>&1 | tee logs/smoke.log
    $PY scripts/inspect_trace.py traces/P-smoke/full --leaks-only | tee logs/sim_audit-smoke.txt
    $PY - "$t0" <<'EOF'
import json, sys, time
from pathlib import Path
eps = [[json.loads(l) for l in p.read_text().splitlines()]
       for p in Path("traces/P-smoke/full").glob("*.jsonl")]
wall = time.time() - int(sys.argv[1])
foot = [e[-1] for e in eps if e and e[-1].get("type") == "footer"]
per = sum(f.get("wall_s", 0) for f in foot) / max(len(foot), 1)
toks = sum((f.get("budget") or {}).get("tokens", 0) for f in foot) / max(len(foot), 1)
print(f"\n{len(foot)} episodes in {wall / 60:.1f} min; "
      f"{sum(bool(f.get('passed')) for f in foot)} passed; "
      f"{per:.0f} s and {toks:,.0f} tokens per episode")
# The real stages run 24 episodes at once. SGLang batches them, so each gets
# a little slower: read this as an order of magnitude, not a quote.
hours = lambda n: n * per / 24 / 3600
print(f"projected sweeps: stage 1 ~{hours(215 * 13):.1f} h (baseline 215x5 + "
      f"collection 215x8), stage 2 evaluation ~{hours(215 * 5):.1f} h, plus two "
      f"LoRA runs (the trainer prints its own ETA)")
EOF
    ;;

  stage1)
    $PY -m pasarbench.rl.split
    serve agent
    $PY scripts/train_rft.py baseline 2>&1 | tee -a logs/baseline.log
    $PY scripts/inspect_trace.py traces/P-base/full --leaks-only > logs/sim_audit.txt
    $PY scripts/train_rft.py collect 2>&1 | tee -a logs/collect.log
    $PY scripts/train_rft.py build --server http://localhost:8000 2>&1 | tee logs/build.log
    for F in A B; do
      $PY scripts/train_rft.py check --data "data/rft/train-$F.jsonl" \
        --server http://localhost:8000 2>&1 | tee "logs/check-$F.txt"
    done
    echo
    echo "Stage 1 done. Before stage 2, read:"
    echo "  logs/sim_audit.txt    leaks and unanswered requests by language (WHAT_FAILED #26)"
    echo "  logs/check-A.txt      what one example trains on, and the server comparison"
    echo "  data/rft/stats.json   per-fold traps with nothing to learn from"
    ;;

  stage2)
    stop
    for F in A B; do
      if [ -f "checkpoints/rft-$F/adapter_config.json" ]; then
        echo "checkpoints/rft-$F already trained; skipping"; continue
      fi
      accelerate launch --num_processes 2 --multi_gpu scripts/train_rft.py train \
        --data "data/rft/train-$F.jsonl" --out "checkpoints/rft-$F" --resume \
        2>&1 | tee -a "logs/train-$F.log"
    done
    serve rft
    REF=""
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
      # Same customer, API agent: the reference row. It shares the customer's
      # GPU with the evaluation, so run them together.
      echo "reference run (deepseek-v4-pro through its API) in the background -> logs/reference.log"
      $PY -m pasarbench.sweep --backend openai --model deepseek-v4-pro \
        --base-url https://api.deepseek.com/v1 --extra-body '{"thinking":{"type":"disabled"}}' \
        --simulator openai --sim-model "$SIM_MODEL" \
        --sim-url http://localhost:8001/v1 --sim-extra-body '{}' --suite all -k 5 --temperature 0 \
        --strategies full --workers 8 --run-id P-ref --resume > logs/reference.log 2>&1 &
      REF=$!
    else
      echo "(no DEEPSEEK_API_KEY: skipping the reference row; RESULTS.md prints its command)"
    fi
    $PY scripts/train_rft.py eval 2>&1 | tee -a logs/eval.log
    if [ -n "$REF" ]; then wait "$REF" || echo "!! reference run failed; see logs/reference.log"; fi
    echo
    echo "Stage 2 done. Copy back to your laptop and regenerate RESULTS.md there:"
    echo "  traces/P-base traces/P-rft traces/P-ref traces/P-collect data/splits data/rft/stats.json"
    ;;

  *)
    echo "usage: bash scripts/gpu_pipeline.sh smoke | stage1 | stage2" >&2
    exit 1
    ;;
esac
