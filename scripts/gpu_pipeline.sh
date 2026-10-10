#!/usr/bin/env bash
# The post-training phase on one GPU machine, in resumable stages: one B200 or
# H200 (both servers share it), or two 80 GB cards (one server each).
#
#   bash scripts/gpu_pipeline.sh smoke    # 16 episodes; projects the hours of the rest
#   bash scripts/gpu_pipeline.sh stage1   # split, baseline, collection, examples, checks
#     ...read logs/sim_audit.txt, logs/check-A.txt, data/rft/stats.json...
#   bash scripts/gpu_pipeline.sh stage2   # two LoRAs, held-out eval (+ reference if keyed)
#
# Run inside tmux (or a notebook cell, or a Modal job). After a crash, an
# eviction or a dropped connection, run the same stage again: sweeps resume per
# episode, training resumes from its last checkpoint, and finished steps are
# skipped. The stage owns the servers and stops them when it exits, so the GPUs
# are free for the next one.
#
# The human step between the stages is deliberate. The customer is a new model
# and the agent a new family: whether the leak detector can read the agent's
# questions (WHAT_FAILED #26), and what the training examples actually contain,
# are things to read before paying for training, not after.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Written by setup_node.sh: the venv and HF_HOME on the persistent filesystem.
# shellcheck disable=SC1091
if [ -f .pasar_env ]; then set +u; source .pasar_env; set -u; fi   # activate is not set -u safe everywhere
mkdir -p logs
PY="${PYTHON:-python}"
STAGE="${1:-}"
# One agent and one customer for every run: the servers, the sweeps, training
# and the reference row all read these, so overriding them here changes every
# step together.
export AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.8-27B}"
export SIM_MODEL="${SIM_MODEL:-RedHatAI/gemma-4-31B-it-FP8-Dynamic}"
# The knobs that decide the bill (on Modal, export them before `modal run`; the
# job receives them). Stage 1 records the values it ran with in
# data/run_settings.env and later stages read them from there, so a new
# terminal cannot quietly change them: runs compared task by task must use the
# same repetitions. Epochs and the training length cap may still be overridden
# in stage 2 (the second is the way out of a CUDA OOM).
SETTINGS=data/run_settings.env
saved() {
  if [ -f "$SETTINGS" ]; then
    local v; v="$(sed -n "s/^$1=//p" "$SETTINGS" | tail -1)"
    # A stage 1 from before v31 recorded no customer: it had the only one
    # there was, version 1 (WHAT_FAILED #38).
    if [ "$1" = PASAR_CUSTOMER ]; then v="${v:-1}"; fi
    echo "$v"
  fi
}
# A stage 1 from before v31 ran with the old customer and the old tools
# (WHAT_FAILED #37, #38): nothing run now would be compared with like.
if [ -f "$SETTINGS" ] && ! grep -q '^PASAR_CUSTOMER=' "$SETTINGS" && [ "$STAGE" != settings ]; then
  echo "!! stage 1 ran before v31 ($SETTINGS records no customer): its runs had the" >&2
  echo "   simulated customer and the tools from before WHAT_FAILED #37 and #38 were" >&2
  echo "   fixed, and nothing run now would be compared with like. Move data/ and" >&2
  echo "   traces/P-* aside and run stage 1 again." >&2
  exit 1
fi
if [ -f "$SETTINGS" ]; then
  for v in PASAR_K PASAR_COLLECT_K PASAR_CUSTOMER; do
    if [ -n "${!v:-}" ] && [ "${!v}" != "$(saved "$v")" ]; then
      echo "!! $v=${!v}, but stage 1 ran with $v=$(saved "$v") ($SETTINGS)." >&2
      echo "   Runs compared task by task must use the same value: unset $v." >&2
      exit 1
    fi
  done
fi
K="${PASAR_K:-$(saved PASAR_K)}";                         K="${K:-5}"        # repetitions per task: baseline, evaluation, reference
# The simulated customer's rules (WHAT_FAILED #38): every step and the
# reference row get the same one, and train_rft.py reads it from here.
CUSTOMER="${PASAR_CUSTOMER:-$(saved PASAR_CUSTOMER)}";    CUSTOMER="${CUSTOMER:-2}"
export PASAR_CUSTOMER="$CUSTOMER"
COLLECT_K="${PASAR_COLLECT_K:-$(saved PASAR_COLLECT_K)}"; COLLECT_K="${COLLECT_K:-8}"   # rollouts per task to learn from
EPOCHS="${PASAR_EPOCHS:-$(saved PASAR_EPOCHS)}";          EPOCHS="${EPOCHS:-2}"         # LoRA epochs per fold
TRAIN_MAX_LEN="${PASAR_TRAIN_MAX_LEN:-$(saved PASAR_TRAIN_MAX_LEN)}"; TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-16384}"   # longest training turn

# How the servers share this machine (scripts/gpu_plan.py): two or more GPUs,
# one server each; one large GPU (B200, H200), both on it, the agent first.
# Every GPU stage logs the plan and the memory in use to logs/gpu.txt.
NGPU=1
layout() {
  LAYOUT="$("$PY" scripts/gpu_plan.py layout)" || exit 1   # prints why nothing fits
  NGPU="$("$PY" scripts/gpu_plan.py count)"
  if [ "$LAYOUT" = shared ]; then export SHARE_GPU=1 AGENT_GPU=0 SIM_GPU=0; fi
  { echo "== $(date -u +%FT%TZ) $STAGE"; "$PY" scripts/gpu_plan.py show; } | tee -a logs/gpu.txt
  # SGLang builds kernels when a server starts, with the nvcc CUDA_HOME names:
  # it must be 12.9 or newer (scripts/cuda_home.py says why). Found, linked and
  # tried on a test kernel before any model loads.
  CUDA_HOME="$("$PY" scripts/cuda_home.py --check 2> >(tee -a logs/gpu.txt >&2))" || exit 1
  export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
}
gpu_memory() {  # what the servers hold once they are up
  { echo "  memory in use ($1): $(nvidia-smi --query-gpu=index,memory.used,memory.total \
      --format=csv,noheader | tr '\n' ';')"; } 2>/dev/null | tee -a logs/gpu.txt || true
}
capacity() {  # $1 = the agent's log: what each server can hold, as SGLang reports it
  local f line most
  for f in "$1" logs/sim.log; do
    line="$(grep -o 'max_total_num_tokens=[0-9]*.*max_running_requests=[0-9]*' "$f" 2>/dev/null \
            | tail -1 || true)"
    if [ -n "$line" ]; then echo "  $(basename "$f" .log): $line" | tee -a logs/gpu.txt; fi
  done
  most="$(grep -o 'max_running_requests=[0-9]*' "$1" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
  if [ -n "$most" ] && [ "$most" -lt 24 ]; then
    echo "  (the agent runs at most $most conversations at once and the sweeps send 24:"
    echo "   the rest wait their turn -- slower, not wrong)" | tee -a logs/gpu.txt
  fi
}

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

healthy() {  # curl where it exists; the Python every stage needs anyway where not
  if command -v curl >/dev/null 2>&1; then curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1
  else "$PY" -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)" \
         "http://127.0.0.1:$1/health" >/dev/null 2>&1; fi
}

wait_ready() {  # port name log pid
  # SGLang answers /health with 503 until the model is loaded and warmed up.
  for _ in $(seq 1 360); do
    if healthy "$1"; then echo "  $2 ready"; return 0; fi
    if ! kill -0 "$4" 2>/dev/null; then
      echo "!! $2 exited while starting; last lines of $3:"; tail -25 "$3"
      if grep -qi "out of memory" "$3"; then
        if [ -n "${SHARE_GPU:-}" ]; then
          echo "   (out of memory on the shared GPU: keep more of it free, e.g."
          echo "    PASAR_RESERVE_GB=16, and run the stage again; logs/gpu.txt has the plan)"
        else
          echo "   (out of memory: retry with SGLANG_ARGS=\"--mem-fraction-static 0.8\")"
        fi
      fi
      exit 1
    fi
    sleep 5
  done
  echo "!! $2 not ready after 30 minutes; see $3"; exit 1
}

serve_agent() {  # $1 = a serve_sglang.sh mode; the agent alone, on GPU 0
  stop
  $SETSID bash scripts/serve_sglang.sh "$1" > "logs/$1.log" 2>&1 & local a=$!
  echo "$a" > logs/servers.pid
  wait_ready 8000 "$1" "logs/$1.log" "$a"
}

serve() {  # $1 = agent | rft
  stop
  if [ -n "${SHARE_GPU:-}" ]; then
    # One card: the customer's memory is worked out from what the agent left,
    # so it starts only once the agent is up.
    echo "starting $1 and then the customer, both on GPU 0"
    $SETSID bash scripts/serve_sglang.sh "$1" > "logs/$1.log" 2>&1 & local a=$!
    echo "$a" > logs/servers.pid
    wait_ready 8000 "$1" "logs/$1.log" "$a"
    $SETSID bash scripts/serve_sglang.sh sim > logs/sim.log 2>&1 & local s=$!
    echo "$a $s" > logs/servers.pid
    wait_ready 8001 customer logs/sim.log "$s"
  else
    echo "starting $1 on GPU ${AGENT_GPU:-0} and the customer on GPU ${SIM_GPU:-1}"
    $SETSID bash scripts/serve_sglang.sh "$1" > "logs/$1.log" 2>&1 & local a=$!
    $SETSID bash scripts/serve_sglang.sh sim > logs/sim.log 2>&1 & local s=$!
    echo "$a $s" > logs/servers.pid
    wait_ready 8000 "$1" "logs/$1.log" "$a"
    wait_ready 8001 customer logs/sim.log "$s"
  fi
  gpu_memory "$1 + customer"
  capacity "logs/$1.log"
}

case "$STAGE" in
  smoke)
    # A smoke test re-run after a fix must test the fixed setup, not reuse the
    # episodes of the broken one.
    rm -rf traces/P-smoke
    layout
    $PY -m pasarbench.rl.split
    serve agent
    t0=$(date +%s)
    $PY scripts/train_rft.py baseline --sample 1 -k 1 --run-id P-smoke --workers 16 \
      2>&1 | tee logs/smoke.log
    $PY scripts/inspect_trace.py traces/P-smoke/full --leaks-only | tee logs/sim_audit-smoke.txt
    gpu_memory "after the smoke sweep"
    $PY - "$t0" "$K" "$COLLECT_K" "$($PY scripts/gpu_plan.py price)" <<'EOF' | tee logs/smoke-projection.txt
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
k, ck, usd = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
print(f"projected sweeps: stage 1 ~{hours(215 * (k + ck)):.1f} h (baseline 215x{k} + "
      f"collection 215x{ck}), stage 2 evaluation ~{hours(215 * k):.1f} h, plus two "
      f"LoRA runs (logs/train-probe.txt has the time per turn). At ~${usd:.2f}/h for "
      f"this machine, the sweeps alone are ~${usd * hours(215 * (2 * k + ck)):.0f}.")
EOF
    # Stage 2 serves LoRA adapters on this model. Find out now, with a small
    # random adapter built from the config, whether SGLang loads, routes to and
    # applies one -- not after training.
    echo; echo "LoRA serving check on $AGENT_MODEL -> logs/lora-probe.txt"
    CUDA_VISIBLE_DEVICES= $PY scripts/train_rft.py probe-adapter --out checkpoints/probe   # CPU only
    serve_agent probe
    $PY scripts/train_rft.py probe-serve 2>&1 | tee logs/lora-probe.txt
    # And training: a few steps on one turn of the longest length stage 2 will
    # train on, on one card with the servers stopped -- the memory it peaks at
    # and the seconds per turn, before hours of sweeps depend on it.
    echo; echo "training check: one ${TRAIN_MAX_LEN}-token turn -> logs/train-probe.txt"
    stop
    CUDA_VISIBLE_DEVICES=0 $PY scripts/train_rft.py train-probe --max-len "$TRAIN_MAX_LEN" \
      2>&1 | tee logs/train-probe.txt
    ;;

  stage1)
    mkdir -p data
    if [ ! -f "$SETTINGS" ]; then
      printf 'PASAR_K=%s\nPASAR_COLLECT_K=%s\nPASAR_EPOCHS=%s\nPASAR_TRAIN_MAX_LEN=%s\nPASAR_CUSTOMER=%s\n' \
        "$K" "$COLLECT_K" "$EPOCHS" "$TRAIN_MAX_LEN" "$CUSTOMER" > "$SETTINGS"
    fi
    echo "settings ($SETTINGS): $(tr '\n' ' ' < "$SETTINGS")"
    layout
    $PY -m pasarbench.rl.split
    serve agent
    $PY scripts/train_rft.py baseline -k "$K" 2>&1 | tee -a logs/baseline.log
    $PY scripts/inspect_trace.py traces/P-base/full --leaks-only > logs/sim_audit.txt
    $PY scripts/train_rft.py collect -k "$COLLECT_K" 2>&1 | tee -a logs/collect.log
    $PY scripts/train_rft.py build --server http://localhost:8000 2>&1 | tee logs/build.log
    failed=""
    for F in A B; do
      $PY scripts/train_rft.py check --data "data/rft/train-$F.jsonl" \
        --server http://localhost:8000 2>&1 | tee "logs/check-$F.txt" || failed="$failed $F"
    done
    if [ -n "$failed" ]; then
      echo
      echo "!! CHECK FAILED for fold(s)$failed. Running stage 1 again will not change it:"
      echo "   read logs/check-A.txt and logs/check-B.txt (GPU_GUIDE.md, step R) and ask."
      exit 1
    fi
    echo
    echo "Stage 1 done. Before stage 2, read:"
    echo "  logs/sim_audit.txt    leaks and unanswered requests by language (WHAT_FAILED #26)"
    echo "  logs/check-A.txt      what one example trains on, and the server comparison"
    echo "  data/rft/stats.json   per-fold traps with nothing to learn from"
    ;;

  stage2)
    stop
    layout
    # The same recipe on any machine: 32 turns per optimizer step, whether one
    # GPU accumulates all 32 or two accumulate 16 each (the nearest whole number
    # on other counts; the adapter's metadata records the actual figure).
    ACCUM=$(( NGPU > 1 ? (32 + NGPU / 2) / NGPU : 32 )); [ "$ACCUM" -ge 1 ] || ACCUM=1
    if [ "$NGPU" -gt 1 ]; then launch=(accelerate launch --num_processes "$NGPU" --multi_gpu)
    else launch=("$PY"); fi
    for F in A B; do
      if [ -f "checkpoints/rft-$F/adapter_config.json" ]; then
        echo "checkpoints/rft-$F already trained; skipping"; continue
      fi
      "${launch[@]}" scripts/train_rft.py train \
        --data "data/rft/train-$F.jsonl" --out "checkpoints/rft-$F" --resume \
        --epochs "$EPOCHS" --max-len "$TRAIN_MAX_LEN" --grad-accum "$ACCUM" \
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
        --sim-url http://localhost:8001/v1 --sim-extra-body '{}' --suite all -k "$K" --temperature 0 \
        --customer "$CUSTOMER" \
        --strategies full --workers 8 --run-id P-ref --resume > logs/reference.log 2>&1 &
      REF=$!
    else
      echo "(no DEEPSEEK_API_KEY: skipping the reference row; RESULTS.md prints its command)"
    fi
    $PY scripts/train_rft.py eval -k "$K" 2>&1 | tee -a logs/eval.log
    if [ -n "$REF" ]; then wait "$REF" || echo "!! reference run failed; see logs/reference.log"; fi
    echo
    echo "Stage 2 done. Copy back to your laptop and regenerate RESULTS.md there:"
    echo "  traces/P-base traces/P-rft traces/P-ref traces/P-collect data/splits data/rft/stats.json"
    ;;

  settings)
    echo "agent $AGENT_MODEL | customer $SIM_MODEL"
    echo "PASAR_K=$K PASAR_COLLECT_K=$COLLECT_K PASAR_EPOCHS=$EPOCHS PASAR_TRAIN_MAX_LEN=$TRAIN_MAX_LEN PASAR_CUSTOMER=$PASAR_CUSTOMER" \
         "($([ -f "$SETTINGS" ] && echo "stage 1 recorded in $SETTINGS" || echo "stage 1 not run yet"))"
    "$PY" scripts/gpu_plan.py show
    ;;

  *)
    echo "usage: bash scripts/gpu_pipeline.sh smoke | stage1 | stage2 | settings" >&2
    exit 1
    ;;
esac
