#!/usr/bin/env bash
# The post-training phase on one GPU machine, in resumable stages: one B200 or
# H200 (both servers share it), or two 80 GB cards (one server each).
#
#   bash scripts/gpu_pipeline.sh smoke    # 16 episodes; projects the hours of the rest
#   bash scripts/gpu_pipeline.sh stage1   # split, baseline, collection, examples, checks
#     ...read logs/sim_audit.txt, logs/check-A.txt, data/rft/stats.json...
#   bash scripts/gpu_pipeline.sh stage2   # two LoRAs, held-out eval (+ reference if keyed)
#   PASAR_ROUND=2 bash scripts/gpu_pipeline.sh grpo   # GRPO from round 2's adapters (v45)
#   PASAR_ROUND=2 bash scripts/gpu_pipeline.sh customer3   # customer 3 against 2 (v47)
#   PASAR_ROUND=2 bash scripts/gpu_pipeline.sh edition3    # the hard edition (v48)
#   PASAR_ROUND=2 bash scripts/gpu_pipeline.sh grpo-measure   # GRPO on its own tasks (v52)
#   bash scripts/gpu_pipeline.sh serving  # phase 3: the agent served three ways (v68)
#
# PASAR_ROUND=2 runs the same stages again as round 2 (v44): the second
# fine-tune, its data balanced by trap (WHAT_FAILED #31), under today's customer
# and tools, with the base model also run on the 62 newer tasks of the five
# hardest traps. Its runs, files and adapters have names of their own (P2-*,
# data/rft2, checkpoints/rft2-*, logs/r2-*, data/run2_settings.env), so round
# 1's stay as they are, here and on the laptop (RUNBOOK phase 4, round 2).
#
# `grpo` (round 2 only) continues each fold's adapter with GRPO on that fold's
# tasks, then evaluates the result on what no model trained on: the held-out
# fold and the 62 newer tasks (RUNBOOK phase 4, GRPO).
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
ROUND="${PASAR_ROUND:-1}"
case "$ROUND" in
  1) P=P;  RFT=rft;  LP="";    SETTINGS=data/run_settings.env ;;
  2) P=P2; RFT=rft2; LP="r2-"; SETTINGS=data/run2_settings.env ;;
  *) echo "!! PASAR_ROUND is 1 or 2, not $ROUND" >&2; exit 1 ;;
esac
# The adapters serve_sglang.sh's rft mode loads: checkpoints/$RFT-A and -B.
export RFT_NAME="$RFT"
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
# in stage 2 (the second is the way out of a CUDA OOM). Each round has its own
# file ($SETTINGS, set above).
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
  echo "   traces/P-* aside and run stage 1 again -- or keep them, and run round 2," >&2
  echo "   which has files of its own: PASAR_ROUND=2 (docs/GPU_GUIDE.md, round 2)." >&2
  exit 1
fi
if [ -f "$SETTINGS" ]; then
  for v in PASAR_K PASAR_COLLECT_K PASAR_CUSTOMER PASAR_PER_TRAP; do
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
# Training episodes per trap and fold, balanced across the trap's tasks; 0 is
# round 1's recipe, at most three a task (WHAT_FAILED #31).
PER_TRAP="${PASAR_PER_TRAP:-$(saved PASAR_PER_TRAP)}"
if [ -z "$PER_TRAP" ]; then if [ "$ROUND" = 1 ]; then PER_TRAP=0; else PER_TRAP=20; fi; fi
# GRPO (the grpo stage): iterations a fold -- run the stage again with more to
# continue from the last one -- rollouts a task, and the learning rate. The
# first grpo run records its rollouts a task with the round's settings, and
# every later iteration uses the same.
GRPO_ITERS="${PASAR_GRPO_ITERS:-1}"
GRPO_K="${PASAR_GRPO_K:-$(saved PASAR_GRPO_K)}";          GRPO_K="${GRPO_K:-8}"
GRPO_LR="${PASAR_GRPO_LR:-2e-5}"
# Traps GRPO never trains on (train_grpo.py --skip-traps; `none` for none): by
# default the photo trap, whose verdicts follow the customer (WHAT_FAILED #43).
GRPO_SKIP="${PASAR_GRPO_SKIP_TRAPS:-$(saved PASAR_GRPO_SKIP_TRAPS)}"
GRPO_SKIP="${GRPO_SKIP:-high_value_photo_required_first}"
# How GRPO's reward reads the voucher trap (train_grpo.py build --reward-reading;
# WHAT_FAILED #46): "checks", as the check scores it, or "policy", where an
# escalation after she declined the voucher is the pass P6.3 and P10 make it.
# Held for every iteration after the first; round 2's first was built by the
# checks, before the reading could be chosen.
GRPO_READING="${PASAR_GRPO_READING:-$(saved PASAR_GRPO_READING)}"
GRPO_READING="${GRPO_READING:-checks}"
# The hard edition's customer (the edition3 stage): customer 3, whom the next
# editions run with, once the customer3 stage has checked her; 2 if that check
# says to keep customer 2 (RUNBOOK 1s). Recorded at its first run, like GRPO's.
E3_CUSTOMER="${PASAR_E3_CUSTOMER:-$(saved PASAR_E3_CUSTOMER)}"; E3_CUSTOMER="${E3_CUSTOMER:-3}"

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

finished() {  # $1 = a run cell: how many of its episodes ran to the end
  "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
n = 0
for f in Path(sys.argv[1]).glob("*.jsonl"):
    try:
        last = json.loads([l for l in f.read_text(encoding="utf-8").splitlines()
                           if l.strip()][-1])
    except (IndexError, ValueError):
        continue                     # empty, or cut off mid-line
    # A server error ends an episode with a footer too, and the sweep runs it
    # again on resume: not finished.
    n += last.get("type") == "footer" and last.get("stop_reason") != "backend_error"
print(n)
EOF
}

grpo_reading() {  # GRPO's reading of the voucher trap, one for every iteration of a run
  case "$GRPO_READING" in
    checks|policy) ;;
    *) echo "!! PASAR_GRPO_READING is checks or policy, not $GRPO_READING" >&2; return 1 ;;
  esac
  if [ -z "$(saved PASAR_GRPO_READING)" ] && [ -f data/grpo2/stats/i1-A.stats.json ] \
      && ! grep -q '"reward_reading"' data/grpo2/stats/i1-A.stats.json; then
    # Built before the reading could be chosen: by the checks.
    printf 'PASAR_GRPO_READING=checks\n' >> "$SETTINGS"
  fi
  if [ -n "$(saved PASAR_GRPO_READING)" ] && [ "$GRPO_READING" != "$(saved PASAR_GRPO_READING)" ]; then
    echo "!! PASAR_GRPO_READING=$GRPO_READING, but GRPO's first iteration was rewarded by" \
         "$(saved PASAR_GRPO_READING) ($SETTINGS): one reading a run (WHAT_FAILED #46)." >&2
    return 1
  fi
  if [ -z "$(saved PASAR_GRPO_READING)" ]; then
    printf 'PASAR_GRPO_READING=%s\n' "$GRPO_READING" >> "$SETTINGS"
  fi
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
      printf 'PASAR_K=%s\nPASAR_COLLECT_K=%s\nPASAR_EPOCHS=%s\nPASAR_TRAIN_MAX_LEN=%s\nPASAR_CUSTOMER=%s\nPASAR_PER_TRAP=%s\n' \
        "$K" "$COLLECT_K" "$EPOCHS" "$TRAIN_MAX_LEN" "$CUSTOMER" "$PER_TRAP" > "$SETTINGS"
    fi
    echo "round $ROUND, settings ($SETTINGS): $(tr '\n' ' ' < "$SETTINGS")"
    layout
    $PY -m pasarbench.rl.split
    serve agent
    $PY scripts/train_rft.py baseline -k "$K" --run-id "$P-base" 2>&1 | tee -a "logs/${LP}baseline.log"
    if [ "$ROUND" != 1 ]; then
      # The 62 newer tasks of the five hardest traps, outside the folds: where
      # the base model fails some of the time is where GRPO can learn.
      $PY scripts/train_rft.py baseline -k "$K" --suite edition2 --run-id "$P-base-e2" \
        2>&1 | tee -a "logs/${LP}baseline.log"
    fi
    $PY scripts/inspect_trace.py "traces/$P-base/full" --leaks-only > "logs/${LP}sim_audit.txt"
    $PY scripts/train_rft.py collect -k "$COLLECT_K" --run-id "$P-collect" 2>&1 | tee -a "logs/${LP}collect.log"
    build=(--server http://localhost:8000 --run "traces/$P-collect/full" --out-dir "data/$RFT")
    if [ "$PER_TRAP" -gt 0 ]; then build+=(--per-trap "$PER_TRAP" --max-per-task "$COLLECT_K"); fi
    $PY scripts/train_rft.py build "${build[@]}" 2>&1 | tee "logs/${LP}build.log"
    failed=""
    for F in A B; do
      $PY scripts/train_rft.py check --data "data/$RFT/train-$F.jsonl" \
        --server http://localhost:8000 2>&1 | tee "logs/${LP}check-$F.txt" || failed="$failed $F"
    done
    if [ -n "$failed" ]; then
      echo
      echo "!! CHECK FAILED for fold(s)$failed. Running stage 1 again will not change it:"
      echo "   read logs/${LP}check-A.txt and logs/${LP}check-B.txt (GPU_GUIDE.md, step R) and ask."
      exit 1
    fi
    echo
    echo "Stage 1 done. Before stage 2, read:"
    echo "  logs/${LP}sim_audit.txt    leaks and unanswered requests by language (WHAT_FAILED #26),"
    echo "                             and photos a customer says she sent but has not (#43)"
    echo "  logs/${LP}check-A.txt      what one example trains on, and the server comparison"
    echo "  data/$RFT/stats.json   per-fold traps with nothing to learn from, and episodes by trap"
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
      if [ -f "checkpoints/$RFT-$F/adapter_config.json" ]; then
        echo "checkpoints/$RFT-$F already trained; skipping"; continue
      fi
      "${launch[@]}" scripts/train_rft.py train \
        --data "data/$RFT/train-$F.jsonl" --out "checkpoints/$RFT-$F" --resume \
        --epochs "$EPOCHS" --max-len "$TRAIN_MAX_LEN" --grad-accum "$ACCUM" \
        2>&1 | tee -a "logs/${LP}train-$F.log"
    done
    serve rft
    REF=""
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
      # Same customer, API agent: the reference row. It shares the customer's
      # GPU with the evaluation, so run them together.
      echo "reference run (deepseek-v4-pro through its API) in the background -> logs/${LP}reference.log"
      $PY -m pasarbench.sweep --backend openai --model deepseek-v4-pro \
        --base-url https://api.deepseek.com/v1 --extra-body '{"thinking":{"type":"disabled"}}' \
        --simulator openai --sim-model "$SIM_MODEL" \
        --sim-url http://localhost:8001/v1 --sim-extra-body '{}' --suite all -k "$K" --temperature 0 \
        --customer "$CUSTOMER" \
        --strategies full --workers 8 --run-id "$P-ref" --resume > "logs/${LP}reference.log" 2>&1 &
      REF=$!
    else
      echo "(no DEEPSEEK_API_KEY: skipping the reference row; RESULTS.md prints its command)"
    fi
    $PY scripts/train_rft.py eval -k "$K" --adapter-prefix "pasar-$RFT-" --run-id "$P-rft" \
      2>&1 | tee -a "logs/${LP}eval.log"
    if [ -n "$REF" ]; then wait "$REF" || echo "!! reference run failed; see logs/${LP}reference.log"; fi
    echo
    echo "Stage 2 done. Copy back to your laptop and regenerate RESULTS.md there:"
    echo "  traces/$P-base traces/$P-rft traces/$P-ref traces/$P-collect data/splits data/$RFT/stats.json"
    ;;

  grpo)
    # GRPO from round 2's fine-tunes (RUNBOOK phase 4, GRPO). For each fold,
    # PASAR_GRPO_ITERS iterations, starting from checkpoints/rft2-<fold>: the
    # adapter served, its rollouts on the fold's tasks it can learn from, their
    # advantages, the servers stopped, one pass of the policy gradient. Then
    # the last iteration's adapters on what no model trained on -- the held-out
    # fold, paired with traces/P2-rft, and the 62 newer tasks, with the RFT
    # adapters and the base model beside them. Run it again after a failure, or
    # with more iterations: finished iterations, sweeps and evaluations are kept.
    if [ "$ROUND" != 2 ]; then
      echo "!! GRPO continues round 2's fine-tunes: run it as PASAR_ROUND=2 (docs/GPU_GUIDE.md, GRPO)" >&2
      exit 1
    fi
    for F in A B; do
      if [ ! -f "checkpoints/$RFT-$F/adapter_config.json" ]; then
        echo "!! no checkpoints/$RFT-$F: GRPO starts from round 2's adapters (its stage 2 first)" >&2
        exit 1
      fi
    done
    for d in "traces/$P-collect/full" "traces/$P-rft/full" "traces/$P-base-e2/full"; do
      if [ ! -d "$d" ]; then
        echo "!! no $d: GRPO picks its tasks from round 2's collection and is compared with" >&2
        echo "   its evaluation and its base run on the newer tasks" >&2
        exit 1
      fi
    done
    if [ -n "$(saved PASAR_GRPO_K)" ] && [ "$GRPO_K" != "$(saved PASAR_GRPO_K)" ]; then
      echo "!! PASAR_GRPO_K=$GRPO_K, but GRPO's first iteration ran with $(saved PASAR_GRPO_K) ($SETTINGS)." >&2
      exit 1
    fi
    if [ -z "$(saved PASAR_GRPO_K)" ]; then printf 'PASAR_GRPO_K=%s\n' "$GRPO_K" >> "$SETTINGS"; fi
    # The tasks are picked once, so the traps left out are too.
    if [ -n "$(saved PASAR_GRPO_SKIP_TRAPS)" ] && [ "$GRPO_SKIP" != "$(saved PASAR_GRPO_SKIP_TRAPS)" ]; then
      echo "!! PASAR_GRPO_SKIP_TRAPS=$GRPO_SKIP, but GRPO's tasks were picked leaving out $(saved PASAR_GRPO_SKIP_TRAPS) ($SETTINGS)." >&2
      exit 1
    fi
    if [ -z "$(saved PASAR_GRPO_SKIP_TRAPS)" ]; then
      printf 'PASAR_GRPO_SKIP_TRAPS=%s\n' "$GRPO_SKIP" >> "$SETTINGS"
    fi
    grpo_reading || exit 1
    stop
    layout
    ACCUM=$(( NGPU > 1 ? (32 + NGPU / 2) / NGPU : 32 )); [ "$ACCUM" -ge 1 ] || ACCUM=1
    if [ "$NGPU" -gt 1 ]; then launch=(accelerate launch --num_processes "$NGPU" --multi_gpu)
    else launch=("$PY"); fi
    G=grpo2
    mkdir -p "data/$G/stats" "data/$G/examples"
    echo "GRPO: $GRPO_ITERS iteration(s) a fold, $GRPO_K rollouts a task, learning rate $GRPO_LR"
    if [ ! -f "data/$G/stats/tasks-A.json" ] || [ ! -f "data/$G/stats/tasks-B.json" ]; then
      # Once: the tasks stay the same from iteration to iteration.
      $PY scripts/train_grpo.py tasks --collect "traces/$P-collect/full" -k "$GRPO_K" \
        --skip-traps "$GRPO_SKIP" --out-dir "data/$G/stats" 2>&1 | tee "logs/${LP}grpo-tasks.log"
    fi
    for F in A B; do
      prev="checkpoints/$RFT-$F"
      for i in $(seq 1 "$GRPO_ITERS"); do
        out="checkpoints/$G-i$i-$F"
        if [ -f "$out/adapter_config.json" ]; then
          echo "$out already trained; skipping"; prev="$out"; continue
        fi
        ex="data/$G/examples/i$i-$F.jsonl"; st="data/$G/stats/i$i-$F.stats.json"
        name="pasar-$(basename "$prev")"
        if [ ! -f "$st" ]; then           # written last: a build cut short is built again
          echo "== fold $F, iteration $i: rollouts of $name" | tee -a "logs/${LP}grpo-$F.log"
          export LORA="$prev"
          serve lora
          $PY scripts/train_grpo.py collect --adapter "$name" -k "$GRPO_K" \
            --tasks-file "data/$G/stats/tasks-$F.json" --run-id "$P-grpo-i$i-collect-$F" \
            2>&1 | tee -a "logs/${LP}grpo-$F.log"
          $PY scripts/train_grpo.py build --run "traces/$P-grpo-i$i-collect-$F/full" \
            --served-model "$AGENT_MODEL:$name" --server http://localhost:8000 \
            --max-len "$TRAIN_MAX_LEN" --reward-reading "$GRPO_READING" \
            --out "$ex" --stats "$st" 2>&1 | tee -a "logs/${LP}grpo-$F.log"
          stop
        fi
        if [ ! -s "$ex" ]; then           # measured by grpo-measure, nothing to train on
          echo "!! $ex is empty: every group of fold $F had one verdict throughout ($st)." >&2
          echo "   The policy has no pass-and-fail left on these tasks: GRPO stops here." >&2
          exit 1
        fi
        echo "== fold $F, iteration $i: the update -> $out" | tee -a "logs/${LP}grpo-$F.log"
        "${launch[@]}" scripts/train_rft.py train --objective grpo --init "$prev" \
          --data "$ex" --out "$out" --resume --epochs 1 --lr "$GRPO_LR" --warmup-steps 3 \
          --max-len "$TRAIN_MAX_LEN" --grad-accum "$ACCUM" 2>&1 | tee -a "logs/${LP}grpo-$F.log"
        prev="$out"
      done
    done
    N="$GRPO_ITERS"
    # The newer tasks' own split, which only shares them out between the two
    # adapters: neither trained on any of them.
    $PY -m pasarbench.rl.split --suite edition2 > /dev/null
    export RFT_NAME="$G-i$N"            # serve_sglang.sh rft: checkpoints/grpo2-iN-A and -B
    serve rft
    $PY scripts/train_rft.py eval -k "$K" --adapter-prefix "pasar-$G-i$N-" --run-id "$P-grpo-i$N" \
      2>&1 | tee -a "logs/${LP}grpo-eval.log"
    $PY scripts/train_rft.py eval -k "$K" --suite edition2 --adapter-prefix "pasar-$G-i$N-" \
      --run-id "$P-grpo-i$N-e2" 2>&1 | tee -a "logs/${LP}grpo-eval.log"
    e2_done="$(finished "traces/$P-rft-e2/full")" || e2_done=0
    if [ "${e2_done:-0}" -lt "$(( 62 * K ))" ]; then   # once, for every N
      export RFT_NAME="$RFT"
      serve rft
      $PY scripts/train_rft.py eval -k "$K" --suite edition2 --adapter-prefix "pasar-$RFT-" \
        --run-id "$P-rft-e2" 2>&1 | tee -a "logs/${LP}grpo-eval.log"
    fi
    export RFT_NAME="$RFT"
    stop
    {
      echo "== GRPO ($N iteration(s)) against RFT on the held-out fold, paired by task"
      $PY scripts/compare_cells.py "traces/$P-rft/full" "traces/$P-grpo-i$N/full"
      echo; echo "== the 62 newer tasks, which no model trained on: base -> RFT"
      $PY scripts/compare_cells.py "traces/$P-base-e2/full" "traces/$P-rft-e2/full"
      echo; echo "== the 62 newer tasks: RFT -> GRPO ($N iteration(s))"
      $PY scripts/compare_cells.py "traces/$P-rft-e2/full" "traces/$P-grpo-i$N-e2/full"
    } > "logs/${LP}grpo-compare-i$N.txt" 2>&1 || echo "!! a comparison failed; the runs are complete"
    echo
    echo "GRPO done. Read, in this order:"
    echo "  logs/${LP}grpo-compare-i$N.txt  the paired comparisons (the result; RUNBOOK phase 4, GRPO)"
    echo "  data/$G/stats/i*-A.stats.json  each iteration's groups, and the policy's pass rate on its tasks"
    ;;

  grpo-measure)
    # The learning curve's next point, without another update (RUNBOOK phase 4,
    # GRPO, the learning curve's second point): the last GRPO iteration's
    # adapters on their own training tasks at temperature 1, rolled out and
    # built exactly as the next iteration's rollouts would be -- its stats, not
    # an update and not an evaluation. Whether GRPO learned the tasks it trained
    # on, which no evaluation reads. A later `grpo` with one more iteration
    # trains on these rollouts rather than rolling out again.
    if [ "$ROUND" != 2 ]; then
      echo "!! GRPO's learning curve is round 2's: run it as PASAR_ROUND=2 (docs/GPU_GUIDE.md, GRPO)" >&2
      exit 1
    fi
    G=grpo2
    N=0
    while [ -f "checkpoints/$G-i$((N + 1))-A/adapter_config.json" ] \
        && [ -f "checkpoints/$G-i$((N + 1))-B/adapter_config.json" ]; do N=$((N + 1)); done
    if [ "$N" -lt 1 ]; then
      echo "!! no checkpoints/$G-i1-A and -B: the grpo stage trains them first" >&2
      exit 1
    fi
    for F in A B; do
      if [ ! -f "data/$G/stats/tasks-$F.json" ] || [ ! -d "traces/$P-grpo-i$N-collect-$F/full" ]; then
        echo "!! no data/$G/stats/tasks-$F.json or traces/$P-grpo-i$N-collect-$F: the rollouts" >&2
        echo "   iteration $N started from are what these are compared with" >&2
        exit 1
      fi
    done
    if [ -n "$(saved PASAR_GRPO_K)" ] && [ "$GRPO_K" != "$(saved PASAR_GRPO_K)" ]; then
      echo "!! PASAR_GRPO_K=$GRPO_K, but GRPO's first iteration ran with $(saved PASAR_GRPO_K) ($SETTINGS)." >&2
      exit 1
    fi
    grpo_reading || exit 1
    i=$((N + 1))
    stop
    layout
    mkdir -p "data/$G/stats" "data/$G/examples"
    for F in A B; do
      st="data/$G/stats/i$i-$F.stats.json"
      if [ -f "$st" ]; then echo "$st already built; skipping"; continue; fi
      name="pasar-$G-i$N-$F"
      echo "== fold $F: $name on its own training tasks, the learning curve's point $i" \
        | tee -a "logs/${LP}grpo-measure.log"
      export LORA="checkpoints/$G-i$N-$F"
      serve lora
      $PY scripts/train_grpo.py collect --adapter "$name" -k "$GRPO_K" \
        --tasks-file "data/$G/stats/tasks-$F.json" --run-id "$P-grpo-i$i-collect-$F" \
        2>&1 | tee -a "logs/${LP}grpo-measure.log"
      $PY scripts/train_grpo.py build --run "traces/$P-grpo-i$i-collect-$F/full" \
        --served-model "$AGENT_MODEL:$name" --server http://localhost:8000 \
        --max-len "$TRAIN_MAX_LEN" --reward-reading "$GRPO_READING" --measure \
        --out "data/$G/examples/i$i-$F.jsonl" --stats "$st" 2>&1 | tee -a "logs/${LP}grpo-measure.log"
      stop
    done
    $PY scripts/grpo_curve.py \
      "traces/$P-grpo-i$N-collect-A/full" "traces/$P-grpo-i$i-collect-A/full" \
      "traces/$P-grpo-i$N-collect-B/full" "traces/$P-grpo-i$i-collect-B/full" \
      > "logs/${LP}grpo-measure-i$i.txt" 2>&1 || echo "!! the comparison failed; the rollouts are complete"
    echo
    echo "GRPO measured on its own tasks. Read logs/${LP}grpo-measure-i$i.txt (RUNBOOK phase 4,"
    echo "GRPO, the learning curve's second point), then data/$G/stats/i$i-*.stats.json."
    ;;

  customer3)
    # Customer 3 against customer 2 (RUNBOOK phase 4, customer 3; WHAT_FAILED
    # #43): the base model at temperature 0 on the photo trap, where customer 2
    # sends photos she was told she has none of, and on the duplicate refund,
    # whose customer promises to check her statement -- the other trap customer
    # 3's new rule reaches. Runs of their own, $P-c2-check and $P-c3-check, the
    # same day, the same agent and customer model; no other stage reads them.
    if [ "$ROUND" != 2 ]; then
      echo "!! the customer check runs on round 2's machine and models: PASAR_ROUND=2" >&2
      exit 1
    fi
    layout
    serve agent
    for C in 2 3; do
      $PY scripts/train_rft.py baseline -k "$K" --customer "$C" \
        --traps high_value_photo_required_first,duplicate_refund_escalate \
        --run-id "$P-c$C-check" 2>&1 | tee -a "logs/${LP}customer3.log"
    done
    stop
    {
      echo "== customer 2 -> customer 3, the same agent, paired by task"
      $PY scripts/compare_cells.py "traces/$P-c2-check/full" "traces/$P-c3-check/full"
      for C in 2 3; do
        echo; echo "== customer $C: photos she said she sent"
        $PY scripts/inspect_trace.py "traces/$P-c$C-check/full" --leaks-only | sed -n '/^4\. /,$p'
      done
    } > "logs/${LP}customer3.txt" 2>&1 || echo "!! a comparison failed; the runs are complete"
    echo
    echo "Customer 3 checked. Read logs/${LP}customer3.txt (RUNBOOK phase 4, customer 3)."
    ;;

  edition3)
    # The hard edition (RUNBOOK 1s): 36 cases that combine the suite's traps,
    # which no model trained on, under one customer -- customer 3 once the
    # customer3 stage has checked her. The base model, round 2's RFT adapters
    # and, if GRPO has run, its last iteration's, each at temperature 0, K
    # times a task; then base -> RFT and RFT -> GRPO paired by task, and the
    # changed request's audit of each run. Runs of their own: $P-base-e3,
    # $P-rft-e3, $P-grpo-iN-e3. Run it again after a failure: finished runs
    # are not served again, and the sweeps resume.
    if [ "$ROUND" != 2 ]; then
      echo "!! the hard edition runs on round 2's models: PASAR_ROUND=2" >&2
      exit 1
    fi
    for F in A B; do
      if [ ! -f "checkpoints/$RFT-$F/adapter_config.json" ]; then
        echo "!! no checkpoints/$RFT-$F: the hard edition runs round 2's adapters (its stage 2 first)" >&2
        exit 1
      fi
    done
    if [ "$E3_CUSTOMER" = 3 ] && [ ! -d "traces/$P-c3-check/full" ]; then
      echo "!! customer 3 has not been checked: run the customer3 stage and read" >&2
      echo "   logs/${LP}customer3.txt first (RUNBOOK phase 4, customer 3) -- or, if that" >&2
      echo "   check said to keep customer 2, PASAR_E3_CUSTOMER=2 (RUNBOOK 1s)" >&2
      exit 1
    fi
    if [ -n "$(saved PASAR_E3_CUSTOMER)" ] && [ "$E3_CUSTOMER" != "$(saved PASAR_E3_CUSTOMER)" ]; then
      echo "!! PASAR_E3_CUSTOMER=$E3_CUSTOMER, but the hard edition's first run had customer $(saved PASAR_E3_CUSTOMER) ($SETTINGS)." >&2
      exit 1
    fi
    if [ -z "$(saved PASAR_E3_CUSTOMER)" ]; then printf 'PASAR_E3_CUSTOMER=%s\n' "$E3_CUSTOMER" >> "$SETTINGS"; fi
    E3_N="$($PY -c 'from pasarbench.generate import generate; print(len(generate(edition=3)[0]))')"
    left() {  # $1 = a run id: whether it has episodes still to run
      local n; n="$(finished "traces/$1/full")" || n=0
      [ "${n:-0}" -lt "$(( E3_N * K ))" ]
    }
    echo "the hard edition: $E3_N tasks x $K, customer $E3_CUSTOMER"
    layout
    $PY -m pasarbench.rl.split --suite edition3 > /dev/null
    if left "$P-base-e3"; then
      serve agent
      $PY scripts/train_rft.py baseline -k "$K" --suite edition3 --customer "$E3_CUSTOMER" \
        --run-id "$P-base-e3" 2>&1 | tee -a "logs/${LP}edition3.log"
    fi
    if left "$P-rft-e3"; then
      export RFT_NAME="$RFT"
      serve rft
      $PY scripts/train_rft.py eval -k "$K" --suite edition3 --customer "$E3_CUSTOMER" \
        --adapter-prefix "pasar-$RFT-" --run-id "$P-rft-e3" 2>&1 | tee -a "logs/${LP}edition3.log"
    fi
    G=grpo2; N="$GRPO_ITERS"; runs=("$P-base-e3" "$P-rft-e3")
    if [ -f "checkpoints/$G-i$N-A/adapter_config.json" ] && [ -f "checkpoints/$G-i$N-B/adapter_config.json" ]; then
      if left "$P-grpo-i$N-e3"; then
        export RFT_NAME="$G-i$N"
        serve rft
        $PY scripts/train_rft.py eval -k "$K" --suite edition3 --customer "$E3_CUSTOMER" \
          --adapter-prefix "pasar-$G-i$N-" --run-id "$P-grpo-i$N-e3" 2>&1 | tee -a "logs/${LP}edition3.log"
      fi
      runs+=("$P-grpo-i$N-e3")
    else
      echo "(no checkpoints/$G-i$N-A and -B: the hard edition runs without GRPO's adapters)"
    fi
    export RFT_NAME="$RFT"
    stop
    {
      echo "== the hard edition, which no model trained on: base -> RFT, paired by task"
      $PY scripts/compare_cells.py "traces/$P-base-e3/full" "traces/$P-rft-e3/full"
      if [ "${#runs[@]}" -gt 2 ]; then
        echo; echo "== the hard edition: RFT -> GRPO ($N iteration(s))"
        $PY scripts/compare_cells.py "traces/$P-rft-e3/full" "traces/$P-grpo-i$N-e3/full"
      fi
      for r in "${runs[@]}"; do
        echo; echo "== $r: did the customer who changes her request change it with her digits?"
        $PY scripts/inspect_trace.py "traces/$r/full" --leaks-only | sed -n '/^4b\. /,$p'
      done
    } > "logs/${LP}edition3.txt" 2>&1 || echo "!! a comparison failed; the runs are complete"
    echo
    echo "The hard edition ran. Read logs/${LP}edition3.txt (RUNBOOK 1s)."
    ;;

  serving)
    # Phase 3 (RUNBOOK): what serving the agent costs, three ways -- the policy
    # in the prompt with SGLang's radix cache on, as every run so far; no
    # policy in the prompt, the agent looking rules up (jit); and the policy in
    # the prompt with the agent's radix cache off. The base agent and the
    # customer, both started afresh for each bar, so every bar's agent gets the
    # memory plan of the first; only the agent's flags change (AGENT_ARGS: the
    # customer keeps her cache, and the load the agent is measured under is the
    # same). The no-cache bar is last: if its server will not start with the
    # flag, the other two are measured. A bar is measured whole -- run the
    # stage again after a failure, and measured bars are kept while the one cut
    # short starts over. The report is rewritten after each bar.
    SERVE_K="${PASAR_SERVE_K:-1}"
    SERVE_SAMPLE="${PASAR_SERVE_SAMPLE:-0}"
    PRICE="$($PY scripts/gpu_plan.py price)"
    report_serving() {
      $PY scripts/serving_bars.py report --prefix "$P" --usd-per-hour "$PRICE" \
        > "logs/${LP}serving.md" || echo "!! the serving report failed; the bars are on disk"
    }
    layout
    PLAN="$($PY scripts/serving_bars.py plan)"
    failed=""
    # The plan on its own descriptor: the servers started in the loop inherit
    # the loop's standard input, and must not be handed the plan's lines.
    while IFS= read -r line <&3; do
      bar="${line%%$'\t'*}"; flags="${line#*$'\t'}"
      if $PY scripts/serving_bars.py done --bar "$bar" --prefix "$P" -k "$SERVE_K" \
          --sample "$SERVE_SAMPLE"; then
        echo "== $bar: measured already (data/serving/$P-serve-$bar.json)"
        continue
      fi
      echo "== $bar${flags:+ (the agent with $flags)}"
      export AGENT_ARGS="$flags"
      serve agent
      $PY scripts/serving_bars.py run --bar "$bar" --prefix "$P" -k "$SERVE_K" \
        --sample "$SERVE_SAMPLE" --customer "$CUSTOMER" --usd-per-hour "$PRICE" 2>&1 \
        | tee -a "logs/${LP}serving.log" || failed="$failed $bar"
      report_serving
    done 3<<< "$PLAN"
    unset AGENT_ARGS
    stop
    report_serving
    echo
    if [ -n "$failed" ]; then
      echo "!! not measured:$failed -- run the stage again; logs/${LP}serving.log says why"
    fi
    echo "Serving measured. Read logs/${LP}serving.md (RUNBOOK phase 3)."
    ;;

  settings)
    echo "agent $AGENT_MODEL | customer $SIM_MODEL | round $ROUND: runs $P-*, adapters checkpoints/$RFT-A and -B"
    echo "PASAR_K=$K PASAR_COLLECT_K=$COLLECT_K PASAR_EPOCHS=$EPOCHS PASAR_TRAIN_MAX_LEN=$TRAIN_MAX_LEN PASAR_CUSTOMER=$PASAR_CUSTOMER PASAR_PER_TRAP=$PER_TRAP" \
         "($([ -f "$SETTINGS" ] && echo "stage 1 recorded in $SETTINGS" || echo "stage 1 not run yet"))"
    if [ "$ROUND" = 2 ]; then
      echo "grpo: PASAR_GRPO_ITERS=$GRPO_ITERS PASAR_GRPO_K=$GRPO_K PASAR_GRPO_LR=$GRPO_LR PASAR_GRPO_SKIP_TRAPS=$GRPO_SKIP PASAR_GRPO_READING=$GRPO_READING"
      echo "edition3: PASAR_E3_CUSTOMER=$E3_CUSTOMER"
    fi
    "$PY" scripts/gpu_plan.py show
    ;;

  *)
    echo "usage: bash scripts/gpu_pipeline.sh smoke | stage1 | stage2 | grpo | grpo-measure | customer3 | edition3 | serving | settings" >&2
    exit 1
    ;;
esac
