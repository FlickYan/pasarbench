#!/usr/bin/env bash
# Serve models for PasarBench with SGLang, on a rented 2x H100 node.
#
#   ./scripts/serve_sglang.sh agent       # GPU 0, port 8000: Qwen3-8B, the model under training
#   ./scripts/serve_sglang.sh sim         # GPU 1, port 8001: the simulated customer, Gemma 4 31B
#   ./scripts/serve_sglang.sh rft         # GPU 0, port 8000: Qwen3-8B with both RFT adapters
#   ./scripts/serve_sglang.sh agent-fp8   # GPU 0: FP8 weights and KV cache, for the serving comparison
#
# gpu_pipeline.sh starts and stops these itself; start them by hand only for
# phase 3 or to debug. Override with AGENT_MODEL, SIM_MODEL, SIM_TOKENIZER,
# AGENT_GPU, SIM_GPU, PORT, SIM_PORT and HOST; SGLANG_ARGS is appended to the
# command (e.g. SGLANG_ARGS="--mem-fraction-static 0.8" after an OOM).
# DRY_RUN=1 prints the command instead of starting it.
#
# Servers listen on 127.0.0.1 without an API key: every client runs on this
# node. Do not set HOST=0.0.0.0 on a machine with a public address.
#
# THE LAYOUT
# ----------
# Rollouts and evaluation: agent on GPU 0, customer on GPU 1, both at once.
# Training: both stopped; LoRA uses both cards. The reference is
# deepseek-v4-pro through its API, re-run against THIS customer so its row is
# comparable.
#
# THE CUSTOMER
# ------------
# Gemma 4 31B-it: another family from the Qwen agent (pasarbench.sweep refuses
# a same-family pair), multilingual across all eight varieties, Apache 2.0 and
# not gated. Its bf16 weights are ~62 GB, which leaves an 80 GB card KV cache
# for only two or three conversations of this benchmark's length (each token
# held in its 50 sliding-window layers takes ~0.8 MB), against 24 agent
# workers. So the default is
# RedHat's FP8-Dynamic checkpoint of the same model: per-channel FP8 weights,
# per-token activations, vision tower and embeddings left in bf16, 99-100% of
# bf16 on the benchmarks its model card reports (IFEval included), ~33 GB.
# It runs with Google's own tokenizer and chat template (the checkpoint was
# first published before Google's later template fixes), and the traces name
# the checkpoint, so every run records which customer it had. On a card with
# room for bf16:
#   SIM_MODEL=google/gemma-4-31B-it ./scripts/serve_sglang.sh sim
#
# SGLang keeps Gemma 4's special tokens in the decoded text so its parsers can
# read them; `--reasoning-parser gemma4` takes any thought channel out of the
# reply. `inspect_trace.py --leaks-only` counts chat-format tokens on both
# sides of every conversation, and the pipeline runs it after the smoke test.
#
# A NEW CUSTOMER IS A NEW INSTRUMENT. Before training anything, read the leak
# and stall audit of the baseline run (RUNBOOK phase 4). The ask-patterns were
# checked against DeepSeek's phrasing, not Qwen's (WHAT_FAILED #26).

set -euo pipefail
MODE="${1:-agent}"
PY="${PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-8B}"
SIM_DEFAULT="RedHatAI/gemma-4-31B-it-FP8-Dynamic"
SIM_MODEL="${SIM_MODEL:-$SIM_DEFAULT}"
if [ -z "${SIM_TOKENIZER:-}" ]; then
  if [ "$SIM_MODEL" = "$SIM_DEFAULT" ]; then SIM_TOKENIZER="google/gemma-4-31B-it"
  else SIM_TOKENIZER="$SIM_MODEL"; fi
fi
AGENT_GPU="${AGENT_GPU:-0}"
SIM_GPU="${SIM_GPU:-1}"
# shellcheck disable=SC2206
extra=(${SGLANG_ARGS:-})

run() {  # the command, or with DRY_RUN=1 just its text
  if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "$@"; echo; else exec "$@"; fi
}

agent=(
  env CUDA_VISIBLE_DEVICES="$AGENT_GPU" "$PY" -m sglang.launch_server
  --model-path "$AGENT_MODEL" --host "$HOST" --port "${PORT:-8000}"
  --context-length 32768
  --tool-call-parser qwen        # Qwen3 writes <tool_call>{json}</tool_call>, the Qwen2.5 format
  --enable-metrics               # Prometheus counters at /metrics; the radix cache is on by default
  --enable-cache-report          # cached prompt tokens in each response's usage, as the APIs report
)

case "$MODE" in
  agent)
    run "${agent[@]}" ${extra[@]+"${extra[@]}"}
    ;;
  rft)
    # Both fold adapters from one server. A request names one as
    # <base>:<adapter> (Qwen/Qwen3-8B:pasar-rft-A); a request naming the bare
    # adapter is answered by the BASE model, without an error, so the sweep
    # checks every fold model against the server's list before it starts.
    for f in checkpoints/rft-A checkpoints/rft-B; do
      [ -f "$f/adapter_config.json" ] || { echo "no adapter in $f -- train it first" >&2; exit 1; }
    done
    run "${agent[@]}" --enable-lora --max-loras-per-batch 2 \
      --lora-paths pasar-rft-A=checkpoints/rft-A pasar-rft-B=checkpoints/rft-B \
      ${extra[@]+"${extra[@]}"}
    ;;
  agent-fp8)
    # H100 has native FP8. Report BOTH throughput AND pass^k: a quantisation
    # that speeds things up and drops task success is not a win, and on a
    # policy-following task the loss shows up in the traps before perplexity.
    run "${agent[@]}" --quantization fp8 --kv-cache-dtype fp8_e4m3 ${extra[@]+"${extra[@]}"}
    ;;
  sim)
    # Short turns, no tools, thinking off (Gemma 4's template default: the
    # sweep sends the customer no chat_template_kwargs). 16k context covers
    # the longest conversation.
    sim=(
      env CUDA_VISIBLE_DEVICES="$SIM_GPU" "$PY" -m sglang.launch_server
      --model-path "$SIM_MODEL" --tokenizer-path "$SIM_TOKENIZER"
      --host "$HOST" --port "${SIM_PORT:-8001}" --context-length 16384
      --enable-metrics --enable-cache-report
    )
    shopt -s nocasematch
    [[ "$SIM_MODEL" == *gemma-4* ]] && sim+=(--reasoning-parser gemma4)
    shopt -u nocasematch
    run "${sim[@]}" ${extra[@]+"${extra[@]}"}
    ;;
  *) echo "unknown mode: $MODE (agent | sim | rft | agent-fp8)" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# MEASURING THE SERVING RESULT (RUNBOOK phase 3)
#
#   from pasarbench.serving import MetricsSnapshot
#   before = MetricsSnapshot.from_text(requests.get(f"{url}/metrics").text)
#   ... run the sweep cell ...
#   after  = MetricsSnapshot.from_text(requests.get(f"{url}/metrics").text)
#   before.diff(after).prefix_cache_hit_rate()   # counters, NOT the gauge
#
# DIFF THE COUNTERS. `sglang:cached_tokens_total` and `sglang:prompt_tokens_total`
# are cumulative over the server's lifetime, and `sglang:cache_hit_rate` is a
# gauge of recent traffic, so reading any of them once mixes in every
# experiment you ran before. MetricsDiff raises if a counter went backwards,
# which means the server restarted mid-run and the numbers must be discarded.
#
# WARM UP FIRST. The first request pays CUDA graph capture and an empty cache.
# Folding it into p99 flatters everything measured after it.
#
# Three bars:
#   preload + radix cache      (shared policy prefix, cache does the work)
#   preload, cache disabled    (SGLANG_ARGS=--disable-radix-cache)
#   jit                        (tiny prefix, extra search_policy round trips)
#
# AND NEVER REPORT A SPEED WIN WITHOUT serving_verdict(). At this suite size a
# 2-point regression is undetectable, INCONCLUSIVE is often the honest answer,
# and per_trap_guard() shows FP8 damage concentrating in specific traps while
# the aggregate barely moves.
# ---------------------------------------------------------------------------
