#!/usr/bin/env bash
# Serve a model for PasarBench on local A100/H100.
#
#   ./scripts/serve_vllm.sh agent-32b     # agent under test
#   ./scripts/serve_vllm.sh reference-70b # strong reference, TP=4
#   ./scripts/serve_vllm.sh sim           # user simulator, small + different family
#   ./scripts/serve_vllm.sh agent-8b-fp8  # the trained policy, FP8 on Hopper
#
# WHY THE TIERS MATTER
# --------------------
# The headline result of the post-training track is a gap-closing number:
#   "an 8B policy trained on PasarBench trajectories recovers X% of the 70B
#    reference's pass^4 at Y% of the cost per resolved conversation."
# That claim needs all three tiers measured on the same task suite with the
# same harness. Serve them separately, sweep each, compare.
#
# THE SIMULATOR MUST BE A DIFFERENT MODEL FAMILY FROM THE AGENT.
# A simulator sharing the agent's weights is unusually easy for that agent to
# satisfy, and it inflates scores in a way that is invisible in the numbers.
# Qwen agent -> Llama simulator, or vice versa. Say so in the writeup.

set -euo pipefail
MODE="${1:-agent-32b}"
PORT="${PORT:-8000}"

# FlashAttention-3 is Hopper-only. On H100 this is a real throughput win and a
# genuine differentiator: almost every public agent benchmark is run on A100 or
# consumer cards and cannot use it.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_USE_V1=1

common=(
  --host 0.0.0.0 --port "$PORT" --api-key EMPTY
  --enable-prefix-caching          # ~2.2k tokens of policy are a shared prefix
  --enable-auto-tool-choice
  --max-model-len 32768
  --gpu-memory-utilization 0.90
)

case "$MODE" in
  agent-8b)
    exec vllm serve Qwen/Qwen3-8B "${common[@]}" \
      --tool-call-parser hermes --tensor-parallel-size 1
    ;;
  agent-8b-fp8)
    # H100 has native FP8 tensor cores. W8A8 + FP8 KV cache roughly halves
    # weight memory and materially raises throughput. Report BOTH throughput
    # AND pass^k -- a quantisation that speeds things up and drops task
    # success is not a win, and on a policy-following task the loss shows up
    # in the traps before it shows up in perplexity.
    exec vllm serve Qwen/Qwen3-8B "${common[@]}" \
      --tool-call-parser hermes --tensor-parallel-size 1 \
      --quantization fp8 --kv-cache-dtype fp8
    ;;
  agent-32b)
    exec vllm serve Qwen/Qwen3-32B "${common[@]}" \
      --tool-call-parser hermes --tensor-parallel-size 2
    ;;
  reference-70b)
    # ~140GB bf16 weights across 4x80GB leaves ~180GB for KV cache. Comfortable.
    exec vllm serve Qwen/Qwen3-72B-Instruct "${common[@]}" \
      --tool-call-parser hermes --tensor-parallel-size 4
    ;;
  sim)
    # Different family from the agent, deliberately. Cheap: it is half the
    # tokens in every episode.
    exec vllm serve meta-llama/Llama-3.1-8B-Instruct "${common[@]}" \
      --tool-call-parser llama3_json --tensor-parallel-size 1 --port 8001
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# MEASURING THE SERVING RESULT
#
#   from pasarbench.serving import MetricsSnapshot, three_bar_report
#   before = MetricsSnapshot.from_text(requests.get(f"{url}/metrics").text)
#   ... run the sweep cell ...
#   after  = MetricsSnapshot.from_text(requests.get(f"{url}/metrics").text)
#   before.diff(after).prefix_cache_hit_rate()   # counters, NOT the gauge
#
# DIFF THE COUNTERS. `vllm:prefix_cache_hits_total` is cumulative over the
# server's lifetime, so reading it once at the end mixes in every experiment
# you ran before. MetricsDiff raises if a counter went backwards, which means
# the server restarted mid-run and the numbers must be discarded.
#
# WARM UP FIRST. The first request pays model load, CUDA graph capture and an
# empty cache. Folding it into p99 flatters everything measured after it.
#
# Three bars:
#   preload + prefix caching   (2.2k-token shared prefix, cache does the work)
#   preload, caching disabled  (--no-enable-prefix-caching)
#   jit                        (tiny prefix, extra search_policy round trips)
#
# JIT can LOSE on latency despite sending far fewer tokens, because the extra
# round trip costs more than the cached prefill it saves. Workload-specific,
# non-obvious, and exactly what an inference team wants to hear.
#
# AND NEVER REPORT A SPEED WIN WITHOUT serving_verdict(). On 186 tasks you
# cannot detect a 2-point regression -- required_n() says you would need ~6,000
# per arm. The honest verdict at this sample size is often INCONCLUSIVE, and
# per_trap_guard() will show you that FP8 damage concentrates in specific
# traps while the aggregate barely moves.
# ---------------------------------------------------------------------------
