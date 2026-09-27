"""
Serve an open model on Modal with an OpenAI-compatible endpoint.

    pip install modal && modal setup
    modal deploy scripts/modal_vllm.py
    # -> https://<workspace>--pasarbench-vllm-serve.modal.run/v1

    python -m pasarbench.sweep --backend openai \
        --model Qwen/Qwen3-8B \
        --base-url https://<workspace>--pasarbench-vllm-serve.modal.run/v1 \
        --api-key EMPTY

WHY THIS MATTERS FOR THE WEEK 7 RESULT
--------------------------------------
Every episode sends the same system prompt: the agent instructions plus ~2.2k
tokens of policy document, unchanged across every step of every task. That is
a large, perfectly shared prefix.

`--enable-prefix-caching` makes vLLM reuse the KV cache for it instead of
recomputing attention over those tokens on every call. The headline number to
report is prefix cache hit rate against time-to-first-token, measured across
your full task suite.

Then run the same sweep with `--policy-mode jit`, where the policy is NOT in
the prompt. Now the shared prefix nearly vanishes, but the agent makes extra
round trips to `search_policy`. Preload+caching versus JIT is a genuine
latency-cost-accuracy trade, it is specific to agent workloads, and almost
nobody measures it. That is your serving result.

Use this only for burst capacity; it is the one place the repo still serves
with vLLM. On a rented node, use scripts/gpu_pipeline.sh, which serves the
agent and the simulated customer with SGLang (scripts/serve_sglang.sh) on one
card each. The customer must not be a Qwen model when the agent is
(pasarbench.sweep refuses the pair).

Cost note: scale_down_window keeps the container warm between sweep batches.
A cold start on a 32B is 4-8 minutes; paying it once per batch rather than
once per task is the difference between a cheap sweep and an expensive one.
"""

import modal

MODEL = "Qwen/Qwen3-32B"
GPU = "H100:2"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.8.5", "huggingface_hub[hf_transfer]==0.30.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

app = modal.App("pasarbench-vllm")
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": cache},
    scale_down_window=15 * 60,
    timeout=60 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=10 * 60)
def serve():
    import subprocess

    subprocess.Popen(" ".join([
        "vllm serve", MODEL,
        "--host 0.0.0.0", f"--port {PORT}",
        "--api-key EMPTY",
        "--enable-prefix-caching",          # the whole point -- see module docstring
        "--enable-auto-tool-choice",
        "--tool-call-parser hermes",        # match to your model's template
        "--max-model-len 32768",
        "--tensor-parallel-size 2",
        "--gpu-memory-utilization 0.90",
    ]), shell=True)


# ---------------------------------------------------------------------------
# Measuring the thing you actually want to report.
#
# vLLM exposes Prometheus metrics at /metrics. The two series that matter:
#
#   vllm:gpu_prefix_cache_hit_rate
#   vllm:time_to_first_token_seconds
#
# Scrape /metrics before and after each sweep cell and diff the counters.
# Report TTFT p50/p99 and hit rate for: preload+caching, preload without
# caching (--no-enable-prefix-caching), and jit. Three bars, one clear story.
#
# Do NOT report a cache hit rate without also reporting task success. A
# strategy that caches beautifully and fails the tasks is not a result.
# ---------------------------------------------------------------------------
