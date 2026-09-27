"""
Serving layer tests.

Two things are load-bearing here.

`test_cost_inversion` encodes the finding: a model that is cheaper per token
and escalates more is usually the MORE expensive model once human handling is
priced in. If that assertion ever flips, the cost model has stopped modelling
the thing a CS org actually pays for.

`test_quality_guard` enforces that no speed win can be declared without a
non-inferiority result, including the INCONCLUSIVE case, which is the honest
verdict at this sample size and the one people quietly skip.

Run: python -m tests.test_serving
"""

from __future__ import annotations

from pasarbench.serving.cost import (ServingConfig, amortised_capacity,
                                     break_even_human_cost, compare, markdown)
from pasarbench.serving.metrics import (MetricsSnapshot, histogram_quantile,
                                        parse_prometheus)
from pasarbench.serving.report import (non_inferiority, per_trap_guard,
                                       proportion_diff_ci, required_n,
                                       serving_verdict, three_bar_report)

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


METRICS_BEFORE = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="Qwen3-8B"} 1000
vllm:prefix_cache_hits_total{model_name="Qwen3-8B"} 300
vllm:prompt_tokens_total{model_name="Qwen3-8B"} 500000
vllm:generation_tokens_total{model_name="Qwen3-8B"} 50000
vllm:time_to_first_token_seconds_bucket{le="0.1"} 10
vllm:time_to_first_token_seconds_bucket{le="0.25"} 20
vllm:time_to_first_token_seconds_bucket{le="0.5"} 30
vllm:time_to_first_token_seconds_bucket{le="1.0"} 40
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 40
"""

METRICS_AFTER = """
vllm:prefix_cache_queries_total{model_name="Qwen3-8B"} 3000
vllm:prefix_cache_hits_total{model_name="Qwen3-8B"} 2100
vllm:prompt_tokens_total{model_name="Qwen3-8B"} 1500000
vllm:generation_tokens_total{model_name="Qwen3-8B"} 140000
vllm:time_to_first_token_seconds_bucket{le="0.1"} 60
vllm:time_to_first_token_seconds_bucket{le="0.25"} 150
vllm:time_to_first_token_seconds_bucket{le="0.5"} 190
vllm:time_to_first_token_seconds_bucket{le="1.0"} 198
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 200
"""


def test_parsing():
    print("\n=== Prometheus parsing ===")
    s = parse_prometheus(METRICS_BEFORE)
    check("comments and blanks are skipped", all(not x["name"].startswith("#") for x in s))
    check("labels are parsed",
          any(x["labels"].get("model_name") == "Qwen3-8B" for x in s))
    snap = MetricsSnapshot.from_text(METRICS_BEFORE)
    check("values are addressable by name",
          snap.value("vllm:prefix_cache_queries_total") == 1000)
    check("histogram buckets are extracted and sorted",
          [le for le, _ in snap.buckets("vllm:time_to_first_token_seconds")][:2]
          == [0.1, 0.25])
    check("malformed lines do not raise", len(parse_prometheus("garbage ??\nx 1")) == 1)


def test_counter_diffing():
    print("\n=== counters are diffed, not read ===")
    d = MetricsSnapshot.from_text(METRICS_BEFORE).diff(
        MetricsSnapshot.from_text(METRICS_AFTER))

    pc = d.prefix_cache_hit_rate()
    check("hit rate is computed over THIS run only",
          pc["hit_rate"] == round(1800 / 2000, 4), str(pc))
    check("reading the gauge naively would have given a different answer",
          pc["hit_rate"] != round(2100 / 3000, 4),
          "lifetime rate 0.70 vs this-run rate 0.90")
    check("the metric name used is recorded for reproducibility",
          pc["metric"] == "vllm:prefix_cache_queries_total")

    tp = d.throughput(wall_seconds=60.0)
    check("prompt tokens diffed", tp["prompt_tokens"] == 1_000_000)
    check("output tok/s computed", tp["output_tok_per_s"] == 1500.0, str(tp))

    restarted = MetricsSnapshot.from_text(METRICS_AFTER).diff(
        MetricsSnapshot.from_text(METRICS_BEFORE))
    try:
        restarted.prefix_cache_hit_rate()
        raised = False
    except ValueError as e:
        raised = "restarted" in str(e)
    check("a counter reset raises instead of silently clamping", raised)


def test_histogram_quantiles():
    print("\n=== histogram quantiles ===")
    b = [(0.1, 50), (0.25, 100), (0.5, 150), (1.0, 200)]
    check("p50 lands in the right bucket by interpolation",
          0.1 <= histogram_quantile(b, 0.5) <= 0.25,
          str(histogram_quantile(b, 0.5)))
    check("p99 is near the tail", histogram_quantile(b, 0.99) > 0.5,
          str(histogram_quantile(b, 0.99)))
    check("p0 is at the bottom edge", histogram_quantile(b, 0.0) is not None)
    check("empty histogram returns None", histogram_quantile([], 0.5) is None)
    check("all-zero histogram returns None", histogram_quantile([(1.0, 0)], 0.5) is None)

    d = MetricsSnapshot.from_text(METRICS_BEFORE).diff(
        MetricsSnapshot.from_text(METRICS_AFTER))
    lat = d.latency()
    check("latency is computed from bucket DELTAS", lat["n"] == 160, str(lat["n"]))
    check("the bucket-resolution caveat is carried",
          "interpolations rather than measurements" in lat["resolution_note"])


SGLANG_BEFORE = """
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="Qwen/Qwen3-8B",is_streaming="false"} 400000
sglang:generation_tokens_total{model_name="Qwen/Qwen3-8B",is_streaming="false"} 40000
sglang:cached_tokens_total{model_name="Qwen/Qwen3-8B",cache_source="device"} 100000
sglang:cache_hit_rate{model_name="Qwen/Qwen3-8B"} 0.97
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="false",le="0.1"} 5
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="false",le="0.5"} 10
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="false",le="+Inf"} 10
"""

SGLANG_AFTER = """
sglang:prompt_tokens_total{model_name="Qwen/Qwen3-8B",is_streaming="false"} 1400000
sglang:generation_tokens_total{model_name="Qwen/Qwen3-8B",is_streaming="false"} 100000
sglang:cached_tokens_total{model_name="Qwen/Qwen3-8B",cache_source="device"} 800000
sglang:cached_tokens_total{model_name="Qwen/Qwen3-8B",cache_source="host"} 100000
sglang:cache_hit_rate{model_name="Qwen/Qwen3-8B"} 0.10
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="false",le="0.1"} 45
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="false",le="0.5"} 90
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="false",le="+Inf"} 100
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="true",le="0.1"} 10
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="true",le="0.5"} 10
sglang:time_to_first_token_seconds_bucket{model_name="Qwen/Qwen3-8B",is_streaming="true",le="+Inf"} 10
"""


def test_sglang_metrics():
    print("\n=== the same numbers from SGLang's counters ===")
    before = MetricsSnapshot.from_text(SGLANG_BEFORE)
    after = MetricsSnapshot.from_text(SGLANG_AFTER)
    check("the engine is recognised from the metric names",
          after.engine() == "sglang" and MetricsSnapshot.from_text(METRICS_AFTER).engine() == "vllm")
    d = before.diff(after)
    pc = d.prefix_cache_hit_rate()
    check("hit rate = cached tokens over prompt tokens, this run only, every cache tier",
          pc["hit_rate"] == round(800_000 / 1_000_000, 4)
          and pc["metric"] == "sglang:prompt_tokens_total", str(pc))
    check("…not the gauge, which says something else entirely",
          pc["hit_rate"] != 0.10)
    tp = d.throughput(wall_seconds=100.0)
    check("throughput reads SGLang's token counters",
          tp["prompt_tokens"] == 1_000_000 and tp["output_tok_per_s"] == 600.0, str(tp))
    lat = d.latency()
    check("TTFT sums the streaming and non-streaming series edge by edge",
          lat["n"] == 100 and lat["metric"] == "sglang:time_to_first_token_seconds",
          str(lat))
    check("…so the buckets never repeat an edge",
          [le for le, _ in after.buckets("sglang:time_to_first_token_seconds")]
          == [0.1, 0.5, float("inf")])


def test_cost_inversion():
    print("\n=== THE finding: cheap per token, expensive per resolution ===")
    # Small model: 3x cheaper to run, escalates far more, passes less
    small = ServingConfig("qwen-8b-fp8", n_conversations=200, passed=150,
                          escalated=44, prompt_tokens=1_800_000,
                          completion_tokens=180_000, wall_seconds=600,
                          gpu="H100-80", n_gpus=1)
    # Big model: 4 GPUs, far more expensive to run, resolves much more
    big = ServingConfig("qwen-72b-bf16", n_conversations=200, passed=182,
                        escalated=20, prompt_tokens=1_800_000,
                        completion_tokens=180_000, wall_seconds=900,
                        gpu="H100-80", n_gpus=4)

    check("the small model has the smaller model bill",
          small.model_cost_usd() < big.model_cost_usd(),
          f"{small.model_cost_usd():.2f} vs {big.model_cost_usd():.2f}")

    r = compare([small, big])
    check("but the big model is cheaper per RESOLVED conversation",
          r["best_per_resolved"] == "qwen-72b-bf16", str(r["best_per_resolved"]))
    check("the inversion is called out explicitly", r["inversion"] is not None)
    check("the explanation names escalation as the cause",
          "escalation" in (r["inversion"] or ""), r["inversion"])

    sm = next(x for x in r["rows"] if x["config"] == "qwen-8b-fp8")
    check("human cost dominates the model bill", sm["model_share"] < 0.25,
          f"model share {sm['model_share']}")
    print(f"       small: ${sm['usd_per_resolved']:.4f}/resolved  "
          f"(model {sm['model_share']:.1%} of total)")
    bg = next(x for x in r["rows"] if x["config"] == "qwen-72b-bf16")
    print(f"       big:   ${bg['usd_per_resolved']:.4f}/resolved  "
          f"(model {bg['model_share']:.1%} of total)")

    check("markdown renders the inversion", "**" in markdown(r))


def test_break_even():
    print("\n=== break-even human cost ===")
    cheap = ServingConfig("cheap", 200, passed=150, escalated=44,
                          prompt_tokens=1_000_000, completion_tokens=100_000,
                          wall_seconds=600, n_gpus=1)
    dear = ServingConfig("dear", 200, passed=185, escalated=15,
                         prompt_tokens=1_000_000, completion_tokens=100_000,
                         wall_seconds=900, n_gpus=4)
    be = break_even_human_cost(cheap, dear)
    check("a crossover point exists", be["crossover_usd"] is not None, str(be))
    check("the cheap model wins below it", be["below_crossover_winner"] == "cheap",
          str(be))
    check("the expensive model wins above it", be["above_crossover_winner"] == "dear")
    check("the note is decision-shaped", "$" in be["note"], be["note"])
    print(f"       {be['note']}")

    same = ServingConfig("same", 200, passed=150, escalated=44,
                         prompt_tokens=1_000_000, completion_tokens=100_000,
                         wall_seconds=600, n_gpus=1)
    check("identical human load reports no crossover",
          break_even_human_cost(cheap, same)["crossover_usd"] is None)

    cap = amortised_capacity(dear)
    check("self-hosted capacity is expressed per GPU-hour",
          cap["resolved_per_gpu_hour"] is not None, str(cap))


def test_quality_guard():
    print("\n=== no speed win without a proven quality result ===")
    held = non_inferiority(0.80, 186, 0.81, 186, margin=0.02)
    check("a tiny drop at n=186 is INCONCLUSIVE, not a pass",
          "INCONCLUSIVE" in held.verdict, held.verdict)
    check("the required sample size is attached", held.needed_n > 186,
          str(held.needed_n))
    print(f"       to rule out a 2-point regression you need ~{held.needed_n} "
          f"tasks per arm, not 186")

    bad = non_inferiority(0.55, 400, 0.80, 400, margin=0.02)
    check("a real regression is named as one", "REAL REGRESSION" in bad.verdict,
          bad.verdict)
    check("a real regression is not non-inferior", not bad.non_inferior)

    good = non_inferiority(0.82, 2000, 0.81, 2000, margin=0.02)
    check("with enough n, non-inferiority is established", good.non_inferior,
          good.verdict)

    diff, lo, hi = proportion_diff_ci(0.8, 100, 0.8, 100)
    check("CI brackets zero for identical rates", lo < 0 < hi, f"{lo},{hi}")
    check("required_n grows as the margin shrinks",
          required_n(0.8, 0.01) > required_n(0.8, 0.05))

    fast_but_broken = {"name": "fp8", "pass_rate": 0.55, "n": 400,
                       "ttft_p50": 0.10, "output_tok_per_s": 3000}
    base = {"name": "bf16", "pass_rate": 0.80, "n": 400,
            "ttft_p50": 0.25, "output_tok_per_s": 1200}
    v = serving_verdict(fast_but_broken, base)
    check("a 2.5x throughput gain with broken quality is REJECTED",
          v["headline"].startswith("REJECT"), v["headline"])

    fine = {"name": "fp8", "pass_rate": 0.805, "n": 3000,
            "ttft_p50": 0.10, "output_tok_per_s": 3000}
    base2 = dict(base, n=3000)
    v2 = serving_verdict(fine, base2)
    check("a genuine win is declared when quality is proven",
          v2["headline"].startswith("WIN"), v2["headline"])

    neutral = {"name": "same", "pass_rate": 0.805, "n": 3000,
               "ttft_p50": 0.249, "output_tok_per_s": 1210}
    check("no material gain yields NEUTRAL, not WIN",
          serving_verdict(neutral, base2)["headline"].startswith("NEUTRAL"))


def test_per_trap_guard():
    print("\n=== quantisation damage concentrates ===")
    base = {f"trap{i}": 0.80 for i in range(16)}
    treat = dict(base)
    treat["trap3"] = 0.55        # one rule broken hard
    treat["trap7"] = 0.58
    g = per_trap_guard(treat, base)
    check("concentrated damage is detected", g["n_regressed"] == 2, str(g))
    check("the aggregate understates it", abs(g["aggregate_delta"]) < 0.04,
          str(g["aggregate_delta"]))
    check("the verdict says so explicitly", "CONCENTRATED DAMAGE" in g["verdict"],
          g["verdict"])
    check("a clean quantisation is not accused",
          "no single rule broke" in per_trap_guard(base, base)["verdict"])


def test_three_bar():
    print("\n=== the prefix-caching three-bar report ===")
    bars = [
        {"name": "preload+cache", "prefix_hit_rate": 0.91, "ttft_p50": 0.11,
         "ttft_p99": 0.48, "output_tok_per_s": 2400, "pass_rate": 0.81,
         "n": 3000, "usd_per_resolved": 0.0141},
        {"name": "preload-nocache", "prefix_hit_rate": 0.0, "ttft_p50": 0.38,
         "ttft_p99": 1.10, "output_tok_per_s": 1500, "pass_rate": 0.810,
         "n": 3000, "usd_per_resolved": 0.0192},
        # JIT sends far fewer tokens and still loses on latency: the extra
        # search_policy round trip costs more than the cached prefill it saves
        {"name": "jit", "prefix_hit_rate": 0.22, "ttft_p50": 0.14,
         "ttft_p99": 0.72, "output_tok_per_s": 2600, "pass_rate": 0.77,
         "n": 3000, "usd_per_resolved": 0.0168},
    ]
    md = three_bar_report(bars, "preload-nocache")
    check("the table renders", "prefix hit" in md)
    check("caching is declared a win", "WIN" in md)
    check("the guard sentence is present", "is a result on its own" in md)
    v = serving_verdict(bars[2], bars[1])
    check("JIT's quality drop is caught despite its throughput gain",
          not v["headline"].startswith("WIN"), v["headline"])
    print(f"       jit vs preload-nocache: {v['headline']}")


def main() -> int:
    test_parsing()
    test_counter_diffing()
    test_histogram_quantiles()
    test_sglang_metrics()
    test_cost_inversion()
    test_break_even()
    test_quality_guard()
    test_per_trap_guard()
    test_three_bar()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
