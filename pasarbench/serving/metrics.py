"""
vLLM metrics.

Three things go wrong when people measure a serving change, and all three
produce a number that looks fine and is not.

1. READING GAUGES INSTEAD OF DIFFING COUNTERS.
   `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` are
   cumulative counters over the server's lifetime. Reading them once at the end
   of a run gives you the hit rate since the process started, which includes
   every experiment you ran before this one. Snapshot before, snapshot after,
   subtract. `MetricsSnapshot.diff` exists because this mistake is easy and
   invisible.

2. NOT WARMING UP.
   The first request pays model load, CUDA graph capture and an empty cache.
   Including it destroys p99 and flatters every subsequent configuration by
   comparison. `warmup_then_measure` makes the cold/warm split explicit rather
   than leaving it to whoever reads the numbers later.

3. MISREADING PREFIX CACHE HITS ON A SEQUENTIAL RUN.
   Episodes in this benchmark share a ~2.2k-token policy prefix, so running
   them back to back gives a spectacular hit rate that says almost nothing
   about production, where requests from different tenants interleave. Report
   the run pattern alongside the number, and ideally measure both.

Histogram quantiles are computed from Prometheus buckets by linear
interpolation, the same way `histogram_quantile()` does it server-side. The
result is bucket-resolution-limited: with a bucket edge at 1.0s and the next at
2.5s, a "p99 = 1.8s" is an interpolation, not a measurement. Say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+|NaN|[-+]Inf)\s*$"
)


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    """Parse the exposition format. Comments and malformed lines are skipped."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE_RE.match(line)
        if not m:
            continue
        raw = m.group("value")
        if raw == "NaN":
            value = float("nan")
        elif raw.endswith("Inf"):
            value = float("inf") if not raw.startswith("-") else float("-inf")
        else:
            value = float(raw)
        labels = {}
        if m.group("labels"):
            for part in re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group("labels")):
                labels[part[0]] = part[1]
        out.append({"name": m.group("name"), "labels": labels, "value": value})
    return out


@dataclass
class MetricsSnapshot:
    samples: list[dict[str, Any]] = field(default_factory=list)
    label: str = ""

    @classmethod
    def from_text(cls, text: str, label: str = "") -> "MetricsSnapshot":
        return cls(parse_prometheus(text), label)

    def value(self, name: str, **labels: str) -> float:
        total = 0.0
        for s in self.samples:
            if s["name"] != name:
                continue
            if all(s["labels"].get(k) == v for k, v in labels.items()):
                total += s["value"]
        return total

    def buckets(self, name: str) -> list[tuple[float, float]]:
        """Cumulative (le, count) pairs for a histogram, sorted by le."""
        out = []
        for s in self.samples:
            if s["name"] != f"{name}_bucket":
                continue
            le = s["labels"].get("le")
            if le is None:
                continue
            out.append((float("inf") if le == "+Inf" else float(le), s["value"]))
        return sorted(out)

    def diff(self, later: "MetricsSnapshot") -> "MetricsDiff":
        return MetricsDiff(before=self, after=later)


@dataclass
class MetricsDiff:
    before: MetricsSnapshot
    after: MetricsSnapshot

    def counter(self, name: str, **labels: str) -> float:
        d = self.after.value(name, **labels) - self.before.value(name, **labels)
        # A negative delta means the server restarted mid-run and the counter
        # reset. Silently clamping would hide a run you must discard.
        if d < 0:
            raise ValueError(
                f"counter {name!r} decreased ({d:.1f}); the server restarted "
                f"during this run -- discard these numbers and re-measure")
        return d

    def bucket_diff(self, name: str) -> list[tuple[float, float]]:
        b0 = dict(self.before.buckets(name))
        return [(le, c - b0.get(le, 0.0)) for le, c in self.after.buckets(name)]

    # ---- the numbers worth reporting ------------------------------------

    def prefix_cache_hit_rate(self) -> dict[str, Any]:
        """Over THIS run only, from counter deltas.

        vLLM has used more than one metric name across versions; try each and
        report which one was found so the number is reproducible.
        """
        for q, h in (("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"),
                     ("vllm:gpu_prefix_cache_queries_total",
                      "vllm:gpu_prefix_cache_hits_total")):
            try:
                queries, hits = self.counter(q), self.counter(h)
            except ValueError:
                raise
            if queries > 0:
                return {"hit_rate": round(hits / queries, 4),
                        "queries": int(queries), "hits": int(hits), "metric": q}
        return {"hit_rate": None, "queries": 0, "hits": 0, "metric": None,
                "note": "no prefix-cache counters found; is --enable-prefix-caching on?"}

    def latency(self, name: str = "vllm:time_to_first_token_seconds"
                ) -> dict[str, Any]:
        b = self.bucket_diff(name)
        total = b[-1][1] if b else 0.0
        if total <= 0:
            return {"n": 0, "p50": None, "p90": None, "p99": None, "metric": name}
        return {
            "n": int(total),
            "p50": histogram_quantile(b, 0.50),
            "p90": histogram_quantile(b, 0.90),
            "p99": histogram_quantile(b, 0.99),
            "metric": name,
            "resolution_note": _resolution_note(b),
        }

    def throughput(self, wall_seconds: float) -> dict[str, Any]:
        prompt = self.counter("vllm:prompt_tokens_total")
        gen = self.counter("vllm:generation_tokens_total")
        return {
            "prompt_tokens": int(prompt), "generation_tokens": int(gen),
            "prompt_tok_per_s": round(prompt / wall_seconds, 1) if wall_seconds else 0,
            "output_tok_per_s": round(gen / wall_seconds, 1) if wall_seconds else 0,
            "wall_seconds": round(wall_seconds, 2),
        }


def histogram_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus-style interpolation over cumulative buckets."""
    if not buckets:
        return None
    buckets = sorted(buckets)
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if le == float("inf"):
                # Everything above the last finite edge. Prometheus returns the
                # last finite bound; report it and let the caller know the
                # histogram is under-bucketed at the tail.
                return prev_le or None
            span = count - prev_count
            if span <= 0:
                return round(le, 4)
            frac = (target - prev_count) / span
            return round(prev_le + frac * (le - prev_le), 4)
        prev_le, prev_count = le, count
    return round(buckets[-1][0], 4)


def _resolution_note(buckets: list[tuple[float, float]]) -> str:
    edges = [le for le, _ in buckets if le != float("inf")]
    if len(edges) < 2:
        return "too few buckets to interpolate meaningfully"
    widest = max(b - a for a, b in zip(edges, edges[1:]))
    return (f"bucket-limited: widest gap {widest:.2f}s, so tail percentiles are "
            f"interpolations rather than measurements")


# --------------------------------------------------------------------------

def warmup_then_measure(fetch_metrics, run_once, warmup: int = 3
                        ) -> dict[str, Any]:
    """Run `warmup` episodes discarded, snapshot, then measure.

    `fetch_metrics()` returns the /metrics text; `run_once()` runs one episode.
    Keeping the warmup explicit is the point -- a cold first request pays model
    load, CUDA graph capture and an empty cache, and folding it into p99
    flatters whatever you measure next.
    """
    import time
    for _ in range(warmup):
        run_once()
    before = MetricsSnapshot.from_text(fetch_metrics(), "before")
    t0 = time.monotonic()
    result = run_once()
    wall = time.monotonic() - t0
    after = MetricsSnapshot.from_text(fetch_metrics(), "after")
    d = before.diff(after)
    return {"result": result, "wall_seconds": wall,
            "prefix_cache": d.prefix_cache_hit_rate(),
            "ttft": d.latency(),
            "throughput": d.throughput(wall),
            "warmup_episodes": warmup}
