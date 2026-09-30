"""
How the agent and the simulated customer share this machine's GPUs.

    python scripts/gpu_plan.py show             # the plan in words (each stage logs it)
    python scripts/gpu_plan.py layout           # split | shared; exit 2 if nothing fits
    python scripts/gpu_plan.py count            # number of GPUs
    python scripts/gpu_plan.py fraction agent   # SGLang --mem-fraction-static, right now
    python scripts/gpu_plan.py fraction sim     #   ...for the customer, once the agent is up
    python scripts/gpu_plan.py price            # $/h of this machine as a Modal job

TWO OR MORE CARDS ("split"): the agent on GPU 0, the customer on GPU 1, each
with SGLang's own memory defaults. Training uses every card.

ONE CARD ("shared"): both servers on GPU 0 -- a B200 (180 GB) holds the agent's
52 GiB of bf16 weights, the customer's 31 GiB of FP8 ones and a KV cache for
each; an H200 (141 GB) does with less room; an 80 GB card cannot. SGLang sizes
its KV cache from the memory free when the server starts:

    pool = free_before_load * f - weights    (f = --mem-fraction-static)

and leaves free_before_load * (1 - f) for CUDA graphs and activations. Because f
is a fraction of what is free at that moment, not of the card, the two servers
start one after the other: the agent first, with f chosen to give it a planned
pool; then the customer, with f chosen to leave RESERVE GiB free for both
servers' runtime. Each fraction is computed at start-up from nvidia-smi, so the
plan holds on any card size. Overrides: PASAR_AGENT_POOL_GB (the agent's pool),
PASAR_RESERVE_GB (the memory left free for both servers' working memory once
both are up, 10 by default: raise it after an out-of-memory error at run time),
MEM_FRACTION (the fraction itself, for one server started by hand), PASAR_LAYOUT.

Tests set PASAR_GPU_INFO="NAME,TOTAL_MIB,FREE_MIB;..." instead of asking nvidia-smi.
Standard library only: it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

AGENT_DEFAULT = "Qwen/Qwen3.8-27B"
SIM_DEFAULT = "RedHatAI/gemma-4-31B-it-FP8-Dynamic"
# Weights as SGLang loads them (GiB), for when they are not downloaded yet:
# the safetensors of each checkpoint, vision tower included.
KNOWN_WEIGHTS_GIB = {AGENT_DEFAULT: 51.8, SIM_DEFAULT: 30.7}

CTX_GIB = 0.75       # a new process's CUDA context, before SGLang measures free memory
LOAD_GIB = 1.0       # buffers allocated while loading, besides the weights
RUNTIME_GIB = 6.0    # one server's CUDA graphs and activations (batch and chunk capped)
MARGIN_GIB = 4.0     # never planned; absorbs growth under load
MIN_POOL_GIB = 12.0  # below this a server cannot keep 24 conversations going
MAX_POOL_GIB = {"agent": 48.0, "sim": 64.0}
# Half the spare memory each. The agent's pool holds 64 KiB of KV per token plus
# ~150 MiB of linear-attention state per running conversation, and SGLang caps
# the agent's concurrency by how many of those states fit; the customer's cache
# costs ~330 KiB per token (48 sliding-window layers kept to their window).
AGENT_SHARE = 0.5


def reserve_gib() -> float:
    """Left free once both servers are up: the customer's CUDA graphs and
    activations, and whatever either server grows by under load."""
    return float(os.environ.get("PASAR_RESERVE_GB") or RUNTIME_GIB + MARGIN_GIB)

# Modal's per-GPU prices ($/h, Sep 2026), and the host a stage job asks for
# (8 cores, 32 GiB). A notebook's CPU and memory cost about three times as much.
PRICE = {"B300": 7.10, "B200": 6.25, "H200": 4.54, "H100": 3.95, "A100": 2.50,
         "RTX PRO 6000": 3.03, "L40S": 1.95}
HOST_PER_HOUR = 8 * 0.0472 + 32 * 0.0080


def gpus() -> list[dict]:
    """[{name, total_gib, free_gib}] for every GPU this process can see."""
    raw = os.environ.get("PASAR_GPU_INFO")
    if raw is None:
        try:
            raw = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return []
        raw = ";".join(raw.strip().splitlines())
    out = []
    for item in filter(None, (x.strip() for x in raw.split(";"))):
        name, total, free = (x.strip() for x in item.rsplit(",", 2))
        out.append({"name": name, "total_gib": float(total) / 1024,
                    "free_gib": float(free) / 1024})
    return out


def _hub() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def weights_gib(model: str) -> float:
    """Size of a checkpoint's safetensors: a local directory, the Hugging Face
    cache, or the table above."""
    local = Path(model)
    if local.is_dir():
        files = list(local.glob("*.safetensors"))
    else:
        snaps = sorted((_hub() / f"models--{model.replace('/', '--')}" / "snapshots").glob("*"),
                       key=lambda p: p.stat().st_mtime)
        files = list(snaps[-1].glob("*.safetensors")) if snaps else []
    if files:
        return sum(f.stat().st_size for f in files) / 2**30
    if model in KNOWN_WEIGHTS_GIB:
        return KNOWN_WEIGHTS_GIB[model]
    raise SystemExit(f"no weights for {model} in {_hub()}: download them first "
                     f"(scripts/download_weights.sh)")


def models() -> tuple[str, str]:
    return (os.environ.get("AGENT_MODEL") or AGENT_DEFAULT,
            os.environ.get("SIM_MODEL") or SIM_DEFAULT)


def plan(cards: list[dict], w_agent: float, w_sim: float) -> dict:
    """The layout, and for one card the pools it leaves each server."""
    forced = os.environ.get("PASAR_LAYOUT", "")
    if not cards:
        return {"layout": "none", "why": "no GPU visible (nvidia-smi found none)"}
    if len(cards) >= 2 and forced != "shared":
        need_agent = w_agent + LOAD_GIB + MIN_POOL_GIB + RUNTIME_GIB + CTX_GIB
        need_sim = w_sim + LOAD_GIB + MIN_POOL_GIB + RUNTIME_GIB + CTX_GIB
        if cards[0]["total_gib"] < need_agent or cards[1]["total_gib"] < need_sim:
            return {"layout": "too-small", "why": (
                f"{len(cards)} x {cards[0]['name']} ({cards[0]['total_gib']:.0f} GiB): the agent "
                f"needs ~{need_agent:.0f} GiB on GPU 0 and the customer ~{need_sim:.0f} on "
                f"GPU 1. Use 80 GB cards, or one H200 or B200.")}
        return {"layout": "split", "why": f"{len(cards)} GPUs: the agent on GPU 0, "
                                          f"the customer on GPU 1"}
    if forced == "split":
        return {"layout": "too-small", "why": "PASAR_LAYOUT=split needs two GPUs"}
    total = cards[0]["total_gib"]
    # The agent's weights, context, load buffers and working memory; the
    # customer's weights, context and load buffers; the reserve (which holds the
    # customer's working memory); what is left is the two caches.
    spare = (total - w_agent - w_sim - (RUNTIME_GIB + CTX_GIB + LOAD_GIB)
             - (CTX_GIB + LOAD_GIB) - reserve_gib())
    if spare < 2 * MIN_POOL_GIB:
        need = total - spare + 2 * MIN_POOL_GIB
        return {"layout": "too-small", "why": (
            f"one {cards[0]['name']} ({total:.0f} GiB) cannot hold the agent ({w_agent:.0f} GiB) "
            f"and the customer ({w_sim:.0f} GiB) with room to run: that takes about "
            f"{need:.0f} GiB. Use one H200 or B200, or two 80 GB cards.")}
    pool_agent = float(os.environ.get("PASAR_AGENT_POOL_GB") or
                       min(max(AGENT_SHARE * spare, MIN_POOL_GIB), MAX_POOL_GIB["agent"]))
    pool_sim = min(spare - pool_agent, MAX_POOL_GIB["sim"])
    if pool_sim < MIN_POOL_GIB:
        return {"layout": "too-small", "why": (
            f"PASAR_AGENT_POOL_GB={pool_agent:.0f} leaves the customer {pool_sim:.0f} GiB "
            f"of cache on this {total:.0f} GiB card; at most "
            f"{spare - MIN_POOL_GIB:.0f} fits")}
    return {"layout": "shared", "total": total, "spare": spare,
            "pool_agent": pool_agent, "pool_sim": pool_sim,
            "why": f"one {cards[0]['name']} ({total:.0f} GiB): both servers on it"}


def fraction(role: str, card: dict, w_agent: float, w_sim: float) -> float:
    """--mem-fraction-static for the server about to start on `card`, from the
    memory free on it now (see the module docstring)."""
    p = plan([card], w_agent, w_sim)
    if p["layout"] != "shared":
        raise SystemExit(p["why"])
    pre = card["free_gib"] - CTX_GIB
    if role == "agent":
        static = w_agent + LOAD_GIB + p["pool_agent"]
        # Room for the agent, its working memory, and the customer after it.
        needed = (static + RUNTIME_GIB + CTX_GIB + w_sim + LOAD_GIB + MIN_POOL_GIB
                  + reserve_gib())
        if needed > pre:
            raise SystemExit(
                f"only {card['free_gib']:.0f} GiB free on the GPU, and the agent and the "
                f"customer need ~{needed + CTX_GIB:.0f}: is another process using it? "
                f"(nvidia-smi; in a notebook, pn.stop())")
    else:
        pool = pre - reserve_gib() - w_sim - LOAD_GIB
        if pool < MIN_POOL_GIB:
            raise SystemExit(
                f"only {card['free_gib']:.0f} GiB left for the customer after the agent: "
                f"a {pool:.0f} GiB cache is too small. Lower PASAR_AGENT_POOL_GB "
                f"(the plan gave the agent {p['pool_agent']:.0f} GiB).")
        static = w_sim + LOAD_GIB + min(pool, MAX_POOL_GIB["sim"])
    return round(min(max(static / pre, 0.05), 0.95), 3)


def price(cards: list[dict]) -> float:
    if os.environ.get("PASAR_USD_PER_HOUR"):
        return float(os.environ["PASAR_USD_PER_HOUR"])
    per = 0.0
    for c in cards:
        per += next((v for k, v in PRICE.items() if k in c["name"].upper()), 4.0)
    return per + HOST_PER_HOUR


def show(cards: list[dict], w_agent: float, w_sim: float) -> str:
    agent, sim = models()
    p = plan(cards, w_agent, w_sim)
    lines = [f"GPUs: " + (", ".join(f"{c['name']} ({c['total_gib']:.0f} GiB, "
                                     f"{c['free_gib']:.0f} free)" for c in cards) or "none"),
             f"agent {agent} ({w_agent:.1f} GiB) | customer {sim} ({w_sim:.1f} GiB)",
             f"layout: {p['layout']} -- {p['why']}"]
    if p["layout"] == "shared":
        lines.append(f"  agent KV+state pool ~{p['pool_agent']:.0f} GiB, customer KV pool up to "
                     f"~{p['pool_sim']:.0f} GiB, {reserve_gib():.0f} GiB kept free for runtime; "
                     f"training in one process, same effective batch")
    if cards:
        lines.append(f"  ~${price(cards):.2f}/h as a Modal job (a notebook adds ~$1-2/h for "
                     f"its CPU and memory)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["show", "layout", "count", "fraction", "price"])
    ap.add_argument("role", nargs="?", choices=["agent", "sim"])
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi index (fraction)")
    a = ap.parse_args(argv)
    cards = gpus()
    if a.what == "count":
        print(len(cards))
        return 0
    if a.what == "price":
        print(f"{price(cards):.2f}")
        return 0
    agent, sim = models()
    w_agent, w_sim = weights_gib(agent), weights_gib(sim)
    if a.what == "show":
        print(show(cards, w_agent, w_sim))
        return 0
    if a.what == "layout":
        p = plan(cards, w_agent, w_sim)
        if p["layout"] in ("split", "shared"):
            print(p["layout"])
            return 0
        print(f"!! {p['why']}", file=sys.stderr)
        return 2
    if not a.role:
        ap.error("fraction needs a role: agent or sim")
    if a.gpu >= len(cards):
        raise SystemExit(f"no GPU {a.gpu} (found {len(cards)})")
    f = fraction(a.role, cards[a.gpu], w_agent, w_sim)
    print(f"{a.role}: --mem-fraction-static {f} ({cards[a.gpu]['free_gib']:.0f} GiB free now)",
          file=sys.stderr)
    print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
