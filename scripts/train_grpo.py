"""
GRPO on PasarBench, from round 2's fine-tunes (docs/RUNBOOK.md, phase 4, GRPO).

THE POINT OF THIS FILE
----------------------
The harness is already the RL environment: the sweep rolls out multi-turn
tool-calling episodes, the verifier scores them with no human and no reward
model, and replay re-scores any recorded episode under today's checks. GRPO
adds the group advantage and the policy update, and nothing else.

    rollouts --> verifier --> group advantage --> policy update --> rollouts

RFT FIRST. GRPO's only meaningful baseline is the rejection-sampled model, not
the base model: a GRPO number against the base model conflates "RL worked"
with "training on correct trajectories worked". So each fold's GRPO starts from
round 2's RFT adapter for that fold and trains on that fold's tasks -- the
tasks the RFT adapter was trained on, nothing new -- and the two are compared
on what neither trained on: the other fold (paired with traces/P2-rft), and
the 62 newer tasks of the five hardest traps, which no model has trained on
(base, RFT and GRPO on the same tasks, the same customer, the same checks).

ONE ITERATION (gpu_pipeline.sh grpo, PASAR_ROUND=2)
---------------------------------------------------
    tasks    once: each fold's tasks the base model failed at least once in
             round 2's collection (8 rollouts a task at temperature 1, within the
             model's own top-k and top-p, #42). A task it always solved is all
             but certain to be a group with nothing in it to learn
    collect  the current adapter served, k rollouts a task at temperature 1 from
             the full softmax (no top-k, no top-p: the sampler the loss assumes),
             conversations saved (the sweep, as RFT's collection)
    build    every episode replayed and rewarded under today's checks; in each
             task's group with a pass and a fail, the rewards standardised and
             every agent turn an example with its episode's advantage
             (pasarbench/rl/grpo.py)
    update   the servers stopped, `train_rft.py train --objective grpo --init
             <adapter>`: one pass of the clipped policy gradient

    python scripts/train_grpo.py tasks
    python scripts/train_grpo.py collect --adapter pasar-rft2-A \\
        --tasks-file data/grpo2/stats/tasks-A.json --run-id P2-grpo-i1-collect-A
    python scripts/train_grpo.py build --run traces/P2-grpo-i1-collect-A/full \\
        --served-model Qwen/Qwen3.8-27B:pasar-rft2-A --server http://localhost:8000 \\
        --out data/grpo2/examples/i1-A.jsonl --stats data/grpo2/stats/i1-A.stats.json

THE THREE THINGS THAT ACTUALLY GO WRONG
---------------------------------------
 1. Degenerate groups. If all k rollouts of a task pass, or all fail, the group
    has nothing to learn from. `build` reports their rate; above ~70%,
    rebalance the tasks, never turn up the learning rate.
 2. Loss masking across turns. Each example is one agent turn after the prompt
    the server built: tool results and the customer's words are context, never
    targets (pasarbench/rl/sft.py, as for RFT).
 3. Length bias. A turn's loss is divided by its episode's token count, so a
    long episode weighs what a short one does.

WHAT TO REPORT
  base -> RFT -> GRPO on what none of them trained on, pass^1 and pass^k, per
  trap, with tokens per resolved conversation. If GRPO does not beat RFT, say
  so: that is a finding, and at 98% on the main suite the likeliest one there.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pasarbench.harness.simulator import CUSTOMER  # noqa: E402
from pasarbench.models import AGENT_DEFAULT, SIM_DEFAULT  # noqa: E402

AGENT = os.environ.get("AGENT_MODEL") or AGENT_DEFAULT
FOLDS = "data/splits/folds.json"
STATS = Path("data/grpo2/stats")
# The sampler the update assumes: the whole softmax, at the temperature asked
# for. SGLang fills every sampling parameter a request leaves out from the
# model's generation_config.json (`--sampling-defaults model`, its default) --
# for Qwen a top-k of 20 and a top-p of 0.95 -- so a request that names only the
# temperature samples from a cut-off distribution that the loss, over the full
# one, never sees (WHAT_FAILED #42). Every one of them is set here.
ON_POLICY = {"top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
             "presence_penalty": 0.0, "frequency_penalty": 0.0}
ROLLOUTS = json.dumps({"chat_template_kwargs": {"enable_thinking": False}, **ON_POLICY})


def passes(cell: str | Path) -> dict[str, tuple[int, int]]:
    """(passed under today's checks, finished episodes) per task in one run cell."""
    from pasarbench.diagnose import load_episodes

    out: dict[str, list[bool]] = {}
    for e in load_episodes(cell, checker="current"):
        out.setdefault(e["task_id"], []).append(bool(e["passed"]))
    return {t: (sum(v), len(v)) for t, v in out.items()}


def pick(collect: dict[str, tuple[int, int]], task_fold: dict[str, str],
         fold: str) -> list[str]:
    """One fold's GRPO tasks: the fold's tasks the collection failed at least
    once. The newer tasks are never picked: they are what GRPO is judged on."""
    return sorted(t for t, (p, n) in collect.items() if task_fold.get(t) == fold and p < n)


def tasks(args) -> None:
    from pasarbench.rescore import all_tasks
    from pasarbench.rl.split import load_folds

    folds = load_folds(args.folds)
    col = passes(args.collect)
    if not col:
        raise SystemExit(f"no finished episodes in {args.collect}: GRPO picks its tasks "
                         f"from round 2's collection (stage 1)")
    every = all_tasks()
    out, short = {}, []
    for fold in args.fold or folds["folds"]:
        chosen = pick(col, folds["task_fold"], fold)
        mine = [t for t in col if folds["task_fold"].get(t) == fold]
        traps = Counter(every[t].trap for t in chosen)
        out[fold] = {"fold": fold, "source": str(args.collect),
                     "rule": "the fold's tasks the collection failed at least once",
                     "tasks": chosen, "by_trap": dict(sorted(traps.items())),
                     "collection_pass": {t: list(col[t]) for t in chosen}}
        print(f"fold {fold}: {len(chosen)} of its {len(mine)} tasks failed at least once in "
              f"{args.collect}")
        print("  by trap: " + (", ".join(f"{t} {k}" for t, k in sorted(traps.items())) or "-"))
        print(f"  {len(chosen)} tasks x k={args.k}: {len(chosen) * args.k} rollouts an iteration")
        if len(chosen) < args.min_tasks:
            short.append(fold)
    if short:
        # Nothing written: the stage would otherwise find the files next time
        # and go ahead on too few.
        raise SystemExit(f"!! fold(s) {', '.join(short)}: fewer than {args.min_tasks} tasks "
                         f"to learn from -- too few for GRPO to say anything (RUNBOOK "
                         f"phase 4, GRPO)")
    for fold, d in out.items():
        path = Path(args.out_dir) / f"tasks-{fold}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=1))
        print(f"-> {path}")


def collect(args) -> None:
    """k rollouts a task of the served adapter, conversations saved."""
    chosen = json.loads(Path(args.tasks_file).read_text())["tasks"]
    if not chosen:
        raise SystemExit(f"{args.tasks_file} lists no tasks")
    cmd = [sys.executable, "-m", "pasarbench.sweep",
           "--backend", "openai", "--model", f"{args.model}:{args.adapter}",
           "--base-url", args.base_url, "--simulator", "openai", "--sim-model", args.sim_model,
           "--sim-url", args.sim_url, "--suite", "all", "--tasks", ",".join(chosen),
           "-k", str(args.k), "--temperature", str(args.temperature), "--strategies", "full",
           "--workers", str(args.workers), "--run-id", args.run_id, "--resume",
           "--max-tokens", "2048", "--extra-body", args.extra_body,
           # the agent's flags (Qwen's enable_thinking) are not the customer's
           "--sim-extra-body", "{}",
           "--customer", str(args.customer), "--save-messages"]
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(cmd))


def build(args) -> None:
    from pasarbench.rescore import all_tasks
    from pasarbench.rl.grpo import build as build_rows, write
    from pasarbench.rl.sft import server_renderer
    from pasarbench.sweep import ALL_SOLUTIONS
    from transformers import AutoTokenizer

    from pasarbench.rl.sft import ServerRenderError
    from pasarbench.tools import schemas

    tok = AutoTokenizer.from_pretrained(args.model)
    server = None
    if args.server:
        server = server_renderer(args.server, args.model)
        try:
            # One probe with a tool list before a few thousand requests.
            server([{"role": "user", "content": "hi"}], schemas(["get_order"]), {})
        except ServerRenderError as e:
            # RFT's build falls back to the local template and `check` then
            # fails it; nothing checks between this build and the update, and
            # the local template renders none of the served prompts (WHAT_FAILED
            # #27). So no fallback.
            raise SystemExit(f"!! {e}\n!! GRPO trains on the prompts the server built: bring "
                             f"the agent server up and build again")
    rows, stats = build_rows(args.run, all_tasks(), tok, ALL_SOLUTIONS, server=server,
                             max_len=args.max_len, expect_model=args.served_model or None)
    if stats.get("prompts_ending_unlike_local_render"):
        raise SystemExit(f"!! {stats['prompts_ending_unlike_local_render']} of the server's "
                         f"prompts do not end in the generation prompt the local template "
                         f"ends in, so the turns rendered after them are not what followed "
                         f"(the server's template or chat_template_kwargs are not these). "
                         f"`train_rft.py check` shows the first; nothing written.")
    if not rows:
        raise SystemExit(f"!! nothing to train on: {stats['groups']} groups, every one with "
                         f"the same verdict throughout ({stats['bands']}). The policy has no "
                         f"pass-and-fail left on these tasks: GRPO stops here.")
    path = write(rows, stats, args.out, args.stats)
    print(f"{stats['episodes']} rollouts on {stats['groups']} tasks; the policy passed "
          f"{stats['passed']} ({stats['pass_rate']:.0%}). Groups: {stats['bands']}; "
          f"degenerate {stats['degenerate_groups']} ({stats['degenerate_rate']:.0%}). "
          f"{stats['examples']} turns from {stats['episodes_used']} rollouts -> {path}")
    if stats["skipped"]:
        print(f"  skipped: {stats['skipped']}")
    if stats["degenerate_rate"] > 0.7:
        print("!! most groups carry no learning signal: the tasks are too easy (or too "
              "hard) for this policy. Do NOT compensate with the learning rate.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tasks", help="each fold's tasks the base model failed at least once")
    t.add_argument("--fold", action="append", choices=["A", "B"],
                   help="one fold (repeatable); default both")
    t.add_argument("--folds", default=FOLDS)
    t.add_argument("--collect", default="traces/P2-collect/full",
                   help="round 2's collection: 8 rollouts a task at temperature 1")
    t.add_argument("-k", type=int, default=8, help="GRPO's rollouts per task, for the count")
    t.add_argument("--min-tasks", type=int, default=6)
    t.add_argument("--out-dir", default=str(STATS))
    t.set_defaults(fn=tasks)

    c = sub.add_parser("collect", help="k rollouts a task of the served adapter")
    c.add_argument("--adapter", required=True, help="the served adapter's name")
    c.add_argument("--tasks-file", required=True)
    c.add_argument("--run-id", required=True)
    c.add_argument("--model", default=AGENT)
    c.add_argument("--base-url", default="http://localhost:8000/v1")
    c.add_argument("--sim-model", default=os.environ.get("SIM_MODEL") or SIM_DEFAULT)
    c.add_argument("--sim-url", default="http://localhost:8001/v1")
    c.add_argument("--customer", type=int, choices=[1, 2],
                   default=int(os.environ.get("PASAR_CUSTOMER") or CUSTOMER),
                   help="the simulated customer's rules (WHAT_FAILED #38)")
    c.add_argument("-k", type=int, default=8)
    c.add_argument("--temperature", type=float, default=1.0)
    c.add_argument("--workers", type=int, default=24)
    c.add_argument("--extra-body", default=ROLLOUTS,
                   help="thinking off, and every sampling parameter at its neutral "
                        "value: the full softmax at --temperature")
    c.add_argument("--dry-run", action="store_true", help="print the command only")
    c.set_defaults(fn=collect)

    b = sub.add_parser("build", help="advantages and per-turn examples")
    b.add_argument("--run", required=True, help="the collection's cell, traces/<run>/full")
    b.add_argument("--model", default=AGENT, help="the tokenizer's model")
    b.add_argument("--served-model", default="",
                   help="the adapter the rollouts came from, as the traces name it")
    b.add_argument("--server", default="", help="agent server for the prompts' token ids")
    b.add_argument("--max-len", type=int, default=16384)
    b.add_argument("--out", required=True)
    b.add_argument("--stats", required=True)
    b.set_defaults(fn=build)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
