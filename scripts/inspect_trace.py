"""
Inspect a run's traces.

    python scripts/inspect_trace.py traces/smoke-deepseek/full
    python scripts/inspect_trace.py traces/smoke-deepseek/full --task T08

Answers, in one pass, the four questions the smoke test tells you to ask --
plus the two that actually matter after a first real run: why did a task fail,
and what will the full sweep cost.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(path: str) -> dict:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    head = next((r for r in recs if r.get("type") == "header"), {})
    foot = next((r for r in reversed(recs) if r.get("type") == "footer"), {})
    steps = [r for r in recs if r.get("type") == "step"]
    events = [r for r in recs if r.get("type") == "event"]
    return {"head": head, "foot": foot, "steps": steps, "events": events,
            "id": Path(path).stem}


def verdicts(eps: list[dict]) -> None:
    print("=" * 72)
    print("WHY EACH TASK PASSED OR FAILED")
    print("=" * 72)
    for e in eps:
        ok = e["foot"].get("passed")
        print(f"\n{e['id']:8s} {'PASS' if ok else 'FAIL'}   "
              f"trap={e['head'].get('trap', '?')}  "
              f"stop={e['foot'].get('stop_reason')}")
        for f in e["foot"].get("failures", []):
            print(f"         -> {f}")


def sequence(e: dict) -> None:
    print("\n" + "=" * 72)
    print(f"TOOL SEQUENCE: {e['id']}")
    print("=" * 72)
    for st in e["steps"]:
        n = st["step"]
        if st.get("model_content"):
            txt = " ".join(st["model_content"].split())[:110]
            print(f"  [{n}] says: {txt}")
        for tr in st.get("tool_results", []):
            mark = "ok " if tr.get("ok") else "ERR"
            args = json.dumps(tr.get("args", {}), ensure_ascii=False)[:90]
            print(f"  [{n}] {mark} {tr['name']}{args}")
            if not tr.get("ok"):
                print(f"         {tr.get('error')}")
    for ev in e["events"]:
        if ev.get("kind") == "user_turn" and ev.get("text"):
            print(f"  customer: {' '.join(ev['text'].split())[:110]}")


def empties(eps: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("2. EMPTY ASSISTANT TURNS  (max_tokens exhausted, or thinking still on)")
    print("=" * 72)
    bad = 0
    for e in eps:
        for st in e["steps"]:
            if not st.get("model_content") and not st.get("tool_calls"):
                bad += 1
                print(f"  {e['id']} step {st['step']}: no content, no tool call")
    print("  none" if not bad else
          f"  {bad} empty turn(s) -- raise --max-tokens, and confirm "
          f"--extra-body disabled thinking")


def leaks(eps: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("3. DID THE SIMULATOR LEAK FACTS BEFORE BEING ASKED?")
    print("=" * 72)
    try:
        from pasarbench.generate import generate
        from pasarbench.tasks import BY_ID
        tasks = {**BY_ID, **{t.task_id: t for t in generate()[0]}}
    except Exception as ex:                                  # noqa: BLE001
        print(f"  (could not load tasks: {ex})")
        return

    leaked = 0
    for e in eps:
        task = tasks.get(e["id"])
        if not task:
            continue
        turns = [ev.get("text", "") for ev in e["events"]
                 if ev.get("kind") == "user_turn"]
        opening = task.opening
        first = [opening] + turns[:1]
        for fact, value in task.hidden_facts.items():
            if any(str(value).lower() in (t or "").lower() for t in first):
                leaked += 1
                print(f"  {e['id']}: volunteered `{fact}` = {value} unprompted")
    print("  none -- the customer made the agent ask" if not leaked else
          f"  {leaked} leak(s). Above ~15% of episodes this invalidates every "
          f"pass rate: the benchmark has become single-turn.")


def cache_and_cost(eps: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("4. TOKENS, CACHE, AND WHAT THE FULL SWEEP WILL COST")
    print("=" * 72)
    prompt = completion = cached = 0
    for e in eps:
        for st in e["steps"]:
            u = st.get("usage", {})
            prompt += u.get("prompt", 0)
            completion += u.get("completion", 0)
            cached += u.get("cached", 0)
    n = len(eps) or 1
    total = prompt + completion
    print(f"  episodes        {n}")
    print(f"  prompt tokens   {prompt:,}   ({prompt / n:,.0f} per episode)")
    print(f"  completion      {completion:,}   ({completion / n:,.0f} per episode)")
    if cached:
        print(f"  cached prompt   {cached:,}  -> hit rate {cached / prompt:.1%}")
        print("     the ~2.2k policy prefix is identical every call, so this")
        print("     should be high. This is your prefix-caching result, free.")
    else:
        print("  cached prompt   not reported by this provider/run")

    ratio = completion / prompt if prompt else 0
    print(f"\n  completion/prompt ratio {ratio:.3f}")
    if ratio > 0.25:
        print("     HIGH. On a tool-calling workload most output is short JSON.")
        print("     A ratio this high suggests reasoning tokens are still being")
        print("     generated -- verify --extra-body reached the request.")
    else:
        print("     normal for tool calling; thinking looks genuinely off")

    per_ep = total / n
    print(f"\n  PROJECTION at {per_ep:,.0f} tokens/episode")
    for label, eps_n in (("smoke (5 tasks, k=1)", 5),
                         ("--sample 2, 5 strategies, k=3", 32 * 5 * 3),
                         ("full 186, 5 strategies, k=3", 186 * 5 * 3),
                         ("full 186, 5 strategies, k=5", 186 * 5 * 5)):
        t = per_ep * eps_n
        print(f"    {label:32s} {eps_n:>5} episodes  {t / 1e6:>7.1f}M tokens")
    print("\n  Multiply by your provider's per-Mtok price. Subsample while")
    print("  iterating; run the full grid only for final numbers.")


def growth(eps: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("WHY TOKENS PER EPISODE ARE HIGH (full-context re-sends everything)")
    print("=" * 72)
    e = max(eps, key=lambda x: len(x["steps"]))
    print(f"  longest episode: {e['id']} ({len(e['steps'])} steps)")
    for st in e["steps"]:
        u = st.get("usage", {})
        print(f"    step {st['step']:>2}  prompt {u.get('prompt', 0):>7,}  "
              f"completion {u.get('completion', 0):>6,}  "
              f"tools exposed {st.get('n_tools', 0)}")
    print("\n  Each step re-sends the entire history plus the policy. Cost grows")
    print("  quadratically in turns, which is exactly what the week-4 context")
    print("  ablation measures -- `trim3` and `window8` attack this directly.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--task", default="", help="show the full tool sequence for one task")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.trace_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no traces in {args.trace_dir}")
    eps = [load(f) for f in files]

    verdicts(eps)
    if args.task:
        e = next((x for x in eps if x["id"] == args.task), None)
        if e:
            sequence(e)
    else:
        for e in eps:
            if not e["foot"].get("passed"):
                sequence(e)

    empties(eps)
    leaks(eps)
    cache_and_cost(eps)
    growth(eps)


if __name__ == "__main__":
    main()
