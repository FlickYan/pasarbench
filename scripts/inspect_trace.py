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
    events = {ev["turn"]: ev for ev in e["events"] if ev.get("kind") == "user_turn"}
    turn = 0
    print(f"  customer: {e.get('opening', '(opening)')}")
    for st in e["steps"]:
        n = st["step"]
        if st.get("model_content"):
            txt = " ".join(st["model_content"].split())[:110]
            print(f"  [{n}] agent: {txt}")
        for tr in st.get("tool_results", []):
            mark = "ok " if tr.get("ok") else "ERR"
            args = json.dumps(tr.get("args", {}), ensure_ascii=False)[:90]
            print(f"  [{n}]   {mark} {tr['name']}{args}")
            if not tr.get("ok"):
                print(f"           {tr.get('error')}")
        if not st.get("tool_calls"):
            ev = events.get(turn)
            if ev and ev.get("text"):
                print(f"  customer: {' '.join(ev['text'].split())[:110]}")
            turn += 1


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

    from pasarbench.harness.types import Message
    from pasarbench.simqa import audit, leak_report

    reports = []
    for e in eps:
        task = tasks.get(e["id"])
        if not task:
            continue
        # Rebuild the transcript IN ORDER. A fact given after the agent asked
        # for it is correct behaviour; only an unprompted one is a leak, and
        # you cannot tell the two apart without the interleaving.
        msgs = [Message("system", ""), Message("user", task.opening)]
        events = {ev["turn"]: ev for ev in e["events"]
                  if ev.get("kind") == "user_turn"}
        turn = 0
        for st in e["steps"]:
            if st.get("model_content"):
                msgs.append(Message("assistant", st["model_content"]))
            if not st.get("tool_calls"):
                ev = events.get(turn)
                if ev and ev.get("text"):
                    msgs.append(Message("user", ev["text"]))
                turn += 1
        reports.append(leak_report(task, msgs))

    if not reports:
        print("  (no matching tasks)")
        return
    a = audit(reports)
    on_request = sum(len(r.revealed_on_request) for r in reports)
    print(f"  episodes {a['transcripts']}   leak rate {a['leak_rate']:.1%}   "
          f"clean {a['clean_rate']:.1%}")
    print(f"  facts revealed ON REQUEST (correct behaviour): {on_request}")
    for r in reports:
        for fact, t in r.leaked:
            print(f"  LEAK {r.task_id}: volunteered `{fact}` at turn {t}, "
                  f"before the agent asked")
    if a["leak_rate"] == 0:
        print("  no leaks -- the customer made the agent ask for everything")
    elif a["leak_rate"] > 0.15:
        print("  above 15%: the benchmark has drifted toward single-turn. "
              "Strengthen the persona prompt before trusting any pass rate.")


def cache_and_cost(eps: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("4. TOKENS, CACHE, AND WHAT THE FULL SWEEP WILL COST")
    print("=" * 72)
    prompt = completion = cached = 0
    sim_prompt = sim_completion = sim_cached = 0
    for e in eps:
        for st in e["steps"]:
            u = st.get("usage", {})
            prompt += u.get("prompt", 0)
            completion += u.get("completion", 0)
            cached += u.get("cached", 0)
        # The simulator accumulates, so the LAST event carries the episode total.
        last = None
        for ev in e["events"]:
            if ev.get("kind") == "user_turn" and ev.get("sim_usage"):
                last = ev["sim_usage"]
        if last:
            sim_prompt += last.get("prompt", 0)
            sim_completion += last.get("completion", 0)
            sim_cached += last.get("cached", 0)
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
    if sim_prompt:
        print(f"\n  SIMULATOR (separate provider, separate price)")
        print(f"    prompt      {sim_prompt:,}   ({sim_prompt / n:,.0f} per episode)")
        print(f"    completion  {sim_completion:,}   ({sim_completion / n:,.0f} per episode)")
        share = sim_prompt / (prompt + sim_prompt)
        print(f"    {share:.0%} of all prompt tokens in the run")
    else:
        print("\n  SIMULATOR tokens not recorded (scripted simulator, or an "
              "older run). Cost below is AGENT ONLY and understates the bill.")

    hit = cached / prompt if prompt else 0.0
    print("\n  COST -- the cache discount dominates, so raw token counts")
    print("  overstate the bill badly. Rates below are ILLUSTRATIVE; check your")
    print("  provider's current pricing page and re-run with --price.")
    # USD per 1M tokens, rechecked 2026-09-15. VERIFY before quoting: these
    # changed twice in the last two months.
    #
    # DeepSeek bills PEAK vs OFF-PEAK. Peak is 01:00-04:00 and 06:00-10:00 UTC;
    # every other hour is half price. Scheduling a long sweep outside those
    # windows halves the agent bill for free -- and Singapore is UTC+8, so peak
    # is 09:00-12:00 and 14:00-18:00 local. Run overnight.
    PRICES = {
        "deepseek-flash":   {"off": (0.003, 0.15, 0.60), "peak": (0.006, 0.30, 1.20)},
        "deepseek-v4-pro":  {"off": (0.022, 0.66, 1.98), "peak": (0.044, 1.32, 3.96)},
    }
    served = ""
    for e in eps:
        served = e["head"].get("requested_model") or served
    band = PRICES.get(served, PRICES["deepseek-flash"])
    p_cached, p_in, p_out = band["off"]
    print(f"\n  agent priced as {served or 'deepseek-flash'} at OFF-PEAK rates "
          f"(peak is 2x)")
    s_cached, s_in, s_out = 0.016, 0.15, 0.47      # qwen3.8-flash, Singapore
    s_hit = sim_cached / sim_prompt if sim_prompt else 0.0
    for label, eps_n in (("--sample 2, 5 strategies, k=3", 32 * 5 * 3),
                         ("full 186, 5 strategies, k=3", 186 * 5 * 3),
                         ("full 186, 5 strategies, k=5", 186 * 5 * 5)):
        pr, co = prompt / n * eps_n, completion / n * eps_n
        agent = ((pr * hit) / 1e6 * p_cached + (pr * (1 - hit)) / 1e6 * p_in
                 + co / 1e6 * p_out)
        spr, sco = sim_prompt / n * eps_n, sim_completion / n * eps_n
        sim = ((spr * s_hit) / 1e6 * s_cached + (spr * (1 - s_hit)) / 1e6 * s_in
               + sco / 1e6 * s_out)
        print(f"    {label:32s} agent ${agent:6.2f} + sim ${sim:5.2f} "
              f"= ${agent + sim:6.2f}")


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
