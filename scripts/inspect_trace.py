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
            "id": Path(path).stem,
            # filenames carry a seed suffix (T01__r2), so task lookups must use
            # the header rather than the stem
            "task_id": head.get("task_id", Path(path).stem.split("__")[0]),
            "language": head.get("language", "?"),
            "trap": head.get("trap", "?"),
            "passed": bool(foot.get("passed"))}


def config(eps: list[dict]) -> None:
    """Print the run configuration FIRST.

    Comparing two runs that differ in more than one variable is the fastest way
    to a confident wrong conclusion. Making the config visible at the top of
    every inspection is the cheapest defence against it.
    """
    h = eps[0]["head"]
    print("=" * 72)
    print("RUN CONFIGURATION -- check this before comparing against another run")
    print("=" * 72)
    for k in ("backend", "requested_model", "simulator", "context",
              "policy_mode", "exposure"):
        if h.get(k) is not None:
            print(f"  {k:18s} {h[k]}")
    served = {e["head"].get("served_model") for e in eps} - {None}
    if served:
        print(f"  {'served_model':18s} {', '.join(sorted(served))}")
    n_per_task = {}
    for e in eps:
        n_per_task[e["id"]] = n_per_task.get(e["id"], 0) + 1
    print(f"  {'tasks':18s} {len(n_per_task)}")
    print(f"  {'episodes':18s} {len(eps)}")
    print("\n  If ANY line above differs from the run you are comparing to, the")
    print("  comparison is confounded. Change one variable at a time.\n")


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
    # Events are numbered from 1: the loop increments tracker.turns BEFORE
    # writing the event. Counting from 0 here shifted every customer line one
    # step later and made the agent look clairvoyant.
    events = {ev["turn"]: ev for ev in e["events"] if ev.get("kind") == "user_turn"}
    turn = 1
    print(f"  customer: {e.get('opening', '(opening message)')}")
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


def reliability(eps: list[dict]) -> None:
    """Split tasks into always-pass, FLAKY, and always-fail.

    The gap between pass^1 and pass^k lives entirely in the flaky bucket, and
    that bucket is the finding: a task the agent gets right 3 times in 5 is a
    conversation you cannot predict. For a CS org that is often worse than a
    task it fails every time, because consistent failure can be routed around.
    """
    from collections import defaultdict
    by_task = defaultdict(list)
    trap = {}
    for e in eps:
        tid = e["task_id"]
        by_task[tid].append(bool(e["foot"].get("passed")))
        trap[tid] = e["head"].get("trap", "?")

    k = max(len(v) for v in by_task.values())
    if k < 2:
        return
    always = [t for t, v in by_task.items() if all(v)]
    never = [t for t, v in by_task.items() if not any(v)]
    flaky = [t for t, v in by_task.items() if any(v) and not all(v)]

    print("\n" + "=" * 72)
    print(f"RELIABILITY  (k={k})  -- where pass^1 and pass^k diverge")
    print("=" * 72)
    n = len(by_task)
    print(f"  always pass  {len(always):>4}/{n}  ({len(always)/n:.1%})")
    print(f"  FLAKY        {len(flaky):>4}/{n}  ({len(flaky)/n:.1%})  <- the finding")
    print(f"  always fail  {len(never):>4}/{n}  ({len(never)/n:.1%})")

    by_trap = defaultdict(lambda: [0, 0])
    for t in flaky:
        by_trap[trap[t]][0] += 1
    for t in by_task:
        by_trap[trap[t]][1] += 1
    print("\n  flaky tasks by trap (a trap that is never flaky is understood;")
    print("  one that is often flaky is where the policy is being half-applied):")
    for tr, (f, tot) in sorted(by_trap.items(), key=lambda kv: -kv[1][0]):
        if f:
            print(f"    {f:>3}/{tot:<3} {tr}")

    if never:
        print(f"\n  never passes ({len(never)}): {', '.join(sorted(never)[:10])}")
        print("    zero variance means zero learning signal -- these contribute")
        print("    nothing to RFT and will still fail after training.")


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


# Chat-format tokens that reached the text: `<|turn>`, `<channel|>`, `<|im_end|>`,
# a `<think>` block, or a `<tool_call>` the server did not parse. SGLang keeps
# Gemma 4's special tokens in the decoded text for its parsers, so a customer
# served without the right one hands the agent "<|channel>thought...", and every
# audit below reads it as speech. An unparsed tool call on the agent's side is
# worse: the harness sends it to the customer as a message.
MARKUP = re.compile(r"<\|[a-z_\"]+\|?>|<[a-z_]+\|>|</?think>|</?tool_call>")


def markup(eps: list[dict]) -> int:
    """Turns on either side that contain chat-format tokens; prints examples."""
    print("\n" + "=" * 72)
    print("0. CHAT-FORMAT TOKENS IN THE TEXT  (a server parser missing or wrong)")
    print("=" * 72)
    sides = {
        "customer": [(e, ev.get("text") or "") for e in eps for ev in e["events"]
                     if ev.get("kind") == "user_turn"],
        "agent": [(e, st.get("model_content") or "") for e in eps for st in e["steps"]],
    }
    total = 0
    for side, rows in sides.items():
        bad = [(e, t, m) for e, t in rows for m in [MARKUP.search(t)] if m]
        total += len(bad)
        if not bad:
            print(f"  {side}: none in {len(rows)} turns")
            continue
        print(f"  !! {side}: {len(bad)} of {len(rows)} turns contain chat-format tokens. "
              f"Fix the server before trusting anything else in this report.")
        for e, t, m in bad[:3]:
            print(f"     {e['id']}: {m.group()!r} in {t[max(0, m.start() - 40):m.end() + 60]!r}")
    return total


def transcript(e: dict, task) -> list:
    """Rebuild one episode's conversation IN ORDER. A fact given after the
    agent asked for it is correct behaviour; only an unprompted one is a leak,
    and you cannot tell the two apart without the interleaving."""
    from pasarbench.harness.types import Message
    msgs = [Message("system", ""), Message("user", task.opening)]
    events = {ev["turn"]: ev for ev in e["events"] if ev.get("kind") == "user_turn"}
    turn = 1
    for st in e["steps"]:
        if st.get("model_content"):
            msgs.append(Message("assistant", st["model_content"]))
        if not st.get("tool_calls"):
            ev = events.get(turn)
            if ev and ev.get("text"):
                msgs.append(Message("user", ev["text"]))
            turn += 1
    return msgs


def leak_reports(eps: list[dict], tasks: dict, patterns: dict | None = None
                 ) -> list[tuple[dict, object]]:
    """One (episode, report) pair per episode.

    The first version looked each episode's report up by task id, so all k
    seeds of a task shared seed 0's verdict and every per-language count came
    out a multiple of k."""
    from pasarbench.simqa import leak_report
    out = []
    for e in eps:
        task = tasks.get(e["task_id"])
        if task:
            out.append((e, leak_report(task, transcript(e, task), patterns=patterns)))
    return out


def same_tasks(eps: list[dict], tasks: dict) -> list[dict]:
    """Keep the traces that were produced by the task definition we are about
    to read them against.

    The audit rebuilds each task from today's generator. If the generator has
    changed since the run, no fact it looks for is in the transcript, and it
    reports a clean-looking rate about nothing: run B predates the order-id fix
    of #9 and scored 1.2% (#26). Traces since then carry a digest of the task;
    older ones are checked by whether the customer ever says the order id."""
    from pasarbench.tasks import task_digest
    stale = {e["id"] for e in eps if e["head"].get("task_digest")
             and e["task_id"] in tasks
             and e["head"]["task_digest"] != task_digest(tasks[e["task_id"]])}
    if stale:
        print(f"  {len(stale)} of {len(eps)} traces came from a different version "
              f"of their task (digest mismatch) and are left out.")
    eps = [e for e in eps if e["id"] not in stale]

    def says_order_id(e: dict, task) -> bool:
        said = task.opening + " ".join(ev.get("text") or "" for ev in e["events"]
                                       if ev.get("kind") == "user_turn")
        return str(task.hidden_facts["order_id"]).lower() in said.lower()

    old = [e for e in eps if not e["head"].get("task_digest")
           and e["task_id"] in tasks and tasks[e["task_id"]].hidden_facts.get("order_id")]
    if old:
        seen = sum(says_order_id(e, tasks[e["task_id"]]) for e in old)
        if seen / len(old) < 0.5:
            print(f"  NOT COMPUTED. In {len(old) - seen} of {len(old)} of these traces the "
                  f"customer never says the order id its task holds today: they were "
                  f"produced by an earlier version of the task generator, and every "
                  f"leak or stall number would be about facts that are not in them.")
            return []
    return eps


def leaks(eps: list[dict], pattern_set: str = "current") -> None:
    print("\n" + "=" * 72)
    print("3. DID THE SIMULATOR LEAK FACTS BEFORE BEING ASKED?")
    print("=" * 72)
    from pasarbench.simqa import ASK_PATTERNS, ASK_PATTERNS_V1
    patterns = {"current": ASK_PATTERNS, "v1": ASK_PATTERNS_V1}[pattern_set]
    if pattern_set != "current":
        print(f"  ask-patterns: {pattern_set} (the list before WHAT_FAILED #26)")
    try:
        from pasarbench.generate import generate
        from pasarbench.tasks import BY_ID
        tasks = {**BY_ID, **{t.task_id: t for t in generate()[0]}}
    except Exception as ex:                                  # noqa: BLE001
        print(f"  (could not load tasks: {ex})")
        return

    from pasarbench.simqa import audit

    eps = same_tasks(eps, tasks)
    if not eps:
        return
    pairs = leak_reports(eps, tasks, patterns)
    if not pairs:
        print("  (no matching tasks)")
        return
    reports = [r for _, r in pairs]
    a = audit(reports)
    on_request = sum(len(r.revealed_on_request) for r in reports)

    # BY LANGUAGE, always. The overall rate is useless here: leaks concentrate
    # in specific languages, and language is exactly the axis the suite is
    # trying to measure. An aggregate hides the bias where it matters.
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    for e, r in pairs:
        per[e.get("language", "?")][1] += 1
        per[e.get("language", "?")][0] += bool(r.leaked)
    print("  leak rate BY LANGUAGE (this is the number that matters):")
    for lang, (bad, tot) in sorted(per.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        rate = bad / tot if tot else 0
        flag = "  <- contaminated" if rate > 0.15 else ("  <- clean" if rate == 0 else "")
        print(f"    {lang:7s} {rate:6.1%}  ({bad}/{tot} episodes){flag}")
    # A rate that follows language is a claim about the simulator OR about the
    # detector's ask-patterns, and the table cannot tell which (#26).
    base = per.get("en", [0, 0])
    base_rate = base[0] / base[1] if base[1] else 0.0
    odd = [lang for lang, (bad, tot) in per.items()
           if tot and bad / tot > max(0.05, 3 * base_rate) and lang not in ("en", "sg-en")]
    if odd:
        print(f"\n  CHECK THE DETECTOR BEFORE THE SIMULATOR: {', '.join(sorted(odd))} "
              f"leak far more often than English. Read the agent line under each "
              f"leak below -- if it asks for the fact in words simqa.ASK_PATTERNS "
              f"lacks, the customer answered a question and the pattern list is "
              f"short (WHAT_FAILED #26).")
    # The mirror image: requests the customer did not answer. Flat across
    # languages on an honest simulator; a gate that cannot read a language's
    # requests stalls there instead of leaking (#26).
    from pasarbench.simqa import stall_report
    stall = defaultdict(lambda: [0, 0])
    for e, _ in pairs:
        task = tasks[e["task_id"]]
        asked, unanswered = stall_report(task, transcript(e, task), patterns)
        stall[e.get("language", "?")][0] += asked
        stall[e.get("language", "?")][1] += unanswered
    print("\n  requests for a fact the customer did NOT answer in the next turn:")
    for lang, (asked, unanswered) in sorted(stall.items()):
        if asked:
            print(f"    {lang:7s} {unanswered / asked:6.1%}  ({unanswered}/{asked} requests)")
    print()
    print(f"  episodes {a['transcripts']}   leak rate {a['leak_rate']:.1%}   "
          f"clean {a['clean_rate']:.1%}")
    print(f"  facts revealed ON REQUEST (correct behaviour): {on_request}")
    for e, r in pairs:
        for fact, t, agent in r.leak_context:
            print(f"  LEAK {e['id']} [{e.get('language', '?')}]: `{fact}` at turn {t}, "
                  f"no request recognised")
            print(f"       agent before it: {agent[-140:]!r}")
    if a["leak_rate"] == 0:
        print("  no leaks -- the customer made the agent ask for everything")
    elif a["leak_rate"] > 0.15:
        print("  above 15%: either the benchmark has drifted toward single-turn, or "
              "the detector cannot read the agent's requests. The lines above "
              "decide which.")


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
    ap.add_argument("--ask-patterns", choices=["current", "v1"], default="current",
                    help="v1 = the list before WHAT_FAILED #26, to reproduce it")
    ap.add_argument("--leaks-only", action="store_true",
                    help="print only the simulator leak audit")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.trace_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no traces in {args.trace_dir}")
    eps = [load(f) for f in files]

    if args.leaks_only:
        markup(eps)
        leaks(eps, args.ask_patterns)
        return

    config(eps)
    markup(eps)
    verdicts(eps)
    if args.task:
        matches = [x for x in eps if args.task in (x["id"], x["task_id"])]
        for e in matches[:3]:
            sequence(e)
    else:
        for e in eps:
            if not e["foot"].get("passed"):
                sequence(e)

    reliability(eps)
    empties(eps)
    leaks(eps, args.ask_patterns)
    cache_and_cost(eps)
    growth(eps)


if __name__ == "__main__":
    main()
