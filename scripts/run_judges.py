"""
Score the labelled sample with both judges.

    python scripts/run_judges.py \
        --model qwen3.8-flash \
        --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
        --extra-body '{"reasoning_effort":"low"}' --workers 12

Writes data/labels/judge_naive.jsonl and judge_decomposed.jsonl, which are what
scripts/make_report.py reads to produce the agreement table.

ONLY RUN THIS AFTER LABELLING IS FINISHED.
Seeing judge output before you label anchors your labels toward it, and the
agreement you then measure is partly your own anchoring. The script refuses to
run if round 1 is incomplete.

USE A DIFFERENT MODEL FAMILY FROM THE AGENT.
The agent under test was deepseek-v4-pro. A judge from the same family rates
its own family's outputs higher -- self-preference bias -- which inflates
agreement for reasons that have nothing to do with the rubric. qwen3.8-flash is
a different family and costs almost nothing at this volume.

TOOL RESULTS, AND HOW THE JUDGE GETS THEM.
Traces record each tool call's NAME, ARGUMENTS, ok/error, the LENGTH of the
payload and, from v19, a DIGEST of it -- not the payload. Without it a judge
cannot check any fact the agent read from a lookup, and a strict one flags them
as invented. The environment is deterministic, so harness/replay.py regenerates
every result by replaying the calls against the world the episode ran in, and
verifies each one against the recorded digest, or length in older traces.
`--traces ... --payloads` gives the judge those results. A length cannot tell a
tracking number from another of the same length, and v18's came from a salted
hash: in a trace without digests, a generated shipment's payload is withheld
rather than shown with a number the agent never saw (WHAT_FAILED #33).
The calibration run against human labels deliberately does NOT: the humans
labelled from the same payload-free view, and the two must see the same thing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pasarbench.harness.backends import OpenAICompatBackend          # noqa: E402
from pasarbench.harness.types import Message, ToolCall               # noqa: E402
from pasarbench.judge.judges import DecomposedJudge, NaiveJudge      # noqa: E402
from pasarbench.judge.rubric import is_complete, overall_acceptable  # noqa: E402
from pasarbench.claims import ENGLISH, false_claims                    # noqa: E402
from pasarbench.harness.replay import replay_payloads                  # noqa: E402

ROOT = Path("data/labels")


def messages_from_trace(path: str, payloads: list[str | None] | None = None
                        ) -> list[Message]:
    """Rebuild the conversation IN ORDER from a trace file.

    Turn ordering matters: the loop numbers user turns from 1, and counting
    from 0 makes the agent look like it acted on information before it was
    given (see WHAT_FAILED #7).

    `payloads` (from harness.replay) are the exact tool results the agent saw,
    one per call in trace order. Without them the judge sees only each call's
    name, arguments and ok/error, and cannot check any fact the agent read
    from a lookup.
    """
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    head = next((r for r in recs if r.get("type") == "header"), {})
    steps = [r for r in recs if r.get("type") == "step"]
    events = {r["turn"]: r for r in recs
              if r.get("type") == "event" and r.get("kind") == "user_turn"}

    msgs: list[Message] = [Message("system", f"[{head.get('policy_mode', '?')} mode]")]
    turn, k = 1, 0
    for st in steps:
        calls = [ToolCall(name=tc["name"], arguments=tc.get("arguments", {}),
                          id=tc.get("id", "")) for tc in st.get("tool_calls", [])]
        if st.get("model_content") or calls:
            msgs.append(Message("assistant", st.get("model_content", ""),
                                tool_calls=calls))
        for tc, tr in zip(calls, st.get("tool_results", [])):
            real = payloads[k] if payloads and k < len(payloads) else None
            k += 1
            # Without a replayed payload, give the judge the verdict and the
            # arguments -- what the trace itself holds.
            body = real if real is not None else json.dumps(
                {"ok": tr.get("ok"), "error": tr.get("error"),
                 "args_echo": tr.get("args")}, ensure_ascii=False)
            msgs.append(Message("tool", body, tool_call_id=tc.id, name=tc.name))
        if not calls:
            ev = events.get(turn)
            if ev and ev.get("text"):
                msgs.append(Message("user", ev["text"]))
            turn += 1
    return msgs


def judge_vs_verifier(cell: Path, mk, mode: str = "batched", workers: int = 8,
                      limit: int = 0, out_root: Path = Path("data/judge_vs_verifier"),
                      payloads: bool = False, policy: bool = False) -> dict:
    """Both judges on every episode of one arm, scored against the VERIFIER.

    No human labels: the ground truth here is the database. The question is
    whether a judge that only reads the transcript agrees with a verifier that
    checks the end state -- and above all whether it accepts transcripts that
    CLAIM an action the database never saw. Those read like finished jobs.
    If the judge passes them, it cannot stand in for state checking, however
    well it agrees with a human on everything else.

    `mk` builds a judge backend; tests pass a fake one.
    """
    from pasarbench.harness.prompts import policy_text
    from pasarbench.sweep import ALL_TASKS
    by_id = {t.task_id: t for t in ALL_TASKS}
    # In preload mode the agent's system prompt held the whole policy. A judge
    # checking "correct per policy.md" gets the same text, or it is guessing.
    ctx = ("=== POLICY THE AGENT WAS BOUND BY (its system prompt held all of it) "
           "===\n" + policy_text()) if policy else ""
    files = sorted(cell.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no traces in {cell}")
    if limit:
        files = files[:limit]

    from pasarbench.rescore import rescore_records

    def one(f: Path) -> dict:
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        head = next((r for r in recs if r.get("type") == "header"), {})
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), {})
        # Ground truth is today's checks, as everywhere else (WHAT_FAILED #30);
        # the verdict the run recorded is kept beside it for --checker recorded.
        truth = rescore_records(recs, f.stem).verdict if foot else False
        steps = [r for r in recs if r.get("type") == "step"]
        task = by_id.get(head.get("task_id"))
        rp = replay_payloads(recs, task) if payloads and task else None
        msgs = messages_from_trace(str(f), rp["payloads"] if rp else None)
        nj = NaiveJudge(mk(), context=ctx).judge(f.stem, msgs)
        dj = DecomposedJudge(mk(), mode=mode, context=ctx).judge(f.stem, msgs)
        texts = [st["model_content"] for st in steps
                 if (st.get("model_content") or "").strip()]
        calls = [(tc["name"], bool(tr.get("ok"))) for st in steps
                 for tc, tr in zip(st.get("tool_calls", []), st.get("tool_results", []))]
        read = task is not None and head.get("language") in ENGLISH
        claims = false_claims({"task": task, "texts": texts, "calls": calls}) if read else []
        s = nj.get("score")
        return {
            "transcript_id": f.stem, "task_id": head.get("task_id"),
            "trap": head.get("trap"), "language": head.get("language"),
            "verifier_passed": truth,
            "verifier_passed_recorded": bool(foot.get("passed")),
            "naive_score": s,
            "naive_reason": nj.get("reason", ""),
            # a naive response with no valid 1-5 score is not a verdict
            "naive_ok": (bool(nj["labels"].get("overall_acceptable"))
                         if isinstance(s, int) and 1 <= s <= 5 else None),
            "dec_ok": (overall_acceptable(dj["labels"])
                       if is_complete(dj["labels"]) and not dj.get("errors") else None),
            "dec_violations": [k for k, v in dj["labels"].items() if v is False],
            # what the judge quoted for each violation: the why behind a reject
            "dec_evidence": {k: (dj.get("evidence") or {}).get(k, {}).get("quote", "")
                             for k, v in dj["labels"].items() if v is False},
            "payload_calls": [rp["verified"], rp["total"]] if rp else None,
            "dec_unparsed": dj.get("unparsed", []),
            "claims_read": read,
            "false_claims": [t for t, _ in claims],
            "claim_quotes": [q for _, q in claims],
        }

    rows, errs = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, f): f for f in files}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:                            # noqa: BLE001
                errs += 1
                print(f"  ! {futs[fut].stem}: {e}")
            if n % 25 == 0 or n == len(files):
                print(f"  {n}/{len(files)}", flush=True)
    rows.sort(key=lambda r: r["transcript_id"])

    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / (f"{cell.parent.name}__{cell.name}"
                      + ("__payloads" if payloads else "")
                      + ("__policy" if policy else "") + ".jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    passed = [r for r in rows if r["verifier_passed"]]
    failed = [r for r in rows if not r["verifier_passed"]]
    claimed = [r for r in failed if r["false_claims"]]

    def rate(group: list[dict], key: str) -> tuple[int, int]:
        v = [r[key] for r in group if r[key] is not None]
        return sum(v), len(v)

    summary: dict = {"n": len(rows), "passed": len(passed), "failed": len(failed),
                     "claimed": len(claimed), "errors": errs}
    if payloads:
        v = sum(r["payload_calls"][0] for r in rows if r["payload_calls"])
        t = sum(r["payload_calls"][1] for r in rows if r["payload_calls"])
        summary["payload_calls"] = (v, t)
        print(f"\ntool results replayed and verified for {v}/{t} calls"
              + ("" if v == t else f" -- {t - v} did not reproduce and fell back "
                 f"to arguments-only"))
    rec = sum(r["verifier_passed_recorded"] for r in rows)
    moved = sum(r["verifier_passed"] != r["verifier_passed_recorded"] for r in rows)
    print(f"\n{len(rows)} episodes: verifier passed {len(passed)}, failed "
          f"{len(failed)} (today's checks"
          + (f"; as the runs recorded them, {rec} and {len(rows) - rec}" if moved else "")
          + f"), of which {len(claimed)} claim an action that never happened.  "
          f"errors {errs}\n")
    print(f"{'judge accepted …':28s} {'verifier passed':>16s} {'verifier FAILED':>16s} "
          f"{'claimed, not done':>18s}")
    for name, key in (("naive (one 1-5 score)", "naive_ok"),
                      ("decomposed (9 criteria)", "dec_ok")):
        cells = [rate(passed, key), rate(failed, key), rate(claimed, key)]
        summary[key] = cells
        print(f"  {name:26s} " + " ".join(
            f"{f'{a}/{b}':>16s}" if i < 2 else f"{f'{a}/{b}':>18s}"
            for i, (a, b) in enumerate(cells)))
    from pasarbench.judge.agreement import agreement
    print()
    for name, key in (("naive", "naive_ok"), ("decomposed", "dec_ok")):
        pairs = [(r["verifier_passed"], r[key]) for r in rows if r[key] is not None]
        if pairs:
            a = agreement([x for x, _ in pairs], [y for _, y in pairs], bootstrap=1000)
            summary[f"{key}_kappa"] = a.kappa
            ci = f"[{a.ci95[0]:+.2f}, {a.ci95[1]:+.2f}]" if a.ci95 else ""
            print(f"  {name:11s} vs verifier: kappa {a.kappa:+.3f} {ci}  "
                  f"(agreement {a.p_o:.0%} of {a.n})")
    print("\n  'verifier FAILED' accepted = the judge passing an episode the "
          "database\n  fails. On 'claimed, not done' that is a transcript that "
          "reads as a\n  finished job over an empty table.")

    if claimed:
        print("\nclaimed, not done -- what each judge said:")
        for r in claimed:
            print(f"  {r['transcript_id']:24s} {', '.join(r['false_claims'])}")
            if r.get("naive_reason"):
                print(f"      naive said: \"{r['naive_reason'][:150]}\"")
            print(f"      naive: score {r['naive_score']} -> "
                  f"{'ACCEPTED' if r['naive_ok'] else 'rejected'}   decomposed: "
                  f"{'ACCEPTED' if r['dec_ok'] else 'rejected'}"
                  + (f" (flagged: {', '.join(r['dec_violations'])})"
                     if r["dec_violations"] else " (flagged nothing)"))
            for crit, quote in r.get("dec_evidence", {}).items():
                if quote:
                    print(f"      {crit} quoted: \"{quote[:140]}\"")
            for q in r["claim_quotes"]:
                print(f"      \"…{q}…\"")
    unparsed = sum(1 for r in rows if r["dec_unparsed"])
    if rows and unparsed > len(rows) * 0.1:
        print(f"\n!! {unparsed} episodes had an unparsed criterion; unparsed "
              f"defaults to SATISFIED, so the decomposed judge reads lenient.")
    unread = sum(1 for r in failed if not r["claims_read"])
    if unread:
        print(f"\n{unread} failed episode(s) not in English: claims not read.")
    print(f"\nwrote {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-flash")
    ap.add_argument("--base-url",
                    default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--extra-body", default="")
    ap.add_argument("--mode", default="batched", choices=["batched", "per_criterion"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only N, for a dry run")
    ap.add_argument("--traces", default="",
                    help="one arm's trace folder, e.g. traces/I-tools2/full+search-300: "
                         "score every episode against the VERIFIER instead of your "
                         "labels (no labels needed; writes data/judge_vs_verifier/)")
    ap.add_argument("--policy", action="store_true",
                    help="with --traces: give both judges the policy the agent was "
                         "bound by (its preload system prompt)")
    ap.add_argument("--payloads", action="store_true",
                    help="with --traces: replay every tool result so the judge sees "
                         "what the agent saw (verified call by call)")
    args = ap.parse_args()

    key = (args.api_key or os.environ.get("PASARBENCH_SIM_API_KEY")
           or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not key:
        raise SystemExit("set PASARBENCH_SIM_API_KEY (or DASHSCOPE_API_KEY)")

    extra = json.loads(args.extra_body) if args.extra_body else {}

    def mk():
        return OpenAICompatBackend(model=args.model, base_url=args.base_url,
                                   api_key=key, temperature=0.0,
                                   max_tokens=1536, extra_body=extra)

    if args.traces:
        print(f"judging {args.traces} with {args.model} against the verifier")
        judge_vs_verifier(Path(args.traces), mk, args.mode, args.workers, args.limit,
                          payloads=args.payloads, policy=args.policy)
        return

    sample = [json.loads(l) for l in
              (ROOT / "sample.jsonl").read_text().splitlines() if l.strip()]
    human = {}
    for l in (ROOT / "human_round1.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            human[r["transcript_id"]] = r

    todo = [s for s in sample if s["transcript_id"] in human
            and is_complete(human[s["transcript_id"]]["labels"])]
    if len(todo) < len(sample) * 0.9:
        raise SystemExit(
            f"only {len(todo)} of {len(sample)} transcripts are fully labelled. "
            f"Finish round 1 first -- running the judges now and labelling after "
            f"would anchor your labels to the model's answers.")
    if args.limit:
        todo = todo[:args.limit]
    print(f"scoring {len(todo)} transcripts with {args.model} "
          f"(mode={args.mode}, workers={args.workers})")

    def score(item):
        tid = item["transcript_id"]
        msgs = messages_from_trace(item["path"])
        nj = NaiveJudge(mk()).judge(tid, msgs)
        dj = DecomposedJudge(mk(), mode=args.mode).judge(tid, msgs)
        for r in (nj, dj):
            r["trap"] = item.get("trap")
            r["language"] = item.get("language")
        return nj, dj

    naive, dec, errs = [], [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(score, it): it for it in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                a, b = f.result()
                naive.append(a)
                dec.append(b)
            except Exception as e:                            # noqa: BLE001
                errs += 1
                print(f"  ! {futs[f]['transcript_id']}: {e}")
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)

    for name, rows in (("judge_naive.jsonl", naive), ("judge_decomposed.jsonl", dec)):
        with (ROOT / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {ROOT / name}  ({len(rows)} rows)")

    unparsed = sum(1 for r in dec if r.get("unparsed"))
    print(f"\nerrors {errs}   transcripts with an unparsed criterion {unparsed}")
    if dec and unparsed > len(dec) * 0.1:
        print("!! over 10% had a criterion the judge did not return cleanly.")
        print("   Unparsed defaults to SATISFIED, so this biases the judge")
        print("   toward leniency. Fix the prompt before trusting the kappa.")

    # The human side of the naive comparison is NOT written here. make_report
    # derives it from the round-1 labels at report time with
    # rubric.overall_acceptable, so it cannot go stale if a label is changed
    # after the judges ran.
    print("\nnext:  python scripts/make_report.py --out RESULTS.md")


if __name__ == "__main__":
    main()
