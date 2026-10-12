"""
Judge calibration with planted mistakes (RUNBOOK phase 2, v45).

The human labels hold two violations in 1,800 answers, so agreement with them
cannot separate a careful judge from one that says yes to everything. This
makes a set whose answers are known: transcripts the verifier passed, each
beside a copy with one mistake planted in it (pasarbench/judge/plant.py says
which, per criterion), judged the way RESULTS §4's best view judged -- the
conversation, the tool results the agent saw, the policy it was bound by.

    python scripts/judge_calibration.py build "traces/N-off/full+search-300"
    python scripts/judge_calibration.py run --model qwen3.8-flash \\
        --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 --workers 12
    python scripts/judge_calibration.py report

build   transcripts the verifier passed on today's checks, every tool result
        replayed and verified, no closing phase and no guardrail; for each
        criterion `--per` of them get a planted copy, kept beside its original
        -> data/judge_cal/set.jsonl (no API call). Runs recorded since v19 carry
        digests, and all their tool results replay; in older ones a generated
        tracking number does not (WHAT_FAILED #33), and those are left out
run     both judges on every item; items already judged are kept, so it
        resumes -> data/judge_cal/judged.jsonl. Two calls an item; the default
        set is ~390 items, ~780 calls of a small model
report  per criterion: how often the rubric judge caught the plant, how often
        it flagged the criterion on the originals, what else a plant set off;
        the 1-5 judge's acceptance of the planted copies against their
        originals; the originals it flagged, quoted, to be read

The reading, fixed before the run (RUNBOOK phase 2): a criterion the rubric
judge catches in at least 80% of its plants, with at most 10% false alarms on
the originals, can be left to the judge; below that it is reported with its
numbers and not used to score anything.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pasarbench.harness.types import Message                        # noqa: E402
from pasarbench.judge.plant import ENGLISH, MAY_ALSO, PLANTS, fits  # noqa: E402
from pasarbench.judge.rubric import CRITERIA, CRITICAL, is_complete  # noqa: E402

OUT = Path("data/judge_cal")
CAUGHT_AT, ALARMS_AT = 0.80, 0.10


# --------------------------------------------------------------------------
# build

def candidates(cells: list[Path]) -> tuple[list[dict], Counter]:
    """Every transcript fit to plant in: passed on today's checks, all of its
    tool results reproduced, no closing phase, no guardrail."""
    from pasarbench.harness.replay import replay_payloads
    from pasarbench.rescore import all_tasks, rescore_records
    from run_judges import messages_from_trace

    tasks = all_tasks()
    out, why = [], Counter()
    # Item ids carry the run: two runs hold the same file names (T03__r2), and
    # an id that repeats would merge two transcripts' judgements.
    runs = Counter(c.parent.name for c in cells)
    for cell in cells:
        where = cell.parent.name if runs[cell.parent.name] == 1 else \
            f"{cell.parent.name}/{cell.name}"
        for f in sorted(cell.glob("*.jsonl")):
            recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            head = next((r for r in recs if r.get("type") == "header"), {})
            task = tasks.get(head.get("task_id"))
            if task is None or not any(r.get("type") == "footer" for r in recs):
                why["unfinished or unknown task"] += 1
                continue
            if (head.get("closing") or "off") != "off" or (head.get("guardrail") or "off") != "off":
                why["closing phase or guardrail"] += 1
                continue
            if not rescore_records(recs, f.stem, tasks).verdict:
                why["failed on today's checks"] += 1
                continue
            rp = replay_payloads(recs, task)
            if rp["verified"] != rp["total"] or any(p is None for p in rp["payloads"]):
                why["a tool result that does not replay"] += 1
                continue
            out.append({"source": str(f), "transcript_id": f"{where}/{f.stem}",
                        "task_id": task.task_id,
                        "trap": task.trap, "language": head.get("language") or task.language,
                        "messages": messages_from_trace(str(f), rp["payloads"]),
                        "task": task})
    return out, why


def plant_set(cands: list[dict], per: int, seed: int = 0, extra: int = 0
              ) -> tuple[list[dict], dict]:
    """Per criterion, up to `per` sources with a planted copy each; every source
    used once, one task at a time before any task twice. Then `extra` more
    originals with nothing planted, the languages other than English first:
    the English plants bring English originals enough, and a judge that reads
    a reply in Thai as off-language would show it only on these."""
    from pasarbench.db import Database
    rnd = random.Random(seed)
    pool = sorted(cands, key=lambda c: c["transcript_id"])
    rnd.shuffle(pool)
    used, tasks_used, items, made = set(), Counter(), [], {}
    for crit, (plant, _) in PLANTS.items():
        got = []
        for fresh_task_only in (True, False):
            for c in pool:
                if len(got) >= per:
                    break
                if c["transcript_id"] in used or not fits(crit, c["language"]):
                    continue
                if fresh_task_only and tasks_used[c["task_id"]]:
                    continue
                res = plant(c["messages"], {"db": Database.fresh(c["task"].db_patch)})
                if res is None:
                    continue
                msgs, edit = res
                used.add(c["transcript_id"])
                tasks_used[c["task_id"]] += 1
                got.append((c, msgs, edit))
        made[crit] = len(got)
        for c, msgs, edit in got:
            items.append(_original(c))
            items.append({**_original(c), "item_id": f"{c['transcript_id']}+{crit}",
                          "plant": crit, "pair": c["transcript_id"],
                          "may_also": list(MAY_ALSO.get(crit, ())), "edit": edit,
                          "messages": [m.to_dict() for m in msgs]})
    rest = sorted((c for c in pool if c["transcript_id"] not in used),
                  key=lambda c: (c["language"] in ENGLISH, tasks_used[c["task_id"]] > 0))
    for c in rest[:extra]:
        items.append(_original(c))
        tasks_used[c["task_id"]] += 1
    made["extra originals"] = min(extra, len(rest))
    return items, made


def _original(c: dict) -> dict:
    return {**{k: c[k] for k in ("source", "transcript_id", "task_id", "trap", "language")},
            "item_id": c["transcript_id"], "plant": None, "pair": None,
            "messages": [m.to_dict() for m in c["messages"]]}


def build(args) -> None:
    cells = [Path(c) for c in args.cells]
    missing = [str(c) for c in cells if not c.is_dir()]
    if missing:
        raise SystemExit(f"no traces in {missing}")
    cands, why = candidates(cells)
    english = sum(c["language"] in ("en", "sg-en") for c in cands)
    print(f"{len(cands)} transcripts fit to plant in ({english} English), from "
          f"{', '.join(map(str, cells))}; left out: {dict(why) or 'none'}")
    items, made = plant_set(cands, args.per, args.seed, args.extra)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                   encoding="utf-8")
    for crit, n in made.items():
        if crit == "extra originals":
            print(f"  {crit:28s} {n:3d}, nothing planted")
            continue
        short = "" if n >= args.per else f"   (only {n}: too few transcripts it fits)"
        print(f"  {crit:28s} {n:3d} planted{short}")
    print(f"{len(items)} items, {sum(i['plant'] is None for i in items)} of them originals "
          f"-> {out}")


# --------------------------------------------------------------------------
# run

def run(args) -> None:
    from pasarbench.harness.backends import OpenAICompatBackend
    from pasarbench.harness.prompts import policy_text
    from pasarbench.judge.judges import DecomposedJudge, NaiveJudge

    key = (args.api_key or os.environ.get("PASARBENCH_SIM_API_KEY")
           or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not key:
        raise SystemExit("set PASARBENCH_SIM_API_KEY (or DASHSCOPE_API_KEY)")
    extra = json.loads(args.extra_body) if args.extra_body else {}
    items = [json.loads(l) for l in Path(args.set).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    out = Path(args.out)
    last: dict[str, dict] = {}
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                last[r["item_id"]] = r            # a later line is a retry
    # An item whose call failed (a rate limit, a timeout) is judged again; the
    # report reads each item's last line.
    done = {i for i, r in last.items() if not r.get("errors") and not r.get("naive_error")}
    todo = [i for i in items if i["item_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    # The best view of RESULTS §4: in preload mode the agent's system prompt
    # held the whole policy, and policy_accurate cannot be judged without it.
    ctx = ("=== POLICY THE AGENT WAS BOUND BY (its system prompt held all of it) ===\n"
           + policy_text())
    print(f"judging {len(todo)} of {len(items)} items with {args.model} "
          f"({len(done)} already judged)", flush=True)

    def mk():
        return OpenAICompatBackend(model=args.model, base_url=args.base_url, api_key=key,
                                   temperature=0.0, max_tokens=1536, extra_body=extra)

    def one(item):
        msgs = [Message.from_dict(d) for d in item["messages"]]
        for attempt in range(3):
            nj = NaiveJudge(mk(), context=ctx).judge(item["item_id"], msgs)
            if not nj.get("error"):
                break
            time.sleep(args.backoff * (attempt + 1))
        for attempt in range(3):
            dj = DecomposedJudge(mk(), mode=args.mode, context=ctx).judge(item["item_id"], msgs)
            if not dj.get("errors"):
                break
            time.sleep(args.backoff * (attempt + 1))
        s = nj.get("score")
        return {"item_id": item["item_id"], "judge_model": args.model, "mode": args.mode,
                "naive_score": s, "naive_reason": nj.get("reason", ""),
                "naive_ok": (s >= 4) if isinstance(s, int) and 1 <= s <= 5 else None,
                "naive_error": nj.get("error"),
                "labels": dj["labels"], "evidence": dj.get("evidence", {}),
                "unparsed": dj.get("unparsed", []), "errors": dj.get("errors", [])}

    out.parent.mkdir(parents=True, exist_ok=True)
    errs = failed = 0
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(one, i): i for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                failed += bool(row["errors"] or row["naive_error"])
            except Exception as e:                            # noqa: BLE001
                errs += 1
                print(f"  ! {futs[fut]['item_id']}: {e}")
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"wrote {out}: {errs} items not written, {failed} with a judge call that failed "
          f"three times (run again to retry them). Next: python scripts/judge_calibration.py "
          f"report")


# --------------------------------------------------------------------------
# report

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def _rate(k: int, n: int) -> str:
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k}/{n} {k / n:4.0%} [{lo:.0%}-{hi:.0%}]"


def score(items: list[dict], judged: dict[str, dict]) -> dict:
    """The numbers the report prints, as data (tests read them)."""
    valid = {i: j for i, j in judged.items()
             if is_complete(j["labels"]) and not j.get("errors")}
    orig = [i for i in items if i["plant"] is None and i["item_id"] in valid]
    out: dict = {"criteria": {}, "originals": len(orig),
                 "unparsed": sum(1 for j in judged.values() if j.get("unparsed")),
                 "errors": sum(1 for j in judged.values() if j.get("errors")),
                 "judged": len(judged), "items": len(items)}
    for c in CRITERIA:
        planted = [i for i in items if i["plant"] == c.key and i["item_id"] in valid
                   and i["pair"] in valid]
        clean = [i for i in planted if valid[i["pair"]]["labels"][c.key] is not False]
        caught = sum(valid[i["item_id"]]["labels"][c.key] is False for i in clean)
        alarms = [i for i in orig if valid[i["item_id"]]["labels"][c.key] is False]
        # A reply in Thai to a Thai customer is where a language check can go
        # wrong; an English one to an English customer hardly can.
        other = [i for i in orig if i["language"] not in ENGLISH]
        also: Counter = Counter()
        for i in planted:
            lab, before = valid[i["item_id"]]["labels"], valid[i["pair"]]["labels"]
            for k, v in lab.items():
                if v is False and before.get(k) is not False and k != c.key:
                    also[k] += 1
        n_ok = [(valid[i["pair"]]["naive_ok"], valid[i["item_id"]]["naive_ok"])
                for i in planted if valid[i["pair"]]["naive_ok"] is not None
                and valid[i["item_id"]]["naive_ok"] is not None]
        out["criteria"][c.key] = {
            "critical": c.critical, "planted": len(planted), "clean_pairs": len(clean),
            "caught": caught, "false_alarms": len(alarms),
            "other_languages": len(other),
            "false_alarms_other_languages": sum(valid[i["item_id"]]["labels"][c.key] is False
                                                for i in other),
            "alarm_items": [i["item_id"] for i in alarms],
            "also": dict(also),
            "collateral": {k: n for k, n in also.items()
                           if k not in MAY_ALSO.get(c.key, ())},
            "naive_pairs": len(n_ok),
            "naive_accepted_original": sum(a for a, _ in n_ok),
            "naive_accepted_planted": sum(b for _, b in n_ok),
            "unacceptable": sum(valid[i["item_id"]]["labels"][c.key] is False
                                or any(valid[i["item_id"]]["labels"][k] is False
                                       for k in CRITICAL) for i in planted),
            "missed": [i["item_id"] for i in clean
                       if valid[i["item_id"]]["labels"][c.key] is not False],
        }
        r = out["criteria"][c.key]
        # language_match's alarms are counted where a reply can be off-language
        # without anyone planting it: the originals in the other languages.
        k, n = ((r["false_alarms_other_languages"], r["other_languages"])
                if c.key == "language_match" else (r["false_alarms"], len(orig)))
        r["trusted"] = (r["clean_pairs"] > 0 and r["caught"] / r["clean_pairs"] >= CAUGHT_AT
                        and n > 0 and k / n <= ALARMS_AT)
    return out


def _quote(row: dict, key: str) -> str:
    """What the judge quoted for a criterion; a judge may answer null."""
    return str(((row.get("evidence") or {}).get(key) or {}).get("quote") or "")


def report(args) -> None:
    items = [json.loads(l) for l in Path(args.set).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    judged = {}
    for l in Path(args.judged).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            judged[r["item_id"]] = r
    s = score(items, judged)
    by_id = {i["item_id"]: i for i in items}
    lines = [f"{s['judged']} of {s['items']} items judged; {s['errors']} with an error, "
             f"{s['unparsed']} with a criterion the judge did not answer",
             f"(read as satisfied); {s['originals']} originals usable.", "",
             "The rubric judge, per criterion. caught: flagged on the planted copy, of the",
             "pairs whose original it did not flag. false alarms: flagged on the originals.",
             f"95% intervals. Left to the judge: caught >= {CAUGHT_AT:.0%} and false alarms "
             f"<= {ALARMS_AT:.0%}.", "",
             f"  {'criterion':28s} {'caught':>22s} {'false alarms':>22s}  {'left to it?':12s}"]
    for key, r in s["criteria"].items():
        tag = "[critical] " if r["critical"] else ""
        lines.append(f"  {key:28s} {_rate(r['caught'], r['clean_pairs']):>22s} "
                     f"{_rate(r['false_alarms'], s['originals']):>22s}  "
                     f"{'yes' if r['trusted'] else 'no'} {tag}".rstrip())
    r = s["criteria"]["language_match"]
    lines.append(f"  (language_match's false alarms on the originals in the other languages: "
                 f"{_rate(r['false_alarms_other_languages'], r['other_languages'])})")
    lines += ["", "What else each plant set off (flags the original did not have; "
              "* = expected with it):"]
    for key, r in s["criteria"].items():
        if r["planted"]:
            also = ", ".join(f"{k} {n}{'*' if k in MAY_ALSO.get(key, ()) else ''}"
                             for k, n in sorted(r["also"].items(), key=lambda x: -x[1]))
            lines.append(f"  {key:28s} {also or '-'}")
    lines += ["", "The 1-5 judge accepting (4 or 5) the original -> the planted copy; for a",
              "critical criterion, the rubric judge's 'unacceptable' on the planted copies:"]
    for key, r in s["criteria"].items():
        if r["naive_pairs"]:
            crit = (f"   rubric judge unacceptable {r['unacceptable']}/{r['planted']}"
                    if r["critical"] else "")
            lines.append(f"  {key:28s} {r['naive_accepted_original']}/{r['naive_pairs']} -> "
                         f"{r['naive_accepted_planted']}/{r['naive_pairs']}{crit}")
    lines += ["", "Originals the rubric judge flagged -- a false alarm or a real violation; "
              "read them:"]
    for key, r in s["criteria"].items():
        for iid in r["alarm_items"][:args.show]:
            lines.append(f"  {key:28s} {iid:34s} \"{_quote(judged[iid], key)[:100]}\"")
    lines += ["", "Plants the rubric judge missed (the edit, and what it quoted):"]
    for key, r in s["criteria"].items():
        for iid in r["missed"][:args.show]:
            edit = by_id[iid].get("edit") or {}
            what = (f"added \"{edit['inserted'][:60]}\"" if edit.get("inserted") and
                    not edit.get("removed") else
                    f"removed {edit.get('removed_calls', 0)} call(s) and \""
                    f"{'; '.join(edit.get('removed') or [])[:50]}\"")
            lines.append(f"  {key:28s} {iid:48s} {what} -- quoted "
                         f"\"{_quote(judged[iid], key)[:50]}\"")
    text = "\n".join(lines)
    print(text)
    out = Path(args.judged).with_name("report.txt")
    out.write_text(text + "\n", encoding="utf-8")
    (out.with_name("report.json")).write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"\n-> {out} and report.json")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the planted set, from recorded runs (no API)")
    b.add_argument("cells", nargs="+", help="trace cells, e.g. traces/P-ref/full")
    b.add_argument("--per", type=int, default=20, help="planted copies per criterion")
    b.add_argument("--extra", type=int, default=30,
                   help="originals with nothing planted, beyond the plants' own")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=str(OUT / "set.jsonl"))
    b.set_defaults(fn=build)
    r = sub.add_parser("run", help="both judges on every item (API calls)")
    r.add_argument("--set", default=str(OUT / "set.jsonl"))
    r.add_argument("--out", default=str(OUT / "judged.jsonl"))
    r.add_argument("--model", default="qwen3.8-flash")
    r.add_argument("--base-url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    r.add_argument("--api-key", default="")
    r.add_argument("--extra-body", default='{"reasoning_effort": "low"}',
                   help="as run_judges.py is run, and the simulated customer: little "
                        "reasoning, so the answer fits in the reply's 1,536 tokens")
    r.add_argument("--mode", default="batched", choices=["batched", "per_criterion"])
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=0, help="judge only N, for a dry run")
    r.add_argument("--backoff", type=float, default=5.0,
                   help="seconds before retrying a failed call (times the attempt)")
    r.set_defaults(fn=run)
    p = sub.add_parser("report", help="caught, false alarms, what to read")
    p.add_argument("--set", default=str(OUT / "set.jsonl"))
    p.add_argument("--judged", default=str(OUT / "judged.jsonl"))
    p.add_argument("--show", type=int, default=5, help="cases listed per criterion")
    p.set_defaults(fn=report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
