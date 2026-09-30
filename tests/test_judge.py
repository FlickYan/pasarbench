"""
Judge calibration tests.

The metrics are where this can go quietly wrong. A kappa implementation that is
subtly incorrect produces a number that looks fine and is not, and nothing
downstream will tell you. So the statistics are tested against hand-computable
cases before any of them are trusted.

Run: python -m tests.test_judge
"""

from __future__ import annotations

import json

from pasarbench.harness.types import Message, ModelResponse, Usage
from pasarbench.judge.agreement import (agreement, ceiling_report, cohens_kappa,
                                        confusion, length_bias, per_criterion,
                                        position_bias)
from pasarbench.judge.judges import (DecomposedJudge, NaiveJudge, PairwiseJudge,
                                     render_transcript)
from pasarbench.judge.label import stratified
from pasarbench.judge.rubric import (CRITERIA, CRITICAL, blank_labels,
                                     derive_verdict, rubric_text)

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


class FixedJudgeBackend:
    """Returns a canned payload. Lets the whole judge pipeline run offline."""
    name, reports_usage = "fixed", False

    def __init__(self, payload, raise_always=False, raw=None):
        self.payload, self.raise_always, self.raw = payload, raise_always, raw
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        if self.raise_always:
            raise ConnectionError("judge down")
        body = self.raw if self.raw is not None else json.dumps(self.payload)
        return ModelResponse(content=body, usage=Usage(100, 40))


def _crit_payload(**overrides):
    crit = {c.key: {"ok": overrides.get(c.key, True), "turn": 1, "quote": "..."}
            for c in CRITERIA}
    return {"criteria": crit}


# --------------------------------------------------------------------------

def test_kappa_math():
    print("\n=== kappa, checked against hand-computable cases ===")
    check("perfect agreement is 1.0",
          abs(cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1.0) < 1e-9)
    check("complete inversion is -1.0",
          abs(cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) - (-1.0)) < 1e-9)

    # 2x2: a=40 b=10 c=10 d=40, n=100. p_o=0.8, p_e=0.5, kappa=0.6
    a = [1] * 50 + [0] * 50
    b = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
    check("known 2x2 case gives kappa=0.6",
          abs(cohens_kappa(a, b) - 0.6) < 1e-9, f"{cohens_kappa(a, b)}")

    check("single-category degenerate case returns 0, not a crash",
          cohens_kappa([1] * 10, [1] * 10) == 0.0)
    check("empty input is safe", cohens_kappa([], []) == 0.0)


def test_prevalence_paradox():
    print("\n=== the kappa paradox is detected, not hidden ===")
    # A GENUINE paradox needs the disagreements to consume most of the tiny
    # minority class. Cells: both_yes=95, human_yes_judge_no=3,
    # human_no_judge_yes=1, both_no=1.  -> p_o=0.96 but kappa=0.315.
    # (An earlier draft of this test used 96/4 vs 96/4 and scored kappa=0.74 --
    # high agreement alone does NOT produce the paradox, and getting this wrong
    # is exactly how people mis-describe their own results.)
    a = [1] * 98 + [0] * 2
    b = [1] * 95 + [0] * 3 + [1] * 1 + [0] * 1
    r = agreement(a, b, bootstrap=200)
    check("raw agreement is high", r.p_o > 0.9, str(r.p_o))
    check("kappa is nonetheless low", r.kappa < 0.5, str(r.kappa))
    check("prevalence index is high", r.prevalence > 0.6, str(r.prevalence))
    check("the paradox is named in the interpretation",
          "KAPPA PARADOX" in r.note, r.note)
    check("pabak rescues the reading", r.pabak > 0.9, str(r.pabak))
    print(f"       {r}")

    # systematic bias: judge says yes far more often than the human
    h = [1] * 30 + [0] * 70
    j = [1] * 30 + [1] * 35 + [0] * 35
    rb = agreement(h, j, bootstrap=200)
    check("systematic bias is flagged", "SYSTEMATIC BIAS" in rb.note, rb.note)


def test_bootstrap():
    print("\n=== bootstrap CIs ===")
    a = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0] * 10
    b = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0] * 10
    r = agreement(a, b, bootstrap=800, seed=1)
    check("CI is produced", r.ci95 is not None)
    check("CI brackets the point estimate",
          r.ci95[0] <= r.kappa <= r.ci95[1], f"{r.ci95} vs {r.kappa}")
    small = agreement(a[:20], b[:20], bootstrap=800, seed=1)
    check("a smaller sample gives a wider interval",
          (small.ci95[1] - small.ci95[0]) > (r.ci95[1] - r.ci95[0]),
          f"{small.ci95} vs {r.ci95}")

    c = confusion(a, b)
    check("confusion splits the two disagreement directions",
          c["human_yes_judge_no"] + c["human_no_judge_yes"] > 0, str(c))


def test_rubric():
    print("\n=== rubric and verdict derivation ===")
    good = {c.key: True for c in CRITERIA}
    check("all satisfied -> good", derive_verdict(good) == "good")

    leak = dict(good, no_data_leak=False)
    check("a critical violation is disqualifying regardless of the rest",
          derive_verdict(leak) == "unacceptable")

    minor = dict(good, tone_professional=False)
    check("one minor violation -> acceptable", derive_verdict(minor) == "acceptable")

    many = dict(good, tone_professional=False, outcome_communicated=False,
                no_unfounded_promise=False)
    check("three minor violations -> poor", derive_verdict(many) == "poor")
    check("critical criteria are the three that matter",
          set(CRITICAL) == {"no_data_leak", "no_hallucinated_facts", "policy_accurate"},
          str(CRITICAL))
    check("rubric text carries guidance for labellers",
          "guidance:" in rubric_text() and len(rubric_text()) > 500)


def test_judges():
    print("\n=== judges ===")
    msgs = [Message("system", "..."),
            Message("user", "where is my parcel"),
            Message("assistant", "", tool_calls=[]),
            Message("assistant", "It was delivered on 8 Nov.")]

    t = render_transcript(msgs)
    check("transcript numbers customer turns", "[0] CUSTOMER:" in t, t[:80])

    dj = DecomposedJudge(FixedJudgeBackend(_crit_payload()), mode="batched")
    r = dj.judge("t1", msgs)
    check("decomposed judge fills every criterion",
          all(r["labels"][c.key] is not None for c in CRITERIA))
    check("decomposed judge derives a verdict", r["verdict"] == "good", r["verdict"])
    check("evidence is captured", bool(r["evidence"]))

    viol = DecomposedJudge(FixedJudgeBackend(_crit_payload(no_data_leak=False)))
    rv = viol.judge("t2", msgs)
    check("a critical violation propagates to the verdict",
          rv["verdict"] == "unacceptable", rv["verdict"])

    per = DecomposedJudge(FixedJudgeBackend({"ok": True, "turn": 1, "quote": "x"}),
                          mode="per_criterion")
    rp = per.judge("t3", msgs)
    check("per-criterion mode makes one call per criterion",
          per.backend.calls == len(CRITERIA), str(per.backend.calls))
    check("per-criterion mode produces the same shape",
          set(rp["labels"]) == {c.key for c in CRITERIA})

    junk = DecomposedJudge(FixedJudgeBackend(None, raw="not json at all"))
    rj = junk.judge("t4", msgs)
    check("unparseable output defaults to satisfied, never invents a violation",
          all(rj["labels"][c.key] is True for c in CRITERIA))
    check("unparsed criteria are counted so the run can be invalidated",
          len(rj["unparsed"]) == len(CRITERIA), str(len(rj["unparsed"])))

    dead = DecomposedJudge(FixedJudgeBackend(None, raise_always=True))
    rd = dead.judge("t5", msgs)
    check("a judge outage does not raise", bool(rd["errors"]))

    nj = NaiveJudge(FixedJudgeBackend({"score": 5, "reason": "fine"}))
    rn = nj.judge("t6", msgs)
    check("naive judge returns an ordinal score", rn["score"] == 5)
    check("naive judge yields only one collapsed label",
          set(rn["labels"]) == {"overall_acceptable"}, str(rn["labels"]))


def test_pairwise_and_bias():
    print("\n=== pairwise, position bias, length bias ===")
    a = [Message("user", "hi"), Message("assistant", "Short and correct.")]
    b = [Message("user", "hi"), Message("assistant", "A much longer answer " * 20)]

    class Positional:
        name, reports_usage = "positional", False
        def chat(self, messages, tools):
            return ModelResponse(content='{"winner":"left"}', usage=Usage(1, 1))

    pj = PairwiseJudge(Positional())
    r = pj.judge_both_orders("i1", a, b)
    check("a pure position-follower yields no usable winner", r["winner"] is None,
          str(r))
    check("both orders are recorded", r["verdict_ab"] and r["verdict_ba"])

    class Consistent:
        name, reports_usage = "consistent", False
        def __init__(self): self.n = 0
        def chat(self, messages, tools):
            self.n += 1
            side = "left" if self.n % 2 == 1 else "right"
            return ModelResponse(content=json.dumps({"winner": side}), usage=Usage(1, 1))

    r2 = PairwiseJudge(Consistent()).judge_both_orders("i2", a, b)
    check("a consistent judge produces a winner", r2["winner"] == "a", str(r2))
    check("order consistency is reported", r2["order_consistent"] is True)

    biased = [{"verdict_ab": "left", "verdict_ba": "left"} for _ in range(9)]
    biased.append({"verdict_ab": "left", "verdict_ba": "right"})
    pb = position_bias(biased)
    check("position bias is detected", "POSITION BIAS" in pb["verdict"], pb["verdict"])
    check("flip rate is quantified", pb["flip_rate"] == 0.9, str(pb["flip_rate"]))

    clean = [{"verdict_ab": "left", "verdict_ba": "right"} for _ in range(10)]
    check("a clean judge is not accused",
          "no meaningful position bias" in position_bias(clean)["verdict"])

    longwins = ([{"chars": 900 + i, "positive": True} for i in range(20)] +
                [{"chars": 100 + i, "positive": False} for i in range(20)])
    lb = length_bias(longwins)
    check("length bias is detected", "LENGTH BIAS" in lb["verdict"], str(lb))
    check("mean lengths per verdict are reported",
          lb["mean_chars_positive"] > lb["mean_chars_negative"], str(lb))

    mixed = [{"chars": 500, "positive": i % 2 == 0} for i in range(40)]
    check("no false length-bias accusation",
          "no strong length effect" in length_bias(mixed)["verdict"])


def test_ceiling():
    print("\n=== the ceiling: judge agreement as a fraction of self-agreement ===")
    keys = ["no_data_leak", "tone_professional"]

    def recs(vals_leak, vals_tone):
        return [{"transcript_id": f"t{i}",
                 "labels": {"no_data_leak": a, "tone_professional": b}}
                for i, (a, b) in enumerate(zip(vals_leak, vals_tone))]

    n = 40
    leak1 = [i % 3 != 0 for i in range(n)]
    tone1 = [i % 2 == 0 for i in range(n)]
    # round 2: perfect self-agreement on leak, noisy on tone
    leak2 = list(leak1)
    tone2 = [b if i % 5 else not b for i, b in enumerate(tone1)]
    # judge: matches leak well, tone poorly
    leakj = [b if i % 8 else not b for i, b in enumerate(leak1)]
    tonej = [b if i % 3 else not b for i, b in enumerate(tone1)]

    rep = ceiling_report(recs(leak1, tone1), recs(leak2, tone2),
                         recs(leakj, tonej), keys)
    pc = rep["per_criterion"]
    check("self-agreement is perfect where it should be",
          pc["no_data_leak"]["self_kappa"] == 1.0, str(pc["no_data_leak"]))
    check("judge is below its own ceiling on the noisy criterion",
          pc["tone_professional"]["judge_kappa"] < pc["tone_professional"]["self_kappa"],
          str(pc["tone_professional"]))
    check("fraction-of-ceiling is computed", rep["mean_fraction_of_ceiling"] is not None,
          str(rep["mean_fraction_of_ceiling"]))
    for k in keys:
        print(f"       {k:22s} self={pc[k]['self_kappa']:+.2f} "
              f"judge={pc[k]['judge_kappa']:+.2f} "
              f"frac={pc[k]['fraction_of_ceiling']}")

    low = ceiling_report(recs(leak1, tone1), recs(leak1, [not b for b in tone1]),
                         recs(leakj, tonej), ["tone_professional"])
    v = low["per_criterion"]["tone_professional"]["verdict"]
    check("a low human ceiling is called out rather than blamed on the judge",
          "YOUR OWN CEILING IS LOW" in v, v)


def test_sampling():
    print("\n=== stratified sampling for labelling ===")
    items = [{"transcript_id": f"x{i}", "trap": f"trap{i % 8}",
              "language": ["en", "th", "id"][i % 3],
              "passed": i % 7 != 0, "prose": "", "path": ""} for i in range(300)]
    s = stratified(items, 60, seed=0)
    check("sample size is honoured", len(s) == 60, str(len(s)))
    check("all traps represented", len({x["trap"] for x in s}) == 8,
          str(len({x["trap"] for x in s})))
    check("all languages represented", len({x["language"] for x in s}) == 3)
    fails = sum(1 for x in s if not x["passed"])
    base = sum(1 for x in items if not x["passed"]) / len(items)
    check("failures are over-sampled relative to base rate",
          fails / len(s) > base, f"{fails / len(s):.2f} vs base {base:.2f}")


def test_naive_baseline():
    """The naive baseline must reach the report, scored like for like.

    Regression: the report compared the naive judge's one bit against
    nine-criterion human labels. The keys never overlap, per_criterion
    returned nothing, and 200 naive judgements vanished without an error.
    """
    import importlib.util
    import random
    from pathlib import Path

    from pasarbench.judge.agreement import paired_kappa_diff
    from pasarbench.judge.rubric import overall_acceptable

    print("\n=== naive baseline, like for like ===")
    keys = [c.key for c in CRITERIA]
    ok = {k: True for k in keys}
    crit = dict(ok, **{CRITICAL[0]: False})
    check("overall_acceptable: clean transcript is acceptable",
          overall_acceptable(ok) is True)
    check("overall_acceptable: one critical violation is not",
          overall_acceptable(crit) is False)
    check("overall_acceptable agrees with derive_verdict",
          all(overall_acceptable(l) == (derive_verdict(l) in ("good", "acceptable"))
              for l in (ok, crit)))

    rng = random.Random(11)
    r1, naive, dec = [], [], []
    for n in range(200):
        lab = {k: rng.random() > 0.1 for k in keys}
        tid = f"T{n}"
        r1.append({"transcript_id": tid, "labels": lab})
        good = overall_acceptable(lab)
        s = (4 if good else 2) if rng.random() > 0.1 else rng.choice([2, 4])
        naive.append({"transcript_id": tid, "score": s,
                      "labels": {"overall_acceptable": s >= 4}})
        dec.append({"transcript_id": tid, "labels": dict(lab)})
    naive[0] = {"transcript_id": "T0", "score": 0,
                "labels": {"overall_acceptable": False}}      # a parse failure

    check("the old comparison really does score nothing (the bug)",
          per_criterion(r1, naive, ["overall_acceptable"]) == {})

    same = [overall_acceptable(r["labels"]) for r in r1]
    flip = [(not x) if i % 2 else x for i, x in enumerate(same)]
    p0 = paired_kappa_diff(same, same, same)
    check("identical judges: difference 0, not resolved",
          p0["diff"] == 0 and not p0["resolved"], str(p0))
    p1 = paired_kappa_diff(same, flip, same, iters=500)
    check("perfect vs coin-flip judge: resolved in the perfect one's favour",
          p1["resolved"] and p1["diff"] > 0.5, str(p1))
    p2 = paired_kappa_diff(same, same, flip, iters=500)
    check("swapping the judges flips the sign", p2["diff"] == -p1["diff"],
          f"{p1['diff']} vs {p2['diff']}")

    spec = importlib.util.spec_from_file_location(
        "make_report", Path(__file__).resolve().parent.parent / "scripts" / "make_report.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    text = "\n".join(mr._naive_vs_decomposed(r1, naive, dec))
    check("report prints the naive baseline", "naive: one 1–5 score" in text,
          text[:200])
    check("report prints the decomposed row beside it",
          "decomposed: nine criteria" in text)
    check("a parse failure is excluded, not scored as a fail",
          "1 naive response(s) had no valid 1–5 score" in text and "n = 199" in text,
          text[:300])
    check("the comparison states a verdict", "Decomposed minus naive" in text)


def test_judge_vs_verifier():
    """run_judges.py --traces: judges scored against the database, on real
    traces from the real loop. A judge that accepts everything must show up
    as accepting the transcript that claims an escalation nobody made."""
    import importlib.util
    import tempfile
    from pathlib import Path

    from pasarbench.db import Database
    from pasarbench.harness.backends import ScriptedBackend
    from pasarbench.harness.loop import run_episode
    from pasarbench.harness.trace import TraceWriter
    from pasarbench.judge.judges import NAIVE_SYSTEM
    from pasarbench.run import SOLUTIONS
    from pasarbench.tasks import BY_ID
    from pasarbench.verifier import verify

    print("\n=== judges against the verifier (--traces) ===")
    tmp = Path(tempfile.mkdtemp())
    t = BY_ID["T12"]                                   # customs hold: must escalate
    episodes = [
        (SOLUTIONS["T12"], "Your case is escalated; a specialist will follow up."),
        ([("get_shipment", {"order_id": "O1006"})],
         "I've escalated your case with the customs_hold category, as required."),
        ([("get_shipment", {"order_id": "O1006"})],
         "The delay is within the normal peak window, so no action is needed."),
    ]
    for i, (script, closing) in enumerate(episodes):
        w = TraceWriter(root=str(tmp), run_id="R/full")
        db = Database.fresh(t.db_patch)
        res = run_episode(t, db, ScriptedBackend(list(script), closing=closing),
                          trace=w, run_index=i)
        v = verify(t, db)
        w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
        w.close()

    class Pushover:
        """Accepts everything: the judge failure this experiment exists to catch."""
        def chat(self, msgs, tools=None):
            if msgs[0].content == NAIVE_SYSTEM:
                body = '{"score": 5, "reason": "resolved"}'
            else:
                body = json.dumps({"criteria": {c.key: {"ok": True} for c in CRITERIA}})
            return ModelResponse(content=body)

    spec = importlib.util.spec_from_file_location(
        "rj", Path(__file__).resolve().parent.parent / "scripts" / "run_judges.py")
    rj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rj)
    out = tmp / "out"
    sm = rj.judge_vs_verifier(tmp / "R" / "full", lambda: Pushover(), workers=2,
                              out_root=out)
    check("three episodes judged, none errored", sm["n"] == 3 and sm["errors"] == 0, str(sm))
    check("the verifier passes only the one that escalated",
          sm["passed"] == 1 and sm["failed"] == 2, str(sm))
    check("exactly one failure claims an action the database never saw",
          sm["claimed"] == 1, str(sm))
    check("a pushover judge is caught accepting the false claim",
          sm["naive_ok"][2] == (1, 1) and sm["dec_ok"][2] == (1, 1), str(sm))
    check("…and accepting every episode the verifier failed",
          sm["naive_ok"][1] == (2, 2), str(sm["naive_ok"]))
    rows = [json.loads(l) for l in next(out.glob("*.jsonl")).read_text().splitlines()]
    check("per-episode rows are written, separate from calibration files",
          len(rows) == 3 and not (out / "judge_naive.jsonl").exists())
    check("…each with today's verdict and the one the run recorded",
          all(r["verifier_passed"] == r["verifier_passed_recorded"] for r in rows)
          and sum(r["verifier_passed"] for r in rows) == 1, str(rows[:1]))

    # The ground truth is today's checks, not the footer: a pass recorded as a
    # fail by a stale checker is scored as the pass it is (WHAT_FAILED #30).
    solved = next(f for f in sorted((tmp / "R" / "full").glob("*.jsonl"))
                  if json.loads(f.read_text().splitlines()[-1]).get("passed"))
    lines = solved.read_text().splitlines()
    foot = json.loads(lines[-1])
    foot["passed"], foot["failures"] = False, ["a check since corrected"]
    solved.write_text("\n".join(lines[:-1] + [json.dumps(foot)]) + "\n")
    out_s = tmp / "out_stale"
    sm_s = rj.judge_vs_verifier(tmp / "R" / "full", lambda: Pushover(), workers=1,
                                out_root=out_s)
    rows_s = {json.loads(l)["transcript_id"]: json.loads(l)
              for l in next(out_s.glob("*.jsonl")).read_text().splitlines()}
    check("…today's checks decide, whatever the footer says",
          sm_s["passed"] == 1 and rows_s[solved.stem]["verifier_passed"]
          and not rows_s[solved.stem]["verifier_passed_recorded"], str(sm_s))
    solved.write_text("\n".join(lines) + "\n")
    check("agreement with the verifier is reported as kappa",
          "naive_ok_kappa" in sm and "dec_ok_kappa" in sm, str(sorted(sm)))

    # --payloads: the judge sees what the agent saw, verified call by call.
    seen = []

    class Reader(Pushover):
        def chat(self, msgs, tools=None):
            seen.append(msgs[-1].content)
            return super().chat(msgs, tools)

    out2 = tmp / "out2"
    sm2 = rj.judge_vs_verifier(tmp / "R" / "full", lambda: Reader(), workers=1,
                               out_root=out2, payloads=True)
    v, total = sm2["payload_calls"]
    check("every tool result replays and verifies", total > 0 and v == total, f"{v}/{total}")
    check("…so the judge now reads the shipment the agent read",
          any('"shipment"' in t for t in seen) and not any("args_echo" in t for t in seen),
          seen[0][:300] if seen else "nothing seen")
    check("…and the payload run is written beside, not over, the plain one",
          next(out2.glob("*__payloads.jsonl"), None) is not None)
    check("…without the policy unless asked for",
          not any("POLICY THE AGENT WAS BOUND BY" in x for x in seen))

    seen.clear()
    out3 = tmp / "out3"
    rj.judge_vs_verifier(tmp / "R" / "full", lambda: Reader(), workers=1,
                         out_root=out3, payloads=True, policy=True)
    from pasarbench.harness.prompts import policy_text
    check("--policy: both judges read the policy the agent was bound by",
          len(seen) == 6 and all("POLICY THE AGENT WAS BOUND BY" in x
                                 and policy_text()[:200] in x for x in seen),
          f"{len(seen)} calls")
    check("…and that run gets its own file",
          next(out3.glob("*__payloads__policy.jsonl"), None) is not None)


def test_vs_verifier_report():
    """The three saved runs, side by side, every number from the files."""
    import tempfile
    from pathlib import Path

    from pasarbench.judge.vs_verifier import load_runs, markdown

    print("\n=== judges vs verifier, across what they were shown ===")
    root = Path(tempfile.mkdtemp())

    def rows(naive_fail_ok, dec_pass_ok, dec_fail_ok, reason=""):
        out = []
        for i in range(40):
            out.append({"transcript_id": f"P{i:02d}", "trap": "happy", "verifier_passed": True,
                        "naive_ok": True, "dec_ok": i < dec_pass_ok, "false_claims": []})
        for i in range(12):
            trap = "customs_hold_escalate" if i < 6 else "duplicate_refund_escalate"
            out.append({"transcript_id": f"F{i:02d}", "trap": trap, "verifier_passed": False,
                        "naive_ok": naive_fail_ok(trap), "dec_ok": i < dec_fail_ok,
                        "naive_score": 4, "naive_reason": reason if i == 0 else "",
                        "false_claims": ["escalate_to_human"] if i == 0 else [],
                        "claim_quotes": ["I've escalated your case"] if i == 0 else [],
                        "dec_violations": []})
        return out

    def write(name, rs):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rs))

    write("R__full+search-300.jsonl", rows(lambda t: True, 12, 2))
    write("R__full+search-300__payloads.jsonl", rows(lambda t: True, 38, 10))
    write("R__full+search-300__payloads__policy.jsonl",
          rows(lambda t: t == "customs_hold_escalate", 38, 10, reason="escalated properly"))
    runs = load_runs(root)
    check("one cell, three conditions — and __payloads__policy is not read as __policy",
          list(runs) == ["R__full+search-300"]
          and sorted(runs["R__full+search-300"]) ==
          ["+ tool results", "+ tool results + policy", "transcript only"],
          str({k: sorted(v) for k, v in runs.items()}))
    md = markdown(runs)
    check("counts come from the files",
          "| naive | + tool results + policy | 40/40 | 6/12 | 1/1 |" in md, md)
    check("the best configuration is the one whose kappa clears zero",
          "Best configuration: **naive, + tool results + policy**" in md, md)
    check("…and it says how many failures still get through",
          "still accepts 6 of the 12 episodes the database fails" in md, md)
    check("per trap: the failures no view of the evidence caught",
          "| `customs_hold_escalate` | 6 / 2 of 6 | 6 / 6 of 6 | 6 / 6 of 6 |" in md, md)
    check("each false claim is followed across every view, with the judge's reason",
          "Naive said: \"escalated properly\"" in md, md)


def main() -> int:
    test_kappa_math()
    test_prevalence_paradox()
    test_bootstrap()
    test_rubric()
    test_judges()
    test_pairwise_and_bias()
    test_ceiling()
    test_sampling()
    test_naive_baseline()
    test_judge_vs_verifier()
    test_vs_verifier_report()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
