"""
Generated-suite validation.

At 16 hand-written tasks you can check the suite by reading it. At 170 you
cannot, so the guarantee has to be mechanical:

    every generated task is solved by its own generated reference solution
    every generated task is failed by the null agent and by tool spam
    locale twins share byte-identical checks and an identical world

The third one is what makes the multilingual number mean anything. If a Thai
task were even slightly easier or harder than its English twin, the gap you
measure would be task difficulty wearing a language costume.

Run: python -m tests.test_generated
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pasarbench.db import NOW, Database, to_sgd, ts
from pasarbench.generate import (DELIVERED_IN_WINDOW, DELIVERED_OUT_WINDOW,
                                 CUSTOMS_LAST_SCAN, MARKETS, generate,
                                 stratified_sample)
from pasarbench.harness.backends import ScriptedBackend
from pasarbench.harness.loop import run_episode
from pasarbench.locales import NEEDS_NATIVE_REVIEW, coverage_report
from pasarbench.tools import call
from pasarbench.verifier import verify

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


TASKS, SOLUTIONS = generate()


def run_script(task, script):
    db = Database.fresh(task.db_patch)
    for name, args in script:
        call(db, name, args)
    return verify(task, db)


def test_all_solvable():
    print(f"\n=== every one of {len(TASKS)} generated tasks is solvable ===")
    failures = []
    for t in TASKS:
        if not run_script(t, SOLUTIONS[t.task_id]).passed:
            failures.append(t.task_id)
    check("reference solution passes every generated task", not failures,
          f"{len(failures)} failed: {failures[:8]}")

    by_trap = defaultdict(list)
    for tid in failures:
        by_trap[next(t.trap for t in TASKS if t.task_id == tid)].append(tid)
    if by_trap:
        for trap, ids in by_trap.items():
            print(f"       {trap}: {len(ids)} -> {ids[:4]}")


def test_solutions_run_through_harness():
    print("\n=== solutions also work through the real loop, not just direct calls ===")
    sample = stratified_sample(TASKS, per_trap=1)
    bad = []
    for t in sample:
        db = Database.fresh(t.db_patch)
        run_episode(t, db, ScriptedBackend(SOLUTIONS[t.task_id]))
        if not verify(t, db).passed:
            bad.append(t.task_id)
    check(f"all {len(sample)} sampled tasks pass through the harness", not bad, str(bad))


def test_null_and_spam_fail():
    print("\n=== nothing passes by accident ===")
    passed_empty = [t.task_id for t in TASKS if run_script(t, []).passed]
    check("no task passes with zero actions", not passed_empty,
          f"{len(passed_empty)}: {passed_empty[:8]}")

    def spam(t):
        return [
            ("verify_identity", {"user_id": t.user_id, "phone_last4": "0000"}),
            ("get_order", {"order_id": t.hidden_facts["order_id"]}),
            ("get_shipment", {"order_id": t.hidden_facts["order_id"]}),
            ("get_payment", {"order_id": t.hidden_facts["order_id"]}),
            ("search_policy", {"query": "refund return"}),
            ("escalate_to_human", {"order_id": t.hidden_facts["order_id"],
                                   "category": "other", "reason": "x"}),
        ]

    spam_pass = [t.task_id for t in TASKS if run_script(t, spam(t)).passed]
    # Read-only "do no harm" traps are legitimately satisfiable by a cautious
    # agent that looks things up and takes no action -- that IS correct
    # behaviour there. Everything else must not be.
    benign = {"cannot_cancel_shipped_order", "address_change_after_dispatch",
              "peak_period_delay_not_compensable"}
    bad = [tid for tid in spam_pass
           if next(t.trap for t in TASKS if t.task_id == tid) not in benign]
    check("tool spam passes no action-requiring task", not bad,
          f"{len(bad)}: {bad[:8]}")
    print(f"       ({len(spam_pass)} benign do-no-harm tasks satisfied by lookup-only, "
          f"as expected)")


def test_locale_twins_identical():
    print("\n=== locale twins differ ONLY in the opening ===")
    by_base = defaultdict(list)
    for t in TASKS:
        by_base[t.task_id.split(".")[0]].append(t)
    twins = {k: v for k, v in by_base.items() if len(v) > 1}
    check("twin groups exist", len(twins) >= 60, str(len(twins)))

    same_checks = all(all(x.checks is g[0].checks for x in g) for g in twins.values())
    check("twins share the identical checks object", same_checks)

    same_world = all(all(x.db_patch == g[0].db_patch for x in g) for g in twins.values())
    check("twins share an identical world", same_world)

    same_persona = all(all(x.persona == g[0].persona for x in g) for g in twins.values())
    check("twins share an identical persona", same_persona)

    differ = all(len({x.opening for x in g}) == len(g) for g in twins.values())
    check("every twin has a distinct opening", differ,
          str([k for k, g in twins.items() if len({x.opening for x in g}) != len(g)][:4]))

    same_sol = all(all(SOLUTIONS[x.task_id] == SOLUTIONS[g[0].task_id] for x in g)
                   for g in twins.values())
    check("twins share an identical reference solution", same_sol)


def test_market_thresholds():
    print("\n=== policy thresholds land correctly in all six currencies ===")
    for m, cfg in MARKETS.items():
        c = cfg["currency"]
        check(f"{m}: normal item is under the SGD 200 photo rule",
              to_sgd(cfg["normal"], c) < 200, f"{to_sgd(cfg['normal'], c):.1f}")
        check(f"{m}: high item is over the SGD 200 photo rule",
              to_sgd(cfg["high"], c) >= 200, f"{to_sgd(cfg['high'], c):.1f}")
        check(f"{m}: voucher is under the SGD 15 cap",
              to_sgd(cfg["voucher"], c) <= 15, f"{to_sgd(cfg['voucher'], c):.2f}")


def test_date_invariants():
    print("\n=== date invariants ===")
    d_in = (NOW - ts(DELIVERED_IN_WINDOW)).days
    d_out = (NOW - ts(DELIVERED_OUT_WINDOW)).days
    d_cus = (NOW - ts(CUSTOMS_LAST_SCAN)).days
    check("in-window delivery is inside 14 days", d_in <= 14, f"{d_in}d")
    check("out-of-window delivery is outside 14 days", d_out > 14, f"{d_out}d")
    check("customs hold clears the >5 day rule", d_cus > 5, f"{d_cus}d")
    check("out-of-window is not marginal", d_out >= 18, f"{d_out}d")


def test_coverage():
    print("\n=== coverage and provenance ===")
    cov = coverage_report()
    check("no missing (trap, language) cell", not cov["missing"], str(cov["missing"][:5]))
    langs = Counter(t.language for t in TASKS)
    check("eight language varieties present, incl. zh-SG and zh-MY",
          len(langs) == 8 and {"zh-SG", "zh-MY"} <= set(langs), str(dict(langs)))
    check("non-English is a substantial share",
          sum(v for k, v in langs.items() if k != "en") >= 90,
          str(sum(v for k, v in langs.items() if k != "en")))
    check("unreviewed languages are declared",
          NEEDS_NATIVE_REVIEW == {"th", "vi"}, str(NEEDS_NATIVE_REVIEW))
    print(f"       !! th and vi are DRAFTED, NOT NATIVE-REVIEWED. Get them "
          f"reviewed before quoting a per-language number.")

    sample = stratified_sample(TASKS, per_trap=2)
    check("stratified sample covers every trap",
          len({t.trap for t in sample}) == 16, str(len({t.trap for t in sample})))
    check("stratified sample is small enough to iterate on", len(sample) <= 40,
          str(len(sample)))


def test_simulator_qa():
    """The leak detector must catch a leaky simulator and clear a good one.

    Built with synthetic transcripts so it runs offline. Point it at real
    transcripts every time the persona prompt or simulator model changes."""
    print("\n=== simulator QA: leak detection ===")
    from pasarbench.harness.types import Message
    from pasarbench.simqa import audit, leak_report

    task = next(t for t in TASKS if t.trap == "happy_path_return_refund"
                and t.language == "en")
    oid = task.hidden_facts["order_id"]
    ph = task.hidden_facts["phone_last4"]

    leaky = [
        Message("system", "..."),
        Message("user", f"Hi, I want to return order {oid}, my phone ends {ph}."),
        Message("assistant", "Sure, let me look that up."),
        Message("user", "Thanks"),
    ]
    r = leak_report(task, leaky)
    check("volunteered order id in the opening is flagged",
          any(f == "order_id" for f, _ in r.leaked), str(r.leaked))
    check("volunteered phone digits are flagged",
          any(f == "phone_last4" for f, _ in r.leaked), str(r.leaked))
    check("a leaky transcript is not clean", not r.clean)

    good = [
        Message("system", "..."),
        Message("user", "Hi, I want to return the blouse I bought last week."),
        Message("assistant", "Happy to help. Could you give me the order number?"),
        Message("user", f"It is {oid} I think"),
        Message("assistant", "Thanks. For security, the last 4 digits of the phone on the account?"),
        Message("user", ph),
    ]
    g = leak_report(task, good)
    check("facts given ON REQUEST are not counted as leaks", not g.leaked, str(g.leaked))
    check("on-request reveals are recorded separately",
          len(g.revealed_on_request) == 2, str(g.revealed_on_request))
    check("a well-behaved transcript is clean", g.clean)

    coach = [
        Message("system", "..."),
        Message("user", "As an AI I should tell you per the policy section P4.2 to use store credit."),
    ]
    c = leak_report(task, coach)
    check("a simulator that coaches the agent is flagged",
          bool(c.broke_character), str(c.broke_character))

    known = {oid}
    fake = [Message("system", "..."), Message("user", "My order GO-FAKE9999 is late")]
    f = leak_report(task, fake, known_ids=known)
    check("hallucinated order ids are flagged", bool(f.invented_ids), str(f.invented_ids))

    a = audit([r, g, c, f])
    check("audit folds to publishable rates",
          a["leak_rate"] == 0.25 and a["clean_rate"] == 0.25, str(a))


def main() -> int:
    test_all_solvable()
    test_solutions_run_through_harness()
    test_null_and_spam_fail()
    test_locale_twins_identical()
    test_market_thresholds()
    test_date_invariants()
    test_coverage()
    test_simulator_qa()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
