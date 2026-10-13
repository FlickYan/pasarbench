"""
Tool exposure + multilingual diagnosis tests.

The important assertions here are about EXPERIMENTAL DESIGN, not code:
`random-N` must have the same token cost as `all-N` and the same reachability
as `oracle`, or the arm cannot separate the confound it exists to separate.

Run: python -m tests.test_exposure
"""

from __future__ import annotations

from pasarbench.db import Database
from pasarbench.diagnose import (attribute_gap, by_language, report,
                                 tool_scaling_report)
from pasarbench.harness.backends import ScriptedBackend
from pasarbench.harness.exposure import (AllTools, CORE_VISIBLE, OracleTools,
                                         RandomSubset, ToolSearch, _rank_tools,
                                         build_exposure, distractor_names,
                                         register_distractors, schema_tokens)
from pasarbench.harness.loop import run_episode
from pasarbench.run import SOLUTIONS
from pasarbench.tasks import BY_ID
from pasarbench.tools import DEFAULT_TOOLS, TOOLS, call

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


TASK = BY_ID["T08"]


def test_distractors():
    print("\n=== distractors are plausible and inert ===")
    names = register_distractors(300)
    check("300 distractors available", len(names) == 300, str(len(names)))
    check("names are unique", len(set(names)) == 300)
    check("no distractor collides with a real tool",
          not (set(names) & set(DEFAULT_TOOLS)))

    db = Database.fresh()
    before = {t: dict(rows) for t, rows in db.tables.items()}
    for n in names[:40]:
        call(db, n, {"entity_id": "X1"})
    mutated = [t for t in before if db.tables[t] != before[t]]
    check("calling 40 distractors mutates no world table", not mutated, str(mutated))

    r = call(db, names[0], {"entity_id": "X1"})
    check("distractors succeed rather than erroring",
          r["ok"] is True,
          "an erroring distractor would teach the agent that unfamiliar tools "
          "fail, which is a different lesson from the one under test")


def test_arms_are_comparable():
    print("\n=== the four arms separate the confound ===")
    orc = OracleTools(SOLUTIONS)
    a100 = AllTools(100 - len(DEFAULT_TOOLS))
    r100 = RandomSubset(100, SOLUTIONS)
    db = Database.fresh(TASK.db_patch)

    t_orc = orc.tools_for(None, db, TASK)
    t_all = a100.tools_for(None, db, TASK)
    t_rnd = r100.tools_for(None, db, TASK)

    # The oracle may remove CHOICES, never information: every lookup, plus
    # only the write actions this task's solution takes.
    from pasarbench.harness.exposure import LOOKUPS
    writes_used = {n for n, _ in SOLUTIONS[TASK.task_id]} - LOOKUPS
    check("oracle exposes every lookup", LOOKUPS <= set(t_orc),
          str(LOOKUPS - set(t_orc)))
    check("oracle exposes only the write actions the task takes",
          set(t_orc) - LOOKUPS == writes_used & set(TOOLS),
          f"{sorted(set(t_orc) - LOOKUPS)} vs {sorted(writes_used)}")
    check("oracle is smaller than the full registry",
          len(t_orc) < len(DEFAULT_TOOLS), str(len(t_orc)))
    needed = {n for n, _ in SOLUTIONS[TASK.task_id]}
    check("oracle contains every needed tool", needed <= set(t_orc),
          str(needed - set(t_orc)))
    check("random-100 also contains every needed tool", needed <= set(t_rnd),
          str(needed - set(t_rnd)))
    check("random-100 and all-100 expose the same COUNT",
          len(t_rnd) == len(t_all), f"{len(t_rnd)} vs {len(t_all)}")

    s_all, s_rnd = schema_tokens(t_all), schema_tokens(t_rnd)
    check("random-100 and all-100 cost roughly the same tokens",
          abs(s_all - s_rnd) / max(s_all, 1) < 0.15, f"{s_rnd} vs {s_all}")
    check("oracle is far cheaper than either",
          schema_tokens(t_orc) < s_all / 4, f"{schema_tokens(t_orc)} vs {s_all}")

    r2 = RandomSubset(100, SOLUTIONS).tools_for(None, db, TASK)
    check("random subset is deterministic per task (reproducible arm)",
          sorted(r2) == sorted(t_rnd))
    other = RandomSubset(100, SOLUTIONS).tools_for(None, db, BY_ID["T01"])
    check("different tasks get different random subsets", sorted(other) != sorted(t_rnd))


def test_progressive_disclosure():
    print("\n=== tool search reveals tools progressively ===")
    ts = ToolSearch(300)
    db = Database.fresh(TASK.db_patch)

    visible0 = ts.tools_for(None, db, TASK)
    check("only a small core is visible initially",
          len(visible0) == len([t for t in CORE_VISIBLE if t in TOOLS]),
          str(len(visible0)))
    check("issue_refund is NOT visible before searching",
          "issue_refund" not in visible0)

    call(db, "search_tools", {"query": "refund money to the customer"})
    visible1 = ts.tools_for(None, db, TASK)
    check("searching reveals more tools", len(visible1) > len(visible0),
          f"{len(visible0)} -> {len(visible1)}")
    check("the searched-for tool becomes visible",
          "issue_refund" in visible1, str(visible1))

    call(db, "search_tools", {"query": "livestream seller claim"})
    visible2 = ts.tools_for(None, db, TASK)
    check("a second search adds more", len(visible2) >= len(visible1))
    check("visibility is derived from the log, so it survives resume",
          ts.tools_for(None, Database.from_dict(db.to_dict()), TASK) == visible2)

    check("search-300 injects far fewer schema tokens than all-300",
          schema_tokens(visible2) < schema_tokens(AllTools(280).tools_for(None, db, TASK)) / 5)


def test_search_ranking_is_english_only():
    print("\n=== the retrieval mechanism behind the multilingual gap ===")
    en = _rank_tools("refund money to the customer")
    check("an English query retrieves relevant tools", bool(en), str(en[:2]))
    check("issue_refund ranks for an English refund query",
          any(n == "issue_refund" for n, _ in en), str([n for n, _ in en]))

    th = _rank_tools("คืนเงินให้ลูกค้า")
    vi = _rank_tools("hoàn tiền cho khách hàng")
    check("the same query in Thai retrieves NOTHING", not th, str(th))
    check("the same query in Vietnamese retrieves nothing useful",
          not any(n == "issue_refund" for n, _ in vi), str([n for n, _ in vi]))
    print("       ^ this is a REAL property of a keyword retriever over English")
    print("         tool descriptions -- and one of the mechanisms behind the")
    print("         multilingual gap, not a bug to hide")


def test_loop_records_exposure():
    print("\n=== the loop records exposure per step ===")
    for spec, lo, hi in (("oracle", 12, 16), ("all-20", 18, 22), ("all-300", 290, 310)):
        db = Database.fresh(TASK.db_patch)
        res = run_episode(TASK, db, ScriptedBackend(SOLUTIONS["T08"]),
                          exposure=build_exposure(spec, SOLUTIONS))
        st = res.steps[0]
        check(f"{spec}: n_tools recorded in range", lo <= st.n_tools <= hi,
              str(st.n_tools))
        check(f"{spec}: schema token cost recorded", st.schema_tokens > 0)

    db = Database.fresh(TASK.db_patch)
    res = run_episode(TASK, db, ScriptedBackend(SOLUTIONS["T08"]),
                      exposure=build_exposure("all-300", SOLUTIONS))
    check("the reference solution still passes with 300 tools exposed",
          __import__("pasarbench.verifier", fromlist=["verify"]).verify(TASK, db).passed)

    # Regression: the loop sent only the visible schemas but DISPATCHED any
    # registered tool by name, so every reduced arm leaked to a good guesser.
    from pasarbench.harness.types import ToolCall
    orc = build_exposure("oracle", SOLUTIONS)
    hidden = sorted(set(DEFAULT_TOOLS) - set(orc.tools_for(None, None, TASK)))[0]
    db = Database.fresh(TASK.db_patch)
    before = len(db.action_log)
    res = run_episode(TASK, db, ScriptedBackend([(hidden, {"order_id": "X"})]),
                      exposure=orc)
    tr = [r for st in res.steps for r in st.tool_results if r["name"] == hidden]
    check(f"a hidden tool ({hidden}) cannot be called by guessing its name",
          tr and not tr[0]["ok"] and "unknown tool" in (tr[0]["error"] or ""), str(tr))
    check("…and it never touches the database", len(db.action_log) == before,
          f"{before} -> {len(db.action_log)}")


def _ep(lang, passed, **kw):
    base = dict(transcript_id=f"{lang}-{id(kw)}", language=lang, market="TH",
                trap="t", passed=passed, stop_reason="done", tokens=9000, steps=5,
                malformed_args=0, search_calls=2, search_misses=0,
                prose_chars=500, user_chars=200, first_failure=None)
    base.update(kw)
    return base


def test_diagnosis():
    print("\n=== multilingual attribution ===")
    eps = [_ep("en", True) for _ in range(20)] + [_ep("en", False) for _ in range(5)]
    # Thai: worse, and the search misses co-move with it
    eps += [_ep("th", True, search_misses=0) for _ in range(12)]
    eps += [_ep("th", False, search_misses=2) for _ in range(13)]

    table = by_language(eps)
    check("per-language table built", set(table) == {"en", "th"}, str(list(table)))
    check("Thai pass rate is lower", table["th"]["pass_rate"] < table["en"]["pass_rate"])
    check("search miss rate is computed", table["th"]["search_miss_rate"] > 0,
          str(table["th"]["search_miss_rate"]))

    attr = attribute_gap(table)
    row = attr["languages"]["th"]
    check("the gap is quantified", row["pass_gap"] > 0.1, str(row["pass_gap"]))
    check("search_miss is identified as co-moving",
          "search_miss" in row["ranked_mechanisms"], str(row["ranked_mechanisms"]))
    check("the next step names the English-keyword-retriever mechanism",
          "keyword matcher" in row["next_step"], row["next_step"])
    check("the causal caveat is carried in the output",
          "CO-MOVEMENTS" in attr["caveat"])

    # budget exhaustion must take priority: it is a harness artefact
    eps2 = [_ep("en", True) for _ in range(20)]
    eps2 += [_ep("th", False, stop_reason="max_steps") for _ in range(15)]
    eps2 += [_ep("th", True) for _ in range(5)]
    row2 = attribute_gap(by_language(eps2))["languages"]["th"]
    check("budget exhaustion is flagged FIRST, above every model explanation",
          "CHECK THIS FIRST" in row2["next_step"], row2["next_step"])

    # tokenisation inflation
    eps3 = [_ep("en", True, tokens=9000, prose_chars=500, user_chars=200)
            for _ in range(20)]
    eps3 += [_ep("vi", False, tokens=16000, prose_chars=500, user_chars=200)
             for _ in range(10)]
    eps3 += [_ep("vi", True, tokens=16000, prose_chars=500, user_chars=200)
             for _ in range(10)]
    t3 = by_language(eps3)
    infl = t3["vi"]["tokens_per_char"] / t3["en"]["tokens_per_char"]
    check("tokenisation inflation is measured", infl > 1.4, f"{infl:.2f}x")

    judge = [{"transcript_id": e["transcript_id"],
              "labels": {"language_match": e["language"] == "en"}} for e in eps]
    t4 = by_language(eps, judge)
    check("language drift comes from the judge, not from state checks",
          t4["th"]["language_drift_rate"] == 1.0,
          str(t4["th"]["language_drift_rate"]))

    md = report(eps, judge)
    check("report renders", "Multilingual diagnosis" in md and "`th`" in md)
    # These episodes have no locale twins, so nothing is controlled and nothing
    # may be attributed; the caveat travels with an attribution when there is
    # one (tests/test_report.py covers that case).
    check("without twins the report attributes nothing, and says why",
          "No locale twins" in md and "co-moving" not in md, md[md.find("### Gap"):][:300])


def test_scaling_report():
    """Contrasts are read only after the loss clears a paired sign test, and
    the composition verdict points the right way round (WHAT_FAILED #21)."""
    from pasarbench.diagnose import _sign_p

    print("\n=== tool scaling reads contrasts only after a paired test ===")
    check("sign test: 4 tasks worse, 0 better is NOT significant", _sign_p(4, 0) > 0.1,
          str(_sign_p(4, 0)))
    check("sign test: 9 worse, 1 better is", _sign_p(9, 1) < 0.05, str(_sign_p(9, 1)))
    check("sign test: no differing tasks is p=1", _sign_p(0, 0) == 1.0)

    traps = [f"t{i:02d}" for i in range(16)]

    def row(exp, n_tools, schema, tokens, steps, lose=()):
        per = {t: (0.5 if t in lose else 1.0) for t in traps}
        return {"exposure": exp, "cell": f"full+{exp}", "n_tools": n_tools,
                "mean_schema_tokens": schema, "mean_tokens": tokens,
                "mean_steps": steps, "pass^1": sum(per.values()) / len(per),
                "stop_reasons": {"done": 96}, "per_trap": per}

    lost = set(traps[:8])
    base = [row("oracle", 14, 1200, 25000, 7.0), row("all-20", 20, 1760, 38000, 7.7)]
    count = base + [row("all-100", 100, 9670, 100000, 8.1, lost),
                    row("random-100", 100, 9750, 89000, 7.1, lost),
                    row("search-300", 5, 790, 46000, 10.4, set(traps[:7]))]
    md = tool_scaling_report(count)
    check("a loss that persists in random-N is read as the NUMBER of tools",
          "NUMBER of tools" in md, md)
    check("…not as the unneeded real tools", "unneeded REAL tools:" not in md)

    comp = base + [row("all-100", 100, 9670, 100000, 8.1, lost),
                   row("random-100", 100, 9750, 89000, 7.1)]
    md2 = tool_scaling_report(comp)
    check("a loss random-N recovers is read as the unneeded REAL tools",
          "unneeded REAL tools:" in md2, md2)

    check("search is compared with the registry", "`search-300` vs `all-20`" in md)
    check("…on total tokens, not only per-call schema", "Total tokens per episode" in md)
    check("worse AND costlier is called dominated", "Dominated at this registry size" in md,
          md)
    print("\n" + "\n".join(l for l in md.splitlines() if l.startswith("- ")))


def test_scaling_report_on_itools2():
    """Your I-tools2 numbers. The old reading called a 5-episode gap
    'SELECTION difficulty, not token cost'."""
    print("\n=== I-tools2: a 5-episode gap is inconclusive, and says so ===")
    names = ["address_change_after_dispatch", "cancel_while_processing",
             "cannot_cancel_shipped_order", "cod_cancel_no_refund_due",
             "cod_cannot_refund_to_original_method", "customs_hold_escalate",
             "duplicate_refund_escalate", "happy_path_return_refund",
             "hazmat_refund_without_return", "high_value_photo_required_first",
             "identity_verification_failure", "livestream_claim_overrides_window",
             "out_of_window_dispute_escalate", "out_of_window_offer_voucher",
             "peak_period_delay_not_compensable", "perishable_refund_without_return"]

    def pt(**low):
        return {t: low.get(t, 1.0) for t in names}

    rows = [
        {"exposure": "oracle", "cell": "full+oracle", "pass^1": 0.979, "mean_tokens": 25285,
         "mean_steps": 7.16, "mean_schema_tokens": 452, "stop_reasons": {"done": 96},
         "per_trap": pt(duplicate_refund_escalate=0.67)},
        {"exposure": "all-20", "cell": "full+all-20", "pass^1": 0.958, "mean_tokens": 38041,
         "mean_steps": 7.73, "mean_schema_tokens": 1762, "stop_reasons": {"done": 96},
         "per_trap": pt(duplicate_refund_escalate=0.67, livestream_claim_overrides_window=0.83,
                        out_of_window_offer_voucher=0.83)},
        {"exposure": "all-100", "cell": "full+all-100", "pass^1": 0.927, "mean_tokens": 101708,
         "mean_steps": 8.11, "mean_schema_tokens": 9670, "stop_reasons": {"done": 96},
         "per_trap": pt(duplicate_refund_escalate=0.67, high_value_photo_required_first=0.83,
                        identity_verification_failure=0.50,
                        livestream_claim_overrides_window=0.83)},
        {"exposure": "random-100", "cell": "full+random-100", "pass^1": 0.927,
         "mean_tokens": 88974, "mean_steps": 7.14, "mean_schema_tokens": 9753,
         "stop_reasons": {"done": 96},
         "per_trap": pt(duplicate_refund_escalate=0.67, identity_verification_failure=0.67,
                        livestream_claim_overrides_window=0.83,
                        out_of_window_offer_voucher=0.67)},
    ]
    md = tool_scaling_report(rows)
    check("the 100-tool loss is INCONCLUSIVE", "INCONCLUSIVE at 100 tools" in md, md)
    check("…and no mechanism is named", "SELECTION" not in md and "NUMBER of tools" not in md, md)
    check("the cost, which IS certain, is stated", "(4.0x)" in md, md)
    check("the trap both 100-tool arms lose on is flagged as a lead",
          "`identity_verification_failure` (oracle 1.00, all 0.50, random 0.67)" in md, md)


def test_reduced_arms_are_discoverable():
    """The oracle must be a ceiling for an agent that has to FIND OUT.

    Regression: the oracle exposed the tools the reference solution calls. The
    reference solution hard-codes order_item_id="OI3" and never calls
    get_order_items, so the arm hid the only way a real agent learns the item
    id -- four traps scored exactly 0.00 and the "ceiling" came in below all-20.
    test_arms_are_comparable checked the oracle holds what the ORACLE calls,
    which is the same wrong assumption, and passed throughout.
    """
    from pasarbench.harness.exposure import ARG_SOURCES
    from pasarbench.sweep import ALL_SOLUTIONS, ALL_TASKS

    print("\n=== reduced arms expose what a discovering agent needs ===")
    orc, rnd = OracleTools(ALL_SOLUTIONS), RandomSubset(100, ALL_SOLUTIONS)
    gaps = []
    for t in ALL_TASKS:
        sol = ALL_SOLUTIONS.get(t.task_id, [])
        must = {ARG_SOURCES[k] for _, a in sol for k in a if k in ARG_SOURCES}
        for arm in (orc, rnd):
            missing = must - set(arm.tools_for(None, None, t))
            if missing:
                gaps.append(f"{arm.name}:{t.task_id}:{sorted(missing)}")
    check(f"every hard-coded id is discoverable, both arms, all {len(ALL_TASKS)} tasks",
          not gaps, "; ".join(gaps[:5]))
    hp = next(t for t in ALL_TASKS if t.trap == "happy_path_return_refund")
    check("oracle for a return exposes get_order_items",
          "get_order_items" in orc.tools_for(None, None, hp))
    cancel = next(t for t in ALL_TASKS if t.trap == "cancel_while_processing")
    t_cancel = set(orc.tools_for(None, None, cancel))
    check("a cancel task's oracle hides the write actions it does not take",
          not {"issue_refund", "initiate_return", "issue_goodwill_voucher"} & t_cancel,
          str(sorted(t_cancel)))


def test_lookups_are_read_only():
    """LOOKUPS is exposed in every reduced arm on the promise that none of them
    changes data. Hold each one to it, against a live database."""
    import copy

    from pasarbench.harness.exposure import LOOKUPS

    print("\n=== every lookup leaves the data unchanged ===")
    t = BY_ID["T01"]
    db = Database.fresh(t.db_patch)
    args = {"get_user_profile": {"user_id": "U002"}, "get_order": {"order_id": "O1003"},
            "list_user_orders": {"user_id": "U002"}, "get_order_items": {"order_id": "O1003"},
            "get_shipment": {"order_id": "O1003"}, "get_product": {"product_id": "P011"},
            "get_payment": {"order_id": "O1003"}, "search_policy": {"query": "return window"},
            "get_livestream_claims": {"livestream_id": "LS1", "product_id": "P011"},
            "check_return_eligibility": {"order_id": "O1003", "order_item_id": "OI3"},
            "calculate_refund_amount": {"order_id": "O1003", "order_item_id": "OI3",
                                        "include_shipping": False},
            "verify_identity": {"user_id": "U002", "phone_last4": "4567"}}
    check("every lookup has a test call", set(args) == LOOKUPS, str(LOOKUPS ^ set(args)))
    for name in sorted(LOOKUPS):
        before = copy.deepcopy(db.tables)
        out = call(db, name, args[name])
        # ok must be True: a call that errors on bad arguments leaves the
        # tables unchanged trivially and would prove nothing.
        check(f"{name}: succeeded, and no table changed",
              out.get("ok") is True and db.tables == before, str(out)[:120])
    check("every lookup is a real registry tool", LOOKUPS <= set(DEFAULT_TOOLS),
          str(LOOKUPS - set(DEFAULT_TOOLS)))


def test_search_is_confined_to_its_arm():
    """Regression: the ranker searched the GLOBAL registry. After random-100
    registered 300 distractors, "search-300" ranked over 320 tools, and its
    difficulty depended on which arms ran before it in the same process."""
    from pasarbench.harness.exposure import (ToolSearch, _rank_tools,
                                             distractor_names, register_distractors)

    print("\n=== search ranks only within its own arm ===")
    ts = ToolSearch(280, seed=0)
    register_distractors(300, 0)                      # what random-100 does
    extra = set(distractor_names(300, 0)[280:])
    leaked = set()
    for q in ("refund", "escalate to a human", "support handoff history",
              "lookup records report", "voucher campaign pool"):
        leaked |= {n for n, _ in _rank_tools(q, 50, universe=ts.universe)} & extra
    check("no distractor from outside the arm is ever ranked", not leaked, str(leaked))

    db = Database.fresh(TASK.db_patch)
    ts.tools_for(None, db, TASK)                      # the loop does this each step
    out = call(db, "search_tools", {"query": "support handoff history", "limit": 50})
    shown = {x["name"] for x in out.get("tools", [])}
    check("what search_tools SHOWS the agent comes from the same pool",
          shown and shown <= ts.universe and not shown & extra,
          str(sorted(shown & extra)))


def test_scaling_report_refuses_a_broken_ceiling():
    """Your I-tools numbers, fed back in. The old report read them as
    'no degradation at 100 tools' because every arm beat a broken oracle."""
    print("\n=== the report refuses to read against a broken ceiling ===")
    zero4 = {"cod_cannot_refund_to_original_method": 0.0, "happy_path_return_refund": 0.0,
             "high_value_photo_required_first": 0.0, "out_of_window_offer_voucher": 0.0,
             "customs_hold_escalate": 1.0}
    ok4 = {"cod_cannot_refund_to_original_method": 1.0, "happy_path_return_refund": 1.0,
           "high_value_photo_required_first": 0.67, "out_of_window_offer_voucher": 0.83,
           "customs_hold_escalate": 1.0}
    rows = [
        {"exposure": "oracle", "cell": "full+oracle", "n_tools": 5, "mean_schema_tokens": 424,
         "mean_tokens": 33128, "pass^1": 0.656, "stop_reasons": {"done": 96}, "per_trap": zero4},
        {"exposure": "all-20", "cell": "full+all-20", "n_tools": 20, "mean_schema_tokens": 1762,
         "mean_tokens": 38848, "pass^1": 0.938, "stop_reasons": {"done": 96}, "per_trap": ok4},
        {"exposure": "all-100", "cell": "full+all-100", "n_tools": 100, "mean_schema_tokens": 9670,
         "mean_tokens": 95731, "pass^1": 0.854, "stop_reasons": {"done": 83, "max_tokens": 13},
         "per_trap": ok4},
        {"exposure": "random-100", "cell": "full+random-100", "n_tools": 100,
         "mean_schema_tokens": 9772, "mean_tokens": 97378, "pass^1": 0.698,
         "stop_reasons": {"done": 66, "max_tokens": 30}, "per_trap": zero4},
        {"exposure": "search-300", "cell": "full+search-300", "n_tools": 5,
         "mean_schema_tokens": 715, "mean_tokens": 40330, "pass^1": 0.781,
         "stop_reasons": {"done": 96},
         "per_trap": dict(ok4, customs_hold_escalate=0.33, out_of_window_offer_voucher=0.33)},
    ]
    md = tool_scaling_report(rows)
    check("a ceiling the registry beats is called out", "not a ceiling" in md, md)
    check("…and no verdict is read against it", "no degradation" not in md, md)
    check("the structural zeros are named", "`happy_path_return_refund`" in md)
    check("budget stops are separated from tool choice",
          "13 of 96 episodes stopped on the per-episode budget" in md, md)
    check("search losses are listed, rounding-safe (0.83 - 0.33)",
          "`out_of_window_offer_voucher`" in md.split("Loses")[-1], md)


def test_audit_uses_todays_verdicts():
    """#30: the audit's "failed" is RESULTS.md's -- today's checks on the
    replayed episode -- and a requirement a visible tool can meet is not a tool
    the arm hid. The photo check now accepts get_order, which search-N shows."""
    import contextlib
    import importlib.util
    import io
    import json
    import tempfile
    from pathlib import Path

    from pasarbench.harness.replay import pre_v19_patch
    from pasarbench.harness.trace import TraceWriter
    from pasarbench.sweep import ALL_SOLUTIONS, ALL_TASKS

    print("\n=== the audit reads today's verdicts ===")
    spec = importlib.util.spec_from_file_location(
        "audit", Path(__file__).resolve().parent.parent / "scripts" / "audit_tool_arms.py")
    au = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(au)
    task = next(t for t in ALL_TASKS if t.task_id == "HVPRF-SG")
    tmp = tempfile.mkdtemp()
    w = TraceWriter(root=tmp, run_id="I-x/full+search-300")
    run_episode(task, Database.fresh(pre_v19_patch(task)), ScriptedBackend([
        ("verify_identity", {"user_id": task.user_id,
                             "phone_last4": task.hidden_facts["phone_last4"]}),
        ("get_order", {"order_id": task.hidden_facts["order_id"]})]),
        exposure=build_exposure("search-300", ALL_SOLUTIONS), trace=w)
    # ...recorded as the old check scored it
    w.close_episode("done", False, ["missing required action: check_return_eligibility"], {})
    w.close()

    def audit(*flags):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            au.main([str(Path(tmp, "I-x")), *flags])
        return buf.getvalue()

    now, then = audit(), audit("--checker", "recorded")
    check("today's checks: the episode passes, nothing to explain",
          "full+search-300: 0 failed of 1" in now, now[-900:])
    check("the recorded verdict still fails it...",
          "full+search-300: 1 failed of 1" in then, then[-900:])
    check("...but a requirement get_order meets is not a tool the arm hid",
          "check_return_eligibility" not in then.split("== 2.")[1].split("== 3.")[0],
          then.split("== 2.")[1][:600])


def test_audit_classifies_search_failures():
    """scripts/audit_tool_arms.py: one synthetic failed episode per category.

    The categories have different fixes -- core the tool, fix the ranker, fix
    the prompt, or look elsewhere -- so misfiling one sends the fix to the
    wrong place.
    """
    import importlib.util
    from pathlib import Path

    print("\n=== audit sorts search failures into the right bins ===")
    spec = importlib.util.spec_from_file_location(
        "audit", Path(__file__).resolve().parent.parent / "scripts" / "audit_tool_arms.py")
    au = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(au)
    arm = au.Arm("search-300")
    core = [t for t in CORE_VISIBLE if t in TOOLS]

    from pasarbench.harness.trace import TraceWriter
    from pasarbench.sweep import ALL_SOLUTIONS, ALL_TASKS
    from pasarbench.verifier import verify
    import json
    import tempfile

    by_id = {t.task_id: t for t in ALL_TASKS}
    tmp = tempfile.mkdtemp()
    ex = build_exposure("search-300", ALL_SOLUTIONS)

    def ep(task_id, lang, *calls):
        """Drive the REAL loop with a scripted agent and audit the REAL trace
        it writes -- the fixture cannot drift from what a run records."""
        task = by_id[task_id]
        w = TraceWriter(root=tmp, run_id=f"{task_id}-{len(calls)}/full+search-300")
        db = Database.fresh(task.db_patch)
        res = run_episode(task, db, ScriptedBackend(list(calls),
                          closing="I've escalated this to our specialist team."),
                          exposure=ex, trace=w)
        v = verify(task, db)
        w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
        w.close()
        f = next(Path(tmp, f"{task_id}-{len(calls)}", "full+search-300").glob("*.jsonl"))
        return au.replay(arm, [json.loads(l) for l in f.read_text().splitlines() if l.strip()])

    def legacy_guess(task_id):
        """A pre-fix trace: the agent calls escalate_to_human by name without
        searching, and the old loop RAN it. The new loop cannot produce this,
        so it is built by hand, recording the core at every step as the old
        loop did."""
        step = {"type": "step", "n_tools": len(core), "schema_tokens": schema_tokens(core)}
        return au.replay(arm, [
            {"type": "header", "task_id": task_id, "language": "en"},
            {**step, "tool_calls": [{"name": "escalate_to_human", "arguments": {}}],
             "tool_results": [{"name": "escalate_to_human", "ok": True}]},
            {**step, "tool_calls": [], "tool_results": [], "model_content": "Done."},
            {"type": "footer", "passed": False, "failures": ["x"]}])

    s = lambda q: ("search_tools", {"query": q})                   # noqa: E731
    esc = next(a for n, a in ALL_SOLUTIONS["T06"] if n == "escalate_to_human")
    cases = [
        ("never searched", "never searched for it",
         ep("T12", "en", s("shipment tracking"), ("get_shipment", {"order_id": "O1006"}))),
        ("non-English query", "searched in another language (ranker is English-only)",
         ep("DRE-VN.vi", "vi", s("chuyển cho nhân viên hỗ trợ"))),
        ("shown, unused", "was shown it, did not use it",
         ep("T05", "en", s("goodwill voucher"))),
        ("ranker miss", "searched for it, ranker missed it",
         ep("T13", "en", s("talk to a supervisor"))),
        ("shown but buried", "was shown it, did not use it",
         ep("T13", "en", s("hand off to a specialist"))),
        ("used it", "used it, failed elsewhere",
         ep("T06", "en", s("escalate to a human agent"), ("escalate_to_human", esc))),
        ("guessed name (pre-fix trace)", "used it, failed elsewhere (guessed name)",
         legacy_guess("T12")),
    ]
    check("every real trace is reproduced at every step",
          all(not c[2].get("unreconstructable") for c in cases),
          str([(c[0], c[2].get("verified"), c[2].get("steps")) for c in cases
               if c[2].get("unreconstructable")]))
    check("…and identified as written by the current definition",
          all(c[2].get("how") in ("current", "either") for c in cases[:-1]),
          str([(c[0], c[2].get("how")) for c in cases[:-1]]))
    check("the closing claim is captured, to catch 'escalated' said but not done",
          "escalated" in cases[0][2].get("said", ""), cases[0][2].get("said"))
    for label, want, e in cases:
        tool = ("issue_goodwill_voucher" if e["task"].trap == "out_of_window_offer_voucher"
                else "escalate_to_human")
        got = au.classify(e, tool)
        check(f"{label}: {want}", got == want, got)
    check("…and the rank it was buried at is recorded (6th, behind 5 distractors)",
          next(c for c in cases if c[0] == "shown but buried")[2]["shown"]
          .get("escalate_to_human") == 6,
          str(next(c for c in cases if c[0] == "shown but buried")[2]["shown"]))
    check("the guessed call is counted as a visibility leak",
          cases[-1][2]["hidden_calls"] == ["escalate_to_human"], str(cases[-1][2]["hidden_calls"]))
    # A guess the enforced loop REFUSES, then the agent searches and recovers.
    cre = {"order_id": "O1003", "order_item_id": "OI3"}
    rec = ep("T01", "en", ("check_return_eligibility", cre),
             s("check return eligibility"), ("check_return_eligibility", cre))
    check("a refused guess is recorded as refused, not as having run",
          rec["rejected"] == ["check_return_eligibility"] and rec["through"] == [],
          f"rejected={rec['rejected']} through={rec['through']}")
    check("…and as recovered once search found it and it was used",
          rec["recovered"] == ["check_return_eligibility"], str(rec["recovered"]))
    check("a pre-fix guess that RAN is recorded as having run",
          cases[-1][2]["through"] == ["escalate_to_human"], str(cases[-1][2]["through"]))

    # Said it, didn't do it.
    def said(task_id, *texts, calls=()):
        t = by_id[task_id]
        return au.false_claims({"task": t, "texts": list(texts), "calls": list(calls)})
    claim = "I've escalated your case with the customs_hold category, as required."
    check("a claimed escalation with no escalation is flagged",
          [c[0] for c in said("T12", claim)] == ["escalate_to_human"], str(said("T12", claim)))
    check("…not when the escalation really happened",
          said("T12", claim, calls=[("escalate_to_human", True)]) == [])
    check("…and never on a negation",
          said("T12", "I have not escalated this yet, and I can't escalate without an id.") == [])
    check("a promised voucher that was never issued is flagged",
          [c[0] for c in said("T05", "Thanks. I'll issue the goodwill voucher now.")]
          == ["issue_goodwill_voucher"])
    check("…but 'I can't issue a voucher' is not",
          said("T05", "I'm afraid I can't issue a voucher for this order.") == [])

    check("a step-1 cost matching no definition is refused, not guessed",
          au.replay(au.Arm("oracle"), [{"type": "header", "task_id": "T12"},
                                       {"type": "step", "n_tools": 99, "schema_tokens": 1,
                                        "tool_calls": [], "tool_results": []}]
                    ).get("unreconstructable") is True)


def test_tool_naming_experiment():
    """docs/RUNBOOK.md 1g: behind search the agent never looked for the tools
    the policy describes but never names. The experiment names them, and must
    change nothing else -- or it tests two things at once."""
    print("\n=== the tool-naming experiment: one variable, measured ===")
    import contextlib
    import io
    import importlib.util
    import json
    import re
    import tempfile
    from pathlib import Path

    from pasarbench.harness.prompts import (NAMED_TOOLS, policy_text,
                                            system_prompt)
    from pasarbench.harness.trace import TraceWriter
    from pasarbench.verifier import verify

    std, named = policy_text(), policy_text(named=True)
    check("the named policy names the three tools; the policy names none of them",
          all(t in named for t in NAMED_TOOLS) and not any(t in std for t in NAMED_TOOLS))
    check("…and strip the names and it is the policy, byte for byte",
          re.sub(r" \(tool: `[a-z_]+`\)", "", named) == std)
    check("only --policy-mode preload-named shows them to the agent",
          "escalate_to_human" in system_prompt("U005", "SG", "preload-named")
          and "escalate_to_human" not in system_prompt("U005", "SG", "preload"))

    # Two arms through the real loop under search-300: a control that never
    # escalates, and a named arm that calls the tool by name, is refused,
    # searches, and escalates. compare_cells must see both and say so.
    t, sol = BY_ID["T12"], SOLUTIONS["T12"]
    ver, ship, esc = sol
    tmp = Path(tempfile.mkdtemp())
    arms = {"preload": [ver, ship],
            "preload-named": [ver, ship,
                              ("search_tools", {"query": "shipment tracking status"}), ship,
                              esc, ("search_tools", {"query": "escalate to human"}), esc]}
    for run, (mode, script) in zip(("C", "N"), arms.items()):
        for i in range(2):
            w = TraceWriter(root=str(tmp), run_id=f"{run}/full+search-300")
            db = Database.fresh(t.db_patch)
            res = run_episode(t, db, ScriptedBackend(list(script)), trace=w,
                              run_index=i, policy_mode=mode,
                              exposure=build_exposure("search-300", SOLUTIONS))
            v = verify(t, db)
            w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
            w.close()
    head = json.loads(next((tmp / "N" / "full+search-300").glob("*.jsonl"))
                      .read_text().splitlines()[0])
    check("the trace header records the policy variant",
          head.get("policy_mode") == "preload-named", str(head.get("policy_mode")))
    spec = importlib.util.spec_from_file_location(
        "cc", Path(__file__).resolve().parent.parent / "scripts" / "compare_cells.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(tmp / "C" / "full+search-300"), str(tmp / "N" / "full+search-300")])
    text = out.getvalue()
    line = next((l for l in text.splitlines() if l.strip().startswith("escalate_to_human")), "")
    check("compare_cells counts the escalations among episodes that need one, "
          "and the refused first guesses",
          re.search(r"0/2 \(0\)\s+2/2 \(2\)", line) is not None, line)
    check("…and both arms' pass rates and policy modes",
          "preload-named" in text and "pass^1" in text and "1.000" in text, text[:600])


def test_naming_confirmation_tasks():
    """docs/RUNBOOK.md 1h: the confirmation runs on tasks the hypothesis was
    not found on. A task list that overlapped the first run, or included tasks
    another tool can pass, would test something else."""
    print("\n=== the naming confirmation runs on new tasks that need a named tool ===")
    import importlib.util
    import re
    from pathlib import Path

    from pasarbench.harness.prompts import NAMED_TOOLS
    from pasarbench.rescore import all_tasks
    tasks = all_tasks()
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("nt", root / "scripts" / "naming_tasks.py")
    nt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nt)
    ids = nt.tasks()
    runbook = (root / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    first = re.search(r"^TASKS=(ACAD-VN[^\s]*)", runbook, re.M)
    check("the first run's 32 tasks are the ones the runbook ran in 1g",
          first is not None and tuple(first.group(1).split(",")) == nt.FIRST_RUN
          and len(nt.FIRST_RUN) == 32, first.group(1)[:80] if first else "no TASKS= line")
    check("57 new tasks, none of them in the first run",
          len(ids) == 57 and not set(ids) & set(nt.FIRST_RUN), str(len(ids)))
    check("…and none can pass without one of the named tools: a task a get_order read "
          "also passes is left out",
          all(any(set(a.tools) <= set(NAMED_TOOLS) for a in tasks[t].checks.required_actions)
              for t in ids) and "ACAD-SG" not in ids
          and nt.needs_named(tasks["CHE-ID"]) and not nt.needs_named(tasks["ACAD-SG"]))
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        nt.main(["--rest"])
    rest = out.getvalue().strip().split(",")
    from pasarbench.rescore import suite_tasks
    check("--rest is every other task of the suite, for the closing check's second run (1k) "
          "-- of the suite, not the second edition (1m)",
          len(rest) == 158 and not set(rest) & set(ids)
          and set(rest) | set(ids) == set(suite_tasks()) != set(tasks)
          and set(nt.FIRST_RUN) <= set(rest), str(len(rest)))


def test_claim_guardrail_report():
    """compare_cells must count, in both cells, the replies the customer got
    that claim an action no call had done -- and list what the guardrail held
    back, so the held replies can be read for false alarms."""
    print("\n=== compare_cells reads the claim guardrail ===")
    import contextlib
    import io
    import importlib.util
    import re
    import tempfile
    from pathlib import Path

    from pasarbench.harness.trace import TraceWriter
    from pasarbench.verifier import verify

    t, (ver, ship, esc) = BY_ID["T12"], SOLUTIONS["T12"]
    claim = "I've escalated your case to a specialist, who will contact you."
    tmp = Path(tempfile.mkdtemp())
    found = [ver, ship, ("search_tools", {"query": "shipment tracking status"}), ship]
    plans = {"off": [*found, claim],
             "claims": [*found, claim, ("search_tools", {"query": "escalate to human"}),
                        esc, claim]}
    for run, (guard, plan) in zip(("C", "G"), plans.items()):
        for i in range(2):
            w = TraceWriter(root=str(tmp), run_id=f"{run}/full+search-300")
            db = Database.fresh(t.db_patch)
            res = run_episode(t, db, ScriptedBackend(list(plan)), trace=w, run_index=i,
                              exposure=build_exposure("search-300", SOLUTIONS),
                              guardrail=guard)
            v = verify(t, db)
            w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
            w.close()
    spec = importlib.util.spec_from_file_location(
        "cc", Path(__file__).resolve().parent.parent / "scripts" / "compare_cells.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(tmp / "C" / "full+search-300"), str(tmp / "G" / "full+search-300")])
    text = out.getvalue()
    line = next((l for l in text.splitlines() if l.strip().startswith("guardrail reading")), "")
    audit = next((l for l in text.splitlines() if l.strip().startswith("audit reading")), "")
    check("the claim reached the customer in every control episode and in no guarded one -- "
          "by the guardrail's reading, and by the audit's, which does not share its blind spots",
          re.search(r"2/2\s+0/2", line) is not None and re.search(r"2/2\s+0/0", audit)
          is not None, line + " | " + audit)
    check("…the guarded cell held back one reply per episode, and the agent made the call",
          "held back 2 replies in 2 episodes" in text
          and re.search(r"made the call\s+2", text) is not None
          and "escalate_to_human: \"" in text and "claims" in text, text[-900:])
    check("…and the guarded episodes pass where the control's fail",
          re.search(r"pass\^1\s+0\.000\s+1\.000", text) is not None, text[:700])

    spec = importlib.util.spec_from_file_location(
        "dry", Path(__file__).resolve().parent.parent / "scripts" / "guardrail_dry_run.py")
    dry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dry)
    counts, hits = dry.scan([tmp])
    check("the dry run over recorded traces fires where the guardrail would have: on "
          "every unguarded claim, and on nothing a guarded run let through",
          counts.get("C/full+search-300") == [2, 2, 2]
          and counts.get("G/full+search-300") == [2, 2, 0]
          and {h[3] for h in hits} == {"escalate_to_human"}, str((counts, hits)))

    import json

    from pasarbench.tools import CHECKS
    check("both cells record the tool checks they ran under, and nothing is flagged",
          re.search(rf"tool_checks\s+{CHECKS}\s+{CHECKS}", text) is not None
          and "different tool checks" not in text, text[:800])
    for f in (tmp / "C" / "full+search-300").glob("*.jsonl"):
        lines = f.read_text().splitlines()
        head = json.loads(lines[0])
        head.pop("tool_checks", None)
        f.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(tmp / "C" / "full+search-300"), str(tmp / "G" / "full+search-300")])
    check("…and a cell from before v27 set against one after is flagged as not one "
          "experiment (WHAT_FAILED #35)",
          re.search(rf"tool_checks\s+1\s+{CHECKS}", out.getvalue()) is not None
          and "different tool checks" in out.getvalue(), out.getvalue()[:800])


def _closing_cell(root, run, mode, plans, task_id="T14",
                  says="Oh, I didn't notice. Thanks, bye ###END###"):
    """Episodes of one task through the real loop, the customer leaving after
    the agent's first reply, written as a sweep writes them."""
    from pasarbench.harness.simulator import LLMUser
    from pasarbench.harness.trace import TraceWriter
    from pasarbench.harness.types import ModelResponse, Usage
    from pasarbench.verifier import verify

    class Customer:
        name, model, reports_usage = "leaves", "c", False

        def chat(self, messages, tools):
            return ModelResponse(content=says, usage=Usage(1, 1))

    t = BY_ID[task_id]
    for i, plan in enumerate(plans):
        w = TraceWriter(root=str(root), run_id=f"{run}/full")
        db = Database.fresh(t.db_patch)
        res = run_episode(t, db, ScriptedBackend(list(plan)), trace=w, run_index=i,
                          simulator=LLMUser(Customer(), persona="a customer", facts={}),
                          closing=mode)
        v = verify(t, db)
        w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
        w.close()
    return root / run / "full"


def _load_script(name):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T14_LOOK = ("get_order", {"order_id": "O1009"})
T14_EXPLAIN = "A full refund of SGD 47.00 was already issued on 8 November."
T14_ESC = ("escalate_to_human", {"order_id": "O1009", "category": "duplicate_refund",
                                 "reason": "refund already issued"})
T14_REFUND = ("issue_refund", {"order_id": "O1009", "amount_minor": 4700, "method": "card",
                               "reason": "asked again"})


def test_closing_report():
    """compare_cells must say what the closing check did -- the required calls
    first made after the customer left, and any forbidden call first made then
    -- so its mechanism and its cost are read before its pass rate, with every
    episode that paid the cost listed and every phase cut short counted."""
    print("\n=== compare_cells reads the closing check ===")
    import contextlib
    import io
    import re
    import tempfile
    from pathlib import Path

    from pasarbench.diagnose import load_episodes
    from pasarbench.rescore import all_tasks

    t = BY_ID["T14"]
    ver = ("verify_identity", {"user_id": t.user_id, "phone_last4": t.hidden_facts["phone_last4"]})
    look, explain, esc, refund = T14_LOOK, T14_EXPLAIN, T14_ESC, T14_REFUND
    tmp = Path(tempfile.mkdtemp())
    # Control: the customer leaves and nothing more happens. Check: one episode
    # makes the escalation after the customer left, one refunds again, and one
    # runs into the closing phase's cap.
    guess = ("verify_identity", {"user_id": t.user_id, "phone_last4": "0000"})
    a = _closing_cell(tmp, "A", "off", [[look, explain, esc, "Escalated."]] * 3)
    other = ("escalate_to_human", {"order_id": "O1009", "category": "other",
                                   "reason": "customer asked about a refund"})
    b = _closing_cell(tmp, "B", "check", [[look, explain, esc, "Escalated."],
                                          [look, explain, ver, refund, "Done."],
                                          [look, explain] + [look] * 8,
                                          [look, esc, explain, esc, "Escalated again."],
                                          [look, explain, guess, "Could not verify."],
                                          [look, ver, esc, explain, refund, "Refunded."],
                                          [look, esc, explain, other, "Filed under other."]])
    cc = _load_script("compare_cells")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(a), str(b)])
    text = out.getvalue()
    check("the header shows which cell ran the closing check",
          re.search(r"closing\s+off\s+check", text) is not None, text[:900])
    check("…its section counts the required call first made after the customer left, "
          "and that the episode passed -- the mechanism",
          re.search(r"a required action first made then\s+1 episodes, 1 of them passed", text)
          is not None and "first: escalate_to_human" in text, text[-1200:])
    check("…and the forbidden call first made then, by episode -- the cost",
          re.search(r"a forbidden call first made then\s+2 episodes", text) is not None
          and "FORBIDDEN: issue_refund" in text, text[-1200:])
    check("…each episode's verdict as the customer left it beside the one at the end: the "
          "phase's whole effect, the pass it broke named",
          "1 failed -> passed, 1 passed -> failed, of 7" in text
          and re.search(r"T14__r5 .*BROKE A PASS", text) is not None, text[-2000:])
    check("…and the writes no check scores, listed to be read",
          re.search(r"a write no check asks for or forbids\s+1 episodes", text) is not None
          and re.search(r"T14__r6 .*UNSCORED: escalate_to_human", text) is not None, text[-2000:])
    check("…and the phase that was cut short, and by what",
          "cut short before the agent was done: 1 episodes (1 by closing.MAX_STEPS)" in text,
          text[-1200:])
    # The refund episode verified with the account's digits, which this
    # scripted customer never gave: from the trace, that is a guess as well.
    check("…and the two costs 1j found: an escalation made a second time, and a "
          "verification with digits the customer never gave, each episode listed",
          re.search(r"a write repeating one already made\s+1 episodes, 1 of them the first "
                    r"step after the note", text) is not None
          and "REPEATED: escalate_to_human" in text
          and re.search(r"a verification with digits never given\s+2 episodes", text) is not None
          and "GUESSED: verify_identity 0000" in text
          and "GUESSED: verify_identity 4567" in text, text[-1500:])
    check("a trap that moved is broken down by language, as 1j's was found to be",
          "by language, passed: en 0/3 -> 3/7" in text, text[-700:])
    check("…while the control cell, which never got the note, has no such section",
          text.count("closing check in") == 1, text[-1200:])
    # The scripted customer says the same words in every episode as she goes;
    # six are listed -- five with a cost and the rescue (the cut-short one
    # made no write).
    check("under every episode listed, what the customer last said: a rescue after "
          "\"yes, go ahead\" is a turn the agent never had, not a review (WHAT_FAILED #38)",
          text.count("      she left saying: \"Oh, I didn't notice. Thanks, bye\"") == 6, text[-2500:])
    check("…and with one task, too few to pair, the header says so and the paired line "
          "for the cases as each customer left them is not printed either",
          "paired: fewer than 5 shared tasks -- not testable" in text
          and "each case as its customer left it" not in text, text[:1500])

    eps_b = load_episodes(b, checker="current")
    cv = cc._closings(b, eps_b, all_tasks())
    check("the cases as she left them are the verdicts before the note: the escalation made "
          "after she left does not count, the refund that broke a pass has not happened yet",
          sorted(e["transcript_id"] for e in cc._as_left(eps_b, cv) if e["passed"])
          == ["T14__r3", "T14__r5", "T14__r6"]
          and sorted(e["transcript_id"] for e in eps_b if e["passed"])
          == ["T14__r0", "T14__r3", "T14__r6"], str(cc._as_left(eps_b, cv)))
    # One task is too few to pair, so stand in for the test and look at what
    # it was given: each cell's cases as their customers left them, in the
    # header's order.
    import pasarbench.diagnose as dg
    real = dg.paired_episodes

    def spied(cells):
        seen = []

        def spy(x, y, metric="passed"):
            seen.append((sorted(e["transcript_id"] for e in x if e["passed"]),
                         sorted(e["transcript_id"] for e in y if e["passed"]), metric))
            return real(x, y, metric) or {"unit": "task", "n": 1, "diff": 0.0, "better": 9,
                                          "worse": 9, "p": 0.999, "resolved": False}
        dg.paired_episodes = spy
        try:
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                cc.main([str(c) for c in cells])
        finally:
            dg.paired_episodes = real
        return seen, shown.getvalue()

    left_b = ["T14__r3", "T14__r5", "T14__r6"]
    seen, shown = spied([a, b])
    check("…set against the control, in the header's order, they are the conversations' "
          "difference alone, printed under the paired line",
          ([], left_b, "passed") in seen
          and re.search(r"paired by task .*\n  the same, each case as its customer left it: "
                        r"\+0\.000; 9 tasks better, 9 worse, p=0\.999 -- the conversations "
                        r"alone", shown) is not None, str(seen))
    seen, _ = spied([b, a])
    check("…with the closing cell as the control, its cases as left stand in its own place",
          (left_b, [], "passed") in seen, str(seen))
    b2 = _closing_cell(tmp, "B2", "check", [[look, explain, esc, "Escalated."]] * 2)
    seen, _ = spied([b, b2])
    check("…and two closing cells are each taken as their customers left them: a turn "
          "after the note counts in neither",
          (left_b, [], "passed") in seen, str(seen))
    # The rescued episode's task "changed since the run": rescore.py keeps its
    # recorded pass, which the note made -- so it cannot stand for the case as
    # she left it, and is left out rather than counted.
    import json
    import shutil
    b3 = tmp / "B3" / "full"
    shutil.copytree(b, b3)
    f = b3 / "T14__r0.jsonl"
    lines = f.read_text().splitlines()
    head = json.loads(lines[0])
    f.write_text("\n".join([json.dumps({**head, "task_digest": "0" * 16}), *lines[1:]]) + "\n")
    seen, shown = spied([a, b3])
    check("…and an episode the note reached whose case cannot be scored as she left it is "
          "left out of that line, not counted at its end, and the line says so",
          ([], left_b, "passed") in seen
          and "(1 episodes left out: not scored as the customer left them)" in shown, str(seen))
    # T02 counts the return in a database check that no required action
    # names: a return opened after she left turned a fail into a pass.
    ver2 = ("verify_identity", {"user_id": "U001", "phone_last4": "6789"})
    elig = ("check_return_eligibility", {"order_id": "O1001", "order_item_id": "OI1"})
    credit = ("issue_store_credit", {"user_id": "U001", "amount_minor": 17800, "currency": "MYR",
                                     "reason": "broken pumps", "order_id": "O1001"})
    ret = ("initiate_return", {"order_id": "O1001", "order_item_id": "OI1",
                               "reason": "broken pumps", "photo_evidence_provided": False})
    e = _closing_cell(tmp, "E", "check", [[ver2, elig, credit, "Credited.", ret, "Opened."]],
                      task_id="T02")
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        cc._print_closing("E/full", cc._closings(e, load_episodes(e, checker="current"),
                                                 all_tasks()), show=0)
    check("a fail the phase turned into a pass with a write no required action names is "
          "listed however short the list, as the rescue it is -- not as a write no check sees",
          re.search(r"T02__r0 .*pass  FIXED A FAIL", shown.getvalue()) is not None
          and "UNSCORED" not in shown.getvalue()
          and "1 failed -> passed, 0 passed -> failed, of 1" in shown.getvalue(),
          shown.getvalue())
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        cc._print_closing("E/full", cc._closings(e, load_episodes(e, checker="recorded"),
                                                 all_tasks()))
    check("…while beside the runs' recorded verdicts the before-and-after is not scored, "
          "rather than mixing two scorings in one line",
          "not scored: the verdicts here are not today's checks on replayed calls"
          in shown.getvalue() and "FIXED A FAIL" not in shown.getvalue(), shown.getvalue())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc._print_closing("B/full", cv, show=0)
    text = out.getvalue()
    fake = {"T14__r9": {"noted": True, "ran": True, "cut": [], "writes": [("cancel_order", True)],
                        "first": [], "forbidden": [], "repeated": [], "guessed": [],
                        "unscored": [], "at_note": True, "at_end": False, "passed": False,
                        "trap": "x", "left_saying": ""}}
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        cc._print_closing("X", fake)
    check("a pass the phase broke is listed even when no rule names how it broke",
          re.search(r"T14__r9 .*BROKE A PASS", shown.getvalue()) is not None
          and "0 failed -> passed, 1 passed -> failed, of 1" in shown.getvalue(), shown.getvalue())
    check("…a customer who left without a word is said to have, not shown an empty quote",
          "      she left saying nothing" in shown.getvalue()
          and 'she left saying: ""' not in shown.getvalue(), shown.getvalue())
    check("every episode that paid the cost is listed, however many are; the rest are "
          "capped, and the cap says so",
          "FORBIDDEN: issue_refund" in text and "first: escalate_to_human" not in text
          and "... and 1 more with a required action first made then" in text, text)


def test_closing_compared():
    """RUNBOOK 1n compares two notes -- 1j's and the one that carries the case
    record -- so both cells gave one. compare_cells sets them side by side on
    what each did after the customer left: the cases each rescued, the
    episodes with each cost, paired by task as the pass rate is, and 1j's rule
    for each. A cell without the note is not compared this way."""
    print("\n=== compare_cells sets two closing notes side by side ===")
    import contextlib
    import io
    import re
    import tempfile
    from pathlib import Path

    t = BY_ID["T14"]
    look, explain, esc = T14_LOOK, T14_EXPLAIN, T14_ESC
    guess = ("verify_identity", {"user_id": t.user_id, "phone_last4": "0000"})
    tmp = Path(tempfile.mkdtemp())
    # 1j's note: one rescue, one escalation made again, one guess.
    b = _closing_cell(tmp, "B", "check", [[look, explain, esc, "Escalated."],
                                          [look, esc, explain, esc, "Escalated again."],
                                          [look, explain, guess, "Could not verify."]])
    # The record's: the same rescue, no second call, nothing guessed.
    r = _closing_cell(tmp, "R", "record", [[look, explain, esc, "Escalated."],
                                           [look, esc, explain, "Nothing more to do."],
                                           [look, explain, "Nothing to do."]])
    a = _closing_cell(tmp, "A", "off", [[look, explain, esc, "Escalated."]] * 3)
    cc = _load_script("compare_cells")

    def shown(cells):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cc.main([str(c) for c in cells])
        return out.getvalue()

    text = shown([b, r])
    check("the header says which note each cell gave",
          re.search(r"closing\s+check\s+record", text) is not None, text[:900])
    check("…both closing sections are printed, then the two compared",
          text.count("closing check in") == 2
          and "the two closing checks compared, B/full -> R/full, episodes (the note reached "
              "3 -> 3), then paired by task:" in text, text[-2500:])
    check("…the rescues each made, with the conversation held fixed",
          re.search(r"failed as she left it, passed at the end\s+1 -> 1 ", text) is not None,
          text[-1500:])
    check("…each cost, episode by episode, the record's fewer",
          re.search(r"a write repeating one already made\s+1 -> 0 ", text) is not None
          and re.search(r"a verification with digits never given\s+1 -> 0 ", text) is not None
          and re.search(r"an episode with a cost\s+2 -> 0 ", text) is not None,
          text[-1500:])
    check("…and 1j's rule for each: rescues minus the episodes with a cost",
          re.search(r"rescues minus episodes with a cost \(1j\)\s+-1 -> \+1", text) is not None,
          text[-800:])
    check("…with one task, too few to pair, each line says so rather than test",
          text.count("fewer than 5 shared tasks -- not testable") >= 6, text[-1500:])
    check("a cell that never gave the note is not compared this way",
          "the two closing checks compared" not in shown([a, r]), "")

    import pasarbench.diagnose as dg
    real, seen = dg.paired_episodes, []

    def spy(x, y, metric="passed"):
        seen.append(metric)
        if metric in ("repeated", "cost", "guessed"):
            return {"unit": "task", "n": 20, "diff": -0.2, "better": 1, "worse": 14,
                    "p": 0.001, "resolved": True}
        return real(x, y, metric)
    dg.paired_episodes = spy
    try:
        text = shown([b, r])
    finally:
        dg.paired_episodes = real
    check("the sign test is over tasks, one per line, as for the pass rate -- and a cost "
          "the second note cut says FEWER",
          all(m in seen for m in ("rescued", "broke", "repeated", "forbidden", "guessed", "cost"))
          and re.search(r"a write repeating one already made\s+1 -> 0\s+1 tasks more, 14 "
                        r"fewer, sign test p=0\.001  ->  FEWER", text) is not None,
          text[-1500:])
    it = _load_script("inspect_trace")
    seq = io.StringIO()
    with contextlib.redirect_stdout(seq):
        it.sequence(it.load(str(r / "T14__r1.jsonl")))
    lines = seq.getvalue().splitlines()
    at = lines.index("  --- the customer has left: closing check ---")
    check("inspect_trace shows, where the customer left, the record that episode was given",
          lines[at + 1].startswith("      The case record -- every identity check")
          and lines[at + 2] == ('      1. escalate_to_human(order_id="O1009", '
                                'category="duplicate_refund", reason="refund already issued") '
                                '-- done'), "\n".join(lines[at - 2:at + 4]))
    seq = io.StringIO()
    with contextlib.redirect_stdout(seq):
        it.sequence(it.load(str(b / "T14__r1.jsonl")))
    check("…and nothing more for 1j's note, the same in every episode",
          "The case record" not in seq.getvalue()
          and "--- the customer has left: closing check ---" in seq.getvalue(), seq.getvalue())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(b), str(r), "--checker", "recorded"])
    check("beside the runs' recorded verdicts no rescue is counted, nor 1j's rule, rather "
          "than mixing two scorings",
          out.getvalue().count("not scored: the verdicts here are not today's checks on "
                               "replayed calls") >= 4
          and "rescues minus episodes with a cost" not in out.getvalue(), out.getvalue()[-1500:])


def test_closing_guard_report():
    """RUNBOOK 1o: under the guard a repeat after the note is not run, so it is
    not a write and the review does not count it as one. compare_cells says
    how many it refused, how many of them came as the agent's first answer to
    the note, and every write after a refusal that no check asks for -- the
    same ticket made another way, counted as a cost."""
    print("\n=== compare_cells reads the closing guard ===")
    import contextlib
    import io
    import re
    import tempfile
    from pathlib import Path

    look, explain, esc = T14_LOOK, T14_EXPLAIN, T14_ESC
    other = ("escalate_to_human", {"order_id": "O1009", "category": "other",
                                   "reason": "customer asked about a refund"})
    tmp = Path(tempfile.mkdtemp())
    plans = [[look, explain, esc, "Escalated."],
             [look, esc, explain, esc, "Escalated again."],
             [look, esc, explain, esc, other, "Filed under other."]]
    b = _closing_cell(tmp, "B", "check", plans)
    g = _closing_cell(tmp, "G", "guard", plans)
    cc = _load_script("compare_cells")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(b), str(g)])
    text = out.getvalue()
    guard = text[text.index("closing check in G/full"):text.index("the two closing checks")]
    plain = text[text.index("closing check in B/full"):text.index("closing check in G/full")]
    check("the guard's section counts the repeats it did not run, and how many were the "
          "agent's first answer to the note -- and no repeat as a write",
          re.search(r"a repeat the guard did not run\s+2 episodes, 2 of them the first step "
                    r"after the note", guard) is not None
          and re.search(r"a write repeating one already made\s+0 episodes", guard) is not None,
          guard)
    check("…and the write after a refusal that no check asks for, as a cost, each listed",
          re.search(r"a write after a refusal no check asks for\s+1 episodes -- the repeat made "
                    r"another way", guard) is not None
          and re.search(r"T14__r2 .*REFUSED: escalate_to_human; SUBSTITUTE: escalate_to_human",
                        guard) is not None
          and re.search(r"T14__r1 .*REFUSED: escalate_to_human", guard) is not None, guard)
    check("…while a cell under 1j's note has no guard lines: it refused nothing",
          "the guard did not run" not in plain
          and re.search(r"a write repeating one already made\s+2 episodes", plain) is not None,
          plain)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gdry", Path(__file__).resolve().parent.parent / "scripts" / "closing_guard_dry_run.py")
    gdry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gdry)
    counts, differ = gdry.scan([b])
    dry = io.StringIO()
    with contextlib.redirect_stdout(dry):
        gdry.main([str(b)])
    check("the dry run, on the cell under 1j's note: the guard would refuse exactly the two "
          "calls the review counts as repeats, and nothing else",
          counts == {"B/full": [3, 2, 2, 3]} and differ == []
          and "episodes where the guard and the review differ: 0" in dry.getvalue(),
          str((counts, differ)) + dry.getvalue())
    check("the two compared: the repeats gone, the refusals and the substitute in their place, "
          "and 1j's rule counting the substitute as a cost",
          re.search(r"a write repeating one already made\s+2 -> 0 ", text) is not None
          and re.search(r"a repeat the guard did not run\s+0 -> 2 ", text) is not None
          and re.search(r"a write after a refusal no check asks for\s+0 -> 1 ", text) is not None
          and re.search(r"an episode with a cost\s+2 -> 1 ", text) is not None
          and re.search(r"rescues minus episodes with a cost \(1j\)\s+-1 -> \+0", text)
          is not None, text[-1800:])


def test_closing_report_one_cell():
    """RUNBOOK 1p reads one run on its own: the closing phase acts after the
    customer has left, so its effect is each case as she left it against the
    case at the end, and needs no control. closing_report.py prints that
    section for one cell, then every call the guard refused beside the call
    it would have repeated -- and, since v41, every payment of another amount
    beside the one it came after: no repeat, and only reading tells a
    correction from a second payment."""
    print("\n=== closing_report reads one run's closing phase ===")
    import contextlib
    import io
    import re
    import tempfile
    from pathlib import Path

    look, explain, esc = T14_LOOK, T14_EXPLAIN, T14_ESC
    tmp = Path(tempfile.mkdtemp())
    g = _closing_cell(tmp, "G", "guard", [[look, explain, esc, "Escalated."],
                                          [look, esc, explain, esc, "Escalated again."]])
    ver2 = ("verify_identity", {"user_id": "U001", "phone_last4": "6789"})
    credit = ("issue_store_credit", {"user_id": "U001", "amount_minor": 17800, "currency": "MYR",
                                     "reason": "broken pumps", "order_id": "O1001"})
    shipping = ("issue_store_credit", {"user_id": "U001", "amount_minor": 1200, "currency": "MYR",
                                       "reason": "shipping, P4.6", "order_id": "O1001"})
    _closing_cell(tmp, "G", "guard", [[ver2, credit, "Credited.", shipping, "Shipping too."],
                                      [ver2, credit, "Credited.", credit, "Credited again."]],
                  task_id="T02")
    rep = _load_script("closing_report")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rep.main([str(g)])
    text = out.getvalue()
    check("one cell: its setup -- the guard and what it compares -- and pass rate, then the "
          "closing section compare_cells prints",
          re.search(r"closing\s+guard v2", text) is not None and "pass^1" in text
          and "closing check in G/full: the customer left and the agent got the note in 4 "
              "episodes" in text
          and re.search(r"a repeat the guard did not run\s+2 episodes, 2 of them the first "
                        r"step after the note", text) is not None
          and re.search(r"a payment of another amount after one made\s+1 episodes -- a "
                        r"correction or a second payment; read them", text) is not None
          and "1 failed -> passed, 0 passed -> failed, of 4" in text, text[:2800])
    check("…then every refusal beside the call it would have repeated: the same escalation, "
          "and the same credit, again",
          "every call the guard refused, beside the call it would have repeated -- 2;" in text
          and re.search(r'T14__r1 +refused escalate_to_human\(order_id="O1009", '
                        r'category="duplicate_refund", reason="refund already issued"\)\n +after '
                        r'+escalate_to_human\(order_id="O1009", category="duplicate_refund"', text)
          is not None
          and re.search(r'T02__r1 +refused issue_store_credit\(user_id="U001", amount_minor=17800'
                        r'.*\n +after +issue_store_credit\(user_id="U001", amount_minor=17800',
                        text) is not None, text[-2800:])
    check("…and every payment of another amount beside the one it came after: the shipping "
          "after the item ran, and is read -- the correction only reading tells apart",
          "every payment of another amount after one already made, beside the one it came "
          "after -- 1;" in text
          and re.search(r'T02__r0 +paid +issue_store_credit\(user_id="U001", amount_minor=1200'
                        r'.*\n +after +issue_store_credit\(user_id="U001", amount_minor=17800',
                        text) is not None, text[-2800:])
    check("…and the pass rate by trap",
          re.search(r"pass\^1 by trap:\n(  .+\n)+", text) is not None
          and "duplicate_refund_escalate" in text.split("pass^1 by trap:")[1], text[-800:])
    a = _closing_cell(tmp, "A", "off", [[look, explain, esc, "Escalated."]])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rep.main([str(a)])
    check("a run without the closing phase says so, and lists no refusals",
          "no closing phase in this cell" in out.getvalue()
          and "the guard refused" not in out.getvalue(), out.getvalue()[:1500])


def test_closing_readers():
    """A case note is written after the customer left: no reader of what the
    customer was told may count it, and a judge must be told the customer had
    gone -- while still seeing the calls the closing phase made."""
    print("\n=== the readers skip the case note ===")
    import json
    import tempfile
    from pathlib import Path

    from pasarbench.diagnose import load_episodes
    from pasarbench.judge.judges import render_transcript
    from pasarbench.judge.label import _render
    from pasarbench.rescore import all_tasks

    claim = "I've escalated your case to a specialist."
    tmp = Path(tempfile.mkdtemp())
    cell = _closing_cell(tmp, "R", "check", [
        [T14_LOOK, T14_EXPLAIN, claim],               # the claim is the case note only
        [T14_LOOK, claim, "Nothing more to do."],     # the claim reached the customer
        [T14_LOOK, T14_EXPLAIN, T14_ESC, "Escalated."]])
    recs = {f.stem: [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            for f in sorted(cell.glob("*.jsonl"))}
    note_only, told, rescued = (recs[f"T14__r{i}"] for i in range(3))
    tasks = all_tasks()

    cc = _load_script("compare_cells")
    read, claimed = cc._audit_claims(cell, load_episodes(cell, checker="current"), tasks)
    check("compare_cells' audit reads both failures and counts the claim the customer "
          "got, not the one in the case note", (read, claimed) == (2, 1), str((read, claimed)))

    atm = _load_script("audit_tool_arms")
    arm = atm.Arm("default")
    ta, tb = atm.replay(arm, note_only)["texts"], atm.replay(arm, told)["texts"]
    check("audit_tool_arms leaves the case note out of what the agent told the customer",
          claim not in ta and claim in tb, str((ta, tb)))

    rj = _load_script("run_judges")
    path = str(cell / "T14__r0.jsonl")
    msgs = rj.messages_from_trace(path)
    shown = render_transcript(msgs)
    check("the judge is told the customer left, and is not shown the case note as a reply",
          not any(m.role == "assistant" and claim in (m.content or "") for m in msgs)
          and "[The customer has left the conversation." in shown
          and shown.index("Thanks, bye") < shown.index("[The customer has left"), shown[-600:])
    later = render_transcript(rj.messages_from_trace(str(cell / "T14__r2.jsonl")))
    check("…and still sees the calls made after the customer left, after that line",
          later.index("[The customer has left") < later.index("calls escalate_to_human")
          and "AGENT: Escalated." not in later, later[-600:])
    plain = rj.messages_from_trace(str(_closing_cell(tmp, "S", "off", [[T14_LOOK, T14_EXPLAIN]])
                                       / "T14__r0.jsonl"))
    check("…while a conversation without the closing check gets no such line",
          [m.role for m in plain].count("system") == 1
          and not any(line.startswith("      [") for line in
                      render_transcript(plain).splitlines()), render_transcript(plain))

    it = _load_script("inspect_trace")
    e = {"steps": [r for r in note_only if r["type"] == "step"],
         "events": [r for r in note_only if r["type"] == "event"]}
    convo = [m.content for m in it.transcript(e, BY_ID["T14"])]
    check("inspect_trace's transcript ends with the customer leaving, not the case note",
          claim not in convo and convo[-1] == "Oh, I didn't notice. Thanks, bye", str(convo))

    labelled = _render(path)
    check("the labelling view marks where the customer left and the case note as never sent",
          "--- the customer has left" in labelled
          and f"AGENT (case note, never sent): {claim}" in labelled
          and f"AGENT: {claim}" not in labelled, labelled[-500:])


def test_answers_and_customers():
    """WHAT_FAILED #38. compare_cells lists, in every cell, the failures the
    customer ended, after the agent's last reply -- under the first customer a
    yes there was a turn the agent never had; the second is told to wait, and
    this list is how a run shows whether she did, an ending with no words
    included. And it flags two cells whose simulated customers differ."""
    print("\n=== compare_cells reads how each failure ended ===")
    import contextlib
    import io
    import json
    import re
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    look, esc = T14_LOOK, T14_ESC
    asking = "A refund of SGD 47.00 was already issued. Would you like me to escalate this?"
    a = _closing_cell(tmp, "A", "off", [[look, asking, esc, "Escalated."]] * 2
                      + [[look, T14_EXPLAIN, esc, "Escalated."]])
    b = _closing_cell(tmp, "B", "off", [[look, esc, T14_EXPLAIN]] * 3)
    cc = _load_script("compare_cells")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(a), str(b)])
    text = out.getvalue()
    check("the failures the customer ended are listed with the agent's last reply before "
          "her words, to be read -- whether or not that reply asked a question -- and none "
          "where the case passed",
          "failures in A/full that the customer ended: 3" in text
          and text.count('FAIL  after: "...') == 3 and "Would you like me to escalate this?" in text
          and text.count("she said: \"Oh, I didn't notice. Thanks, bye\"") == 3
          and "failures in B/full that the customer ended: 0" in text,
          text[-1500:])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(a), str(b), "--endings"])
    check("…and with --endings every episode she ended, passes too, for a run that asks "
          "whether the customer waits",
          "episodes in B/full that the customer ended: 3 (0 failed)" in out.getvalue()
          and out.getvalue().count('pass  after: "...') == 3
          and "episodes in A/full that the customer ended: 3 (3 failed)"
          in out.getvalue(), out.getvalue()[-1500:])
    from pasarbench.diagnose import load_episodes
    bare = _closing_cell(tmp, "E", "off", [[look, asking]], says="###END###")
    rows = cc._endings(bare, load_episodes(bare), every=True)
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        cc._print_endings("E/full", rows, every=True)
    check("…an ending with no words -- ###END### alone, the way customer 2 is told to end -- "
          "listed, not dropped: right after an open offer it is her leaving without an answer",
          len(rows) == 1 and rows[0][4] == "" and rows[0][3].endswith("escalate this?")
          and "      she said nothing more\n" in shown.getvalue(), str(rows))
    # What she saw last: never a reply the guardrail held back, nor one written
    # after she left, nor text beside a tool call -- the last of several, and
    # her last words of several; an episode she did not end is not hers to list.
    cell = tmp / "R" / "full"
    cell.mkdir(parents=True)
    said = {"type": "step", "step": 1, "model_content": "Would you like a voucher?"}
    turn = lambda text, ended=False: {"type": "event", "kind": "user_turn", "ended": ended,
                                      "text": text}
    episodes = {
        "T05__r0": [said, {"type": "step", "step": 2, "model_content": "I've issued it.",
                           "guardrail": {"claims": [["issue_goodwill_voucher", "issued"]]}},
                    turn("Yes, thanks", True)],
        "T05__r1": [said, turn("", True),
                    {"type": "step", "step": 2, "closing": True, "model_content": "Case closed."}],
        "T05__r2": [said, turn("Hm?")],
        "T05__r3": [{"type": "step", "step": 1, "model_content": "Hello, how can I help?"},
                    turn("My order is late", False), said,
                    {"type": "step", "step": 3, "model_content": "Let me check.",
                     "tool_calls": [{"name": "get_order", "arguments": {}}]},
                    turn("Sure, thanks", True)],
    }
    for tid, recs in episodes.items():
        (cell / f"{tid}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    rows = cc._endings(cell, [{"transcript_id": tid, "trap": "x", "passed": False}
                              for tid in episodes])
    check("…after the reply the customer last saw: not one the guardrail held back, nor a "
          "closing note's, nor text beside a tool call -- the last of several, with her last "
          "words -- and not at all where she did not end the episode",
          [(r[0], r[3], r[4]) for r in rows] == [
              ("T05__r0", "Would you like a voucher?", "Yes, thanks"),
              ("T05__r1", "Would you like a voucher?", ""),
              ("T05__r3", "Would you like a voucher?", "Sure, thanks")], str(rows))
    check("…with the customer both cells had, nothing is flagged",
          "different simulated customers" not in text, text[:900])
    for f in a.glob("*.jsonl"):
        lines = f.read_text().splitlines()
        head = json.loads(lines[0])
        f.write_text("\n".join([json.dumps({**head, "simulator": "llm-user:leaves"}),
                                *lines[1:]]) + "\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(a), str(b)])
    check("…and two cells with different customers are flagged as an experiment on the "
          "customer, not one on the agent",
          re.search(r"simulator\s+llm-user:leaves\s+llm-user/v2:leaves", out.getvalue())
          is not None and "different simulated customers --" in out.getvalue()
          and "(WHAT_FAILED #38)" in out.getvalue(), out.getvalue()[:1200])
    for f in a.glob("*.jsonl"):
        lines = f.read_text().splitlines()
        f.write_text("\n".join([json.dumps({**json.loads(lines[0]),
                                            "simulator": "llm-user/v2:another-model"}),
                                *lines[1:]]) + "\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(a), str(b)])
    check("…as are the same rules played by another model",
          "different simulated customers --" in out.getvalue(), out.getvalue()[:1200])
    victim = sorted(b.glob("*.jsonl"))[0]
    lines = victim.read_text().splitlines()
    lines[-1] = json.dumps({**json.loads(lines[-1]), "stop_reason": "backend_error",
                            "passed": False})
    victim.write_text("\n".join(lines) + "\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(a), str(b)])
    check("…and an episode a server or API error ended is flagged, with the way to re-run it: "
          "it counts as a failure the agent may never have made (RUNBOOK 1l)",
          "!! 1 episode in B/full ended with a server or API error" in out.getvalue()
          and "--resume" in out.getvalue() and "A/full ended with" not in out.getvalue(),
          out.getvalue()[:1500])

    # Tool checks 3: what each cancellation did to the payment, as the trace
    # recorded it, and the refunds refused as already settled -- one in the
    # step after the cancellation (the agent had read the tool's note), one
    # in the same response as it (the agent had not).
    from pasarbench.tools import SETTLED
    cell = tmp / "S" / "full"
    cell.mkdir(parents=True)
    head = {"type": "header", "task_id": "CWP-SG.sg-en", "tool_checks": 3}
    cancel = {"name": "cancel_order", "ok": True, "payment": "refunded"}
    refused = {"name": "issue_refund", "ok": False, "error": SETTLED + "nothing left"}
    for n, steps in enumerate([[[cancel], [refused]], [[cancel, refused]],
                               [[{**cancel, "payment": "released"}]],
                               [[{**cancel, "payment": "unsettled"}]]]):
        recs = [head] + [{"type": "step", "step": i + 1, "tool_results": trs}
                         for i, trs in enumerate(steps)]
        (cell / f"CWP-SG.sg-en__r{n}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n")
    counted = [{"transcript_id": f.stem} for f in sorted(cell.glob("*.jsonl"))]
    (cell / "CWP-SG.sg-en__r9.jsonl").write_text(      # unfinished: no episode to count
        json.dumps(head) + "\n" + json.dumps({"type": "step", "step": 1,
                                              "tool_results": [cancel]}) + "\n")
    done, n_refused, later = cc._settlements(cell, counted)
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown):
        cc._print_settlements("S/full", done, n_refused, later)
    check("…and under tool checks 3 it counts what each cancellation did to the payment -- "
          "one it could not settle too -- and the refunds refused as already settled, "
          "telling one made after the agent had read the tool's note from one made in the "
          "same response, over the episodes the comparison counts",
          dict(done) == {"refunded": 2, "released": 1, "unsettled": 1}
          and (n_refused, later) == (2, 1)
          and "what cancellations in S/full did to the payment: refunded 2, released 1, "
              "unsettled 1; refunds or credit refused as already settled: 2, 1 of them in a "
              "step after the cancellation" in shown.getvalue(),
          shown.getvalue())


def _language_cell(root, run, task_id, plans, replies=("Oh, I didn't notice.",),
                   closing="off", first=0, **setup):
    """Episodes of one task -- a generated one too -- through the real loop,
    the customer saying `replies` in turn and then leaving, written as a sweep
    writes them, under an agent model called "m". `setup`: run_episode's
    exposure, guardrail or context, as a sweep would pass them."""
    from pasarbench.harness.simulator import ScriptedUser
    from pasarbench.harness.trace import TraceWriter
    from pasarbench.rescore import all_tasks
    from pasarbench.verifier import verify

    t = all_tasks()[task_id]
    for i, plan in enumerate(plans):
        w = TraceWriter(root=str(root), run_id=f"{run}/full")
        db = Database.fresh(t.db_patch)
        agent = ScriptedBackend(list(plan))
        agent.model = "m"
        res = run_episode(t, db, agent, trace=w, run_index=first + i,
                          simulator=ScriptedUser(list(replies)), closing=closing, **setup)
        v = verify(t, db)
        w.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
        w.close()
    return root / run / "full"


def test_language_lead():
    """RUNBOOK 1r: the duplicate-refund trap fails far more often outside
    English. language_lead.py asks of each conversation whether the agent
    escalated while the customer was there -- an escalation after she left is
    the closing phase's, not the conversation's -- and whether it came in the
    turn it found the refund or after she said more; compares each local task
    with its English twin only in the cells that ran both, leaves a cell with
    one side only out, and asks the same of every other trap the policy
    answers with an escalation."""
    print("\n=== language_lead reads the trap by language ===")
    import contextlib
    import io
    import re
    import tempfile
    from pathlib import Path

    order = {"order_id": "GO-A096E7"}
    look = ("get_order", order)
    esc = ("escalate_to_human", {**order, "category": "duplicate_refund",
                                 "reason": "refund already issued"})
    told = "Đơn hàng đã được hoàn tiền rồi."
    tmp = Path(tempfile.mkdtemp())
    a = _language_cell(tmp, "A", "DRE-VN", [[look, esc, "I have passed this to a colleague."],
                                            [look, "A refund was already issued.", esc,
                                             "I have passed this on as well."]])
    _language_cell(tmp, "A", "DRE-VN.vi", [[look, told], ["Bạn cho mình xin mã đơn hàng nhé?"]])
    # escalated only after she left: the closing phase's, not the conversation's
    _language_cell(tmp, "A", "DRE-VN.vi", [[look, told, esc, "Đã ghi chú."]], replies=(),
                   closing="check", first=2)
    che = ("escalate_to_human", {"order_id": "GO-5BF80C", "category": "customs_hold",
                                 "reason": "parcel held at customs"})
    _language_cell(tmp, "A", "CHE-VN", [[che, "Escalated."]])
    _language_cell(tmp, "A", "CHE-VN.vi", [[che, "Đã chuyển."]])
    # English only: no twin to compare with, so left out
    _language_cell(tmp, "B", "DRE-VN", [[look, "A refund was already issued."]])

    lead = _load_script("language_lead")
    when = sorted((r["task"], r["escalated"], r["when"]) for r in lead.scan([tmp])
                  if r["trap"] == lead.TRAP)
    check("each conversation: escalated in the turn it found the refund, after she said "
          "more, never -- and an escalation after she left does not count",
          when == [("DRE-VN", False, None), ("DRE-VN", True, lead.AFTER),
                   ("DRE-VN", True, lead.BEFORE), ("DRE-VN.vi", False, None),
                   ("DRE-VN.vi", False, None), ("DRE-VN.vi", False, None)], str(when))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        lead.main([str(tmp)])
    text = out.getvalue()
    check("the cells that ran both sides, and the one left out",
          "== m: 6 episodes of the duplicate-refund trap in 2 cells, 5 of them in the 1 cells "
          "that ran it in English and another language both" in text
          and "left out, one side only: B/full" in text
          and re.search(r"\n  A/full +2/2 100% +0/3   0%\n", text) is not None, text[:1500])
    check("by language, with how many escalated in the turn it found the refund",
          re.search(r"\n    en +2/2 100%   1 of them before she said more\n    vi +0/3   0%   "
                    r"0 of them before she said more\n", text) is not None, text[:1500])
    check("each local task against its English twin in the cells that ran both, and the sign "
          "test",
          re.search(r"DRE-VN\.vi +0/3   0%   English twin DRE-VN: 2/2 100%\n  below their "
                    r"English twin 1 of 1, above 0, level 0; mean gap -100 points; sign test "
                    r"p=1\.0000", text) is not None, text[:2500])
    check("the same split within each path to the lookup -- the episode that never looked "
          "the order up has none",
          re.search(r"0 customer turns before it, identity not checked first +2/2 100% +0/2   0%",
                    text) is not None, text[-1800:])
    check("…and every trap the policy answers with an escalation, twin against twin",
          re.search(r"customs_hold_escalate +below their English twin 0 of 1, above 0, level 1",
                    text) is not None
          and re.search(r"duplicate_refund_escalate +below their English twin 1 of 1", text)
          and re.search(r"out_of_window_dispute_escalate +-\n", text) is not None, text[-1800:])
    check("…and every trap on the verdicts the runs recorded, the episode with a closing "
          "phase left out, the largest gap first (WHAT_FAILED #40)",
          re.search(r"every trap, passed -- on the verdicts the runs recorded, in the cells "
                    r"without a closing phase; with 2 traps, a gap stands on its own at "
                    r"p < 0\.0250:\n    duplicate_refund_escalate +below their English twin 1 "
                    r"of 1, above 0, level 0; mean gap -100 points; sign test p=1\.0000\n    "
                    r"customs_hold_escalate +below their English twin 0 of 1, above 0, level 1",
                    text) is not None, text[-1800:])


def test_claim_reader_check():
    """RUNBOOK 1u calibrates the claim reader on recorded replies before it
    holds back a live one: the replies the English list holds back, a sample
    of every language, and the duplicate refund's replies after the lookup --
    where an alarm on the refund paid before would send the agent to pay it
    again. The report reads the quotes first, then that, then English against
    the list, then every other language."""
    print("\n=== claim_reader_check calibrates the reader on recorded replies ===")
    import contextlib
    import io
    import json
    import re
    import tempfile
    from pathlib import Path

    from pasarbench.harness.types import ModelResponse, Usage

    tmp = Path(tempfile.mkdtemp())
    order = {"order_id": "GO-A096E7"}
    look = ("get_order", order)
    _language_cell(tmp, "A", "DRE-VN", [[look, "I've escalated your case to a specialist.",
                                         "Noted."]])
    _language_cell(tmp, "A", "DRE-VN.vi", [[look, "Đơn hàng đã được hoàn tiền vào ngày 9/11.",
                                            "Cảm ơn bạn."]])
    _language_cell(tmp, "A", "CHE-VN.vi", [["Tôi đã chuyển yêu cầu của bạn lên bộ phận "
                                            "chuyên trách.", "Xin cảm ơn."]])
    crc = _load_script("claim_reader_check")
    sample = tmp / "s.jsonl"
    with contextlib.redirect_stdout(io.StringIO()):
        crc.main(["sample", str(tmp), "--out", str(sample), "--per-language", "5",
                  "--per-language-refund", "5"])
    rows = [json.loads(l) for l in sample.read_text(encoding="utf-8").splitlines()]
    by = {r["text"]: r for r in rows}
    en = by["I've escalated your case to a specialist."]
    vi = by["Đơn hàng đã được hoàn tiền vào ngày 9/11."]
    check("every reply the agent wrote her, with what was done when it was written: the "
          "list's alarm, the duplicate refund after the lookup",
          en["phrase"] == [["escalate_to_human", "I've escalated your case to a specialist."]]
          and "the English list holds it back" in en["why"] and en["after_lookup"]
          and vi["after_lookup"] and not vi["phrase"] and "get_order" in vi["done"]
          and not by["Tôi đã chuyển yêu cầu của bạn lên bộ phận chuyên trách."]["after_lookup"],
          str(rows[:2]))

    class Reader:
        """Reads an escalation in English and Vietnamese -- and, wrongly, the refund
        paid before the conversation as one this agent made."""
        model = "fake-reader"

        def chat(self, messages, tools):
            t = messages[1].content
            claims = ([{"action": "escalate", "quote": "I've escalated your case"}]
                      if "escalated" in t else
                      [{"action": "escalate", "quote": "Tôi đã chuyển yêu cầu của bạn"}]
                      if "chuyển yêu cầu" in t else
                      [{"action": "refund", "quote": "đã được hoàn tiền"},
                       {"action": "voucher", "quote": "a voucher nobody wrote"}]
                      if "hoàn tiền" in t else [])
            return ModelResponse(content=json.dumps({"claims": claims}, ensure_ascii=False),
                                 usage=Usage(1, 1))
    import argparse
    with contextlib.redirect_stdout(io.StringIO()):
        crc.cmd_read(argparse.Namespace(sample=str(sample), out="", cache=str(tmp / "c.json"),
                                        workers=2), backend=Reader())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        crc.main(["report", str(tmp / "s.read.jsonl")])
    text = out.getvalue()
    check("the quotes first: a claim whose words are not in the reply is dropped and counted, "
          "and over 5% the reader is not used",
          "3 claims kept, 1 dropped" in text and "(25.0%)" in text
          and "!! more than 5% dropped: the reader invents" in text, text[:600])
    check("…then the refund already paid: every alarm on a payment in the duplicate refund "
          "after the lookup, each to read",
          "the reader holds back 1 for a payment no call made" in text
          and "đã được hoàn tiền" in text.split("ENGLISH")[0], text)
    check("…English against the list, and every alarm outside English",
          "both hold back 1, the list only 0, the reader only 0" in text
          and "Tôi đã chuyển yêu cầu của bạn" in text.split("BY LANGUAGE")[1], text)
    check("…under the quote check it names (WHAT_FAILED #50)",
          "read by fake-reader under quote check 2: 3 claims kept" in text
          and "!! read under more than one" not in text, text[:300])

    def row(rid, lang, text, phrase, unbacked, **kw):
        return {"id": rid, "language": lang, "text": text, "phrase": phrase,
                "unbacked": unbacked, "read": unbacked, "after_lookup": False,
                "reader": "fake-reader", "dropped": 0, "unreadable": False, **kw}
    esc = "escalate_to_human"
    mixed = tmp / "mixed.read.jsonl"
    mixed.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        row("X/full/CHE-MY.ms__r0#4", "ms", "I've escalated your case.",
            [[esc, "I've escalated your case"]], [[esc, "I've escalated your case"]],
            quote_check=2),
        row("X/full/OOWDE-MY.ms__r2#6", "ms", "If it's the refund, I'll escalate it now.",
            [[esc, "I'll escalate it now"]], [], quote_check=2),
        row("X/full/CHE-ID.id__r0#13", "id", "Kasus Anda sudah saya eskalasi.", [],
            [[esc, "Kasus Anda sudah saya eskalasi"]])]), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        crc.main(["report", str(mixed)])
    text = out.getvalue()
    lang = text.split("BY LANGUAGE")[1]
    check("…outside English, the list's alarms beside the reader's -- English words in a "
          "Malay conversation, held by both; the list's alone, to read -- and a reading "
          "under two quote checks said to be two readers'",
          "ms         1 of    2; the list    2, both    1" in lang
          and "I've escalated your case\"  [list too]" in lang
          and "Kasus Anda sudah saya eskalasi\"" in lang
          and "eskalasi\"  [list too]" not in lang
          and "I'll escalate it now" in lang.split("the list only")[1]
          and "under quote check 1 and 2" in text
          and "!! read under more than one quote check" in text, text)

    class Never:
        model = "fake-reader"

        def chat(self, messages, tools):
            raise AssertionError("the reader was asked")
    old = tmp / "old.read.jsonl"
    old.write_text(json.dumps({"id": "x", "read": [], "dropped": 1}) + "\n", encoding="utf-8")
    kept = old.read_text(encoding="utf-8")
    try:
        crc.cmd_read(argparse.Namespace(sample=str(sample), out=str(old),
                                        cache=str(tmp / "c.json"), workers=1), backend=Never())
        stopped = ""
    except SystemExit as e:
        stopped = str(e)
    # The cache as quote check 1 left it: no number, and its claims parsed away.
    cache = json.loads((tmp / "c.json").read_text(encoding="utf-8"))
    for v in cache.values():
        v.pop("quote_check", None)
        v["dropped"], v["claims"] = v["dropped"] + len(v["claims"]), []
    (tmp / "c.json").write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        crc.cmd_read(argparse.Namespace(sample=str(sample), out=str(tmp / "again.read.jsonl"),
                                        cache=str(tmp / "c.json"), workers=1), backend=Never())
    again = {r["text"]: r for r in (json.loads(l) for l in (
        tmp / "again.read.jsonl").read_text(encoding="utf-8").splitlines())}
    check("a reading made under quote check 1 is kept: `read` stops before asking anything; "
          "into another file, the answers cached under check 1 are read again from their "
          "raw text, without a call",
          "quote check 1" in stopped and "--out" in stopped
          and old.read_text(encoding="utf-8") == kept
          and len(again) == 6 and all(r["quote_check"] == 2 for r in again.values())
          and again["I've escalated your case to a specialist."]["read"]
          == [["escalate_to_human", "I've escalated your case"]], stopped)
    one = tmp / "one.read.jsonl"
    one.write_text(json.dumps(row("X/full/A__r0#1", "en", "Hi.", [], [])) + "\n",
                   encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        crc.main(["report", str(one)])
    check("…and a reading from before v61 is reported as made under check 1, with what "
          "that check could not find", "under quote check 1:" in out.getvalue()
          and "markdown and Thai marks were part of the match" in out.getvalue(),
          out.getvalue()[:400])
    from pasarbench.claim_reader import ClaimReader

    class Cut(Reader):
        """The same reader, its answer to one reply cut short."""

        def chat(self, messages, tools):
            if messages[1].content == "Xin cảm ơn.":
                return ModelResponse(content='{"claims": [', usage=Usage(1, 1))
            return super().chat(messages, tools)
    _language_cell(tmp, "B", "CHE-VN.vi", [["Tôi đã chuyển yêu cầu của bạn lên bộ phận "
                                            "chuyên trách.", "Xin cảm ơn."]],
                   guardrail="reader", claim_reader=ClaimReader(Cut()))

    class Counted(Reader):
        calls = 0

        def chat(self, messages, tools):
            Counted.calls += 1
            return super().chat(messages, tools)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        crc.cmd_delivered(argparse.Namespace(cells=[str(tmp / "B" / "full")],
                                             cache=str(tmp / "d.json"), workers=2),
                          backend=Counted())
    guarded = out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        crc.cmd_delivered(argparse.Namespace(cells=[str(tmp / "A" / "full")],
                                             cache=str(tmp / "c.json"), workers=2),
                          backend=Reader())
    text = out.getvalue()
    check("after a live run: every reply the customer saw that the reader reads as a claim "
          "no call backed, by cell and language, each with its verdict, to read",
          f"{tmp / 'A' / 'full'}: 3 of 3 episodes, 3 of them failed (en 1, vi 2)" in text
          and text.count("  failed") == 3
          and "6 replies, read now; 0 answers not the JSON asked for (those replies unread) "
              "and 1 claims dropped" in text, text)
    check("…the guarded arm read as the run read it, asking the reader nothing, with the "
          "answer it could not give counted",
          f"{tmp / 'B' / 'full'}: 0 of 1 episodes" in guarded and Counted.calls == 0
          and "as the run read them; 1 answers not the JSON asked for" in guarded, guarded)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _load_script("compare_cells").main([str(tmp / "A" / "full"), str(tmp / "B" / "full"),
                                            "--checker", "recorded"])
    text = out.getvalue()
    check("compare_cells names the reader a cell read with, and says its guardrail reading "
          "is not the other cell's instrument",
          re.search(r"claim_reader\s+-\s+fake-reader", text) is not None
          and "B/full with the reader fake-reader" in text
          and "A/full with the English list" in text, text[:1500])


def test_closing_room_compared():
    """v49: a closing phase may have room of its own. Two cells, one with it
    and one without, are not one experiment after the customer left, and
    compare_cells says so; two cells without it print as they always did."""
    print("\n=== compare_cells flags a closing phase with room of its own ===")
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    order = {"order_id": "GO-A096E7"}
    look = ("get_order", order)
    esc = ("escalate_to_human", {**order, "category": "duplicate_refund",
                                 "reason": "refund already issued"})
    tmp = Path(tempfile.mkdtemp())
    plans = [[look, "A refund was already issued.", esc, "Escalated."]]
    a = _language_cell(tmp, "A", "DRE-VN", plans, closing="check")
    b = _language_cell(tmp, "B", "DRE-VN", plans, closing="check", closing_room=True)
    c = _language_cell(tmp, "C", "DRE-VN", plans, closing="check")
    cc = _load_script("compare_cells")

    def compare(x, y):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cc.main([str(x), str(y), "--checker", "recorded"])
        return out.getvalue()
    flagged, plain = compare(a, b), compare(a, c)
    check("one cell with room and one without: the row, and the warning",
          "closing_room" in flagged and "!! one cell gave the closing phase room of its own"
          in flagged, flagged[:900])
    check("…two without it: neither", "closing_room" not in plain, plain[:900])


def test_language_swap():
    """RUNBOOK 1t holds a recorded duplicate-refund conversation fixed at the
    moment the agent reads the refund and changes only its language. A prefix
    must be what the agent saw -- the system prompt, every message, every
    payload replayed and checked, the world as those calls left it -- and only
    a conversation that can be rebuilt exactly is one; the arms change only the
    words, through one cache; the agent then continues until it next speaks;
    and the report pairs the arms by prefix, both directions apart."""
    print("\n=== the language swap: a conversation held fixed, its language changed ===")
    import argparse
    import contextlib
    import io
    import json
    import os
    import tempfile
    from pathlib import Path

    from pasarbench.harness.exposure import build_exposure
    from pasarbench.harness.prompts import system_prompt
    from pasarbench.harness.types import ModelResponse, ToolCall, Usage
    from pasarbench.rescore import all_tasks
    from pasarbench.swap import (Translator, arm_messages, continue_agent, decision_prefix,
                                 local_language)
    from pasarbench.sweep import ALL_SOLUTIONS

    tasks = all_tasks()
    tmp = Path(tempfile.mkdtemp())
    traces = tmp / "traces"

    def plan(tid, escalate):
        oid = tasks[tid].hidden_facts["order_id"]
        look = ("get_order", {"order_id": oid})
        esc = ("escalate_to_human", {"order_id": oid, "category": "duplicate_refund",
                                     "reason": "refund already issued"})
        return ["What is your order number?", look] + ([esc] if escalate else []) + ["Noted."]

    def cell(run, tid, escalates, **setup):
        oid = tasks[tid].hidden_facts["order_id"]
        return _language_cell(traces, run, tid, [plan(tid, e) for e in escalates],
                              replies=(oid, "Oh, I didn't notice."), **setup)
    a = cell("A", "DRE-MY", [True, False])
    cell("A", "DRE-MY.ms", [False])
    cell("A", "DRE-MY.zh-MY", [False])
    cell("A", "DRE-SG.sg-en", [True])
    cell("A", "DRE-PH", [False])
    cell("B", "DRE-MY.ms", [False], guardrail="claims")
    cell("C", "DRE-MY.ms", [False], exposure=build_exposure("search-300", ALL_SOLUTIONS))
    stale = a / "DRE-MY__r9.jsonl"
    recs = [json.loads(l) for l in (a / "DRE-MY__r0.jsonl").read_text().splitlines()]
    stale.write_text("\n".join(json.dumps({**r, "task_digest": "000000000000"}
                                          if r["type"] == "header" else r) for r in recs) + "\n")
    ls = _load_script("language_swap")
    found, why = ls.find_prefixes([traces], "m")
    check("the conversations that can be rebuilt exactly, and why the others cannot: a claim "
          "guardrail, a task changed since the run, a market with no other language",
          len(found) == 6 and why == {"claim guardrail": 1,
                                      "task or world changed since the run": 1,
                                      "no other language in its market": 1}, str(why))
    p = next(x for x in found if x.pid == "A.full.DRE-MY.r0")
    t = tasks["DRE-MY"]
    roles = [m["role"] for m in p.messages]
    payload = json.loads(p.messages[-1]["content"])
    check("a prefix is the agent's context up to the lookup: its system prompt, her opening and "
          "turns, its replies and calls, and the lookup's payload with the refund already paid",
          roles == ["system", "user", "assistant", "user", "assistant", "tool"]
          and p.messages[0]["content"] == system_prompt(t.user_id, t.market, "preload")
          and p.messages[1]["content"] == t.opening
          and p.messages[3]["content"] == t.hidden_facts["order_id"]
          and p.messages[4]["tool_calls"][0]["name"] == "get_order"
          and payload["order"]["existing_refunds"]
          and (p.steps, p.turns) == (2, 1), str(roles))
    other = next(x for x in found if x.pid == "A.full.DRE-MY.r1")
    check("…with what the recording did next: escalated before its next reply, or not",
          p.escalated and not other.escalated and not p.escalated_later)
    by = {x.pid: x for x in found}
    check("English is swapped into its market's other language, every other into English, "
          "Singlish into Singapore's Chinese",
          by["A.full.DRE-MY.ms.r0"].target == "en" and p.target in ("ms", "zh-MY")
          and by["A.full.DRE-SG.sg-en.r0"].target == "zh-SG"
          and local_language(by["A.full.DRE-SG.sg-en.r0"]) == "zh-SG"
          and by["C.full.DRE-MY.ms.r0"].head["exposure"] == "search-300")
    lookup = next(i for i, r in enumerate(recs) if r.get("type") == "step"
                  and any(tc["name"] == "get_order" for tc in r["tool_calls"]))
    both = [dict(r) for r in recs]
    both[lookup] = {**both[lookup],
                    "tool_calls": both[lookup]["tool_calls"] + [{"id": "x", "name":
                                                                 "escalate_to_human",
                                                                 "arguments": {}}],
                    "tool_results": both[lookup]["tool_results"] + [{"ok": True}]}
    windowed = [{**r, "context": "window8+pin"} if r["type"] == "header" else r for r in recs]
    check("…and none where the escalation came in the lookup's own step, or under a context "
          "strategy that showed the agent less",
          decision_prefix(both, t)[1] == "escalated in the lookup's own step"
          and decision_prefix(windowed, t)[1] == "context strategy not full")
    picked = ls.choose(found, 1)
    pfile = tmp / "found.jsonl"
    pfile.write_text("".join(json.dumps(x.to_dict(), default=str) + "\n" for x in found))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = ls.main(["check", str(pfile)])
    check("before any spend, each prefix continued with a probe is shown what the recording's "
          "next step was: as many messages, the same tools -- under tool search too",
          code == 0 and "6 prefixes continued with a probe; 0 shown other" in out.getvalue(),
          out.getvalue())
    check("one prefix a group at most -- each language other than English, in each "
          "direction -- never chosen by what the agent did",
          sorted(ls.group(x) for x in picked)
          == sorted({ls.group(x) for x in found}) and len(picked) == len({ls.group(x)
                                                                          for x in found}))

    class FakeTranslator:
        model = "fake-tr"
        calls = 0

        def chat(self, messages, tools):
            FakeTranslator.calls += 1
            lang = messages[0].content.split("Translate it into ")[1].split(". Keep")[0]
            text = messages[1].content.split("] ", 1)[-1]
            return ModelResponse(content=f"[{lang}] {text}", usage=Usage(1, 1))
    cache = tmp / "tr.json"
    tr = Translator(FakeTranslator(), cache)
    ms = by["A.full.DRE-MY.ms.r0"]
    swap, back = arm_messages(ms, "swap", tr), arm_messages(ms, "back", tr)
    calls = FakeTranslator.calls
    arm_messages(ms, "swap", tr)
    tr.save()
    again = Translator(FakeTranslator(), cache)
    arm_messages(ms, "back", again)
    words = [i for i, m in enumerate(ms.messages) if m["role"] in ("user", "assistant")
             and m["content"].strip()]
    check("the arms change the words and nothing else: the system prompt, the calls and the "
          "payloads as recorded, every reply and turn translated -- and back again",
          arm_messages(ms, "orig", tr) == ms.messages
          and all(swap[i] == ms.messages[i] and back[i] == ms.messages[i]
                  for i in range(len(ms.messages)) if i not in words)
          and all(swap[i]["content"].startswith("[English] ")
                  and back[i]["content"].startswith("[Bahasa Malaysia] ") for i in words)
          and [m.get("tool_calls") for m in swap] == [m.get("tool_calls") for m in ms.messages],
          str([m["content"][:30] for m in swap]))
    check("…through one cache: a text is translated once, and a rerun reads it from the file",
          FakeTranslator.calls == calls == 2 * len(words) and calls == FakeTranslator.calls)

    class FakeAgent:
        reports_usage = False
        name, model = "fake", "m"

        def chat(self, messages, tools):
            """Escalates when her last turn is in English or Singlish -- after
            finding the tool, where the tools are behind search."""
            last = next(m.content for m in reversed(messages) if m.role == "user")
            done = any(tc.name == "escalate_to_human" for m in messages for tc in m.tool_calls)
            if last.startswith(("[English]", "[Singlish")) and not done:
                if "escalate_to_human" not in {t["function"]["name"] for t in tools}:
                    return ModelResponse(tool_calls=[ToolCall("search_tools", {
                        "query": "escalate to a human agent"})], usage=Usage(1, 1))
                order = json.loads(next(m.content for m in reversed(messages)
                                        if m.role == "tool" and '"order"' in m.content)
                                   )["order"]["order_id"]
                return ModelResponse(tool_calls=[ToolCall("escalate_to_human", {
                    "order_id": order, "category": "duplicate_refund",
                    "reason": "already refunded"})], usage=Usage(1, 1))
            return ModelResponse(content="That refund was already paid.", usage=Usage(1, 1))
    tm = tasks[ms.task_id]
    out_swap = continue_agent(ms, swap, tm, FakeAgent())
    out_orig = continue_agent(ms, ms.messages, tm, FakeAgent())
    check("the agent continues from the prefix until it next speaks to the customer, in the "
          "prefix's world: an escalation it makes there is counted, with its category",
          out_swap.escalated and out_swap.category == "duplicate_refund"
          and out_swap.calls == ["escalate_to_human"] and out_swap.reply
          and not out_orig.escalated and out_orig.calls == [] and out_orig.stop == "done",
          str((out_swap, out_orig)))

    run = argparse.Namespace(prefixes=str(tmp / "P.jsonl"), limit=0, k=2,
                             out_root=str(tmp / "swap"), run_id="W", dry_run=False,
                             translator_model="fake-tr", cache=str(tmp / "tr2.json"),
                             workers=3, max_tokens=256, max_steps=4,
                             base_url="http://localhost:9/v1",
                             translator_url="http://localhost:9/v1", translator_extra_body="")
    with contextlib.redirect_stdout(io.StringIO()):
        ls.main(["prefixes", str(traces), "--model", "m", "--out", run.prefixes,
                 "--per-group", "8"])
        ls.cmd_run(run, agent_factory=lambda p: FakeAgent(),
                   translator_backend=FakeTranslator())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ls.main(["report", str(tmp / "swap" / "W")])
    text = out.getvalue()
    heads = [json.loads(f.read_text().splitlines()[0])
             for f in (tmp / "swap" / "W").rglob("*.jsonl")]
    arms = (tmp / "swap" / "W" / "arms.md").read_text(encoding="utf-8")
    check("…and every prefix's words in its three arms side by side, to read before the "
          "results", "## A.full.DRE-MY.ms.r0" in arms and "  - orig: " in arms
          and "  - swap: [English] " in arms and "  - back: [Bahasa Malaysia] " in arms,
          arms[:600])
    check("the run writes each continuation's trace with the arm in its header and the task's "
          "own id, under the source's tool exposure",
          len(heads) == 6 * 3 * 2 and all(h["task_id"] in tasks and h["swap"]["arm"]
                                           in ("orig", "swap", "back") for h in heads)
          and {h["exposure"] for h in heads if h["swap"]["pid"].startswith("C.")}
          == {"search-300"}, str(heads[0])[:300])
    check("the report: by arm and direction, the recording reproduced, and English against "
          "the other language paired by prefix, each direction on its own",
          "36 continuations of m from 6 prefixes" in text
          and "all prefixes: +1.000; 6 prefixes higher, 0 lower, sign test p=0.031  ->  "
              "LANGUAGE: English higher" in text
          and "recorded in English or Singlish: fewer than 5 prefixes" in text
          and "the orig arm, by the majority of its runs, agrees with it on 4 of 6" in text
          and "under 75%" in text and "THE TRANSLATOR" in text, text)
    check("…and how much room the agent left the test: here it escalated in half the "
          "translated arms' runs", "ROOM: the translated arms escalated in 50.0% of their runs\n"
          in text and "NO ROOM" not in text, text)

    class Always(FakeAgent):
        """Escalates in every language: an agent that leaves the test no room."""

        def chat(self, messages, tools):
            if any(tc.name == "escalate_to_human" for m in messages for tc in m.tool_calls):
                return ModelResponse(content="Noted.", usage=Usage(1, 1))
            if "escalate_to_human" not in {t["function"]["name"] for t in tools}:
                return ModelResponse(tool_calls=[ToolCall("search_tools", {
                    "query": "escalate to a human agent"})], usage=Usage(1, 1))
            order = json.loads(next(m.content for m in reversed(messages)
                                    if m.role == "tool" and '"order"' in m.content)
                               )["order"]["order_id"]
            return ModelResponse(tool_calls=[ToolCall("escalate_to_human", {
                "order_id": order, "category": "duplicate_refund",
                "reason": "already refunded"})], usage=Usage(1, 1))
    full = argparse.Namespace(**{**vars(run), "run_id": "W-always", "k": 1})
    with contextlib.redirect_stdout(io.StringIO()):
        ls.cmd_run(full, agent_factory=lambda p: Always(), translator_backend=FakeTranslator())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ls.main(["report", str(tmp / "swap" / "W-always")])
    check("v62, RUNBOOK 1w: an agent that escalates in every language leaves the test nothing "
          "to see -- its INCONCLUSIVE is NO ROOM, not a null",
          "all prefixes: +0.000" in out.getvalue() and "INCONCLUSIVE" in out.getvalue()
          and "escalated in 100.0% of their runs -- outside 10%-90%: an INCONCLUSIVE above "
              "is NO ROOM" in out.getvalue(), out.getvalue())

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ls.main(["report", str(tmp / "swap" / "W"), "--alpha", "0.0167"])
    g = {"diff": -0.01, "better": 20, "worse": 7, "p": 0.019}
    check("a test resolves below the p the reading names -- 1w's three agents, 0.0167 each -- "
          "and a verdict points the way its sign test does, not the mean's; the room's edges "
          "are 10% and 90%, inclusive",
          "a test resolves at p < 0.0167" in out.getvalue()
          and "6 prefixes higher, 0 lower, sign test p=0.031  ->  INCONCLUSIVE" in out.getvalue()
          and "->  UP" in ls._line("x", g, "UP", "DOWN")
          and all("NO ROOM" in ls._room(r) for r in (0.0, 0.1, 0.9, 1.0))
          and not any("NO ROOM" in ls._room(r) for r in (0.1001, 0.5, 0.8999)),
          out.getvalue()[:300])

    # RUNBOOK 1v: the policy in the conversation's language.
    import re
    from pasarbench.harness.prompts import policy_text
    from pasarbench.swap import (POLICY_BREAK, POLICY_LANGUAGES, TRANSLATE_POLICY, Prefix,
                                 policy_arm_messages, policy_missing, policy_unchangeable,
                                 translate_policy)
    from pasarbench.tools import policy_sections, policy_shown_as, search_policy

    code_of = {name: code for code, name in POLICY_LANGUAGES.items()}
    script_word = {"zh-MY": "字", "zh-SG": "字", "th": "คำ", "vi": "từ"}

    def as_kata(text, lang):
        """A "translation" that keeps every code, number, section number and
        mark of markdown and changes every word: what a search of it finds is
        its own. A word is "kata", or one in the language's own script, every
        word, as a translation into it would be (WHAT_FAILED #51); the mark at
        its end names the language the prompt asked for, or "?"."""
        code = code_of.get(lang, "?")
        word = script_word.get(code)
        swap = (lambda x: re.sub(r"\b[A-Za-z]+\b", word, x)) if word else \
            (lambda x: re.sub(r"\b[A-Za-z]{3,}\b", "kata", x))
        kept = re.split(r"(`[^`]*`)", text)
        return "".join(x if x.startswith("`") else swap(x) for x in kept) + f"\n<!-- in {code} -->"

    class PolicyTranslator(FakeTranslator):
        """A chat as FakeTranslator translates it; a piece of the policy as
        as_kata does -- or, told to, with a code lost."""
        lose = ""

        def chat(self, messages, tools):
            prompt = messages[0].content
            if "customer-service policy" not in prompt:
                return super().chat(messages, tools)
            lang = prompt.split("Translate it into ")[1].split(", in the register")[0]
            text = messages[1].content.replace(self.lose, "") if self.lose else messages[1].content
            return ModelResponse(content=as_kata(text, lang), usage=Usage(1, 1))
    pt = Translator(PolicyTranslator(), tmp / "tr3.json")
    ms_policy, lost = translate_policy(policy_text(), "ms", pt.policy)
    secs = policy_sections(ms_policy)
    english = policy_sections(policy_text())
    check("the policy is translated piece by piece -- its preamble, then P1 to P11 -- each "
          "piece checked, the lines of --- kept, and keyed by its prompt apart from the chat's",
          not lost and ms_policy.count("<!-- in ms -->") == 12
          and ms_policy.count(POLICY_BREAK) == policy_text().count(POLICY_BREAK)
          and [h.split(".")[0] for h, _ in secs] == [h.split(".")[0] for h, _ in english]
          and "kata" in secs[9][1] and "refund" not in secs[9][1]
          and pt.key("x", "ms", ("policy", "p")) != pt.key("x", "ms"), str(lost))
    pieces = [c.strip() for c in policy_text().split(POLICY_BREAK)]
    p7, p8, p10 = pieces[7], pieces[8], pieces[10]
    no_bullet = "\n".join(line for line in p10.splitlines() if "already been refunded" not in line)
    check("…what each piece must keep: its codes, every section number even where Chinese or "
          "Thai touches it, every number as often as the English has it, its list items and "
          "its bold -- so a rule left out, a 5 made 15 or a **not** dropped is named",
          policy_missing(p7, p7.replace(" P4.1 / P4.2", "按照P4.1/P4.2中")
                         .replace("P3.1, P3.3 and P6", "P3.1、P3.3和P6")) == []
          and "7 list item(s), 6 in the translation" in policy_missing(p10, no_bullet)
          and policy_missing(p8, p8.replace("more than 5 days", "more than 15 days"))
          == ["5 (x2, 1 in the translation)"]
          and policy_missing(p8, p8.replace("**not**", "not"))
          == ["6 bold(s), 5 in the translation"])
    p2, p5 = pieces[2], pieces[5]
    check("…where the English stresses a word in capitals, a script without them may stress "
          "it in bold: one bold more for each such word the translation does not keep in "
          "capitals -- a bold added anywhere else, or one dropped, is still named "
          "(WHAT_FAILED #51)",
          policy_missing(p2, p2.replace("NOT possible", "**不可**能")) == []
          and policy_missing(p5, p5.replace("IS the amount", "**คือ** the amount")) == []
          and policy_missing(p2, p2.replace("Do not attempt it", "**Do not** attempt it"))
          == ["4 bold(s), 5 in the translation"]
          and policy_missing(p2, p2.replace("NOT", "**NOT**"))
          == ["4 bold(s), 5 in the translation"]
          and policy_missing(p2, p2.replace("NOT", "不").replace("**P2.4**", "P2.4"))
          == ["4 bold(s), 3 in the translation (1 more allowed, for words in capitals)"])
    zh_p1 = as_kata(pieces[1], POLICY_LANGUAGES["zh-MY"])
    ms_p1 = as_kata(pieces[1], POLICY_LANGUAGES["ms"])
    check("…and, told its language, a piece must be in it: Chinese in Chinese characters, "
          "Thai in Thai, Vietnamese in its own letters, and in every language not the "
          "English's words -- Chinese (Malaysia)'s P1 came back in Malay, every code and "
          "number in place (WHAT_FAILED #51); the prompt names Chinese in its own script, asks "
          "for every sentence in the language, and for bold on the English's bold alone -- not, "
          "as v65's did, a capital's stress in bold, which had the translator bold every "
          "prohibition",
          policy_missing(pieces[1], zh_p1, "zh-MY") == []
          and policy_missing(pieces[1], ms_p1, "ms") == []
          and policy_missing(pieces[1], as_kata(pieces[1], POLICY_LANGUAGES["th"]), "th") == []
          and policy_missing(pieces[1], as_kata(pieces[1], POLICY_LANGUAGES["vi"]), "vi") == []
          and policy_missing(pieces[1], ms_p1, "zh-MY")
          == ["not in its language: 0% of its letters are Chinese characters"]
          and policy_missing(pieces[1], ms_p1, "vi")
          == ["not in its language: 0% of its letters are Vietnamese letters"]
          and all(policy_missing(e, e, lang)[-1].startswith(("not in its language: 0%",
                                                             "not translated: 100%"))
                  for e in pieces for lang in POLICY_LANGUAGES)
          and policy_missing(pieces[1], ms_p1) == []
          and all("简体中文" in POLICY_LANGUAGES[z] for z in ("zh-MY", "zh-SG"))
          and "none in another language" in TRANSLATE_POLICY
          and "on the words the English sets in bold, and on no others" in TRANSLATE_POLICY
          and "capitals" not in TRANSLATE_POLICY)
    loser = PolicyTranslator()
    loser.lose = "`identity_unverified`"
    _, lost = translate_policy(policy_text(), "ms", Translator(loser).policy)
    check("…and a piece that lost a code is named, with what it lost",
          lost == [(1, ["`identity_unverified`"])], str(lost))
    db = Database.fresh(tasks["DRE-MY"].db_patch)
    found = search_policy(db, "refund already escalate")["results"]
    at = [next(i for i, (h, _) in enumerate(english) if h == x["section"]) for x in found]
    with policy_shown_as(secs):
        shown = search_policy(db, "refund already escalate")["results"]
        local = search_policy(db, "kata")["results"]
    try:
        with policy_shown_as(secs[:3]):
            pass
        misaligned = False
    except ValueError:
        misaligned = True
    check("search_policy finds what it always finds, in the English, and inside "
          "policy_shown_as returns those sections in the other language; a query in her "
          "language, which finds nothing in the English, finds the policy she is read; "
          "outside it, as before; sections that would not pair up are refused",
          [x["section"] for x in shown] == [secs[i][0] for i in at] and len(at) == 3
          and all(x["text"] == secs[i][1] for x, i in zip(shown, at))
          and len(local) == 3 and not search_policy(db, "kata")["results"]
          and search_policy(db, "refund already escalate")["results"] == found and misaligned)

    traces2 = tmp / "traces2"

    def cell2(run, tid, **setup):
        oid = tasks[tid].hidden_facts["order_id"]
        return _language_cell(traces2, run, tid, [plan(tid, False)],
                              replies=(oid, "Oh, I didn't notice."), **setup)
    for tid in ("DRE-MY", "DRE-MY.ms", "DRE-VN.vi", "DRE-SG.sg-en"):
        cell2("A", tid)
    for tid in ("DRE-MY.zh-MY", "DRE-VN"):
        cell2("N", tid, policy_mode="preload-named")
    p2 = tmp / "P2.jsonl"
    with contextlib.redirect_stdout(io.StringIO()):
        ls.main(["prefixes", str(traces2), "--model", "m", "--out", str(p2)])
    two = {x.pid: x for x in ls.load_prefixes(p2)}
    named = two["N.full.DRE-MY.zh-MY.r0"]
    zh_policy, _ = translate_policy(policy_text(named=True), "zh-MY", pt.policy)
    arms2 = {arm: policy_arm_messages(named, arm, pt, zh_policy if arm == "policy" else None)
             for arm in ("local", "policy", "english")}
    check("1v's arms: local and policy are the same conversation in its language, english the "
          "same in English; only the policy arm's system prompt differs -- its policy, named "
          "tools and all, in her language, the instructions around it as they were",
          arms2["local"][1:] == arms2["policy"][1:]
          and arms2["local"][1:] == arm_messages(named, "back", pt)[1:]
          and arms2["english"][1:] == arm_messages(named, "swap", pt)[1:]
          and arms2["local"][0] == named.messages[0]
          and policy_text(named=True) not in arms2["policy"][0]["content"]
          and "(字: `escalate_to_human`)" in arms2["policy"][0]["content"]
          and "<!-- in zh-MY -->" in arms2["policy"][0]["content"]
          and arms2["policy"][0]["content"].startswith(named.messages[0]["content"][:300]),
          arms2["policy"][0]["content"][:200])
    searched = Prefix.from_dict({**named.to_dict(), "pid": "searched", "messages":
                                 named.messages[:2] + [{"role": "tool", "name": "search_policy",
                                                        "tool_call_id": "s", "content": "{}"}]
                                 + named.messages[2:]})
    jit = Prefix.from_dict({**named.to_dict(), "pid": "jit",
                            "head": {**named.head, "policy_mode": "jit"}})
    sfile = tmp / "P-searched.jsonl"
    sfile.write_text("".join(json.dumps(x.to_dict(), default=str) + "\n"
                             for x in (named, searched, jit)), encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ls.cmd_run(argparse.Namespace(**{**vars(run), "prefixes": str(sfile), "run_id": "V0",
                                             "policy": True, "dry_run": True}))
        refused = ""
    except SystemExit as e:
        refused = str(e)
    check("a prefix whose policy its prompt cannot change -- one that read it from "
          "search_policy before the lookup, one with no policy in its prompt -- stops a run "
          "before anything is spent, named",
          policy_unchangeable(named) == "" and "2 prefixes cannot" in refused
          and "searched: it read the policy from search_policy" in refused
          and "jit: policy mode 'jit' puts no policy in the prompt" in refused, refused)

    class PolicyReader(FakeAgent):
        """Searches the policy, then escalates if her last turn is in English, or
        if it read the policy in her language in its prompt and in the search;
        and says which policy it found."""

        def chat(self, messages, tools):
            found = [m.content for m in messages if m.role == "tool" and '"results"' in m.content]
            if not found:
                return ModelResponse(tool_calls=[ToolCall("search_policy", {
                    "query": "refund already escalate"})], usage=Usage(1, 1))
            last = next(m.content for m in reversed(messages) if m.role == "user")
            translated = "<!-- in " in found[-1]
            theirs = "<!-- in " in messages[0].content and translated
            if (theirs or last.startswith(("[English]", "[Singlish"))) and not any(
                    tc.name == "escalate_to_human" for m in messages for tc in m.tool_calls):
                order = json.loads(next(m.content for m in reversed(messages)
                                        if m.role == "tool" and '"order"' in m.content)
                                   )["order"]["order_id"]
                return ModelResponse(tool_calls=[ToolCall("escalate_to_human", {
                    "order_id": order, "category": "duplicate_refund",
                    "reason": "already refunded"})], usage=Usage(1, 1))
            return ModelResponse(content="searched " + ("translated" if translated else
                                                        "english"), usage=Usage(1, 1))
    run2 = argparse.Namespace(**{**vars(run), "prefixes": str(p2), "run_id": "V",
                                 "cache": str(tmp / "tr3.json"), "policy": True, "k": 1})
    bad = argparse.Namespace(**{**vars(run2), "run_id": "V-bad", "cache": str(tmp / "tr4.json")})
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ls.cmd_run(bad, agent_factory=lambda p: PolicyReader(), translator_backend=loser)
        stopped = ""
    except SystemExit as e:
        stopped = str(e)
    check("a run with the policy in her language does not start while a piece of the policy "
          "has lost something: it names the piece, and no continuation runs",
          "the run does not start" in stopped and "piece 1: `identity_unverified`"
          in out.getvalue() and not list((tmp / "swap" / "V-bad").rglob("*.jsonl")),
          out.getvalue() + stopped)
    step = argparse.Namespace(prefixes=str(p2), run_id="VP", out_root=str(tmp / "swap"),
                              cache=str(tmp / "tr5.json"), workers=2, limit=0,
                              translator_model="fake-tr", translator_url="http://localhost:9/v1",
                              translator_extra_body="")
    with contextlib.redirect_stdout(io.StringIO()):
        good = ls.cmd_policy(step, translator_backend=PolicyTranslator())
        lossy = ls.cmd_policy(argparse.Namespace(**{**vars(step), "run_id": "VP2",
                                                    "cache": str(tmp / "tr6.json")}),
                              translator_backend=loser)
    p10s = (tmp / "swap" / "VP" / "policy-translated" / "P10.md").read_text(encoding="utf-8")
    check("the policy step alone: every translation written out, and P10 in English and each "
          "language side by side to read before the run -- as the plain text has it and, "
          "where a prompt names the tools, as that text has it -- and it fails while a piece "
          "is lossy",
          good == 0 and lossy == 1 and "## English\n" in p10s and "## zh-MY\n" in p10s
          and "## English, the tools named" in p10s and "## zh-MY, the tools named" in p10s
          and "## vi, the tools named" in p10s and "## ms, the tools named" not in p10s
          and "already been refunded" in p10s and p10s.count("(tool: `escalate_to_human`)") == 1
          and p10s.count("(字: `escalate_to_human`)") == 1
          and p10s.count("<!-- in ") == 6, p10s[:300])
    with contextlib.redirect_stdout(io.StringIO()):
        ls.cmd_run(run2, agent_factory=lambda p: PolicyReader(),
                   translator_backend=PolicyTranslator())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ls.main(["report", str(tmp / "swap" / "V")])
    text = out.getvalue()
    recs2 = [[json.loads(l) for l in f.read_text().splitlines()]
             for f in (tmp / "swap" / "V").rglob("*.jsonl")]
    heads2 = [r[0] for r in recs2]
    said = {(r[0]["swap"]["arm"], next(x for x in r if x.get("kind") == "swap_outcome")["reply"])
            for r in recs2}
    written = sorted(f.name for f in (tmp / "swap" / "V" / "policy-translated").glob("*.md"))
    check("the run: three arms a prefix, each trace saying its arm, its conversation's "
          "language and its policy's; search_policy in her language in the policy arm only; "
          "every policy it read written out beside the traces",
          len(heads2) == 6 * 3 and all(h["swap"]["design"] == "policy" for h in heads2)
          and all(h["swap"]["language"] == (h["swap"]["local_language"]
                                            if h["swap"]["arm"] != "english" else
                                            ("sg-en" if h["swap"]["source_language"] == "sg-en"
                                             else "en")) for h in heads2)
          and {(h["swap"]["arm"], h["swap"]["policy_language"] == "en") for h in heads2}
          == {("local", True), ("english", True), ("policy", False)}
          and said == {("local", "searched english"), ("english", "searched english"),
                       ("policy", "searched translated")}
          and "zh-MY.named.md" in written and "zh-MY.plain.md" in written
          and "vi.named.md" in written and "ms.plain.md" in written and "P10.md" in written,
          str((said, written)))
    check("the report: 1t's test again, then the policy in her language against in English, "
          "then what is left of the gap -- and search_policy counted in each arm",
          "18 continuations of m from 6 prefixes" in text
          and "THE REPLICATION" in text and "LANGUAGE again: English higher" in text
          and "all prefixes: +1.000; 6 prefixes higher, 0 lower, sign test p=0.031  ->  "
              "POLICY: higher in her language" in text
          and text.split("THE GAP")[1].split("\n")[1].strip().startswith(
              "all prefixes: +0.000") and "INCONCLUSIVE" in text.split("THE GAP")[1]
          and "policy   6 of 6 continuations; 0 searches found nothing" in text, text)

    def srow(pid, arm, esc, first_english):
        return {"pid": pid, "arm": arm, "escalated": esc, "dup": esc, "design": "policy",
                "english_source": first_english, "local_language": "ms", "calls": [],
                "searched_nothing": 0}
    rows = [srow(f"e{i}", a, a == "english", True) for i in range(6)
            for a in ("local", "policy", "english")]
    rows += [srow(f"o{i}", a, a != "local", False) for i in range(6)
             for a in ("local", "policy", "english")]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ls._report_policy(rows)
    text = out.getvalue()
    test, gap = text.split("THE TEST")[1].split("THE GAP")
    check("…each direction apart: here the policy in her language moves only the "
          "conversations first held in another language, and English stays higher than the "
          "rest",
          "recorded in English/Singlish: +0.000; 0 prefixes higher, 0 lower, sign test "
          "p=1.000  ->  INCONCLUSIVE" in test
          and "recorded in another language: +1.000; 6 prefixes higher, 0 lower, sign test "
              "p=0.031  ->  higher" in test
          and "all prefixes: +0.500; 6 prefixes higher, 0 lower, sign test p=0.031  ->  "
              "English still higher" in gap
          and text.split("THE REPLICATION")[1].count("English higher") == 3, text)
    keys = {k: os.environ.pop(k) for k in list(os.environ)
            if k.endswith("_API_KEY") or k.startswith("PASARBENCH_")}
    try:
        os.environ["DEEPSEEK_API_KEY"] = "sk-agent"
        a2 = argparse.Namespace(base_url="https://api.deepseek.com/v1",
                                translator_url="https://dashscope-intl.aliyuncs.com/"
                                               "compatible-mode/v1")
        try:
            ls._keys(a2)
            refused = False
        except SystemExit as e:
            refused = "not sent to another provider" in str(e)
        os.environ["DASHSCOPE_API_KEY"] = "sk-translator"
        check("each provider gets its own key: the agent's is never sent to the translator's",
              refused and ls._keys(a2) == ("sk-agent", "sk-translator"))
    finally:
        for k in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
            os.environ.pop(k, None)
        os.environ.update(keys)


def test_add_model():
    """v55: one more model on the hard edition with one command (scripts/
    add_model.py; RUNBOOK 1s). Its key is typed, and sent only to its own
    endpoint: the runs get it in their environment with every other provider's
    key taken out (WHAT_FAILED #47); a key that is another provider's is
    refused; a Qwen model is refused, the customer being one; and the reading
    is written only once no conversation is left that an API error ended."""
    print("\n=== add_model: one more model, its key to its own endpoint ===")
    import contextlib
    import importlib.util
    import io
    import json
    import os
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from pasarbench.generate import EDITION3_PARTS
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("add_model", root / "scripts" / "add_model.py")
    am = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(am)
    cwd = os.getcwd()
    keys = {k: os.environ.pop(k) for k in list(os.environ) if k.endswith("_API_KEY")}
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = am.main(["--provider", "openai", "--model", "gpt-x", "--dry-run"])
        text = out.getvalue()
        parts = ",".join(dict.fromkeys(EDITION3_PARTS.values()))
        check("the plan: the 36 cases and their parts as RUNBOOK 1s runs them, against "
              "customer 3, resumable, on the provider's endpoint",
              rc == 0 and "--suite edition3 --run-id V-gpt-x" in text
              and f"--traps {parts} --run-id V-gpt-x-parts" in text
              and "--customer 3" in text and "--sim-model qwen3.8-flash" in text
              and "--resume" in text and "--base-url https://api.openai.com/v1" in text, text)

        def plan(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                am.main([*argv, "--dry-run"])
            return out.getvalue()
        glm = plan("--provider", "zai", "--model", "glm-5.3-flash")
        mimo = plan("--provider", "mimo", "--model", "mimo-v2.6-flash")
        grok = plan("--provider", "xai", "--model", "grok-4.3")
        slow = plan("--provider", "zai", "--model", "glm-5.3-flash", "--workers", "3")
        check("v59: fewer conversations at once for a provider that limits an account's "
              "open requests, the same runs otherwise",
              slow.count("--workers 3 --resume") == 2
              and slow.replace("--workers 3", "--workers 8") == glm, slow)
        check("Z.ai, Xiaomi and xAI each at its own endpoint: GLM-5.3-Flash, which cannot stop "
              "reasoning, at low effort with 4,096 tokens a step; MiMo, and Grok 4.3, with "
              "reasoning off",
              "--base-url https://api.z.ai/api/paas/v4 --extra-body "
              '{"reasoning_effort":"low"} --max-tokens 4096' in glm
              and "--run-id V-glm-5.3-flash\n" in glm
              and "--base-url https://api.xiaomimimo.com/v1 --extra-body "
                  '{"thinking":{"type":"disabled"}} --simulator' in mimo
              and "--max-tokens" not in mimo
              and "--base-url https://api.x.ai/v1 --extra-body "
                  '{"reasoning_effort":"none"} --simulator' in grok, glm + mimo + grok)

        tmp = Path(tempfile.mkdtemp())
        am.ROOT = tmp
        ran = []

        def fake_run(cmd, env=None, stdout=None, stderr=None):
            ran.append((cmd, env))
            if stdout is not None:
                stdout.write("read\n")
            return SimpleNamespace(returncode=0)

        am.subprocess = SimpleNamespace(run=fake_run, STDOUT=-2)
        am.preflight = lambda backend, url, model, key, extra, max_tokens=2048: model + "-served"
        os.environ.update(DEEPSEEK_API_KEY="sk-dsk", DASHSCOPE_API_KEY="sk-ds")
        for run in ("V-dsk", "V-dsk-parts"):
            (tmp / "traces" / run / "full").mkdir(parents=True)
            (tmp / "traces" / run / "full" / "T__r0.jsonl").write_text(
                json.dumps({"type": "header", "requested_model": "deepseek-v4-pro"}) + "\n"
                + json.dumps({"type": "footer", "stop_reason": "done"}) + "\n")
        answers = iter(["y"])
        with contextlib.redirect_stdout(io.StringIO()):
            rc = am.main(["--provider", "openai", "--model", "gpt-x"],
                         ask=lambda prompt: next(answers),
                         secret=lambda prompt: "sk-oai-typed")
        sweeps = [(c, e) for c, e in ran if "pasarbench.sweep" in c]
        others = [(c, e) for c, e in ran if "pasarbench.sweep" not in c]
        check("the typed key reaches the two runs as this run's key, with no other "
              "provider's key beside it and in no command line",
              rc == 0 and len(sweeps) == 2
              and all(e["PASARBENCH_API_KEY"] == "sk-oai-typed"
                      and e["PASARBENCH_SIM_API_KEY"] == "sk-ds"
                      and "DEEPSEEK_API_KEY" not in e and "DASHSCOPE_API_KEY" not in e
                      for _, e in sweeps)
              and not any(k in " ".join(c) for c, _ in ran
                          for k in ("sk-oai-typed", "sk-ds", "sk-dsk")), ran[:2])
        record = (tmp / "logs" / "add_model-gpt-x.log").read_text(encoding="utf-8")
        check("v59: the run keeps its own record in logs/ -- what ran, the check requests, "
              "each sweep's end, the reading -- and no key in it",
              "start: gpt-x on https://api.openai.com/v1" in record
              and "check requests ok: answered as gpt-x-served" in record
              and "sweep V-gpt-x ended, exit code 0" in record
              and "sweep V-gpt-x-parts ended, exit code 0" in record
              and "reading written: V-gpt-x.txt, paired with dsk" in record
              and not any(k in record for k in ("sk-oai-typed", "sk-ds", "sk-dsk")), record)
        check("…its reading and, for each model run before it, the pairing on the cases "
              "and on the parts, each run with no key at all",
              (tmp / "V-gpt-x.txt").exists()
              and any(c[-2:] == ["traces/V-dsk/full", "traces/V-gpt-x/full"] for c, _ in others)
              and any(c[-2:] == ["traces/V-dsk-parts/full", "traces/V-gpt-x-parts/full"]
                      for c, _ in others)
              and all(not any(k.endswith("_API_KEY") for k in e) for _, e in others),
              [c for c, _ in others])

        def refused(argv, typed="sk-oai-typed"):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    am.main(argv, ask=lambda prompt: "n", secret=lambda prompt: typed)
                return ""
            except SystemExit as e:
                return str(e)
        check("another provider's key typed in is refused before anything is sent",
              "DEEPSEEK_API_KEY" in refused(["--provider", "openai", "--model", "gpt-y"],
                                            typed="sk-dsk"))
        check("a Qwen model is refused: the customer is one",
              "is a Qwen model" in refused(["--provider", "openrouter", "--model",
                                            "qwen/qwen-max"]))
        check("a retired model its provider answers for with another is refused, with the "
              "one to run",
              "Run grok-4.3." in refused(["--provider", "xai", "--model",
                                          "grok-4-1-fast-non-reasoning"]))
        check("a name that holds another model's runs is refused",
              "already holds deepseek-v4-pro" in refused(["--provider", "openai", "--model",
                                                         "gpt-z", "--name", "dsk"]))
        (tmp / "traces" / "V-gpt-x" / "full").mkdir(parents=True)
        (tmp / "traces" / "V-gpt-x" / "full" / "T__r0.jsonl").write_text(
            json.dumps({"type": "header", "requested_model": "gpt-x"}) + "\n"
            + json.dumps({"type": "footer", "stop_reason": "backend_error"}) + "\n")
        (tmp / "V-gpt-x.txt").unlink()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = am.main(["--provider", "openai", "--model", "gpt-x", "--yes"],
                         ask=lambda prompt: "y", secret=lambda prompt: "sk-oai-typed")
        def broke(*a, **k):
            raise SystemExit("!! the check requests failed, so nothing ran:\n   HTTP 429 from "
                             "https://api.z.ai/api/paas/v4/chat/completions: {\"error\":{\"code\":"
                             "\"1113\",\"message\":\"Insufficient balance\"}}")
        kept, am.preflight = am.preflight, broke
        stopped = refused(["--provider", "zai", "--model", "glm-5.3-flash", "--name", "glm"],
                          typed="zai-typed-1")
        am.preflight = kept
        zrec = (tmp / "logs" / "add_model-glm.log").read_text(encoding="utf-8")
        check("…and a run that stops at the check requests says why there too, the "
              "provider's own words, so it can be read after the terminal is gone",
              "1113" in stopped and "stopped at the check requests, nothing ran" in zrec
              and "Insufficient balance" in zrec and "zai-typed-1" not in zrec
              and not (tmp / "traces" / "V-glm").exists(), zrec)
        check("a conversation an API error ended: run it again, and no reading until then",
              rc == 1 and "ended on a server or API error" in out.getvalue()
              and not (tmp / "V-gpt-x.txt").exists(), out.getvalue()[-300:])
    finally:
        os.chdir(cwd)
        for k in [k for k in os.environ if k.endswith("_API_KEY")]:
            os.environ.pop(k)
        os.environ.update(keys)


def test_add_model_check():
    """v56: which keys work and which models, before any run (add_model.py
    --check): each provider whose key is set, the models its key lists, and
    its newest fast models looked at the way a run uses them -- an order looked
    up with a tool, and an answer from what the tool returned. The file it
    writes names the variables, never their values."""
    print("\n=== add_model --check: which keys work, which models ===")
    import contextlib
    import importlib.util
    import io
    import json
    import os
    import tempfile
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("add_model", root / "scripts" / "add_model.py")
    am = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(am)
    seen = []

    class Fake(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": "fast-1"}, {"id": "big-2"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            if any(m["role"] == "tool" for m in req["messages"]):
                msg = {"role": "assistant", "content": "It has shipped with J&T Express."}
            else:
                msg = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "get_order", "arguments": '{"order_id": "GO-10001"}'}}]}
            self._send({"model": req["model"], "choices": [{"message": msg,
                                                           "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 20,
                                  "prompt_tokens_details": {"cached_tokens": 900 * (
                                      len(seen) > 1)}}})

    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    try:
        r = am.probe("openai", url, "fast-1", "EMPTY", '{"thinking":{"type":"disabled"}}')
        check("the probe is the loop a run is: the order looked up with a tool, the tool's "
              "answer sent back, an answer from it -- with the model's settings, at "
              "temperature 0, its 20 tools",
              r["tool"] == "get_order" and r["answered"] and r["requests"] == 2
              and r["prompt"] == 2000 and r["cached"] == (900, 1000) and len(seen) == 2
              and seen[0]["thinking"] == {"type": "disabled"} and seen[0]["temperature"] == 0
              and len(seen[0]["tools"]) == 20
              and seen[1]["messages"][-1]["role"] == "tool", (r, seen[-1]["messages"][-2:]))
        check("…and the models a key lists, from its /models",
              am.listed_models(url, "EMPTY") == ["big-2", "fast-1"])
    finally:
        srv.shutdown()

    check("GLM-5.3 and its Flash cannot stop reasoning: low effort and room for it; every "
          "other model of the table with reasoning off where its provider has a switch",
          am.settings_for("zai", "glm-5.3-flash") == ('{"reasoning_effort":"low"}', 4096)
          and am.settings_for("zai", "glm-4.7-flash")[0] == '{"thinking":{"type":"disabled"}}'
          and am.settings_for("mimo", "mimo-v2.6-flash")[0] == '{"thinking":{"type":"disabled"}}'
          and am.settings_for("xai", "grok-4.3") == ('{"reasoning_effort":"none"}', 2048)
          and am.settings_for("xai", "grok-4.20-0309-non-reasoning") == ("", 2048))

    tmp = Path(tempfile.mkdtemp())
    am.ROOT = tmp
    keys = {k: os.environ.pop(k) for k in list(os.environ) if k.endswith("_API_KEY")}
    calls = []

    def fake_probe(backend, url, model, key, extra, max_tokens=2048):
        calls.append((url, model, key, extra, max_tokens))
        if model == "grok-3-mini":
            raise RuntimeError("HTTP 404 from https://api.x.ai/v1/chat/completions: no model "
                               "for key xai-secret-1")
        return {"served": model, "requests": 2, "tool": "get_order", "answered": True,
                "prompt": 9000, "completion": 80, "seconds": 1.5, "cached": (4000, 4500)}

    am.probe = fake_probe
    am.listed_models = lambda url, key: ([
        "grok-4.3", "grok-4.20-0309-reasoning", "grok-4-1-fast-non-reasoning", "grok-2-mini",
        "grok-3-mini", "grok-5-fast", "grok-4.8-mini", "grok-imagine-image-fast"]
        if "x.ai" in url else ["glm-5.3-flash", "glm-4.5-air"])
    try:
        os.environ.update(GROK_API_KEY="xai-secret-1", GLM_API_KEY="glm-secret-2\r",
                          DASHSCOPE_API_KEY="ds-secret-3", MIMO_KEY="mimo-secret-4")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = am.main(["--check"], ask=lambda prompt: "n", secret=lambda prompt: "")
        text = (tmp / "models_check.txt").read_text(encoding="utf-8")
        check("--check finds each provider's key under the names it goes by, tries its newest "
              "fast models the key lists, says which it skipped and why one failed",
              rc == 0 and "key from GROK_API_KEY" in text and "key from GLM_API_KEY" in text
              and "grok-4.3                         OK" in text
              and 'settings {"reasoning_effort":"none"}' in text
              and "grok-4.20-0309-non-reasoning     not in your key's list" in text
              and "grok-3-mini                      FAIL HTTP 404" in text
              and "no model for key <key>" in text
              and "glm-5.3-flash                    OK" in text
              and "glm-5.3-flashx                   not in your key's list" in text
              and 'settings {"reasoning_effort":"low"}' in text
              and "The customer, qwen3.8-flash on DashScope, key from DASHSCOPE_API_KEY: OK"
              in text, text)
        tried = [m for u, m, _, _, _ in calls if "x.ai" in u]
        check("…and the fast models its key lists beyond those, newest first, three at most: "
              "no image model, no retired one, no model of another tier",
              tried == ["grok-4.3", "grok-5-fast", "grok-4.8-mini", "grok-3-mini"]
              and [m for u, m, _, _, _ in calls if "z.ai" in u] == ["glm-5.3-flash"], tried)
        check("a key under a name nothing reads is named, with the name to give it; a line "
              "end after a key is not sent",
              "MIMO_KEY is set, but nothing here reads that name: for Xiaomi (MiMo), name it "
              "MIMO_API_KEY" in text
              and all("\r" not in k for _, _, k, _, _ in calls), text)
        check("…each key sent only to its own provider, and the file naming the variables, "
              "never their values",
              all((("x.ai" in u) == (k == "xai-secret-1")) and (("z.ai" in u) == (k == "glm-secret-2"))
                  and (("dashscope" in u) == (k == "ds-secret-3")) for u, _, k, _, _ in calls)
              and "API key variables set (names only): DASHSCOPE_API_KEY, GLM_API_KEY, "
                  "GROK_API_KEY" in text
              and not any(v in text for v in ("xai-secret-1", "glm-secret-2", "ds-secret-3",
                                              "mimo-secret-4")),
              calls)

        # v57: the keys a .env held under names the shell or the runs do not read,
        # typed at --check instead, tried, and saved under the name the runs read.
        tmp2 = Path(tempfile.mkdtemp())
        am.ROOT = tmp2
        (tmp2 / ".env").write_text("DEEPSEEK_API_KEY=sk-dsk-9\nPASARBENCH_SIM_API_KEY=ds-sim-8\n"
                                   "ZAI_API_KEY = zai-typo-7\nmimo-bare-6\nGROK=grok-misnamed-5\n")
        for k in [k for k in os.environ if k.endswith("_API_KEY")] + ["MIMO_KEY"]:
            os.environ.pop(k, None)
        os.environ.update(DEEPSEEK_API_KEY="sk-dsk-9", PASARBENCH_SIM_API_KEY="ds-sim-8")
        calls.clear()
        typed = {"xAI": "sk-dsk-9", "Z.ai": "zai-new-4", "Xiaomi": "mimo-bad-3"}
        asked = []

        def secret(prompt):
            asked.append(prompt.split(" API key")[0])
            return next((v for k, v in typed.items() if prompt.startswith(k)), "")

        def fake_probe2(backend, url, model, key, extra, max_tokens=2048):
            calls.append((url, model, key, extra, max_tokens))
            if key == "mimo-bad-3":
                raise RuntimeError(f"HTTP 401 from {url}/chat/completions: bad key {key}")
            return {"served": model, "requests": 2, "tool": "get_order", "answered": True,
                    "prompt": 9000, "completion": 80, "seconds": 1.5, "cached": (4000, 4500)}
        am.probe = fake_probe2
        am.listed_models = lambda url, key: ("HTTP 401: invalid key" if key == "mimo-bad-3"
                                             else ["glm-5.3-flash"] if "z.ai" in url
                                             else ["deepseek-v4-pro"])
        with contextlib.redirect_stdout(io.StringIO()):
            rc = am.main(["--check"], ask=lambda prompt: "y", secret=secret)
        text = (tmp2 / "models_check.txt").read_text(encoding="utf-8")
        env_after = (tmp2 / ".env").read_text(encoding="utf-8")
        check("v57: with keys missing, --check says by name only what .env sets and which "
              "lines the shell does not read -- a space round '=', a line that is not "
              "NAME=value, a name nothing reads -- and never a value",
              rc == 0 and "No key in this shell for xAI (Grok), Z.ai (GLM), Xiaomi (MiMo)." in text
              and ".env sets (names only): DEEPSEEK_API_KEY, PASARBENCH_SIM_API_KEY, GROK" in text
              and "line 3: ZAI_API_KEY has a space round its '='" in text
              and "line 4: not NAME=value, so the shell does not read it" in text
              and "GROK: nothing here reads that name -- for xAI (Grok), name it XAI_API_KEY"
              in text, text)
        check("…asks for each missing key, refuses one that is another provider's before "
              "sending it, and saves a key that worked under the name the runs read, in "
              "place of the misspelt line, readable by you alone",
              asked == ["xAI (Grok)", "Z.ai (GLM)", "Xiaomi (MiMo)"]
              and "that is the key in DEEPSEEK_API_KEY, another provider's -- not sent" in text
              and not any("x.ai" in u for u, _, _, _, _ in calls)
              and "saved to .env as ZAI_API_KEY" in text
              and "ZAI_API_KEY=zai-new-4\n" in env_after and "zai-typo-7" not in env_after
              and "DEEPSEEK_API_KEY=sk-dsk-9\n" in env_after and "GROK=grok-misnamed-5" in env_after
              and (tmp2 / ".env").stat().st_mode & 0o077 == 0
              and os.environ.get("ZAI_API_KEY") == "zai-new-4", (asked, env_after))
        check("…a typed key no request worked with is not saved, and the customer's key is "
              "found as the sweep finds it, PASARBENCH_SIM_API_KEY first; no key in the file",
              "no request with it worked, so it is not saved" in text
              and "MIMO_API_KEY" not in env_after
              and "The customer, qwen3.8-flash on DashScope, key from PASARBENCH_SIM_API_KEY: OK"
              in text and any(k == "ds-sim-8" and "dashscope" in u for u, _, k, _, _ in calls)
              and not any(v in text for v in ("sk-dsk-9", "ds-sim-8", "zai-new-4", "mimo-bad-3",
                                              "zai-typo-7", "mimo-bare-6", "grok-misnamed-5")),
              text)

        tmp3 = Path(tempfile.mkdtemp())
        (tmp3 / ".env").write_text('DEEPSEEK_API_KEY=sk-from-file\n'
                                   'export ZAI_API_KEY="zai.v57" # Z.ai\n'
                                   'XAI_API_KEY=xai#7 # xAI\nMIMO_API_KEY = spaced\n')
        for k in [k for k in os.environ if k.endswith("_API_KEY")]:
            os.environ.pop(k)
        os.environ["DEEPSEEK_API_KEY"] = "sk-from-shell"
        loaded = am.load_dotenv(tmp3 / ".env")
        src = (root / "scripts" / "add_model.py").read_text()
        check("v57: the script reads .env itself, as `set -a; . ./.env` would -- quotes and a "
              "comment after a space taken off, a line with a space round '=' not read -- and "
              "keeps what this shell set, saying so by name when the two differ",
              loaded == ["ZAI_API_KEY", "XAI_API_KEY"] and os.environ["ZAI_API_KEY"] == "zai.v57"
              and os.environ["XAI_API_KEY"] == "xai#7"
              and os.environ["DEEPSEEK_API_KEY"] == "sk-from-shell"
              and "MIMO_API_KEY" not in os.environ
              and any("this shell's value is not .env's for DEEPSEEK_API_KEY" in n
                      for n in am.env_report(tmp3 / ".env"))
              and 'load_dotenv(ROOT / ".env")\n    raise SystemExit(main())' in src, loaded)
    finally:
        os.environ.pop("MIMO_KEY", None)
        for k in [k for k in os.environ if k.endswith("_API_KEY")]:
            os.environ.pop(k)
        os.environ.update(keys)


def test_export_hf():
    """v44 publishes the recorded runs as a Hugging Face dataset: one line per
    episode with its setup and verdict as columns beside the whole trace, a
    list of the runs, and a card written from what was exported -- and it
    refuses to publish a trace holding a key, a token or a home path, and
    replaces email addresses."""
    print("\n=== export_hf packs the traces as a dataset, and nothing that must stay ===")
    import contextlib
    import io
    import json
    import tempfile
    from pathlib import Path

    order = {"order_id": "GO-A096E7"}
    look = ("get_order", order)
    esc = ("escalate_to_human", {**order, "category": "duplicate_refund",
                                 "reason": "refund already issued"})
    tmp = Path(tempfile.mkdtemp())
    traces = tmp / "traces"
    cell = _language_cell(traces, "A", "DRE-VN", [[look, esc, "Passed on."],
                                                  [look, "Already refunded."]],
                          replies=("my email is someone@example.com",))
    _language_cell(traces, "B", "DRE-VN.vi", [[look, "Đã hoàn tiền rồi."]])
    ex = _load_script("export_hf")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ex.main([str(traces), "--out", str(tmp / "hf")])
    rows = [json.loads(l) for l in (tmp / "hf" / "data" / "A.jsonl").read_text(
        encoding="utf-8").splitlines()]
    check("one line per episode, its setup and verdict as columns beside the whole trace",
          len(rows) == 2 and all(list(r) == list(ex.FIELDS) for r in rows)
          and {r["passed"] for r in rows} == {True, False}
          and all(r["trap"] == "duplicate_refund_escalate" and r["language"] == "en"
                  and r["cell"] == "full" and r["trace"].startswith('{"type": "header"')
                  for r in rows), [list(r)[:6] for r in rows])
    check("…email addresses replaced and counted, the rest of the trace as written",
          all("example.com" not in r["trace"] for r in rows)
          and any("my email is [email]" in r["trace"] for r in rows)
          and "2 email addresses replaced in 2 episodes" in out.getvalue(), out.getvalue())
    runs = [json.loads(l) for l in (tmp / "hf" / "runs.jsonl").read_text().splitlines()]
    readme = (tmp / "hf" / "README.md").read_text(encoding="utf-8")
    check("a list of the run cells, and a card with the configs the dataset viewer reads",
          [(r["run"], r["cell"], r["episodes"]) for r in runs] == [("A", "full", 2),
                                                                   ("B", "full", 1)]
          and readme.startswith("---\nlicense: mit") and "data_files: data/*.jsonl" in readme
          and "- vi" in readme and "3 episodes in 2 runs" in readme, readme[:400])
    leak = Path(cell) / "DRE-VN__r0.jsonl"
    leak.write_text(leak.read_text(encoding="utf-8").replace(
        "Passed on.", "key sk-abcdefghijklmnopqrstu1234"), encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ex.main([str(traces), "--out", str(tmp / "hf2")])
        refused = ""
    except SystemExit as e:
        refused = str(e)
    check("…and a trace holding a key stops the export, named, before anything is written",
          "DRE-VN__r0.jsonl: an API key" in refused and not (tmp / "hf2").exists(), refused)


def test_photo_claims_compared():
    """WHAT_FAILED #43: on the photo trap, customer v2 says she sent the photos
    she was told she does not have, and the trap's verdict then follows her.
    compare_cells counts, in each cell, the episodes she says it in and the
    failures after and without it, so a reader of a paired comparison -- GRPO
    against RFT on the held-out fold -- sees whose verdicts that trap's are."""
    print("\n=== compare_cells counts the photos a customer says she sent ===")
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    look = ("get_order", {"order_id": "O1005"})
    ask = "Please send photos of the torn strap first; I cannot open the return without them."
    opened = ("initiate_return", {"order_id": "O1005", "order_item_id": "OI5",
                                  "reason": "torn strap", "photo_evidence_provided": True})
    tmp = Path(tempfile.mkdtemp())
    held = _language_cell(tmp, "H", "T07", [[look, ask]],
                          replies=("I can't send photos now, I'll send them tonight.",))
    took = _language_cell(tmp, "G", "T07", [[look, ask, opened, "Your return is open."]],
                          replies=("(Attached: photo_1.jpg) Here are the photos.",))
    cc = _load_script("compare_cells")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(held), str(took)])
    text = out.getvalue()
    block = text[text.find("customers told they have no photos"):]
    rows = {line.split("  ")[1].strip(): line.split()[-2:] for line in block.splitlines()[1:4]}
    check("each cell's episodes she said it in, and the failures after and without it",
          rows == {"episodes she said it in": ["0/1", "1/1"], "failures after it": ["0", "1"],
                   "failures without it": ["0", "0"]}, block[:600])
    check("…and what that makes of the trap's row, when she said it anywhere",
          "the photo trap's verdict follows her line" in block, block[:600])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(held), str(held)])
    check("…said only where she said it",
          "follows her line" not in out.getvalue()
          and "customers told they have no photos" in out.getvalue(), out.getvalue()[-600:])


def test_voucher_choice_compared():
    """WHAT_FAILED #46: on the voucher trap the agents offer "the voucher, or a
    human to review a full refund", and the customer -- who wants her money and
    will accept a voucher -- may take the human. P6.3 and P10 then ask for the
    escalation, and the trap's check fails it. simqa finds that road in a trace,
    diagnose reads it as the policy does, and compare_cells counts it in each
    cell and prints the paired line read that way beside the result."""
    print("\n=== the voucher trap's customer who declines the voucher ===")
    import contextlib
    import io
    import json
    import tempfile
    from pathlib import Path

    from pasarbench.diagnose import load_episodes, voucher_choice
    from pasarbench.rescore import all_tasks
    from pasarbench.simqa import declined_voucher, only_the_voucher
    from pasarbench.sweep import ALL_SOLUTIONS

    tasks = all_tasks()
    ids = ["OOWOV-SG", "OOWOV-MY", "OOWOV-ID", "OOWOV-TH", "OOWOV-PH"]
    offer = ("It is past the 14-day window. I can offer a goodwill voucher, or pass your "
             "case to a human agent to review a full refund. Which would you like?")
    tmp = Path(tempfile.mkdtemp())
    for tid in ids:
        verify, look, voucher = ALL_SOLUTIONS[tid]
        order = tasks[tid].hidden_facts["order_id"]
        esc = ("escalate_to_human", {"order_id": order, "category": "out_of_window_dispute",
                                     "reason": "voucher declined, customer asks for a refund"})
        _language_cell(tmp, "V", tid, [[look, offer, verify, voucher, "Your voucher is issued."]],
                       replies=("The voucher is fine. My digits are "
                                + tasks[tid].hidden_facts["phone_last4"] + ".", "Thanks."))
        _language_cell(tmp, "H", tid, [[look, offer, esc, "A human agent will be in touch."]],
                       replies=("I'd rather have a full refund. Please escalate this.",
                                "Thanks."))

    def recs(cell, tid):
        f = tmp / cell / "full" / f"{tid}__r0.jsonl"
        return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    took, gave = recs("V", "OOWOV-MY"), recs("H", "OOWOV-MY")

    def step(text, *calls):
        return {"type": "step", "model_content": text,
                "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                "tool_results": [{"name": n, "ok": True} for n, _ in calls]}
    esc = ("escalate_to_human", {"order_id": "O1", "category": "out_of_window_dispute"})
    her = {"type": "event", "kind": "user_turn", "text": "A refund, please."}
    early = [step(offer + " I have passed it on.", esc)]
    unasked = [step("It is past the window."), her, step("", esc)]
    paid = [step(offer), her, step("", ("issue_refund", {"order_id": "O1"}), esc)]
    over = step("", ("issue_goodwill_voucher", {"user_id": "U1", "amount_minor": 300000,
                                                "currency": "VND"}))
    over["tool_results"] = [{"name": "issue_goodwill_voucher", "ok": False,
                             "error": "voucher exceeds SGD 15 equivalent cap (P6.2)"}]
    took_refused = [step(offer), {"type": "event", "kind": "user_turn",
                                  "text": "I'll take the voucher."}, over,
                    {"type": "event", "kind": "user_turn", "text": "Then escalate, please."},
                    step("", esc)]
    check("v60, WHAT_FAILED #49: a voucher the agent tried and the tool refused means she "
          "took it -- the escalation she then asked for is not her declining it",
          not declined_voucher("two_orders_voucher_and_cancel", took_refused)
          and not declined_voucher("out_of_window_offer_voucher", took_refused))
    check("the road is found: the voucher offered, her answer, then the dispute's "
          "escalation and no payment -- not where she took the voucher, nor an escalation "
          "made before she could answer or with no voucher offered, nor one beside a "
          "refund, nor on another trap",
          declined_voucher("out_of_window_offer_voucher", gave)
          and declined_voucher("out_of_window_offer_voucher", [step(offer), her, step("", esc)])
          and not declined_voucher("out_of_window_offer_voucher", took)
          and not any(declined_voucher("out_of_window_offer_voucher", r)
                      for r in (early, unasked, paid))
          and not declined_voucher("duplicate_refund_escalate", gave))
    lines = gave[-1].get("failures") or []
    check("…and it fails on the voucher's lines alone, which only the voucher needed",
          gave[-1].get("passed") is False and only_the_voucher("out_of_window_offer_voucher",
                                                                lines)
          and not only_the_voucher("out_of_window_offer_voucher",
                                   lines + ["forbidden action taken: issue_refund {}"])
          and not only_the_voucher("two_orders_voucher_and_cancel",
                                   ["missing required action: verify_identity {}"]), str(lines))
    eps = load_episodes(tmp / "H" / "full", checker="recorded")
    read, n = voucher_choice(tmp / "H" / "full", eps, checker="recorded")
    check("diagnose reads them as P6.3 and P10 do, and counts them",
          n == {"episodes": 5, "declined": 5, "read_as_pass": 5, "failed_without": 0}
          and all(e["passed"] for e in read) and not any(e["passed"] for e in eps), str(n))
    cc = _load_script("compare_cells")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(tmp / "V" / "full"), str(tmp / "H" / "full"), "--checker", "recorded"])
    text = out.getvalue()
    block = text[text.find("customers offered the voucher"):]
    rows = {line.split("  ")[1].strip(): line.split()[-2:] for line in block.splitlines()[1:4]}
    check("compare_cells: each cell's episodes she declined it in, those failed on the "
          "voucher alone, and the trap's other failures",
          rows == {"episodes she declined it in": ["0/5", "5/5"],
                   "failed on the voucher alone": ["0", "5"],
                   "trap failures without it": ["0", "0"]}
          and "P6.3 and P10 ask for the escalation the check fails" in block, block[:700])
    check("…and the paired line again, read as the policy reads those escalations, beside "
          "the result and marked as after the fact",
          "paired by task (5 shared): -1.000; 0 tasks better, 5 worse" in text
          and "read as P6.3 and P10 read them (after the fact, WHAT_FAILED #46): +0.000; 0 "
          "tasks better, 0 worse, p=1.000" in text, text[:1500])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cc.main([str(tmp / "V" / "full"), str(tmp / "V" / "full"), "--checker", "recorded"])
    check("…neither where she never declined it",
          "read as P6.3 and P10" not in out.getvalue()
          and "P6.3 and P10 ask for" not in out.getvalue(), out.getvalue()[-700:])


def test_read_failures():
    """The reading fixed before round 2 and GRPO: every episode a run fails on
    a task its control always passed, read for another trap's answer (#31,
    #44). read_failures.py sets out what that needs -- per trap, the writes no
    control episode on those tasks made; per failure, the writes, the failure
    lines and the conversation -- and flags a customer who claimed photos
    (#43)."""
    print("\n=== read_failures.py sets out the failures to read ===")
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    from pasarbench.run import SOLUTIONS
    verify, check_win, voucher = SOLUTIONS["T05"]
    escalate = ("escalate_to_human", {"order_id": "O1004", "category": "out_of_window_dispute",
                                      "reason": "customer wants a full refund"})
    offer = "I can offer a goodwill voucher; or I can pass this to a human agent."
    tmp = Path(tempfile.mkdtemp())
    control = _language_cell(tmp, "B", "T05", [[verify, check_win, offer, voucher, "Done."]] * 2,
                             replies=("Yes please.", "Thanks."))
    treated = _language_cell(tmp, "F", "T05", [[verify, check_win, offer, voucher, "Done."],
                                               [verify, check_win, offer, escalate, "Escalated."]],
                             replies=("A human, please.", "Thanks."))
    look = ("get_order", {"order_id": "O1005"})
    ask = "Please send photos of the torn strap first; I cannot open the return without them."
    opened = ("initiate_return", {"order_id": "O1005", "order_item_id": "OI5",
                                  "reason": "torn strap", "photo_evidence_provided": True})
    _language_cell(tmp, "B", "T07", [[look, ask]] * 2,
                   replies=("I can't send photos now, I'll send them tonight.",))
    _language_cell(tmp, "F", "T07", [[look, ask, opened, "Your return is open."], [look, ask]],
                   replies=("(Attached: photo_1.jpg) Here are the photos.",))
    rf = _load_script("read_failures")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rf.main([str(control), str(treated), "--width", "200"])
    text = out.getvalue()
    per = text[text.find("per trap"):text.find("=" * 20)]
    check("it finds the two failures on tasks the control always passed, and no pass",
          "F/full: 2 failure(s) on a task the control passed every time in B/full" in text
          and text.count("control 2/2") == 2 and "T05__r0  [" not in text, text[:900])
    check("…per trap, the write the control never made: an escalation, by its category, "
          "on the voucher trap",
          "escalate_to_human[out_of_window_dispute] 1" in per
          and "out_of_window_offer_voucher" in per, per)
    check("…each failure's writes with what they were for, its failure lines, and the "
          "conversation",
          "writes: escalate_to_human(category=out_of_window_dispute, order_id=O1004, reason=customer "
          "wants a full refund)" in text
          and "failed: " in text and "C: A human, please." in text
          and "A: I can offer a goodwill voucher" in text, text[-2500:])
    check("…and a customer who said she sent the photos she does not have is flagged, "
          "with what that makes of the verdict",
          "T07__r0" in text and "CUSTOMER CLAIMED PHOTOS" in text
          and "1 of them after the customer said she sent photos" in text, text)
    check("…as is an escalation after the voucher offer and her answer, to be read as "
          "P6.3 and P10 read it (WHAT_FAILED #46)",
          "T05__r1  [out_of_window_offer_voucher, en]  control 2/2  stop=done  ESCALATED "
          "AFTER THE VOUCHER OFFER" in text
          and "1 of them escalated after the customer was offered the voucher" in text, text)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rf.main([str(control), str(treated), "--trap", "high_value_photo_required_first",
                 "--no-conversation"])
    one = out.getvalue()
    check("--trap reads one trap; --no-conversation keeps the writes and failure lines",
          "1 failure(s)" in one and "T05__" not in one and "C: " not in one
          and "writes: initiate_return(" in one, one)


def main() -> int:
    test_distractors()
    test_arms_are_comparable()
    test_progressive_disclosure()
    test_search_ranking_is_english_only()
    test_loop_records_exposure()
    test_diagnosis()
    test_scaling_report()
    test_scaling_report_on_itools2()
    test_reduced_arms_are_discoverable()
    test_scaling_report_refuses_a_broken_ceiling()
    test_audit_classifies_search_failures()
    test_audit_uses_todays_verdicts()
    test_lookups_are_read_only()
    test_search_is_confined_to_its_arm()
    test_tool_naming_experiment()
    test_naming_confirmation_tasks()
    test_claim_guardrail_report()
    test_closing_report()
    test_closing_compared()
    test_closing_guard_report()
    test_closing_report_one_cell()
    test_closing_readers()
    test_answers_and_customers()
    test_language_lead()
    test_closing_room_compared()
    test_claim_reader_check()
    test_language_swap()
    test_photo_claims_compared()
    test_voucher_choice_compared()
    test_read_failures()
    test_add_model()
    test_add_model_check()
    test_export_hf()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
