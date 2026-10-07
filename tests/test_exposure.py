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
    line = next((l for l in text.splitlines() if l.strip().startswith("episodes with one")), "")
    check("the claim reached the customer in every control episode and in no guarded one",
          re.search(r"2/2\s+0/2", line) is not None, line)
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
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
