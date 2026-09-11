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

    check("oracle exposes only what the task needs", len(t_orc) <= 8, str(len(t_orc)))
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
    for spec, lo, hi in (("oracle", 1, 10), ("all-20", 18, 22), ("all-300", 290, 310)):
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
    check("report carries the causal caveat", "CO-MOVEMENTS" in md)


def test_scaling_report():
    print("\n=== tool scaling reads the contrasts, not the trend ===")
    rows = [
        {"exposure": "oracle", "n_tools": 6, "mean_schema_tokens": 500,
         "mean_tokens": 8000, "pass^1": 0.82},
        {"exposure": "all-20", "n_tools": 20, "mean_schema_tokens": 1800,
         "mean_tokens": 9200, "pass^1": 0.80},
        {"exposure": "all-100", "n_tools": 100, "mean_schema_tokens": 9600,
         "mean_tokens": 17000, "pass^1": 0.66},
        # selection-dominated: guaranteeing reachability does NOT recover it
        {"exposure": "random-100", "n_tools": 100, "mean_schema_tokens": 9700,
         "mean_tokens": 17100, "pass^1": 0.68},
        {"exposure": "search-100", "n_tools": 100, "mean_schema_tokens": 600,
         "mean_tokens": 9800, "pass^1": 0.77},
    ]
    md = tool_scaling_report(rows)
    check("selection vs token cost is adjudicated",
          "SELECTION difficulty" in md, md)
    check("the search arm's recovery is quantified", "search-100` recovers" in md)
    check("schema token saving is reported", "fewer schema tokens" in md)

    rows2 = list(rows)
    rows2[3] = {"exposure": "random-100", "n_tools": 100, "mean_schema_tokens": 9700,
                "mean_tokens": 17100, "pass^1": 0.81}   # recovery -> token cost
    check("the opposite verdict is reached when the data says so",
          "TOKEN COST" in tool_scaling_report(rows2))
    print("\n" + "\n".join(tool_scaling_report(rows).splitlines()[-4:]))


def main() -> int:
    test_distractors()
    test_arms_are_comparable()
    test_progressive_disclosure()
    test_search_ranking_is_english_only()
    test_loop_records_exposure()
    test_diagnosis()
    test_scaling_report()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
