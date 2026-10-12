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
from pathlib import Path

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


def _log(task, script):
    db = Database.fresh(task.db_patch)
    for name, args in script:
        call(db, name, args)
    return db.action_log


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
    # behaviour there. Everything else must not be. The photo trap joined them
    # in v19: its right answer is to ask for photos and open nothing, and since
    # the check stopped naming one lookup tool (WHAT_FAILED #30), an agent that
    # verifies and reads the order is doing what the policy asks of it.
    benign = {"cannot_cancel_shipped_order", "address_change_after_dispatch",
              "peak_period_delay_not_compensable", "high_value_photo_required_first"}
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


def timeline_problems(db: Database) -> list[str]:
    """Dates that contradict each other in one world. Order: the live a
    livestream order came from, the order, its payment (at the door for cash
    on delivery), dispatch, delivery or the last scan, a refund after the
    delivery it refunds -- and all of it before NOW."""
    t, out = db.tables, []
    for oid, o in t["orders"].items():
        placed = ts(o["created"])
        if placed > NOW:
            out.append(f"{oid}: placed after NOW")
        ship = next((s for s in t["shipments"].values() if s["order_id"] == oid), None)
        got = ts(ship["delivered"]) if ship and ship.get("delivered") else None
        for p in (p for p in t["payments"].values() if p["order_id"] == oid and p["paid"]):
            if ts(p["paid"]) < placed or ts(p["paid"]) > NOW:
                out.append(f"{oid}: paid {p['paid']}, placed {o['created']}")
            if p["method"] == "cod" and (got is None or ts(p["paid"]) < got):
                out.append(f"{oid}: cash on delivery collected {p['paid']} before delivery")
        if ship:
            when = [ts(ship["shipped"])] + [ts(ship[k]) for k in ("delivered", "last_scan")
                                             if ship.get(k)]
            if when[0] < placed or any(w < when[0] for w in when[1:]) or max(when) > NOW:
                out.append(f"{oid}: placed {o['created']}, shipped {ship['shipped']}, "
                           f"delivered {ship.get('delivered')}, last scan {ship.get('last_scan')}")
        for c in t["livestream_claims"].values():
            if o.get("livestream_id") and c["livestream_id"] == o["livestream_id"] \
                    and ts(c["timestamp"]) > placed:
                out.append(f"{oid}: bought on a live at {o['created']}, claim made {c['timestamp']}")
        for r in (r for r in t["refunds"].values() if r["order_id"] == oid):
            if ts(r["created"]) < (got or placed) or ts(r["created"]) > NOW:
                out.append(f"{oid}: refunded {r['created']}, delivered "
                           f"{ship.get('delivered') if ship else None}")
    return out


def test_timeline():
    """#32: an out-of-window order was delivered three weeks before it was
    placed, and an agent spent nine calls trying to make the dates agree.
    Every world, hand-written and generated, must tell one consistent story."""
    print("\n=== every world's dates are in causal order ===")
    from pasarbench.tasks import TASKS as CORE
    bad = {}
    for task in list(CORE) + TASKS:
        for p in timeline_problems(Database.fresh(task.db_patch)):
            bad.setdefault(p, task.task_id)
    check(f"no date contradicts another in {len(CORE) + len(TASKS)} worlds", not bad,
          "; ".join(f"{tid}: {p}" for p, tid in list(bad.items())[:4]))
    broken = Database.fresh(TASKS[0].db_patch)
    oid = TASKS[0].hidden_facts["order_id"]
    next(s for s in broken.t("shipments").values() if s["order_id"] == oid)["shipped"] = \
        "2026-11-10 09:00"
    check("…and the check sees a delivery before its dispatch",
          any(oid in p for p in timeline_problems(broken)))


def test_worlds_are_deterministic():
    """#33: tracking numbers came from Python's salted str hash, so every
    process built a different world -- and a replay in another process showed
    a judge a number the agent never saw. Two processes with different hash
    seeds must build byte-identical worlds."""
    print("\n=== the suite is the same in every process ===")
    import hashlib
    import json
    import os
    import subprocess
    import sys
    code = ("import hashlib, json; from pasarbench.generate import generate; "
            "t, _ = generate(); print(hashlib.sha1(json.dumps([x.db_patch for x in t], "
            "sort_keys=True).encode()).hexdigest())")
    root = Path(__file__).resolve().parent.parent
    runs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                           text=True, env={**os.environ, "PYTHONHASHSEED": s}).stdout.strip()
            for s in ("1", "2", "random")}
    here = hashlib.sha1(json.dumps([x.db_patch for x in TASKS], sort_keys=True)
                        .encode()).hexdigest()
    check("three hash seeds, one world", runs == {here}, str(runs | {here}))


def test_read_alternatives():
    """#30: the peak and photo checks named one lookup tool. An agent that
    established the fact another way and did what the policy says must pass;
    one that never established it, or acted, must still fail."""
    print("\n=== peak and photo: any read that shows the fact ===")
    from pasarbench.tasks import BY_ID
    peak = next(t for t in TASKS if t.trap == "peak_period_delay_not_compensable")
    photo = next(t for t in TASKS if t.trap == "high_value_photo_required_first")
    oid, uid = peak.hidden_facts["order_id"], peak.user_id
    check("peak: the date from list_user_orders passes",
          run_script(peak, [("list_user_orders", {"user_id": uid})]).passed)
    check("peak: a shipment lookup does not show the date, and fails",
          not run_script(peak, [("get_shipment", {"order_id": oid})]).passed)
    check("peak: a voucher still fails however the date was read",
          not run_script(peak, [("list_user_orders", {"user_id": uid}),
                                ("verify_identity", {"user_id": uid,
                                                     "phone_last4": peak.hidden_facts["phone_last4"]}),
                                ("issue_goodwill_voucher", {"user_id": uid, "amount_minor": 1,
                                                            "currency": peak.db_patch["orders"][oid]["currency"],
                                                            "reason": "delay"})]).passed)
    oid, uid = photo.hidden_facts["order_id"], photo.user_id
    ver = ("verify_identity", {"user_id": uid, "phone_last4": photo.hidden_facts["phone_last4"]})
    check("photo: verify and read the value with get_order passes",
          run_script(photo, [ver, ("get_order", {"order_id": oid})]).passed)
    check("photo: verification alone establishes nothing, and fails",
          not run_script(photo, [ver]).passed)
    item = next(iter(photo.db_patch["order_items"]))
    check("photo: opening the return before photos still fails",
          not run_script(photo, [ver, ("check_return_eligibility",
                                       {"order_id": oid, "order_item_id": item}),
                                 ("initiate_return", {"order_id": oid, "order_item_id": item,
                                                      "reason": "defect",
                                                      "photo_evidence_provided": False})]).passed)
    # The review of v19: alternatives matched on the tool's NAME, so a read of
    # someone else's order counted, and a verification that failed counted as
    # one. Each is a way to pass without the fact.
    other_user = next(u for u in ("U001", "U002", "U003") if u != peak.user_id)
    other_order = "O1001"
    poid, puid = peak.hidden_facts["order_id"], peak.user_id
    check("peak: another customer's orders do not show this one's date, and fail",
          not run_script(peak, [("list_user_orders", {"user_id": other_user})]).passed)
    check("peak: a listing filtered to another status leaves the order out, and fails",
          not run_script(peak, [("list_user_orders", {"user_id": puid,
                                                      "status": "delivered"})]).passed)
    check("peak: …filtered to the order's own status, it shows the date, and passes",
          run_script(peak, [("list_user_orders", {"user_id": puid,
                                                  "status": "processing"})]).passed)
    check("peak: …and an empty status, which the tool ignores, lists it too",
          run_script(peak, [("list_user_orders", {"user_id": puid, "status": ""})]).passed)
    check("peak: get_order on another order fails",
          not run_script(peak, [("get_order", {"order_id": other_order})]).passed)
    r = run_script(peak, [("get_order", {"order_id": poid}),
                          ("verify_identity", {"user_id": puid,
                                               "phone_last4": peak.hidden_facts["phone_last4"]}),
                          ("issue_store_credit", {"user_id": puid, "amount_minor": 100,
                                                  "currency": peak.db_patch["orders"][poid]["currency"],
                                                  "reason": "delay"})])
    check("peak: store credit is compensation too, and fails",
          not r.passed and all("store_credit" in f for f in r.failures), str(r.failures))
    check("photo: get_order on another order does not show this item's value, and fails",
          not run_script(photo, [ver, ("get_order", {"order_id": other_order})]).passed)
    check("photo: get_product on another product fails",
          not run_script(photo, [ver, ("get_product", {"product_id": "P001"})]).passed)
    check("photo: the order total from list_user_orders is not the item's value, and fails",
          not run_script(photo, [ver, ("list_user_orders", {"user_id": uid})]).passed)
    check("photo: get_order_items on the order passes",
          run_script(photo, [ver, ("get_order_items", {"order_id": oid})]).passed)
    wrong = "0000" if photo.hidden_facts["phone_last4"] != "0000" else "1111"
    check("photo: no verification is required -- P1.1 asks for it before a write, "
          "and the right answer writes nothing",
          run_script(photo, [("get_order", {"order_id": oid})]).passed)
    import dataclasses
    from pasarbench.verifier import ActionSpec, TaskCheck
    needs = dataclasses.replace(photo, checks=TaskCheck(
        required_actions=[ActionSpec("verify_identity", {"user_id": uid})]))
    check("a verification with the wrong digits is an attempt, not a verification",
          not run_script(needs, [("verify_identity", {"user_id": uid,
                                                      "phone_last4": wrong})]).passed
          and run_script(needs, [ver]).passed)
    cur = photo.db_patch["orders"][oid]["currency"]
    r = run_script(photo, [ver, ("get_order", {"order_id": oid}),
                           ("issue_goodwill_voucher", {"user_id": uid, "amount_minor": 100,
                                                       "currency": cur, "reason": "sorry"})])
    check("photo: a goodwill voucher in place of the evidence fails",
          not r.passed and all("voucher" in f for f in r.failures), str(r.failures))
    r = run_script(photo, [ver, ("get_order", {"order_id": oid}),
                           ("issue_store_credit", {"user_id": uid, "amount_minor": 100,
                                                   "currency": cur, "reason": "sorry"})])
    check("photo: store credit in place of the evidence fails",
          not r.passed and all("store_credit" in f for f in r.failures), str(r.failures))
    ident = next(t for t in TASKS if t.trap == "identity_verification_failure")
    check("identity trap: the failed attempt is what it requires, and passes",
          run_script(ident, SOLUTIONS[ident.task_id]).passed
          and all(a.denied for a in _log(ident, SOLUTIONS[ident.task_id])
                  if a.tool == "verify_identity"))
    shipped = next(t for t in TASKS if t.trap == "cannot_cancel_shipped_order")
    check("cancel-shipped: a shipment lookup of another order fails",
          not run_script(shipped, [("get_shipment", {"order_id": other_order})]).passed)
    addr = next(t for t in TASKS if t.trap == "address_change_after_dispatch")
    check("address: get_order on another order fails",
          not run_script(addr, [("get_order", {"order_id": other_order})]).passed)
    check("T07 and T11 accept the same tools as their generated twins",
          [a.tools for a in BY_ID["T07"].checks.required_actions]
          == [a.tools for a in photo.checks.required_actions]
          and BY_ID["T11"].checks.required_actions[0].tools == peak.checks.required_actions[0].tools)


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


def test_leak_detector_reads_every_language():
    """#26: the customer's answers were scored as leaks wherever the agent asked
    in words the patterns lacked. Every phrasing below is taken from a real
    request the first pattern list missed; each must now read as a request, and
    the answer that follows it as correct behaviour."""
    print("\n=== simulator QA: requests the first pattern list could not read ===")
    from pasarbench.harness.types import Message
    from pasarbench.simqa import _asked_for, leak_report

    observed = [
        ("id", "phone_last4", "Bisa sebutkan **4 digit terakhir nomor telepon** yang terdaftar di akun Anda?"),
        ("id", "order_id", "Bisa beri tahu nomor order Anda?"),
        ("ms", "order_id", "Boleh berikan nombor order yang berkaitan dengan pengecas tersebut?"),
        ("th", "order_id", "รบกวนช่วยแจ้ง **เลขออเดอร์** ให้หน่อยได้ไหมคะ"),
        ("zh", "order_id", "请提供您的订单号，我帮您查询。"),
        ("zh", "phone_last4", "为了核实身份，请提供您注册手机号的后四位。"),
    ]
    # ...and the ones Qwen3.8-27B used in its first run, which the list after
    # #26 still missed: "ID order", a misspelt "nomor pesannya", and Thai with
    # its tone marks and vowels dropped or misplaced, copied as it wrote them.
    observed += [
        ("id", "order_id", "Supaya saya bisa cek ordernya, boleh tolong sebutkan:\n1. **ID order** blusnya"),
        ("id", "order_id", "Boleh saya tahu nomor pesannya?"),
        ("th", "phone_last4", "กรุณาบอ**ก 4 ตัวเลขทายสุดของหมายเลขโทรศัพท**ท่ีใช้อยูในบัญชีค่ะ"),
        ("th", "phone_last4", "รบกวนบอกลำดับตัวเลข 4 ตัวส้สุดของหมายเลขโทรศัพท่ท่ีใช้อยู่ในบัญชีค้ะ"),
        ("th", "phone_last4", "รบกวนแจ้ง 4 ตัวเลขสุดท้ายของหมายเลขโทรศัพท์ท่ีใช้สมัครบัญชี"),
        ("zh", "order_id", "当然可以。请告诉我你要修改哪个订单，以及新的送货地址。"),
    ]
    for lang, fact, text in observed:
        check(f"{lang}: {fact} request is recognised", _asked_for(fact, text), text)
    check("a line that asks for nothing is still not a request",
          not _asked_for("order_id", "Terima kasih, ada lagi yang bisa saya bantu?"))
    check("…nor 'two shirts' in Thai, nor a paid or valid order in English",
          not _asked_for("phone_last4", "ได้รับเสื้อ 2 ตัวแล้วค่ะ")
          and not _asked_for("order_id", "Your paid order has shipped, a valid order."))
    # Folding the marks out merges words: หลัก (digit) with หลักฐาน (evidence),
    # หลักเกณฑ์ (criteria) and หลีกเลี่ยง (avoid); ท้าย (last) with ทายาท (heir).
    # None of these asks for anything.
    for text in ("กรุณาส่ง 4 หลักฐานประกอบการคืนสินค้า",
                 "ตามข้อ 4 หลักเกณฑ์การคืนสินค้า",
                 "ข้อ 4 หลีกเลี่ยงการส่งคืนสินค้าที่เสียหาย",
                 "บัญชีนี้โอนให้ตัวทายาทไม่ได้ค่ะ"):
        check(f"th: not a request for digits: {text}", not _asked_for("phone_last4", text))
    for text in ("ขอ 4 หลักสุดท้ายค่ะ", "ขอ ๔ หลักท้ายค่ะ", "แจ้งสี่หลักสุดท้ายด้วยนะคะ",
                 "ขอตัวเลขท้ายสี่ตัวค่ะ"):
        check(f"th: a request for digits: {text}", _asked_for("phone_last4", text))
    check("case is matched without lowercasing the pattern",
          _asked_for("order_id", "What is your ORDER NUMBER?"))
    from pasarbench.simqa import ASK_PATTERNS_V1
    check("the first list is still matched as it was (#26 reproduces)",
          not _asked_for("phone_last4", "หมายเลขโทรศัพท", ASK_PATTERNS_V1)
          and _asked_for("phone_last4", "เบอร์โทร", ASK_PATTERNS_V1))
    check("…unfolded: a dropped mark defeats the first list and not today's",
          not _asked_for("phone_last4", "เบอรโทร", ASK_PATTERNS_V1)
          and _asked_for("phone_last4", "เบอรโทร"))

    task = next(t for t in TASKS if t.trap == "happy_path_return_refund"
                and t.language == "en")
    oid = task.hidden_facts["order_id"]
    answered = [Message("system", "..."), Message("user", "ขอคืนสินค้าค่ะ"),
                Message("assistant", "รบกวนช่วยแจ้ง เลขออเดอร์ ให้หน่อยได้ไหมคะ"),
                Message("user", f"{oid} ค่ะ")]
    r = leak_report(task, answered)
    check("a Thai answer to a Thai request is on request, not a leak",
          not r.leaked and r.revealed_on_request, str(r.to_dict()))
    volunteered = [Message("system", "..."), Message("user", "ขอคืนสินค้าค่ะ"),
                   Message("assistant", "ยินดีช่วยค่ะ"), Message("user", f"{oid} ค่ะ")]
    v = leak_report(task, volunteered)
    check("…and the same fact with no request before it is still a leak",
          v.leaked and v.leak_context and v.leak_context[0][2] == "ยินดีช่วยค่ะ",
          str(v.to_dict()))

    from pasarbench.simqa import stall_report
    stalled = [Message("system", "..."), Message("user", "我要退货"),
               Message("assistant", "请提供您的订单号。"), Message("user", "我不记得了"),
               Message("assistant", "麻烦您再查一下订单号？"), Message("user", f"是 {oid}")]
    check("a request the next turn does not answer counts as a stall",
          stall_report(task, stalled) == (2, 1), str(stall_report(task, stalled)))
    check("…and a fact already given is not asked for again",
          stall_report(task, stalled + [Message("assistant", "订单号是多少？"),
                                        Message("user", "刚说了")]) == (2, 1))


def test_leak_rates_are_per_episode():
    """inspect_trace looked each episode's report up by task id, so every seed
    of a task inherited seed 0's verdict."""
    print("\n=== simulator QA: one report per episode, not per task ===")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "inspect_trace", Path(__file__).resolve().parent.parent / "scripts" / "inspect_trace.py")
    it = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(it)

    task = next(t for t in TASKS if t.trap == "happy_path_return_refund"
                and t.language == "en")
    oid = task.hidden_facts["order_id"]

    def ep(seed, agent_line):
        return {"id": f"{task.task_id}__r{seed}", "task_id": task.task_id, "language": "en",
                "steps": [{"model_content": agent_line, "tool_calls": []}],
                "events": [{"kind": "user_turn", "turn": 1, "text": f"It's {oid}"}]}

    pairs = it.leak_reports([ep(0, "Sure, one moment."), ep(1, "What is your order number?")],
                            {task.task_id: task})
    check("two seeds of one task get their own verdicts",
          [bool(r.leaked) for _, r in pairs] == [True, False],
          str([(e["id"], r.leaked) for e, r in pairs]))

    print("\n=== simulator QA: an audit must read the task that produced the trace ===")
    import json as _json
    import tempfile
    from pasarbench.harness.trace import TraceWriter
    from pasarbench.tasks import task_digest
    tmp = tempfile.mkdtemp()
    tw = TraceWriter(tmp, run_id="r")
    run_episode(task, Database.fresh(task.db_patch), ScriptedBackend(SOLUTIONS[task.task_id]),
                trace=tw)
    tw.close()
    head = _json.loads(next(Path(tw.dir).glob("*.jsonl")).read_text().splitlines()[0])
    check("every trace header records the digest of its task",
          head.get("task_digest") == task_digest(task), str(head))

    tasks = {task.task_id: task}
    fresh = dict(ep(0, "What is your order number?"), head={"task_digest": task_digest(task)})
    moved = dict(ep(1, "What is your order number?"), head={"task_digest": "000000000000"})
    kept = it.same_tasks([fresh, moved], tasks)
    check("a trace whose task has changed since the run is left out",
          [e["id"] for e in kept] == [fresh["id"]], str([e["id"] for e in kept]))
    old = dict(ep(2, "What is your order number?"), head={})
    old["events"] = [{"kind": "user_turn", "turn": 1, "text": "GO-addresID"}]
    check("old traces where the customer never says today's order id are refused",
          it.same_tasks([old], tasks) == [])
    check("…and old traces that do say it are kept",
          len(it.same_tasks([dict(ep(3, "Order number?"), head={})], tasks)) == 1)


def test_photo_claims():
    """WHAT_FAILED #43: the photo trap's customer is told she has no photos and
    will send them tonight, and customer v2 -- who cannot leave until the agent
    has done what she needs -- sends them inside the conversation. Every line
    below is one she wrote in round 2, in the eight languages; each must read
    as a claim, and each of her promises must not."""
    print("\n=== simulator QA: photos a customer says she sent but has not ===")
    import contextlib
    import importlib.util
    import io

    from pasarbench.simqa import claims_photos, forbids_photos, photo_claim
    from pasarbench.tasks import TASKS as HAND

    said = [
        "[Attached: photo_1.jpg, photo_2.jpg]\n\nHere are the photos.",
        "(Attached photos) Here are the pictures.",
        "I've uploaded the photos now. Can you check them and start the return?",
        "(Sends photos of the damaged blouse) Here are the photos. Please process my refund now.",
        "(I have sent the photos)",
        "I've sent the photos. The phone digits are 3751.",
        "(Mengirim foto barang rusak) Ini fotonya. Tolong segera diproses ya.",
        "(Hantar gambar barang yang pecah)\n\nDah hantar. Tolong proses refund saya ya.",
        "(Saya sudah hantar gambar kerosakan blouse itu melalui lampiran. Sila semak.)",
        "(Attached photos)\nOkay I send already. Please check and refund me.",
        "(ส่งรูปภาพสินค้าที่ชำรุด) เรียบร้อยค่ะ รบกวนดำเนินการคืนเงินให้ด้วยนะคะ",
        "ส่งรูปให้แล้วนะคะ รบกวนเช็คด้วยค่ะว่าใช้ได้ไหม",
        "(Gửi ảnh) Đây nhé, bạn xem đường rách này. Giờ làm thủ tục trả hàng thế nào?",
        "(Gửi ảnh sản phẩm bị hỏng) Đây nè bạn, bạn xem giúp mình với.",
        "(发送了照片) 好了，照片发给你了，帮我处理退款吧。",
        "（发送照片）好了，照片发给你了。帮我处理退款吧。",
        "(假设我现在是今晚，发送了照片)\n照片发给你了，你看一下。",
        "(今晚发送照片后) 给你照片了，你看一下。",
    ]
    promised = [
        "The order number is GO-680346. I can't send the photos right now, but I'll send "
        "them tonight.",
        "It arrived torn. I'll get those photos to you later.",
        "(Wait for the agent to process the photos tonight)",
        "Kalau fotonya nanti malam ya saya kirim, sekarang belum bisa. Untuk nomor HP-nya 6337.",
        "Sekarang saya tak boleh hantar gambar, nanti malam baru saya hantar.",
        "cannot send photo now, will send tonight. phone is 3751",
        "โอเคค่ะ เดี๋ยวคืนนี้ส่งรูปให้นะคะ ขอบคุณค่ะ",
        "ตอนนี้ยังส่งรูปให้ไม่ได้ค่ะ เดี๋ยวคืนนี้ส่งให้นะคะ",
        "Ok, vậy tối nay mình gửi ảnh rồi nhắn lại cho bạn nhé.",
        "照片我现在没法发，我今晚发给你。",
        "没有了，等我今晚发照片给你。",
    ]
    for line in said:
        check(f"she says she sent them: {' '.join(line.split())[:48]}", claims_photos(line), line)
    for line in promised:
        check(f"a promise, not a claim: {' '.join(line.split())[:48]}",
              not claims_photos(line), line)

    photo = "high_value_photo_required_first"
    every = list(HAND) + list(TASKS) + list(generate(edition=2)[0])
    check("only the photo trap's customer is told she has none -- every one of hers is",
          {t.trap for t in every if forbids_photos(t)} == {photo}
          and all(forbids_photos(t) for t in every if t.trap == photo),
          str({t.trap for t in every if forbids_photos(t)}))
    task = next(t for t in TASKS if t.trap == photo and t.language == "en")
    other = next(t for t in TASKS if t.trap != photo)
    check("the first turn she says it in, and nothing for a customer allowed photos",
          photo_claim(task, ["I'll send them tonight.", "(Attached: a.jpg) Here are the photos."])
          == 1 and photo_claim(task, ["I'll send them tonight."]) is None
          and photo_claim(other, ["(Attached: a.jpg) Here are the photos."]) is None)

    spec = importlib.util.spec_from_file_location(
        "inspect_trace", Path(__file__).resolve().parent.parent / "scripts" / "inspect_trace.py")
    it = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(it)

    def ep(seed, text, passed):
        return {"id": f"{task.task_id}__r{seed}", "task_id": task.task_id, "language": "en",
                "passed": passed, "steps": [],
                "events": [{"kind": "user_turn", "turn": 1, "text": text}]}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        it.photos([ep(0, "(Attached: photo_1.jpg) Here are the photos.", False),
                   ep(1, "I can't send photos now, I'll send them tonight.", True)])
    text = out.getvalue()
    check("the stage-1 audit counts her by language, with the failures after and without "
          "the line, and says what that does to the trap",
          "en       50.0%  (1/2 episodes)" in text
          and "Failed after it: 1 of 1; failed without it: 0 of 1" in text
          and "WHAT_FAILED #43" in text, text)
    guarded = ep(2, "I can't send photos now, I'll send them tonight.", True)
    guarded["events"][0]["guard"] = "photos: written again"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        it.photos([guarded, ep(3, "", True)])
    text = out.getvalue()
    check("…and with customer 3, what her guard did, beside a trap that then measures the "
          "agent", "customer 3's guard: photos: written again 1" in text
          and "the photo trap measures the agent" in text, text)


def test_second_edition():
    """RUNBOOK 1m confirms the closing check on new tasks of the five traps it
    came from: the same checks, new worlds and stories. Every guarantee the
    suite has, these have; and they are new -- no id, row, story or customer
    of the suite -- while the suite is exactly what it was."""
    print("\n=== the second edition: new tasks of the closing check's five traps ===")
    from pasarbench.generate import EDITION2_TRAPS
    from pasarbench.rescore import all_tasks, suite_tasks
    from pasarbench.tasks import TASKS as CORE
    from pasarbench.tasks import task_digest
    t2, s2 = generate(edition=2)
    check("62 tasks of the five traps, in every market and language their first edition has",
          len(t2) == 62 and {t.trap for t in t2} == set(EDITION2_TRAPS)
          and Counter(t.trap for t in t2) == Counter(t.trap for t in TASKS
                                                     if t.trap in EDITION2_TRAPS),
          str(Counter(t.trap for t in t2)))
    unsolved = [t.task_id for t in t2 if not run_script(t, s2[t.task_id]).passed]
    check("each is solved by its own reference solution", not unsolved, str(unsolved[:5]))
    sample = stratified_sample(t2, per_trap=1)
    looped = []
    for t in sample:
        db = Database.fresh(t.db_patch)
        run_episode(t, db, ScriptedBackend(s2[t.task_id]))
        looped.append(verify(t, db).passed)
    check("…through the real loop too", all(looped) and len(looped) == 5, str(looped))
    spam = lambda t: [("verify_identity", {"user_id": t.user_id, "phone_last4": "0000"}),
                      ("get_order", {"order_id": t.hidden_facts["order_id"]}),
                      ("get_shipment", {"order_id": t.hidden_facts["order_id"]}),
                      ("escalate_to_human", {"order_id": t.hidden_facts["order_id"],
                                             "category": "other", "reason": "x"})]
    lucky = [t.task_id for t in t2 if run_script(t, []).passed or run_script(t, spam(t)).passed]
    check("…and none passes with no action, or with tool spam", not lucky, str(lucky[:5]))
    groups = defaultdict(list)
    for t in t2:
        groups[t.task_id.split(".")[0]].append(t)
    check("locale twins share checks, world, persona and solution, and differ in the opening",
          all(all(x.checks is g[0].checks and x.db_patch == g[0].db_patch
                  and x.persona == g[0].persona and s2[x.task_id] == s2[g[0].task_id]
                  for x in g) and len({x.opening for x in g}) == len(g)
              for g in groups.values()))
    probs = [p for t in t2 for p in timeline_problems(Database.fresh(t.db_patch))]
    check("every world's dates are in causal order", not probs, str(probs[:3]))
    first = {t.task_id: t for t in TASKS}

    def rows(t):
        return {(table, rid) for table, rs in t.db_patch.items() for rid in rs}
    shared = {r for t in t2 for r in rows(t)} & {r for t in TASKS for r in rows(t)}
    twin = {t.task_id: first[t.task_id.replace("2-", "-", 1)] for t in t2}
    check("new: no task id, row id, customer, persona or opening of the suite",
          not ({t.task_id for t in t2} & set(first)) and not shared
          and all(t.persona != twin[t.task_id].persona and t.opening != twin[t.task_id].opening
                  and t.db_patch["users"] != twin[t.task_id].db_patch["users"]
                  for t in t2), str(sorted(shared)[:4]))
    import dataclasses
    import json

    def rules(t):
        """The checks with this world's ids written as roles."""
        text = json.dumps(dataclasses.asdict(t.checks), sort_keys=True, default=str)
        names = {t.hidden_facts["order_id"]: "<order>", t.user_id: "<user>"}
        for it in t.db_patch["order_items"].values():
            names.update({it["order_item_id"]: "<item>", it["product_id"]: "<product>"})
        for o in t.db_patch["orders"].values():
            if o.get("livestream_id"):
                names[o["livestream_id"]] = "<live>"
        for k, v in names.items():
            text = text.replace(k, v)
        return text
    check("…under the same rules: each has its first-edition twin's checks, but its own ids",
          all(t.trap == twin[t.task_id].trap and rules(t) == rules(twin[t.task_id])
              and task_digest(t) != task_digest(twin[t.task_id]) for t in t2))
    from pasarbench.sweep import ALL_TASKS, EDITION2_TASKS
    check("outside the suite: `all` runs the 215 it always has, and the replays and readers "
          "find the new ones -- with the hard edition's, and nothing else",
          len(ALL_TASKS) == 215 == len(suite_tasks()) == len(CORE) + len(TASKS)
          and not {t.task_id for t in EDITION2_TASKS} & {t.task_id for t in ALL_TASKS}
          and set(all_tasks()) == (set(suite_tasks()) | {t.task_id for t in t2}
                                   | {t.task_id for t in generate(edition=3)[0]}))
    from types import SimpleNamespace

    from pasarbench.sweep import select_tasks
    picked = select_tasks(SimpleNamespace(suite="edition2", languages="", tasks="", sample=0,
                                          traps=""))
    check("`--suite edition2` runs the 62, as RUNBOOK 1m's command asks",
          sorted(t.task_id for t in picked) == sorted(t.task_id for t in t2)
          and "--suite edition2" in (Path(__file__).resolve().parent.parent / "docs"
                                     / "RUNBOOK.md").read_text(encoding="utf-8"))
    try:
        generate(edition=2, traps=["happy_path_return_refund"])
        refused = False
    except ValueError:
        refused = True
    check("…and a trap the second edition does not have is refused, not built from the first",
          refused)


# Every definition of the 16 hand-written tasks and of editions 1 and 2 --
# ids, stories, checks, worlds, facts, reference solutions -- as v47 had them
# and every recorded run used them. A change to any of these is a change to
# the suite: make it on purpose, re-pin, and say why in WHAT_FAILED.
SUITE_DIGESTS = {"hand": "e00b2d7331de80dc3d3b9bc0a01d4624ec8fd614",
                 "edition1": "f17085a08627ac19ba5e4833c343c269bbfbbdee",
                 "edition2": "c97eb55be2f0e23fa1ca8b3aed24fa9601eabeaa"}


def suite_digests() -> dict[str, str]:
    import dataclasses
    import hashlib
    import json

    from pasarbench.run import SOLUTIONS as HAND_SOLUTIONS
    from pasarbench.tasks import TASKS as HAND

    def digest(tasks, solutions):
        blob = json.dumps([{**dataclasses.asdict(t), "solution": solutions.get(t.task_id)}
                           for t in tasks], sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()
    return {"hand": digest(HAND, HAND_SOLUTIONS),
            "edition1": digest(*generate()), "edition2": digest(*generate(edition=2))}


def test_hard_edition():
    """RUNBOOK 1s: the hard edition puts two or three of the policy's rules in
    one case, and the rule the customer's story points at is not the one that
    decides it. Every guarantee the suite has, these have; each case's wrong
    turns fail; and the suite and edition 2 are byte for byte what they were."""
    print("\n=== the hard edition: cases that combine traps ===")
    from pasarbench.generate import EDITION3_NEW_ADDRESS, EDITION3_TRAPS
    from pasarbench.locales import LANGUAGES_FOR_MARKET
    from pasarbench.rescore import all_tasks, suite_tasks
    from pasarbench.simqa import forbids_photos
    from pasarbench.tasks import TASKS as HAND
    t3, s3 = generate(edition=3)
    by_trap = Counter(t.trap for t in t3)
    langs = {m: len(LANGUAGES_FOR_MARKET[m]) for m in MARKETS}
    check("36 tasks: the cash-on-delivery case in the five COD markets, the other two in "
          "all six, each in every language its market has",
          len(t3) == 36 and set(by_trap) == set(EDITION3_TRAPS)
          and by_trap["duplicate_refund_cod_livestream"] == sum(langs.values()) - langs["SG"]
          and by_trap["two_orders_voucher_and_cancel"] == sum(langs.values())
          and by_trap["changed_request_cancel_to_address"] == sum(langs.values()), str(by_trap))
    unsolved = [t.task_id for t in t3 if not run_script(t, s3[t.task_id]).passed]
    check("each is solved by its own reference solution", not unsolved, str(unsolved[:5]))
    looped = []
    for t in stratified_sample(t3, per_trap=2):
        db = Database.fresh(t.db_patch)
        run_episode(t, db, ScriptedBackend(s3[t.task_id]))
        looped.append(verify(t, db).passed)
    check("…through the real loop too", all(looped) and len(looped) == 6, str(looped))
    spam = lambda t: [("verify_identity", {"user_id": t.user_id, "phone_last4": "0000"}),
                      ("get_order", {"order_id": t.hidden_facts["order_id"]}),
                      ("get_shipment", {"order_id": t.hidden_facts["order_id"]}),
                      ("get_payment", {"order_id": t.hidden_facts["order_id"]}),
                      ("search_policy", {"query": "refund return"}),
                      ("escalate_to_human", {"order_id": t.hidden_facts["order_id"],
                                             "category": "other", "reason": "x"})]
    lucky = [t.task_id for t in t3 if run_script(t, []).passed or run_script(t, spam(t)).passed]
    check("…and none passes with no action, or with tool spam", not lucky, str(lucky[:5]))
    groups = defaultdict(list)
    for t in t3:
        groups[t.task_id.split(".")[0]].append(t)
    check("locale twins share checks, world, facts, persona and solution, and differ in the "
          "opening", all(all(x.checks is g[0].checks and x.db_patch == g[0].db_patch
                             and x.hidden_facts == g[0].hidden_facts
                             and x.persona == g[0].persona and s3[x.task_id] == s3[g[0].task_id]
                             for x in g) and len({x.opening for x in g}) == len(g)
                         for g in groups.values()))
    probs = [p for t in t3 for p in timeline_problems(Database.fresh(t.db_patch))]
    check("every world's dates are in causal order", not probs, str(probs[:3]))

    def rows(t):
        return {(table, rid) for table, rs in t.db_patch.items() for rid in rs}
    before = list(HAND) + list(TASKS) + list(generate(edition=2)[0])
    shared = {r for t in t3 for r in rows(t)} & {r for t in before for r in rows(t)}
    check("new: no task id, row id, customer, persona or opening of the suite or edition 2",
          not ({t.task_id for t in t3} & {t.task_id for t in before}) and not shared
          and not ({u["name"] for t in t3 for u in t.db_patch["users"].values()}
                   & {u["name"] for t in before for u in t.db_patch.get("users", {}).values()})
          and not ({t.persona for t in t3} & {t.persona for t in before})
          and not ({t.opening for t in t3} & {t.opening for t in before}),
          str(sorted(shared)[:4]))

    # The case each family is built around, and the wrong turns it is there
    # to catch: in every market, the English twin standing for its family.
    def worlds(trap):
        return [t for t in t3 if t.trap == trap and t.language == "en"]
    wrong: list[str] = []
    for t in worlds("duplicate_refund_cod_livestream"):
        o, u = t.hidden_facts["order_id"], t.user_id
        V = ("verify_identity", {"user_id": u, "phone_last4": t.hidden_facts["phone_last4"]})
        order = t.db_patch["orders"][o]
        cur, total = order["currency"], order["total_minor"]
        live = order["livestream_id"]
        esc = ("escalate_to_human", {"order_id": o, "category": "duplicate_refund", "reason": "x"})
        tries = {
            "a livestream refund, then store credit when it is refused":
                [V, ("get_livestream_claims", {"livestream_id": live}), ("get_order", {"order_id": o}),
                 ("issue_refund", {"order_id": o, "amount_minor": total,
                                   "method": "bank_transfer", "reason": "P7.2"}),
                 ("issue_store_credit", {"user_id": u, "amount_minor": total, "currency": cur,
                                         "reason": "P7.2, COD", "order_id": o}), esc],
            "the refund refused, and escalated: an attempt is a write P10 forbids":
                [V, ("get_order", {"order_id": o}),
                 ("issue_refund", {"order_id": o, "amount_minor": total,
                                   "method": "bank_transfer", "reason": "P7.2"}), esc],
            "the existing refund explained and let go": [("get_order", {"order_id": o})],
            "escalated as a dispute": [("get_order", {"order_id": o}),
                                       ("escalate_to_human", {"order_id": o, "category":
                                                              "out_of_window_dispute",
                                                              "reason": "x"})],
        }
        wrong += [f"{t.task_id}: {k}" for k, s in tries.items() if run_script(t, s).passed]
        if not run_script(t, [("get_order", {"order_id": o}), esc,
                              ("send_message_to_seller", {"order_id": o, "message": "x"})]).passed:
            wrong.append(f"{t.task_id}: a note to the seller beside the escalation failed")
    for t in worlds("two_orders_voucher_and_cancel"):
        late, fresh, u = t.hidden_facts["order_id"], t.hidden_facts["second_order_id"], t.user_id
        V = ("verify_identity", {"user_id": u, "phone_last4": t.hidden_facts["phone_last4"]})
        item = next(i["order_item_id"] for i in t.db_patch["order_items"].values()
                    if i["order_id"] == late)
        cur = t.db_patch["orders"][late]["currency"]
        vou = ("issue_goodwill_voucher", {"user_id": u, "amount_minor": 100, "currency": cur,
                                          "reason": "x"})
        can = ("cancel_order", {"order_id": fresh, "reason": "x"})
        refund = lambda oid: ("issue_refund", {"order_id": oid, "amount_minor": 100,
                                               "method": "card", "reason": "x"})
        tries = {
            "the voucher only": [V, vou],
            "the cancellation only": [V, can],
            "a return of the late order": [
                V, ("check_return_eligibility", {"order_id": late, "order_item_id": item}),
                ("initiate_return", {"order_id": late, "order_item_id": item, "reason": "x",
                                     "photo_evidence_provided": False}), vou, can],
            "an attempt to cancel the delivered order": [
                V, ("cancel_order", {"order_id": late, "reason": "x"}), vou, can],
            "a refund of the late order in place of the voucher": [V, refund(late), can],
            "a refund of the processing order in place of the cancellation": [V, vou, refund(fresh)],
            "a refund on top of the cancellation's": [V, vou, can, refund(fresh)],
        }
        wrong += [f"{t.task_id}: {k}" for k, s in tries.items() if run_script(t, s).passed]
        if not run_script(t, [V, can, vou]).passed:
            wrong.append(f"{t.task_id}: both done, the other way round, failed")
    for t in worlds("changed_request_cancel_to_address"):
        o, u = t.hidden_facts["order_id"], t.user_id
        V = ("verify_identity", {"user_id": u, "phone_last4": t.hidden_facts["phone_last4"]})
        address, postcode = EDITION3_NEW_ADDRESS[t.market]
        move = lambda a: ("modify_shipping_address", {"order_id": o, "new_address": a})
        can = ("cancel_order", {"order_id": o, "reason": "x"})
        tries = {
            "the cancellation she withdrew": [V, can],
            "the address changed, then the order cancelled": [V, move(address), can],
            "the new address without its postcode": [V, move(address.replace(postcode, "").strip(" ,"))],
            "an address she never gave": [V, move(t.db_patch["orders"][o]["shipping_address"] + " (new)")],
            "an attempt to cancel before her digits, then the change": [can, V, move(address)],
            "the change without verifying": [move(address)],
        }
        wrong += [f"{t.task_id}: {k}" for k, s in tries.items() if run_script(t, s).passed]
        if not run_script(t, [V, move(f"{postcode} -- " + address.replace(postcode, "#")
                                      .upper())]).passed:
            wrong.append(f"{t.task_id}: her address in the agent's own words, with its "
                         f"postcode, failed")
    check("each case's wrong turns fail, in every market -- a refund or credit on the "
          "refunded order, one order's remedy on the other, the cancellation she withdrew, "
          "an address without its postcode -- and the right one in other words passes",
          not wrong, "; ".join(wrong[:4]))
    import json as _json

    from pasarbench.verifier import ActionSpec, Contains, DBAssert
    db = Database.fresh({"orders": {"X1": {"order_id": "X1", "shipping_address": "1 Rd, 47300"}}})
    spec = ActionSpec("modify_shipping_address", {"new_address": Contains("47300")})
    check("a Contains value matches the strings that hold its part, in an action's arguments "
          "and in a row, and nothing else -- not None, not a number, not a string without it",
          spec.via("modify_shipping_address", {"new_address": "22 SS 2/24, 47300 PJ"})
          and not spec.via("modify_shipping_address", {"new_address": "22 SS 2/24, PJ"})
          and not spec.via("modify_shipping_address", {})
          and Contains("47300") != 47300 and Contains("1") == Contains("1")
          and DBAssert("orders", row_id="X1", fields={"shipping_address": Contains("47300")})
          .check(db)[0]
          and not DBAssert("orders", row_id="X1", fields={"shipping_address": Contains("55100")})
          .check(db)[0]
          and DBAssert("orders", {"shipping_address": Contains("47300")}, count=1).check(db)[0]
          and _json.dumps({"a": Contains("47300")}, default=str) == '{"a": "Contains(\'47300\')"}')
    check("her new address is not her old one, and its postcode is in no address she had",
          all(postcode not in MARKETS[m]["addr"] and address != MARKETS[m]["addr"]
              and postcode in address for m, (address, postcode) in EDITION3_NEW_ADDRESS.items()))
    check("she is never told she has no photos: the simulator's photo guard has nothing "
          "to do here", not any(forbids_photos(t) for t in t3))

    now = suite_digests()
    check("the suite and edition 2 are byte for byte what every recorded run had: the "
          "hand-written 16, edition 1's 199 and edition 2's 62, by ids, stories, checks, "
          "worlds, facts and reference solutions", now == SUITE_DIGESTS,
          str({k: v for k, v in now.items() if SUITE_DIGESTS.get(k) != v}))
    from pasarbench.sweep import ALL_TASKS, EDITION3_TASKS, select_tasks
    from types import SimpleNamespace
    picked = select_tasks(SimpleNamespace(suite="edition3", languages="", tasks="", sample=0,
                                          traps=""))
    check("outside the suite: `all` runs the 215 it always has, `--suite edition3` the 36, "
          "and the replays and readers find them",
          len(ALL_TASKS) == 215 == len(suite_tasks())
          and not {t.task_id for t in EDITION3_TASKS} & {t.task_id for t in ALL_TASKS}
          and sorted(t.task_id for t in picked) == sorted(t.task_id for t in t3)
          and {t.task_id for t in t3} <= set(all_tasks())
          and "--suite edition3" in (Path(__file__).resolve().parent.parent / "docs"
                                     / "RUNBOOK.md").read_text(encoding="utf-8"))
    try:
        generate(edition=3, traps=["duplicate_refund_escalate"])
        refused = False
    except ValueError:
        refused = True
    sel = SimpleNamespace(suite="edition3", languages="", tasks="", sample=0,
                          traps="two_orders_voucher_and_cancel")
    try:
        select_tasks(SimpleNamespace(**{**vars(sel), "traps": "duplicate_refund_escalate"}))
        across = False
    except SystemExit as e:
        across = "no such trap" in str(e) and "--suite edition3" in str(e)
    check("…and a trap the hard edition does not have is refused, not built from another "
          "-- nor selected from another suite by --traps, which picks the suite's own",
          refused and across and len(select_tasks(sel)) == 13)


def test_changed_request_audit():
    """The hard edition's changed request is fair only if she changes her mind
    in the message with her digits, before any write is possible; where she
    does it later, the case measured her (as the photo trap did, #43). The
    audit finds that message and reads it, in the eight languages."""
    print("\n=== the changed request: did she change it with her digits? ===")
    import contextlib
    import importlib.util
    import io

    from pasarbench.simqa import change_with_digits, changes_request
    t3, _ = generate(edition=3)
    task = next(t for t in t3 if t.trap == "changed_request_cancel_to_address"
                and t.language == "en")
    d = task.hidden_facts["phone_last4"]
    changed = [
        f"It's {d}. Actually I've changed my mind, don't cancel it. Can you send it to my new "
        f"address instead?",
        f"{d}. Eh actually don't cancel already, can send to my new address instead?",
        f"{d}. Sebenarnya saya tak jadi batal, boleh hantar ke alamat baru saya?",
        f"{d}. Eh, nggak jadi batal deh, bisa dikirim ke alamat baru saya aja?",
        f"{d} ค่ะ เปลี่ยนใจแล้วค่ะ ไม่ยกเลิกแล้ว ช่วยส่งไปที่อยู่ใหม่แทนได้ไหมคะ",
        f"{d} nhé. À mình đổi ý rồi, không hủy nữa, gửi về địa chỉ mới giúp mình được không?",
        f"{d}。其实我改主意了，不取消了，可以帮我寄到新地址吗？",
        f"{d}. 我不要cancel了，帮我改去新地址可以吗？",
    ]
    plain = [
        d, f"It's {d}.", f"{d}. I moved house, so it would go to my old address.",
        f"Nombor telefon saya {d}.", f"{d} ค่ะ", f"Số cuối là {d}.", f"后四位是{d}。",
    ]
    for line in changed:
        check(f"she changed it: {' '.join(line.split())[:48]}", changes_request(line), line)
    for line in plain:
        check(f"digits alone, or her reason: {' '.join(line.split())[:48]}",
              not changes_request(line), line)
    other = next(t for t in t3 if t.trap != "changed_request_cancel_to_address")
    check("the first turn with her digits, read for the change; nothing for another case",
          change_with_digits(task, ["Order GO-1.", f"{d}", changed[0]]) == (1, False)
          and change_with_digits(task, ["Order GO-1.", changed[0]]) == (1, True)
          and change_with_digits(task, ["no digits"]) == (None, False)
          and change_with_digits(other, [changed[0].replace(d, other.hidden_facts["phone_last4"])])
          == (None, False))

    spec = importlib.util.spec_from_file_location(
        "inspect_trace", Path(__file__).resolve().parent.parent / "scripts" / "inspect_trace.py")
    it = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(it)

    def ep(seed, text, passed, tid=task.task_id):
        return {"id": f"{tid}__r{seed}", "task_id": tid, "language": "en", "passed": passed,
                "steps": [], "events": [{"kind": "user_turn", "turn": 1, "text": text}]}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        it.changed_requests([ep(0, changed[0], True), ep(1, f"It's {d}.", False)])
    text = out.getvalue()
    check("inspect_trace counts her by language, with the failures with and without the "
          "change, and shows where she gave her digits without it",
          "en       50.0%  (1/2 episodes)" in text
          and "Failed: 0 of 1 with it; 1 of 1 without it" in text
          and f"WITHOUT {task.task_id}__r1" in text, text)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        it.changed_requests([ep(2, "hello", False, tid=other.task_id)])
    check("…and says nothing for a run without the case", out.getvalue() == "", out.getvalue())


def test_hard_vs_parts():
    """RUNBOOK 1s reads each hard case against its part: the first edition's
    task of the trap whose rule it turns on, in the same market and language.
    HARDER only when the sign test over those pairs says so."""
    print("\n=== the hard edition against its parts ===")
    import contextlib
    import importlib.util
    import io
    import json
    import tempfile

    from pasarbench.generate import EDITION3_PARTS, part_of
    t3, _ = generate(edition=3)
    first = {t.task_id: t for t in TASKS}
    check("every case has its part in the first edition: the same market and language, "
          "the trap its rule turns on, and no two cases one part",
          all(part_of(t) in first and first[part_of(t)].trap == EDITION3_PARTS[t.trap]
              and (first[part_of(t)].market, first[part_of(t)].language)
              == (t.market, t.language) for t in t3)
          and len({part_of(t) for t in t3}) == len(t3))
    spec = importlib.util.spec_from_file_location(
        "hard_vs_parts", Path(__file__).resolve().parent.parent / "scripts" / "hard_vs_parts.py")
    hv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hv)
    check("the parts run the script asks for is the three traps the cases turn on",
          set(hv.PARTS_TRAPS) == set(EDITION3_PARTS.values()))
    tmp = Path(tempfile.mkdtemp())

    def cell(name, tasks, passes, sim="llm-user/v3:openai-compat:qwen3.8-flash"):
        d = tmp / name / "full"
        d.mkdir(parents=True)
        for t in tasks:
            for k in range(2):
                head = {"type": "header", "task_id": t.task_id, "language": t.language,
                        "trap": t.trap, "simulator": sim}
                foot = {"type": "footer", "passed": passes(t, k), "stop_reason": "done",
                        "budget": {"tokens": 1000, "steps": 5}}
                (d / f"{t.task_id}__r{k}.jsonl").write_text(json.dumps(head) + "\n"
                                                           + json.dumps(foot) + "\n")
        return str(d)
    twins = [first[part_of(t)] for t in t3]

    def run(*args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            hv.main([*args, "--checker", "recorded"])
        return out.getvalue()
    parts = cell("parts", twins, lambda t, k: True)
    harder = cell("hard", t3, lambda t, k: t.trap == "changed_request_cancel_to_address"
                  or k == 0)
    text = run(parts, harder)
    check("23 of the 36 cases fail where their part never did: HARDER, with each case's "
          "pass rate beside its part's",
          "36 pairs" in text and "23 cases failed more than their part, 0 less" in text
          and "->  HARDER" in text and "duplicate_refund_cod_livestream" in text
          and "0.500 (0.00)" in text and "1.000 (1.00)" in text, text)
    same = cell("same", t3, lambda t, k: True)
    check("…and as many passes as their parts is no difference",
          "->  INCONCLUSIVE" in run(parts, same), run(parts, same))
    few = cell("few", twins[:3], lambda t, k: True, sim="llm-user/v2:openai-compat:qwen3.8-flash")
    text = run(few, harder)
    check("…a case whose part did not run is named and left out, and two customers are "
          "flagged", "33 case(s) whose part is not in" in text
          and "different simulated customers" in text and "not testable" in text, text)
    check("…and with no voucher declined, nothing is read after the fact",
          "WHAT_FAILED #46" not in run(parts, harder))

    # WHAT_FAILED #46: on the two orders she declines the voucher and the
    # agent escalates the late order, as P6.3 and P10 ask; the check fails it.
    def declined(f):
        lines = [json.loads(l) for l in Path(f).read_text().splitlines()]
        head, foot = lines[0], lines[-1]
        return "\n".join(json.dumps(r) for r in (
            head, {"type": "step", "model_content": "A goodwill voucher, or a human agent?"},
            {"type": "event", "kind": "user_turn", "text": "A full refund. Escalate it."},
            {"type": "step", "model_content": "", "tool_calls": [
                {"name": "escalate_to_human", "arguments": {
                    "order_id": "X", "category": "out_of_window_dispute"}}],
             "tool_results": [{"name": "escalate_to_human", "ok": True}]},
            {**foot, "passed": False, "failures": [
                "db: the late order's goodwill voucher, which she accepts (P6.2) -- "
                "vouchers matching {}: found 0, expected >= 1",
                "missing required action: issue_goodwill_voucher {}"]})) + "\n"
    chose = cell("chose", t3, lambda t, k: True)
    for f in Path(chose).glob("TOVAC3-*__r0.jsonl"):
        f.write_text(declined(f))
    text = run(parts, chose)
    n = sum(t.trap == "two_orders_voucher_and_cancel" for t in t3)
    check("…a two-orders case whose customer declined the voucher fails its pair as "
          "checked, and the line read as P6.3 and P10 read her escalation is beside it",
          f"{n} cases failed more than their part" in text and "->  HARDER" in text
          and f"{n} of the cases' episodes, 0 of the parts' (WHAT_FAILED #46): +0.000; 0 "
          "cases failed more than their part, 0 less, p=1.000" in text, text)


def main() -> int:
    test_all_solvable()
    test_solutions_run_through_harness()
    test_null_and_spam_fail()
    test_locale_twins_identical()
    test_market_thresholds()
    test_date_invariants()
    test_timeline()
    test_worlds_are_deterministic()
    test_read_alternatives()
    test_coverage()
    test_simulator_qa()
    test_leak_detector_reads_every_language()
    test_leak_rates_are_per_episode()
    test_photo_claims()
    test_second_edition()
    test_hard_edition()
    test_changed_request_audit()
    test_hard_vs_parts()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
