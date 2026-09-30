"""
Re-scoring recorded episodes under today's checks (pasarbench/rescore.py).

A checker change moves every recorded verdict it touches, and the report shows
both scorings. These tests hold the replay to what it claims: it rebuilds the
world the episode ran in and the state it ended in, it scores that with
today's checks, and whatever it cannot rebuild keeps its recorded verdict and
says so -- including the tracking numbers v18 drew from a salted hash.

Run: python -m tests.test_rescore
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

from pasarbench.db import Database
from pasarbench.generate import generate
from pasarbench.harness import TraceWriter, run_episode
from pasarbench.harness.backends import ScriptedBackend
from pasarbench.harness.replay import pre_v19_patch, replay_payloads
from pasarbench.rescore import all_tasks, cells, rescore_dir, rescore_file, summary
from pasarbench.run import SOLUTIONS
from pasarbench.verifier import verify

PASS, FAIL = [], []
GEN, GEN_SOL = generate()
SOL = {**SOLUTIONS, **GEN_SOL}


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + ("" if ok else f"  -- {str(detail)[:600]}"))


def _episode(tmp, task, script, world=None, run_id="R/full", i=0):
    tw = TraceWriter(root=tmp, run_id=run_id)
    db = Database.fresh(task.db_patch if world is None else world)
    res = run_episode(task, db, ScriptedBackend(script), trace=tw, run_index=i)
    v = verify(task, db)
    tw.close_episode(res.stop_reason.value, v.passed, v.failures, res.budget)
    return Path(tmp, run_id, f"{task.task_id}__r{i}.jsonl"), v


def _read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write(path, recs):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


def _as_v18(path, passed=None, failures=None):
    """What a trace from before v19 looks like: no world digest, no payload
    digests -- and, optionally, the verdict the old checker gave it."""
    recs = _read(path)
    for r in recs:
        r.pop("world_digest", None)
        for tr in r.get("tool_results") or []:
            tr.pop("result_sha1", None)
        if r.get("type") == "footer" and passed is not None:
            r["passed"], r["failures"] = passed, failures or []
    _write(path, recs)


def _no_tracking(patch):
    p = copy.deepcopy(patch)
    for s in p.get("shipments", {}).values():
        if "tracking_no" in s and s["order_id"].startswith("GO-"):
            s["tracking_no"] = "?"
    return hashlib.sha1(json.dumps(p, sort_keys=True, ensure_ascii=False,
                                   default=str).encode()).hexdigest()[:12]


def test_pre_v19_world():
    print("\n=== the world a pre-v19 trace ran in is rebuilt exactly ===")
    ref = json.loads((Path(__file__).parent / "fixtures" / "pre_v19_worlds.json").read_text())
    tasks = all_tasks()
    bad = [tid for tid, t in tasks.items() if _no_tracking(pre_v19_patch(t)) != ref.get(tid)]
    check(f"all {len(tasks)} worlds match the v18 generator's, tracking numbers aside",
          not bad and set(ref) == set(tasks), bad[:5])
    moved = sum(_no_tracking(t.db_patch) != ref[tid] for tid, t in tasks.items())
    check("…and v19's worlds differ from them where the dates were fixed", moved >= 64,
          str(moved))


def test_unchanged_checks_keep_their_verdicts():
    print("\n=== same checks, same state: the replay changes no verdict ===")
    tasks = all_tasks()
    ids = ["T01", "T08", "T13", "HRWR-ID.id", "OOWDE-TH.th", "CHE-MY", "LCOW-VN"]
    with tempfile.TemporaryDirectory() as tmp:
        for tid in ids:
            _episode(tmp, tasks[tid], SOL[tid])
        # and one that fails, for the other direction
        _episode(tmp, tasks["T11"], [("verify_identity", {"user_id": "U003", "phone_last4": "6789"}),
                                      ("issue_goodwill_voucher", {"user_id": "U003", "amount_minor": 20000,
                                                                  "currency": "IDR", "reason": "delay"})])
        rows = rescore_dir(Path(tmp, "R", "full"))
        check("every episode is re-scored", all(r.status == "rescored" for r in rows),
              [(r.transcript_id, r.status, r.detail) for r in rows])
        check("…to the verdict it recorded", all(r.current == r.recorded for r in rows),
              [(r.transcript_id, r.recorded, r.current) for r in rows if r.current != r.recorded])
        check("…passes and a failure among them",
              sum(r.verdict for r in rows) == len(ids)
              and not next(r for r in rows if r.task_id == "T11").verdict,
              [(r.transcript_id, r.verdict) for r in rows])
        f = Path(tmp, "R", "full", "CHE-MY__r0.jsonl")
        rp = replay_payloads(_read(f), tasks["CHE-MY"])
        check("a v19 trace is verified by digest, every payload, in today's world",
              rp["by"] == "digest" and rp["verified"] == rp["total"] > 0
              and rp["world"] == "same", {k: rp[k] for k in ("by", "verified", "total", "world")})
        s = summary(rows)
        check("the summary shows both scorings and nothing moved",
              s["recorded"] == s["current"] and s["fail_to_pass"] == s["pass_to_fail"] == 0
              and s["payloads"][0] == s["payloads"][1], s)


def test_old_checker_verdict_moves():
    print("\n=== a pre-v19 episode failed by the old peak check is re-scored ===")
    tasks = all_tasks()
    task = next(t for t in GEN if t.trap == "peak_period_delay_not_compensable")
    with tempfile.TemporaryDirectory() as tmp:
        # It read the order date from list_user_orders and gave nothing away --
        # what the policy asks -- and the old check failed it for not calling
        # get_order. It ran in the pre-v19 world and its trace has no digests.
        path, _ = _episode(tmp, task, [("list_user_orders", {"user_id": task.user_id})],
                           world=pre_v19_patch(task))
        oid = task.hidden_facts["order_id"]
        _as_v18(path, passed=False,
                failures=[f"missing required action: get_order {{'order_id': '{oid}'}}"])
        r = rescore_file(path)
        check("today's checks pass it, and the move is recorded",
              r.recorded is False and r.current is True and r.moved == "fail→pass",
              r.to_dict())
        check("…replayed in the pre-v19 world, by length",
              "pre-v19" in r.detail and r.payloads[0] == r.payloads[1] == 1, r.to_dict())
        s = summary([r])
        check("the summary counts it once, in both scorings",
              s["recorded"] == 0.0 and s["current"] == 1.0 and s["fail_to_pass"] == 1, s)


def test_what_cannot_be_rebuilt_keeps_its_verdict():
    print("\n=== an episode the replay cannot rebuild is never re-scored on a guess ===")
    tasks = all_tasks()
    with tempfile.TemporaryDirectory() as tmp:
        path, v = _episode(tmp, tasks["T01"], SOL["T01"])
        recs = _read(path)
        for rec in recs:          # a write the run saw fail, which succeeds on replay
            for tr in rec.get("tool_results") or []:
                if tr["name"] == "initiate_return":
                    tr["ok"], tr["error"] = False, "(something the run saw)"
        foot = recs[-1]
        foot["passed"], foot["failures"] = False, ["db: returns found 0"]
        _write(path, recs)
        r = rescore_file(path)
        check("a call whose outcome differs marks the episode diverged",
              r.status == "diverged" and r.current is None, r.to_dict())
        check("…and it counts with the verdict it recorded", r.verdict is False)

        path2, _ = _episode(tmp, tasks["T03"], SOL["T03"], run_id="R2/full")
        recs = _read(path2)
        recs[0]["task_id"] = "T99"
        _write(path2, recs)
        r2 = rescore_file(path2)
        check("a task today's suite no longer has is kept as recorded",
              r2.status == "unknown_task" and r2.verdict is True, r2.to_dict())

        path3, _ = _episode(tmp, tasks["T04"], SOL["T04"], run_id="R3/full")
        _write(path3, _read(path3)[:-1])
        r3 = rescore_file(path3)
        check("an episode that never finished is not counted", r3.status == "unfinished"
              and summary([r3])["episodes"] == 0, r3.to_dict())

        # Run B predates the order-id fix (#9): its calls name orders today's
        # world does not have. They fail in the replay as they failed in the
        # run -- so success alone agrees -- but the payloads do not.
        path4, _ = _episode(tmp, tasks["T15"], SOL["T15"], run_id="R4/full")
        _as_v18(path4)
        recs = _read(path4)
        for rec in recs:
            for tr in rec.get("tool_results") or []:
                if tr["name"] == "get_order":
                    tr["result_chars"] += 7
        _write(path4, recs)
        r4 = rescore_file(path4)
        check("a payload that differs from the recording marks the world changed",
              r4.status == "world_changed" and r4.current is None and r4.verdict is True,
              r4.to_dict())

        # ...and when it is older than the task digest, the ids it names decide
        idt = next(t for t in GEN if t.trap == "identity_verification_failure")
        path6, _ = _episode(tmp, idt, [("verify_identity", {"user_id": idt.user_id,
                                                            "phone_last4": "1234"}),
                                       ("escalate_to_human", {"order_id": "GO-identiSG",
                                                              "category": "identity_unverified",
                                                              "reason": "x"})],
                            world=pre_v19_patch(idt), run_id="R6/full")
        _as_v18(path6, passed=True)
        recs = _read(path6)
        recs[0].pop("task_digest", None)
        _write(path6, recs)
        r6 = rescore_file(path6)
        check("a trace older than the digest that names an order today's world lacks "
              "is kept as recorded", r6.status == "task_changed" and r6.verdict is True,
              r6.to_dict())

        path5, _ = _episode(tmp, tasks["T16"], SOL["T16"], run_id="R5/full")
        recs = _read(path5)
        recs[0]["task_digest"] = "000000000000"
        _write(path5, recs)
        r5 = rescore_file(path5)
        check("a task whose opening or facts changed since the run is kept as recorded",
              r5.status == "task_changed" and r5.current is None, r5.to_dict())


def test_salted_tracking_numbers():
    print("\n=== #33: a tracking number no one can rebuild is never 'verified' ===")
    tasks = all_tasks()
    task = tasks["CHE-MY"]
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _episode(tmp, task, SOL["CHE-MY"], world=pre_v19_patch(task))
        _as_v18(path)
        rp = replay_payloads(_read(path), task)
        ship = [i for i, (n, _) in enumerate(SOL["CHE-MY"]) if n == "get_shipment"]
        check("the generated shipment's payload is withheld",
              all(rp["payloads"][i] is None for i in ship) and ship, rp["payloads"])
        check("…and every other payload is verified by length",
              rp["verified"] == rp["total"] - len(ship) and rp["by"] == "length", rp)
        r = rescore_file(path)
        check("the verdict does not depend on it: still re-scored",
              r.status == "rescored" and r.current is True, r.to_dict())
        # A tampered digest is caught where a length never could be.
        path2, _ = _episode(tmp, task, SOL["CHE-MY"], run_id="R2/full")
        recs = _read(path2)
        n = 0
        for rec in recs:
            for tr in rec.get("tool_results") or []:
                if tr["name"] == "get_shipment":
                    tr["result_sha1"], n = "000000000000", n + 1
        _write(path2, recs)
        rp2 = replay_payloads(recs, task)
        check("a payload whose digest differs is withheld, lengths notwithstanding",
              rp2["verified"] == rp2["total"] - n and n > 0, rp2)


def test_same_error_on_every_interpreter():
    """An argument the tool does not take came back as Python's own TypeError
    text, and Python 3.13 appends "Did you mean 'user_id'?" to it. A run on
    3.13 told the agent more than a run on 3.12, and a replay on 3.13 could
    not reproduce a 3.12 recording: P-ref's OOWOV-TH.th__r0 was kept at its
    recorded verdict on one machine and re-scored on another."""
    print("\n=== the environment says the same thing on every Python ===")
    import sys
    from pasarbench.db import Database
    from pasarbench.tools import call
    out = call(Database.fresh(), "issue_goodwill_voucher",
               {"amount_minor": 30000, "currency": "THB", "order_id": "GO-F22ED7",
                "reason": "goodwill", "user_id": "GU-TH"})
    want = ("bad arguments: issue_goodwill_voucher() got an unexpected keyword "
            "argument 'order_id'")
    check(f"an unknown argument reads as every recorded run shows it "
          f"(Python {sys.version_info.major}.{sys.version_info.minor})",
          out == {"ok": False, "error": want}, out)
    out = call(Database.fresh(), "get_order", {"order_id": "O1001", "orderId": "O1001",
                                               "user": "U001"})
    check("…and names the first of several, in the order given, as Python did",
          out["error"].endswith("argument 'orderId'"), out)


def test_cells_and_loader():
    print("\n=== the report reads today's verdicts through the same loader ===")
    from pasarbench import diagnose
    tasks = all_tasks()
    task = next(t for t in GEN if t.trap == "high_value_photo_required_first")
    with tempfile.TemporaryDirectory() as tmp:
        ver = ("verify_identity", {"user_id": task.user_id,
                                   "phone_last4": task.hidden_facts["phone_last4"]})
        path, _ = _episode(tmp, task, [ver, ("get_order", {"order_id": task.hidden_facts["order_id"]})],
                           world=pre_v19_patch(task), run_id="P-base/full")
        _as_v18(path, passed=False, failures=["missing required action: check_return_eligibility"])
        _episode(tmp, tasks["T01"], SOL["T01"], run_id="P-base/full", i=1)
        check("cells() finds traces/<run>/<cell>", cells(tmp) == [Path(tmp, "P-base", "full")],
              cells(tmp))
        rec = diagnose.load_episodes(Path(tmp, "P-base", "full"))
        cur = diagnose.load_episodes(Path(tmp, "P-base", "full"), checker="current")
        check("recorded by default", [e["passed"] for e in rec] == [False, True],
              [e["passed"] for e in rec])
        check("today's checks when asked, with the recorded verdict alongside",
              [e["passed"] for e in cur] == [True, True]
              and [e["passed_recorded"] for e in cur] == [False, True]
              and cur[0]["first_failure"] is None, cur)
        diagnose.use_checker("current")
        try:
            check("…and use_checker() switches the default for every caller",
                  [e["passed"] for e in diagnose.load_episodes(Path(tmp, "P-base", "full"))]
                  == [True, True])
        finally:
            diagnose.use_checker("recorded")

        # A stray folder of traces beside the runs used to hide every run.
        stray = Path(tmp, "scratch")
        stray.mkdir()
        (stray / path.name).write_text(path.read_text())
        check("…a stray folder of traces does not hide the runs beside it",
              cells(tmp) == [Path(tmp, "P-base", "full"), stray], cells(tmp))
        # An episode that never finished is left out everywhere alike: by the
        # sweep's summary, by rescore's, and by the report's loader.
        cut = Path(tmp, "P-base", "full", "T01__r9.jsonl")
        recs = _read(Path(tmp, "P-base", "full", "T01__r1.jsonl"))
        _write(cut, [r for r in recs if r.get("type") != "footer"])
        from pasarbench.harness.trace import summarise_run
        from pasarbench.rescore import rescore_dir, summary
        sm = summary(rescore_dir(Path(tmp, "P-base", "full")))
        check("…and an unfinished episode is left out of every count alike",
              len(diagnose.load_episodes(Path(tmp, "P-base", "full"))) == 2
              and sm["episodes"] == 2 and sm["unfinished"] == 1
              and summarise_run(Path(tmp, "P-base", "full"))["episodes"] == 2, sm)


def main() -> int:
    test_pre_v19_world()
    test_unchanged_checks_keep_their_verdicts()
    test_old_checker_verdict_moves()
    test_what_cannot_be_rebuilt_keeps_its_verdict()
    test_salted_tracking_numbers()
    test_same_error_on_every_interpreter()
    test_cells_and_loader()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
