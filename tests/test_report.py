"""
scripts/make_report.py -- the file every quoted number comes from.

These tests hold the report to its own rules on the cases that broke it:
a verdict read off point estimates, a tie broken silently in favour of a broken
run, and a table of kappas computed over labels with nothing in them to agree
about. Each is built from the smallest data that reproduces it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import tempfile
from pathlib import Path

from pasarbench.analyze import markdown_report

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + ("" if ok else f"  -- {detail[:600]}"))


def _report():
    spec = importlib.util.spec_from_file_location(
        "make_report", Path(__file__).resolve().parent.parent / "scripts" / "make_report.py")
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    return mr


def _gap(diff, worse, better, p, n=32):
    return {"unit": "task", "n": n, "diff": diff, "worse": worse, "better": better,
            "p": p, "resolved": p < 0.05}


def test_context_verdicts_need_a_paired_test():
    print("\n=== context ablation: no ranking off point estimates ===")
    rows = [{"strategy": s, "pass^1": p, "pass^k": p - 0.05, "mean_tokens": t,
             "per_trap": {}, "per_language": {}}
            for s, p, t in (("full", 0.917, 38720), ("window8", 0.979, 38256),
                            ("window4", 0.885, 43344))]
    paired = {"window8": {"pass": _gap(0.062, 0, 4, 0.125), "tokens": _gap(-460, 9, 7, 0.80)},
              "window4": {"pass": _gap(-0.031, 3, 1, 0.625), "tokens": _gap(4600, 5, 20, 0.004)}}
    md = markdown_report(rows, "full", paired=paired)
    check("a 6-point gap at p=0.125 is INCONCLUSIVE, not a ranking",
          "INCONCLUSIVE on accuracy" in md and "Dominated" not in md, md)
    check("a real cost difference is still reported",
          "`window4` is **costlier**" in md, md)
    check("tokens read the right way round (negative = cheaper)",
          "20 costlier" in md, md)
    legacy = markdown_report(rows, "full")
    check("without paired data the old table is unchanged", "frontier" in legacy)

    rows[2]["per_trap"] = {"a": 0.5, "b": 1.0}
    rows[0]["per_trap"] = {"a": 1.0, "b": 1.0}
    rows[1]["per_trap"] = {"a": 1.0, "b": 1.0}
    md = markdown_report(rows, "full", paired=paired)
    line = next(l for l in md.splitlines() if l.startswith("**`window4`**"))
    check("an unresolved strategy's trap dips are leads, not a located loss",
          "leads to read, not losses" in line and "diffuse" not in line
          and "concentrated" not in line, line)


def test_ties_are_said_out_loud():
    print("\n=== a tie is not broken silently ===")
    mr = _report()
    cands = [("G-gated/full", 1075, "g"), ("C-clean/full", 1075, "c"),
             ("F-clean-sim/full", 1075, "f"), ("D-nozh/full", 930, "d")]
    payload, note = mr._choose(cands, "", "--multilingual-run")
    check("a tie is announced, naming the others", "a tie at 1075 episodes" in note
          and "`F-clean-sim/full`" in note and "`G-gated/full`" in note, note)
    check("…and broken the same way every time", payload == "c", str(payload))
    payload, note = mr._choose(cands, "D-nozh", "--multilingual-run")
    check("pinning overrides it", payload == "d" and "pinned" in note, note)


def test_replication_table():
    print("\n=== several pinned runs: the gap, run by run ===")
    mr = _report()
    rng = random.Random(3)

    def run(gap_th):
        eps = []
        for i in range(16):
            base = f"T{i:02d}"
            for k in range(3):
                eps.append({"task_id": base, "transcript_id": f"{base}__r{k}",
                            "language": "en", "passed": rng.random() < 0.9})
                eps.append({"task_id": f"{base}.th", "transcript_id": f"{base}.th__r{k}",
                            "language": "th", "passed": rng.random() < 0.9 - gap_th})
                eps.append({"task_id": f"{base}.vi", "transcript_id": f"{base}.vi__r{k}",
                            "language": "vi", "passed": rng.random() < 0.9})
        return eps

    md = mr._replication([("C-clean", run(0.6), ""), ("D-nozh", run(0.6), "")])
    check("a large gap in both runs replicates",
          "| `th` |" in md and "**replicates**" in md.split("| `th` |")[1].split("\n")[0], md)
    check("no gap in either run reads as a null",
          "null in every run" in md.split("| `vi` |")[1].split("\n")[0], md)
    check("…and no gated run, no gate note", "fact gate" not in md)

    def near_miss():
        # 9 twins worse, 2 better, 5 tied: a big gap at p = 0.065 by design
        eps = []
        for i in range(16):
            base = f"T{i:02d}"
            eps.append({"task_id": base, "transcript_id": f"{base}__r0",
                        "language": "en", "passed": i < 14, "simulator": "llm-user:x+gated"})
            eps.append({"task_id": f"{base}.id", "transcript_id": f"{base}.id__r0",
                        "language": "id", "passed": i >= 9, "simulator": "llm-user:x+gated"})
        return eps

    clean = [dict(e, language=e["language"]) for e in run(0.0)
             if e["language"] in ("en",)] + [
        dict(e, language="id", task_id=e["task_id"].replace(".vi", ".id"),
             transcript_id=e["transcript_id"].replace(".vi", ".id"))
        for e in run(0.0) if e["language"] == "vi"]
    md = mr._replication([("C-clean", clean, ""), ("G-gated", near_miss(), "")])
    row = md.split("| `id` |")[1].split("\n")[0]
    check("a large gap that just misses is named, not called nothing",
          "closest: `G-gated`" in row, row)
    check("a gated run is flagged from its own trace headers",
          "`G-gated` ran with the fact gate" in md and "`C-clean` ran" not in md, md[-600:])


def test_calibration_without_variance():
    print("\n=== labels with nothing in them to agree about ===")
    mr = _report()
    from pasarbench.judge.rubric import CRITERIA
    keys = [c.key for c in CRITERIA]
    tmp = Path(tempfile.mkdtemp())
    lab = tmp / "data" / "labels"
    lab.mkdir(parents=True)
    r1 = [{"transcript_id": f"T{i}", "labels": {k: True for k in keys}, "labeller": "me",
           "at": "2026-09-18T10:00:00+00:00"} for i in range(50)]
    r1[0]["labels"][keys[0]] = False
    sample = [{"transcript_id": f"T{i}", "passed": i >= 10} for i in range(50)]
    for name, rows in (("human_round1", r1), ("sample", sample)):
        (lab / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        md = mr.section_judge()
    finally:
        os.chdir(cwd)
    check("the report says the labels cannot calibrate anything",
          "These labels cannot calibrate a judge" in md, md[:500])
    check("…counts the failed transcripts rated clean",
          "9 of the 10 sampled transcripts that FAILED" in md, md[:700])

    r2 = [{"transcript_id": f"T{i}", "labels": {k: True for k in keys},
           "labeller": "friend", "at": "2026-09-19T10:00:00+00:00"} for i in range(10)]
    (lab / "human_round2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in r2))
    try:
        os.chdir(tmp)
        md = mr.section_judge()
    finally:
        os.chdir(cwd)
    check("…and with a second rater, the ceiling is undefined, not '0% of achievable'",
          "inter-annotator (me vs friend): undefined." in md
          and "of achievable agreement" not in md, md[-800:])
    check("…and the header does not call a second rater a retest",
          "round 2 (retest)" not in md and "round 2 = 10" in md, md[:120])

    naive = [{"transcript_id": f"T{i}", "score": 5, "labels": {"overall_acceptable": True}}
             for i in range(50)]
    dec = [{"transcript_id": f"T{i}",
            "labels": {k: not (k == "no_hallucinated_facts" and i % 2) for k in keys}}
           for i in range(50)]
    for name, rows in (("judge_naive", naive), ("judge_decomposed", dec)):
        (lab / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        os.chdir(tmp)
        md = mr.section_judge()
    finally:
        os.chdir(cwd)
    check("two zero kappas are not reported as two indistinguishable judges",
          "indistinguishable" not in md and "zero by construction" in md
          and "the nine-criterion judge on 48%" in md, md[md.find("Decomposed judge's"):][:900])


def test_attribution_needs_a_resolved_gap():
    print("\n=== no mechanism for a gap the paired test calls noise ===")
    from pasarbench.diagnose import load_episodes, report
    tmp = Path(tempfile.mkdtemp())

    def write(gap_th):
        d = Path(tempfile.mkdtemp(dir=tmp))
        for i in range(16):
            for lang, task, ok in (("en", f"T{i:02d}", True),
                                   ("th", f"T{i:02d}.th", i >= int(16 * gap_th))):
                (d / f"{task}__r0.jsonl").write_text(
                    json.dumps({"type": "header", "task_id": task, "language": lang}) + "\n"
                    + json.dumps({"type": "footer", "passed": ok, "stop_reason": "done",
                                  "budget": {"tokens": 1000, "steps": 5}}) + "\n")
        return report(load_episodes(d))

    md = write(0.0625)            # one twin worse: a 6-point gap, p = 1.0
    check("a gap the sign test calls noise gets no attribution",
          "no gap to attribute" in md and "**`th`**" not in md, md[md.find("### Gap"):][:400])
    md = write(0.75)              # twelve twins worse
    check("a resolved gap still does, with the causal caveat",
          "**`th`** -- paired gap" in md and "CO-MOVEMENTS" in md,
          md[md.find("### Gap"):][:400])


def test_noise_floor():
    print("\n=== the same configuration, run twice ===")
    mr = _report()
    tmp = Path(tempfile.mkdtemp())
    for run, cell, passing in (("H", "full", range(0, 7)), ("I", "full+all-20", range(0, 10))):
        d = tmp / run / cell
        d.mkdir(parents=True)
        for i in range(10):
            (d / f"T{i:02d}__r0.jsonl").write_text(
                json.dumps({"type": "header", "task_id": f"T{i:02d}", "language": "en"}) + "\n"
                + json.dumps({"type": "footer", "passed": i in passing, "stop_reason": "done"}) + "\n")
    md = mr.section_noise(tmp, "H/full,I/full+all-20")
    check("the noise floor is the paired difference between the two runs",
          "7/10 vs 10/10 passed" in md and "3 tasks better and 0 worse" in md, md)
    check("…and it says it is an assertion, not an inference", "by your assertion" in md)
    check("nothing is printed unless a pair is named", mr.section_noise(tmp, "") == "")


def test_post_training():
    print("\n=== post-training: base, fold-swapped RFT, reference ===")
    mr = _report()
    from pasarbench.generate import generate
    from pasarbench.rl.split import make_folds, other_fold
    from pasarbench.tasks import TASKS, task_digest
    tasks = (list(TASKS) + list(generate()[0]))[:40]
    folds = make_folds(list(TASKS) + list(generate()[0]))
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / "splits").mkdir(parents=True)
    (tmp / "data" / "splits" / "folds.json").write_text(json.dumps(folds))
    fm = {"A": "Qwen/Qwen3-8B:pasar-rft-A", "B": "Qwen/Qwen3-8B:pasar-rft-B"}

    def run(name, model_of, passes, temp=0.0,
            sim="llm-user:openai-compat:RedHatAI/gemma-4-31B-it-FP8-Dynamic",
            setup=None):
        d = tmp / "traces" / name / "full"
        d.mkdir(parents=True)
        for i, t in enumerate(tasks):
            for k in range(2):
                head = {"type": "header", "task_id": t.task_id, "language": t.language,
                        "trap": t.trap, "requested_model": model_of(t), "simulator": sim,
                        "agent_temperature": temp, "task_digest": task_digest(t)}
                foot = {"type": "footer", "passed": passes(i, k), "stop_reason": "done",
                        "budget": {"tokens": 1000, "steps": 5}}
                (d / f"{t.task_id}__r{k}.jsonl").write_text(json.dumps(head) + "\n"
                                                           + json.dumps(foot) + "\n")
        if setup:
            (d.parent / "summary.json").write_text(json.dumps([{"setup": setup}]))

    run("P-base", lambda t: "Qwen/Qwen3-8B", lambda i, k: i % 3 != 0)
    held_out = lambda t: fm[other_fold(folds["task_fold"][t.task_id])]
    run("P-rft", held_out, lambda i, k: True,
        setup={"fold_models": "A=Qwen/Qwen3-8B:pasar-rft-A,B=Qwen/Qwen3-8B:pasar-rft-B"})
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        md = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("RFT is compared with base, paired by task, and the gain resolves",
          "RFT, each on its held-out fold" in md and "is better than base" in md, md[:1500])
    check("the routing is verified episode by episode",
          "80/80 RFT episodes answered by the model that did not train" in md, md)
    check("the missing reference prints the command that fills it",
          "Reference: **NOT MEASURED**" in md and "--run-id P-ref" in md)
    check("same customer, same temperature, tasks unchanged: all checked",
          "✓ one customer in every run" in md and "✓ agent temperature 0" in md
          and "✓ tasks unchanged since the split" in md, md[md.find("Checks"):][:600])

    import shutil
    shutil.rmtree(tmp / "traces" / "P-rft")
    leak = lambda t: fm[folds["task_fold"][t.task_id]] if t.task_id == tasks[0].task_id else held_out(t)
    run("P-rft", leak, lambda i, k: True,
        setup={"fold_models": "A=Qwen/Qwen3-8B:pasar-rft-A,B=Qwen/Qwen3-8B:pasar-rft-B"})
    run("P-ref", lambda t: "deepseek-v4-pro", lambda i, k: True,
        sim="llm-user:openai-compat:qwen3.8-flash")
    try:
        os.chdir(tmp)
        md = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("a task scored by the model that trained on it is called a leak",
          "LEAKED" in md and "78/80" in md, md[md.find("Checks"):][:800])
    check("a reference run with a different customer fails the customer check",
          "✗ **one customer in every run" in md, md[md.find("Checks"):][:800])

    empty = Path(tempfile.mkdtemp())
    try:
        os.chdir(empty)
        md = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("with nothing run, the section says what to run first",
          md.startswith("**NOT MEASURED**") and "gpu_pipeline.sh stage1" in md
          and "setup_node.sh" in md, md[:400])


def main() -> int:
    test_context_verdicts_need_a_paired_test()
    test_ties_are_said_out_loud()
    test_replication_table()
    test_calibration_without_variance()
    test_attribution_needs_a_resolved_gap()
    test_noise_floor()
    test_post_training()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
