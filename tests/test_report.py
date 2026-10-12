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

    from pasarbench.tools import CHECKS

    def run(name, model_of, passes, temp=0.0,
            sim="llm-user:openai-compat:RedHatAI/gemma-4-31B-it-FP8-Dynamic",
            setup=None):
        d = tmp / "traces" / name / "full"
        d.mkdir(parents=True)
        for i, t in enumerate(tasks):
            for k in range(2):
                head = {"type": "header", "task_id": t.task_id, "language": t.language,
                        "trap": t.trap, "requested_model": model_of(t), "simulator": sim,
                        "agent_temperature": temp, "task_digest": task_digest(t),
                        "tool_checks": CHECKS}
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
    check("the missing reference prints the command that fills it, with the base run's "
          "customer and repetitions",
          "Reference: **NOT MEASURED**" in md and "--run-id P-ref" in md
          and "--customer 1 " in md and " -k 2 " in md, md[md.find("Reference:"):][:700])
    check("same customer, same temperature, same tools, tasks unchanged: all checked",
          "✓ one customer in every run" in md and "✓ agent temperature 0" in md
          and "✓ one tool version in every run" in md
          and "✓ tasks unchanged since the split" in md, md[md.find("Checks"):][:600])
    for f in (tmp / "traces" / "P-rft" / "full").glob("*.jsonl"):
        lines = f.read_text().splitlines()
        lines[0] = json.dumps({**json.loads(lines[0]), "tool_checks": CHECKS - 1})
        f.write_text("\n".join(lines) + "\n")
    try:
        os.chdir(tmp)
        md3 = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("…and a fine-tune evaluated under other tools than its base fails the tool check: "
          "since v3 a cancellation settles the payment (WHAT_FAILED #37)",
          "✗ **one tool version in every run" in md3, md3[md3.find("Checks"):][:800])

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
    shutil.rmtree(tmp / "traces" / "P-ref")
    for f in (tmp / "traces" / "P-base" / "full").glob("*.jsonl"):
        lines = f.read_text().splitlines()
        lines[0] = json.dumps({**json.loads(lines[0]),
                               "simulator": "llm-user/v2:openai-compat:gemma"})
        f.write_text("\n".join(lines) + "\n")
    try:
        os.chdir(tmp)
        md = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("…and a base run with customer 2 gets a reference command for customer 2: the "
          "name a run recorded says which it had (WHAT_FAILED #38)",
          "--customer 2 " in md and "--customer 1 " not in md, md[md.find("Reference:"):][:700])
    for f in (tmp / "traces" / "P-base" / "full").glob("*.jsonl"):
        lines = f.read_text().splitlines()
        lines[0] = json.dumps({**json.loads(lines[0]), "tool_checks": CHECKS - 1})
        f.write_text("\n".join(lines) + "\n")
    try:
        os.chdir(tmp)
        md = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("…while a base run under older tools gets no command: a reference made now would "
          "run today's, and not be compared with like (WHAT_FAILED #37)",
          "would not be compared with like" in md and "--run-id P-ref" not in md,
          md[md.find("Reference:"):][:700])

    # Round 2 (v46): shown first once its base model has run, with its own
    # collection's stats, and round 1 after it as context, never paired.
    try:
        os.chdir(tmp)
        before = mr.section_training_rounds(Path("traces"))
        alone = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("before round 2 has run, section 6 is round 1's alone", before == alone,
          before[:300])
    v2 = "llm-user/v2:openai-compat:RedHatAI/gemma-4-31B-it-FP8-Dynamic"
    fm2 = {"A": "Qwen/Qwen3-8B:pasar-rft2-A", "B": "Qwen/Qwen3-8B:pasar-rft2-B"}
    run("P2-base", lambda t: "Qwen/Qwen3-8B", lambda i, k: True, sim=v2)
    run("P2-rft", lambda t: fm2[other_fold(folds["task_fold"][t.task_id])],
        lambda i, k: True, sim=v2,
        setup={"fold_models": "A=Qwen/Qwen3-8B:pasar-rft2-A,B=Qwen/Qwen3-8B:pasar-rft2-B"})
    photo = next(t for t in tasks if t.trap == "high_value_photo_required_first")
    said = tmp / "traces" / "P2-base" / "full" / f"{photo.task_id}__r0.jsonl"
    head, foot = [json.loads(l) for l in said.read_text().splitlines()]
    said.write_text("\n".join(json.dumps(r) for r in (
        head, {"type": "event", "kind": "user_turn", "turn": 1,
               "text": "(Attached: photo_1.jpg) Here are the photos."},
        {**foot, "passed": False})) + "\n")
    (tmp / "data" / "rft2").mkdir(parents=True)
    (tmp / "data" / "rft2" / "stats.json").write_text(json.dumps({
        "source": "traces/P2-collect/full", "episodes": 80, "pass_rate": 0.9,
        "examples": {"A": 10, "B": 10}, "checker": "current",
        "selection": {"per_trap": 20, "max_per_task": 8}, "per_trap_pass_rate": {},
        "traps_with_zero_signal": {"A": [], "B": []}}))
    try:
        os.chdir(tmp)
        both = mr.section_training_rounds(Path("traces"))
    finally:
        os.chdir(cwd)
    two, _, one = both.partition("### Round 1, as context")
    check("once it has, round 2 comes first under its own runs, and round 1 after it as "
          "context",
          both.startswith("### Round 2") and "`P2-base`" in two and "`P2-rft`" in two
          and "`P-base`" not in two and "`P-base`" in one and "never paired" in both,
          both[:1200])
    check("…with its own collection: 20 episodes a trap and fold, from its own stats",
          "`traces/P2-collect/full`" in two and "20 a trap and fold" in two,
          two[two.find("Collection"):][:400])
    check("…its reference command named for round 2",
          "--run-id P2-ref" in two, two[two.find("Reference:"):][:500])
    check("…and the photo trap's customer who said she sent photos she has none of, "
          "counted beside the per-trap table (WHAT_FAILED #43)",
          f"She said she had sent them in 1 of {2 * sum(t.trap == photo.trap for t in tasks)} "
          f"base episodes (failed after it 1 of 1, without it 0 of" in two
          and "follows her line" in two
          and "RFT" not in two[two.find("*The photo trap"):].split("\n")[0],
          two[two.find("photo trap"):][:400])
    check("…and no voucher note where no customer declined the voucher",
          "voucher trap's customer" not in two, two[-800:])
    vt = next(t for t in tasks if t.trap == "out_of_window_offer_voucher")
    gave = tmp / "traces" / "P2-rft" / "full" / f"{vt.task_id}__r1.jsonl"
    head, foot = [json.loads(l) for l in gave.read_text().splitlines()]
    gave.write_text("\n".join(json.dumps(r) for r in (
        head, {"type": "step", "model_content": "A goodwill voucher, or a human agent?"},
        {"type": "event", "kind": "user_turn", "text": "A human, please."},
        {"type": "step", "model_content": "", "tool_calls": [{"name": "escalate_to_human",
                                                              "arguments": {
            "order_id": "O1004", "category": "out_of_window_dispute"}}],
         "tool_results": [{"name": "escalate_to_human", "ok": True}]},
        {**foot, "passed": False, "failures": [
            "missing required action: issue_goodwill_voucher {'user_id': 'U004'}"]})) + "\n")
    try:
        os.chdir(tmp)
        two = mr.section_training_rounds(Path("traces")).partition("### Round 1")[0]
    finally:
        os.chdir(cwd)
    check("…while one who did is counted beside it, with what that makes of the verdict "
          "(WHAT_FAILED #46)",
          f"declined it and asked for her money or a human in 1 of "
          f"{2 * sum(t.trap == vt.trap for t in tasks)} RFT episodes (1 of them failed on "
          f"the voucher's lines alone)" in two and "follows her choice" in two,
          two[two.find("voucher trap's customer"):][:500])

    empty = Path(tempfile.mkdtemp())
    try:
        os.chdir(empty)
        md = mr.section_training(Path("traces"))
    finally:
        os.chdir(cwd)
    check("with nothing run, the section says what to run first",
          md.startswith("**NOT MEASURED**") and "gpu_pipeline.sh stage1" in md
          and "setup_node.sh" in md, md[:400])


def test_planted_mistakes_section():
    """Section 4 carries the planted-mistake calibration from its report.json
    (scripts/judge_calibration.py report): caught and false alarms with their
    intervals, the bar fixed before the run, and the 1-5 judge beside them --
    language_match's false alarms counted on the other languages, as the
    calibration counts them."""
    print("\n=== section 4: the planted-mistake calibration ===")
    mr = _report()
    tmp = Path(tempfile.mkdtemp())
    gone = mr.section_planted(tmp / "report.json")
    check("before the calibration has run, the section says so and how to run it",
          "**NOT MEASURED**" in gone and "judge_calibration.py run" in gone, gone)
    crit = lambda **k: {"critical": False, "planted": 20, "clean_pairs": 20, "caught": 0,
                        "false_alarms": 0, "other_languages": 50,
                        "false_alarms_other_languages": 0, "naive_pairs": 20,
                        "naive_accepted_original": 20, "naive_accepted_planted": 20,
                        "trusted": False, **k}
    (tmp / "report.json").write_text(json.dumps({
        "judged": 390, "items": 390, "originals": 210, "criteria": {
            "no_data_leak": crit(critical=True, caught=20, naive_accepted_planted=0,
                                 trusted=True),
            "verification_before_action": crit(caught=1, false_alarms=3),
            "language_match": crit(clean_pairs=19, caught=12, false_alarms=2,
                                   false_alarms_other_languages=2)}}))
    md = mr.section_planted(tmp / "report.json")
    check("per criterion: caught, false alarms with their intervals, left to the judge or "
          "not, and the 1-5 judge's acceptance before and after the plant",
          "| `no_data_leak` (critical) | 20/20 (100%; 84%–100%) | 0/210 (0%; 0%–2%) | "
          "**yes** | 20/20 → 0/20 |" in md
          and "| `verification_before_action` | 1/20 (5%; 1%–24%) | 3/210 (1%; 0%–4%) | no |"
          in md, md)
    check("…language_match's false alarms on the originals in the other languages",
          "| `language_match` | 12/19 (63%; 41%–81%) | 2/50 (4%; 1%–13%) | no |" in md, md)
    check("…and which criteria the judge may be left with, the rest scoring nothing",
          "1 of 3 criteria can be left to the judge: `no_data_leak`." in md
          and "scores nothing" in md, md[-500:])


def test_grpo_section():
    """Section 6's GRPO block reads the runs as RUNBOOK phase 4 fixed before
    GRPO ran: on the 62 newer tasks GRPO against RFT ("GRPO helped" only at
    p < 0.05) and RFT against the base model; the held-out fold as the
    regression check; each iteration's build stats as the learning curve; and
    the checks that make the rows comparable -- one customer, temperature 0,
    every episode answered by the adapter that did not train on its fold."""
    print("\n=== section 6: GRPO, as its reading was fixed ===")
    mr = _report()
    from pasarbench.generate import generate
    from pasarbench.rl.split import make_folds, other_fold
    from pasarbench.tasks import TASKS
    from pasarbench.tools import CHECKS
    newer = list(generate(edition=2)[0])
    main = list(TASKS) + list(generate()[0])
    held = main[:40]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / "splits").mkdir(parents=True)
    f2, f1 = make_folds(newer), make_folds(main)
    (tmp / "data" / "splits" / "folds-e2.json").write_text(json.dumps(f2))
    (tmp / "data" / "splits" / "folds.json").write_text(json.dumps(f1))
    v2 = "llm-user/v2:openai-compat:RedHatAI/gemma-4-31B-it-FP8-Dynamic"

    def run(name, tasks, model_of, passes, fold_models=None):
        d = tmp / "traces" / name / "full"
        d.mkdir(parents=True)
        for i, t in enumerate(tasks):
            for k in range(2):
                head = {"type": "header", "task_id": t.task_id, "language": t.language,
                        "trap": t.trap, "requested_model": model_of(t), "simulator": v2,
                        "agent_temperature": 0.0, "tool_checks": CHECKS}
                foot = {"type": "footer", "passed": passes(i, k), "stop_reason": "done",
                        "budget": {"tokens": 1000, "steps": 5}}
                (d / f"{t.task_id}__r{k}.jsonl").write_text(json.dumps(head) + "\n"
                                                           + json.dumps(foot) + "\n")
        if fold_models:
            (d.parent / "summary.json").write_text(json.dumps(
                [{"setup": {"fold_models": ",".join(f"{f}={m}" for f, m in
                                                    fold_models.items())}}]))

    def section():
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            return mr.section_grpo(Path("traces"))
        finally:
            os.chdir(cwd)

    run("P2-base-e2", newer, lambda t: "Qwen/Qwen3.8-27B", lambda i, k: i % 4 != 0)
    md = section()
    check("before GRPO has run: the gate as read on the base model, and the command",
          "pass^1 0.742" in md and "below 0.95, so GRPO runs" in md
          and "**NOT MEASURED**" in md and "gpu_pipeline.sh grpo" in md, md)

    rft = {f: f"Qwen/Qwen3.8-27B:pasar-rft2-{f}" for f in "AB"}
    grpo = {f: f"Qwen/Qwen3.8-27B:pasar-grpo2-i1-{f}" for f in "AB"}
    routed = lambda models, folds: (lambda t: models[other_fold(folds["task_fold"][t.task_id])])
    run("P2-rft-e2", newer, routed(rft, f2), lambda i, k: i % 5 != 0, rft)
    run("P2-grpo-i1-e2", newer, routed(grpo, f2), lambda i, k: True, grpo)
    run("P2-rft", held, routed(rft, f1), lambda i, k: i % 7 != 3, rft)
    run("P2-grpo-i1", held, routed(grpo, f1), lambda i, k: i % 7 != 4, grpo)
    st = tmp / "data" / "grpo2" / "stats"
    st.mkdir(parents=True)
    for f, rate in (("A", 0.5), ("B", 0.625)):
        (st / f"i1-{f}.stats.json").write_text(json.dumps(
            {"episodes": 64, "groups": 8, "pass_rate": rate, "degenerate_rate": 0.25,
             "examples": 300, "model": f"Qwen/Qwen3.8-27B:pasar-rft2-{f}"}))
        (st / f"tasks-{f}.json").write_text(json.dumps(
            {"tasks": [f"X{i}" for i in range(8)],
             "skipped_traps": ["high_value_photo_required_first"],
             "skipped_tasks": ["HVPRF-A", "HVPRF-B"]}))
    md = section()
    check("the result, as fixed before the run: GRPO against RFT on the newer tasks, "
          "\"GRPO helped\" only when the sign test resolves it",
          "- **The result**, GRPO against RFT, paired by task: **GRPO helped** — +0.210; "
          "13 tasks better, 0 worse" in md, md[:1600])
    check("…RFT against the base model on the same tasks, the newer tasks' table with "
          "all three, and their per-trap table",
          "- **RFT on tasks it never saw**, RFT against the base model:" in md
          and "| GRPO, 1 iteration | `P2-grpo-i1-e2` | 62 | 1.000 |" in md
          and "| trap | base | RFT | GRPO, 1 iteration |" in md, md[:2200])
    check("…the held-out fold as the regression check, with the traps that moved and how "
          "to read them",
          "**The held-out fold**" in md and "read_failures.py traces/P2-rft/full "
          "traces/P2-grpo-i1/full" in md and "RFT 0.850" in md and "GRPO 0.850" in md
          and "Traps that moved:" in md, md[md.find("held-out"):][:900])
    check("…the learning curve from each iteration's build, and the traps its tasks left "
          "out",
          "| 1 | A | `pasar-rft2-A` | 64 on 8 tasks | 50% | — | 25% | 300 |" in md
          and "| 1 | B | `pasar-rft2-B` | 64 on 8 tasks | 62% | — | 25% | 300 |" in md
          and "- fold A: 8 tasks; 2 of high_value_photo_required_first left out" in md
          and "— built before the build counted the voucher she declined (v52)." in md
          and "(measured)" not in md, md[md.find("learning curve"):][:900])
    # The learning curve's second point (grpo-measure, v52): GRPO's adapters on
    # the same tasks, measured and not trained on.
    for f, rate in (("A", 0.75), ("B", 0.8)):
        (st / f"i2-{f}.stats.json").write_text(json.dumps(
            {"episodes": 64, "groups": 8, "pass_rate": rate, "degenerate_rate": 0.5,
             "examples": 120, "voucher_declined": 3,
             "model": f"Qwen/Qwen3.8-27B:pasar-grpo2-i1-{f}"}))
    md = section()
    check("…and the point grpo-measure adds: GRPO's adapters on the same tasks, measured and "
          "not trained on, with the voucher she declined counted and where to read the pairing",
          "| 2 (measured) | A | `pasar-grpo2-i1-A` | 64 on 8 tasks | 75% | 3 | 50% | 0 |" in md
          and "\"She declined the voucher\": rollouts the checks failed where P6.3 and P10 "
              "ask for the escalation (WHAT_FAILED #46), counted whichever way the reward "
              "read them; — built before" in md
          and "Point 2 is GRPO's last adapters on the same tasks" in md
          and "logs/r2-grpo-measure-i2.txt" in md, md[md.find("learning curve"):][:1500])
    for f in ("A", "B"):
        (st / f"i2-{f}.stats.json").unlink()
    md = section()
    check("…and the checks: one customer, temperature 0, every episode answered by the "
          "adapter that did not train on its fold",
          "✓ one customer in every run" in md and "✓ agent temperature 0 in every "
          "evaluation" in md and "✓ `P2-grpo-i1-e2`: 124/124 episodes" in md
          and "✓ `P2-rft-e2`: 124/124" in md and "✓ `P2-grpo-i1`: 80/80" in md,
          md[md.find("Checks"):])
    check("…and with no voucher declined anywhere, no line read after the fact",
          "After the fact" not in md and "WHAT_FAILED #46" not in md, md[:2500])

    # WHAT_FAILED #46: RFT's voucher-trap episodes in which she declined the
    # voucher and the agent escalated, failing on the voucher's lines alone.
    vouch = [t for t in newer if t.trap == "out_of_window_offer_voucher"]
    offer = "I can offer a goodwill voucher, or pass this to a human agent."
    for t in vouch[:6]:
        for k in range(2):
            f = tmp / "traces" / "P2-rft-e2" / "full" / f"{t.task_id}__r{k}.jsonl"
            h, foot = [json.loads(l) for l in f.read_text().splitlines()]
            f.write_text("\n".join(json.dumps(r) for r in (
                h, {"type": "step", "model_content": offer, "tool_calls": [],
                    "tool_results": []},
                {"type": "event", "kind": "user_turn", "text": "A refund, please."},
                {"type": "step", "model_content": "",
                 "tool_calls": [{"name": "escalate_to_human", "arguments": {
                     "order_id": "X", "category": "out_of_window_dispute"}}],
                 "tool_results": [{"name": "escalate_to_human", "ok": True}]},
                {**foot, "passed": False, "failures": [
                    "db: vouchers matching {}: found 0, expected >= 1",
                    "missing required action: issue_goodwill_voucher {}"]})) + "\n")
    md = section()
    check("the voucher trap's customer who declined the voucher is counted beside the "
          "newer tasks' table, and the comparisons are read again as P6.3 and P10 read "
          "her escalations -- after the fact, beside the result, never instead of it",
          f"12 of {2 * len(vouch)} RFT episodes (12 of them failed on the voucher's lines "
          f"alone)" in md
          and "- *After the fact, beside the result and never instead of it* (WHAT_FAILED "
              "#46)" in md
          and "GRPO against RFT: " in md and "RFT against the base model: " in md
          and "the base model's pass^1 0.742" in md
          and md.find("- **The result**") < md.find("After the fact"), md[:3000])
    one = tmp / "traces" / "P2-grpo-i1-e2" / "full" / f"{newer[0].task_id}__r0.jsonl"
    h, foot = [json.loads(l) for l in one.read_text().splitlines()]
    trained_on = grpo[f2["task_fold"][newer[0].task_id]]
    one.write_text(json.dumps({**h, "requested_model": trained_on}) + "\n"
                   + json.dumps(foot) + "\n")
    md = section()
    check("a newer task answered by the adapter of its own half is called a leak",
          "✗ **`P2-grpo-i1-e2`: 123/124" in md and "LEAKED" in md, md[md.find("Checks"):])


def test_edition3_section():
    """Section 6's hard edition on round 2's models, read as RUNBOOK phase 4
    fixed before the run: room first (nothing more read at 0.95 or above),
    then RFT against the base model and GRPO against RFT, paired by case; the
    case table; the changed request's customer and the voucher choice beside
    them; and one customer, temperature 0 and every case answered by the
    adapter its split gives it."""
    print("\n=== section 6: the hard edition on round 2's models ===")
    mr = _report()
    from pasarbench.generate import generate
    from pasarbench.rl.split import make_folds, other_fold
    from pasarbench.simqa import CHANGED_REQUEST
    from pasarbench.tools import CHECKS
    cases = list(generate(edition=3)[0])
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data" / "splits").mkdir(parents=True)
    f3 = make_folds(cases)
    (tmp / "data" / "splits" / "folds-e3.json").write_text(json.dumps(f3))
    v3 = "llm-user/v3:openai-compat:RedHatAI/gemma-4-31B-it-FP8-Dynamic"

    def run(name, model_of, passes, fold_models=None):
        d = tmp / "traces" / name / "full"
        d.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(cases):
            for k in range(2):
                head = {"type": "header", "task_id": t.task_id, "language": t.language,
                        "trap": t.trap, "requested_model": model_of(t), "simulator": v3,
                        "agent_temperature": 0.0, "tool_checks": CHECKS}
                digits = t.hidden_facts.get("phone_last4", "0000")
                said = ([{"type": "event", "kind": "user_turn",
                          "text": f"{digits}. Actually don't cancel it, send it to my new "
                                  f"address instead."}] if t.trap == CHANGED_REQUEST else [])
                foot = {"type": "footer", "passed": passes(i, k), "stop_reason": "done",
                        "budget": {"tokens": 1000, "steps": 5}}
                (d / f"{t.task_id}__r{k}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in (head, *said, foot)) + "\n")
        if fold_models:
            (d.parent / "summary.json").write_text(json.dumps(
                [{"setup": {"fold_models": ",".join(f"{f}={m}" for f, m in
                                                    fold_models.items())}}]))

    def section():
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            return mr.section_edition3(Path("traces"))
        finally:
            os.chdir(cwd)

    md = section()
    check("before the hard edition has run on round 2's models: what is missing and the "
          "command", "**NOT MEASURED**" in md and "gpu_pipeline.sh edition3" in md, md)
    run("P2-base-e3", lambda t: "Qwen/Qwen3.8-27B", lambda i, k: True)
    md = section()
    check("room first: a base model at 0.95 or above, and nothing more is read",
          "the base model passes 1.000, at or above 0.95" in md
          and "nothing more is read" in md and "RFT on combined cases" not in md, md)

    import shutil
    shutil.rmtree(tmp / "traces" / "P2-base-e3")
    run("P2-base-e3", lambda t: "Qwen/Qwen3.8-27B", lambda i, k: i % 4 != 0)
    rft = {f: f"Qwen/Qwen3.8-27B:pasar-rft2-{f}" for f in "AB"}
    grpo = {f: f"Qwen/Qwen3.8-27B:pasar-grpo2-i1-{f}" for f in "AB"}
    routed = lambda models: (lambda t: models[other_fold(f3["task_fold"][t.task_id])])  # noqa: E731
    run("P2-rft-e3", routed(rft), lambda i, k: True, rft)
    run("P2-grpo-i1-e3", routed(grpo), lambda i, k: i % 9 != 1, grpo)
    md = section()
    check("…then RFT against the base model and GRPO against RFT, paired by case, in the "
          "words fixed before the run, GRPO's beside its own result",
          "- **Room:** the base model passes 0.750, below 0.95" in md
          and "RFT against the base model, paired by case: **RFT better** — +0.250; 9 tasks "
              "better, 0 worse, p=0.004" in md
          and "beside GRPO's own result on the 62 newer tasks, never instead of it: "
              "**INCONCLUSIVE**" in md
          and "| GRPO, 1 iteration | `P2-grpo-i1-e3` | 36 |" in md, md[:2500])
    check("…each case's passes by model, and the customer and the voucher choice said even "
          "when there is nothing to set aside",
          "| case | base | RFT | GRPO, 1 iteration |" in md
          and "| `changed_request_cancel_to_address` |" in md
          and "her phone digits came with her change of mind" in md
          and "she never declined it" in md, md[md.find("| case"):][:1500])
    check("…and the checks: one customer, temperature 0, every case answered by the adapter "
          "its split gives it",
          "✓ one customer in every run" in md and "✓ agent temperature 0 in every run" in md
          and "✓ `P2-rft-e3`: 72/72 episodes" in md and "✓ `P2-grpo-i1-e3`: 72/72" in md,
          md[md.find("Checks"):])
    # Her digits without the change: hers, set aside and counted.
    ch = next(t for t in cases if t.trap == CHANGED_REQUEST)
    f = tmp / "traces" / "P2-rft-e3" / "full" / f"{ch.task_id}__r0.jsonl"
    h, _, foot = [json.loads(l) for l in f.read_text().splitlines()]
    f.write_text("\n".join(json.dumps(r) for r in (
        h, {"type": "event", "kind": "user_turn", "text": ch.hidden_facts["phone_last4"]},
        {**foot, "passed": False})) + "\n")
    one = tmp / "traces" / "P2-grpo-i1-e3" / "full" / f"{cases[0].task_id}__r0.jsonl"
    h1, *rest = [json.loads(l) for l in one.read_text().splitlines()]
    one.write_text("\n".join(json.dumps(r) for r in (
        {**h1, "requested_model": grpo[f3["task_fold"][cases[0].task_id]]}, *rest)) + "\n")
    md = section()
    check("a changed request whose digits came without the change is counted as hers, and a "
          "case answered by the adapter of its own half is called a leak",
          "without the change of mind her persona puts in that message in 1 of " in md
          and "RFT episodes (1 of them failed)" in md
          and "✗ **`P2-grpo-i1-e3`: 71/72" in md and "LEAKED" in md, md[-1500:])


def test_both_scorings():
    """#30: the checker the runs were scored with demanded one lookup tool on
    two traps. The report builds on today's checks, shows the recorded
    verdicts beside them, and says when a conclusion moves between the two."""
    print("\n=== one checker for every number, and both scorings shown ===")
    mr = _report()
    from pasarbench import diagnose
    from pasarbench.generate import generate
    from pasarbench.harness.replay import pre_v19_patch
    from tests.test_rescore import _as_v18, _episode
    gen, _ = generate()
    peak = [t for t in gen if t.trap == "peak_period_delay_not_compensable"]
    photo = [t for t in gen if t.trap == "high_value_photo_required_first"]
    tmp = Path(tempfile.mkdtemp())
    traces = tmp / "traces"
    for t in peak + photo:
        oid = t.hidden_facts["order_id"]
        ver = ("verify_identity", {"user_id": t.user_id,
                                   "phone_last4": t.hidden_facts["phone_last4"]})
        item = next(iter(t.db_patch["order_items"]))
        for k in range(2):
            # base: establishes the fact another way, which the old check failed
            base = ([("list_user_orders", {"user_id": t.user_id})] if t in peak
                    else [ver, ("get_order", {"order_id": oid})])
            f, _ = _episode(traces, t, base, world=pre_v19_patch(t),
                            run_id="P-base/full", i=k)
            _as_v18(f, passed=False, failures=["missing required action: "
                                               + ("get_order" if t in peak
                                                  else "check_return_eligibility")])
            # RFT: calls the tool the old check named
            rft = ([("get_order", {"order_id": oid})] if t in peak else
                   [ver, ("check_return_eligibility", {"order_id": oid, "order_item_id": item})])
            f, _ = _episode(traces, t, rft, world=pre_v19_patch(t), run_id="P-rft/full", i=k)
            _as_v18(f)
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        diagnose.use_checker("current")
        md = mr.section_training(Path("traces"))
        md7 = mr.section_rescoring(Path("traces"))
        diagnose.use_checker("recorded")
        old = mr.section_training(Path("traces"))
    finally:
        diagnose.use_checker("recorded")
        os.chdir(cwd)
    check("today's checks: base and RFT tie, INCONCLUSIVE",
          "RFT vs base: **INCONCLUSIVE**" in md and "| 1.000 |" in md, md[:1600])
    check("…the recorded verdicts are shown beside them",
          "The same runs, as recorded at run time" in md
          and "52 of 52" in md, md[md.find("The same runs"):][:900])
    check("…and the conclusion that moved is named",
          "The checker change moves a conclusion here" in md
          and "resolved +1.000" in md, md[md.find("The same runs"):][:1200])
    check("--checker recorded rebuilds it on the recorded verdicts alone",
          "**RFT is better than base**" in old and "as recorded at run time" not in old,
          old[:1400])
    check("section 7: every run, both scorings, what moved",
          "| `P-base/full` | 52 | 0.000 → 1.000 |" in md7
          and "| `P-rft/full` | 52 | 1.000 → 1.000 |" in md7, md7)
    check("…by trap, and only where the checks changed",
          "| `peak_period_delay_not_compensable` | 26 | 0 |" in md7
          and "| `high_value_photo_required_first` | 26 | 0 |" in md7
          and "pass to fail" not in md7, md7)

    rows = [{"strategy": "full", "cell": "full", "pass^1": 0.0, "pass^k": 0.0,
             "per_trap": {}, "mean_tokens": 1}]
    diagnose.use_checker("current")
    try:
        now = mr._rows_now(rows, traces / "P-base")
    finally:
        diagnose.use_checker("recorded")
    check("ablation rows are recomputed from their episodes under today's checks",
          now[0]["pass^1"] == 1.0 and now[0]["pass^1_recorded"] == 0.0
          and now[0]["per_trap"]["peak_period_delay_not_compensable"] == 1.0, now)
    check("…and left as the sweep wrote them under --checker recorded",
          mr._rows_now(rows, traces / "P-base") is rows)


def test_ablation_sections_under_both_checkers():
    """Sections 1-3 were rebuilt on today's checks with nothing to say what the
    recorded verdicts concluded, or that a conclusion had moved -- the tool
    search result went from p=0.021 to p=0.070 without a word in section 2.
    And an arm with no episodes kept the sweep's rates silently."""
    print("\n=== ablation sections: both scorings, and the fallbacks named ===")
    mr = _report()
    from pasarbench import diagnose
    from pasarbench.generate import generate
    from pasarbench.harness.replay import pre_v19_patch
    from tests.test_rescore import _as_v18, _episode
    gen, _ = generate()
    tasks = [t for t in gen if t.trap in ("peak_period_delay_not_compensable",
                                          "high_value_photo_required_first")]
    tmp = Path(tempfile.mkdtemp())
    for t in tasks:
        oid = t.hidden_facts["order_id"]
        ver = ("verify_identity", {"user_id": t.user_id,
                                   "phone_last4": t.hidden_facts["phone_last4"]})
        item = next(iter(t.db_patch["order_items"]))
        peak = t.trap.startswith("peak")
        # `full` reads the fact another way (failed then, passes now);
        # `window8` calls the tool the old check named (passes both).
        other = [("list_user_orders", {"user_id": t.user_id})] if peak \
            else [ver, ("get_order", {"order_id": oid})]
        named = [("get_order", {"order_id": oid})] if peak else \
            [ver, ("check_return_eligibility", {"order_id": oid, "order_item_id": item})]
        f, _ = _episode(tmp / "traces", t, other, world=pre_v19_patch(t), run_id="X/full")
        _as_v18(f, passed=False, failures=["missing required action: get_order"])
        f, _ = _episode(tmp / "traces", t, named, world=pre_v19_patch(t), run_id="X/window8")
        _as_v18(f)
    rows = [{"strategy": s, "cell": s, "pass^1": p, "pass^k": p, "mean_tokens": 100,
             "per_trap": {}, "per_language": {}}
            for s, p in (("full", 0.0), ("window8", 1.0), ("window4", 0.5))]
    (tmp / "traces" / "X" / "summary.json").write_text(json.dumps(rows))
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        diagnose.use_checker("current")
        md = mr.section_context([Path("traces/X")], "X")
        diagnose.use_checker("recorded")
        old = mr.section_context([Path("traces/X")], "X")
    finally:
        diagnose.use_checker("recorded")
        os.chdir(cwd)
    check("section 1 under today's checks: the two arms tie",
          "`window8` vs `full`" in md and "| `window8` vs `full` | +1.000; 0 worse, "
          f"{len(tasks)} better, p=0.000 | +0.000; 0 worse, 0 better, p=1.000 |" in md, md)
    check("…and says which conclusion the checker change moved",
          "The checker change moves a conclusion here:** `window8` vs `full`: better "
          "(p=0.000) recorded, inconclusive (p=1.000) now" in md, md)
    check("an arm with no episodes is named, not passed off as re-scored",
          "Not re-scored: `window4`" in md, md)
    check("--checker recorded: neither note", "verdicts the run recorded" not in old
          and "Not re-scored" not in old and "moves a conclusion" not in old, old)


def test_judges_against_todays_verifier():
    """The judge files carry the verdicts recorded when the judges ran. Under
    today's checker the ground truth is re-scored, and the payload views built
    from salted tracking numbers (#33) are counted, not left to prose."""
    print("\n=== judges against the verifier: today's ground truth, #33 flagged ===")
    mr = _report()
    from pasarbench import diagnose
    from pasarbench.generate import generate
    from pasarbench.harness.replay import pre_v19_patch
    from pasarbench.run import SOLUTIONS
    from tests.test_rescore import _as_v18, _episode
    gen, sol = generate()
    photo = next(t for t in gen if t.trap == "high_value_photo_required_first")
    che = next(t for t in gen if t.task_id == "CHE-MY")
    tmp = Path(tempfile.mkdtemp())
    cell = tmp / "traces" / "I-x" / "full+search-300"
    ver = ("verify_identity", {"user_id": photo.user_id,
                               "phone_last4": photo.hidden_facts["phone_last4"]})
    f1, _ = _episode(tmp / "traces", photo, [ver, ("get_order", {"order_id":
                     photo.hidden_facts["order_id"]})], world=pre_v19_patch(photo),
                     run_id="I-x/full+search-300")
    _as_v18(f1, passed=False, failures=["missing required action: check_return_eligibility"])
    f2, _ = _episode(tmp / "traces", che, sol["CHE-MY"], world=pre_v19_patch(che),
                     run_id="I-x/full+search-300")
    _as_v18(f2)
    jv = tmp / "data" / "judge_vs_verifier"
    jv.mkdir(parents=True)
    rows = [{"transcript_id": f.stem, "task_id": f.stem.split("__")[0], "trap": t.trap,
             "verifier_passed": ok, "naive_ok": True, "dec_ok": True, "naive_score": 5,
             "false_claims": []} for f, t, ok in ((f1, photo, False), (f2, che, True))]
    (jv / "I-x__full+search-300__payloads.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        diagnose.use_checker("current")
        md = mr.section_judge_vs_verifier(Path("traces"))
        diagnose.use_checker("recorded")
        old = mr.section_judge_vs_verifier(Path("traces"))
    finally:
        diagnose.use_checker("recorded")
        os.chdir(cwd)
    check("ground truth is re-scored, and the section says how much moved",
          "1 of the 2 episodes' verdicts differ from the ones the runs recorded" in md
          and "the verifier passed 2 and failed 0" in md, md[:900])
    check("…recorded ground truth under --checker recorded",
          "the verifier passed 1 and failed 1" in old and "differ from" not in old,
          old[:900])
    check("the salted tracking numbers are counted in the payload views",
          "1 of 2 episodes looked up a generated shipment" in md
          and "showed the judge a tracking number the agent never saw" in md, md[:1500])

    # A file written since v19 holds today's verdict and the recorded one, and
    # its payload views withhold what no replay can rebuild (#33): the note
    # must say so rather than warn about numbers the judge was never shown.
    rows = [{**r, "verifier_passed": True, "verifier_passed_recorded": r["verifier_passed"],
             "payload_calls": [1, 2] if r["transcript_id"] == f2.stem else [2, 2]}
            for r in rows]
    (jv / "I-x__full+search-300__payloads.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    try:
        os.chdir(tmp)
        diagnose.use_checker("current")
        md = mr.section_judge_vs_verifier(Path("traces"))
        diagnose.use_checker("recorded")
        old = mr.section_judge_vs_verifier(Path("traces"))
    finally:
        diagnose.use_checker("recorded")
        os.chdir(cwd)
    check("a file holding both verdicts reads today's by default…",
          "the verifier passed 2 and failed 0" in md
          and "1 of the 2 episodes' verdicts differ" in md, md[:900])
    check("…and the recorded one under --checker recorded",
          "the verifier passed 1 and failed 1" in old, old[:900])
    check("a view that withheld the unrebuildable payload says so, instead of "
          "warning about a number it never showed",
          "These views withhold it" in md and "never saw" not in md, md[:1500])


def test_figure():
    """scripts/make_figure.py draws the README's figure from numbers it reads
    off the traces. The SVGs in docs/img must be the ones those numbers draw,
    say what the docs say, and not keep a title the numbers stop bearing out."""
    print("\n=== the results figure ===")
    import contextlib
    import io
    import re
    import xml.etree.ElementTree as ET
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("make_figure", root / "scripts" / "make_figure.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    img = root / "docs" / "img"
    numbers = json.loads((img / "results.json").read_text())

    tmp = Path(tempfile.mkdtemp())
    with contextlib.redirect_stdout(io.StringIO()):
        code = mf.main(["--numbers", str(img / "results.json"), "--out", str(tmp)])
    drawn = sorted(p.name for p in tmp.iterdir())
    sizes = {}
    for (mode, layout), name in mf.FILES.items():
        svg = ET.fromstring((tmp / name).read_text(encoding="utf-8"))
        sizes[name] = (int(svg.get("width")), int(svg.get("height"))) == mf.LAYOUTS[layout]["size"]
    check("redrawn from its numbers: four well-formed SVGs at their layouts' sizes, nothing else",
          code == 0 and drawn == sorted(mf.FILES.values()) and all(sizes.values()),
          f"{code} {drawn} {sizes}")
    stale = [n for n in mf.FILES.values()
             if (tmp / n).read_bytes() != (img / n).read_bytes()]
    check("the committed SVGs are the ones results.json draws", not stale,
          f"redraw with: python scripts/make_figure.py --numbers docs/img/results.json -- {stale}")

    wide = (img / "results.svg").read_text(encoding="utf-8")
    texts = " | ".join(re.findall(r"<text [^>]*>([^<]*)</text>", wide))
    quoted = ["44%", "80%", "38 tasks better, 4 worse", "p = 6 × 10⁻⁸", "of 96 per arm · p = 0.06",
              "| 5 |", "| 0 |", "+0.3", "−6.6", "fine-tuned p = 0.84, reference p = 0.004",
              "+7.5", "every p ≥ 0.12, in two independent runs"]
    check("it shows the numbers the README and the blog quote",
          all(q in texts for q in quoted), str([q for q in quoted if q not in texts]))
    check("…and its description carries them for a screen reader",
          all(q in wide.split("</desc>")[0] for q in ("44% to 80%", "5 to 0", "+2.3 to +0.3",
                                                      "+4.0 to −6.6", "at best −4.7")),
          wide.split("</desc>")[0][-600:])

    ok_ink, ok_marks = True, True
    for (mode, layout), name in mf.FILES.items():
        t = mf.THEMES[mode]
        svg = (img / name).read_text(encoding="utf-8")
        ok_ink &= set(re.findall(r'<text [^>]*fill="([^"]+)"', svg)) <= {t["ink"], t["ink2"], t["muted"]}
        marks = re.findall(r"<(path|circle) [^>]*>(.*?)</\1>", svg)
        ok_marks &= bool(marks) and all(inner.startswith("<title>") for _, inner in marks)
    check("text wears text colours, never a series colour, in all four", ok_ink)
    check("every bar and dot carries its value as a tooltip", ok_marks)
    # The reference palette's own steps (the dataviz skill's palette.md): the
    # dark accent and surface are stepped for the dark surface, not reused.
    dark = (img / "results-dark.svg").read_text(encoding="utf-8")
    light_only = ("#fcfcfb", "#0b0b0b", "#e1e0d9", "#2a78d6")
    check("dark is its own palette, with no light-mode colour left in it",
          all(c in dark for c in ("#1a1a19", "#3987e5"))
          and not any(c in dark for c in light_only),
          str([c for c in light_only if c in dark]))

    check("the tests behind the panels: exact sign test and Fisher test",
          abs(mf._sign_p(4, 38) - numbers["naming"]["p"]) < 1e-12
          and abs(mf._fisher_p(5, 96, 0, 96) - numbers["guardrail"]["p"]) < 1e-12
          and mf._sign_p(0, 0) == 1.0 and abs(mf._fisher_p(0, 9, 0, 9) - 1.0) < 1e-12,
          f"{mf._sign_p(4, 38)} {mf._fisher_p(5, 96, 0, 96)}")
    shown = [mf._p(5.65e-8), mf._p(9.6e-4), mf._p(0.00366), mf._p(0.0593), mf._signed(-6.6),
             mf._signed(0.04), mf._signed(2.33)]
    check("p and signs print as a reader expects: a real exponent, a real minus",
          shown == ["p = 6 × 10⁻⁸", "p = 1 × 10⁻³", "p = 0.004", "p = 0.06", "−6.6",
                    "0.0", "+2.3"], str(shown))

    moved = json.loads(json.dumps(numbers))
    moved["language"]["D-nozh/full"]["th"]["p"] = 0.02
    moved["finetune"]["fixed"]["p_rft"] = 0.01
    (tmp / "moved.json").write_text(json.dumps(moved))
    out_dir = tmp / "moved"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = mf.main(["--numbers", str(tmp / "moved.json"), "--out", str(out_dir)])
    said = buf.getvalue()
    check("a title the numbers stop bearing out is not drawn under the old words",
          code == 1 and not out_dir.exists() and "No language differed" in said
          and "checker quirk" in said and mf.unsupported(numbers) == [], said)

    pages = {"README.md": "docs/img/", "docs/BLOG.md": "img/"}
    linked = {page: all(f'srcset="{pre}{n}"' in (root / page).read_text(encoding="utf-8")
                        or f'src="{pre}{n}"' in (root / page).read_text(encoding="utf-8")
                        for n in mf.FILES.values())
              for page, pre in pages.items()}
    check("README and the blog offer all four, by paths that resolve from each page",
          all(linked.values()) and all((root / page).parent.joinpath(pre, n).is_file()
                                       for page, pre in pages.items() for n in mf.FILES.values()),
          str(linked))


def test_hard_edition_section():
    """Section 8 (v60): RUNBOOK 1s on several agents, read as fixed before the
    runs -- each agent's cases against their parts, paired, in its words; the
    cases' table; every pair of agents on the cases and on the parts' whole run,
    and which of them the hard edition separates where the parts do not; the
    customer and the voucher choice; one customer and temperature 0."""
    print("\n=== section 8: the hard edition against its parts, several agents ===")
    mr = _report()
    from pasarbench.generate import EDITION3_PARTS, generate, part_of
    from pasarbench.rescore import all_tasks
    from pasarbench.simqa import CHANGED_REQUEST
    tasks = all_tasks()
    cases = [t for t in generate(edition=3)[0] if t.trap in EDITION3_PARTS]
    twins = {t.task_id: part_of(t) for t in cases}
    tmp = Path(tempfile.mkdtemp())
    v3 = "llm-user/v3:openai-compat:qwen3.8-flash"

    def run(name, model, start, case_ok, part_ok):
        for cell, ids, ok in ((f"V-{name}", [t.task_id for t in cases], case_ok),
                              (f"V-{name}-parts", sorted(set(twins.values())), part_ok)):
            d = tmp / "traces" / cell / "full"
            d.mkdir(parents=True, exist_ok=True)
            for tid in ids:
                t = tasks[tid]
                for k in range(3):
                    head = {"type": "header", "task_id": tid, "language": t.language,
                            "trap": t.trap, "requested_model": model, "simulator": v3,
                            "agent_temperature": 0.0, "started_at": start,
                            "agent_extra_body": {"thinking": {"type": "disabled"}}}
                    said = ([{"type": "event", "kind": "user_turn", "text":
                              f"{t.hidden_facts.get('phone_last4', '0000')}. Actually don't "
                              f"cancel it, send it to my new address instead."}]
                            if t.trap == CHANGED_REQUEST else [])
                    foot = {"type": "footer", "passed": ok(t), "stop_reason": "done",
                            "budget": {"tokens": 1000, "steps": 5}}
                    (d / f"{tid}__r{k}.jsonl").write_text(
                        "\n".join(json.dumps(r) for r in (head, *said, foot)) + "\n")

    def section():
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            return mr.section_hard_edition(Path("traces"))
        finally:
            os.chdir(cwd)

    md = section()
    check("before any agent has run both: what is missing and the command",
          "**NOT MEASURED**" in md and "scripts/add_model.py" in md, md)
    every = lambda t: True                                                   # noqa: E731
    run("x", "model-x", "2026-10-13T01:00", every, every)
    run("y", "model-y", "2026-10-13T02:00", every,
        lambda t: t.trap != "duplicate_refund_escalate")
    run("z", "model-z", "2026-10-13T03:00", lambda t: t.trap != "duplicate_refund_cod_livestream",
        every)
    md = section()
    check("each agent's cases against their parts, paired, in the words fixed before the "
          "run: EASIER where the parts fail more, HARDER where the cases do",
          "| x | `model-x` | `{\"thinking\": {\"type\": \"disabled\"}}` | 1.000 (1.00) | 1.000 "
          "(1.00) | **INCONCLUSIVE**" in md
          and "| y | `model-y` |" in md and "**EASIER** — +0.278; 0 cases failed more than "
              "their part, 10 less, p=0.002" in md
          and "**HARDER** — -0.278; 10 cases failed more than their part, 0 less, p=0.002"
              in md and "- **Harder:** shown for z." in md, md[:3000])
    check("…the cases' table, each cell the case's passes and its part's",
          "| `duplicate_refund_cod_livestream` (`duplicate_refund_escalate`) | 30/30 · 30/30 "
          "| 30/30 · 0/30 | 0/30 · 30/30 |" in md, md[md.find("| case"):][:800])
    check("…and every pair on the cases and on the parts' run: the hard edition separates "
          "x and z, which the parts do not, and the parts x and y, which the cases do not",
          "| x → y | **INCONCLUSIVE**" in md
          and "the parts separate them, the cases do not |" in md
          and "| x → z | **z worse** — -0.278; 0 tasks better, 10 worse, p=0.002 | "
              "**INCONCLUSIVE**" in md
          and "- **Separation:** the hard edition separates x–z, which the parts do not. "
              "The parts separate x–y, which the cases do not." in md,
          md[md.find("| agents"):][:1500])
    check("…the customer and the voucher choice said even when there is nothing to set "
          "aside, and one customer at temperature 0",
          "her phone digits came with her change of mind" in md
          and "she never declined it" in md
          and "- ✓ one customer in every run: `llm-user/v3:openai-compat:qwen3.8-flash`" in md
          and "- ✓ agent temperature 0 in every run" in md
          and "- ✓ each agent's two runs: one model, one setting, every case's part" in md,
          md[md.find("*The customer"):])


def main() -> int:
    test_context_verdicts_need_a_paired_test()
    test_ties_are_said_out_loud()
    test_replication_table()
    test_calibration_without_variance()
    test_attribution_needs_a_resolved_gap()
    test_noise_floor()
    test_post_training()
    test_planted_mistakes_section()
    test_grpo_section()
    test_edition3_section()
    test_hard_edition_section()
    test_both_scorings()
    test_ablation_sections_under_both_checkers()
    test_judges_against_todays_verifier()
    test_figure()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
