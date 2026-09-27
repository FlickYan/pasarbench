"""
Train/eval folds for post-training.

    python -m pasarbench.rl.split --out data/splits/folds.json

THE UNIT OF SPLITTING IS THE FAMILY, NOT THE TASK
-------------------------------------------------
`HRWR-ID` and `HRWR-ID.id` are locale twins: the same world, the same checks,
the same correct actions, in two languages. Split by task id and one twin can
land in training while the other is evaluated -- the model is then tested on a
translation of an example it was trained on, and the number measures memory.
A family is every task that shares a world: the id before the first ".".

TWO FOLDS, AND EVERY TASK IS EVALUATED ONCE
-------------------------------------------
One model is trained on fold A and evaluated on fold B; another is trained on
fold B and evaluated on fold A. Together they put all 215 tasks in the held-out
set, paired by task against the base model -- the tool-scaling arms had 32 tasks
each, and 32 is why so many of their verdicts were INCONCLUSIVE.

Families are dealt into folds trap by trap, so both folds see every trap. What
the held-out number measures is therefore "new worlds, markets and languages
for policy situations the model has seen", not "policy situations it has never
seen". Say that in the writeup; a leave-traps-out split is a different, harder
question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FOLDS = ("A", "B")


def family_of(task_id: str) -> str:
    """Every task sharing a world: `HRWR-ID.id` -> `HRWR-ID`, `T01` -> `T01`."""
    return task_id.split(".")[0]


def make_folds(tasks, seed: int = 0) -> dict[str, Any]:
    """Deal families into two folds, trap by trap, reproducibly.

    Within each trap the families are shuffled with a seed derived from the
    trap name, then dealt alternately. The fold that receives a trap's first
    family alternates from trap to trap, so an odd family count per trap does
    not always land its extra family in the same fold."""
    from ..tasks import task_digest

    fam_trap: dict[str, str] = {}
    for t in tasks:
        f = family_of(t.task_id)
        if fam_trap.setdefault(f, t.trap) != t.trap:
            raise ValueError(f"family {f} spans two traps: {fam_trap[f]} and {t.trap}")

    by_trap: dict[str, list[str]] = defaultdict(list)
    for f, trap in fam_trap.items():
        by_trap[trap].append(f)

    family_fold: dict[str, str] = {}
    for i, trap in enumerate(sorted(by_trap)):
        fams = sorted(by_trap[trap])
        rnd = random.Random(f"{seed}:{trap}")
        rnd.shuffle(fams)
        for j, f in enumerate(fams):
            family_fold[f] = FOLDS[(i + j) % 2]

    task_fold = {t.task_id: family_fold[family_of(t.task_id)] for t in tasks}
    digests = {t.task_id: task_digest(t) for t in tasks}
    return {
        "seed": seed,
        "unit": "family (task id before the first '.')",
        "folds": list(FOLDS),
        "task_fold": dict(sorted(task_fold.items())),
        "task_digest": dict(sorted(digests.items())),
        # One hash over the whole assignment, so a report can say which split
        # it was computed on and two runs can be checked for the same split.
        "split_digest": hashlib.sha1(json.dumps(sorted(task_fold.items()))
                                     .encode()).hexdigest()[:12],
    }


def load_folds(path: str | Path, tasks=None) -> dict[str, Any]:
    """Read a folds file and, given tasks, refuse it if any task has changed
    since the split was made -- a split over yesterday's tasks silently
    evaluates today's model on something else."""
    folds = json.loads(Path(path).read_text())
    if tasks is not None:
        from ..tasks import task_digest
        known = folds["task_digest"]
        changed = [t.task_id for t in tasks
                   if t.task_id in known and known[t.task_id] != task_digest(t)]
        missing = [t.task_id for t in tasks if t.task_id not in known]
        if changed or missing:
            raise SystemExit(
                f"{path} does not describe these tasks: {len(changed)} changed "
                f"since the split was made, {len(missing)} not in it "
                f"(e.g. {(changed or missing)[:3]}). Re-make the split and "
                f"re-train; never evaluate across a changed task set.")
    return folds


def other_fold(fold: str) -> str:
    return FOLDS[1 - FOLDS.index(fold)]


def summary(folds: dict[str, Any], tasks) -> str:
    tf = folds["task_fold"]
    by_trap: dict[str, Counter] = defaultdict(Counter)
    by_lang: dict[str, Counter] = defaultdict(Counter)
    for t in tasks:
        by_trap[t.trap][tf[t.task_id]] += 1
        by_lang[t.language][tf[t.task_id]] += 1
    n = Counter(tf.values())
    fam_n = Counter({f: tf[t.task_id] for t in tasks
                     for f in [family_of(t.task_id)]}.values())
    out = [f"split {folds['split_digest']} (seed {folds['seed']}): "
           f"A = {n['A']} tasks in {fam_n['A']} families, "
           f"B = {n['B']} tasks in {fam_n['B']} families",
           "", f"{'trap':40s}   A    B"]
    for trap in sorted(by_trap):
        c = by_trap[trap]
        out.append(f"{trap:40s} {c['A']:3d}  {c['B']:3d}")
    out += ["", f"{'language':40s}   A    B"]
    for lang in sorted(by_lang):
        c = by_lang[lang]
        out.append(f"{lang:40s} {c['A']:3d}  {c['B']:3d}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/splits/folds.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ..generate import generate
    from ..tasks import TASKS
    tasks = list(TASKS) + list(generate()[0])
    folds = make_folds(tasks, seed=args.seed)
    out = Path(args.out)
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("split_digest") != folds["split_digest"]:
            raise SystemExit(
                f"{out} already holds a different split ({old.get('split_digest')}). "
                f"Models trained on it would be evaluated against the wrong "
                f"held-out set. Move it aside deliberately if you mean to re-split.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(folds, indent=1))
    print(summary(folds, tasks))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
