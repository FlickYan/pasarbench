"""
Re-score recorded episodes against today's checks.

A trace holds every tool call an episode made. The environment is
deterministic, so replaying those calls against the world the episode ran in
rebuilds its final database (harness/replay.py), and today's verifier scores
that state. No model is called and nothing is re-run: this costs seconds.

Why it exists. v19 found that two checks demanded one specific lookup tool
where the policy only needs the fact the lookup establishes (WHAT_FAILED #30).
Fixing a check changes what every recorded verdict means, and re-running
thousands of episodes to find out would cost what the runs cost. Re-scoring
makes a checker change cheap to measure -- and the report shows both scorings,
so a change to the rules can never quietly move a headline number.

What is trusted. Every replayed call is compared with the recording: its
success, and its payload, by digest from v19 and by length before. If either
differs, the world or the state is not the one the episode had, and the
episode keeps its recorded verdict, counted as not re-scorable: never
re-scored on a guess. So does an episode whose task has changed since, no
longer exists, or never finished. A task's change shows in the task_digest a
trace records; a trace older than the digest is ruled out if a call names a
generated id today's world lacks -- run B's calls name orders from before the
id change of #9, and an escalation carrying one replays at the same length, so
payloads alone would let it through. The one payload let through unchecked is
a pre-v19 tracking number, which nothing can rebuild (#33) and no check
reads.

    python scripts/rescore.py                     # every run under traces/
    python scripts/rescore.py traces/P-base --changed
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tasks import task_digest
from .verifier import verify

_TASKS: dict[str, Any] | None = None
_CACHE: dict[tuple[str, int, int], "Rescored"] = {}


def all_tasks() -> dict[str, Any]:
    """Every task today's code defines, hand-written and generated, by id."""
    global _TASKS
    if _TASKS is None:
        from .generate import generate
        from .tasks import TASKS
        gen, _ = generate()
        _TASKS = {t.task_id: t for t in list(TASKS) + gen}
    return _TASKS


@dataclass
class Rescored:
    transcript_id: str
    task_id: str
    trap: str
    language: str
    recorded: bool | None                 # the verdict in the trace's footer
    current: bool | None                  # today's checks on the rebuilt state
    status: str   # rescored | diverged | world_changed | task_changed | unknown_task | unfinished
    failures_recorded: list[str] = field(default_factory=list)
    failures_current: list[str] = field(default_factory=list)
    detail: str = ""
    payloads: tuple[int, int] = (0, 0)    # replayed payloads that match the recording, of all

    @property
    def verdict(self) -> bool:
        """What to count: today's verdict where the episode could be rebuilt,
        else the recorded one."""
        return bool(self.current) if self.current is not None else bool(self.recorded)

    @property
    def failures(self) -> list[str]:
        return self.failures_current if self.current is not None else self.failures_recorded

    @property
    def moved(self) -> str:
        """"fail→pass", "pass→fail" or ""."""
        if self.current is None or self.recorded is None or self.current == bool(self.recorded):
            return ""
        return "fail→pass" if self.current else "pass→fail"

    def to_dict(self) -> dict[str, Any]:
        return {"transcript_id": self.transcript_id, "task_id": self.task_id,
                "trap": self.trap, "language": self.language,
                "recorded": self.recorded, "current": self.current,
                "status": self.status, "moved": self.moved,
                "failures_recorded": self.failures_recorded,
                "failures_current": self.failures_current, "detail": self.detail,
                "payloads": list(self.payloads)}


def rescore_records(recs: list[dict[str, Any]], transcript_id: str = "",
                    tasks: dict[str, Any] | None = None) -> Rescored:
    """One episode's parsed trace records, scored by today's checks."""
    from .harness.replay import replay
    tasks = tasks if tasks is not None else all_tasks()
    head = next((r for r in recs if r.get("type") == "header"), {})
    foot = next((r for r in reversed(recs) if r.get("type") == "footer"), None)
    tid = head.get("task_id") or transcript_id.split("__")[0]
    base = dict(transcript_id=transcript_id, task_id=tid,
                trap=head.get("trap", "?"), language=head.get("language", "?"),
                recorded=None if foot is None else bool(foot.get("passed")),
                failures_recorded=(foot or {}).get("failures") or [])
    if foot is None:
        return Rescored(**base, current=None, status="unfinished",
                        detail="no footer: the episode never finished")
    task = tasks.get(tid)
    if task is None:
        return Rescored(**base, current=None, status="unknown_task",
                        detail=f"no task {tid!r} in today's suite")
    if head.get("task_digest") and head["task_digest"] != task_digest(task):
        return Rescored(**base, current=None, status="task_changed",
                        detail="the task's opening or facts changed since the run")
    if not head.get("task_digest"):
        # Older than the digest (#26). Run B's calls name the orders of a
        # generator that has since changed its ids (#9): an escalation carries
        # its order id and replays at the same length, so payloads alone pass
        # it, and today's check -- which looks for today's id -- would fail an
        # episode that passed. Any generated id today's world lacks rules the
        # episode out.
        from .db import Database
        known = {x for rows in Database.fresh(task.db_patch).tables.values()
                 for rid, row in rows.items()
                 for x in (rid, *(v for v in row.values() if isinstance(v, str)))}
        stale = sorted({v for st in recs if st.get("type") == "step"
                        for tc in st.get("tool_calls") or []
                        for v in (tc.get("arguments") or {}).values()
                        if isinstance(v, str) and re.fullmatch(r"G[A-Z]-\w+", v)
                        and v not in known})
        if stale:
            return Rescored(**base, current=None, status="task_changed",
                            detail=f"calls name {stale[0]}, which today's world does not "
                                   f"have: the trace predates today's task")
    r = replay(recs, task)
    if r.diverged or r.mismatched:
        kind, first = (("diverged", r.diverged[0]) if r.diverged
                       else ("world_changed", r.mismatched[0]))
        return Rescored(**base, current=None, status=kind,
                        payloads=(r.verified, r.total),
                        detail=f"{len(r.diverged) + len(r.mismatched)} call(s) replay "
                               f"differently, first: {first}")
    v = verify(task, r.db)
    return Rescored(**base, current=v.passed, status="rescored",
                    failures_current=v.failures, payloads=(r.verified, r.total),
                    detail=f"{r.total} calls, world {r.world}"
                           + (f", {len(r.unrecoverable)} tracking number(s) not "
                              f"recoverable (#33)" if r.unrecoverable else ""))


def rescore_file(path: str | Path, tasks: dict[str, Any] | None = None) -> Rescored:
    """One trace file, cached on its path, size and modification time."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if tasks is None and key in _CACHE:
        return _CACHE[key]
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    out = rescore_records(recs, path.stem, tasks)
    if tasks is None:
        _CACHE[key] = out
    return out


def rescore_dir(cell: str | Path, tasks: dict[str, Any] | None = None) -> list[Rescored]:
    """Every episode in one cell directory (traces/<run>/<cell>)."""
    return [rescore_file(f, tasks) for f in sorted(Path(cell).glob("*.jsonl"))]


def cells(root: str | Path = "traces") -> list[Path]:
    """Every cell under a traces root, a run, or a cell itself: each directory
    one or two levels down that holds traces. (Only one level was searched when
    it found any, so a stray folder of traces beside the runs hid every run.)"""
    root = Path(root)
    if any(root.glob("*.jsonl")):
        return [root]
    return sorted({p for pat in ("*", "*/*") for p in root.glob(pat)
                   if p.is_dir() and any(p.glob("*.jsonl"))})


def summary(rows: list[Rescored]) -> dict[str, Any]:
    """Headline counts for one cell: both scorings, and what moved."""
    done = [r for r in rows if r.recorded is not None]
    n = len(done) or 1

    def passk(get) -> float:
        by: dict[str, list[bool]] = {}
        for r in done:
            by.setdefault(r.task_id, []).append(get(r))
        return sum(all(v) for v in by.values()) / (len(by) or 1)

    return {
        "episodes": len(done),
        "recorded": sum(bool(r.recorded) for r in done) / n,
        "current": sum(r.verdict for r in done) / n,
        "recorded_k": passk(lambda r: bool(r.recorded)),
        "current_k": passk(lambda r: r.verdict),
        "fail_to_pass": sum(r.moved == "fail→pass" for r in done),
        "pass_to_fail": sum(r.moved == "pass→fail" for r in done),
        "not_rescorable": sum(r.current is None for r in done),
        "unfinished": len(rows) - len(done),
        "payloads": (sum(r.payloads[0] for r in done), sum(r.payloads[1] for r in done)),
    }
