"""
Traces.

Every number you will eventually put in a table comes from here. Write the
trace first and the analysis later, never the other way round -- an episode you
did not record is an episode you cannot explain, and "why did it fail?" is the
question that produces findings.

One JSONL file per run. Line 1 is a header, then one line per step, then a
footer with the verdict. Grep-able, streamable, diff-able between runs.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, root: str | Path = "traces", run_id: str | None = None):
        self.root = Path(root)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._path: Path | None = None
        self._t0 = 0.0

    def open_episode(self, task_id: str, meta: dict[str, Any] | None = None,
                     run_index: int | None = None, **kw: Any) -> None:
        """`run_index` distinguishes repeat seeds of the same task. Without it,
        k>1 overwrites its own traces and you analyse one episode in k."""
        meta = {**(meta or {}), **kw}
        self.close()
        name = task_id if run_index is None else f"{task_id}__r{run_index}"
        self._path = self.dir / f"{name}.jsonl"
        self._fh = self._path.open("w", encoding="utf-8")
        self._t0 = time.monotonic()
        self._write({"type": "header", "task_id": task_id,
                     "run_index": run_index,
                     "started_at": datetime.now(timezone.utc).isoformat(), **meta})

    def step(self, record: dict[str, Any]) -> None:
        self._write({"type": "step", **record})

    def event(self, kind: str, **payload: Any) -> None:
        self._write({"type": "event", "kind": kind, **payload})

    def messages(self, messages: list[dict[str, Any]], **payload: Any) -> None:
        """The conversation exactly as the agent saw it, for training.

        Traces otherwise keep which tools were called, not what came back
        (#23), which is enough to audit an episode and not enough to train on
        it: a training example is the prompt the model saw, token for token.
        Written only when a sweep asks for it (--save-messages)."""
        self._write({"type": "messages", "messages": messages, **payload})

    def close_episode(self, stop_reason: str, passed: bool | None = None,
                      failures: list[str] | None = None,
                      budget: dict[str, Any] | None = None) -> None:
        self._write({"type": "footer", "stop_reason": stop_reason,
                     "passed": passed, "failures": failures or [],
                     "budget": budget or {},
                     "wall_s": round(time.monotonic() - self._t0, 3)})
        self.close()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def _write(self, obj: dict[str, Any]) -> None:
        if self._fh:
            self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()


class NullTrace:
    """No-op, for unit tests and quick loops."""

    run_id = "null"
    dir = Path(".")

    def open_episode(self, task_id: str, meta: dict[str, Any] | None = None,
                     run_index: int | None = None, **kw: Any) -> None: ...
    def step(self, record: dict[str, Any]) -> None: ...
    def event(self, kind: str, **payload: Any) -> None: ...
    def messages(self, messages: list[dict[str, Any]], **payload: Any) -> None: ...
    def close_episode(self, *a: Any, **kw: Any) -> None: ...
    def close(self) -> None: ...


def read_episode(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def customer_saw(step: dict[str, Any]) -> bool:
    """Whether a step's text counts as said to the customer: not held back by
    the claim guardrail (guardrail.py), and not written after the customer had
    left (the closing check, closing.py). Its tool calls ran either way.

    Not quite "reached": text the agent writes beside a tool call is never
    shown to the simulated customer either (simulator.py), and the readers
    have always counted it. Leaving it out changes no false-claim count in any
    run recorded up to v28 -- checked on all of them -- so it is kept, and
    this reads as it always has for every step but these two kinds."""
    return not step.get("guardrail") and not step.get("closing")


def summarise_run(run_dir: str | Path) -> dict[str, Any]:
    """Fold a whole run directory into headline numbers. Cheap enough to call
    after every sweep; keep the output next to the traces."""
    run_dir = Path(run_dir)
    episodes, passed, tokens, steps, stops = 0, 0, 0, 0, {}
    for f in sorted(run_dir.glob("*.jsonl")):
        recs = read_episode(f)
        foot = next((r for r in reversed(recs) if r.get("type") == "footer"), None)
        if not foot:
            continue
        episodes += 1
        passed += bool(foot.get("passed"))
        b = foot.get("budget") or {}
        tokens += b.get("tokens", 0)
        steps += b.get("steps", 0)
        stops[foot["stop_reason"]] = stops.get(foot["stop_reason"], 0) + 1
    return {"episodes": episodes, "passed": passed,
            "pass_rate": round(passed / episodes, 4) if episodes else 0.0,
            "mean_tokens": round(tokens / episodes, 1) if episodes else 0,
            "mean_steps": round(steps / episodes, 2) if episodes else 0,
            "stop_reasons": stops}
