"""
Agents that run through the harness.

Deliberately thin. The adapter's only job is to satisfy the same
`run(task, db) -> int` interface that ReferenceAgent already satisfies, so
every scoring path from week 1 keeps working unchanged. That is the payoff for
having defined the interface before building the loop.
"""

from __future__ import annotations

from typing import Any, Callable

from .db import Database
from .harness.backends import ScriptedBackend
from .harness.context import ContextStrategy, FullContext
from .harness.loop import restore, run_episode, snapshot
from .harness.simulator import ScriptedUser, SilentUser
from .harness.trace import NullTrace
from .harness.types import Budget, EpisodeResult


class HarnessAgent:
    """Wraps loop.run_episode so it can be scored by pasarbench.run."""

    def __init__(self, backend_factory: Callable[[Any], Any],
                 *, context: ContextStrategy | None = None,
                 budget: Budget | None = None,
                 simulator_factory: Callable[[Any], Any] | None = None,
                 policy_mode: str = "preload",
                 trace_factory: Callable[[], Any] | None = None,
                 tool_names: list[str] | None = None,
                 label: str = "harness"):
        self.backend_factory = backend_factory
        self.context = context or FullContext()
        self.budget = budget or Budget()
        self.simulator_factory = simulator_factory or (lambda t: SilentUser())
        self.policy_mode = policy_mode
        self.trace_factory = trace_factory or (lambda: NullTrace())
        self.tool_names = tool_names
        self.name = f"{label}[{self.context.name},{policy_mode}]"
        self.last: EpisodeResult | None = None
        self.trace = self.trace_factory()

    def run(self, task, db: Database) -> int:
        res = run_episode(
            task, db, self.backend_factory(task),
            simulator=self.simulator_factory(task),
            context=self.context,
            budget=self.budget,
            trace=self.trace,
            tool_names=self.tool_names,
            policy_mode=self.policy_mode,
        )
        self.last = res
        return res.budget["steps"]


def scripted_harness_agent(solutions: dict[str, list[tuple[str, dict]]],
                           **kw) -> HarnessAgent:
    """The reference solutions, driven through the real loop rather than
    around it. If this scores 1.0 the harness is wired correctly."""
    return HarnessAgent(
        backend_factory=lambda t: ScriptedBackend(solutions.get(t.task_id, [])),
        label="scripted-harness", **kw)


def run_with_interrupt(task, backend, *, at_step: int, **kw) -> tuple[EpisodeResult, str]:
    """Run until `at_step`, snapshot, then resume from the blob.

    Returns the final result and the intermediate snapshot, so a test can
    assert that interrupting changes nothing about the outcome.
    """
    db = Database.fresh(task.db_patch)
    first = run_episode(task, db, backend, interrupt_after_steps=at_step, **kw)
    blob = snapshot(first.state, db)

    state2, db2 = restore(blob)
    second = run_episode(task, db2, backend, state=state2, **kw)
    second.state.messages = second.state.messages  # explicit: state carried over
    return second, blob


def final_db_after(task, backend, **kw) -> Database:
    db = Database.fresh(task.db_patch)
    run_episode(task, db, backend, **kw)
    return db
