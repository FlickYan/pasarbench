"""
Trajectory collection and dataset construction.

Three products, in ascending order of difficulty and descending order of
return on effort:

  RFT / rejection-sampling SFT
      Roll out k times, keep what the verifier says passed, fine-tune on it.
      No RL machinery at all. On a verifiable environment this usually
      recovers most of the achievable gain, and it is what you should ship
      first. If someone asks why you did not start with GRPO, this is the
      answer: RFT is the baseline GRPO has to beat, and often does not by much.

  DPO
      Needs pairs. Full-trajectory pairs are easy and blunt. Step-level pairs
      at the FIRST DIVERGENCE are sharper -- both trajectories share an
      identical prefix, so the gradient is about the one decision that
      actually differed rather than about everything downstream of it. Both
      are implemented; prefer divergence pairs.

  GRPO / RLVR
      Group-relative advantage straight off the verifier, no critic. See
      reward.py. Only worth it once RFT has plateaued.

Everything here reads from rollouts you already have. Traces are the source of
truth; nothing is regenerated for training that was not scored first.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..db import Database
from ..harness.loop import run_episode
from ..harness.types import EpisodeResult, Message
from ..tasks import Task
from ..verifier import Result, verify
from .reward import RewardBreakdown, RewardConfig, compute_reward, group_advantages


@dataclass
class Rollout:
    task_id: str
    trap: str
    market: str
    language: str
    messages: list[Message]
    result: Result
    reward: RewardBreakdown
    steps: int
    tokens: int
    stop_reason: str
    seed: int = 0

    @property
    def passed(self) -> bool:
        return self.result.passed

    @property
    def action_signature(self) -> str:
        """Identity of the trajectory by its tool-call sequence. Used to
        deduplicate RFT data -- 40 rollouts of an easy task collapse to two or
        three distinct solutions, and training on 40 copies of the same one
        just sharpens a mode you already had."""
        calls = []
        for m in self.messages:
            for tc in m.tool_calls:
                calls.append(f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)})")
        return hashlib.sha1("|".join(calls).encode()).hexdigest()[:16]


def rollout_task(task: Task, backend_factory: Callable[[Task], Any], k: int = 8,
                 cfg: RewardConfig | None = None,
                 reference_steps: int | None = None, **episode_kw) -> list[Rollout]:
    """k independent rollouts of one task. Temperature must be > 0 upstream or
    every rollout is identical and the group carries no signal."""
    out: list[Rollout] = []
    for seed in range(k):
        db = Database.fresh(task.db_patch)
        res: EpisodeResult = run_episode(task, db, backend_factory(task), **episode_kw)
        v = verify(task, db, n_turns=res.budget["steps"], tokens=res.budget["tokens"])
        r = compute_reward(task, db, res.budget["steps"], reference_steps, cfg, v)
        out.append(Rollout(task.task_id, task.trap, task.market, task.language,
                           res.state.messages, v, r, res.budget["steps"],
                           res.budget["tokens"], res.stop_reason.value, seed))
    return out


# --------------------------------------------------------------------------
# RFT / SFT
# --------------------------------------------------------------------------

def sft_examples(rollouts: Iterable[Rollout], dedupe: bool = True,
                 max_per_task: int = 4) -> list[dict[str, Any]]:
    """Passing trajectories only, deduplicated by tool-call signature.

    Format is a plain message list. THE TRAINER MUST MASK LOSS TO ASSISTANT
    TOKENS ONLY. Training on tool observations teaches the model to hallucinate
    tool outputs, which is the single most common way agent SFT goes wrong and
    it is invisible until the model starts inventing order records.
    """
    seen: set[tuple[str, str]] = set()
    per_task: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []

    for r in sorted(rollouts, key=lambda x: (-x.reward.reward, x.steps)):
        if not r.passed:
            continue
        key = (r.task_id, r.action_signature)
        if dedupe and key in seen:
            continue
        if per_task[r.task_id] >= max_per_task:
            continue
        seen.add(key)
        per_task[r.task_id] += 1
        out.append({
            "task_id": r.task_id,
            "trap": r.trap,
            "market": r.market,
            "language": r.language,
            "reward": r.reward.reward,
            "steps": r.steps,
            "messages": [m.to_dict() for m in r.messages],
        })
    return out


# --------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------

def _assistant_steps(messages: list[Message]) -> list[tuple[int, Message]]:
    return [(i, m) for i, m in enumerate(messages) if m.role == "assistant"]


def _step_key(m: Message) -> str:
    if m.tool_calls:
        return "|".join(f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}"
                        for tc in m.tool_calls)
    return f"TEXT:{(m.content or '')[:80]}"


def divergence_pairs(rollouts: list[Rollout], max_pairs_per_task: int = 3
                     ) -> list[dict[str, Any]]:
    """Pair a passing and a failing rollout at their FIRST differing decision.

    The shared prefix becomes the prompt, so the preference is about one
    action rather than about an entire trajectory. Pairs whose divergence is
    only a difference in free text (not tool calls) are dropped -- phrasing
    preferences are noise here and will drift the model's tone without
    improving task success.
    """
    by_task: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_task[r.task_id].append(r)

    pairs: list[dict[str, Any]] = []
    for task_id, rs in by_task.items():
        good = sorted([r for r in rs if r.passed], key=lambda x: x.steps)
        bad = sorted([r for r in rs if not r.passed], key=lambda x: x.reward.reward)
        made = 0
        for g in good:
            for b in bad:
                if made >= max_pairs_per_task:
                    break
                gs, bs = _assistant_steps(g.messages), _assistant_steps(b.messages)
                idx = None
                for n in range(min(len(gs), len(bs))):
                    if _step_key(gs[n][1]) != _step_key(bs[n][1]):
                        idx = n
                        break
                if idx is None:
                    continue
                g_msg, b_msg = gs[idx][1], bs[idx][1]
                if not g_msg.tool_calls and not b_msg.tool_calls:
                    continue          # text-only divergence: skip
                prefix = g.messages[:gs[idx][0]]
                pairs.append({
                    "task_id": task_id,
                    "trap": g.trap,
                    "divergence_step": idx,
                    "prompt": [m.to_dict() for m in prefix],
                    "chosen": g_msg.to_dict(),
                    "rejected": b_msg.to_dict(),
                    "reward_gap": round(g.reward.reward - b.reward.reward, 4),
                    "rejected_failure": (b.result.failures or ["unknown"])[0],
                })
                made += 1
    return pairs


def trajectory_pairs(rollouts: list[Rollout], max_pairs_per_task: int = 2
                     ) -> list[dict[str, Any]]:
    """Blunter full-trajectory preferences. Use only when divergence pairs are
    too scarce -- for example on tasks the policy never solves."""
    by_task: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_task[r.task_id].append(r)
    pairs = []
    for task_id, rs in by_task.items():
        good = [r for r in rs if r.passed]
        bad = sorted([r for r in rs if not r.passed], key=lambda x: x.reward.reward)
        for g, b in zip(good, bad):
            pairs.append({
                "task_id": task_id, "trap": g.trap,
                "prompt": [m.to_dict() for m in g.messages[:2]],
                "chosen": [m.to_dict() for m in g.messages[2:]],
                "rejected": [m.to_dict() for m in b.messages[2:]],
                "reward_gap": round(g.reward.reward - b.reward.reward, 4),
            })
            if len(pairs) >= max_pairs_per_task:
                break
    return pairs


# --------------------------------------------------------------------------
# GRPO groups
# --------------------------------------------------------------------------

def grpo_groups(rollouts: list[Rollout]) -> list[dict[str, Any]]:
    by_task: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_task[r.task_id].append(r)
    groups = []
    for task_id, rs in by_task.items():
        rewards = [r.reward.reward for r in rs]
        groups.append({
            "task_id": task_id,
            "trap": rs[0].trap,
            "rewards": rewards,
            "advantages": group_advantages(rewards),
            "pass_rate": round(sum(r.passed for r in rs) / len(rs), 3),
            "degenerate": len(set(rewards)) <= 1,
        })
    return groups


# --------------------------------------------------------------------------

def corpus_stats(rollouts: list[Rollout]) -> dict[str, Any]:
    """Read this before training anything.

    The number that decides whether RFT is viable is per-trap pass rate. A trap
    at 0.0 contributes NO SFT data, so fine-tuning cannot teach it -- it will
    still be broken after training and you need to say so rather than hide it
    behind an aggregate. A trap at 1.0 contributes data you did not need.
    Learning happens in the middle band.
    """
    by_trap: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_trap[r.trap].append(r)

    traps = {}
    for trap, rs in sorted(by_trap.items()):
        pr = sum(x.passed for x in rs) / len(rs)
        traps[trap] = {
            "n": len(rs), "pass_rate": round(pr, 3),
            "distinct_solutions": len({x.action_signature for x in rs if x.passed}),
            "usable_for_rft": 0.0 < pr < 1.0 or pr == 1.0,
            "unlearnable": pr == 0.0,
        }
    total = len(rollouts)
    return {
        "rollouts": total,
        "pass_rate": round(sum(r.passed for r in rollouts) / total, 4) if total else 0,
        "mean_reward": round(sum(r.reward.reward for r in rollouts) / total, 4) if total else 0,
        "gated_by_forbidden": sum(1 for r in rollouts if r.reward.gated),
        "traps_with_zero_signal": [t for t, v in traps.items() if v["unlearnable"]],
        "per_trap": traps,
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path
