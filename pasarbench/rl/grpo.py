"""
GRPO on PasarBench, one agent turn at a time (docs/RUNBOOK.md, phase 4, GRPO).

One iteration:

  1. collect   the current policy rolls out k times at temperature 1 on each task
               it trains on, its conversations saved (scripts/train_grpo.py
               collect: the sweep, as RFT's collection);
  2. build     every episode is replayed and rewarded under today's checks
               (`episode_reward`), the rewards are standardised within each
               task's group (`reward.group_advantages`), and every agent turn of
               an episode in a group that is not degenerate becomes one example:
               the prompt as the serving engine built it, the turn as the model
               wrote it, and its episode's advantage (`build`);
  3. update    `train_rft.py train --objective grpo` takes a clipped
               policy-gradient pass over them (`grpo_loss`), starting from the
               last adapter.

Why per turn: for the reason RFT trains per turn (pasarbench/rl/sft.py). The
prompt of a turn is the conversation the model saw, token for token, from the
server's /tokenize; the turn is the only thing that gets a gradient; tool
results and the customer's words are context. That is the loss masking across
turns that multi-turn GRPO needs, with nothing to align by hand.

Every turn of an episode carries the episode's advantage: the reward is the
verifier's, given once at the end. A turn's loss is divided by the number of
tokens in all of its episode's turns together, so each episode weighs the same
however long it ran -- the length bias a sum of log-probabilities has.

A group teaches something only when its verdicts differ: some of its k
episodes passed and some did not. The reward also takes 0.005 off for each
step beyond the reference path, so k passing episodes of different lengths
have rewards that differ in the third decimal -- and standardising within the
group would turn that into advantages as large as a pass against a fail. Such
a group is degenerate here, as one with every episode failed is: the shaped
part of the reward orders the episodes inside a group that has a pass and a
fail, and decides nothing on its own (WHAT_FAILED #41).

The ratio is the trainer's own: the log-probabilities the turn had under the
adapter the iteration started from, computed by the same forward pass training
uses before the first step. No KL term: the adapter is the only thing that
moves, and the clip bounds each step.

The rollouts must come from the distribution the loss is over: the full softmax
at temperature 1. SGLang completes a request from the model's own
generation_config.json -- for Qwen3.8 a top-k of 20 and a top-p of 0.95 -- so
the collection names every sampling parameter (scripts/train_grpo.py), and
`build` refuses rollouts whose headers show anything else (WHAT_FAILED #42).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .reward import RewardConfig, compute_reward, group_advantages
from .sft import TemplateDrift, chat_kwargs, episode_examples, read_trace, tokenize_example

# How a rollout must have been sampled for the policy gradient to be the
# policy's: the full softmax. train_grpo.py collect sends these.
ON_POLICY = {"top_p": 1.0, "top_k": -1, "min_p": 0.0}

# Episodes the harness stopped, not the policy: a server error, an interrupted
# run, the wall clock (which a busy server reaches as easily as a slow agent).
# The benchmark scores them as failures; as training signal they are noise,
# and they are left out of their groups.
HARNESS_STOPS = ("backend_error", "interrupted", "max_wall_seconds")


def episode_reward(recs: list[dict], task, reference_steps: int | None,
                   cfg: RewardConfig | None = None):
    """The reward of one recorded episode under today's checks, from its
    replayed calls; None when the calls do not replay as recorded."""
    from ..harness.replay import replay

    r = replay(recs, task)
    if r.diverged or r.mismatched:
        return None
    foot = next((x for x in reversed(recs) if x.get("type") == "footer"), {})
    steps = (foot.get("budget") or {}).get("steps")
    if steps is None:
        steps = sum(1 for x in recs if x.get("type") == "step")
    return compute_reward(task, r.db, steps, reference_steps, cfg)


def build(run_dir: str | Path, tasks: dict[str, Any], tokenizer,
          solutions: dict[str, list] | None = None, cfg: RewardConfig | None = None,
          server=None, max_len: int = 16384, expect_model: str | None = None,
          episodes: list[dict] | None = None) -> tuple[list[dict], dict[str, Any]]:
    """GRPO examples from one collection run, and the stats to read first.

    `episodes` stands in for reading `run_dir` (tests). Returns one row per
    agent turn, each with its episode's `advantage` and `norm` (the token count
    of the episode's turns together)."""
    eps = episodes if episodes is not None else [
        read_trace(p) for p in sorted(Path(run_dir).glob("*.jsonl"))]
    if not eps:
        raise SystemExit(f"no traces in {run_dir}")
    if any(e["messages"] is None for e in eps):
        raise SystemExit(f"{run_dir}: collect with --save-messages; GRPO trains on the "
                         f"conversations")
    kw = {json.dumps(chat_kwargs(e["header"]), sort_keys=True) for e in eps}
    if len(kw) != 1:
        raise SystemExit(f"{run_dir} mixes chat_template_kwargs {sorted(kw)}")
    kwargs = json.loads(kw.pop())
    models = Counter(e["header"].get("requested_model") for e in eps)
    if expect_model and set(models) != {expect_model}:
        raise SystemExit(f"{run_dir} was collected from {dict(models)}, not {expect_model}: "
                         f"GRPO trains the policy that sampled the episodes")
    for e in eps:
        h = e["header"]
        if h.get("context") != "full" or h.get("guardrail", "off") != "off" \
                or h.get("closing", "off") != "off":
            raise SystemExit(f"{e['path']}: GRPO trains on plain full-context episodes "
                             f"(no guardrail, no closing check)")
        body = h.get("agent_extra_body") or {}
        if any(body.get(k) != v for k, v in ON_POLICY.items()) \
                or h.get("agent_temperature") != 1.0:
            # A request that leaves them out gets the model's own defaults from
            # SGLang -- a top-k and a top-p -- and its rollouts come from a
            # distribution the loss never sees (WHAT_FAILED #42). The loss reads
            # the logits as they are, so the temperature is 1.
            raise SystemExit(f"{e['path']}: sampled at temperature "
                             f"{h.get('agent_temperature')} with "
                             f"{({k: body.get(k) for k in ON_POLICY})}, not the full softmax "
                             f"at 1 ({ON_POLICY}): collect with `train_grpo.py collect`")

    solutions = solutions or {}
    by_task: dict[str, list[tuple[dict, Any]]] = defaultdict(list)
    skipped: Counter = Counter()
    for e in eps:
        tid = e["header"]["task_id"]
        task = tasks.get(tid)
        if task is None:
            skipped["unknown task"] += 1
            continue
        if not e["footer"]:
            # Never finished (an interrupted run): not a result, here as in
            # every report (diagnose.load_episodes).
            skipped["unfinished"] += 1
            continue
        stop = e["footer"].get("stop_reason")
        if stop in HARNESS_STOPS:
            skipped[f"stopped by the harness ({stop})"] += 1
            continue
        recs = [e["header"], *e["steps"], *([e["footer"]] if e["footer"] else [])]
        rb = episode_reward(recs, task, len(solutions.get(tid, [])) or None, cfg)
        if rb is None:
            skipped["does not replay"] += 1
            continue
        by_task[tid].append((e, rb))

    rows: list[dict] = []
    groups = []
    for tid, members in sorted(by_task.items()):
        passed = [rb.passed for _, rb in members]
        mixed = 0 < sum(passed) < len(passed)
        adv = group_advantages([rb.reward for _, rb in members]) if mixed else []
        groups.append({"task_id": tid, "trap": tasks[tid].trap, "k": len(members),
                       "passed": sum(passed),
                       "pass_rate": round(sum(passed) / len(passed), 3),
                       "degenerate": not mixed})
        if not mixed:
            skipped["episode in a degenerate group"] += len(members)
            continue
        for (e, rb), a in zip(members, adv):
            if a == 0:
                continue
            try:
                exs, sk = episode_examples(e, tokenizer, kwargs, server)
            except (TemplateDrift, ValueError):
                # A turn the local template cannot render as the model wrote
                # it, or a trace whose calls do not line up with its messages:
                # the episode is left out, and counted (a server that cannot
                # render is not caught here: that stops the build).
                skipped["episode whose turns cannot be rendered"] += 1
                continue
            skipped.update(sk)
            kept = []
            for ex in exs:
                ids = tokenize_example(tokenizer, ex)
                n_tok = len(ids["input_ids"])
                if n_tok > max_len:
                    skipped["longer than max_len"] += 1
                    continue
                kept.append((ex, n_tok, sum(l != -100 for l in ids["labels"])))
            norm = sum(c for _, _, c in kept)
            for ex, n_tok, _ in kept:
                rows.append({"task_id": tid, "trap": tasks[tid].trap, "fold": "grpo",
                             "tokens": n_tok, "advantage": a, "reward": rb.reward,
                             "passed": rb.passed, "norm": norm, **ex})

    dead = sum(g["degenerate"] for g in groups)
    bands = Counter("always" if g["pass_rate"] == 1 else "never" if g["pass_rate"] == 0
                    else "sometimes" for g in groups)
    by_trap: dict[str, list[float]] = defaultdict(list)
    for g in groups:
        by_trap[g["trap"]].append(g["pass_rate"])
    scored = sum(g["k"] for g in groups)
    stats = {
        "source": str(run_dir),
        "model": next(iter(models)),
        "chat_template_kwargs": kwargs,
        "rendered_by": "server /tokenize" if server is not None else "local tokenizer",
        "episodes": len(eps),
        # The policy that sampled them, at the collection's temperature, on
        # these tasks: iteration by iteration, the learning curve on the
        # training tasks (the held-out evaluation is the result).
        "passed": sum(g["passed"] for g in groups),
        "pass_rate": round(sum(g["passed"] for g in groups) / scored, 4) if scored else 0.0,
        "groups": len(groups),
        "degenerate_groups": dead,
        "degenerate_rate": round(dead / len(groups), 4) if groups else 0.0,
        "bands": dict(bands),
        "per_trap_pass_rate": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_trap.items())},
        "examples": len(rows),
        # As in RFT's build: prompts the local template renders otherwise (not
        # an error: training uses the server's), and prompts that do not END
        # as the local render does -- the turn would not follow them, and
        # train_grpo.py build refuses those.
        "prompts_unlike_local_render": (sum(r.get("prompt_matches_local") is False
                                            for r in rows) if server is not None else None),
        "prompts_ending_unlike_local_render": (
            sum(r.get("prompt_end_matches_local") is False for r in rows)
            if server is not None else None),
        "episodes_used": len({r["trace"] for r in rows}),
        "tokens": sum(r["tokens"] for r in rows),
        "mean_abs_advantage": (round(sum(abs(r["advantage"]) for r in rows) / len(rows), 4)
                               if rows else 0.0),
        "skipped": dict(skipped),
        "learning_tasks": [g["task_id"] for g in groups if not g["degenerate"]],
    }
    return rows, stats


def grpo_loss(logits, labels, advantage: float, norm: float, old=None, clip: float = 0.2):
    """The clipped policy-gradient loss of one turn.

    logits [1, L, V] and labels [1, L] as the model returns and the collator
    builds them (-100 outside the turn); `old` the turn's per-token
    log-probabilities under the policy the iteration started from (None: this
    pass's own, detached -- the first step of an iteration, where the ratio is
    1). Returns the loss and the per-token log-probabilities of the turn."""
    import torch
    import torch.nn.functional as F

    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets != -100
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                          targets.clamp(min=0).reshape(-1), reduction="none")
    logp = -nll.view_as(targets)[mask]
    ref = logp.detach() if old is None else old.to(logp.device, logp.dtype)
    if ref.shape != logp.shape:
        raise ValueError(f"old log-probabilities for {ref.numel()} tokens, turn has "
                         f"{logp.numel()}")
    ratio = torch.exp(logp - ref)
    a = torch.as_tensor(advantage, dtype=logp.dtype, device=logp.device)
    objective = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    return -objective.sum() / max(float(norm), 1.0), logp.detach()


def write(rows: list[dict], stats: dict, path: str | Path, stats_path: str | Path) -> Path:
    """The examples, then their stats. Each is written whole and renamed into
    place, and the stats come last: a build cut short leaves no stats file, and
    gpu_pipeline.sh builds again rather than training on half an iteration."""
    path, stats_path = Path(path), Path(stats_path)
    for p in (path, stats_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    tmp = stats_path.with_name(stats_path.name + ".tmp")
    tmp.write_text(json.dumps(stats, indent=2))
    tmp.replace(stats_path)
    return path
