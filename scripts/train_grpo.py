"""
GRPO on PasarBench.

THE POINT OF THIS FILE
----------------------
The harness you built in week 2 is already the RL environment. `run_episode`
generates a multi-turn tool-calling trajectory; `verify` scores it with no
human and no reward model. `rollout_task` is literally the rollout function.
Nothing new has to be built for the environment side -- that is the whole
argument for having built the harness properly first, and it is the sentence
to lead with when describing this project.

    rollouts --> verifier --> group advantage --> policy update --> rollouts
                                                                        |
                        the closed loop JD #1 describes as self-evolving

RUN RFT FIRST. GRPO'S ONLY MEANINGFUL BASELINE IS A REJECTION-SAMPLED MODEL,
not the base model. A GRPO number reported against the base model conflates
"RL worked" with "training on correct trajectories worked", and that
conflation is the first thing an interviewer will probe.

TWO VIABLE PATHS ON 4x H100
---------------------------
  A. verl with a custom agent loop + custom reward.
     Mature multi-turn tool-calling support, hybrid engine that sleeps vLLM
     during the training phase so rollout and training share all four cards.
     Recommended if you want results rather than a systems exercise.

  B. This file: your own loop, vLLM for rollout, FSDP for the update.
     More work, more to go wrong, and far more to talk about. The token
     plumbing below is the part that is genuinely hard.

MEMORY, 8B, 4x H100 80GB
------------------------
  colocated (recommended): policy FSDP ~32 GB/GPU, vLLM woken only during
      rollout with weights reloaded from the sharded policy. Reference model
      omitted entirely by setting kl_coef=0 -- on a verifiable reward with a
      strong SFT init, KL to reference buys little and costs 16 GB plus a
      forward pass per step.
  split (simpler): GPUs 0-2 train with ZeRO-3, GPU 3 runs vLLM at TP=1.
      Wastes ~25% of the cluster during the training phase but is far easier
      to debug. Start here.

THE THREE THINGS THAT ACTUALLY GO WRONG
---------------------------------------
 1. Degenerate groups. If all k rollouts of a task pass, or all fail, the
    group has zero variance and contributes no gradient. `degenerate_group_rate`
    exists to measure this. Above ~70% you are burning rollout compute on
    nothing; fix it by rebalancing the task mix toward the middle band, not by
    turning up the learning rate.
 2. Loss masking across turns. Only assistant tokens get loss. Tool
    observations and simulator turns must be masked or the policy learns to
    predict its own environment, which is the multi-turn version of the SFT
    failure in train_rft.py.
 3. Length bias. Longer trajectories accumulate more tokens and, under a naive
    sum-of-logprobs objective, more gradient. Normalise per-token within a
    trajectory or short correct solutions get out-competed by rambling ones.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pasarbench.harness.backends import OpenAICompatBackend
from pasarbench.harness.simulator import LLMUser
from pasarbench.harness.types import Budget
from pasarbench.rl.collect import grpo_groups, rollout_task
from pasarbench.rl.reward import RewardConfig, degenerate_group_rate
from pasarbench.run import SOLUTIONS
from pasarbench.tasks import TASKS


def collect_groups(args) -> list[dict]:
    """One GRPO iteration's worth of rollouts. Pure environment interaction --
    no training code touches this, which is exactly why the harness split was
    worth doing."""
    agent = lambda t: OpenAICompatBackend(model=args.model, base_url=args.base_url,
                                          api_key="EMPTY", temperature=args.temperature)
    sim = OpenAICompatBackend(model=args.sim_model, base_url=args.sim_url,
                              api_key="EMPTY", temperature=0.8)

    all_rollouts = []
    for task in TASKS:
        all_rollouts.extend(rollout_task(
            task, agent, k=args.group_size, cfg=RewardConfig(),
            reference_steps=len(SOLUTIONS.get(task.task_id, [])),
            simulator=LLMUser(sim, task.persona, task.hidden_facts, task.language),
            budget=Budget(max_steps=args.max_steps),
            policy_mode=args.policy_mode,
        ))

    groups = grpo_groups(all_rollouts)
    dead = degenerate_group_rate([g["rewards"] for g in groups])

    print(f"\ngroups={len(groups)}  degenerate={dead:.1%}")
    if dead > 0.7:
        print("!! most groups carry no learning signal. Rebalance the task mix")
        print("   toward traps the policy solves sometimes-but-not-always, or")
        print("   raise temperature / group_size. Do NOT compensate with LR.")

    by_band = defaultdict(int)
    for g in groups:
        pr = g["pass_rate"]
        by_band["never (0.0)" if pr == 0 else
                "always (1.0)" if pr == 1 else "learnable"] += 1
    print("group bands:", dict(by_band))
    print("\nper-trap pass rate:")
    for g in sorted(groups, key=lambda x: x["pass_rate"]):
        flag = "   <-- no signal" if g["degenerate"] else ""
        print(f"  {g['pass_rate']:.2f}  {g['trap']}{flag}")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/rft-8b")   # NOT the base model
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--sim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sim-url", default="http://localhost:8001/v1")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--policy-mode", default="preload")
    ap.add_argument("--out", default="data/grpo/iter0.json")
    args = ap.parse_args()

    groups = collect_groups(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(groups, indent=2))
    print(f"\nwrote {args.out}")

    print("""
NEXT: the policy update. What this scaffold does NOT do yet, in order of
difficulty, so you know exactly what you are signing up for.

  1. Token alignment. You need per-token logprobs for the assistant spans of
     each trajectory under BOTH the sampling policy (vLLM, `logprobs=0` at
     generation) and the current policy (a forward pass under FSDP). vLLM and
     HF must tokenise identically -- verify this explicitly on a few
     trajectories before trusting a single gradient. Chat-template drift
     between the two is the classic silent killer here and it produces a
     ratio that looks plausible and is wrong.

  2. Turn-level masking. Build a boolean mask over the full episode token
     sequence that is True only on assistant-generated tokens. Tool results
     and simulator turns are context, not actions.

  3. The GRPO objective. Advantage is already computed per trajectory in
     `groups[i]["advantages"]`; broadcast it across that trajectory's
     assistant tokens, clip the ratio, normalise PER TOKEN within a trajectory
     (see length bias, above), and set kl_coef=0 to start.

  4. Weight sync. After the update, push weights back into vLLM before the
     next rollout iteration. With the split layout this is a checkpoint save
     plus a server restart -- slow but obviously correct. Colocate later.

If step 1 looks like more systems work than you want, use verl and keep this
file as your reward and rollout definition. That is not a retreat: the
environment, the verifiable reward and the harness are the contribution, and
they are yours either way.

WHAT TO REPORT
  base -> RFT -> GRPO, on held-out tasks, as pass^1 AND pass^4, with the
  per-trap breakdown. Then the gap-closing line against the 70B reference:
  cost per resolved conversation for each. Three models, one table. If GRPO
  does not beat RFT, say so -- that is a finding, and pretending otherwise is
  the one thing that will actually cost you the interview.
""")


if __name__ == "__main__":
    main()
