"""
Rejection-sampling fine-tuning (RFT) on PasarBench.

    # 1. serve the model you want to improve
    ./scripts/serve_vllm.sh agent-8b

    # 2. collect rollouts, keep what the verifier passed
    python scripts/train_rft.py collect --k 8 --temperature 1.0 \
        --model Qwen/Qwen3-8B --base-url http://localhost:8000/v1

    # 3. train on them
    accelerate launch --config_file scripts/fsdp4.yaml \
        scripts/train_rft.py train --data data/rft/sft.jsonl

DO THIS BEFORE GRPO. Always.
---------------------------
On a verifiable environment, rejection sampling recovers most of the
achievable gain with none of the RL infrastructure. It is also the baseline
GRPO must beat -- and frequently does not beat by much. If you skip it you
have no way to attribute your GRPO gain to the algorithm rather than to
simply training on correct trajectories, and that is the first question a
post-training interviewer will ask.

MEMORY, 8B FULL-PARAMETER, 4x H100 80GB (320GB total)
-----------------------------------------------------
    params   bf16              16 GB
    grads    bf16              16 GB
    optimiser fp32 m,v + master 96 GB
    ------------------------------------
    states                    128 GB  -> /4 with ZeRO-3 = 32 GB per GPU
    activations, seq 8192, grad ckpt, mbs 2  ~15 GB per GPU
    ------------------------------------
    ~47 GB of 80 GB. Comfortable; seq 16384 also fits.

A 32B needs LoRA at this scale: base bf16 64 GB sharded to 16 GB/GPU, adapter
and its optimiser states negligible. Full-parameter 32B does not fit on four
cards and is not worth chasing -- the story is the 8B closing the gap on the
70B, not training the biggest thing you can.

THE FAILURE MODE THAT RUINS AGENT SFT
-------------------------------------
Training on tool OBSERVATIONS teaches the model to hallucinate tool output.
Loss must be masked to assistant tokens only. It is invisible in the loss
curve and shows up later as a model that invents order records instead of
calling get_order. `assistant_only_loss=True` below is load-bearing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pasarbench.harness.backends import OpenAICompatBackend
from pasarbench.harness.context import FullContext
from pasarbench.harness.simulator import LLMUser
from pasarbench.harness.types import Budget
from pasarbench.rl.collect import (corpus_stats, divergence_pairs, rollout_task,
                                   sft_examples, write_jsonl)
from pasarbench.rl.reward import RewardConfig
from pasarbench.run import SOLUTIONS
from pasarbench.tasks import TASKS

OUT = Path("data/rft")


def collect(args) -> None:
    agent = lambda t: OpenAICompatBackend(model=args.model, base_url=args.base_url,
                                          api_key="EMPTY",
                                          temperature=args.temperature)
    sim_backend = OpenAICompatBackend(model=args.sim_model, base_url=args.sim_url,
                                      api_key="EMPTY", temperature=0.8)

    rollouts = []
    for task in TASKS:
        rs = rollout_task(
            task, agent, k=args.k, cfg=RewardConfig(),
            reference_steps=len(SOLUTIONS.get(task.task_id, [])),
            simulator=LLMUser(sim_backend, task.persona, task.hidden_facts, task.language),
            context=FullContext(), budget=Budget(max_steps=args.max_steps),
            policy_mode=args.policy_mode,
        )
        rollouts.extend(rs)
        pr = sum(r.passed for r in rs) / len(rs)
        print(f"{task.task_id} {task.trap:38s} pass={pr:.2f} "
              f"mean_r={sum(r.reward.reward for r in rs)/len(rs):+.3f}", flush=True)

    stats = corpus_stats(rollouts)
    print("\n" + json.dumps({k: v for k, v in stats.items() if k != "per_trap"}, indent=2))

    # The number that decides whether RFT can work at all.
    dead = stats["traps_with_zero_signal"]
    if dead:
        print(f"\n!! {len(dead)} trap(s) never solved in {args.k} samples: {dead}")
        print("   RFT CANNOT teach these -- there is no correct trajectory to")
        print("   train on. They will still fail after training. Report them")
        print("   by name rather than hiding them in an aggregate, and treat")
        print("   them as the argument for exploration (higher k, higher")
        print("   temperature) or for GRPO with partial credit.")

    write_jsonl(sft_examples(rollouts, dedupe=True, max_per_task=args.max_per_task),
                OUT / "sft.jsonl")
    write_jsonl(divergence_pairs(rollouts), OUT / "dpo_pairs.jsonl")
    write_jsonl([{**r.reward.to_dict(), "trap": r.trap, "steps": r.steps,
                  "tokens": r.tokens, "stop": r.stop_reason} for r in rollouts],
                OUT / "rollouts.jsonl")
    with (OUT / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {OUT}/sft.jsonl, dpo_pairs.jsonl, rollouts.jsonl, stats.json")


def train(args) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    print(f"{len(rows)} verified trajectories")

    # Held-out split BY TASK, not by row. Splitting by row leaks: two rollouts
    # of the same task differ only by sampling noise, so a row split reports a
    # number that is closer to training accuracy than to generalisation.
    task_ids = sorted({r["task_id"] for r in rows})
    holdout = set(task_ids[::5])
    train_rows = [r for r in rows if r["task_id"] not in holdout]
    eval_rows = [r for r in rows if r["task_id"] in holdout]
    print(f"train {len(train_rows)} / eval {len(eval_rows)} "
          f"(held-out tasks: {sorted(holdout)})")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,          # effective batch 64 on 4 GPUs
        learning_rate=1e-5,                     # low: the base model is already
                                                # good; RFT is a nudge, not a
                                                # re-teach. 2e-5+ degrades tool
                                                # formatting on small sets.
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=8192,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        assistant_only_loss=True,               # LOAD-BEARING -- see module docstring
        report_to="none",
    )

    SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list([{"messages": r["messages"]} for r in train_rows]),
        eval_dataset=Dataset.from_list([{"messages": r["messages"]} for r in eval_rows]),
        processing_class=tok,
    ).train()

    print(f"\ndone -> {args.out}")
    print("Now re-serve the checkpoint and re-run the sweep. The ONLY number "
          "that counts is pass^k on held-out tasks, measured through the same "
          "harness. Training loss is not a result.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--model", default="Qwen/Qwen3-8B")
    c.add_argument("--base-url", default="http://localhost:8000/v1")
    c.add_argument("--sim-model", default="meta-llama/Llama-3.1-8B-Instruct")
    c.add_argument("--sim-url", default="http://localhost:8001/v1")
    c.add_argument("--k", type=int, default=8)
    c.add_argument("--temperature", type=float, default=1.0)   # >0 or no diversity
    c.add_argument("--max-steps", type=int, default=30)
    c.add_argument("--max-per-task", type=int, default=4)
    c.add_argument("--policy-mode", default="preload")
    c.set_defaults(fn=collect)

    t = sub.add_parser("train")
    t.add_argument("--model", default="Qwen/Qwen3-8B")
    t.add_argument("--data", default="data/rft/sft.jsonl")
    t.add_argument("--out", default="checkpoints/rft-8b")
    t.add_argument("--epochs", type=float, default=2.0)
    t.set_defaults(fn=train)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
