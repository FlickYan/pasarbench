"""
Rejection-sampling fine-tuning (RFT) on PasarBench, sized for 1-2 rented H100s.

    # 0. the split, once. Every later step reads it.
    python -m pasarbench.rl.split

    # 1. with the agent and the simulated customer served (serve_sglang.sh)
    python scripts/train_rft.py baseline          # base model, all 215 tasks, k=5, T=0
    python scripts/train_rft.py collect           # base model, all 215 tasks, k=8, T=1

    # 2. examples, then look at them before spending anything
    python scripts/train_rft.py build
    python scripts/train_rft.py check --data data/rft/train-A.jsonl --server http://localhost:8000

    # 3. servers down, one LoRA per fold (both GPUs: prefix with accelerate launch)
    python scripts/train_rft.py train --data data/rft/train-A.jsonl --out checkpoints/rft-A
    python scripts/train_rft.py train --data data/rft/train-B.jsonl --out checkpoints/rft-B

    # 4. serve both adapters, then score each on the fold it did not train on
    python scripts/train_rft.py eval

Every sweep step prints the exact `pasarbench.sweep` command it runs, resumes
after a crash, and writes traces the audits and the report read.

DO THIS BEFORE GRPO. Always.
---------------------------
On a verifiable environment, rejection sampling recovers most of the
achievable gain with none of the RL infrastructure. It is also the baseline
GRPO must beat -- and frequently does not beat by much. Skip it and a GRPO gain
cannot be told apart from simply training on correct trajectories, which is
the first question a post-training interviewer will ask.

WHY LoRA ON TWO H100s
---------------------
Full-parameter 8B needs ~128 GB of weights, gradients and Adam states before
activations: 64 GB per card under ZeRO-3 on two cards, which leaves too little
for 16k-token sequences. LoRA keeps the 16 GB base frozen and trains ~1% of the
parameters, which fits one card with room to spare. With a few thousand
examples per fold, LoRA is also the right size of update, and SGLang serves
the adapters directly -- no merge step, both folds from one server.

THE FAILURE MODE THAT RUINS AGENT SFT
-------------------------------------
Training on tool OBSERVATIONS teaches the model to invent tool output. Here
loss only ever touches the completion of an example -- the agent's own turn --
and `check` prints exactly which text that is. Read it before training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/rft")
FOLDS = "data/splits/folds.json"
NO_THINKING = json.dumps({"chat_template_kwargs": {"enable_thinking": False}})

from pasarbench.models import SIM_DEFAULT  # noqa: E402  (after the sys.path line)


# --------------------------------------------------------------------------
# sweeps: baseline, collection, evaluation

def _sweep(args, run_id: str, k: int, temperature: float, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", "pasarbench.sweep",
           "--backend", "openai", "--model", args.model, "--base-url", args.base_url,
           "--simulator", "openai", "--sim-model", args.sim_model,
           "--sim-url", args.sim_url, "--suite", "all", "-k", str(k),
           "--temperature", str(temperature), "--strategies", "full",
           "--workers", str(args.workers), "--run-id", run_id, "--resume",
           "--max-tokens", "2048", "--extra-body", args.extra_body,
           # the agent's flags (Qwen's enable_thinking) are not the customer's
           "--sim-extra-body", "{}", *extra]
    if args.sample:
        cmd += ["--sample", str(args.sample)]
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(cmd))


def baseline(args) -> None:
    """The base model, scored exactly as the fine-tunes will be."""
    _sweep(args, args.run_id or "P-base", args.k, 0.0, [])


def collect(args) -> None:
    """Sampled rollouts to learn from, with the full conversation saved."""
    _sweep(args, args.run_id or "P-collect", args.k, args.temperature,
           ["--save-messages"])


def evaluate(args) -> None:
    """Each fine-tune on the fold it did not train on, in one sweep.

    SGLang serves an adapter under <base>:<adapter>; the sweep checks both
    names against the server's adapter list before the first episode, because
    a bare adapter name would be answered by the base model."""
    fm = ",".join(f"{f}={args.model}:{args.adapter_prefix}{f}" for f in ("A", "B"))
    _sweep(args, args.run_id or "P-rft", args.k, 0.0,
           ["--folds", args.folds, "--fold-models", fm])


# --------------------------------------------------------------------------
# examples

def _tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{name} has no chat template; cannot render prompts")
    return tok


def build(args) -> None:
    out_dir = Path(args.out_dir)
    from pasarbench.generate import generate
    from pasarbench.rl.sft import build as build_examples
    from pasarbench.rl.split import load_folds
    from pasarbench.tasks import TASKS

    from pasarbench.rl.sft import ServerRenderError, server_renderer
    from pasarbench.tools import schemas

    tasks = {t.task_id: t for t in list(TASKS) + list(generate()[0])}
    folds = load_folds(args.folds, tasks.values())
    server = None
    if args.server:
        server = server_renderer(args.server, args.served_model or args.model)
        try:
            # One probe with a tool list before a few thousand requests: a
            # server that is down, or cannot render tools, is better found here.
            server([{"role": "user", "content": "hi"}], schemas(["get_order"]), {})
        except ServerRenderError as e:
            print(f"!! {e}\n!! rendering LOCALLY instead. Against SGLang these "
                  f"prompts will not be the served ones (`check --server` fails "
                  f"them): bring the agent server up and build again before "
                  f"training.", flush=True)
            server = None
    files, stats = build_examples(args.run, tasks, folds, _tokenizer(args.model),
                                  max_per_task=args.max_per_task,
                                  max_len=args.max_len, expect_model=args.model,
                                  server=server)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold, rows in files.items():
        p = out_dir / f"train-{fold}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {p}: {len(rows)} turns, {stats['tokens'][fold]:,} tokens")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\ncollection pass rate {stats['pass_rate']:.3f} over {stats['episodes']} "
          f"episodes; skipped {stats['skipped'] or 'nothing'}")
    if server is not None:
        n = sum(stats["examples"].values())
        print(f"prompt token ids: the server's, for all {n} examples; "
              f"{stats['prompts_unlike_local_render']} of them are not what the local "
              f"template renders (SGLang dumps the tool list its own way)")
        if stats["prompts_ending_unlike_local_render"]:
            print(f"!! {stats['prompts_ending_unlike_local_render']} server prompts do not "
                  f"end in the local generation prompt: the turns would not follow them. "
                  f"`check` stops on this; find out why before training.")
    for fold, dead in stats["traps_with_zero_signal"].items():
        if dead:
            print(f"!! fold {fold}: {len(dead)} trap(s) never solved, so model "
                  f"rft-{fold} gets nothing to learn for them: {dead}")
    print("\nNext: python scripts/train_rft.py check --data data/rft/train-A.jsonl"
          " --server http://localhost:8000")


def _server_tokens(server: str, model: str, ex: dict, kwargs: dict):
    """Token ids the serving engine builds for this example's prompt, or the
    reason it could not say. The same request `build --server` made."""
    from pasarbench.rl.sft import (ServerRenderError, api_messages, read_trace,
                                   server_renderer)
    from pasarbench.tools import schemas

    rec = read_trace(ex["trace"])["messages"]
    msgs = api_messages(rec["messages"])[:ex["message_index"]]
    names = rec["tool_names"][ex["turn"]]
    try:
        return server_renderer(server, model)(msgs, schemas(names) if names else None,
                                              kwargs), None
    except ServerRenderError as e:
        return None, str(e)


def _first_difference(tok, a: list[int], b: list[int]) -> str:
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return (f"first difference at token {k} of {len(a)}/{len(b)}: "
            f"{tok.decode(a[max(0, k - 10):k + 10])!r} vs {tok.decode(b[max(0, k - 10):k + 10])!r}")


def check(args) -> None:
    """Show what one example trains on, and prove its prompt is the served one."""
    from pasarbench.rl.sft import TURN_END
    tok = _tokenizer(args.model)
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{args.data} is empty")
    stats = json.loads((Path(args.data).parent / "stats.json").read_text())
    kwargs = stats.get("chat_template_kwargs") or {}

    rnd = random.Random(0)
    sample = rnd.sample(rows, min(args.n, len(rows)))
    ex = next((r for r in sample if r["turn"] > 0), sample[0])
    print(f"example: {ex['task_id']} turn {ex['turn']} ({ex['tokens']} tokens)\n")
    print("… PROMPT, last 600 chars (no loss) " + "…" * 20)
    print(ex["prompt"][-600:])
    print("=== COMPLETION (loss) " + "=" * 50)
    print(ex["completion"])
    print("=" * 72)

    problems = []
    for r in rows:
        c = r["completion"]
        if not c.strip():
            problems.append(f"{r['task_id']} turn {r['turn']}: empty completion")
        elif not any(m and c.endswith(m) for m in (tok.eos_token, *TURN_END)):
            problems.append(f"{r['task_id']} turn {r['turn']}: completion does not "
                            f"end with an end-of-turn marker")
        if "<tool_response>" in c:
            problems.append(f"{r['task_id']} turn {r['turn']}: a tool result is in "
                            f"the trained text")
    lens = sorted(r["tokens"] for r in rows)
    print(f"{len(rows)} examples; tokens median {lens[len(lens) // 2]:,}, "
          f"max {lens[-1]:,}, total {sum(lens):,}")

    served_ids = all(r.get("prompt_ids") is not None for r in rows)
    if served_ids:
        print(f"\nall {len(rows)} examples train on the server's own prompt token ids "
              f"(built with --server); each turn is tokenized locally.")
        ends = [r for r in rows if r.get("prompt_end_matches_local") is False]
        if ends:
            problems.append(
                f"{len(ends)} server prompt(s) do not end in the generation prompt the "
                f"local template ends in, so the locally rendered turn does not follow "
                f"them (the server's template or chat_template_kwargs are not these). "
                f"{ends[0]['task_id']} turn {ends[0]['turn']} ends "
                f"{tok.decode(ends[0]['prompt_ids'][-8:])!r}, the local render "
                f"{tok.decode(tok(ends[0]['prompt'], add_special_tokens=False)['input_ids'][-8:])!r}")
        unlike = [r for r in rows if r.get("prompt_matches_local") is False]
        if unlike:
            r = unlike[0]
            local = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            print(f"  {len(unlike)} of {len(rows)} of those prompts are not what the local "
                  f"template renders, so a build without the server would have trained on "
                  f"prompts the model never saw. {r['task_id']} turn {r['turn']}, server vs "
                  f"local, {_first_difference(tok, r['prompt_ids'], local)}")
    if args.server:
        ok = bad = 0
        for r in sample:
            served, err = _server_tokens(args.server, args.served_model or args.model,
                                         r, kwargs)
            if err:
                print(f"  server check unavailable: {err}")
                break
            mine = (r["prompt_ids"] if served_ids
                    else tok(r["prompt"], add_special_tokens=False)["input_ids"])
            if served == mine:
                ok += 1
            else:
                bad += 1
                print(f"  MISMATCH {r['task_id']} turn {r['turn']}: server vs "
                      f"{'stored' if served_ids else 'local'}, "
                      f"{_first_difference(tok, served, mine)}")
        if ok or bad:
            what = "the ids stored in this file" if served_ids else "the local rendering"
            print(f"\nserver check: {ok} of {ok + bad} prompts the server renders are "
                  f"token-identical to {what}")
            if bad and served_ids:
                problems.append(f"the server now renders {bad} prompt(s) differently from "
                                f"when these examples were built (another model, template "
                                f"or SGLang version?) -- rebuild with `build --server "
                                f"{args.server}`")
            elif bad:
                problems.append(f"{bad} prompt(s) differ from what the server renders"
                                f" -- rebuild with `build --server {args.server}` so "
                                f"training uses the server's own tokens")
    else:
        print("\n(no --server: prompts not compared against the serving engine. Run "
              "it with the agent server up before training.)")
    for p in problems[:10]:
        print("  PROBLEM:", p)
    if problems:
        raise SystemExit(f"\n{len(problems)} problem(s). Do not train on this file "
                         f"until they are understood.")
    print("\nno problems found")


# --------------------------------------------------------------------------
# training

def train(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{args.data} is empty")
    folds = {r["fold"] for r in rows}
    tok = _tokenizer(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    from pasarbench.rl.sft import tokenize_example

    class Turns(torch.utils.data.Dataset):
        # Loss on the agent's own turn and nothing else: the prompt holds the
        # policy, the customer and every tool result.
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return tokenize_example(tok, rows[i])

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"]))
                                       for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"]))
                                    for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"])
                                            + [0] * (n - len(b["input_ids"]))
                                            for b in batch]),
        }

    bf16 = torch.cuda.is_available() and not args.fp32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if bf16 else torch.float32,
        attn_implementation=args.attn)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,          # turns run 3k-16k tokens; no padding waste
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,                  # LoRA; full fine-tuning would want ~1e-5
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,             # a rented node can vanish
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=Turns(), data_collator=collate)
    resume = args.resume and any(Path(args.out).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "pasarbench_meta.json").write_text(json.dumps({
        "base_model": args.model, "data": args.data, "folds": sorted(folds),
        "examples": len(rows), "epochs": args.epochs, "lr": args.lr,
        "lora_r": args.lora_r}, indent=2))
    print(f"\nadapter -> {args.out}. Serve it with serve_sglang.sh rft and run "
          f"`train_rft.py eval`. Training loss is not a result; held-out pass^k is.")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def served(p, k: int):
        p.add_argument("--model", default="Qwen/Qwen3-8B")
        p.add_argument("--base-url", default="http://localhost:8000/v1")
        # SIM_MODEL overrides the customer here and in serve_sglang.sh together
        p.add_argument("--sim-model", default=os.environ.get("SIM_MODEL") or SIM_DEFAULT)
        p.add_argument("--sim-url", default="http://localhost:8001/v1")
        p.add_argument("-k", type=int, default=k)
        p.add_argument("--workers", type=int, default=24)
        p.add_argument("--extra-body", default=NO_THINKING,
                       help="request flags for the agent; the default turns Qwen3 "
                            "thinking off, and the same flags reach training "
                            "through the traces")
        p.add_argument("--run-id", default="")
        p.add_argument("--sample", type=int, default=0,
                       help="N tasks per trap instead of all 215 (smoke tests only)")
        p.add_argument("--dry-run", action="store_true", help="print the command only")

    b = sub.add_parser("baseline", help="base model, k=5, T=0: the number to beat")
    served(b, 5)
    b.set_defaults(fn=baseline)

    c = sub.add_parser("collect", help="base model, k=8, T=1: episodes to learn from")
    served(c, 8)
    c.add_argument("--temperature", type=float, default=1.0)   # >0 or no diversity
    c.set_defaults(fn=collect)

    e = sub.add_parser("eval", help="each adapter on the fold it did not train on")
    served(e, 5)
    e.add_argument("--folds", default=FOLDS)
    e.add_argument("--adapter-prefix", default="pasar-rft-",
                   help="adapters are loaded as <prefix>A and <prefix>B, and "
                        "requested as <model>:<prefix>A")
    e.set_defaults(fn=evaluate)

    bd = sub.add_parser("build", help="per-turn examples, one file per fold")
    bd.add_argument("--run", default="traces/P-collect/full")
    bd.add_argument("--folds", default=FOLDS)
    bd.add_argument("--model", default="Qwen/Qwen3-8B")
    bd.add_argument("--max-per-task", type=int, default=3)
    bd.add_argument("--max-len", type=int, default=16384)
    bd.add_argument("--out-dir", default=str(OUT))
    bd.add_argument("--server", default="",
                    help="agent server (e.g. http://localhost:8000): take every "
                         "prompt's token ids from the server's own rendering")
    bd.add_argument("--served-model", default="")
    bd.set_defaults(fn=build)

    ck = sub.add_parser("check", help="print what gets loss; compare with the server")
    ck.add_argument("--data", default="data/rft/train-A.jsonl")
    ck.add_argument("--model", default="Qwen/Qwen3-8B")
    ck.add_argument("--server", default="", help="agent server, e.g. http://localhost:8000")
    ck.add_argument("--served-model", default="")
    ck.add_argument("--n", type=int, default=5)
    ck.set_defaults(fn=check)

    t = sub.add_parser("train", help="LoRA on one fold's examples")
    t.add_argument("--data", default="data/rft/train-A.jsonl")
    t.add_argument("--model", default="Qwen/Qwen3-8B")
    t.add_argument("--out", default="checkpoints/rft-A")
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lora-r", type=int, default=32)
    t.add_argument("--grad-accum", type=int, default=16)
    t.add_argument("--warmup-steps", type=int, default=10)
    t.add_argument("--save-steps", type=int, default=50)
    t.add_argument("--attn", default="sdpa")
    t.add_argument("--fp32", action="store_true", help="CPU smoke tests only")
    t.add_argument("--resume", action="store_true",
                   help="continue from the latest checkpoint in --out")
    t.set_defaults(fn=train)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
