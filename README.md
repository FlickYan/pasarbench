# PasarBench

A verifiable tool-agent **environment** for Southeast Asian e-commerce customer
service. 215 tasks (16 hand-written, 199 generated) across 6 markets and 8
language varieties, 20 tools (303 with distractors), one policy document, and
verification on final database state rather than text.

Because verification is programmatic, it is also a **reward function** — so the
same artefact measures agents and trains them.

```
rollouts ──▶ verifier ──▶ group advantage ──▶ policy update ──▶ rollouts
                                                                   │
                                       no human, no reward model in the loop
```

`pasar` = market (Malay/Indonesian). Hardware: a laptop for everything but serving and post-training, which run on one rented B200 (or H200, or two H100s: Modal Notebook, Modal jobs or Lambda, [docs/GPU_GUIDE.md](docs/GPU_GUIDE.md)).

**[Runbook →](docs/RUNBOOK.md)** · **[Results →](RESULTS.md)** · **[What failed →](docs/WHAT_FAILED.md)** ·
**[Writeup kit →](docs/WRITEUP.md)** · **[Blog post →](docs/BLOG.md)** · **[Build log →](docs/README_weekly.md)**

## What it found

Agent `deepseek-v4-pro`, simulated customer and judges `qwen3.8-flash`; for
post-training, agent `Qwen3.8-27B` against a `gemma-4-31B-it` customer. Every
number is regenerated from traces by `scripts/make_report.py` and
`scripts/audit_tool_arms.py`, on today's checks — the runs' recorded verdicts
are re-scored by replaying their tool calls (`scripts/rescore.py`).

- **The fine-tune learned the verifier, not the policy.** RFT gained 2.3 points
  (p = 0.18) because two checks demanded one lookup tool where the policy only
  needs the fact. Corrected and re-scored — nothing re-run — the gain is 0.3
  points (p = 0.84), and the API reference the quirk had put 4 points above the
  base model is 6.6 below it (p = 0.004; on pass^k it was never ahead, and now
  trails by 9.7). A review found the first fix too loose; tightened, it moved no
  verdict ([WHAT_FAILED #30](docs/WHAT_FAILED.md)).
- **Search-based tool exposure cost 22% more tokens and bought nothing**: 11.5
  points below exposing all 20 tools (7 tasks worse, 1 better, p = 0.07 — it
  read p = 0.021 before the checker fix). Of 21 times a failed episode needed a
  hidden tool, the agent never searched for it 17 times — always a tool the
  policy describes but never names. Naming those three tools in the policy, in
  a re-run against a same-day control, took escalations from 12 to 21 of the 24
  episodes that needed one and pass^1 from 0.844 to 0.938 (6 tasks better, 1
  worse, p = 0.125), for 18% more tokens ([RUNBOOK 1g](docs/RUNBOOK.md)).
- **The agent claimed actions it never took** — only when the tool was out of
  sight: 3 of 14 readable search-arm failures, 0 of 19 elsewhere. The same-day
  control repeated it, 3 of 14; with the tools named, 0 of 6. A guardrail that
  holds such a reply back before the customer sees it is built and tested: read
  over the 9,958 recorded episodes, it would have fired 25 times, each on a
  claim no call had backed ([RUNBOOK 1i](docs/RUNBOOK.md)). It has not run live
  yet, and nor has a confirmation of the naming result on 57 new tasks (1h).
- **An LLM judge could not stand in for the database.** Its best configuration —
  with the tool results and the policy — reached κ = 0.72 and still accepted 6
  of 15 failed episodes. It caught one of three false claims and noticed another
  but passed it; on its first run, given the same evidence, it had credited the
  one it caught.
- **No language differed from English beyond noise**, in two independent runs.
  The only significant gaps in the project came from a fact-gating bug, and the
  simulator "leaks" behind it were the detector's ([WHAT_FAILED #26](docs/WHAT_FAILED.md)).

---

## Quickstart

```bash
./run_tests.sh                    # every suite, from any directory
```

Or individually, **from the repo root** (the one containing `README.md`, not the
`pasarbench/` package inside it — that nesting is the most common first-run
mistake):

```bash
python -m pasarbench.run          # null agent → 0.0, reference agent → 1.0
python -m tests.test_traps        # 10 naive solutions, all must be rejected
python -m tests.test_harness      # 38 harness invariants, the claim guardrail
python -m tests.test_reward       # 30 reward + dataset invariants
python -m tests.test_generated    # 108 checks over all 199 generated tasks: dates in causal order, the same world in every process, reads of the task's own order, simulator QA
python -m tests.test_rescore      # 29 re-scoring invariants: the pre-v19 world, replay by digest and by length, the same tool errors on every Python, nothing re-scored on a guess
python -m tests.test_context      # 39 context-strategy + analysis invariants
python -m tests.test_judge        # 83 judge + agreement-statistics invariants, judged against today's checks
python -m tests.test_exposure     # 121 tool-scaling + diagnosis invariants, the tool-naming experiment and its confirmation tasks, the guardrail's report and dry run
python -m tests.test_serving      # 54 metrics (SGLang and vLLM), cost and quality-guard invariants
python -m tests.test_report       # 48 report invariants: paired verdicts, ties, replication, calibration, noise floor, post-training, both scorings in every section
python -m tests.test_training     # 160 checks: split, customer family, collection and resume, examples (Qwen3 and Qwen3.8 formats) against an SGLang-like server, adapter routing, the LoRA probe, serve and Modal scripts, one-GPU memory plans, the notebook helper (progress in place, an interrupt-proof cleanup), the CUDA compiler, run settings, LoRA steps on Qwen3 and Qwen3.8's hybrid architecture, training data picked by today's checks, a push script that keeps runs and keys off GitHub (118 without transformers/torch)

python -m pasarbench.sweep --backend scripted --suite all
python scripts/make_report.py     # regenerates RESULTS.md; never hand-fill it
python scripts/rescore.py         # what today's checks change in every recorded run
```

The benchmark, harness, reward and judge layers have **zero dependencies**
(Python 3.10+). Model backends use `urllib`. Everything in `requirements.txt` is
for the training track only.

---

## What makes it different from a weekend benchmark

**Verification is on final database state, not text.** The agent's prose is
never scored for correctness. A task passes when the right rows exist with the
right values, the required calls happened, no forbidden call happened, and
ordering constraints hold. Text-similarity scoring is what makes most homemade
agent benchmarks meaningless.

**Tools enforce data integrity, not policy judgment.** `issue_refund` blocks a
COD refund because there is genuinely no instrument to refund to. But
`initiate_return` will happily open a return 19 days after delivery — that is a
*policy* error and catching it is the agent's job. If the tools enforced
everything, every agent would score 100% and the benchmark would measure
nothing. This is the line most homemade benchmarks get wrong in the safe
direction.

**Every generated task ships with a generated reference solution.** At 16 tasks
you can check a suite by reading it. At 199 you cannot, so the guarantee is
mechanical: the suite asserts every task is solved by its own solution, fails
under the null agent, and fails under a tool-spam adversary.

**Locale twins share a byte-identical checks object** — asserted with `is`, not
equality. A Thai task and its English twin have the same world and the same pass
criteria; only the words differ. That is what licenses attributing a score gap
to language at all.

---

## The suite

| axis | decides | values |
|---|---|---|
| **trap** | the checks — what correct behaviour is | 16 |
| **market** | the world — currency, payment rails, order data | SG MY ID TH PH VN |
| **language** | the surface form — what the customer types | en, sg-en, zh-SG, ms, zh-MY, id, th, vi |

Chinese is included because Singapore is majority ethnic Chinese and Malaysia
has a large Chinese-speaking population — a SEA e-commerce benchmark that omits
it is not modelling those two markets. Both use **simplified** characters
officially; traditional belongs to Taiwan, Hong Kong and Macau, which are not
markets here, so a "traditional" variant would be inauthentic rather than more
thorough. Written Cantonese is likewise a Hong Kong register: a
Cantonese-speaking shopper in KL types standard written Chinese to support, so
Mandarin-register simplified is the faithful choice for a text benchmark.

SEA-specific content that no US-built benchmark has: **cash on delivery** (no
instrument to refund to), **livestream purchase disputes**, **zero-minor-unit
currencies** (IDR and VND — an agent that divides by 100 is wrong in two of the
six markets), peak-sale SLA extensions, and customs holds.

### The flagship trap

`livestream_claim_overrides_window`: delivered 15 days ago, so every surface
signal says out-of-window → offer a voucher. But P7.2 waives the window when a
seller's livestream claim went unfulfilled — an exception in a later section
overriding an earlier rule. The agent must notice the purchase came from a live,
call `get_livestream_claims`, and understand policy precedence. Models that
pattern-match on "14 days" fail it, and the trace shows exactly why.

---

## The harness

No framework. `harness/loop.py` is ~130 lines of control flow and every branch
exists because something breaks without it.

```
harness/types.py       messages, tool calls, budgets, episode state
harness/backends.py    scripted/mute/confused/failing + OpenAI-compat + Anthropic
harness/context.py     5 pluggable context strategies — the ablation seam
harness/exposure.py    tool-exposure arms + 300 distractors + progressive disclosure
harness/simulator.py   scripted + persona-driven user simulators
harness/prompts.py     preload vs JIT policy modes
harness/trace.py       JSONL traces, one file per episode
harness/loop.py        the loop
```

Validated by running the reference solutions **through** the loop rather than
around it: `scripted_harness_agent(SOLUTIONS)` scores 1.0, `MuteBackend` scores
0.0.

Four decisions worth defending:

- **An unanswered tool call corrupts the conversation.** When the tool-call
  budget runs out mid-batch the loop still emits an error result for each
  remaining call. Skipping them is the obvious implementation and produces a 400
  on the next request.
- **Context units are atomic.** An assistant message with tool calls plus the
  tool messages answering it is one unit truncation must never split. Most
  homegrown sliding windows have this bug; it surfaces as an error rate people
  blame on the model.
- **Backend exceptions end one episode, not the sweep.**
- **Malformed tool arguments are data, not exceptions** — returned to the model
  as a recoverable error. Small models emit broken JSON constantly, and
  disproportionately on non-Latin scripts, which is exactly the signal the
  multilingual diagnosis is looking for.

Interrupt and resume are real: `snapshot()` serialises episode state plus the
database, `restore()` continues, and the test asserts an identical final DB.

---

## Experiments

### Context ablation — 5 strategies × 2 policy modes

`full`, `window8`/`window4`, `trim3` (shrink old tool payloads to their
ok/error verdict), `summarize4` (running **structured** summary), `notes4`
(agent-written scratchpad).

**JIT is not a context strategy.** `policy_mode` governs the system prompt;
context strategies govern conversation history. They are orthogonal — folding
them into one axis is why many published context ablations cannot be read.

Two decisions decide whether `summarize` works: **structured, not free-text**
(a generic summary prompt drops exactly what the policy gates on — whether
identity was verified, and what eligibility returned), and **running, not
redone** (only new units are folded in, so cost is O(n) not O(n²)).

`notes4` has a failure mode that hides itself: if the agent never writes notes,
it degrades to a naked sliding window and the pass rate still looks like a
measurement. `note_discipline` is reported every sweep for that reason.

### Tool scaling — the confound that ruins two-arm studies

Token cost and selection difficulty degrade together, so 20-vs-300 is
unreadable. Four arms separate them, and **`random-N`** does the work: N tools
at random but always containing the ones the task needs — same token cost as
`all-N`, same reachability as `oracle`. The sweep warns you if you run without
the pair.

Measured on the reference path: `oracle` injects 431 schema tokens per call,
`all-100` injects 9,670, `search-300` injects 445 while reaching 300 tools.

### Multilingual diagnosis — mechanisms, in the order you must rule them out

1. **Budget exhaustion** — if non-English hits the ceiling more often, the gap
   is a *harness artefact* and every conclusion drawn before raising the budget
   is wrong. Checked first, above every model explanation.
2. **Tokenisation inflation** — tokens per character vs English.
3. **Retrieval miss** — `search_policy` and `search_tools` are keyword matchers
   over English text. `_rank_tools("refund money to the customer")` returns
   `issue_refund`; `คืนเงินให้ลูกค้า` returns **nothing**. That is a property of
   the system, not a bug to hide.
4. **Malformed tool arguments** — the harness's distinct recoverable error.
5. **Language drift** — the agent replies in English to a Thai customer. Passes
   every state check; caught only by the judge.

`attribute_gap` carries its own caveat into the output: these are
**co-movements, not causal attributions.**

### Judge calibration

Nine binary criteria targeting what state verification structurally cannot see —
data leakage, hallucinated facts, policy accuracy (all critical), plus
`language_match`, which is why a model can pass every state check while replying
to a Thai customer in English.

Kappa is reported with **prevalence and PABAK**, because on a criterion
satisfied 95% of the time two labellers can agree 96% and still score κ = 0.32.
And with a bootstrap CI, because a move from 0.62 to 0.68 with overlapping
intervals is not an improvement.

**The ceiling:** a judge cannot agree with you more than you agree with
yourself. Round 2 re-labels a subset blind, and agreement is reported as a
fraction of that test-retest κ.

### Serving and cost

**The metric is cost per *resolved* conversation.** Escalations consume human
agent time and failures come back as second contacts, so at any realistic wage a
few points of escalation rate outweigh the entire model bill — which lands at a
low single-digit percentage of total cost in both arms of the worked example.
A model 6× cheaper to run can be 3.8× more expensive per resolution.

**No speed win is declared without a proven quality result.** `serving_verdict`
returns WIN only on non-inferiority at a stated margin. The third verdict is the
honest one: at 215 tasks a 2-point regression is undetectable (`required_n` says
~6,000 per arm), so INCONCLUSIVE is frequently correct.

### Post-training

The verifier is a **verifiable reward** — no human, no reward model. The reward
is designed not to be hacked and it is tested: partial credit is clipped below
the pass floor (`min(passing) > max(failing)` asserted over the whole suite), a
forbidden action *gates* credit to zero rather than subtracting, and excess
steps cost. The tool-spam policy is in the test suite as an adversary.

Order of work: **RFT first** — on a verifiable environment it recovers most of
the gain with no RL infrastructure, and it is the only meaningful GRPO baseline.
A GRPO number reported against the base model conflates "RL worked" with
"training on correct trajectories worked."

**What it did:** nothing measurable, once two checks that preferred a lookup
tool were corrected — on those traps the fine-tune had learned the preference
([WHAT_FAILED #30](docs/WHAT_FAILED.md)) — and it moved behaviour between traps
([#31](docs/WHAT_FAILED.md)). RESULTS.md §6 has both scorings. `train_rft.py
build` now picks training episodes by today's checks, not the verdicts the
collection recorded.

The design, sized for one rented B200 (`docs/RUNBOOK.md`, phase 4; every step
of running it, for a first-time GPU user, in `docs/GPU_GUIDE.md`):

- **Qwen3.8-27B, LoRA in bf16.** No quantized base, so the model trained is the
  model evaluated; logits only at the trained turn, because a 248k vocabulary
  makes full logits the largest tensor in the step. 32 turns per optimizer
  step whether one card accumulates them or several share them.

- **Two folds by family.** Locale twins stay together and every trap is in
  both folds; each fine-tune is scored only on the fold it did not train on, so
  all 215 tasks are held out once, paired by task against the base model.
- **One example per agent turn**, with loss on that turn only and the prompt's
  token ids taken from the serving SGLang's `/tokenize` — the server renders
  the tool list from its own dump of it, so a local render of the same
  template is not what the model saw. A whole-conversation render is wrong for
  Qwen3 (`pasarbench/rl/sft.py`).
- **A customer from another model family** (Gemma 4 31B for a Qwen agent),
  enforced by the sweep, with the API reference re-run against the same
  customer. Serving is SGLang: agent (bf16) and customer (FP8) on one card,
  each server's memory worked out from what is free when it starts
  (`scripts/gpu_plan.py`), or one card each on a two-GPU machine.

---

## Repo map

```
pasarbench/
  db.py tools.py policy.md tasks.py verifier.py   the environment
  generate.py locales.py                          199 generated tasks, 3 orthogonal axes
  harness/                                        the runtime, and the claim guardrail
  rl/                                             reward, folds, per-turn SFT examples
  judge/                                          rubric, judges, agreement stats
  analyze.py diagnose.py simqa.py                 turning sweeps into findings
  rescore.py                                      recorded episodes, today's checks
  serving/                                        metrics, cost, quality guards
  sweep.py run.py                                 entry points
scripts/    gpu_pipeline.sh gpu_plan.py modal_pipeline.py pasar_notebook.py
            cuda_home.py setup_node.sh download_weights.sh
            serve_sglang.sh train_rft.py train_grpo.py make_report.py
            audit_tool_arms.py inspect_trace.py run_judges.py rescore.py
            compare_cells.py naming_tasks.py guardrail_dry_run.py
            slurm_train.sh slurm_sweep.sh modal_vllm.py
tests/      12 suites, 650+ assertions
docs/       RUNBOOK.md  GPU_GUIDE.md  WHAT_FAILED.md  WRITEUP.md  BLOG.md  README_weekly.md
```

---

## Limitations

Stated plainly, because a limitations section nobody wrote is the first thing a
careful reader notices.

- **215 tasks cannot detect a 2-point regression.** ~6,000 per arm would be
  needed. The tool arms are 32 tasks each. Several verdicts in this work are
  honestly INCONCLUSIVE.
- **Few models.** The ablations are one agent (`deepseek-v4-pro`) judged by
  one judge (`qwen3.8-flash`); post-training is one base model (`Qwen3.8-27B`),
  one recipe, one run.
- **A check encodes a choice about what establishes a fact.** v19 found two
  that demanded one lookup tool, and one that demanded identity verification
  where the policy asks for it only before a write
  ([WHAT_FAILED #30](docs/WHAT_FAILED.md)). Others may still be waiting: what a
  trap's failures have in common is where to look.
- **The false-claim check reads English only**, and only the phrasings it was
  written for — the same kind of instrument as the leak detector, so its counts
  are a floor.
- **Thai and Vietnamese translations are drafted, not native-reviewed.** English,
  Singlish, Malay, Indonesian and both Chinese varieties were written directly;
  Singlish deliberately was not machine-translated, because the particles and
  code-switching are the part that breaks agents.
- **52 of 199 generated tasks are "do no harm" traps** that a cautious
  lookup-only agent passes for free — and so does one that stalls: 4 of the 234
  photo-trap episodes the v19 checks promoted never asked for photos.
  `tests/test_generated.py` exempts them explicitly rather than pretending
  otherwise. They are the argument for the judge.
- **Prose-only policy rules are unverifiable by state checks.** P1.4 (never
  disclose another customer's data) cannot be checked by looking at the
  database; those tasks were left out rather than checked badly.
- **`scripts/train_grpo.py` collects and scores groups but does not do the
  policy update.** The three remaining pieces — token alignment between SGLang
  and HF, turn-level loss masking, per-token advantage normalisation — are named
  explicitly in that file rather than glossed.
- **The leak audit is only as good as its ask-patterns.** They cover the
  phrasings `deepseek-v4-pro` used across four runs and `Qwen3.8-27B` across
  three, where no reveal outside English was unprompted
  ([WHAT_FAILED #26](docs/WHAT_FAILED.md)). Another agent can ask in words the
  list lacks, and the customer's answers will read as leaks: read the line
  `inspect_trace.py` prints under each one first.
- **Default cost rates are illustrative placeholders.** GPU hourly rates vary by
  3× across providers.
