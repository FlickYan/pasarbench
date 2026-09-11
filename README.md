# PasarBench

A verifiable tool-agent **environment** for Southeast Asian e-commerce customer
service. 186 tasks across 6 markets and 6 language varieties, 20 tools (303 with
distractors), one policy document, and verification on final database state
rather than text.

Because verification is programmatic, it is also a **reward function** — so the
same artefact measures agents and trains them.

```
rollouts ──▶ verifier ──▶ group advantage ──▶ policy update ──▶ rollouts
                                                                   │
                                       no human, no reward model in the loop
```

`pasar` = market (Malay/Indonesian). Hardware assumed: 1× A100 to 4× H100.

**[Runbook →](docs/RUNBOOK.md)** · **[Results →](RESULTS.md)** · **[What failed →](docs/WHAT_FAILED.md)** ·
**[Writeup kit →](docs/WRITEUP.md)** · **[Build log →](docs/README_weekly.md)**

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
python -m tests.test_harness      # 28 harness invariants
python -m tests.test_reward       # 30 reward + dataset invariants
python -m tests.test_generated    # 47 checks over all 170 generated tasks
python -m tests.test_context      # 39 context-strategy + analysis invariants
python -m tests.test_judge        # 51 judge + agreement-statistics invariants
python -m tests.test_exposure     # 47 tool-scaling + diagnosis invariants
python -m tests.test_serving      # 48 metrics, cost and quality-guard invariants

python -m pasarbench.sweep --backend scripted --suite all
python scripts/make_report.py     # regenerates RESULTS.md; never hand-fill it
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
you can check a suite by reading it. At 186 you cannot, so the guarantee is
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
| **language** | the surface form — what the customer types | en, sg-en, ms, id, th, vi |

SEA-specific content that no US-built benchmark has: **cash on delivery** (no
instrument to refund to), **livestream purchase disputes**, **zero-minor-unit
currencies** (IDR and VND — an agent that divides by 100 is wrong in half the
markets), peak-sale SLA extensions, and customs holds.

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
honest one: at 186 tasks a 2-point regression is undetectable (`required_n` says
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

---

## Repo map

```
pasarbench/
  db.py tools.py policy.md tasks.py verifier.py   the environment
  generate.py locales.py                          186 tasks, 3 orthogonal axes
  harness/                                        the runtime
  rl/                                             reward + trajectory datasets
  judge/                                          rubric, judges, agreement stats
  analyze.py diagnose.py simqa.py                 turning sweeps into findings
  serving/                                        metrics, cost, quality guards
  sweep.py run.py                                 entry points
scripts/    serve_vllm.sh train_rft.py train_grpo.py modal_vllm.py
            make_report.py slurm_train.sh slurm_sweep.sh
tests/      9 suites, ~300 assertions
docs/       RUNBOOK.md  WHAT_FAILED.md  WRITEUP.md  README_weekly.md
```

---

## Limitations

Stated plainly, because a limitations section nobody wrote is the first thing a
careful reader notices.

- **186 tasks cannot detect a 2-point regression.** ~6,000 per arm would be
  needed. Several verdicts in this work are honestly INCONCLUSIVE.
- **Thai and Vietnamese translations are drafted, not native-reviewed.** English,
  Singlish, Malay and Indonesian were written directly; Singlish deliberately
  was not machine-translated, because the particles and code-switching are the
  part that breaks agents.
- **33 of 186 tasks are "do no harm" traps** that a cautious lookup-only agent
  passes for free. `tests/test_generated.py` exempts them explicitly rather than
  pretending otherwise. They are the argument for the judge.
- **Prose-only policy rules are unverifiable by state checks.** P1.4 (never
  disclose another customer's data) cannot be checked by looking at the
  database; those tasks were left out rather than checked badly.
- **`scripts/train_grpo.py` collects and scores groups but does not do the
  policy update.** The three remaining pieces — token alignment between vLLM and
  HF, turn-level loss masking, per-token advantage normalisation — are named
  explicitly in that file rather than glossed.
- **`LLMUser` has not been run against a real model.** Its persona prompt is
  unvalidated until `simqa` reports a leak rate on real transcripts.
- **Default cost rates are illustrative placeholders.** GPU hourly rates vary by
  3× across providers.
