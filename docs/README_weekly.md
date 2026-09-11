# Build log (chronological)

The landing page is `../README.md`. This is the week-by-week version it was condensed from, kept because it carries design rationale in more depth than a README should.

---

# PasarBench

A verifiable tool-agent **environment** for Southeast Asian e-commerce
customer service. 186 tasks across 6 markets and 6 language varieties, 20
tools, one policy document, state-based verification -- and because
verification is programmatic, a reward function you can train against.

`pasar` = market (Malay/Indonesian).

```
rollouts --> verifier --> group advantage --> policy update --> rollouts
                                                                    |
                              the closed loop, with no reward model in it
```

**Hardware assumed:** 1x A100 to 4x H100.

---

## Quickstart

```bash
python -m pasarbench.run          # null agent -> 0.0, reference agent -> 1.0
python -m tests.test_traps        # 10 naive solutions, all must be rejected
python -m tests.test_harness      # 28 harness invariants
python -m tests.test_reward       # 30 reward + dataset invariants
python -m tests.test_generated    # 47 checks over all 170 generated tasks
python -m tests.test_context      # 39 context-strategy + analysis invariants
python -m tests.test_judge        # 51 judge + agreement-statistics invariants
python -m tests.test_exposure     # 47 tool-scaling + diagnosis invariants
python -m tests.test_serving      # 48 metrics, cost and quality-guard invariants
python -m pasarbench.sweep --backend scripted --suite all
```

All three suites pass with zero dependencies on Python 3.10+. Real model
backends use urllib, so there is still nothing to install.

Against a real model:

```bash
python -m pasarbench.sweep --backend openai --model Qwen/Qwen3-8B \
    --base-url https://<app>.modal.run/v1 --api-key EMPTY \
    --simulator openai --sim-model gpt-4o-mini \
    --strategies full,window8,trim3 --policy-mode preload -k 3
```

---

## Why it is built this way

**Verification is on final database state, not text.** The agent's prose is
never scored. A task passes when the right rows exist with the right values,
the required tool calls happened, no forbidden call happened, and the ordering
constraints hold. Text-similarity scoring is what makes most homemade agent
benchmarks meaningless, and it is the first thing an interviewer will probe.

**Tools enforce data integrity, not policy judgment.** `issue_refund` blocks a
COD refund because there is genuinely no instrument to refund to. But
`initiate_return` will happily open a return outside the 14-day window — that
is a *policy* error, and catching it is the agent's job. If the tools enforced
everything, the benchmark would measure nothing. This split is the single most
important design decision here after state-based verification.

**Tools return structured errors, never exceptions.** Every failure comes back
as `{"ok": False, "error": "...(P4.2)"}` with the policy section cited. An
agent can recover from that. This is a harness design choice, and error
recovery quality is one of the things worth ablating in week 4.

**The clock is frozen** at `NOW = 2026-11-11 10:00 SGT`, inside the 11.11 sale
window. Nothing calls `datetime.now()`. Every relative-time rule is
deterministic and reruns are reproducible.

**Money is integer minor units with a per-currency exponent.** IDR and VND have
zero minor digits; SGD/MYR/THB/PHP have two. This is real SEA texture and a
live trap — an agent that assumes "divide by 100" is wrong in half the markets.

---

## The world

6 users across SG/MY/ID/TH/PH/VN · 5 sellers · 12 products · 10 orders
· 4 payment methods including COD · shipments with customs states
· **livestream claims** — what the seller actually said on the live

That last table is the differentiator. No US-built benchmark has it, and it
powers the flagship trap.

---

## The 16 traps

| ID | Trap | What a careless agent does wrong |
|----|------|----------------------------------|
| T01 | happy path | refunds shipping on a change-of-mind return (P4.6) |
| T02 | COD refund method | calls `issue_refund` on a cash order (P4.2) |
| T03 | cancel while processing | — baseline, should pass |
| T04 | cancel after dispatch | attempts a cancel that policy forbids (P2.2) |
| T05 | out of window | refunds anyway because the customer is unhappy |
| T06 | out-of-window dispute | caves and refunds instead of escalating (P6.3) |
| T07 | high-value defect claim | opens the return before asking for photos (P3.3) |
| T08 | **livestream misrepresentation** | never pulls the claim; treats it as out-of-window |
| T09 | perishable | asks the customer to ship mouldy food back (P4.5) |
| T10 | hazmat | asks the customer to post a battery charger (P3.4) |
| T11 | peak-period delay | hands out a voucher for a 10-hour "delay" (P8.2) |
| T12 | customs hold | refunds a COD order nobody has paid for (P4.3) |
| T13 | identity failure | cancels for a caller who failed verification (P1.1) |
| T14 | duplicate refund | pays twice (P10) |
| T15 | address after dispatch | edits a locked address (P9.2) |
| T16 | COD cancel | refunds money that was never collected (P2.4) |

**T08 is the one to lead with.** The order was delivered 15 days ago, so every
surface signal says "out of window, offer a voucher". But P7.2 waives the window
when a livestream claim went unfulfilled — an exception buried in a later
section that overrides the earlier rule. An agent has to (a) notice the purchase
came from a live, (b) call `get_livestream_claims`, and (c) know that a later
policy section beats an earlier one. Models that pattern-match on "14 days" fail
this, and the failure is legible in the trace.

---

## Known limitations

Write these in the paper too. Naming them is worth more than hiding them.

- **Policy compliance that only shows up in prose is unverifiable.** P1.4 says
  never disclose another customer's data. A state-based verifier cannot check
  that the agent didn't *say* something. Those tasks were deliberately left out
  rather than checked badly. If you want them, they need a judge — which is
  exactly what week 5 builds.
- **16 tasks is small.** Confidence intervals on a 16-task set are wide. Get to
  50+ before quoting a headline number, and always report per-trap breakdowns
  rather than a single aggregate.
- **The reference solutions are one correct path, not the only one.** The
  verifier accepts any action sequence that satisfies the constraints, but the
  constraints encode my reading of the policy. Where the policy is ambiguous,
  that is a benchmark bug — file it against `policy.md`, not the task.
- **English only in week 1.** The `language` and `market` fields exist and are
  unused. That is deliberate: get the mechanism right monolingually first.
- **No user simulator yet.** `persona` and `hidden_facts` are written and
  unused. Week 3.

---

## Week 2 handoff

`ReferenceAgent` replays a script. Replace it with a real loop; nothing else
changes. The interface is one method:

```python
class LLMAgent:
    name = "llm-<model>-<strategy>"
    def run(self, task: Task, db: Database) -> int:
        # 1. build system prompt (policy doc, or a search_policy tool for JIT)
        # 2. loop: model -> tool calls -> tools.call(db, name, args) -> observations
        # 3. user turns come from the simulator, seeded with task.persona
        #    and task.hidden_facts
        # 4. respect task.max_turns and a token budget
        # return the number of turns used
```

`tools.schemas()` gives you the function-calling schemas. `tools.call()` is the
dispatcher — it already validates required args and never raises.

Two things to build in from the first version, because retrofitting them is
painful:

1. **Per-step trace logging to disk** (model in, tool calls out, observation,
   token counts). Every result you will eventually report comes from these
   traces. Without them you cannot do error analysis, and error analysis is the
   part that produces findings.
2. **A pluggable context module.** The full-context version is the baseline;
   sliding window, summarisation, note-taking and JIT retrieval are the same
   interface with different implementations. That interface *is* the week-4
   ablation, so define it now.

---

## Scaling the suite (week 3)

`generate.py` expands 16 hand-written tasks to **186** along three orthogonal
axes:

| axis | decides | values |
|---|---|---|
| trap | the **checks** -- what correct behaviour is | 16 |
| market | the **world** -- currency, payment rails, order data | SG MY ID TH PH VN |
| language | the **surface form** -- what the customer types | en, sg-en, ms, id, th, vi |

Orthogonality is the whole point. A Thai task and its English twin share an
identical world instance and the **identical checks object** -- asserted by
`is`, not by equality. Only the words differ. So a gap between the `en` and
`th` rows of the sweep is attributable to language and nothing else. Most
multilingual agent evals compare tasks that also differ in difficulty, and
their language gap is difficulty wearing a costume.

**Every generated task ships with a generated reference solution.** At 16
tasks you can check the suite by reading it; at 186 you cannot, so the
guarantee is mechanical. `tests/test_generated.py` asserts every generated task
is solved by its own solution, fails under the null agent and under tool spam,
and that thresholds land correctly in all six currencies (normal item under the
SGD 200 photo rule, high item over it, voucher under the SGD 15 cap).

### Cost control

```bash
python -m pasarbench.sweep --suite all --sample 2      # 32 tasks, all 16 traps
python -m pasarbench.sweep --suite all --languages en,th   # one language pair
python -m pasarbench.sweep --suite all -k 4            # final numbers
```

`--sample N` is stratified **by trap**, never uniform. A uniform subsample of
30 from 186 misses traps entirely, and a missing trap is precisely the signal
you are trying to read.

### The simulator gets audited too

`simqa.py` checks the thing nobody checks. A simulator that volunteers the
order id in its opening turn converts a multi-turn information-gathering task
into single-turn instruction following: scores rise, the benchmark stops
measuring what it claims, and nothing in the results looks wrong.

It separates **leaked** (fact appeared before the agent asked) from **revealed
on request** (correct behaviour), and flags broken character and hallucinated
order ids. Run it after every persona-prompt change and publish `leak_rate`
next to the pass rates. Above ~0.15, the pass rates describe an easier
benchmark than the one you wrote up.

### Translation provenance -- state this in the writeup

| | |
|---|---|
| en, sg-en | written directly. Singlish is **not** machine-translated: the particles and code-switching are the part that breaks agents, and a translation model smooths them away |
| ms, id | written directly, informal register |
| th, vi | **drafted, not native-reviewed** |

Get Thai and Vietnamese reviewed before quoting a per-language number. Saying
which languages were reviewed and by whom is worth more than a clean-looking
table.

## The context ablation (week 4)

Five strategies behind one interface, swappable with nothing else changing:

| strategy | keeps | cost |
|---|---|---|
| `full` | everything | baseline |
| `window8` / `window4` | system + pinned opening + last N units | cheapest, bluntest |
| `trim3` | every turn, but old tool payloads shrink to their ok/error verdict | conversation structure is cheap; tool JSON is what fills the window |
| `summarize4` | a running structured summary + recent units | one extra model call per fold |
| `notes4` | an agent-written scratchpad + recent units | one extra tool the agent must remember to use |

### JIT is not a context strategy

`policy_mode` (preload vs JIT) governs the **system prompt**; context
strategies govern **conversation history**. They are orthogonal, so sweep them
as a 5x2 grid. Folding them into one axis is why many published context
ablations cannot be read -- a single "JIT" row conflates a smaller system
prompt with a different history policy, and you cannot tell which produced the
number.

### Two decisions inside `Summarize` that decide whether it works

**Structured, not free-text.** A generic "summarise this" prompt reliably drops
exactly what the policy gates on: whether identity was verified, and what the
eligibility check returned. The agent then re-runs a write it already ran, or
acts without verifying. Pinning a schema with `identity_verified` and
`facts_established` as required keys is what makes this strategy competitive
rather than catastrophic.

**Running, not redone.** The summary lives in `EpisodeState` and only new units
are folded in, so cost is O(conversation) not O(conversation^2), and it
survives interrupt/resume. Re-summarising the whole history every step is the
naive implementation and it costs more than the raw context it replaces.
`tests/test_context.py` asserts the backend is called once per new batch.

### `notes4` has a failure mode that hides itself

If the agent never calls `write_note`, the strategy silently degrades to a
naked sliding window with no pinned facts -- and the pass rate still looks like
a real measurement. `note_discipline` is reported in every sweep for exactly
this reason, and the sweep prints a warning below 50%. Publish that number next
to the pass rate or the row is uninterpretable.

### From table to finding

```python
from pasarbench.analyze import what_compaction_lost, markdown_report
what_compaction_lost("traces/run/full", "traces/run/window4")
```

`pareto_frontier` drops strategies that are worse on **both** axes -- showing a
dominated option as a "trade-off" is the most common way an ablation table
misleads. `degradation_matrix` reports per-trap deltas and labels the loss
*concentrated* or *diffuse*, which is the difference between a mechanism and a
vibe. `what_compaction_lost` lists the specific tasks that pass under `full`
and fail under a strategy, classifies what the verifier objected to, and says
what that class of failure usually means:

> *took a forbidden action* -- the agent lost a constraint it had already
> established, most often the eligibility verdict or the payment method. This
> is the expensive failure: a policy violation, not an omission.

The claim to aim at is not "window4 is 4 points worse." It is: **"window4 saves
61% of tokens for 4 points of pass^1, and all of that loss sits in three traps,
every one of which needs a fact established more than four units earlier.
Sliding windows do not fail uniformly -- they fail on long-range dependencies,
and here is the trace."**

## Serving (week 7)

### The metric is cost per *resolved* conversation

Not cost per conversation, and definitely not tokens per dollar. Every
escalated conversation consumes a human agent's time and every failed one comes
back as a second contact, so at any realistic agent wage a few points of
escalation rate outweigh the entire model bill.

Worked example from `tests/test_serving.py` — an 8B FP8 model on one H100
versus a 72B on four:

| | model bill | model share of total | **$/resolved** |
|---|---|---|---|
| `qwen-8b-fp8` | $0.50 | 0.2% | **$2.93** |
| `qwen-72b-bf16` | $3.00 | 2.4% | **$0.77** |

The small model is **6x cheaper to run and 3.8x more expensive per resolved
conversation.** Anyone quoting tokens-per-dollar picks the wrong one. The model
bill is under 3% of total cost in both arms — which is the real reason
throughput is a weak objective on this workload.

`break_even_human_cost` turns that into a decision: the crossover here is
**$0.03 per human contact**, meaning the bigger model wins at essentially any
wage. Note the model separates *escalated* from *failed*: several traps require
escalation, so escalating correctly costs human time without being a quality
problem, and conflating the two would punish the right behaviour.

### No speed win is declared without a proven quality result

`serving_verdict` returns WIN only when the quality difference is **non-inferior
at a stated margin** — not "about the same", which is what eyeballing
overlapping error bars actually means. A 2.5x throughput gain with a 25-point
pass drop returns REJECT.

The third verdict is the honest one and the one people skip. **At 186 tasks you
cannot detect a 2-point regression**; `required_n` says you would need ~6,000
per arm. So the correct answer is often INCONCLUSIVE with a sample size
attached, not a win.

`per_trap_guard` exists because **quantisation damage concentrates**. A model
that loses 1 point overall but 25 points on `cod_cannot_refund_to_original_
method` has not degraded smoothly — it has stopped following one specific rule,
and the aggregate hides that completely. On a policy-following workload this
shows up in the traps long before it shows up in perplexity.

### Measuring vLLM without fooling yourself

Three mistakes, all of which produce a number that looks fine:

**Diff counters, don't read gauges.** `vllm:prefix_cache_hits_total` is
cumulative over the server's lifetime, so reading it once at the end folds in
every experiment you ran before. `MetricsDiff` raises if a counter went
backwards — that means the server restarted mid-run and the numbers must be
discarded rather than clamped.

**Warm up.** The first request pays model load, CUDA graph capture and an empty
cache; including it destroys p99 and flatters whatever you measure next.

**Say that tail percentiles are interpolated.** Prometheus histogram quantiles
are bucket-resolution-limited, and `latency()` carries the widest bucket gap in
its output so you can't quietly present an interpolation as a measurement.

The three-bar result to aim for — preload+cache, preload without cache, and JIT
— has a non-obvious outcome: **JIT sends far fewer tokens and can still lose on
latency**, because the extra `search_policy` round trip costs more than the
cached prefill it saves.

## Tool scaling (week 6)

### Two arms cannot answer this question

"Agents get worse with more tools" is easy to show and useless, because two
things degrade at once: **token cost** (300 schemas is ~25k tokens injected on
every call) and **selection difficulty** (300 near-neighbours make picking
harder). A 20-vs-300 comparison cannot separate them.

| arm | what it isolates |
|---|---|
| `oracle` | ceiling: only the ~6 tools the task needs |
| `all-20` | the real registry, no distractors |
| `all-N` | both effects, N in {50, 100, 300} |
| **`random-N`** | **N tools at random but ALWAYS containing the needed ones** |
| `search-N` | small core + `search_tools`, progressive disclosure |

`random-N` is the arm that does the work: same token cost as `all-N`, same
reachability guarantee as `oracle`. If it tracks `all-N`, the damage is token
cost and dilution. If it tracks `oracle`, the damage is selection. The sweep
**warns you** if you run without an `all-N`/`random-N` pair, because the result
would not be readable.

Measured on the reference path: `oracle` injects 431 schema tokens per call,
`all-100` injects 9,670, and `search-300` injects 445 while reaching 300 tools.

300 distractors are plausible internal marketplace tools (`get_seller_payout`,
`check_risk_velocity_check`, `list_campaign_voucher_pool`) and completely
inert. They deliberately **succeed** rather than erroring — an erroring
distractor would teach the agent that unfamiliar tools fail, which is a
different lesson from the one under test.

Progressive disclosure derives visibility by replaying `search_tools` calls
from the action log, so there is no extra state and interrupt/resume works
unchanged.

## Multilingual diagnosis (week 6)

"Non-English scores lower" is not a finding. The finding is the decomposition,
and this suite can do it because locale twins share an identical world and
byte-identical checks — so the gap itself is attributable to language, and only
the *mechanism* remains to be established.

| mechanism | how it is measured |
|---|---|
| **budget exhaustion** | `max_*` stop reasons by language |
| **tokenisation inflation** | tokens per character vs `en` |
| **retrieval miss** | `search_policy` / `search_tools` returning nothing |
| **malformed tool arguments** | the harness's distinct recoverable error |
| **language drift** | the judge's `language_match` criterion |

Two of these are worth calling out.

**Budget exhaustion is checked first, above every model explanation.** If Thai
episodes hit the step ceiling more often than English ones, the gap is a
*harness artefact*, and every conclusion drawn before raising the budget and
re-running is wrong. Conflating a budget ceiling with a model weakness is the
most common error in multilingual agent evals, and `_next_step` refuses to
suggest anything else until it is ruled out.

**The retrieval mechanism is real and already in the code.** `search_policy`
and `search_tools` are keyword matchers over English text. `test_exposure.py`
demonstrates it directly: `_rank_tools("refund money to the customer")` returns
`issue_refund`; the same query as `คืนเงินให้ลูกค้า` returns **nothing at all**.
That is a property of the system, not a bug to hide — and it is the mechanism
people most often misread as "the model is worse at Thai."

`attribute_gap` carries its own caveat in the output: these are **co-movements,
not causal attributions**. They tell you which traces to read and which
ablation to run. Presenting the correlation table as an explanation is the
mistake the docstring exists to prevent.

## Judge calibration (week 5)

### The judge is for what the verifier structurally cannot see

`verify()` already decides pass/fail programmatically. A judge that re-scores
that adds noise and measures nothing. This one audits only the agent's **prose**:

| criterion | why a state check cannot see it |
|---|---|
| `no_data_leak` **[critical]** | P1.4 is about what the agent *said* |
| `no_hallucinated_facts` **[critical]** | an amount asserted with no tool result behind it |
| `policy_accurate` **[critical]** | what the agent *told* the customer the rules are |
| `verification_before_action` | asking, not just calling |
| `no_unfounded_promise` | a delivery date not in the tracking record |
| `outcome_communicated` | did the customer end up knowing what happens |
| `escalation_explained` | silently escalating is a violation |
| `tone_professional` | the lowest-agreement criterion in any rubric |
| `language_match` | **a model can pass every state check while replying to a Thai customer in English** |

That last row is what turns the multilingual sweep from "non-English scores
lower" into a diagnosis.

### Nine binary criteria, not a 1-5 score

Decomposition helps for three separate reasons, and it is worth being able to
name all three: binary judgments are far more reproducible than ordinal ones;
failure becomes attributable (`kappa 0.89 on data leakage, 0.41 on tone` tells
you which half of the rubric to cut, `kappa 0.71 overall` does not); and
criteria can be weighted by consequence. A data leak is not one fifth of a
five-point scale -- `derive_verdict` makes any critical violation disqualifying
regardless of the rest.

Evidence is required on every judgment. A turn index and a short quote forces
grounding and is the cheapest single thing that raises agreement.

### Report kappa with the number that makes kappa readable

On a criterion satisfied 95% of the time, two labellers can agree 96% and still
score **kappa = 0.32**. Chance agreement is already near-total, so almost
nothing is left for kappa to credit. Every criterion therefore reports five
numbers -- `p_o`, `kappa`, `prevalence`, `bias`, `pabak` -- and
`interpret()` names the paradox explicitly when it sees it, rather than letting
you describe a fine criterion as broken.

Kappa also gets a bootstrap CI. At 200 labels the interval is wide; a move from
0.62 to 0.68 with half-overlapping intervals is not an improvement, and
claiming it is will not survive a careful interviewer.

### The ceiling -- the part almost nobody does

A judge cannot agree with you more than you agree with yourself. Round 2
re-shows a 30-item subset, reshuffled, **never displaying your round-1
answers**, and `ceiling_report` expresses judge agreement as a fraction of that
test-retest kappa:

> kappa 0.71, human test-retest ceiling 0.78, so **91% of achievable
> agreement** -- further prompt tuning is fitting noise.

It also catches the case that matters most: when your *own* ceiling on a
criterion is below 0.4, the verdict says so and tells you to fix the guidance
or cut the criterion rather than tune the judge against a target you cannot
hit yourself.

### Bias probes

`position_bias` requires every pair judged in **both orders** -- a judge that
picks whichever answer is shown first is not comparing content, and
`PairwiseJudge.judge_both_orders` returns no winner when the verdict flips.
`length_bias` measures the correlation between response length and a positive
verdict, because judges reward verbosity by default and in customer service
that is backwards: a correct one-line refusal beats three paragraphs of hedging.

### Labelling discipline

```bash
python -m pasarbench.judge.label sample  --traces traces/run/full --n 200
python -m pasarbench.judge.label annotate --round 1
#   ... a week later ...
python -m pasarbench.judge.label annotate --round 2 --retest 30
```

Three rules the tooling enforces rather than suggests: **label before seeing
any judge output** (anchoring is not recoverable), **the retest round is
blind**, and **sample stratified by trap, language and pass/fail** -- uniform
sampling gives you 180 clean transcripts and near-zero signal on the criteria
that matter. The annotator deliberately hides whether the task passed.

## Post-training (weeks 7+)

The verifier is a **verifiable reward**: programmatic pass/fail, no human and
no learned reward model in the path. That is what makes this an environment
rather than a leaderboard, and it is the part of the project worth leading
with.

```
pasarbench/rl/reward.py    verifier -> reward, with anti-hacking guarantees
pasarbench/rl/collect.py   rollouts -> RFT / DPO / GRPO datasets
scripts/train_rft.py       rejection-sampling SFT (do this first)
scripts/train_grpo.py      GRPO scaffold; the harness IS the rollout engine
scripts/serve_vllm.sh      agent-8b / agent-32b / reference-70b / sim tiers
```

### The reward is designed not to be hacked, and it is tested

The naive shaped reward gives partial credit per satisfied `required_action`,
and a policy learns within a few hundred steps that enumerating every tool
harvests that credit without solving anything. Three defences, all enforced
and all asserted in `tests/test_reward.py`:

1. Partial credit is **clipped below the pass floor** -- a passing episode
   always beats a non-passing one. Asserted as a property across the full task
   set: `min(passing) > max(failing)`.
2. A forbidden action **gates** partial credit to zero rather than subtracting
   from it. Otherwise the policy trades one policy violation for several
   satisfied requirements, which is exactly what a CS org cannot ship.
3. Steps beyond the reference path are penalised, so enumeration costs.

The tool-spam policy is included in the test suite and must score below every
reference rollout. It is how this repo caught that **T15 was weakly specified**
-- an agent that called `get_shipment` incidentally passed it. Fixed by also
requiring `get_order`, which is substantively correct anyway.

### The order of work, and why

| | why |
|---|---|
| 1. RFT / rejection sampling | On a verifiable env this recovers most of the gain with no RL infra. It is also the **only meaningful baseline for GRPO** -- reporting GRPO against the base model conflates "RL worked" with "training on correct trajectories worked". |
| 2. DPO on divergence pairs | Pairs are cut at the first differing decision, so both sides share an identical prefix and the gradient is about one action rather than a whole trajectory. |
| 3. GRPO / RLVR | Group-relative advantage straight off the verifier, no critic. Only worth it once RFT plateaus. |

### The headline claim to aim at

> An 8B policy trained on PasarBench trajectories recovers *X%* of the 70B
> reference's pass^4 at *Y%* of the cost per resolved conversation.

Three models, one table, per-trap breakdown, held-out tasks. **If GRPO does not
beat RFT, report that.** It is a finding, and it is more credible than the
alternative.

### Memory, 8B full-parameter SFT, 4x H100 (320 GB)

| | |
|---|---|
| params bf16 | 16 GB |
| grads bf16 | 16 GB |
| optimiser fp32 m, v + master | 96 GB |
| **states, ZeRO-3 sharded /4** | **32 GB per GPU** |
| activations, seq 8192, grad ckpt | ~15 GB per GPU |

~47 of 80 GB. Comfortable; seq 16384 also fits. A 32B needs LoRA at this scale,
and full-parameter 32B is not worth chasing -- the story is the 8B closing the
gap on the 70B, not training the largest thing that fits.

### Serving results this hardware unlocks

H100-specific and worth reporting because almost every public agent benchmark
runs on A100 or consumer cards and cannot produce them:

- **FP8 W8A8 + FP8 KV cache** (native Hopper tensor cores). Report throughput
  *and* pass^k -- on a policy-following task, quantisation damage shows up in
  the traps long before it shows up in perplexity.
- **FlashAttention-3** (Hopper only).
- **Prefix caching against the ~2,200-token policy prefix** re-sent on every
  step. The non-obvious result: JIT mode sends far fewer tokens and can still
  *lose* on latency, because the extra `search_policy` round trip costs more
  than the cached prefill it saves.

## What is still stubbed

- `LLMUser` is written but unused; week 3 turns it on and tunes the persona
  prompt. Until then episodes are single-turn via `SilentUser`.
- `scripts/train_grpo.py` collects and scores groups but does not yet do the
  policy update. The three remaining pieces -- token alignment between vLLM and
  HF, turn-level loss masking, per-token advantage normalisation -- are named
  explicitly in that file rather than glossed.
- Negative "do no harm" traps (`cannot_cancel_shipped_order`,
  `peak_period_delay_not_compensable`, `address_change_after_dispatch`) are
  inherently weak under state-only verification: passing means nothing bad
  happened, which a cautious lookup-only agent achieves for free. 33 of the 186
  tasks are in this category and `tests/test_generated.py` exempts them
  explicitly rather than pretending otherwise. They are the strongest argument
  for the week-5 judge.
- `LLMUser` is now wired into `train_rft.py` and `sweep.py` but has not been
  run against a real model. Its persona prompt is unvalidated until `simqa`
  reports a leak rate on actual transcripts.
- Token counts from backends that do not report usage come from a 4-chars-per-
  token approximation, which is roughly right for English and roughly WRONG for
  Thai and Vietnamese. The trace records which backend supplied each number.
  Never quote an approximated figure as measured.

## Expanding the task set

To 30+: each trap generalises across markets. T02 (COD) works identically in
ID, PH and VN. T07 (photo threshold) works on any item over SGD 200. Clone the
task, change the order and the persona, keep the trap label stable so per-trap
grouping still works.

To multilingual: keep `task_id` stable and suffix the locale (`T08.vi`,
`T08.sg-en` for Singlish). Translate `opening` and `persona`; leave `checks`
untouched. Identical checks across locales are what let you attribute a score
drop to language rather than to task difficulty — which is the whole point of
the multilingual diagnosis.

Do **not** machine-translate the Singlish variants. Write them yourself or get
a native speaker. Code-switching is the part that actually breaks agents, and
a translation model will smooth it out into standard English.
