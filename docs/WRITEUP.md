# Writeup kit

Blog post, resume lines, interview prep.

**Every number in this file is `[N]`.** Fill them only from `RESULTS.md`, which
is generated from files on disk and refuses to fabricate. Do not hand-fill
either document — run the command, regenerate, copy across. A published post
containing two numbers you never ran is a worse outcome than no post.

---

# Part 1 — The blog post

**Target:** 1,800–2,400 words. One idea, three results, one honest limitation.

**Title options** (lead with the finding, not the artefact):
- *Your multilingual agent eval is measuring your token budget*
- *We built a customer-service agent benchmark in six SEA languages. The
  language gap wasn't the model.*
- *Cheaper per token, more expensive per resolution*

Pick one. The first is the strongest: it promises a specific, contrarian,
checkable claim.

---

### Opening — 150 words, lead with the trap

Do **not** open with "I built a benchmark." Open with the finding.

> An agent handling a Thai customer resolved `[N]%` of cases. The same agent,
> handling the identical customer in English — same order, same database, same
> pass criteria, only the words changed — resolved `[N]%`.
>
> The obvious conclusion is that the model is worse at Thai. It is also, at
> least partly, wrong. `[N]` points of that gap came from episodes hitting the
> step budget, because Thai tokenises `[N]x` worse and the same conversation
> ran out of room. Another `[N]` points came from the policy retriever, which
> is a keyword matcher over English text and returns nothing at all for
> `คืนเงินให้ลูกค้า`.
>
> Neither is a model weakness. Both are mine.

### Section 1 — Why locale twins (300 words)

The argument, in order: most multilingual evals compare tasks that also differ
in difficulty, so the language gap is difficulty wearing a costume. Three
orthogonal axes — trap decides the checks, market decides the world, language
decides only the surface form. Twins share the **identical checks object**,
asserted with `is`, not equality.

That design is what licenses the whole post. Say so plainly, then show the
assertion.

### Section 2 — The verifier (250 words)

State-based, not text-similarity: verify the final database, not the prose.
Tools enforce data integrity, not policy judgment — `issue_refund` blocks a COD
refund because there is no instrument, but `initiate_return` will happily open a
return 19 days after delivery, because catching *that* is the agent's job. If
the tools enforced everything, every agent would score 100%.

One paragraph on the flagship trap: delivered 15 days ago, every surface signal
says out-of-window, but a later policy section waives the window when a
livestream claim went unfulfilled. Models that pattern-match on "14 days" fail
it and the trace shows exactly why.

### Section 3 — The diagnosis (600 words, the core)

Table: pass rate, malformed-arg rate, search-miss rate, language-drift rate,
budget-exhaustion rate, tokens-per-char, by language.

Then walk the mechanisms **in the order you must rule them out**:

1. **Budget exhaustion first.** If non-English episodes hit the ceiling more
   often, the gap is a harness artefact and every other conclusion drawn before
   raising the budget is wrong. `[N]` points here.
2. **Tokenisation inflation.** `[N]x` tokens per character vs English.
3. **Retrieval miss.** Show the two-line demo: the English query returns
   `issue_refund`; the Thai query returns nothing. `[N]` points.
4. **Malformed tool arguments.** `[N]%` of non-Latin episodes vs `[N]%` English.
5. **Language drift.** The agent replies in English to a Thai customer — passes
   every state check, fails the judge's `language_match`. `[N]%`.

Then the caveat, in your own voice: **these are co-movements, not causal
attributions.** They told me which traces to read. State which mechanism you
then confirmed by reading, and how.

### Section 4 — Two shorter results (400 words)

**Tool scaling.** The confound most people ship: token cost and selection
difficulty degrade together, so a 20-vs-300 comparison is unreadable. The
`random-N` arm — N tools at random but always containing the needed ones —
separates them. Verdict: `[SELECTION | TOKEN COST | both]`. `search-N` recovers
`[N]` points while injecting `[N]` fewer schema tokens per call.

**Cost.** The inversion: `[model A]` is `[N]x` cheaper per token and `[N]x` more
expensive per **resolved** conversation, because escalations cost human agent
time and the model bill is `[N]%` of total cost. Break-even at `$[N]` per human
contact.

### Section 5 — What failed (300 words)

Pick two from `docs/WHAT_FAILED.md`. Recommended: the `deepcopy` aliasing bug
(silent, invisible at k=1, corrupted every `pass^k`, found by scaling rather
than by review) and the kappa-paradox test you got wrong. The second is
counter-intuitively the more persuasive one.

### Closing — 150 words

The limitation, stated first and without hedging: **186 tasks cannot detect a
2-point regression** — `required_n` says ~6,000 per arm — so several verdicts
in this work are honestly INCONCLUSIVE. Thai and Vietnamese translations are
drafted and not native-reviewed. 33 of 186 tasks are "do no harm" traps that a
cautious lookup-only agent passes for free.

Then the repo link. No call to action, no "reach out." The work is the pitch.

---

# Part 2 — Resume lines

Rules: one line each, a **number**, and the mechanism. "Built an AI agent with
LangChain" is worth nothing; the number and the mechanism are the entire value.

Pick three. Do not list all six.

> **Built a verifiable multi-turn agent environment** for SEA e-commerce support
> — 186 tasks across 6 markets and 6 language varieties, 20 tools, state-based
> verification. Generated tasks ship with generated reference solutions; the
> suite asserts every task is solvable and every trap discriminative (`[N]`
> automated invariants).

> **Wrote the agent runtime from scratch** (no framework) — tool dispatch,
> budgets, structured error recovery, interrupt/resume, per-step tracing.
> Ablated 5 context strategies: `[N]%` token reduction at `[N]` points of
> pass^1, with the loss concentrated in `[N]` traps requiring long-range
> dependencies.

> **Isolated the cause of a `[N]`-point multilingual performance gap** using
> locale twins with byte-identical pass criteria, attributing it across budget
> exhaustion, `[N]x` tokenisation inflation, and an English-only policy
> retriever rather than to model capability.

> **Calibrated an LLM-as-judge against 200 hand-labelled transcripts** —
> decomposed 9-criterion rubric raised Cohen's κ from `[N]` to `[N]`, reported
> against a `[N]` human test-retest ceiling (`[N]%` of achievable agreement).

> **Separated token cost from selection difficulty in tool scaling** with a
> control arm holding schema size constant while guaranteeing reachability;
> retrieval-based exposure recovered `[N]` points at `[N]` fewer schema tokens
> per call.

> **Post-trained an 8B policy against the environment's own verifiable reward**
> — rejection-sampling SFT then GRPO, recovering `[N]%` of a 72B reference's
> pass^4 at `[N]%` of cost per resolved conversation.

---

# Part 3 — Interview prep

### The 60-second version

> I built a customer-service agent environment for Southeast Asian e-commerce —
> 186 tasks, six markets, six languages. Verification is on final database
> state, not text, which means it's also a verifiable reward, so I could train
> against it rather than just measure.
>
> The result I'd lead with: I found a `[N]`-point gap between Thai and English,
> then decomposed it. Because locale twins share an identical world and
> byte-identical pass criteria, the gap is attributable to language — and most
> of it turned out to be my harness, not the model. Budget exhaustion from
> tokenisation inflation, and a policy retriever that's a keyword matcher over
> English text.
>
> The thing I'd want to talk about is the verifier design, because that's where
> most homemade agent benchmarks go wrong.

### Five stories worth having ready

**1. Why tools enforce integrity but not policy.** `issue_refund` blocks a COD
refund; `initiate_return` allows an out-of-window return. If tools enforced
everything the benchmark would measure nothing. This is the single best
signal that you understand what a benchmark is *for*.

**2. The aliasing bug.** Silent, invisible at k=1, corrupted every `pass^k`,
found by scaling to 170 tasks rather than by review. Lands the point that
guarantees catch bug classes and code review catches instances.

**3. The `random-N` arm.** Two arms cannot separate token cost from selection
difficulty. Shows experimental design, not implementation.

**4. The judge ceiling.** "κ 0.71 against a 0.78 human test-retest ceiling, so
91% of achievable agreement — further tuning is fitting noise." Almost nobody
measures their own ceiling.

**5. Cost per resolved conversation.** The inversion, and the fact that the
model bill is a low single-digit percentage of total cost. Reframes an
inference question as a business one.

### Questions you will be asked

**"How do you know the tasks are correct?"** Every generated task ships with a
generated reference solution and the suite asserts it passes; ten naive-but-
plausible solutions must fail; a tool-spam adversary must fail everything that
requires action. That last one found a real defect in T15.

**"Isn't LLM-as-judge unreliable?"** Yes, which is why I measured it rather than
assumed. Naive 1-5 scoring reached κ `[N]`; a 9-criterion binary rubric with
required evidence reached `[N]`. And I reported it against my own test-retest
ceiling, because a judge cannot beat that.

**"Why not just use τ-bench?"** I did use its core design decision — verify
final state, not text. What it doesn't have is COD, livestream disputes,
multi-currency with zero-minor-unit currencies, or six languages with identical
checks, and those are exactly the SEA cases this role touches.

**"Did GRPO beat RFT?"** `[answer honestly]`. If not, say so — it's a finding,
and RFT is the only meaningful baseline for GRPO anyway. Claiming a win against
the base model instead conflates "RL worked" with "training on correct
trajectories worked."

**"What would you do next?"** Three, in order: get Thai and Vietnamese
natively reviewed; grow the suite toward the `required_n` needed to detect a
2-point regression; replace the keyword policy retriever with multilingual
embeddings and re-measure the gap — the diagnosis predicts `[N]` points should
close, which is a falsifiable claim.

### One thing not to do

Do not present the co-movement table as an explanation. Say "this narrowed
which traces to read, and here's what I found when I read them." The distinction
is small and it is exactly the kind of thing a research engineer notices.
