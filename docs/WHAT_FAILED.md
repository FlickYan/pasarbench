# What failed

Fifteen real defects found while building this, in the order of how much
damage they would have done. Every one is reproducible from the git history.

The pattern worth extracting: **almost none of the serious bugs were found by
reading code.** They were found by mechanisms built to catch a whole class — a
reference solution that must pass, an adversarial solution that must fail, a
scale-up that exercises a code path differently, a real model run. That is the
argument for spending the first two weeks on guarantees rather than features.

The exception is #9, which no mechanism here could have caught and a human
reading transcripts did. Automation catches the classes you thought of.

---

## 1. Patch rows were aliased into the database — every `pass^k` was silently wrong

**What happened.** `Database.fresh(patch)` inserted rows for new ids by
reference:

```python
db.tables[table][row_id] = fields     # the task's own dict, not a copy
```

The first episode's tool calls then mutated the task definition itself. Run 2
of the same task started from a world where the order had already been
cancelled and the return was already open.

**How it was found.** Not by reading it. By generating 170 tasks whose locale
twins reuse the same world rows, which made the corruption visible as nine
tasks failing their own reference solution — but only the non-English variants,
because they ran second.

**Why it matters.** Invisible at `k=1`. Every `pass^k` number the project would
ever have produced was wrong, and nothing in the output would have said so. The
hand-written 16 tasks never triggered it because they patch *existing* rows,
where `.update()` copies values in.

**Fix.** One `copy.deepcopy`. The comment explaining why is longer than the fix,
deliberately.

---

## 2. The tool set was computed once per episode

**What happened.** `run_episode` built the function-calling schema list once,
before the loop.

**Why it matters.** Correct until progressive disclosure existed. The moment
`search_tools` could reveal new tools mid-episode, the agent could be told about
a tool it was never actually offered — or worse, the arm labelled `search-300`
would have measured something other than search.

**How it was found.** Writing the exposure test. The bug never fired because no
feature had needed per-step recomputation before.

**Lesson.** A correct assumption has a shelf life. This one expired silently
two weeks after it was written.

---

## 3. A policy threshold collided with the happy-path task

**What happened.** `check_return_eligibility` flagged photo evidence as required
on item value alone. The happy-path task was a SGD 219 keyboard returned for
change of mind — over the SGD 200 threshold, so the "clean" task demanded photo
evidence for a non-defect return.

**How it was found.** Writing the reference solution. The task was unsolvable as
specified and there was no way to see that by reading the policy.

**Fix.** Two changes, and the second is the real one: lower the item price, *and*
reword P3.3 so the photo rule applies to **defect or discrepancy claims**, not
to value alone. The policy was genuinely ambiguous; the collision exposed it.

**Lesson.** Writing reference solutions is not busywork to satisfy a test. It is
the only mechanism that reads your policy adversarially.

---

## 4. A rule landed exactly on its own boundary

**What happened.** The customs-hold task sat at 5 days 15 hours against a policy
that says "more than 5 days." Python's `timedelta.days` truncates to 5, so the
correct answer was genuinely undefined — a model could reason either way and be
right.

**How it was found.** A world-sanity assertion in the adversarial test suite.

**Fix.** Move the timestamp to 6 days 15 hours. Not reword the policy — a real
policy has boundaries, and the benchmark's job is to not test them by accident.

---

## 5. A task passed under tool spam

**What happened.** `T15` (address change after dispatch) required only
`get_shipment` and forbade `modify_shipping_address`. An agent that enumerated
tools called `get_shipment` incidentally, changed nothing, and passed.

**How it was found.** The reward-hacking adversary in `tests/test_reward.py` —
a policy that calls every tool to harvest partial credit. It passed T15, which
was supposed to be impossible.

**Fix.** Also require `get_order`. Substantively correct anyway: you cannot tell
a customer the address is locked without checking the order status.

**The honest residue.** "Do no harm" traps are structurally weak under
state-only verification — passing means nothing bad happened, which a cautious
lookup-only agent achieves for free. 36 of 199 tasks are in this category and
`tests/test_generated.py` **exempts them explicitly** rather than pretending
otherwise. They are the argument for the judge.

---

## 6. A policy rule had two reasonable readings, and the verifier accepted one

**What happened.** P4.4 read *"A refund for a returnable item is issued only
after the return is initiated."* A model handled the happy-path return
correctly right up to the end, opened the return, told the customer it was
approved, and stopped — never issuing the refund. Real shops refund when the
item arrives back, so "only after" is naturally read as a precedence
constraint rather than an obligation to act now.

**How it was found.** Running the suite against a stronger model. The weaker
`deepseek-flash` passed the task; `deepseek-v4-pro` reasoned its way into the
*other* valid reading and was marked wrong. **The benchmark was penalising the
more careful reader.**

**Why it matters.** The task was measuring whether a model guesses the author's
intent, not whether it follows the policy. Left unfixed, it would have shown up
as "the stronger model is worse at returns", which is a confidently wrong
finding and exactly the kind that survives into a writeup.

**Fix.** Reword the policy, not the task: the refund must now explicitly be
issued in the same conversation. Reference solutions already did this, so every
suite stayed green.

**Lesson.** Where a policy is ambiguous, that is a benchmark bug — file it
against `policy.md`, never against the model. A rule only one reading satisfies
is a rule that tests mind-reading.

---

## 7. Customer turns were displayed one step late

**What happened.** The trace inspector numbered user turns from 0, but the loop
increments `tracker.turns` *before* writing the event, so they start at 1. Every
customer line rendered one speaking step later than it happened.

**Why it matters more than a display bug should.** It made the agent look
clairvoyant — calling `get_order("O1003")` two steps *before* the customer
supplied the order id, and verifying `phone_last4=4567` before it was given.
Reading that trace, the obvious conclusion is that the simulator is leaking
answers into the prompt. It was not. An off-by-one in a debugging view came
within one step of producing a fabricated finding about evaluation integrity.

---

## 8. A task was unsolvable by anything that had to look

**What happened.** `duplicate_refund_escalate` requires the agent to notice a
refund has already been issued and escalate rather than pay twice. It scored
**0.00 across every episode**. Reading the tool registry explains why: nothing
exposed prior refunds. `get_order` returned the order, `get_payment` returned
the payment, `check_return_eligibility` returned an existing-*return* count.
Nothing returned existing refunds. An agent could not have known.

**Why the reference-solution guarantee missed it.** This is the important part.
The reference solution passes because it calls
`escalate_to_human(category="duplicate_refund")` directly -- it is an **oracle**.
It already knows the answer, so it never needs to discover anything.

> A reference solution proves a task is solvable BY SOMETHING THAT KNOWS THE
> ANSWER. It does not prove the task is solvable by an agent that can only
> observe.

That is a real limitation of the central guarantee in this repo, and it took a
real model run to expose it. The adversarial suite could not catch it either:
tool spam and naive solutions test whether wrong behaviour is rejected, not
whether right behaviour is *reachable*.

**Fix.** `get_order` now returns `existing_refunds` and `existing_returns`.

**What to add.** A third guarantee alongside solvable-and-failable:
**observable** -- for every required action, is the information that motivates
it reachable through some read tool from the starting state? That check is
mechanisable and this repo does not yet have it.

---

## 9. Order IDs contained the answer

**What happened.** Generated identifiers were built from the trap name:

```python
tag = f"{trap[:6]}{market}".replace("_", "")   # "cannot" + "TH"
oid = f"GO-{tag}"                              # GO-cannotTH
```

So `GO-duplicMY` carried "duplic" from `duplicate_refund_escalate`,
`GO-hazmatID` said "hazmat" outright, `GO-livestMY` said "livest". The customer
hands that string to the agent in turn one. **The case type was in the prompt
before a single tool call.** Separately, every generated user verified with
`0000`, so the identity answer was learnable rather than askable, and the
identity-failure trap (whose customer says `1234`) was trivially distinguishable.

**How it was found.** Not by any test. By a human reading transcripts during
judge labelling and noticing the ids all looked alike.

**Why nothing caught it.** Every guarantee in this repo asks *is the right
behaviour reachable, and is the wrong behaviour rejected*. None of them asks
**is the answer reachable too cheaply**. Reference solutions, adversarial
solutions and observability checks are all blind to a leak that makes a task
easier rather than impossible.

**Fix.** Ids are now a truncated hash of (trap, market) -- deterministic, so
runs stay reproducible and trace filenames stay stable, but carrying no signal.
Phone digits are per-user, derived the same way, excluding `1234` and
all-same-digit strings.

**What this costs.** Every pass rate measured before this fix was taken under a
prompt that contained a hint. Whether the model used it is unknown; the point
is that it cannot be ruled out, so those numbers are not the ones to publish.

**The missing guarantee.** Alongside solvable, failable and observable:
**uninformative** -- no identifier, name or field visible to the agent may
correlate with the task's label. Mechanisable, and this repo did not have it.

---

## 10. I quantified an effect that did not replicate

**What happened.** Comparing runs B and C, the `duplicate_refund_escalate` trap
dropped 0.70 -> 0.41 after semantic order ids were replaced with hashes. The
95% CI was [+0.120, +0.460] -- it excluded zero, so I reported it as a measured
effect: *"a leaky identifier was worth 29 points."*

**The replication.** Run D used the identical 186-task set with only the id
change, and scored **0.55** on the same trap. Diff +0.150, CI [-0.029, +0.329].
Crosses zero.

C and D disagree by 14 points on the same trap running the same code. At ~55-65
episodes per arm, a 15-point effect is unresolvable -- `required_n` says ~168
episodes per arm -- and one CI happening to exclude zero is what small samples
do sometimes.

**Why this is the worst one on the list.** Every guard needed to prevent it was
already in this repo and written by me. `serving_verdict` refuses to declare a
win without non-inferiority. `required_n` exists to say "you cannot see this at
your sample size." `README.md` states that 186 tasks cannot detect a 2-point
regression. I bypassed all of it because a single number looked clean.

**The correct claim.** The identifier leak was a real defect and removing it was
right regardless. Its measured effect on pass rate: **not resolvable at n=930**
(pass^1 +0.018, CI [-0.008, +0.044]). A confound does not need to move the
number to be worth fixing.

**Lesson.** Replication beats significance. One comparison at small n is a
hypothesis, not a result -- especially when you are the one who wanted it to be
true.

---

## 11. The simulator obeyed the rules only in some languages

**What happened.** The persona prompt instructs the customer to reveal facts
only when asked. Run D's leak audit: **132 leak events, and every single one in
Indonesian, Malay or Thai. Zero in English, Singlish or Vietnamese.**

**Why it is not a script or tokenisation story.** Indonesian and Malay are
Latin-script. Thai is not. Vietnamese is Latin-with-diacritics and is perfectly
clean. The split does not follow script, so it is instruction-following under
language shift -- and specifically a NEGATIVE constraint ("do not say X until
asked"), which is the hardest kind to hold.

**Why it mattered more than a QA nuisance.** It biased exactly the measurement
the suite exists for. Leaked facts make a task easier, so Indonesian, Malay and
Thai were being scored under a more helpful customer than English was. The
language null holds cleanly only for the languages with zero leaks.

**Attempted fix, and it BACKFIRED.** Restating the hold-back rule in the target
language raised the leak rate from 11.3% to 22.4%. Chinese, added in the same
round, leaked heavily from the start. A plausible mechanism: a negative
instruction that names the forbidden fact ("do not say your order number")
raises that fact's salience -- the don't-think-of-an-elephant problem, and
exactly the wrong tool for the job.

**The actual fix: stop asking.** `--gate-facts` withholds each fact from the
simulator's own prompt until the agent has asked for it. You cannot leak what
you were never told. A fallback releases everything after six turns so a missed
ask-pattern cannot stall an episode.

**And the structural fix broke the experiment.** Gating dropped the leak rate
(22.8% -> 15.3%) and destroyed the thing it was protecting. Indonesian fell to
0.463 and Chinese to 0.585/0.650 while English held at 0.910, and EVERY trap
dropped, including ones that had been at 1.00.

The cause: the gate decides "has the agent asked?" using `ASK_PATTERNS`, which
are English regexes plus a few translations and contain **nothing for Chinese**.
The agent asks `请提供您的订单号`, no pattern matches, the fact is never
released, the customer stonewalls, and the episode fails. That is a
language-dependent handicap far LARGER than the leak it removed -- and it looks
exactly like a model capability result.

**Three attempts, three failures, escalating in damage:**

| attempt | leak rate | what it cost |
|---|---|---|
| tighten the English prompt | 18.8% -> 14.6% | nothing, barely helped |
| restate the rule in-language | 14.6% -> 22.4% | made it worse |
| withhold facts until asked | 22.8% -> 15.3% | **broke the comparison entirely** |

**Fix.** `--gate-facts` now raises on any language without ask-patterns rather
than silently handicapping it. Run G is discarded.

**Lesson, and it is the one worth telling.** I spent three runs and roughly ten
dollars trying to eliminate a bias I had already measured and bounded. The
correct move after attempt one was to report the bound: *Vietnamese and Singlish
give a clean null; Indonesian, Malay, Thai and Chinese were measured under a
more helpful customer, so their nulls are lower bounds.* A measured, stated
limitation is a result. A fix that introduces a bigger confound is not.

**The second-order finding.** The capability degradation this project set out to
find in the *agent* showed up in the *simulator* instead -- because the
simulator's instruction is harder. Answering a question in Thai is easy;
declining to volunteer something in Thai is not.

---

## 12. The kappa-paradox test did not contain a paradox

**What happened.** The test asserted that 96%-agreement data would produce a low
kappa. It produced **kappa = 0.74**. The test failed and the implementation was
correct.

**Why it matters.** High raw agreement alone does not create the paradox — the
disagreements have to consume most of the tiny minority class. The corrected
case (98/2 vs 96/4) gives p_o = 0.96 with kappa = 0.315.

**Lesson.** This is precisely how people mis-describe their own agreement
results, so the wrong version is kept in the test file as a comment rather than
deleted. Being able to say *"I got this wrong first"* about a statistic is worth
more in an interview than the statistic.

---

## 13. A test asserted on state the code had already mutated

**What happened.** The interrupt/resume test snapshotted an episode, restored
it, resumed — then asserted `state2.step == 2`. But `run_episode` mutates
`state2` in place, so by assertion time it had advanced to 5.

**Fix.** Capture the value before resuming. Test bug, not code bug.

---

## 14. Two locale variants were byte-identical

**What happened.** The Singlish and English openings for
`identity_verification_failure` were the same string. The "twins differ in the
opening" invariant caught it.

**Why it matters more than it looks.** The whole multilingual claim rests on
locale twins differing *only* in surface form. A twin that does not differ at
all is a silent duplicate inflating the English sample.

---

## 15. The distractor pool ran out before 300

**What happened.** `all-300` produced a 258-tool registry. Domain × field × verb
combinatorics topped out at 235 distractors.

**Fix.** More domains. Trivial — but the arm was mislabelled until it was
caught, and a mislabelled arm is a wrong result, not a cosmetic one.

---

## 16. The label format could not tell one rater from two

**What happened.** Round 2 of labelling finished 49 minutes after round 1. I
read the timestamps as a same-sitting test-retest — recall contaminating the
second pass — and shipped a caveat into `make_report.py` saying the ceiling was
inflated and the judge's fraction of it therefore *conservative*. Round 2 had
been labelled by a different person. `--labeller` defaulted to `"me"`, so the
file recorded one rater twice, and the caveat asserted the wrong statistic with
the wrong direction of bias.

**Why it matters.** Test-retest (one rater, twice) and inter-annotator (two
raters, once) are the same arithmetic on files with the same shape, and their
biases on `fraction_of_ceiling` point in *opposite* directions. Same-sitting
retest inflates the denominator and understates the judge; a second rater
agrees less than you do with yourself, which shrinks the denominator and
*flatters* the judge. Getting the kind wrong does not add noise to the
headline number — it inverts how the headline can be read.

**Fix.** The report now reads the `labeller` field, names which kind of
agreement it is computing, and refuses to interpret when the rater is not
recorded. `label set-rater` stamps a round after the fact without touching any
label. For the inter-annotator case it prints each rater's satisfied rate and a
per-language breakdown of the second rater's, because a second rater who
cannot read Thai presses enter on Thai transcripts — and enter means
SATISFIED. The aggregate rate averages that away; the per-language spread does
not.

**Lesson.** Metadata that decides how a number may be read belongs in the
record, not in the memory of whoever ran the command. A default of `"me"` was
a guess about the user dressed up as a value.

---

## 17. The naive baseline vanished from the report without an error

**What happened.** 200 naive-judge verdicts were written to disk and the
report showed nothing about them — no number, and no `NOT MEASURED` either.
The report compared the naive judge's single bit, `overall_acceptable`,
against the human's nine-criterion labels. The key exists on one side only, so
`per_criterion` found no pairs, returned an empty dict, and the `if` guarding
the output line was false. `run_judges.py` even wrote the matching human bit to
`human_overall.jsonl` for exactly this comparison. Nothing read it.

Two smaller defects surfaced in the same pass. The unmeasured count was
`body.count("**NOT MEASURED**")`, which included the legend line explaining
the marker, so it could never reach zero. And the diagnosis printed when the
decomposed judge lost to the naive one blamed unparsed criteria defaulting to
SATISFIED — a cause that can only make a judge *more lenient* — for a loss
that, on the fixture that exposed it, was entirely false alarms.

**Why it matters.** The report's design rule is that every gap prints
`NOT MEASURED` with the command that fills it. This gap printed nothing,
because the marker lives on the path for *no data*, and the data existed.
Rule 4 — report the naive baseline — was violated by the tool that states it.

**Fix.** One definition of the one-bit target, `rubric.overall_acceptable`,
used for the human, the naive judge and the collapsed decomposed judge, so the
two kappas answer the same question on the same transcripts. The comparison is
a paired bootstrap of the difference, not two CIs side by side. A naive
response with no valid 1–5 score is excluded rather than scored as a fail. If
the data exists but nothing aligns, the section now says so under the marker.
The count is per section and names them. The loss diagnosis reads the
direction of disagreement — harsh or lenient — off the confusion matrix, and
the per-criterion table carries both columns so the harsh criteria can be
found. A test builds the exact shapes that failed and asserts the baseline
reaches the page.

**Lesson.** "Print a marker when data is missing" only covers the branch where
the code notices. Absence has to be asserted where the *output* is built, not
inferred from where the input is loaded.

---

## 18. The ceiling arm was below the floor — defect #5, rebuilt

**What happened.** In the tool-scaling run the `oracle` arm — only the tools
the task needs, the designed *ceiling* — scored 0.656. `all-20`, the plain
registry, scored 0.938. Four traps scored exactly 0.00 under oracle and were
passed under all-20.

The oracle exposed "the tools the reference solution calls". The reference
solution passes `order_item_id="OI3"` because it already knows; it never calls
`get_order_items`. A real agent can only learn `OI3` from `get_order_items`, so
on every trap whose resolution needs an item — returns, eligibility checks, a
voucher capped at 20% of *item* value — the arm was unsolvable. The prediction
"zero exactly where the verifier needs an item id" matched 15 of 16 traps; the
sixteenth is the voucher, where the policy needs the item's price. The
`random-100` control guarantees the same set, and on the 32 sampled tasks it
happened to omit `get_order_items` for both tasks of exactly those four traps:
the same four zeros.

This is #5 again. `get_order` carries a comment saying a reference solution
proves a task is solvable by something that knows the answer, not by something
that has to find out. That lesson was applied to the database payload and not
to the one other place that derives "what an agent needs" from a solution.

Found in the same pass: the report read the run as **"no degradation at 100
tools"** — every arm beat the broken oracle, so every "drop" was negative. The
summary pointed all five arms' `trace_dir` at one directory that does not
exist. And at 100 tools, 13–30 episodes per arm hit the per-episode token
budget, so pass^1 mixed cost with tool choice.

**Fix.** `needed_tools` adds, for every argument a solution hard-codes that a
real agent must discover, the tool that reveals it; oracle and random-N share
it. A test walks all 215 tasks and fails if any hard-coded id is undiscoverable
in either arm. The report now checks the ceiling *before* reading anything
against it, names structural zeros, reports budget stops and a finished-only
pass rate as their own columns, and compares retrieval against the real
registry. `trace_dir` records the cell.

**Second pass — the first fix was a list of cases.** It added the lookup that
reveals each value a solution hard-codes. The audit of the same run then showed
random-100 agents calling `get_livestream_claims` on tasks whose solution never
does: the policy lets a livestream claim override the return window, so a
careful agent rules that out before denying a return. The reference solution
skips the check because it already knows there is no claim — #5 a third time,
now for *exceptions to rule out* rather than values to learn. Under the
enforced loop the patched oracle would have rejected that call.

So the rule became structural: **an oracle may remove choices, never
information.** Every read-only lookup is exposed; only the write actions the
task does not take are hidden. A test calls each lookup against a live
database and fails if one changes a table — the rule rests on that promise.

**Lesson.** The design had an invariant — ceiling ≥ baseline — and nothing
asserted it. An invariant that is not checked in code is not a property of the
experiment; it is a hope about it. And a fix that enumerates the cases found so
far is a fix for those cases; the next one arrives with the next run.

---

## 19. The arms controlled what the agent was told, not what it could do

**What happened.** Every exposure arm sent the model only the visible tool
schemas, but the loop *dispatched* any registered tool by name. An agent that
guessed `issue_refund` — not a hard guess — got a working tool the arm was
supposed to hide. Found while testing the audit script: under `search-300` the
scripted agent never searched and made 56 successful calls to tools it was
never shown.

**Why it matters.** A reduced arm measures "can the agent find and pick the
right tool". With the leak it also measured "can it guess a tool name", and
the two are inseparable from the pass rate. Default-exposure runs — every
language and context result — show all real tools, so nothing was hidden and
they are unaffected. The tool-scaling arms were.

**Fix.** A call to a tool not visible at that step returns `unknown tool` —
not "hidden", which would leak that it exists — never touches the database,
and is logged as a `hidden_tool_call` event. `scripts/audit_tool_arms.py`
replays each arm's visibility from traces recorded *before* the fix, matching
the recorded schema cost at step 1 to decide which arm definition produced
them, and counts how many calls in a past run went through the gap.

**Lesson.** An exposure experiment has to control the action space, not just
the prompt.

---

## 20. One arm's distractors leaked into the next arm's search

**What happened.** Distractor tools register into one global registry, and the
search ranker ranked over all of it. The arms of a sweep run in one process, in
order: by the time `search-300` ran, `random-100` had registered 300
distractors, so "search-300" was searching 320 tools. Run alone, or first, it
would have searched 300. An arm's difficulty depended on which arms happened to
run before it.

**Why it went unnoticed.** Distractor lists are prefix-stable, so the extra 20
were a consistent superset and every run with the same arm order reproduced
exactly. Reproducible is not the same as correct: the label said 300.

**Fix.** Each search arm ranks within its own tools only — the 20 real ones
plus its own distractors — and the `search_tools` result the agent reads comes
from the same pool as the visibility check, so the agent cannot be shown a
tool it then cannot call. A test registers extra distractors first and asserts
none ever reaches the ranking. The audit replays both behaviours and reports
which one reproduces a trace at every step.

---

## 21. The tool-scaling verdict pointed the wrong way, and fired on noise

**What happened.** On the first valid tool-scaling run, oracle passed 94 of 96
episodes and both 100-tool arms passed 89. The report printed "SELECTION
difficulty, not token cost". Two things were wrong with that line.

It named a mechanism from a five-episode gap. Paired by task, that difference
is a handful of tasks worse and none better — a sign test nowhere near
significance. The reading code had no noise check at all; it compared point
estimates against 0.7× and 0.3× thresholds.

And the mapping was inverted. `all-100` and `random-100` hold the same number
of tools at the same schema cost; they differ only in composition — `all-100`
keeps every unneeded real write tool, `random-100` swaps most of them for
distractors. So a loss that `random-100` shares cannot come from the real
tools it dropped; it comes with the *number*. The code called that case
"selection", and the module docstring beside it said the opposite. The labels
also overclaimed: at a fixed N these arms cannot separate token cost from
count-driven difficulty, because both move together.

**Why it survived.** Its test asserted the same inverted mapping. A test
written from the same misunderstanding as the code proves the two agree, not
that either is right. It was found by asking what an exact tie between the two
100-tool arms would *mean*, when the first real run produced one.

**Fix.** Every contrast is a paired sign test over tasks before it is read;
the unit is the task, not the episode. An unresolved loss prints INCONCLUSIVE
with the numbers, and the cost — which is certain — beside it. The composition
verdict is re-derived and named for what these arms can distinguish. A trap
both 100-tool arms lose on is printed as a lead to read, not a result. The
search comparison reports total tokens and steps, not only schema tokens per
call — per call, search looked 55% cheaper; per episode it cost 22% more.

---

## 22. The distractors' disclaimer made them match the words it disclaimed

**What happened.** Every distractor's description ends "Not used for customer
refunds, returns or cancellations" — written to make them unmistakably
irrelevant. A keyword ranker cannot read "not". Each distractor gained a point
on any query containing *refund*, *return* or *cancel*. For "escalate duplicate
refund claim", five `*_catalog_duplicate_listing` distractors scored 4 (name
match on *duplicate* plus the disclaimer's *refund*), `escalate_to_human`
scored 3, and the six-result limit cut it. Three of the thirteen escalation
failures in the clean search run were this, not the agent.

**How it was found.** The audit classified those three as "searched for it,
ranker missed it" — the only non-discovery failures on escalation — and
replaying the query through the ranker with scores showed where the points
came from.

**Not fixed, deliberately.** The clean tool-scaling run used these
distractors; changing their text would make any later search arm
incomparable with it. The writeup reports the three as retrieval failures and
names the cause. A future run should drop the disclaimer, or add an embedding
ranker as its own arm — negation blindness is a property of keyword search
worth measuring, not only a bug to remove.

---

## What this list is for

Two things.

**In the README**, it is evidence the project was built rather than generated.
Anyone can produce a clean repo; a specific, load-bearing failure list is hard
to fake.

**In an interview**, #1 and #6 are the two to tell. #1 because the bug class —
silent corruption invisible at the default configuration — is the one senior
engineers actually worry about, and because the fix came from a *guarantee*
rather than from code review. #6 because "my benchmark penalised the better
model until I found the ambiguity" demonstrates the instinct that separates
people who build evals from people who run them.
