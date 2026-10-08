# What failed

Thirty-three real defects found while building this. The first fifteen are in
the order of how much damage they would have done; the rest are in the order
they were found. Every one is reproducible from the git history.

The pattern worth extracting: **almost none of the serious bugs were found by
reading code.** They were found by mechanisms built to catch a whole class — a
reference solution that must pass, an adversarial solution that must fail, a
scale-up that exercises a code path differently, a real model run. That is the
argument for spending the first two weeks on guarantees rather than features.

The exceptions are #9, which no mechanism here could have caught and a human
reading transcripts did, #26, a measured "finding" that one printed line per
case undid, and #30, where the checker's own quirk became the thing a
fine-tune learned. Automation catches the classes you thought of.

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
lookup-only agent achieves for free. 52 of the 199 generated tasks are in this
category since v19 (the photo trap joined them, #30), and
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

> **Correction — see #26.** The simulator was not leaking. The leak detector
> could not read how the agent asks for a fact in Indonesian, Malay, Thai or
> Chinese, so the customer's answers were scored as volunteered facts. With the
> agent's own phrasings added, no reveal outside English is unprompted in any
> of the four runs that can be checked. The entry is kept as written: the three
> fixes and the broken gate happened. The diagnosis did not.

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

## 23. The judges graded transcripts with the tool results cut out

**What happened.** Scored against the verifier on the search arm, the
decomposed judge rejected 52 of the 79 episodes the database passed. Traces
store each call's name, arguments, ok/error and payload length — not the
payload. The judge saw "the item value is THB 1,500" with no lookup result
behind it, and did exactly what its rules say: "a claim is hallucinated if no
tool result in the transcript supports it."

Given the results, the same judge accepted 71 of 79 correct episodes instead
of 27. Its apparent strictness on failures went with it: it had accepted only
3 of 17 failures, which looked like discrimination; with the results it
accepted 13 of 17. It had been rejecting nearly everything, and the failures
were simply included.

**How it was found.** The limitation was written into `run_judges.py` from the
first version, as a caveat to "state in the writeup". It was never measured.
What forced it was a number that could not be taken at face value: a judge
failing two thirds of work the database called correct.

**Fix.** The environment is deterministic — fixed clock, counter ids, no
randomness — so `harness/replay.py` regenerates every result by replaying the
recorded calls against the task's fresh database, and accepts a result only if
it matches the recorded payload length. Byte-identical on all 640 calls of the
215 reference solutions; 629 of 629 verified on the real run. (Both claims
held inside one process only: a tracking number drawn from Python's salted
hash differed between processes at the same length, and the length check
passed it — #33.) `--payloads` gives the judge those results. Found alongside it: neither judge saw the
policy either — the trace keeps a placeholder system message, and the renderer
drops system messages — so `policy_accurate`, "correct per policy.md", was
judged without policy.md. `--policy` closes that. The calibration run against
human labels keeps the old view on purpose: the humans labelled from it too.

**Lesson.** A documented limitation is not a measured one. This caveat was the
difference between a judge that rejects two thirds of correct work and one
that rejects a tenth.

*The numbers above are as the runs recorded them. On the v19 checks (#30) two
search-arm episodes move to pass: the judge rejects 54 of 81 correct episodes
without the tool results and 9 of 81 — a ninth — with them; 12 of 81 in the
judges' re-run after #33.*

---

## 24. The calibration sample had nothing in it for the rubric to find

**What happened.** 200 transcripts, labelled on nine criteria: 1,800 human
judgments, of which 2 were "violated". Every per-criterion kappa came out 0 —
not because the judge disagreed, but because there was nothing to agree about.
The report printed a table of zeros annotated KAPPA PARADOX, which reads as a
failed judge when it means an unmeasurable one.

**Why.** The sample was stratified to over-represent failures, and it did: 43
of the 200 transcripts failed the verifier. 41 of those 43 were labelled clean
on every criterion — and that is very likely correct. All 43 were *action*
failures: 28 left the wrong end state, 14 skipped a required action, one took a
forbidden one. None was a failure of language, and the rubric scores language:
leaks, invented facts, wrong statements of policy, tone. An agent that
politely does the wrong thing satisfies all nine. The labelling tool made
"satisfied" a single keystroke, which cannot have helped, but is not needed to
explain the result.

**What it cost, and what replaced it.** The human-agreement numbers measure
nothing and are reported as such. The judges were instead scored against the
verifier on the search arm, where 17 of 96 episodes fail (15 on the v19
checks, #30) — a ground truth with variance, and the experiment that produced
the project's clearest judge result. The 41 of 43 is the same on either
checker.

**Lesson.** Look at label variance after the first twenty labels, not after two
hundred. And sample for the property the rubric measures: a set stratified on
*task* failure is not a set rich in *communication* failure.

---

## 25. A tie-break put a broken run behind the headline language finding

**What happened.** The report picks the largest qualifying run for the
multilingual section. Three runs tied at 1,075 episodes; the tie broke
alphabetically in reverse, which chose `G-gated` — the run whose fact-gating
had no Chinese ask-patterns and stonewalled (#11, #26). That run holds the only
significant language gaps in the project: zh-MY +26 points (p < 0.01) and
zh-SG +29 points (p = 0.02). In `C-clean`, the same tasks without gating, both
are within nine points of English with p ≥ 0.62. (Under the v19 checks, #30:
+21 and +28 points in G, and within three points in C.) The report attributed the
fake gap to tokenisation.

**Why it matters.** Nothing flagged it. The section named its source run in
italics, but a tie read as a choice, and "largest run" sounded principled.

**Fix.** A tie is announced in bold with the runs it tied against, and broken
the same way every time; `--multilingual-run` takes several runs and adds a
replication table with a sign test per language per run. The language claim
now rests on `C-clean` and `D-nozh` together, where it is a null in both.

**Lesson.** A selection rule that is silent in the tie case is a coin flip
with a principled name.

---

## 26. The simulator never leaked. The detector could not read the questions

**What happened.** #11 reported that the simulated customer volunteered facts
before being asked: 11.3% of episodes in run D, every one in Indonesian, Malay
or Thai. Three runs went into fixing it. Checking whether the language result
still needed #11's caveat, I printed the agent's message before each flagged
leak. Outside English, every one asked for the fact. The agent writes
"4 digit terakhir", "nomor order", "เลขออเดอร์", "订单号", "手机号"; the patterns
knew "4 angka", "nomor pesanan", "หมายเลขคำสั่ง", and no Chinese at all. The
customer answered a question the detector could not read, and the answer was
counted as a leak.

The requests it missed in runs C and D: "4 digit" 148, "订单号" 132, "手机号" 107,
"nomor order" 55, "เลขออเดอร์" 35, "nombor order" 21, "id pesanan" 3.

| run | flagged, first patterns | flagged, with the agent's phrasings |
|---|---|---|
| C | 233 of 1,075 (21.7%) | 2 |
| D | 105 of 930 (11.3%) | 0 |
| F | 241 of 1,075 (22.4%) | 1 |
| G | 139 of 1,075 (12.9%) | 0 |

The three left are English: two customers who had failed identity
verification repeating their order number while pushing back, and one
answering "is this the order you meant?".

A fifth run cannot be checked at all, and finding that out was a second
defect. Run B predates the order-id fix of #9: its customers say
`GO-addresID`, today's generator gives the same task `GO-B6FE71`, and in 850 of
its 850 generated episodes the order id the audit looks for never appears. The
audit printed 1.2% for it anyway — a clean-looking rate about nothing.

**What it undoes.**

- **#11's backfire.** Restating the rule in the target language "raised the
  leak rate from 11.3% to 22.4%" — that is run D to run F. F also added
  Chinese, which the detector could not read at all. Without its Chinese
  episodes, F flags 105 of 930: the same count as D. The restatement changed
  nothing, and the elephant mechanism explained a number that was not there.
- **The language caveat.** Indonesian, Malay, Thai and Chinese were never
  measured under a more helpful customer. Their nulls are not lower bounds.
- **Run G.** The gate used the same patterns, so it withheld exactly the facts
  the detector could not see being asked for. The customer left the agent's
  request unanswered in the next turn 74% of the time in Indonesian and 85–86%
  in Chinese, against 17% in English; ungated, the same languages sit at
  10–25%. #11 blamed the missing Chinese patterns, but Indonesian was
  "covered" and fell further, to 0.463.
- **#25.** Those are the gaps a tie-break then put in the report as the
  language finding. One pattern list, four symptoms: a leak rate, a fix that
  seemed to backfire, a fix that broke the comparison, and a false result.

**Why it survived.** The rate looked like a finding: concentrated in some
languages, zero in others, with a plausible mechanism. The audit printed each
case as "volunteered `order_id` at turn 2, before the agent asked" — the
conclusion, not the evidence. It also looked each episode up by task id, so all
five seeds of a task shared seed 0's verdict and every per-language count was a
multiple of five.

**Fix.** The patterns include the phrasings the agent uses in every language,
and a test holds six requests in the phrasings the first list missed. Every
flagged leak prints the agent's message before it, and the audit reports the
mirror image — requests the customer did not answer — by language, which is how
a stalling gate shows up. A language that leaks far more often than English
gets a warning to check the detector before the simulator. Counts are per
episode. The gate runs only in languages whose patterns were checked this way.
Traces now record a digest of the task that produced them, and the audit
leaves out any trace whose task has changed since — or, for older traces
without one, refuses a run in which the customer mostly never says today's
order id.

**Lesson.** A detector's rate is a claim about the detector until someone reads
what it flagged. Here the deciding evidence was one line per case, in traces I
had for three runs.

**It happened again, and was caught.** The first post-training run changed the
agent to Qwen3.8-27B. Its audit flagged 20 leaks, all in Indonesian and Thai,
and printed the warning this entry added: check the detector before the
simulator. Every one followed a request in words the list lacked: "ID order",
a misspelt "nomor pesannya", and Thai with its tone marks and vowels dropped or
misplaced ("หมายเลขโทรศัพท" for "หมายเลขโทรศัพท์"); the fine-tune later asked
"哪个订单" in Chinese. The patterns now include them, and Thai is matched with
its combining marks folded out on both sides. The three post-training runs
flag 0 of 3,225 episodes; the four earlier runs are unchanged at 3 of 4,155.
Folding has a cost of its own: with the marks gone, หลัก (digit) reads like
หลักฐาน (evidence) and หลีกเลี่ยง (avoid), and ท้าย (last) like ทายาท (heir), so
"4 หลักฐาน" read as a request for the phone digits. The patterns now exclude
those words and a test holds the negatives; no audit count moved.

---

## 27. The training pipeline was built for the week-1 suite

**What happened.** Before renting GPUs for post-training, I read the training
scripts against the suite as it is now. They had been written in week 1 and
never re-read. As written they would have:

- **trained on 16 tasks.** `train_rft.py` and `train_grpo.py` imported the
  hand-written `TASKS`; the 199 generated tasks, and their reference solutions,
  were never loaded.
- **evaluated on translations of training data.** The held-out split was by
  task id, so `HRWR-ID` could train while its twin `HRWR-ID.id` — the same
  world, the same checks, the same correct actions — was scored as unseen.
- **trained on prompts the model never saw.** The whole conversation was
  rendered once and masked to the assistant turns. Qwen3's template in
  non-thinking mode puts the empty `<think></think>` block only on the *last*
  assistant turn, but the model generated every turn after one; every earlier
  turn was a target under a prefix that never existed at inference.
  `tests/test_training.py` reproduces it with a Qwen3-style template.
- **trained without the tool list** the model is shown at inference, and
  **written no traces**, so none of the audits in this document could have run
  on the rollouts.
- **used a customer that cannot write half the languages.** The default
  simulator was Llama-3.1-8B, whose supported languages include Thai but not
  Indonesian, Malay, Vietnamese or Chinese. The published customer, Qwen,
  cannot be reused either: the agent under training is Qwen.
- **pointed the reference tier at `Qwen/Qwen3-72B-Instruct`**, which is not
  among Qwen3's released sizes.

**Fix.** The pipeline runs on all 215 tasks through the sweep, so every episode
is a trace and every audit applies. The split is by family, two folds, every
trap in both (`pasarbench/rl/split.py`); each fine-tune is scored only on the
fold it did not train on, and the report verifies that episode by episode.
Each example is one agent turn, rendered with the model's own template, with a
prefix assertion that fails loudly on a template that renders differently in
and out of context; the prompt's token ids come from the serving engine
itself, and `train_rft.py check` re-renders a sample through it token by
token. The customer is Gemma 4 31B, and the sweep refuses an agent and
customer from one family. The reference is deepseek-v4-pro through its API,
against the same customer.

**Moving the serving to SGLang** (from vLLM, before any GPU time was spent)
needed three more of these, all found by reading its source:

- It answers a request naming a bare LoRA adapter with the **base model**,
  without an error, although `/v1/models` lists the adapter under that very
  name. The evaluation would have scored the base model as its own
  fine-tune. Adapters are now requested as `Qwen/Qwen3.8-27B:pasar-rft-A`,
  and the sweep checks every fold model against the server before the first
  episode.
- It hands the chat template each tool as its own pydantic dump —
  `"strict": false`, `"defer_loading": null`, description before name — so
  every training prompt rendered locally would have differed from the one the
  model saw, in the tool block. Prompts now come from the server's
  `/tokenize`; that endpoint cannot render a finished turn, so the turn is
  tokenized locally.
- It keeps Gemma 4's special tokens in the decoded text so its parsers can
  read them. The customer is served with `--reasoning-parser gemma4`, and the
  audit counts chat-format tokens on both sides of every conversation.

**A last read before the first GPU hour**, by a reviewer who had not written
the code, found four ways the pipeline would have spent money on nothing:
resume kept episodes a dead API key or a crashed server had ended, so "run it
again" re-ran none of them; a new terminal silently reset the repetitions
between the baseline and the evaluation it is compared with; a LoRA probe with
zero weights would have passed whether or not the server applied the adapter;
and on a GPU machine the setup's own tests failed, because transformers routes
Qwen3.8's linear attention to GPU kernels even for the CPU toy model. All four
are fixed, and the first three have tests.

**Lesson.** Unlike most of this list, this was found by reading, and it was
found because GPU time costs money and API time had not. Code written against
an early version of the data encodes assumptions that were true when written;
the suite grew by thirteen times and the training scripts did not notice.

---

## 28. Two servers on one GPU: the second one's fraction is of what is left

**Symptom (caught before any GPU hour).** The only cards on offer were single
ones: one H100, H200, B200 or B300. Everything was sized for two H100s, one
server per card. One H100 cannot hold the agent's 52 GiB and the customer's
31 GiB of weights; one B200 can, with ~80 GiB left for both KV caches. The
obvious move — the same `--mem-fraction-static` split in half for each server
— does not work.

**Cause.** SGLang sizes its cache as `pool = free_before_load * f - weights`,
where `free_before_load` is the memory free *when that server starts*, not the
card's size, and keeps `free_before_load * (1 - f)` for CUDA graphs and
activations. Start both at once and each measures the other's half-loaded
weights as used, or not, depending on timing; start them in order and the same
`f` means something different for the second. And SGLang's defaults on a
180 GB card (16k prefill chunks, decode graphs up to batch 512) reserve working
memory sized for one server alone.

**Fix.** On one card the agent starts first, with `f` chosen to give it a
planned pool; the customer starts once the agent answers, with `f` worked out
from `nvidia-smi` at that moment so that a reserve stays free for both
servers' runtime (`scripts/gpu_plan.py`, tested by simulating the agent's use;
`PASAR_RESERVE_GB` raises it, and is the one knob that helps after an
out-of-memory at run time — a smaller agent cache would only have grown the
customer's).
Batch size and prefill chunks are capped at what 24 conversations need. The
customer's sliding-window pool, by default 0.8x the tokens of its full-attention
pool at ~768 KiB a token, is cut to 0.3x: a conversation needs one 1024-token
window there but its whole length in the full-attention layers. Training runs
in one process with 32 gradient-accumulation steps, the 32 turns per step two
cards gave, so the recipe does not depend on the hardware; the smoke stage now
trains a few steps on one maximum-length turn, so an OOM or a kernel that
does not run on Blackwell turns up in minutes. B300 was ruled out on paper:
Modal requires CUDA 13.1 for it, and SGLang 0.5.20's torch is built on 13.0.

A reviewer reading SGLang's pool code found a limit the two-card layout had
all along: the agent keeps a ~150 MiB linear-attention state per running
conversation, five per request with the radix cache's copies, and SGLang's
default gives those states under half the pool. That capped the agent near a
dozen conversations on an H100 or an H200 while the sweeps send 24 — nothing
wrong, only slow. The agent now gives them 70% of its pool
(`--mamba-full-memory-ratio 2.5`), and every stage logs what each server can
hold (`logs/gpu.txt`) with a note when the agent's limit is under 24.

**Lesson.** A memory knob is only portable if you know what it is a fraction
*of*. Read the formula before splitting a card — and read what else draws on
the same pool.

---

## 29. The first GPU minute: SGLang compiles, and nothing had a compiler

**Symptom.** The first smoke test on a B200 in a Modal Notebook stopped before
the agent loaded a single weight: `RuntimeError: … NVCC version must be at
least 12.9`, from DeepGEMM, called by SGLang's model runner at start-up.

**Cause.** `pip install sglang` is not the whole runtime. SGLang 0.5.20 compiles
kernels when a server starts — its own JIT kernels, FlashInfer's, DeepGEMM's —
with the nvcc it finds through `CUDA_HOME`, then `PATH`, then `/usr/local/cuda`.
torch 2.13 brings the CUDA 13 *runtime* through pip but no compiler. The
notebook image had a CUDA toolkit older than 12.9 on its PATH, so that is the
one DeepGEMM found; the Modal job image had none at all, and no C++ compiler
for nvcc's host code either. Every test in this repo runs without a GPU and
without nvcc, so none of them could see it.

**Fix.** Every install path (the notebook helper, the Modal image, the Lambda
setup) now adds pip's `cuda-toolkit[nvcc,cccl]` at the version torch pins, and
`scripts/cuda_home.py` points `CUDA_HOME` at it — preferring it over whatever
the machine has — after two symlinks that make its layout what nvcc's own
profile and the JIT link lines expect (`lib64`, `libcudart.so`; without them
the link fails, which was checked by building a kernel with it). Every GPU
stage now builds a test kernel for the card before any model loads, and so do
`pn.setup()` and the Modal `test` stage, where no GPU is billed yet. DeepGEMM,
which serves block-FP8 and MoE GEMMs that neither model has, is off unless
asked for.

**Lesson.** The dependency list of an inference engine includes the compiler
it runs at start-up. Pin it like torch, and check it on the target before the
expensive part.

---

## 30. The checker preferred a tool, and the fine-tune learned the preference

**Symptom.** The first post-training result. Qwen3.8-27B fine-tuned on its own
passing episodes (RFT) scored 0.882 against the base model's 0.859 — 22 tasks
better, 13 worse, p = 0.18: a gain, not a significant one. deepseek-v4-pro, the
reference, scored 0.899. Per trap, most of the movement sat in two places: the
peak-period delay (7 of 70 base episodes passed, 26 of 70 after RFT) and the
high-value photo rule (3 → 7 of 70).

**Cause.** Reading the failures, not the rates. All 63 of the base model's
peak-period failures failed on one line — `missing required action: get_order`
— and so did all 44 of the fine-tune's. 59 of the base model's 67 photo
failures failed only on a missing `check_return_eligibility`. In every one the
outcome was right: no voucher for a delay inside the extended sale SLA, no
return opened before photos, nothing forbidden. And the agent had established
the fact the rule turns on another way: `list_user_orders` shows when an order
was placed, `get_order` what an item is worth. The two checks named one tool
where the policy needs only the fact. The check for a shipped order that cannot
be cancelled already said why that is wrong — "requiring one specific call
tests tool preference, not policy compliance" — and accepted any order lookup;
these two had never been read against it.

RFT trains on the episodes the checker passes, so on these traps it was
trained on the preference — and learned it there: on the peak trap the
fine-tune called `get_order` in 26 of 70 held-out conversations, the base
model in 7. Not as a general habit: across the suite it called `get_order`
slightly less than the base model did (672 of 1,075 conversations against
717). deepseek-v4-pro calls it in 95% of its conversations (the base model in
67%), so the same checks favoured it.

**What it moved.** Nothing was re-run: every episode's recorded tool calls
were replayed and scored by the corrected checks (`scripts/rescore.py`).
8,941 of the 9,766 recorded episodes reproduce call for call — checked by
each result's length, since every run predates the digests v19 adds, and never
on a tracking number (#33) — and are re-scored. The other 825 keep their
recorded verdicts: 824 from run B, whose calls name orders from before the id
change of #9, and one in run F.

| | as recorded | today's checks |
|---|---|---|
| base, pass^1 (pass^k) | 0.859 (0.805) | 0.979 (0.930) |
| RFT | 0.882 (0.837) | 0.981 (0.949) |
| reference | 0.899 (0.800) | 0.913 (0.833) |
| RFT vs base, paired | +0.023; 22 better, 13 worse, p = 0.175 | +0.003; 13 better, 11 worse, p = 0.839 |
| reference vs base, paired | +0.040; 39 better, 32 worse, p = 0.477 | −0.066; 12 better, 32 worse, **p = 0.004** |
| search-300 vs all-20 (tool scaling) | −0.135; 9 worse, 1 better, p = 0.021 | −0.115; 7 worse, 1 better, p = 0.070 |

Twenty-two of the fine-tune's 25-episode gain were the checker (923 → 948
passing, recorded; 1,052 → 1,055 now); its pass^k lead shrinks from 3.2 points
to 1.9. On pass^1 the reference and the base model swap places. On pass^k —
every seed of a task passing, the number this benchmark calls the one that
matters — the reference was already level with the base model (0.800 against
0.805), and the correction opens a 9.7-point gap. And it reached back to the
first experiments: the search arm keeps `get_order` in view but hides
`check_return_eligibility` behind search, so two of its failures were this
quirk, and the one significant accuracy result of the tool-scaling study is no
longer significant. The judges' agreement with the verifier moves too (best
κ 0.65 → 0.72 on the judges' re-run, 0.56 → 0.61 on their first), as do the
noise floor and the context and language tables a little; RESULTS.md shows
each paired test under both scorings and names any conclusion that changed —
only these two did. No verdict moved from pass to fail, and none moved in a trap whose checks did not change — which is what a
correct replay of a relaxed check has to show, and `rescore.py` prints it per
trap.

**Fix.** Both checks accept any read of the task's own order that shows the
fact: its date on the peak trap (`get_order`, or `list_user_orders` for this
customer, unfiltered or filtered to the order's status), the item's value on
the photo trap (`check_return_eligibility`, `get_order`, `get_order_items`,
`get_product` of this item's product, `calculate_refund_amount` for this
item). P3.2 requires the eligibility check before `initiate_return`, and a
return is what this task forbids. The photo trap is now a do-no-harm trap like
the peak trap, and the spam tests exempt it for the same reason. It also
required `verify_identity` — kept at first, since the customer is asking for a
return — until that was the only thing its failures had in common: 20 of the
21 left across the three runs, none of which called it. P1.1 puts verification
before a *write*, P1.3 lets the agent answer without it, and the right answer
here writes nothing: the same mistake as naming one lookup. Dropped, it
promotes 48 more episodes across all runs, every one of them asking for
photos, and leaves one failure on the trap in the three runs: the base model
opening the return and refunding on photos the customer never sent.

The first version of the fix was looser, and a review of it found three ways
to pass without the fact. Alternatives matched on the tool's name, so reading
another customer's orders counted as reading this one's; a `verify_identity`
whose digits did not match counted as a verification; and neither trap forbade
compensating another way — store credit on the peak trap, a voucher or store
credit on the photo trap. Each alternative now names its target, a call must
work *and* not be denied (the identity trap, where verifying must fail, asks
for the attempt), both traps forbid and assert against the compensation, and
the order total `list_user_orders` shows, shipping included, is no longer
taken for the item's value. Re-scored, none of it moves a verdict: every
promoted episode read its own order, none compensated, and none relied on a
failed verification. The promoted episodes were read as well: all 117 on the
peak trap tell the customer about the extended sale SLA, and 230 of the 234 on
the photo trap ask for photos. The four that do not are one tool-scaling arm's
(`I-tools`, random-100), where the agent stalled asking for an item id — and a
do-no-harm check passes a stall.

Every recorded verdict can now be re-scored by replaying the episode
(`pasarbench/rescore.py`): against the world it ran in, refusing any episode
whose replay does not reproduce its recording. RESULTS.md is built on today's
checks and shows the recorded verdicts beside them — every section's paired
tests, and sections 6 and 7 in full — so a checker change can never quietly
move a headline number again. And `train_rft.py build` now picks its passing
episodes by today's checks (`--checker`), so the next fine-tune cannot learn a
check that has since been corrected.

**Lesson.** A checker that prefers a tool is a reward that prefers a tool. RFT
is meant to learn the policy from its successes; it learned what the checker
counted as success. Before training on a verifier, read what its passes have in
common that the policy does not require.

---

## 31. Fine-tuning carried answers from one trap into its neighbours

**Symptom.** Outside the two traps of #30 the fine-tune's net change was +2
episodes — gains on duplicate-refund escalation (64 → 70 of 70) and
out-of-window vouchers (61 → 66), losses on out-of-window disputes (69 → 64),
customs holds (54 → 49) and cash-on-delivery refunds (54 → 52). Per-trap moves
are leads at ~13 tasks a trap, so the losses were read, episode by episode.

**Cause.** Each loss is a neighbouring trap's right answer in the wrong place.

- **Out-of-window disputes.** In four episodes the fine-tune refunded a
  wrong-colour item three weeks out of the window, in full with shipping, and
  three times messaged the seller, calling it "seller misrepresentation" or a
  seller-side fault; one cited P7.2 by name. That is the livestream trap's
  answer. P7.2 covers livestream claims only; the rule here is P6.3: escalate.
  The livestream trap passes nearly every time, so it filled its whole quota
  of the training data. (The base model did this once. A fifth episode
  escalated under the wrong category; the sixth is #32.)
- **Customs holds.** The fine-tune messaged the seller in 7 of 55 episodes
  (base 3) and escalated in 49 (base 54), telling the customer to wait on
  tracking — the peak-delay trap's "wait, it is not compensable".
- **A COD refund in Thai** (2 of 5 passes, base 5 of 5): store credit "for a
  lost shipment" with no return opened, after a customer who had said only
  their order id, their digits and that store credit was fine.

**Why it matters.** The training mix was whatever passed at temperature 1.0,
capped at three episodes a task: every task the model already solves fills its
quota, and the traps it rarely solved — the two of #30 — barely appear. The
solved traps' answers leak into the ones that look like them — livestream into
out-of-window, peak delay into customs — and the net number hides it.

**Fix.** Not run: a second fine-tune is a GPU run, and this one ends the
budget. What it would change is written down instead: build the data with
today's checks (#30; `train_rft.py build` now does by default), balance it by
trap rather than by task, and read the per-trap table, not the aggregate, as
the result.

**Lesson.** Fine-tuning on your own successes teaches the successes you have
most of. A net-zero change can be two real moves in opposite directions.

---

## 32. Out-of-window orders were delivered before they were placed

**Symptom.** One RFT episode on the Thai out-of-window dispute told the
customer the system's dates did not make sense — shipped 2 November, delivered
20 October — then checked eligibility and searched the policy nine times each,
repeating itself, until it ran out of tokens without escalating.

**Cause.** It was right. The generator placed every order on 1 November and
shipped it the next day, and made a case out of the window by moving only the
delivery date back to 20 October. 64 of the 215 tasks had dates that
contradict each other: 39 out-of-window orders delivered three weeks before
they were placed; 13 peak-sale orders paid nine days before they were placed;
10 cash-on-delivery orders paid before delivery; a hand-written order
delivered before it shipped (T01); and an earlier refund issued before the
delivery it refunded (T14). The reference solutions passed regardless: they
are oracles, and no check reads a date.

**Fix.** Dates run in causal order — the live a livestream order came from,
the order, payment (at the door for COD), dispatch, delivery, and all of it
before NOW — and a test checks every world, hand-written and generated. That
moved dates in 87 worlds: the 64, and 23 more whose only change is the
livestream now falling 20 minutes before its order instead of on 20 October.
Only dates changed, so a trace recorded before the fix is replayed against the
world it ran in (`pre_v19_patch`, checked against digests of the v18 worlds),
and traces now record a `world_digest` — of the seed database with the task's
patch applied — so a replay can tell.

**Lesson.** A world has to be consistent before an agent can be graded in it.
The agent that noticed was the careful one: the contradiction penalised
reading.

---

## 33. The replay verified a tracking number the agent never saw

**What happened.** Found while fixing #32: a generated shipment's tracking
number was `abs(hash(order_id))`, and Python salts `str` hashes per process.
Every process built the same shipment with a different number — always two
letters and eight digits.

**Why it mattered.** The judges were shown tool results by a replay in a new
process (#23), verified against the length each result had at run time. A
different tracking number has the same length, so it verified. In `CHE-MY__r2`
the agent read `MY39959518` from `get_shipment` and told the customer so; the
judge, shown `MY38135001`, wrote that the agent "made a factual error by
reporting the wrong tracking number", and the rubric judge flagged the same
line as a hallucination. In that first run, that transcript was the one false
claim either judge rejected — and it was rejected for my replay's error. Two of
the 96 judged episodes had looked a generated shipment up.

**Fix.** Tracking numbers come from sha1 like every other generated id, and a
test builds the suite under three hash seeds and compares. Traces record a
digest of every tool result; the replay verifies content by it, and falls back
to length only for older traces — where a generated shipment's result is
withheld and counted as unverifiable, never shown. RESULTS.md counts the judged
episodes it touched, and says whether a view showed the wrong number or
withheld it.

**The re-run.** The judges were re-run with those two results withheld (627 of
629 calls shown). Given the tool results, both still marked `CHE-MY__r2` down
over the shipment — a tracking number "not present in any tool output", a last
scan taken for invention — which is true of what they were shown: a withheld
result is missing evidence, and a strict judge reads missing evidence as
invention. And the re-run moved more than those two episodes. The best κ went
from 0.61 to 0.72. On `OOWOV-TH__r0`, whose evidence did not change, the 1–5
judge given the tool results and the policy now rejected the voucher the agent
never issued — it "never actually called a tool to issue the voucher" — where
the first run had credited it with "a goodwill voucher per P6.2 with accurate
currency calculation". One run of an LLM judge is one sample of it.

**It happened again, across interpreters.** Re-scoring on Python 3.13 kept one
more episode at its recorded verdict than on 3.10: P-ref's `OOWOV-TH.th__r0`,
where the agent passed `order_id` to `issue_goodwill_voucher`. The tool's
error was Python's own TypeError text, and 3.13 appends "Did you mean
'user_id'?" — so the payload was 25 characters longer than the one recorded,
and the replay refused it, correctly. Worse than the replay: an agent run on
3.13 is told more than one run on 3.12. `tools.call` now reports an argument
the tool does not take in its own words, the ones every recorded run shows;
a test pins them, and the suites run on 3.10 to 3.13.

**Lesson.** A length check is a checksum that cannot see a substitution of the
same length. "Deterministic" has to be tested across processes; within one, a
salted hash looks perfectly stable. And across interpreters: an environment
that passes exception text to the agent inherits every change to it.

---

## 34. The guardrail was graded by its own detector

**What happened.** The claim guardrail (RUNBOOK 1i) holds back a reply that
claims an action no tool call has done, and tells the agent instead. Version
v24 read first-person claims only — "I've escalated", "I'll issue the voucher
now" — and the rule written before its run graded it the same way: the
replies the customer got that make a first-person claim no call backs. On the
run (`L-off` against `L-claims`, 2026-10-07) that went from 2 of 96 to 0 of 96,
a pass by the rule. It held back two replies, both real claims, and both times
the agent made the call and the episode passed. But the audit, which reads more
phrasings, found three false claims reaching customers in the guarded run —
against three in the control. All three read "Your voucher has been issued".

**Why.** Passives were left out on purpose: in the duplicate-refund trap "your
refund was issued on 9 November" is true, and read as the agent's claim it
would send the agent towards a second refund. The same exclusion let "your
voucher has been issued" through, though no task starts with a voucher. And
the measure of success read claims exactly as the guardrail did, so it could not
see what the guardrail could not see: by construction, it could only pass.

**Fix.** v25 reads "we" and passives for the two actions no record of can
predate a conversation in this world — vouchers and escalations — and still
reads refunds in the first person only. A passive needs a subject that points
at something ("your voucher", "this"): a first draft without one fired three
times on "No voucher has been issued", an agent saying what it had NOT done,
which the dry run over recorded traces (`scripts/guardrail_dry_run.py`) caught
before any run did. `compare_cells.py` now prints the audit's reading beside
the guardrail's, and the runbook grades a guarded run by the audit's. Read by
v25, each arm of the recorded run has three false claims: the two first-person
claims the guardrail caught, it fixed; the three passives it never looked at
got through. Graded the new way, v25's own run (`L2-off` against `L2-claims`,
2026-10-08) took the audit's count from 5 of 15 readable failures to 0 of 13.
Reading every failure by hand then found what both readers miss — "your case
has been flagged", a promise never kept — about as often in each arm
(RUNBOOK 1i).

**Lesson.** A fix graded by the instrument that defines it passes by
construction. Grade it with one it does not share, and write down which before
the run. It is #26 again — a detector's blind spot read as the world's — in the
other direction: there the detector invented a problem, here it hid one.

---

## What this list is for

Two things.

**In the README**, it is evidence the project was built rather than generated.
Anyone can produce a clean repo; a specific, load-bearing failure list is hard
to fake.

**In an interview**, #1, #6, #26 and #30 are the ones to tell. #1 because the
bug class — silent corruption invisible at the default configuration — is the
one senior engineers actually worry about, and because the fix came from a
*guarantee* rather than from code review. #6 because "my benchmark penalised
the better model until I found the ambiguity" demonstrates the instinct that
separates people who build evals from people who run them. #26 because a
measured bias, three runs of fixes and a false headline all came from one
unchecked pattern list, and one printed line per case settled it. #30 because
it is #6 at training time: the fine-tune learned the checker's quirk, the
quirk had ranked the models the wrong way round, and replaying the recorded
episodes measured all of it without a GPU.
