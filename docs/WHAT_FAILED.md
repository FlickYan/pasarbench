# What failed

Nine real defects found while building this, in the order of how much damage
they would have done. Every one is reproducible from the git history.

The pattern worth extracting: **none of the serious bugs were found by reading
code.** They were found by mechanisms built to catch a whole class — a
reference solution that must pass, an adversarial solution that must fail, a
scale-up that exercises a code path differently. That is the argument for
spending the first two weeks on guarantees rather than features.

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
lookup-only agent achieves for free. 33 of 186 tasks are in this category and
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

## 9. The kappa-paradox test did not contain a paradox

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

## 10. A test asserted on state the code had already mutated

**What happened.** The interrupt/resume test snapshotted an episode, restored
it, resumed — then asserted `state2.step == 2`. But `run_episode` mutates
`state2` in place, so by assertion time it had advanced to 5.

**Fix.** Capture the value before resuming. Test bug, not code bug.

---

## 11. Two locale variants were byte-identical

**What happened.** The Singlish and English openings for
`identity_verification_failure` were the same string. The "twins differ in the
opening" invariant caught it.

**Why it matters more than it looks.** The whole multilingual claim rests on
locale twins differing *only* in surface form. A twin that does not differ at
all is a silent duplicate inflating the English sample.

---

## 12. The distractor pool ran out before 300

**What happened.** `all-300` produced a 258-tool registry. Domain × field × verb
combinatorics topped out at 235 distractors.

**Fix.** More domains. Trivial — but the arm was mislabelled until it was
caught, and a mislabelled arm is a wrong result, not a cosmetic one.

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
