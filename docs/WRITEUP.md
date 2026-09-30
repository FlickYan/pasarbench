# Writeup kit

Blog post, resume lines, interview prep — drafted from the measured results.

**Every number here comes from a file generated from disk**, and each carries a
hidden comment naming it. The generators:

```bash
python scripts/make_report.py --out RESULTS.md --tools-run I-tools2 \
  --multilingual-run C-clean,D-nozh,G-gated \
  --noise-pair H-context/full,I-tools2/full+all-20
python scripts/rescore.py                                                          # what the checker fix moved
python scripts/audit_tool_arms.py traces/I-tools2
python scripts/inspect_trace.py traces/D-nozh/full --leaks-only                    # leaks now
python scripts/inspect_trace.py traces/D-nozh/full --leaks-only --ask-patterns v1  # leaks as first reported
```

**One checker for every number.** v19 corrected two checks that demanded one
specific lookup tool (WHAT_FAILED #30), and every recorded episode was
re-scored by replaying its tool calls — nothing was re-run. The numbers below
are on today's checks; where the correction moved a number this post quotes,
the post says so, with the recorded value beside it. `make_report.py --checker
recorded` rebuilds the report as the runs first scored it, and RESULTS.md
shows every section's paired tests both ways.

Full judge reasons (RESULTS.md truncates them) are in
`data/judge_vs_verifier/I-tools2__full+search-300__payloads__policy.jsonl`.

If you change a number, change it by re-running, never by hand. A published
post containing a number you never ran is a worse outcome than no post.

**Models, as served.** Agent `deepseek-v4-pro`; simulated customer and judges
`qwen3.8-flash` — a different family from the agent, to avoid self-preference.
No alias re-routing was recorded in the traces. Runs of 2026-09-18 to 09-19
(language) and 2026-09-25 (context, tools, judges); the two judge views with
tool results were re-run on 2026-09-30, withholding the two results no replay
can rebuild (WHAT_FAILED #33). Post-training, 2026-09-29:
agent `Qwen/Qwen3.8-27B` (base, and a LoRA fine-tune per fold) served with
SGLang on one B200, customer `gemma-4-31B-it` (FP8) on the same card, and
`deepseek-v4-pro` through its API against the same customer as the reference.

This draft is in your voice. Rewrite anything that doesn't sound like you — the
numbers are the part not to touch.

---

# Part 1 — The blog post

**Target:** ~2,000 words. **Title options**, finding first:

- *The agent said it escalated. The database said it didn't.*
- *My fine-tune learned my verifier*
- *Your LLM judge believes your agent*
- *Read the database, not the transcript*

---

## The agent said it escalated. The database said it didn't.

> "I've escalated your case with the `customs_hold` category, as required for a
> shipment held at customs for more than 5 days."

A customer-service agent sent that to a customer in Malaysia whose parcel was
stuck at customs. The category is right, the policy is right, the threshold is
right. It is also false. The agent had searched the *policy* for the
escalation category, and never called the escalation *tool*. The database held
no escalation.
<!-- audit_tool_arms.py §3 CHE-MY__r2; trace: search_policy("escalate customs hold category"), no escalate_to_human; verifier: escalations found 0, expected >= 1 -->

I gave an LLM judge the conversation, the tool results the agent had seen,
and the policy it was working under. The judge credited the agent with having
"escalated appropriately". It did mark the transcript down — for trying to
cancel an order that had already shipped, which the agent had. The escalation
it described never happened.
<!-- RESULTS.md §4, CHE-MY__r2, "+ tool results + policy": naive 3 → rejected; full reason in …__payloads__policy.jsonl; the refused cancel_order is in the trace -->

On another conversation it went further. The agent had told the customer "I'm
escalating this for further investigation". The judge wrote that the agent
"failed to actually call an escalation tool after the search returned
irrelevant results" — and passed it, 4 out of 5.
<!-- RESULTS.md §4, DRE-PH__r2, "+ tool results + policy": naive 4 → accepted; full reason in …__payloads__policy.jsonl -->

This post is about the distance between what an agent says and what it does,
and what it takes to measure it. Four results: what happens when an agent has
to *search* for its tools; whether an LLM judge can tell a solved case from an
unsolved one; a language effect that turned out to be my own regex; and a
fine-tune that learned my verifier instead of my policy.

### The benchmark

PasarBench is a customer-service environment for Southeast Asian e-commerce:
215 tasks across 16 traps, 6 markets and 8 language varieties — including
Singapore English, Malay, Indonesian, Thai, Vietnamese, and Chinese as written
in Singapore and Malaysia — with 20 tools over a mock order database.
<!-- RESULTS.md "The suite" -->

A task passes if the **database** ends in the right state: the refund issued,
the escalation filed, the forbidden action never taken. Not if the reply reads
well. The tools enforce data integrity and nothing else — `issue_refund`
refuses a cash-on-delivery order because there is no card to refund, but
`initiate_return` will open a return 19 days after delivery, because catching
*that* is the agent's job. If the tools enforced policy, every agent would
score 100%.

Each task has **locale twins**: the same world and byte-identical pass
criteria, in another language. Only the words change, so a gap between twins
is language and nothing else.

### Result 1: the agent searches for tools it already knows about

Production tool registries can run to hundreds of tools, so a common pattern is
to show the agent a small core plus a `search_tools` tool and let it retrieve
the rest. I compared five exposures on the same 32 tasks, three seeds each: an
oracle set (every lookup, plus only the actions the task needs); all 20 tools;
100 (20 plus 80 plausible distractors); a random 100 that always contains what's
needed; and search over 300.
<!-- RESULTS.md §2 -->

With 100 tools the agent was not measurably less accurate than with the oracle
set — 4 tasks worse, none better, p = 0.125 — but each episode cost 2.7 times
the tokens of the 20-tool registry.
<!-- RESULTS.md §2 Reading: INCONCLUSIVE at 100 tools; 101,708 vs 38,041 = 2.7x -->

Search cost more and bought nothing: 22% more tokens per episode, and 11.5
points below showing all 20 tools — 7 tasks worse, 1 better, p = 0.07, not a
significant loss. (It was significant, at p = 0.021, until Result 4: two of the
failures were my checker's, not the agent's.) Each call carried 55% fewer
tokens of tool definitions, but searching added almost three steps per
conversation, and every step re-sends the conversation.
<!-- RESULTS.md §2: -0.115, 7 worse 1 better, p=0.070 (recorded: -0.135, p=0.021); 46,423 vs 38,041 (+22%); steps 10.4 vs 7.7; schema 793 vs 1,762 -->

Where search did fail, the reason was specific. The ranker is deterministic, so
I replayed every query the agent issued and reconstructed exactly what it was
shown. There were 21 cases of a failed
episode needing a tool the arm had hidden. In 17 of them, the agent **never
searched for it**. The three tools it didn't look for — escalation, shipment
lookup, goodwill voucher — have one thing in common: the policy describes those
actions but never names the tools. The tools the policy *does* name, the agent
called without searching; the harness refused them as unknown, and 23 times out
of 26 it then found them by search and used them.
<!-- audit_tool_arms.py §2 (never searched 10+5+2 of 21; "NOT named in policy") and §1 (26 refused, 23 found & used, 26/26 policy-named) -->

It searched for tools it knew existed. It did not search for tools it would
have had to imagine. In five of the six failed customs-hold episodes it never
looked up the shipment at all, and told the customer the delay was normal for
the 11.11 sale.
<!-- audit_tool_arms.py §2: get_shipment never searched 5 of 6; "last said" lines for CHE-MY r0/r1, CHE-VN r0/r1/r2 -->

And this is where the false claims came from. Three of the fourteen search-arm
failures I could read said an action had happened that never did. In the four
arms where the needed tools were always visible: none, in twenty failures. The
numbers are small, but the shape is right — an agent that knows what it should
do and can't find the tool to do it says it did it anyway.
<!-- audit_tool_arms.py §3: search-300 read 14, claimed-not-done 3; all-100 0/7, all-20 0/4, oracle 0/2, random-100 0/7 -->

### Result 2: an LLM judge takes the agent's word for it

The usual way to evaluate conversations at scale is an LLM judge. I had
calibrated two against 200 hand-labelled transcripts, and that calibration
measured nothing: of 1,800 human judgments, 2 were violations. Forty-one of
the 43 transcripts that failed the verifier were labelled clean on every
criterion — probably correctly. Every one was an action failure, and the rubric
scores language. An agent that politely does the wrong thing satisfies it.
<!-- RESULTS.md §4 calibration notice; WHAT_FAILED #24 -->

So I scored the judges against the database instead, on the 96 search-arm
episodes: 81 passed, 15 failed, 3 of those claiming an action that never
happened. (79 and 17 before the checker fix of Result 4, which moved two
episodes to pass; the table below is on today's checks.) Two judges — a single
1–5 score, and a nine-criterion rubric — each under three views: the
transcript alone, plus the tool results, plus the policy.
<!-- RESULTS.md §4 "Judges against the verifier"; recorded: RESULTS --checker recorded, 79/17 -->

| judge | what it saw | correct accepted | failures accepted | kappa vs database |
|---|---|---|---|---|
| 1–5 score | transcript | 73/79 | 11/15 | 0.22 |
| 1–5 score | + tool results | 74/81 | 11/15 | 0.20 |
| 1–5 score | + tool results + policy | 80/80 | 6/15 | **0.72** [0.48, 0.90] |
| 9 criteria | transcript | 27/81 | 3/15 | 0.06 |
| 9 criteria | + tool results | 69/81 | 9/15 | 0.23 |
| 9 criteria | + tool results + policy | 72/81 | 10/15 | 0.23 |
<!-- RESULTS.md §4 table, ground truth on today's checks; the 1–5 judge gave no valid score on 2 transcript-only episodes and 1 with everything -->

Four things in that table.

**What the judge can see decides everything.** Without the tool results, the
rubric judge rejected two thirds of the correct episodes, 53 of the 54 for
"hallucinated facts" — facts the agent had read from tool results the judge
couldn't see. Given the results, it rejected 12.
<!-- data/judge_vs_verifier/I-tools2__full+search-300.jsonl against today's verdicts: 54 correct rejected, 53 no_hallucinated_facts; …__payloads.jsonl: 69/81 -->

**Only one configuration beats chance, and it still misses two in five.** The
1–5 judge with the tool results and the policy reaches κ = 0.72 — and accepts 6
of the 15 episodes the database fails. (On the recorded verdicts: κ = 0.65, 8
of 17.)

**Given everything, the judges caught one false claim in three.** The 1–5
judge rejected the fake voucher: the agent "never actually called a tool to
issue the voucher—telling the customer it was 'arranged' without performing
the write action". It noticed the fake escalation on the duplicate refund and
passed it anyway. On the customs hold it wrote that the agent had "escalated
appropriately" — though shown the same tool results without the policy, it
had caught that one: the agent "claimed to have escalated without evidence of
actually performing that action". The rubric judge, given everything, passed
the voucher and the duplicate refund without a flag.
<!-- RESULTS.md §4, the three false-claim transcripts under every view; full reasons in …__payloads__policy.jsonl -->

**And one run of a judge is one sample.** The two views with tool results are
the judges' second run. The first showed them a tracking number my replay had
got wrong (below); the re-run withholds it, which changed the evidence for two
episodes and nothing else. The best κ still moved from 0.61 to 0.72 — and on the voucher,
the same judge given the same transcript, tool results and policy had credited
the agent the first time: it "offered a goodwill voucher per P6.2 with accurate
currency calculation".
<!-- first run: WHAT_FAILED #33; re-run: RESULTS.md §4 -->

The lesson is not that LLM judges are useless. It is that whether an action
happened is a question about state, and state can be checked. Check outcomes
against the database, check claimed actions against the tool log —
mechanically — and keep the LLM judge for what can't be checked: tone,
clarity, language.

### Result 3: the language effect was my regex

The multilingual comparison is the one the benchmark was built for. In two
independent runs — the full suite, and the suite without Chinese — no language
differed from English beyond noise: the largest paired gap was 7.5 points,
Thai doing *better* than English, and every sign test had p ≥ 0.12. (Before
the checker fix of Result 4: 9.2 points, Singapore Chinese doing better, every
p ≥ 0.38. The null holds either way.)
<!-- RESULTS.md §3 replication table, C-clean (215 tasks) and D-nozh (186); recorded: §3 "Paired gaps that differ" -->

For a while, that result carried a caveat. My leak audit said the simulated
customer volunteered facts before being asked — 11% of episodes in one run, 22%
in another, almost all in Indonesian, Malay, Thai or Chinese. I spent three
runs trying to fix the simulator.
<!-- inspect_trace.py --leaks-only --ask-patterns v1: D-nozh 105/930 (11.3%), F-clean-sim 241/1,075 (22.4%) -->

Writing this post, I printed the agent's message before each flagged leak.
Every one outside English answered a question: "4 digit terakhir nomor
telepon", "nomor order", "เลขออเดอร์", "订单号". My detector knew the textbook
phrasings and nothing in Chinese, so it scored the customer's *answer* as a
leak. With the agent's own phrasings added, 3 of 4,155 episodes are flagged
across four runs, all of them in English.
<!-- WHAT_FAILED #26; inspect_trace.py --leaks-only on C, D, F, G: 2 + 0 + 1 + 0 of 1,075 + 930 + 1,075 + 1,075 -->

It went further than a wrong number. My third fix withheld each fact from the
simulator until the agent asked for it — judged by the same patterns. In
Indonesian and Chinese, the customer now stonewalled: it left 74% and 85% of
the agent's requests unanswered, against 17% in English. That run holds the
only significant language gaps in the project: Chinese 23 and 28 points below
English (26 and 29 on the recorded verdicts). When my report picked a run for
its language section, a tie-break chose that one, and pointed at
tokenisation.
<!-- inspect_trace.py traces/G-gated/full --leaks-only: requests not answered id 207/280, zh-MY 478/561, en 183/1,093; RESULTS.md §3 replication table, G-gated column; WHAT_FAILED #25, #26 -->

One pattern list produced a leak rate, a fix that seemed to backfire, a fix
that broke the comparison, and a false finding. What settled it was one printed
line per case.

How much does a result move on its own? The same configuration, run twice,
scored 89/96 and 92/96 — 3 points, p = 0.38 (88/96 and 4 points, p = 0.22, on
the recorded verdicts). The best context-management strategy I tested beat the
baseline by 5 points, at p = 0.13 (6 points, p = 0.06, recorded). I can't tell
it from re-running.
<!-- RESULTS.md "Noise floor"; §1 paired table (window8 +0.052, p=0.125; recorded +0.062, p=0.062) -->

### Result 4: my fine-tune learned my verifier

Then I trained on it. Qwen3.8-27B with LoRA, rejection-sampling fine-tuning:
collect the model's own conversations, keep the ones the verifier passes, train
on those, and score each fine-tune only on the tasks it didn't train on. Against
the base model: 22 tasks better, 13 worse, +2.3 points, p = 0.18. Not
significant, but pointing the right way, and I was ready to write it up as that.
<!-- RESULTS.md §6, "as recorded at run time" table: 0.859 → 0.882 -->

Most of it sat in one trap: telling a customer that a delay during the 11.11
sale isn't compensable. The base model passed 7 of 70 conversations there, the
fine-tune 26. So I read the base model's 63 failures. Every one failed on the
same line — `missing required action: get_order` — and every one had done the
right thing: no voucher, no refund, after reading the order's date from
`list_user_orders`. My check demanded one lookup; the policy needs only the
date. A photo-evidence check had the same bug. The fine-tune learns from the
conversations the checker passes, so on that trap it had learned to call
`get_order`: 26 of its 70 conversations there did, against the base model's 7.
Only there — across the whole suite it called `get_order` a little less than
the base model.
<!-- rescore.py; WHAT_FAILED #30: 63 of 63 peak failures, 59 of 67 photo failures, only the named read missing; get_order on the peak trap's own order: base 7/70, RFT 26/70; any get_order: base 717, RFT 672 of 1,075 -->

The environment is deterministic, so I didn't re-run anything. I corrected both
checks and replayed the tool calls of every conversation I had recorded —
8,941 of 9,766 reproduced call for call; the rest keep their original verdict
— and scored the rebuilt databases again. Base and fine-tune now tie: 0.972 and
0.977, 16 tasks better, 13 worse, p = 0.71. Twenty of the fine-tune's 25 extra
passes were my checker. The API model I used as a reference calls `get_order`
in 95% of its conversations, and the quirk had flattered it too: it goes from
4 points above the base model to 6.7 points below it, p = 0.007. On pass^k —
every seed of a task passing — it had never been ahead (0.800 against 0.805),
and now trails by 9.7 points.
<!-- RESULTS.md §6 (both tables) and §7; get_order in 1,024 of 1,075 reference episodes; pass^k 0.800/0.805 recorded, 0.819/0.916 now -->

It reached back to Result 1. Search keeps `get_order` in view and hides the
eligibility check, so two of search's failures were this quirk, and its
accuracy loss stopped being significant.

Before quoting any of it I went back over the fix, and it was too loose: a
read of *another* customer's orders counted, a verification with the wrong
digits counted, and nothing stopped store credit standing in for the forbidden
voucher. Tightened, it moved not one verdict. Then I read the conversations it
had promoted: all 117 on the sale-delay trap explain the extended delivery
window, and 182 of 186 on the photo trap ask for photos. A replay can re-score
everything; only reading says the re-scoring means what you think.
<!-- WHAT_FAILED #30 Fix: 0 of 9,766 verdicts moved; promoted episodes read: peak 117/117, photo 182/186 -->

What the fine-tune really changed is small and mixed. It got better at
escalating duplicate refunds, and worse at out-of-window disputes: in four
conversations it refunded a wrong-colour blouse three weeks late, and in three
of them messaged the seller — the right answer to a *different* trap, the
livestream one, which it already passed nearly every time and so trained on in
full.
<!-- RESULTS.md §6 per-trap table; WHAT_FAILED #31 -->

A verifier is a reward function the moment you train on it, and its quirks
become the model's habits. The replay that caught this cost seconds. Without
it, the headline would have been a fine-tune that "improved" and an API model
ranked above the one it trails.

### What failed

`docs/WHAT_FAILED.md` has 33 entries. Most were found by running experiments
and refusing a number that couldn't be right. Two more:

**The ceiling was below the floor.** The oracle arm — by design the best case —
scored below the full registry. I had defined what a task needs as "what the
reference solution calls", and the reference solution already knows the item
ID a real agent has to look up. I had documented exactly that bug class in an
earlier module, and rebuilt it in a new one. The fix became a rule: an oracle
may remove choices, never information.

**The judges were grading with the evidence cut out.** Traces stored which
tools were called, not what came back. I noted it as a limitation on day one
and never measured it. Measured, it was the difference between a judge that
rejects two thirds of correct work and one that rejects a seventh. The
environment is deterministic, so replaying the calls regenerates every result —
or nearly: a tracking number drawn from Python's salted hash came out different
in the replay, at the same length, my length check waved it through, and the
judges' first run was shown a number the agent never saw. The re-run withholds
those two results (627 of 629 shown), and traces now carry a digest of each
result.
<!-- RESULTS.md §4 (27/81 → 69/81); run_judges.py --payloads output: 627/629; WHAT_FAILED #33 -->

### Limits

Each tool arm is 32 tasks, so these tests see large effects only; several
verdicts are honestly inconclusive. The language comparison has 13–16 twin
pairs per language: it rules out large gaps, not small ones. One agent model
and one judge model for the first three results, one base model for the
fourth. The false-claim check reads English only, and only the phrasings I
wrote — the same kind of instrument as the leak detector, so its three are a
floor. The Thai and Vietnamese translations are not native-reviewed. The
fine-tune is one run of one recipe; a second, with training data balanced
across traps, is designed but not run. Serving cost is built but not measured.
<!-- RESULTS.md §3 paired table (pairs 13–16); audit_tool_arms.py §3 header -->

The code, the trace generators, and all 33 failures are in the repo.

---

# Part 2 — Resume lines

One line each, a number, and the mechanism. Pick three.

> **Built PasarBench**, a verifiable customer-service agent environment for SEA
> e-commerce — 215 tasks, 6 markets, 8 language varieties, 20 tools — scored on
> end-state database, with locale twins sharing byte-identical pass criteria.

> **Caught a fine-tune learning my verifier instead of my policy**: RFT on
> Qwen3.8-27B gained 2.3 points (p = 0.18) only because two checks demanded one
> lookup tool the policy didn't require; re-scoring 9,766 recorded episodes by
> deterministic replay — no GPU — cut the gain to 0.5 (p = 0.71) and put the
> API reference 6.7 points below the base model (p = 0.007), where the quirk
> had put it 4 points above.

> **Showed an LLM judge cannot stand in for state verification**: its best
> configuration reached κ = 0.72 against the database and still accepted 6 of 15
> failed episodes, crediting actions the tool log showed never happened. Removed
> the missing-evidence confound with deterministic tool-result replay.

> **Measured search-based tool exposure**: 22% more tokens for no measurable
> accuracy gain (11.5 points lower, p = 0.07); replayed every query to show the
> agent never searched for tools its policy described but didn't name (17 of
> 21 missing-tool cases).

> **Traced a measured simulator bias to my own detector**: an 11–22% "leak"
> rate concentrated in four languages, and three runs of fixes, came from
> ask-patterns that couldn't read the agent's questions; corrected, 3 of 4,155
> episodes are flagged, all English. The same blind spot had produced the
> project's only significant language gaps.

> **Documented 33 defects**, most caught by refusing implausible results —
> including a "ceiling" arm that scored below baseline, a judge evaluated
> without the evidence it was judging, and a fine-tune rewarded for a checker's
> quirk.

---

# Part 3 — Interview prep

### The 60-second version

> I built a customer-service agent environment for Southeast Asian e-commerce —
> 215 tasks, six markets, eight languages — that checks the database at the end
> of a conversation instead of the text of the reply.
>
> The result I'd lead with: when the agent had to search for its tools, it
> sometimes never found the escalation tool — and told the customer "I've
> escalated your case" anyway. I gave an LLM judge those conversations with
> the tool results and the full policy. It credited one fake escalation, and
> for the other wrote that the escalation tool was never called — and passed
> it anyway. So I'd check actions against state and use judges for language.
>
> The thing I'd want to talk about is how I found my own bugs. The best two: a
> simulator bias I spent three runs fixing was my regex; and a fine-tune whose
> gain was almost all my checker's quirk — which I measured by replaying nearly
> 9,000 recorded conversations instead of paying for a GPU re-run.

### Six stories worth having ready

**1. "I've escalated your case."** State versus text, told through two
transcripts: a reply that reads perfectly over an empty table, and a judge that
noticed the missing call and passed it anyway.

**2. The bias that was my detector.** A leak rate that followed language,
three runs of fixes, a fix that broke the comparison, and a tie-break that put
the breakage in the report as a finding. One printed line per case undid all
of it. Lands: a detector's rate is a claim about the detector until you read
what it flagged.

**3. The ceiling below the floor.** A design invariant — oracle ≥ baseline —
that nothing asserted. When it broke, the cause was a bug class I'd already
documented once. Lands: invariants belong in code, and a fix that lists cases
fixes those cases.

**4. Replay instead of re-run.** A deterministic environment doesn't need to
store what it can recompute. Regenerating tool results removed a confound for
free, and the same replay later re-scored every recorded episode when two
checks changed. The recorded payload lengths let it prove itself call by call —
until a salted hash changed a tracking number at the same length. Lands: verify
content, not size, and test "deterministic" across processes.

**5. The calibration that measured nothing.** 1,800 labels, 2 violations. Not
a labelling failure — a sampling one: stratifying on task failure doesn't give
you communication failures. Say what you'd do differently.

**6. The fine-tune that learned the verifier.** +2.3 points that were almost
all one check demanding `get_order`. Reading 63 failures found it; replaying
every recorded episode measured it for free; the same quirk had ranked the API
model above the base model on pass^1. Then read what the fix promoted, because
the first version of it was too loose. Lands: a verifier is a reward function
once you train on it — read what its passes have in common before you do.

### Questions you will be asked

**"Isn't an LLM judge fine with a good enough prompt?"** The evidence mattered
more than the prompt. Giving the rubric judge the tool results took it from
rejecting two thirds of correct work to a seventh; adding the policy took the
other judge from κ = 0.20 to 0.72. And both still passed claims the tool log
contradicted — which ones changed when I re-ran the same judge on the same
evidence. A claim about an action is checkable, so check it.

**"Why is search worse — isn't retrieval the fix for big tool sets?"** At 32
tasks it isn't measurably less accurate (p = 0.07); it is 22% more expensive.
Retrieval fixes finding a tool you're looking for. The failures here were tools
the agent never looked for, because nothing told it they existed. The test I'd
run next names those tools in the policy: if search recovers, that's the
mechanism.

**"How do you know your false-claim detector isn't wrong the way the leak
detector was?"** I don't, fully — it's the same kind of instrument. So it's
scoped to English, the report says other languages are counted, not read, and
I read each of the three against the tool log. What it can't tell me is how
many claims it missed, so three is a floor.

**"How do you know the language null isn't just low power?"** It is partly low
power, and I'd say so: 13–16 twin pairs per language rule out large gaps, not
small ones. What makes it believable is that it held across two independent
runs, and that the one "significant" gap traced to a bug.

**"Is 3 points of run-to-run noise normal?"** It's what I measured for this
agent and simulator at 32 tasks. It's why every comparison in the report is a
paired test, and why several verdicts say INCONCLUSIVE.

**"So did RFT work?"** Not measurably, once the checker was fixed: +0.5 points,
p = 0.71. What it did do is move behaviour between traps — better at
escalating duplicate refunds, worse at out-of-window disputes, where it copied
the livestream trap's answer. The next run would build its data with the
corrected checks and balance it by trap, and I'd read the per-trap table as the
result. GRPO isn't worth it yet: at 0.97 the suite is nearly saturated for this
model, so it needs harder traps first.

**"Why didn't you re-run the evaluation after fixing the checker?"** I didn't
need to. The environment is deterministic, so replaying each episode's tool
calls rebuilds its final database exactly, and the fixed checks score that.
Every replay is verified against what was recorded — a call that succeeds on
replay where it failed in the run, or a result that differs, and the episode
keeps its old verdict. 8,941 of 9,766 reproduced; all but one of the rest are
a run from before an ID format change. Then I read the conversations the fix
had promoted — a replay proves the state, not that the state means what the
check assumes.

**"What would you do next?"** Name the missing tools in the policy and re-run
search — a direct test of the mechanism. Add the claim check as a guardrail and
measure how many false claims it stops. A second fine-tune on trap-balanced
data built with the corrected checks. Get the Thai and Vietnamese translations
reviewed. Then serving cost.

### One thing not to do

Don't present a co-movement as a cause, and don't round INCONCLUSIVE up to a
finding. Several of the most useful results here are nulls; say them as nulls.
