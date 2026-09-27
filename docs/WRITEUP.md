# Writeup kit

Blog post, resume lines, interview prep — drafted from the measured results.

**Every number here comes from a file generated from disk**, and each carries a
hidden comment naming it. The generators:

```bash
python scripts/make_report.py --out RESULTS.md --tools-run I-tools2 \
  --multilingual-run C-clean,D-nozh,G-gated \
  --noise-pair H-context/full,I-tools2/full+all-20
python scripts/audit_tool_arms.py traces/I-tools2
python scripts/inspect_trace.py traces/D-nozh/full --leaks-only                    # leaks now
python scripts/inspect_trace.py traces/D-nozh/full --leaks-only --ask-patterns v1  # leaks as first reported
```

Full judge reasons (RESULTS.md truncates them) are in
`data/judge_vs_verifier/I-tools2__full+search-300__payloads__policy.jsonl`.

If you change a number, change it by re-running, never by hand. A published
post containing a number you never ran is a worse outcome than no post.

**Models, as served.** Agent `deepseek-v4-pro`; simulated customer and judges
`qwen3.8-flash` — a different family from the agent, to avoid self-preference.
No alias re-routing was recorded in the traces. Runs of 2026-09-18 to 09-19
(language) and 2026-09-25 (context, tools, judges).

This draft is in your voice. Rewrite anything that doesn't sound like you — the
numbers are the part not to touch.

---

# Part 1 — The blog post

**Target:** ~2,000 words. **Title options**, finding first:

- *The agent said it escalated. The database said it didn't.*
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

I gave an LLM judge the conversation, every tool result the agent had seen,
and the policy it was working under. The judge credited the agent with having
"escalated properly", then marked it down for reporting the wrong tracking
number.
<!-- RESULTS.md §4, CHE-MY__r2, "+ tool results + policy": naive 3 → rejected; full reason in …__payloads__policy.jsonl -->

On another conversation it went further. The agent had told the customer "I'm
escalating this for further investigation". The judge wrote that "the actual
escalation tool call is not visible in the transcript … suggesting the
escalation may not have been fully executed" — and passed it, 4 out of 5.
<!-- RESULTS.md §4, DRE-PH__r2, "+ tool results + policy": naive 4 → accepted; full reason in …__payloads__policy.jsonl -->

This post is about the distance between what an agent says and what it does,
and what it takes to measure it. Three results: what happens when an agent has
to *search* for its tools; whether an LLM judge can tell a solved case from an
unsolved one; and a language effect that turned out to be my own regex.

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

Search was worse on both axes: 13.5 points below showing all 20 tools (9 tasks
worse, 1 better, p = 0.021), *and* 22% more tokens per episode. Each call
carried 55% fewer tokens of tool definitions, but searching added almost three
steps per conversation, and every step re-sends the conversation.
<!-- RESULTS.md §2: -0.135, p=0.021; 46,423 vs 38,041 (+22%); steps 10.4 vs 7.7; schema 793 vs 1,762 -->

Why? The ranker is deterministic, so I replayed every query the agent issued
and reconstructed exactly what it was shown. There were 23 cases of a failed
episode needing a tool the arm had hidden. In 17 of them, the agent **never
searched for it**. The three tools it didn't look for — escalation, shipment
lookup, goodwill voucher — have one thing in common: the policy describes those
actions but never names the tools. The tools the policy *does* name, the agent
called without searching; the harness refused them as unknown, and 23 times out
of 26 it then found them by search and used them.
<!-- audit_tool_arms.py §2 (never searched 10+5+2 of 23; "NOT named in policy") and §1 (26 refused, 23 found & used, 26/26 policy-named) -->

It searched for tools it knew existed. It did not search for tools it would
have had to imagine. In five of the six failed customs-hold episodes it never
looked up the shipment at all, and told the customer the delay was normal for
the 11.11 sale.
<!-- audit_tool_arms.py §2: get_shipment never searched 5 of 6; "last said" lines for CHE-MY r0/r1, CHE-VN r0/r1/r2 -->

And this is where the false claims came from. Three of the sixteen search-arm
failures I could read said an action had happened that never did. In the four
arms where the needed tools were always visible: none, in twenty failures. The
numbers are small, but the shape is right — an agent that knows what it should
do and can't find the tool to do it says it did it anyway.
<!-- audit_tool_arms.py §3: search-300 read 16, claimed-not-done 3; all-100 0/7, all-20 0/4, oracle 0/2, random-100 0/7 -->

### Result 2: an LLM judge takes the agent's word for it

The usual way to evaluate conversations at scale is an LLM judge. I had
calibrated two against 200 hand-labelled transcripts, and that calibration
measured nothing: of 1,800 human judgments, 2 were violations. Forty-one of
the 43 transcripts that failed the verifier were labelled clean on every
criterion — probably correctly. Every one was an action failure, and the rubric
scores language. An agent that politely does the wrong thing satisfies it.
<!-- RESULTS.md §4 calibration notice; WHAT_FAILED #24 -->

So I scored the judges against the database instead, on the 96 search-arm
episodes: 79 passed, 17 failed, 3 of those claiming an action that never
happened. Two judges — a single 1–5 score, and a nine-criterion rubric — each
under three views: the transcript alone, plus the tool results, plus the
policy.
<!-- RESULTS.md §4 "Judges against the verifier" -->

| judge | what it saw | correct accepted | failures accepted | kappa vs database |
|---|---|---|---|---|
| 1–5 score | transcript | 71/77 | 13/17 | 0.19 |
| 1–5 score | + tool results | 74/79 | 14/17 | 0.14 |
| 1–5 score | + tool results + policy | 76/79 | 8/17 | **0.56** [0.33, 0.76] |
| 9 criteria | transcript | 27/79 | 3/17 | 0.08 |
| 9 criteria | + tool results | 71/79 | 13/17 | 0.15 |
| 9 criteria | + tool results + policy | 73/79 | 13/17 | 0.19 |
<!-- RESULTS.md §4 table (the 1–5 judge's transcript-only row has 2 unparsed) -->

Three things in that table.

**What the judge can see decides everything.** Without the tool results, the
rubric judge rejected two thirds of the correct episodes, 51 of the 52 for
"hallucinated facts" — facts the agent had read from tool results the judge
couldn't see. Given the results, it rejected 8.
<!-- data/judge_vs_verifier/I-tools2__full+search-300.jsonl: 52 correct rejected, 51 no_hallucinated_facts; …__payloads.jsonl: 71/79 -->

**Only one configuration beats chance, and it still misses half.** The 1–5
judge with the tool results and the policy reaches κ = 0.56 — and accepts 8 of
the 17 episodes the database fails.

**Given everything, neither judge rejected a false claim for being false.** The
1–5 judge credited all three fake actions: the agent "escalated properly";
"followed P10's escalation requirement"; "offered a goodwill voucher per P6.2
with accurate currency calculation". The one it rejected, it rejected for other
errors, a wrong tracking number among them. The rubric judge rejected the same
transcript, for the tracking number. With the tool results but not the policy,
it did catch the voucher line — and filed it as a minor "unfounded promise", so
the transcript passed.
<!-- RESULTS.md §4, the three false-claim transcripts under every view; full reasons in …__payloads__policy.jsonl -->

The lesson is not that LLM judges are useless. It is that whether an action
happened is a question about state, and state can be checked. Check outcomes
against the database, check claimed actions against the tool log —
mechanically — and keep the LLM judge for what can't be checked: tone,
clarity, language.

### Result 3: the language effect was my regex

The multilingual comparison is the one the benchmark was built for. In two
independent runs — the full suite, and the suite without Chinese — no language
differed from English beyond noise: the largest paired gap was 9 points, and
every sign test had p ≥ 0.38.
<!-- RESULTS.md §3 replication table, C-clean (215 tasks) and D-nozh (186) -->

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
only significant language gaps in the project: Chinese 26 and 29 points below
English. When my report picked a run for its language section, a tie-break
chose that one, and pointed at tokenisation.
<!-- inspect_trace.py traces/G-gated/full --leaks-only: requests not answered id 207/280, zh-MY 478/561, en 183/1,093; RESULTS.md §3 replication table, G-gated column; WHAT_FAILED #25, #26 -->

One pattern list produced a leak rate, a fix that seemed to backfire, a fix
that broke the comparison, and a false finding. What settled it was one printed
line per case.

How much does a result move on its own? The same configuration, run twice,
scored 88/96 and 92/96 — 4 points, p = 0.22. The best context-management
strategy I tested beat the baseline by 6 points, at p = 0.06. I can't tell it
from re-running.
<!-- RESULTS.md "Noise floor"; §1 paired table (window8 +0.062, p=0.062) -->

### What failed

`docs/WHAT_FAILED.md` has 26 entries. Most were found by running experiments
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
rejects two thirds of correct work and one that rejects a tenth. The
environment is deterministic, so replaying the calls regenerates every result —
629 of 629 verified against the lengths recorded at run time.
<!-- RESULTS.md §4 (27/79 → 71/79); run_judges.py --payloads output: 629/629 -->

### Limits

Each tool arm is 32 tasks, so these tests see large effects only; several
verdicts are honestly inconclusive. The language comparison has 13–16 twin
pairs per language: it rules out large gaps, not small ones. One agent model,
one judge model. The false-claim check reads English only, and only the
phrasings I wrote — the same kind of instrument as the leak detector, so its
three are a floor. The Thai and Vietnamese translations are not native-reviewed.
Serving cost and post-training are built but not yet run — they need GPUs.
<!-- RESULTS.md §3 paired table (pairs 13–16); audit_tool_arms.py §3 header -->

The code, the trace generators, and all 26 failures are in the repo.

---

# Part 2 — Resume lines

One line each, a number, and the mechanism. Pick three.

> **Built PasarBench**, a verifiable customer-service agent environment for SEA
> e-commerce — 215 tasks, 6 markets, 8 language varieties, 20 tools — scored on
> end-state database, with locale twins sharing byte-identical pass criteria.

> **Showed an LLM judge cannot stand in for state verification**: its best
> configuration reached κ = 0.56 against the database and still accepted 8 of 17
> failed episodes, crediting actions the tool log showed never happened. Removed
> the missing-evidence confound with deterministic tool-result replay (629/629
> verified).

> **Measured search-based tool exposure**: 13.5 points less accurate than
> exposing all tools (p = 0.021) at 22% more tokens; replayed every query to show
> the agent never searched for tools its policy described but didn't name
> (17 of 23 missing-tool cases).

> **Traced a measured simulator bias to my own detector**: an 11–22% "leak"
> rate concentrated in four languages, and three runs of fixes, came from
> ask-patterns that couldn't read the agent's questions; corrected, 3 of 4,155
> episodes are flagged, all English. The same blind spot had produced the
> project's only significant language gaps.

> **Documented 26 defects**, most caught by refusing implausible results —
> including a "ceiling" arm that scored below baseline and a judge evaluated
> without the evidence it was judging.

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
> every tool result and the full policy. It credited the fake escalations, and
> in one case wrote that the escalation call wasn't visible and passed it
> anyway. So I'd check actions against state and use judges for language.
>
> The thing I'd want to talk about is how I found my own bugs. The best one: a
> simulator bias I spent three runs fixing was my regex.

### Five stories worth having ready

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
free — and the recorded payload lengths let the replay prove itself call by
call.

**5. The calibration that measured nothing.** 1,800 labels, 2 violations. Not
a labelling failure — a sampling one: stratifying on task failure doesn't give
you communication failures. Say what you'd do differently.

### Questions you will be asked

**"Isn't an LLM judge fine with a good enough prompt?"** The evidence mattered
more than the prompt. Giving the rubric judge the tool results took it from
rejecting two thirds of correct work to a tenth; adding the policy took the
other judge from κ = 0.14 to 0.56. And both believed claims the tool log
contradicted. A claim about an action is checkable, so check it.

**"Why is search worse — isn't retrieval the fix for big tool sets?"**
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

**"Is 4 points of run-to-run noise normal?"** It's what I measured for this
agent and simulator at 32 tasks. It's why every comparison in the report is a
paired test, and why several verdicts say INCONCLUSIVE.

**"What would you do next?"** Name the missing tools in the policy and re-run
search — a direct test of the mechanism. Add the claim check as a guardrail and
measure how many false claims it stops. Get the Thai and Vietnamese
translations reviewed. Then serving cost and post-training on GPUs.

### One thing not to do

Don't present a co-movement as a cause, and don't round INCONCLUSIVE up to a
finding. Several of the most useful results here are nulls; say them as nulls.
