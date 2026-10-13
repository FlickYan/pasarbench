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
python scripts/make_figure.py --traces traces                                      # the figure in README and the blog
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

`docs/BLOG.md` is Part 1 as a post ready to publish: no source comments, no
recorded-verdict asides, and the two "What failed" stories folded into a closing
list of lessons. Its numbers are the ones below, so a number that changes here
changes there too.

**Models, as served.** Agent `deepseek-v4-pro`; simulated customer and judges
`qwen3.8-flash` — a different family from the agent, to avoid self-preference.
No alias re-routing was recorded in the traces. Runs of 2026-09-18 to 09-19
(language) and 2026-09-25 (context, tools, judges); the two judge views with
tool results were re-run on 2026-09-30, withholding the two results no replay
can rebuild (WHAT_FAILED #33); the tool-naming experiment and its control ran
on 2026-10-02. Post-training, 2026-09-29:
agent `Qwen/Qwen3.8-27B` (base, and a LoRA fine-tune per fold) served with
SGLang on one B200, customer `gemma-4-31B-it` (FP8) on the same card, and
`deepseek-v4-pro` through its API against the same customer as the reference.
Round 2 and GRPO, 2026-10-11 to 10-12: the same agent, served the same way,
against the second version of that customer (WHAT_FAILED #38).

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
and what it takes to measure it. Five results: what happens when an agent has
to *search* for its tools; whether an LLM judge can tell a solved case from an
unsolved one; a language effect that turned out to be my own regex, and one
that didn't; a fine-tune that learned my verifier instead of my policy; and a
second look after the customer leaves, which rescued what the agent let go and
then did things twice.

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

183 of the 215 tasks have **locale twins**: the same world and byte-identical
pass criteria, in another language. Only the words change, so a gap between
twins is language and nothing else.
<!-- python -c "from collections import Counter; from pasarbench.rescore import all_tasks; t = all_tasks(); c = Counter(id(x.checks) for x in t.values()); print(sum(c[id(x.checks)] > 1 for x in t.values()), 'of', len(t))"  -> 183 of 215; the 32 without a twin are the 16 PH tasks (English only) and the hand-written T01-T16 -->

### Result 1: the agent searches for tools it already knows about

Production tool registries can run to hundreds of tools, so a common pattern is
to show the agent a small core plus a `search_tools` tool and let it retrieve
the rest. I compared five exposures on the same 32 tasks, three seeds each: an
oracle set (every lookup, plus only the actions the task needs); all 20 tools;
100 (20 plus 80 plausible distractors); a random 100 that always contains what's
needed; and search over 300.
<!-- RESULTS.md §2 -->

With 100 tools the agent was not measurably less accurate than with the oracle
set — 3 tasks worse, none better, p = 0.25 (4 and p = 0.125 on the recorded
verdicts) — but each episode cost 2.7 times the tokens of the 20-tool registry.
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
arms where the needed tools were always visible: none, in nineteen failures. The
numbers are small, but the shape is right — an agent that knows what it should
do and can't find the tool to do it says it did it anyway.
<!-- audit_tool_arms.py §3: search-300 read 14, claimed-not-done 3; all-100 0/6, all-20 0/4, oracle 0/2, random-100 0/7 -->

If not knowing the tools existed was the problem, naming them should fix it.
The named policy is the same text with the three tool names added where it
describes each action — *Escalate with category `customs_hold` (tool:
`escalate_to_human`)* — and nothing else. I ran it on the same 32 tasks, three
seeds each, back to back with a control that had no names. The control repeated
the first search run: 81 of 96 passed both times, and it made three false
claims again, on the same three tasks.
<!-- RUNBOOK 1g; --policy-mode preload-named (pasarbench/harness/prompts.py, NAMED_MARK); traces/J-preload and traces/J-preload-named, 2026-10-02 08:17-08:28 UTC; compare_cells.py: control 0.844 = I-tools2 search-300 0.844 (81/96, pass^k 0.750 both); audit_tool_arms.py traces/J-preload §3: claimed-not-done 3 of 14 read (CHE-MY, DRE-PH, OOWOV-TH) -->

With the names, the agent went looking. In 21 of the 24 conversations that
needed an escalation, it called the tool by name before it had been shown it;
the harness refused, and the agent searched for the tool and used it — the
pattern it had always shown with tools the policy named. Escalations went from
12 of 24 to 21, shipment lookups from 3 of 18 to 12. Accuracy rose from 0.844 to
0.938, most of the way to the 0.958 of showing all 20 tools: 6 tasks better,
1 worse, p = 0.125 — the direction the explanation predicts, but 32 tasks can't
confirm it. None of the six failures left claimed an action that hadn't
happened.
<!-- python scripts/compare_cells.py traces/J-preload/full+search-300 traces/J-preload-named/full+search-300: escalate_to_human 12/24 (0) -> 21/24 (21); get_shipment 3/18 -> 12/18; pass^1 0.844 -> 0.938; paired +0.094, 6 better 1 worse, p=0.125 INCONCLUSIVE. audit_tool_arms.py traces/J-preload-named §1: 88 hidden calls refused, 87 found & used; §3: read 6, claimed-not-done 0 -->

So I wrote down beforehand what would count as confirmation, and asked again,
of 57 tasks the first run hadn't used: every task that can't pass without one
of the three tools. Without the names the agent passed 44% of those
conversations; with them, 80%. Thirty-eight tasks got better and four worse,
p < 0.000001, and every language moved the same way. Escalations went from 63
of the 135 conversations that needed one to 117, and the false claims went with
them: 12 of the 42 failures I could read claimed an action never taken without
the names, none of 13 with them. What still failed was mostly judgment, not
search — duplicate refunds explained and closed without the escalation the
policy asks for.
<!-- RUNBOOK 1h (rule fixed before the run); python scripts/naming_tasks.py (57 tasks); traces/K-preload and traces/K-preload-named, 2026-10-07 17:57-18:13 UTC; compare_cells.py: pass^1 0.439 -> 0.801, paired +0.363, 38 better 4 worse, sign test p = 5.7e-8 BETTER; escalate_to_human 63/135 -> 117/135 (110), get_shipment 20/27 -> 27/27, voucher 18/36 -> 32/36; audit reading 12/42 -> 0/13; per language en 42->67/78, id 2->9/15, ms 7->13/15, sg-en 3->7/9, th 8->14/15, vi 3->9/12, zh-MY 5->9/15, zh-SG 5->9/12; audit_tool_arms.py traces/K-preload-named §2: 34 failed, 16 DRE never escalated, 5 DRE category duplicate_refund_claim, 7 IVF order_id unknown/blank, 2 IVF never escalated, 4 vouchers never searched -->

Naming didn't make search cheap: the extra calls cost 13 to 18% more tokens per
conversation, and showing all 20 tools had been cheaper still. If an agent has
to search for its tools, name them where its instructions describe the action.
If there are only twenty, show them.
<!-- compare_cells.py: J tokens 44,187 -> 52,354 (+18%), p=0.002; K tokens 40,019 -> 45,224 (+13%), 39 costlier 18 cheaper, p=0.008; RESULTS.md §2 all-20: 38,041 tokens, 0.958 (I-tools2, 2026-09-25) -->

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

Then I gave the judge answers to find. Labels with almost no violations
calibrate nothing, so I made the violations: one mistake planted in a copy of a
transcript the database passed — another customer's order read out, a total no
tool gave, a 30-day return window, the identity check taken out — beside the
untouched original. Against a bar fixed before the run, four of nine criteria
passed: leaks, invented facts, an unexplained escalation, rudeness. The worst was
the one about an action. With the identity check taken out, the rubric judge
flagged it once in 20, quoting the agent's request for the digits as the check;
the 1–5 judge passed 19 of those 20 copies. To a judge, asking for a check is
the check.
<!-- RUNBOOK, planted mistakes, "What it found"; judge_calibration.py report: verification_before_action caught 1/20, the 1-5 judge 18/20 -> 19/20; trusted: no_data_leak, no_hallucinated_facts, escalation_explained, tone_professional -->

The lesson is not that LLM judges are useless. It is that whether an action
happened is a question about state, and state can be checked. Check outcomes
against the database, check claimed actions against the tool log —
mechanically — and keep the LLM judge for what can't be checked: tone,
clarity, language — calibrated too, since on planted mistakes it caught every
rude line, but only two in three English sentences dropped into a reply in
another language.

So I built that check into the agent. A guardrail reads each reply before the
customer does; if it claims an action no tool call in the conversation has
done, the reply is held back and the agent is told which call would make it
true. It reads the reply and the calls, never the task's answer, so a deployed
agent could run it. Behind search it held back two replies, "I've escalated
your case" and "I'll issue the goodwill voucher now", and both times the agent
made the call and the case passed.
<!-- RUNBOOK 1i; pasarbench/harness/guardrail.py (v24, first-person only); traces/L-off and traces/L-claims, 2026-10-07 18:13-18:22 UTC; compare_cells.py: held back 2 replies in 2 episodes (OOWDE-MY__r0, OOWOV-TH__r0), made the call 2; both episodes passed -->

It still let three false claims through, all like "Your voucher has been
issued". I had told it to ignore passives, because in the duplicate-refund
trap "your refund was issued on 9 November" is true, and read as a claim it
would push the agent towards a second refund. And it passed the test I had
written for it, because the test read claims the way the guardrail did: from
two false claims reaching customers to none. The audit, which reads more
phrasings, counted three with the guardrail and three without. The next
version reads passives for vouchers and escalations, which no conversation in
this world starts with, and is graded by the audit's reading, not its own.
<!-- WHAT_FAILED #34; v24 reading 2/96 -> 0/96 (the rule written for v24); audit_tool_arms.py traces/L-off and traces/L-claims §3: claimed-not-done 3 of 15 read -> 3 of 18 (L-claims: OOWOV-TH__r1, OOWOV-TH__r2, OOWOV-VN__r1, all passive); v25 reading 3/96 and 3/96; pass^1 0.812 -> 0.802, 2 better 3 worse, p=1.0; scripts/guardrail_dry_run.py traces: v25 fires on 46 of 10,492, 44 in failed episodes -->

Graded that way, the next version worked. Against a same-day control on the
same 32 tasks, false claims reaching customers went from 5 of 15 readable
failures to none of 13. It held back four replies, all real, and each time the
agent found the tool, made the call and passed. Two were passives: in the
control, one customer was told "Your goodwill voucher of VND 220,000 has been
issued" over an empty vouchers table, and the case failed; in the guarded run
the same claim was held back twice, and twice the voucher was issued. Five
against none is p = 0.06 on the count alone; the held replies, each of which can
be read, are the stronger evidence. The pass rate rose only within noise (0.823
to 0.865, p = 0.69), and there was little more for it to do: ten of the
thirteen guarded failures never escalated at all. A check on what the agent
says can make its words match its actions. It cannot make it decide to act.

Reading every failure by hand shows the boundary. Claims in the words my two
readers know — "I've escalated", "I'm escalating this now", "has been issued" —
are in six of the control's English failures and none of the guarded run's;
two more in the control are in Vietnamese, which neither reads, and the
guarded run had no failure outside English to test that. Other words got
through in both, two in the control and three guarded: a voucher promised
and never issued, "it will be reviewed by our team" with no escalation, and
once a word I had no pattern for — "your case has been flagged with all the
details confirmed". A list of phrasings will always be one step behind. Whether
the customer was told something would happen is a question about language,
which a model can answer; whether it happened is a question about state, which
the tool log answers.
<!-- RUNBOOK 1i, v25; traces/L2-off and traces/L2-claims, 2026-10-08 08:45-08:56 UTC; compare_cells.py: audit reading 5/15 -> 0/13, guardrail reading 6/96 -> 0/96; held 4 in 4 episodes (DRE-PH__r2, OOWDE-MY__r0, OOWOV-VN__r0, OOWOV-VN__r2), made the call 4, all 4 passed; L2-off OOWOV-VN__r2 failed after "Your goodwill voucher of **VND 220,000** has been issued."; Fisher's exact 5/96 vs 0/96, two-sided p=0.059; pass^1 0.823 -> 0.865, 4 better 2 worse, p=0.688; tokens -1,564/ep, p=0.597; guarded failures: CHE-MY x2, CHE-VN x3, DRE-PH x2, T14 x3 never escalated; hand reading of all failures (RUNBOOK 1i): claims in known words 6/15 English -> 0/13 (control: the audit's 5 + LCOW-MY__r2), plus control OOWDE-VN.vi__r1, __r2 in Vietnamese (no guarded failure outside English), other words 2 -> 3 (control LCOW-MY__r0 promise, T14__r1 implied review; guarded OOWOV-VN__r1 promise, DRE-PH__r1 implied review, HPRR-PH__r2 "has been flagged") -->

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
only significant language gaps in the project: Chinese 21 and 28 points below
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

The null is the suite's, though, and one rule is not in it. Pooled over sixteen
traps, a gap on one moves a language's mean by a few points. P10 asks for an
escalation when a customer chases a refund already paid, "even if you have
explained the existing refund and the customer accepts it" — the one escalation
that goes against the obvious service move. Read back over every recorded run,
deepseek-v4-pro made it in 59% of English conversations and in 2–8% of Malay and
Chinese ones; all 14 local-language tasks made it less often than their English
twin, by 38 points on average (p = 0.0001), in the runs recorded before anyone
looked as well as on new tasks. It decides when it reads the refund. In English
it writes "Since a refund has already been issued for this order, I need to
escalate this for review", and does; in Vietnamese it lists the same refund,
asks the customer to check her card statement, and stops. On the policy's three
other escalations the languages are level; trap by trap, this is the only gap
that stands once sixteen are counted; and Qwen3.8-27B shows no gap. What
recorded runs can't separate is the language from everything else that differs
between two conversations; that takes a run that changes only the language.
<!-- RUNBOOK 1r; python scripts/language_lead.py traces, deepseek-v4-pro, the 16 cells that ran the trap on both sides, 778 episodes: en 223/378, sg-en 37/60, id 30/60, vi 22/60, th 21/60, zh-SG 4/50, ms 4/60, zh-MY 1/50; twins 14 of 14 below, mean -38 points, sign test p=0.0001; the 7 runs before the lead (B, C, D, F, G, K-preload, P-ref) 7/7, -21, p=0.016; second edition (Q, R, S) 7/7, -47, p=0.016; 341 of 342 escalations in the turn of the first get_order; customs hold, identity, out-of-window dispute twins level 8, 3, 11 of 14, mean gaps 0, +3, -1; P-base (Qwen3.8-27B) 38/40 vs 26/30, P-ref 25/40 vs 12/30; Q-off DRE2-VN 3/3 escalated, DRE2-VN.vi 0/3; every trap on the recorded verdicts, cells without a closing phase: DRE 12 of 14 below, -34 points, p=0.0005 against 0.05/16=0.0031, next below -5 to -8, livestream +12 p=0.031, identity +14 p=0.227 -->

So I ran it. I held 96 recorded conversations fixed at the moment the agent
reads the refund — the refund, the tools and the world unchanged — translated
every turn of hers and every reply of its into the other language, and let it
decide again, three times each. The same conversation got the escalation more
often in English: in 47 conversations more often, in 16 less (p = 0.0001), and
the same way in both directions — moved into English, a conversation's
escalations went from 16% of runs to 35%; moved out of it, from 50% to 31%.
Translated there and back, without changing the language, it did not change
significantly (p = 0.36). By language — read, not tested, 16 conversations each
— it is Malay, Chinese and Vietnamese; in Thai and Indonesian the agent decided
as it does in English. The language is a little over half of the gap between
these recordings. The rest stays with the conversation — her words and the
agent's own earlier replies — which a run that swaps each alone would separate.
<!-- RUNBOOK 1t, run on 2026-10-13, data/swap/W-swap, W.txt: 96 prefixes x 3 arms x 3 = 864 continuations of deepseek-v4-pro, translator qwen3.8-flash; reproduction 75/96; swap vs back, English minus the other language: +0.191, 47 vs 16, p=0.0001; recorded in English +0.194, 25 vs 10, p=0.017; recorded in another language +0.188, 22 vs 6, p=0.004; orig vs back +0.024, 25 vs 18, p=0.36; escalated, orig/swap/back: recorded in English 45/31/50%, recorded in another language 16/35/16%; English vs it: id 40/42%, ms 40/6%, th 25/21%, vi 65/46%, zh-MY 33/10%, zh-SG 52/15% -->

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
— and scored the rebuilt databases again. Base and fine-tune now tie: 0.979 and
0.981, 13 tasks better, 11 worse, p = 0.84. Twenty-two of the fine-tune's 25
extra passes were my checker. The API model I used as a reference calls `get_order`
in 95% of its conversations, and the quirk had flattered it too: it goes from
4 points above the base model to 6.6 points below it, p = 0.004. On pass^k —
every seed of a task passing — it had never been ahead (0.800 against 0.805),
and now trails by 9.7 points. Up to 1.9 of those 6.6 points are a habit the
environment should have stopped: the reference escalated unverified callers
under order "unknown", and the escalation tool took it (WHAT_FAILED #35). Had
every one been refused and then fixed, it would still trail, by 4.7 points
(p = 0.012).
<!-- RESULTS.md §6 (both tables) and §7; get_order in 1,024 of 1,075 reference episodes; pass^k 0.800/0.805 recorded, 0.833/0.930 now; #35 bound: 21 reference failures, all identity-trap escalations naming no order, none in base or RFT; best case 0.932 against 0.979, 12 better 29 worse, p=0.0115; pass^k 179 -> at most 182 of 215 tasks -->

It reached back to Result 1. Search keeps `get_order` in view and hides the
eligibility check, so two of search's failures were this quirk, and its
accuracy loss stopped being significant.

Before quoting any of it I went back over the fix, and it was too loose: a
read of *another* customer's orders counted, a verification with the wrong
digits counted, and nothing stopped store credit standing in for the forbidden
voucher. Tightened, it moved not one verdict. And the same question — does
the policy require this? — caught one more requirement: the photo trap still
demanded identity verification, which the policy asks for only before a
write, in a case whose right answer writes nothing. Then I read the
conversations the fixes had promoted: all 117 on the sale-delay trap explain
the extended delivery window, and 230 of 234 on the photo trap ask for photos.
A replay can re-score everything; only reading says the re-scoring means what
you think.
<!-- WHAT_FAILED #30 Fix: tightening moved 0 of 9,766 verdicts; promoted episodes read: peak 117/117, photo 230/234 -->

What the fine-tune really changed is small and mixed. It got better at
escalating duplicate refunds, and worse at out-of-window disputes: in four
conversations it refunded a wrong-colour blouse three weeks late, and in three
of them messaged the seller — the right answer to a *different* trap, the
livestream one, which it already passed nearly every time and so trained on in
full.
<!-- RESULTS.md §6 per-trap table; WHAT_FAILED #31 -->

So I ran it again the way that called for: the training data balanced by trap,
twenty passing conversations each, picked by the corrected checks, against a
fixed customer. The same null — 0.962 against 0.954, 11 tasks better and 14
worse, p = 0.69 — and what I had committed to in advance was the failures, not
the number. The livestream refund fell back to the base model's rate. The sale
delay's answer still turned up on customs holds — "it's 11.11, the delay is
normal, please wait" — in four Chinese conversations. And there was a new one:
the photo trap, now as present in the data as any other, lent out its rule.
Asked to refund a MYR 186 serum, the fine-tune asked for photos "since it's an
item worth over SGD 200" in all five conversations; MYR 186 is SGD 56, and the
base model never asked. Balancing decides which successes spread. It doesn't
stop them spreading.
<!-- RUNBOOK phase 4, round 2, "What it found"; WHAT_FAILED #44: compare_cells.py traces/P2-base/full traces/P2-rft/full; T02 photos asked 5/5 RFT, 0/5 base; CHE-MY.zh-MY r1-r4 -->

Then GRPO, from those fine-tunes: the model trained on the difference between
its passing and failing attempts at the same task, and judged, under a rule
written before it ran, on 62 newer tasks no model had trained on. +2.6 points
against the fine-tune it started from, 7 tasks better and 4 worse, p = 0.55:
inconclusive. Eight of the eleven tasks that moved were one trap's, and reading
them changed what the number was. Outside the return window the policy offers a
goodwill voucher; if the customer rejects it and insists on a refund, the case
goes to a human. The agents offer both at once — "the voucher, or a human agent
to review a full refund" — and my simulated customer, who wants her money back
but will take a voucher, often took the human: "I'd rather have a full refund.
Please escalate this." The agent escalated, as the policy says, and the check,
which wants a voucher, failed it. Forty-eight of the trap's 55 failures that
round were this. Read the way the policy reads them, GRPO's gain is zero — and
the base model passes 96% of the newer tasks, above the 95% at which I had said
I would hold GRPO back. Half the room I ran it for was the customer's choice.
And measured afterwards on the tasks it had trained on, it showed no gain there
either: 91.2% to 92.0%, p = 0.71. One update that small moved nothing I could
measure. On a hard edition built to give it room, where the base model passes
86%, it was inconclusive again.
<!-- RUNBOOK phase 4, GRPO, "What it found"; WHAT_FAILED #46; RESULTS.md §6: P2-rft-e2 0.919 -> P2-grpo-i1-e2 0.945, 7 better 4 worse, p=0.549; read as P6.3/P10 read them +0.000, 3 better 2 worse; P2-base-e2 0.923 -> 0.961; voucher-trap failures in round 2's seven runs 55, escalations after a declined voucher 48; RUNBOOK phase 4, the learning curve's second point: logs/r2-grpo-measure-i2.txt, P2-grpo-i1-collect -> P2-grpo-i2-collect 0.912 -> 0.920, 16 better 13 worse, p=0.711; RUNBOOK phase 4, the hard edition: P2-base-e3 0.861, P2-rft-e3 0.939 -> P2-grpo-i1-e3 0.889, 2 better 5 worse, p=0.453 -->

A verifier is a reward function the moment you train on it, and its quirks
become the model's habits. The replay that caught this cost seconds. Without
it, the headline would have been a fine-tune that "improved" and an API model
ranked above the one it trails.

### Result 5: a second look after the customer leaves

That escalation is the one agents let go in any language. The customer chases
a refund, the agent finds it was already paid and says so, the customer says
thanks, and the conversation ends with nothing filed — though the policy calls
a duplicate claim a fraud or system-error signal that must be recorded.

So when the customer ends the conversation, the agent now gets one more turn
she never sees: check the case against the policy and make any call it
requires. Under a rule written before the run, on the tasks that suggested it,
pass^1 went from 0.85 to 0.96 (11 tasks better, 2 worse, p = 0.02). On 62 new
tasks, with a simulated customer who waits for what she agreed to, it went from
0.83 to 0.94 (15 better, 5 worse, p = 0.04), and 21 of its 25 rescues were that
escalation.
<!-- RUNBOOK 1j (traces/M-off, M-check), 1m (Q-off, Q-check): compare_cells.py, closing.review as left vs at end -->

The cost was the interesting part. In 22 of 186 conversations the agent
escalated a case it had already escalated — something the conversations alone
did twice in 372. Nearly every second ticket was its first move after the note,
made with the first one in plain view. So I showed it the record of its own
calls and told it not to repeat one: 26 repeats in 186 conversations became 24.
Then I stopped asking. With the note unchanged, a write that repeats one already
made is not run after the customer leaves: the 20 second tickets became none,
and all 24 rescues stayed.
<!-- RUNBOOK 1m (22/186 vs 2/372), 1n (R-check 26 vs R-record 24 of 186, p=0.66; 90 of 91 second calls the first move after the note, v41 count), 1o (S-check 20 repeats vs S-guard 0, rescues 24 and 24) -->

The first version of that refusal compared a payment without its amount. On the
rest of the suite it refused the shipping credit the policy asks for after a
credit for the item — a correction it read as a repeat. With the amount in what
it compares, it broke no case on the other 158 tasks, both of its refusals were
the same call again, and the correction went through.
<!-- RUNBOOK 1p (T-guard: 3 shipping credits refused, WHAT_FAILED #39), 1q (U-guard, v41: 456 closing phases, none broken, 2 refusals DRE-PH__r2 and LCOW-MY.ms__r0, CCRTOM-ID.id__r2's IDR 37,000 credit through) -->

A note in the prompt found what the agent let go. Only the harness could stop it
doing it twice.

### What failed

`docs/WHAT_FAILED.md` has 51 entries. Most were found by running experiments
and refusing a number that couldn't be right. Four more:

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

**The tool took what its own schema refused.** The escalation tool lists
eight categories and takes an order id, and it checked neither:
`duplicate_refund_claim` and order "unknown" went into the database, the agent
was told it had escalated, and the check failed it. The same escalation passed
the one hand-written identity task, whose check doesn't ask which order, all 30
times, and failed the generated ones, whose check does, 147 times in 148.
Bounded, it moves no conclusion: the naming effect can only grow, and the API
reference's deficit to the base model shrinks from 6.6 points to at least 4.7.
The tools now check their inputs, and every run replays under the checks it
ran under — re-scored, all 10,684 recorded episodes come out as before.
<!-- WHAT_FAILED #35; bounds: K named 0.801 -> at most 0.871, control 0.439 -> at most 0.474; P-ref 0.913 -> at most 0.932 against 0.979, p=0.0115; I-tools2 and the fine-tune unchanged; C-clean/D-nozh largest paired gap 6.2 points, every p >= 0.25; rescore v25 vs v27 over 10,684 episodes: 0 differences -->

**The customer sent photos she didn't have.** The simulated customer used to
say "yes, I'll take the voucher" and leave before the agent could issue it, so
I told her to wait for what she'd agreed to. On the photo trap she is told she
has no photos and will send them tonight — and now that she couldn't leave,
tonight arrived inside the conversation: "(Attached: photo_of_damage.jpg) Here
are the photos." The agent opened the return. She did it in 64 of 210
conversations in the second round, every one of them failed, and the trap's
score became a measure of her. One trap's fix was another trap's bug, and
nothing checked that her persona still held; now the audit does.
<!-- WHAT_FAILED #43: claims 26/70 base, 32/70 RFT, 6/70 deepseek, all failed; round 1: 1 in 210 -->

### Limits

Each tool arm is 32 tasks, so these tests see large effects only; several
verdicts are honestly inconclusive. The language comparison has 13–16 twin
pairs per language: it rules out large gaps, not small ones. One agent model
and one judge model for the first three results, one base model for the
fourth. The false-claim check reads English only, and only the phrasings I
wrote — the same kind of instrument as the leak detector, so its counts are a
floor: in the re-run, the one failure it couldn't read told the customer, in
Vietnamese, that the case had gone to the complaints team. It hadn't. The
guardrail shares that blind spot: three guarded failures told the customer
something in words it doesn't read — a voucher promised and never issued, a
review implied with no escalation, "your case has been flagged" — and its live
test is one run of 192 conversations. The Thai and Vietnamese translations are
not native-reviewed. The fine-tune is one recipe run twice; the second, with
training data balanced across traps, moved the leak rather than stopping it,
and its photo-trap numbers are the simulated customer's. GRPO ran one
iteration, judged on tasks whose room was partly the customer's choice. Serving
cost is built but not measured.
<!-- RESULTS.md §3 paired table (pairs 13–16); audit_tool_arms.py §3 header; traces/J-preload OOWDE-VN.vi__r2: "Trường hợp của bạn đã được chuyển lên bộ phận xử lý khiếu nại", escalate_to_human never searched for, never called; traces/L2-claims OOWOV-VN__r1, DRE-PH__r1, HPRR-PH__r2 (RUNBOOK 1i, v25) -->

The code, the trace generators, and all 51 failures are in the repo.

---

# Part 2 — Resume lines

One line each, a number, and the mechanism. Pick three.

> **Built PasarBench**, a verifiable customer-service agent environment for SEA
> e-commerce — 215 tasks, 6 markets, 8 language varieties, 20 tools — scored on
> end-state database, with locale twins sharing byte-identical pass criteria.

> **Caught a fine-tune learning my verifier instead of my policy**: RFT on
> Qwen3.8-27B gained 2.3 points (p = 0.18) only because two checks demanded one
> lookup tool the policy didn't require; re-scoring 9,766 recorded episodes by
> deterministic replay — no GPU — cut the gain to 0.3 (p = 0.84) and put the
> API reference at least 4.7 points below the base model (6.6 as measured,
> p = 0.004), where the quirk had put it 4 points above.

> **Showed an LLM judge cannot stand in for state verification**: its best
> configuration reached κ = 0.72 against the database and still accepted 6 of 15
> failed episodes, crediting actions the tool log showed never happened. Removed
> the missing-evidence confound with deterministic tool-result replay.

> **Diagnosed search-based tool exposure**: replayed every query to show
> the agent never searched for tools its policy described but didn't name (17
> of 21 missing-tool cases); naming them, confirmed on 57 new tasks under a
> rule written before the run, lifted pass^1 from 0.44 to 0.80 (38 tasks
> better, 4 worse, p < 10⁻⁶) and took false claims from 12 of 42 readable
> failures to none.

> **Built a runtime check against false action claims**: a reply claiming a
> write no tool call had made is held back, and the agent is told which call
> would make it true. The first version passed only by its own reading; graded
> by an audit it doesn't share, under a rule written before the run, the second
> took false claims reaching customers from 5 of 15 readable failures to none
> against a same-day control, and all four replies it stopped ended in the real
> call and a passed case.

> **Traced a measured simulator bias to my own detector**: an 11–22% "leak"
> rate concentrated in four languages, and three runs of fixes, came from
> ask-patterns that couldn't read the agent's questions; corrected, 3 of 4,155
> episodes are flagged, all English. The same blind spot had produced the
> project's only significant language gaps.

> **Built a check that runs after the customer leaves**: the agent gets one hidden
> turn to test the case against the policy. Under rules written before each run
> it raised pass^1 from 0.85 to 0.96, and from 0.83 to 0.94 on 62 new tasks
> (p = 0.04), mostly escalations the policy requires after a satisfied customer;
> its cost, a case escalated twice, went from 20 to none once the harness refused
> repeats, with no case broken on 158 other tasks.

> **Found a language effect a suite-wide null hid**: no language differed from
> English overall, but deepseek-v4-pro made one counter-intuitive escalation in
> 59% of English conversations and 2–8% of Malay and Chinese ones; all 14
> local-language tasks fell below their English twin (p = 0.0001), and the
> decision came the moment it read the refund. Held fixed there and only
> translated, the same conversation got it more often in English, in both
> directions (47 conversations against 16, p = 0.0001).

> **Ran GRPO under a reading fixed before it ran, and found its test's room was
> the simulated customer's**: one iteration on Qwen3.8-27B gained 2.6 points on
> 62 unseen tasks (p = 0.55); 48 of the moved trap's 55 failures were an
> escalation the policy asks for and the check failed, and read as the policy
> reads them the gain was zero.

> **Documented 51 defects**, most caught by refusing implausible results —
> including a "ceiling" arm that scored below baseline, a judge evaluated
> without the evidence it was judging, a fine-tune rewarded for a checker's
> quirk, a guardrail graded by its own detector, a tool that took what its
> own schema refused, a simulated customer whose "yes" ended the episode, and
> a model's quotes counted as invented for leaving out the reply's bold.

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

### Eight stories worth having ready

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

**7. Asking versus refusing.** A hidden note after the customer leaves rescued
the escalations the agent let go, and made it escalate some cases twice — its
first move after the note, with the first ticket in view. Telling it not to
repeat itself changed nothing; refusing the repeat removed it and kept every
rescue. Then the refusal refused a correction, because it compared a payment
without its amount. Lands: enforce in the harness what a prompt can only ask,
and define "the same call" the way the policy does.

**8. The null that hid a rule.** Two runs, no language beyond noise — and one
trap where the agent made the escalation in 59% of English conversations and
2% of Malaysian Chinese ones, every twin pointing the same way; held fixed at
that moment and translated, the same conversation moved with its language.
Lands: a null over an average is a null for the average; keep a table by trap
beside it.

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
the agent never looked for, because nothing told it they existed. So I named
them in the policy: on the first 32 tasks that was the direction, not proof
(p = 0.125), so I wrote down what would count and ran 57 tasks the first run
hadn't used. Pass^1 went from 0.44 to 0.80, 38 tasks better and 4 worse. It
cost 13 to 18% more tokens; at twenty tools, just show them.

**"Did the guardrail work?"** The second version did, by a rule written before
its run: false claims reaching customers went from 5 of 15 readable failures to
none of 13, and all four replies it held back were real — each time the agent
made the call and the case passed. The first version is the better story. It
held back two false claims, but three others got through as passives — "Your
voucher has been issued" — which I had told it to ignore, and it passed its
test anyway, because the test read claims the way the guardrail did. The audit,
which reads more phrasings, counted three with it and three without. A fix
graded by its own detector passes by construction. Two limits I'd say up front:
what still gets through is in other words — "your case has been flagged", a
promise never kept — about as often with the guardrail as without; and neither
version moved the pass rate, because most failures are an agent that never
acted. A guardrail can make words match actions, not cause the action.

**"So what about the agent that never acted?"** I gave it a second look. When
the customer ends the conversation, the agent gets one note she never sees:
check the case against the policy, and make any call it requires that no call
has made. On the tasks that suggested it — the agent explains the earlier
refund, the customer says thanks, and it closes, though the policy says to
escalate anyway — pass^1 went from 0.85 to 0.96, 11 tasks better and 2 worse,
under a rule I wrote before the run. On the other 158 tasks it did no harm, and
replaying every episode as the customer left it and as it ended shows it broke
no case in either run. It isn't free: it escalated cases a second time, 18
times, which no conversation did without it in 1,290, and twice tried identity
digits it had made up. By the rule I wrote before the second run it's worth
keeping as an option, 27 rescues against 23 costs. Reading them one by one made
it about even. Five "rescues" were customers who'd said "yes, go ahead, thanks"
in one message, which ended the episode before the agent could act: the note
gave it a turn the harness owed it. Two "costs" were a shipping credit the
policy asks for, and two refunds it made after a cancellation paid back a card
authorization that had never been charged. That leaves 22 against 23. What
only the review did is narrower: the 19 actions the agent had let go, 18 of
them on one trap.
So I built 62 new tasks of the traps that suggested it — new customers, products
and stories under the same rules — and ran it again with a customer who waits:
0.83 to 0.94, 15 tasks better and 5 worse, p = 0.04, under a rule written before
the run. Twenty-one of its 25 rescues were that same let-go escalation. And the
same cost: in 22 of 186 episodes it opened a second escalation for a case the
conversation had already escalated, which the conversations alone did twice in
372. It works, and it costs one case in eight a second ticket.
I tried to take the cost away by putting the record of its own calls in the
note, with "a call the record shows as done is done: do not make it again". It
made the second call anyway, 24 times in 186 against 26 without the record — 23
escalations and a goodwill voucher — every time as its very first move after the
note, with the first call listed as done right there; one agent admitted the
duplicate voucher afterwards. Across every run, 91 of 93 second calls were that
first move. You can't instruct away a call the agent makes in answer to the
instruction; what's left is to refuse it. So the harness refuses it now: after
the note, a write that repeats one already made isn't run, and the agent is told
it was already done. On the same 62 tasks that took the second tickets from 20
to none and left the rescues at 24 each; after every refusal the agent just
wrote its case note, nearly always naming the ticket that stood. That's the
version I'd keep on the traps it was built for. On the 158 tasks it wasn't, it
broke nothing and stopped the one repeat it met, but three times it refused the
shipping credit the policy asks for after a credit for a broken item, and one
agent said exactly why: "the amounts differ." It had been deciding what counts
as the same payment without looking at the amount. With the amount added, the
same 158 tasks again: the shipping credit went through, the only refusals were
the same call twice, and nothing broke. That's the version I'd turn on
everywhere.
<!-- RUNBOOK 1j, 1k; traces/M-* 2026-10-09, traces/N-* 2026-10-10; compare_cells.py (v30): 1j pass^1 0.848 -> 0.959, 11 better 2 worse, p=0.022; 1k 0.964 -> 0.951, 6 better 10 worse, p=0.454; as left -> at end 21 -> 0 and 6 -> 0 (171, 472); as left, paired: 2 better 3 worse p=1.000, 6 better 15 worse p=0.078; rescued 21 + 6 (1j: 18 DRE let go, IVF-VN__r1 let go, IVF-MY.zh-MY__r2 said yes and left, T13__r1 promise; 1k: HRWR-ID.id__r2, LCOW-SG.zh-SG__r0, PRWR-MY.zh-MY__r0, __r1 said yes and left, CCNRD-MY__r0, IVF-SG.sg-en__r1 promises); costs 17 + 6 (repeated 15 + 6, of 1k's 6 two CCRTOM shipping credits P4.6 asks for; guessed 2); repeated writes before the note 0 of 1,290 conversations ; 1m (Q-off vs Q-check, edition 2, 62 tasks, customer 2, 2026-10-11): pass^1 0.833 -> 0.935, 15 better 5 worse p=0.041; as left 6 better 12 worse p=0.238; rescued 25 (21 DRE2, 4 IVF2), all passed; repeated escalations 22 (13 IVF2, 7 OOWDE2, 2 CHE2); forbidden 0, guessed 0; unscored 2 (OOWOV2-ID.id__r0, OOWOV2-MY.ms__r2: yes to a voucher before verification, failed); repeated writes before the note 2 of 372 (Q-off CHE2-MY__r0, OOWDE2-MY.zh-MY__r0); DRE2 0.38 -> 0.82 ; 1n (R-check vs R-record, 2026-10-12): pass^1 0.973 -> 0.962, 2 better 5 worse p=0.453; repeats 26 -> 24, 9 tasks more 12 fewer p=0.664, all 24 at the first closing step with the first call in the record as done; at the first step, all runs: M 15/15, N 5/6, Q 22/22, R 25/26 and 24/24; rescues 27 -> 21 p=0.180; second voucher OOWOV2-SG.zh-SG__r1 ; 1o (S-check vs S-guard, 2026-10-12): pass^1 0.973 -> 0.946, 2 better 5 worse p=0.453 (as left the same -0.027, 5 better 10 worse p=0.302); costs 20 -> 0, 0 more 15 fewer p<0.001; refused 23, all at the first step, 22 notes name ESC0001; substitutes 0; rescues 24 -> 24 p=1.000; control repeats incl. 2 vouchers; dry run S-check 20/20 ; 1p (T-guard, 158 tasks, 474 episodes, 2026-10-12): pass^1 0.970; 465 closing phases, 0 passes broken, 4 rescues, 0 forbidden/guessed/substituted; refused 4 calls in 3 episodes: LCOW-TH__r2 same message again, CCRTOM-ID.id__r0 and CCRTOM-ID__r1 (x2) the IDR 37,000 shipping credit after 700,000; with amount and currency in a payment's target the dry run changes only N-check 6 -> 4 ; 1q (U-guard, guard v2, 158 tasks, 2026-10-12): pass^1 0.973; 456 phases, 0 broken, 1 rescue (T14__r1); refused 2, both the same call (DRE-PH__r2, LCOW-MY.ms__r0); another amount 1, CCRTOM-ID.id__r2 IDR 37,000 after 700,000; 27 cut short by max_tokens, 3 failing as left -->

**"What did the second look find about the benchmark?"** Two holes. Cancelling a
card order, the agent told the customer her money would go back — 42 times in 42
in one control, in every language — and nothing in the world did it: no rule, no
tool, no check says what happens to that money, so every one passed. The closing
review kept the promise three times, and the checks couldn't say whether that
was right. And the simulated customer often said yes and goodbye in one message;
the episode ended on it before the agent read it, which is more than half of one
trap's failures across every run. Counted at their worst, neither reverses a
result I report — one would cross p < 0.05. Both are changed now: a cancellation
settles the payment, as real platforms do, and the simulated customer is told to
wait until what she agreed to is done. A check run says she does — none of 252
conversations ended on her yes, against 7 for the old customer the same day, and
the voucher trap went from 0.81 to 1.00 — and every cancellation settled as the
tool told the agent. Every recorded run replays under the version it ran with
and re-scores exactly as before.
<!-- WHAT_FAILED #37: N-off cancel_while_processing 42/42 told of a refund, 0 refunded, 42 passed; all runs 690 cancelled, 10 refunded (P-base 7, N-check 3). #38: 63 failures in 11,329 episodes without a closing phase, 57 in out_of_window_offer_voucher (of its 106 failures); upper bound, all 63 passing: naming 0.474 -> 0.825 (39 better, 3 worse), RFT +0.1, reference -6.6, largest language gap 6.2 (C-clean th), p >= 0.25; I-tools2 search vs all-20 -12.5 p=0.039 ; 1l (O-c1 vs O-c2, 2026-10-11): pass^1 0.893 -> 0.917, 14 better 8 worse p=0.286; yes-endings 7 vs 0 of 252; vouchers 34/42 -> 42/42; cancellations 39 refunded + 3 released per arm, 0 refused as settled -->

**"How do you know your false-claim detector isn't wrong the way the leak
detector was?"** I don't, fully — it's the same kind of instrument. So it's
scoped to English, the report says other languages are counted, not read, and
I read each of the three against the tool log. What it can't tell me is how
many claims it missed, so three is a floor. For the guardrail's run I read
every failure by hand to find out: it had missed some, in words like "your case
has been flagged", and the docs report them beside its counts.

**"How do you know the language null isn't just low power?"** It is partly low
power, and I'd say so: 13–16 twin pairs per language rule out large gaps, not
small ones. What makes it believable is that it held across two independent
runs, and that the one "significant" gap traced to a bug. And it is the suite's
null, not every rule's: on the duplicate refund the gap is 38 points, and the
pooled test averages it away (Result 3).

**"Is 3 points of run-to-run noise normal?"** It's what I measured for this
agent and simulator at 32 tasks. It's why every comparison in the report is a
paired test, and why several verdicts say INCONCLUSIVE.

**"So did RFT work?"** Not measurably, once the checker was fixed: +0.3 points,
p = 0.84. What it did do is move behaviour between traps — better at escalating
duplicate refunds, worse at out-of-window disputes, where it copied the
livestream trap's answer. So the second run built its data with the corrected
checks and balanced it by trap, and I read the per-trap table as the result:
the same null, and the leak moved — the photo trap, newly well represented,
lent its rule to a cash-on-delivery refund. GRPO couldn't show much on the
held-out tasks — at 0.96 they are nearly saturated for this model — so it was
judged on the 62 newer tasks no model trained on, where the base model passed
92%. One iteration: +2.6 points, p = 0.55, inconclusive. And half of that 8%
of room was one trap's customer choosing a human over a voucher, an escalation
the policy asks for and the check failed; read as the policy reads it, the gain
is zero and the base model passes 96% (WHAT_FAILED #46). The hard edition gave
room — the base model passes 86% of it — and there GRPO was inconclusive too,
−5 points against RFT, p = 0.45.

**"Why didn't you re-run the evaluation after fixing the checker?"** I didn't
need to. The environment is deterministic, so replaying each episode's tool
calls rebuilds its final database exactly, and the fixed checks score that.
Every replay is verified against what was recorded — a call that succeeds on
replay where it failed in the run, or a result that differs, and the episode
keeps its old verdict. 8,941 of 9,766 reproduced; all but one of the rest are
a run from before an ID format change. Then I read the conversations the fix
had promoted — a replay proves the state, not that the state means what the
check assumes.

**"What would you do next?"** Make the suite harder. Strong models pass 91–98%
of it, and their failures sit in a handful of traps, so a table of frontier
models would bunch at the top. The hard edition is built (v48): 36 cases that
combine traps — a duplicate refund on a cash-on-delivery livestream order, two
orders in one chat, a customer who changes her request with her phone digits —
each read against the suite's task of the rule it turns on, in the same market
and language, so "harder" is a paired test and not an impression (RUNBOOK 1s).
On round 2's models it left room — the base model passes 86% of it, against 92%
of the newer tasks — and was hard where it was built to be: two orders in one
chat, where the failing agents asked for photos of cushion covers worth about
SGD 58 as if they were worth over SGD 200, outside a return window most of them
told the customer was still open, and almost always without the one call that
would have told them both. In Singapore, where the price is in SGD, no model
failed it. Then I ran models from four labs on it, one command each
(`scripts/add_model.py`, each key sent to its own provider only), against one
customer — a customer who leaves once she has promised what she can't do in the
conversation (WHAT_FAILED #43), checked on round 2's machine: under her no claim
of photos reached the agent, where the old customer's did in 31 of 70
conversations. For the five of them the hard edition was not harder than its
parts, and it separated no two the parts did not: where they failed, they failed
the duplicate refund the parts already test, and DeepSeek-V4-Pro failed it less
often inside the livestream story than alone (RUNBOOK 1s). It was hard for
round 2's 27B models; for API models the next edition needs combinations that
defeat them. Reading those failures found two more broken instruments: every
generated customer's profile shows a phone that does not end in the digits that
verify her (WHAT_FAILED #48), and the voucher choice's reader took a refused
voucher for one she declined (#49).
<!-- RUNBOOK 1s, run on 2026-10-13; RESULTS.md §8: cases against their parts, pass^1: deepseek-v4-pro 0.944 vs 0.833 EASIER p=0.039, deepseek-flash 0.991 vs 1.000, mimo-v2.6-flash 0.991 vs 0.991, grok-4.3 0.685 vs 0.731 p=0.289, glm-5.3-flash 0.954 vs 0.944; the parts separate deepseek-v4-pro from deepseek-flash, mimo and glm (p<=0.006), the cases separate no pair the parts do not; duplicate refund twins: deepseek-v4-pro 12/30 alone, 25/30 in the livestream case -->
GRPO ran one iteration from the second fine-tune: inconclusive,
and the room it was judged in was partly the voucher trap's customer (RUNBOOK
phase 4, GRPO; WHAT_FAILED #46); on the hard edition, inconclusive again.
Measured on the tasks it trained on, the one iteration showed no gain there
either (0.912 → 0.920, p = 0.71): not an update that learned its tasks and
didn't carry them over, but one too small to move anything, so the next needs
more signal before another held-out test. Along the way, the three that were
waiting are done — two runs and a calibration. The language: held fixed at the
moment the agent reads the refund and only translated, the same conversation
got the escalation more often in English, both ways (RUNBOOK 1t, above). Room
for the closing phase: given its own, no phase was cut short — the episode's
budget had cut 27 of 456 — and no rescue turned up that the budget had hidden
(RUNBOOK 1q). And a model that reads what the
customer was told in place of the guardrail's phrase list, the check against
the tool log kept mechanical: calibrated on 626 recorded replies, it held back
in English every claim the list did and nothing more, but outside English it
took the refund paid before the conversation for a payment the agent made —
four times, in four languages — so it is not used, and the list stays (RUNBOOK
1u). Its calibration found one more broken instrument, its own quote check,
which counted the reply's bold as invention (WHAT_FAILED #50). The same swap on
three other agents could not test the language effect: two of them escalate a
refund already paid in nearly every run, in every language, and the third,
GLM, decides it differently from one run to the next, so holding the
conversation fixed did not hold its decision fixed (RUNBOOK 1w; Grok, which
escalated in none of its conversations, was left out). Next: the same
conversations under a policy in her language (RUNBOOK 1v), whose translation,
read before the run, had a section in Malay that its check had passed
(WHAT_FAILED #51). Then serving cost.
<!-- RUNBOOK 1q, U-room, run on 2026-10-13: closing phases cut 0 of 469 (U-guard 27 of 456); of U-guard's 27, 25 ran, none wrote; rescues 3 (CCRTOM-MY.ms__r1, T14__r1, T14__r2), none a phase U-guard had cut; passed->failed 0, forbidden 0, refusals 4 each the same call again, another amount 1 (CCRTOM-ID.id__r1 IDR 37,000 after 700,000); pass^1 0.970 vs 0.973, 4 better 5 worse p=1.0. RUNBOOK 1u, C.txt: 626 replies, 261 kept, 47 dropped (15.3%) under quote check 1; refund already paid 4 alarms of 167, all the refund paid before, all outside English; English both 132, list only 4 (offers), reader only 0; outside English 25 alarms: 19 English sentences the list holds too, 1 Indonesian claim, 5 false alarms; under quote check 2, 2 dropped (0.6%) and 5 refund alarms -->

### One thing not to do

Don't present a co-movement as a cause, and don't round INCONCLUSIVE up to a
finding. Several of the most useful results here are nulls; say them as nulls.
