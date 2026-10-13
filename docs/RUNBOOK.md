# Runbook

Ordered by what you can do **right now**, not by chapter number.

## The headline: most of the remaining work needs no GPU

| phase | needs | cost | blocked by GPU queue? |
|---|---|---|---|
| **0. Verify** | laptop | free | no |
| **1. Context + tool + multilingual sweeps** | an API key | ~$15–40 | **no** |
| **2. Judge calibration** | your own eyes, ~6 hours | free | **no** |
| **3. Serving measurements** | the Phase 4 node | an hour or two, optional | yes |
| **4. Post-training** | 1× B200 (or H200, or 2× H100), rented | ~10–20 hours of GPU time | yes |

Phases 1 and 2 produce **three of the five results** in `RESULTS.md`, including
the multilingual diagnosis — the one to lead the writeup with. Phase 2 needs no
compute at all and is the highest-value week in the project. Do not sit idle
waiting for a node.

---

# Updating an already-pushed repo

A new version arrives as an archive of code and docs — no traces, data, logs,
`.env` or RESULTS.md — so unpack it over your working copy, where `.git` and
your runs already are, and commit what changed. Commit one file at a time and
each file on GitHub shows its own message, rather than one message for all:

```bash
cd ~/Downloads && tar -xf pasarbench-MMDD-vNN.tar   # in the folder that holds pasarbench/
cd pasarbench
bash run_tests.sh                     # never commit a red tree
git status                            # what the new version changed
git add -A                            # .gitignore keeps .env, traces/ and data/ out
git commit -m "What changed in this file" -- path/to/file      # once per file
python scripts/make_report.py --out RESULTS.md --tools-run I-tools2 \
  --multilingual-run C-clean,D-nozh,G-gated --noise-pair H-context/full,I-tools2/full+all-20
git commit -m "Regenerate RESULTS.md" -- RESULTS.md
git status                            # nothing left to commit
git push
```

`git commit -- <file>` commits that file alone and leaves the rest staged for
the next commit; list several files after the `--` to give them one message.
GitHub shows each file with the message of the last commit that changed it.
Unpacking never deletes: a file a new version drops stays on disk, and in git,
until you `git rm` it.

Once per folder, check that git will never upload your keys:

```bash
git check-ignore .env      # prints .env when it is ignored; nothing means it is NOT
```

**Only if the histories have diverged** — `git push` is rejected and you want
GitHub to match this folder exactly — replace GitHub's history with one commit:

```bash
./scripts/push_to_github.sh git@github.com:<you>/pasarbench.git
```

It refuses to start unless `.gitignore` keeps `.env`, `traces/`, `data/` and
`checkpoints/` out; runs the full suite and refuses to push a broken tree; asks
you to type `REPLACE`; then builds one commit authored by you, checks that it
holds no `.env` file and no run artifact, and force-pushes it. It deletes
nothing in the folder, and the history it replaced stays in `git reflog main`.
Before v23 it deleted `traces/`, `data/`, `logs/` and `checkpoints/` from the
folder it ran in: never run an older copy in your working folder.

Then confirm the update landed:

```bash
git log origin/main --oneline -5      # GitHub's copy has your commits
python -c "import pasarbench.sweep as m; print(m.__file__)"
```

That second command is the one that catches the real problem: if you have two
extractions on disk, Python may still be importing the old one.

**HTTPS vs SSH.** GitHub removed password authentication, so an `https://` URL
needs a personal access token rather than your account password. `git@github...`
with an SSH key is less friction if you already have one set up.

---

# Phase 0 — verify (10 minutes, laptop)

```bash
git clone <your-repo> && cd pasarbench
./run_tests.sh
```

Expect: reference 1.0, null 0.0, every suite green, and a scripted
end-to-end sweep. If anything fails here, stop — everything downstream inherits it.

### `ModuleNotFoundError: No module named 'pasarbench'`

Almost always a working-directory problem, made easy to hit by the nesting: the
repo root is called `pasarbench` and it *contains* a package also called
`pasarbench`.

```
pasarbench/            <- run from HERE (has README.md, LICENSE)
├── pasarbench/        <- NOT here (has db.py, policy.md)
├── tests/
└── scripts/
```

```bash
pwd && ls               # see README.md? correct. see db.py? cd ..
```

`./run_tests.sh` resolves the repo root from its own location and works from
anywhere, so use it rather than remembering where you are. Two other causes
worth ruling out: Python older than 3.10 (`python3 --version`), and `python`
pointing at Python 2 or a Windows Store stub — use `python3`, or set
`PYTHON=/path/to/python ./run_tests.sh`.

---

# Phase 1 — the sweeps (no GPU, ~$15–40 of API credit)

Any OpenAI-compatible endpoint works. Use a **cheap** model for the user
simulator, and a **different family** from the agent — a simulator sharing the
agent's weights is unusually easy for that agent to satisfy and quietly
inflates every score.

### 1a. Smoke test on 5 tasks first — always

```bash
export OPENAI_API_KEY=sk-...

python -m pasarbench.sweep \
  --backend openai --model gpt-4o-mini \
  --simulator openai --sim-model gpt-4o-mini \
  --suite core --tasks T01,T02,T08,T13,T16 \
  --strategies full -k 1 --run-id smoke
```

#### Supplying your API key

```bash
export DEEPSEEK_API_KEY=sk-...        # then pass NO --api-key flag
```

Each key is found by the endpoint it goes to (v55; WHAT_FAILED #47):
`PASARBENCH_API_KEY` (this run's agent key) or `PASARBENCH_SIM_API_KEY` (the
customer's, which the claim reader and the translator share) when set;
otherwise the variable of the provider that endpoint is on — `DEEPSEEK_API_KEY`
for api.deepseek.com, `DASHSCOPE_API_KEY` for DashScope, `OPENAI_API_KEY` for
api.openai.com, and `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`,
`XAI_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `GROQ_API_KEY` for
theirs. Never another provider's: with only `DEEPSEEK_API_KEY` set, an agent on
api.openai.com gets no key and the sweep stops, naming the variable to set,
rather than sending a request and handing you a 401. A local server gets none.

**Do not pass `--api-key sk-...` on a shared cluster.** Command-line arguments
land in `/proc/<pid>/cmdline`, which is world-readable — any other user on the
node can read your key out of `ps aux`. Environment variables live in
`/proc/<pid>/environ`, readable only by you and root.

Three ways to set it, safest first:

```bash
# 1. A .env file, already in .gitignore. Best for repeated runs.
nano .env                  # one NAME=value per line, no spaces, no quotes;
                           # Ctrl+O then Return saves, Ctrl+X quits
set -a; source .env; set +a

# 2. Interactive, never echoed and never in shell history.
read -rs -p "DeepSeek key: " DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY   # bash
read -rs "DEEPSEEK_API_KEY?DeepSeek key: " && export DEEPSEEK_API_KEY      # zsh (macOS)

# 3. Plain export with a LEADING SPACE, which keeps it out of ~/.bash_history
#    when HISTCONTROL includes ignorespace (bash default on most distros).
#    zsh, the macOS default, does this only after `setopt HIST_IGNORE_SPACE`.
 export DEEPSEEK_API_KEY=sk-...
```

For the runs below, `.env` holds two lines: `DEEPSEEK_API_KEY=...` for the
agent and `DASHSCOPE_API_KEY=...` for the simulated customer. Edit it in an
editor rather than with `echo 'KEY=...' > .env`, which leaves the key in your
shell history and replaces the whole file.

Check it took without printing it:

```bash
cut -d= -f1 .env                      # the names in the file, never the keys
echo "${#DEEPSEEK_API_KEY} chars"     # ~35 for a DeepSeek key
```

If you ever paste a key into a terminal on a shared machine, or commit one,
rotate it. It is thirty seconds of work and the alternative is someone else
spending your credits.

#### Provider notes

**DeepSeek** — `./scripts/smoke_deepseek.sh` (set `DEEPSEEK_API_KEY`). Model is
**`deepseek-flash`**, endpoint `https://api.deepseek.com/v1`.

> **Pin the canonical name, never a retired alias.** DeepSeek released V4.1
> Flash on 2026-09-10 and made `deepseek-flash` the preferred identifier;
> `deepseek-v4-flash` is retired and only *temporarily* routed to V4.1 for
> compatibility. Ask for a retired alias and the provider silently serves you
> something else — which means a sweep run in August and one run in September
> under the same string compare two different models, with nothing in the
> output saying so.
>
> The harness now reads the `model` field the provider returns, records it in
> every trace header as `served_model`, and prints a one-time warning when it
> differs from what you asked for. **Report the served name and the date in
> your writeup**, not the string you typed.

> **V4 runs with thinking ENABLED by default**, and reasoning tokens count
> toward `max_tokens`. Left on, it can exhaust the budget before a tool call is
> emitted (empty assistant turns), and it corrupts two measurements this project
> depends on: the token axis of the context ablation becomes reasoning
> verbosity, and tokens-per-char in the multilingual diagnosis becomes
> unreadable. Always pass:
>
> ```bash
> --extra-body '{"thinking":{"type":"disabled"}}'
> ```
>
> Re-enable it later as a deliberate, separately-labelled arm if you want to
> measure what thinking buys here. Running with it on by accident is the
> problem, not thinking itself.

DeepSeek also reports `prompt_cache_hit_tokens` in `usage`, which the backend
now records as `Usage.cached_tokens`. The ~2.2k-token policy prefix is identical
on every call, so **you get a prefix-caching measurement from the API with no
GPU** — part of Phase 3, early and free.

**Anthropic** — `--backend anthropic --model <id>`, key in `ANTHROPIC_API_KEY`.

**Any OpenAI-compatible endpoint** — `--base-url` plus `--api-key`. No code
change is needed; `OpenAICompatBackend` handles the wire format and
`--extra-body` carries anything provider-specific.

Read the traces before spending anything more. You are checking three things:
the agent actually calls tools, the simulator withholds `hidden_facts` until
asked, and episodes terminate rather than burning the step budget.

### 1b. Inspect the smoke run before spending anything more

```bash
python scripts/inspect_trace.py traces/smoke-deepseek/full
python scripts/inspect_trace.py traces/smoke-deepseek/full --task T08
```

Answers all four smoke-test questions plus the two that matter after a real
run: exactly which check each failing task broke, and what the full sweep will
cost at the token rate you just measured.

### 1c. Audit the simulator before trusting any number

```python
from pasarbench.diagnose import load_episodes
from pasarbench.simqa import leak_report, audit, VERDICT
from pasarbench.generate import generate
from pasarbench.harness.trace import read_episode

tasks = {t.task_id: t for t in generate()[0]}
# build LeakReports from your traces, then:
print(audit(reports)); print(VERDICT)
```

`leak_rate > 0.15` means the persona prompt is not holding and your pass rates
describe an easier benchmark than the one you wrote up — **or** that the
detector cannot read how the agent asks in that language. `python
scripts/inspect_trace.py <trace_dir>` prints the agent's message before every
flagged leak; if it asks for the fact, extend `simqa.ASK_PATTERNS`, not the
persona prompt (WHAT_FAILED #26). Settle which it is here, not later.

### 1d. Context ablation — subsample first

```bash
python -m pasarbench.sweep \
  --backend openai --model gpt-4o-mini \
  --simulator openai --sim-model gpt-4o-mini \
  --suite all --sample 2 \
  --strategies full,window8,window4,trim3,notes4 \
  --summarizer-model gpt-4o-mini \
  -k 3 --run-id ctx-pilot
```

`--sample 2` is 32 tasks covering all 16 traps. Only after the pilot looks sane:
drop `--sample`, raise `-k 5`, and run the full 186.

**Cost arithmetic before you run it.** Episodes are ~9k tokens each. Full suite
× 5 strategies × k=5 = 4,650 episodes ≈ 42M tokens, plus a simulator call per
user turn. On a cheap model that is tens of dollars; on a frontier model it is
hundreds. Check the number before pressing enter.

### 1e. Tool scaling — include the pair or the result is unreadable

```bash
python -m pasarbench.sweep \
  --backend openai --model gpt-4o-mini --simulator openai --sim-model gpt-4o-mini \
  --suite all --sample 2 --strategies full \
  --exposure oracle,all-20,all-100,random-100,all-300,random-300,search-300 \
  -k 3 --run-id toolscale
```

`all-N` **and** `random-N` must both be present. The sweep warns you if they are
not: without the pair you cannot tell token cost from selection difficulty.

### 1f. Multilingual — the run that produces the headline

```bash
python -m pasarbench.sweep \
  --backend openai --model gpt-4o-mini --simulator openai --sim-model gpt-4o-mini \
  --suite all --strategies full -k 3 --run-id multiling
```

Then:

```python
from pasarbench.diagnose import load_episodes, report
print(report(load_episodes("traces/multiling/full")))
```

**Read the budget-exhaustion column first.** If non-English episodes hit
`max_steps` more often than English ones, raise `--max-steps` and re-run before
concluding anything. Every other reading is invalid until that is ruled out.

### 1g. Does naming the tools fix search? (API only, no GPU)

Behind `search_tools` the agent never looked for three tools — escalation,
shipment lookup, goodwill voucher — and the policy describes those actions
without naming a tool; the tools it does name, it found (WRITEUP, Result 1).
That is a hypothesis until it is tested. `--policy-mode preload-named` gives
the agent the same policy with those three names added where each action is
described, and nothing else (a test strips them and gets the policy back byte
for byte). Run it with a same-day control on the 32 tasks of `I-tools2`, so
that a model update since September cannot pass for the effect:

```bash
set -a; . ./.env; set +a      # your keys, as for the I-tools runs: agent and customer
TASKS=ACAD-VN,CCNRD-PH,CCNRD-TH,CCRTOM-PH,CCRTOM-TH,CCSO-VN.vi,CHE-MY,CHE-VN,CWP-ID,CWP-SG.sg-en,DRE-PH,HPRR-PH,HPRR-SG.sg-en,HRWR-MY,HRWR-SG.sg-en,HVPRF-PH,IVF-MY,IVF-SG.sg-en,LCOW-MY,LCOW-TH,OOWDE-MY,OOWDE-VN.vi,OOWOV-TH,OOWOV-VN,PPDNC-SG,PPDNC-TH,PRWR-VN.vi,T04,T07,T09,T14,T15
for MODE in preload preload-named; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
    --policy-mode $MODE --run-id J-$MODE --workers 8
done
python scripts/compare_cells.py traces/J-preload/full+search-300 traces/J-preload-named/full+search-300
python scripts/audit_tool_arms.py traces/J-preload-named
```

192 episodes, the size of two tool arms. Read `compare_cells.py`'s paired line first: at 32 tasks only a
large effect resolves. Then the line for each named tool — how often an episode
that needed it called it — and the audit's breakdown of why the remaining
failures failed. If the named arm calls the tools and closes most of the gap to
`all-20` (0.958 in `I-tools2`), naming is the mechanism. If it searches for
them and still fails, the ranker is; if it still never looks, it is neither.

**Run on 2026-10-02**, the two arms back to back within eleven minutes. The
control matched `I-tools2`'s search arm: 81 of 96, pass^k 0.75, and three
claimed-not-done failures on the same three tasks. With the names:

| | control | named |
|---|---|---|
| escalated, of the 24 episodes that need it | 12 | 21 (21 called it by name first) |
| looked up the shipment, of 18 | 3 | 12 |
| issued the voucher, of 6 | 4 | 5 |
| pass^1 (pass^k) | 0.844 (0.750) | 0.938 (0.875) |
| tokens per episode | 44,187 | 52,354 |
| failures claiming an action never taken (English) | 3 of 14 | 0 of 6 |

Paired by task, 6 better and 1 worse, p = 0.125: INCONCLUSIVE at 32 tasks.
Tokens +18%, 25 tasks costlier and 7 cheaper, p = 0.002. The named arm's 88
calls to tools it had not been shown were all refused, and 87 ended with the
tool found by search and used. By the reading above the agent now calls the
tools and closes most of the gap, so naming is the mechanism, with the pass
rate itself still to be confirmed on more tasks. Of its six failures, three
are T14 never escalating a duplicate refund — no search at all, a decision
rather than a missing tool. WRITEUP, Result 1 has the write-up.

### 1h. Confirm the naming effect on new tasks (API only, no GPU)

1g moved the pass rate the way the explanation predicts — 6 tasks better, 1
worse — but at 32 tasks that is p = 0.125. This asks the same question of the
57 tasks the first run did not include that cannot pass without one of the
three named tools; `scripts/naming_tasks.py` lists them. A task a `get_order`
read also passes (an address change after dispatch) is left out: naming
predicts nothing there. The 32 stay out too: they produced the hypothesis and
its first test, so they cannot also confirm it. In 1g every task that moved was
one of this kind — 6 better and 1 worse of its 10.

**The reading, fixed before the run:**

- **Confirmed** if `compare_cells.py`'s paired line says BETTER: the named
  policy wins the sign test over these 57 tasks at p < 0.05. INCONCLUSIVE or
  WORSE is reported as it comes, and the blog's sentence on naming changes to
  say so.
- The mechanism is the tool lines: the named arm should call
  `escalate_to_human` and `get_shipment` in more of the episodes that need
  them. A pass-rate gain without that is not this effect.
- Tokens are reported either way. The verdict is on the 57 alone; a number
  pooled with the first run may sit beside it, labelled as pooled.

```bash
set -a; . ./.env; set +a
TASKS=$(python scripts/naming_tasks.py)
python scripts/naming_tasks.py --count          # 57
for MODE in preload preload-named; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
    --policy-mode $MODE --run-id K-$MODE --workers 8
done
python scripts/compare_cells.py traces/K-preload/full+search-300 traces/K-preload-named/full+search-300
python scripts/audit_tool_arms.py traces/K-preload-named
```

342 episodes, under twice 1g, the two arms back to back.

**Run on 2026-10-07** (`K-preload`, `K-preload-named`, back to back): confirmed.

| | control | named |
|---|---|---|
| pass^1 (pass^k) | 0.439 (0.228) | 0.801 (0.667) |
| escalated, of the 135 episodes that need it | 63 | 117 (110 called it by name first) |
| looked up the shipment, of 27 | 20 | 27 |
| issued the voucher, of 36 | 18 | 32 |
| tokens per episode | 40,019 | 45,224 |
| failures claiming an action never taken (audit, English) | 12 of 42 | 0 of 13 |

Paired by task, 38 better and 4 worse, p = 6 × 10⁻⁸: BETTER. Tokens +13%, 39
tasks costlier and 18 cheaper, p = 0.008. Every language moved the same way
(English 42 → 67 of 78, Indonesian 2 → 9 of 15, Thai 8 → 14 of 15, ...). Of the
named arm's 34 failures, 16 are duplicate refunds closed without the escalation
P10 asks for even after the refund is explained, 5 are duplicate refunds
escalated under `duplicate_refund_claim`, a category the tool's schema does not
offer, 7 are identity failures escalated with no order id (`"unknown"` or
blank), 2 are identity failures never escalated, and 4 are vouchers never
issued. Twenty-two are the agent deciding not to act and twelve get the
action's details wrong; none is a tool it looked for and could not find. All
twelve are inputs the tool should have refused — a category its schema does
not list, an order called "unknown" — and since v27 it does (WHAT_FAILED #35).
Counted as fixed once refused, the named arm would be at most 0.871 and the
control, with 6 of its own, 0.474: the effect can only be larger. Runs before
v27 replay under the checks they ran under, so these numbers stand as
measured.

### 1i. Does a claim check stop the false claims? (API only, no GPU)

Behind search the agent told customers "I've escalated your case" with no
escalation in the database, and a judge believed it (WRITEUP, Results 1 and 2).
`--guardrail claims` reads each reply before it reaches the customer. A reply
that claims a write action no successful call in the conversation has done is
held back — the customer never sees it — and the agent gets a note naming the
claim and the tool that would back it, asking it to make the call or reword the
reply; at most two notes per episode, so an agent that insists cannot loop. The
check reads only the reply and the conversation's own calls, never the task's
answer, so a deployed agent could run it (`pasarbench/harness/guardrail.py`).
It reads English only. It reads first-person claims for every write action,
and "we" and passives ("your voucher has been issued") only for vouchers and
escalations, the two actions no record of can predate a conversation in this
world: "your refund was issued on 9 November" is true in the duplicate-refund
trap, and read as a claim it would send the agent towards a second refund.
v24 read first-person claims only, and its run (below) is why v25 reads more.

**Before the run**, read what it would have done to the runs already recorded:

```bash
python scripts/guardrail_dry_run.py traces
```

On the 10,492 finished episodes recorded up to 1i — 6,717 in English, the only
ones it reads — v25 would fire on 46. Fourteen are in I-tools' broken oracle
arm, which hid both a lookup the task needed and the escalation tool
(WHAT_FAILED #18); 28 are behind search without the names (I-tools 5, I-tools2
3, J-preload 4, K-preload 13, L-off 3); 3 are the passives v24 let through in
L-claims; one is an "I'll escalate it now" in P-base. Forty-four are in failed
episodes. The two in passing ones are promises made before the call — "I'll
escalate it now", "I'm escalating it — would you like me to proceed?" — and
holding them back would only have moved the call earlier. In the 9,482
episodes with the tools in view it fires once, on that P-base promise, and it
never fires behind search once the tools are named. A first v25 draft fired on
"No voucher has been issued" three times; a passive now needs a subject that
points at something (WHAT_FAILED #34).

Run it on 1g's 32 tasks, with a control the same day:

```bash
set -a; . ./.env; set +a
TASKS=$(python scripts/naming_tasks.py --first-run)
for G in off claims; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
    --guardrail $G --run-id L2-$G --workers 8
done
python scripts/compare_cells.py traces/L2-off/full+search-300 traces/L2-claims/full+search-300
python scripts/audit_tool_arms.py traces/L2-claims
```

**The reading, fixed before the run:**

- **Primary: the audit's reading**, `compare_cells.py`'s line "audit reading,
  failures read" — failed English episodes whose replies claim a required
  action the database never saw. The guardrail works if the guarded arm has
  none, or none but replies it let through after two notes ("claimed again").
  Not the guardrail's own reading: graded by that, v24 passed by construction
  (WHAT_FAILED #34).
- **Read every held reply** it lists. One that claims nothing, or claims what
  happened before the conversation, is a false alarm; count them. A false alarm
  that leads to a wrong action — a guarded episode failing where the control
  passed, on a forbidden write — is the guardrail's cost, and is reported.
- **Secondary:** the paired pass rate and tokens. The note names the tool, so
  part of any pass-rate gain is naming (1g).

192 episodes, the size of 1g. Run ids `L2-*`: `L-*` is v24's run, below.

**Run on 2026-10-07, v24** (`L-off`, `L-claims`, back to back), first-person
claims only. It held back two replies, both real — "I've escalated your case
for review under our out-of-window dispute process", "I'll issue the goodwill
voucher now" — and both times the agent made the call and the episode passed.
By the rule written for v24, the guardrail's own reading, it passed: 2 of 96
episodes with a false claim reaching the customer, then 0. By the audit's
reading it did not: 3 of 15 readable failures in the control, 3 of 18 guarded,
all three of those reading "Your voucher has been issued" — a passive v24 did
not read. Read by v25, each arm has three. Pass^1 0.812 against 0.802, 2 tasks
better and 3 worse, p = 1.0; tokens +4%, p = 0.86. One guarded episode stopped
on the token budget, one the guardrail never touched. v25 is the fix, and the
run above is its test.

**Run on 2026-10-08, v25** (`L2-off`, `L2-claims`, back to back): it works, by
the reading fixed before the run.

| | control | guarded |
|---|---|---|
| failures claiming a required action never taken (audit, English) | 5 of 15 | 0 of 13 |
| episodes with a reply claiming a write no call had done (guardrail's reading) | 6 of 96 | 0 of 96 |
| replies held back | | 4: 4 calls made, 4 episodes passed |
| pass^1 (pass^k) | 0.823 (0.781) | 0.865 (0.781) |
| tokens per episode | 45,063 | 43,499 |

The four held replies are all real claims: "I'll escalate this now" (DRE-PH__r2),
"I've escalated your case with the category `out_of_window_dispute`"
(OOWDE-MY__r0), "Your goodwill voucher of VND 220,000 has been issued to your
account" (OOWOV-VN__r0) and "Your voucher has been arranged" (OOWOV-VN__r2) —
the last two passives only v25 reads. Each time the agent searched for the
tool, made the call, and the episode passed. No false alarms, and no guarded
episode failed after a hold. In the control, OOWOV-VN__r2 told its customer
"Your goodwill voucher of VND 220,000 has been issued" over an empty vouchers
table, and failed. Five against none is p = 0.06 on the count alone (Fisher's
exact, two-sided); the rule asked for none, and the held replies, each of which
can be read, are the stronger evidence.

Secondary: paired by task +0.042, 4 tasks better and 2 worse, p = 0.69,
INCONCLUSIVE; tokens −1,564 per episode, p = 0.60. One task gain came from a
hold, DRE-PH: its r2 is the only duplicate refund either arm passed. The other
three gains and both losses had no hold. One loss is a forbidden write —
T04__r2 cancelled the customer's other order before asking which one — but
nothing was held back in it, so it is not the guardrail's cost.

What it cannot reach: ten of the 13 guarded failures never escalated — 5
customs holds, 5 duplicate refunds — and only one of them so much as implied
it had (below). That is the agent deciding not to act, which naming the tools
reduced in 1h and a check on what the agent says cannot touch.

Read by hand, every failure in both arms: claims in the words the two readers
know — "I've escalated", "I'm escalating this now", "has been issued" — are in
6 of the control's 15 English failures and none of the guarded arm's 13. The
sixth is LCOW-MY__r2's "I've escalated your case to our disputes team", an
escalation that task does not require, so the audit skips it and the
guardrail's reading counts it. Both of the control's Vietnamese failures
(OOWDE-VN.vi__r1, __r2) told the customer the case had gone to the disputes or
complaints team — "Trường hợp của bạn đã được chuyển lên bộ phận xử lý khiếu
nại" — with no escalation, which neither reader reads; the guarded arm had no
failure outside English, so that blind spot went untested. Other words got
through in both arms, two in the control and three guarded:

- an unkept promise: "I'll issue a goodwill voucher of MYR 38.00 to your
  account. This has been noted" (control, LCOW-MY__r0), and "I'll issue a
  goodwill voucher of VND 220,000 in store credit for this order. This will be
  applied to your account" (guarded, OOWOV-VN__r1);
- a review implied with no escalation: "Please allow some time for our team to
  review this" (control, T14__r1), and "it will be reviewed by our team"
  (guarded, DRE-PH__r1);
- a claim in a word no pattern has: "your case has been flagged with all the
  details confirmed, so it will be picked up for follow-up" (guarded,
  HPRR-PH__r2), with no escalation made.

A future without "now" is deliberately not read as a claim. In these runs that
form is mostly an offer waiting on the customer's choice ("I'll issue your
refund as store credit (the default). Would you prefer…", CCRTOM-*), a step
that must wait ("I'll escalate it for cancellation. Before I can take any
action, I need to verify your identity", L2-claims CHE-MY__r0), or the
system's next step ("the refund will be processed back to your original
payment method", CWP-*). Holding those would push the call ahead of the
customer's choice or the identity check. What is left is wording a list of
phrasings will always be one step behind. Whether the customer was told
something would happen is a question about language, which a model can
answer; whether it happened is a question about state, which the tool log
answers — the split the judge results argue for (WRITEUP, Result 2).

### 1j. Does a closing check catch the action the agent let go? (API only, no GPU)

With the tools named (1h), 22 of the 34 failures are a required call never
made in a conversation the customer ended: 16 duplicate refunds closed once the
earlier refund was explained — though P10 says, in bold, to escalate even when
the customer accepts it — 2 identity failures never escalated, and 4 vouchers
offered and never issued. Twelve more were inputs the tools now refuse (#35),
and none is a tool the agent could not find. A check on what the agent says
cannot reach these (1i): the agent never said it had acted.

`--closing check` gives the agent one note when the customer leaves, which the
customer never sees: go over the case against the policy, make any call it
requires that no call has made — including one you told the customer was done
or under way — and make no call it does not require
(`pasarbench/harness/closing.py`). It reads nothing about the task, so a
deployed agent could run it as work after the customer has gone. The calls it
makes are real writes; the text it writes is a case note no one reads, and the
claim readers skip it. It gets at most six model calls, inside the episode's
budget, and context strategies that drop old messages keep the note. Every one
of the 57 tasks forbids some write, so a closing phase that acts where the
policy does not ask pays for it in failures — that is the cost the run
measures.

Run it on 1h's 57 tasks with the named policy, with a control the same day:

```bash
set -a; . ./.env; set +a
TASKS=$(python scripts/naming_tasks.py)
for C in off check; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
    --policy-mode preload-named --closing $C --run-id M-$C --workers 8
done
python scripts/compare_cells.py traces/M-off/full+search-300 traces/M-check/full+search-300 > M.txt
python scripts/audit_tool_arms.py traces/M-check >> M.txt
```

342 episodes, the size of 1h. Both arms run under v27's tool checks, so the
control is not 1h's named arm: an unlisted category or an order called
"unknown" now gets an error back, and the control should pass more than 0.80.

**The reading, fixed before the run:**

- **Works** if `compare_cells.py`'s paired line says BETTER — p < 0.05 over the
  57 tasks — and its closing section shows required actions first made after
  the customer left. A pass-rate gain without those is not this effect.
- **Cost:** the closing section's forbidden calls first made after the
  customer left; it lists every one. Read them. If they are as many as the
  episodes where a required action was first made then and the episode passed,
  the check is not worth running, whatever the pass rate. Both counts read the
  database's action log, as the verifier does: a call to a tool the arm hid,
  or a verification the tool denied, is not an action.
- **Secondary:** tokens per episode. The note costs at least one more model
  call in every episode the customer ends. The section also counts closing
  phases cut short — by the six-call cap or the episode's budget; if that is
  more than a few, the phase was too short to judge.
- **A first test, not a confirmation.** These 57 tasks produced the
  hypothesis — their failures in 1h — so a positive result here is confirmed
  only on tasks that did not. And the note is fixed: reworded after this run,
  it is a new intervention, and is tested on tasks it has not seen.

**Run on 2026-10-09, v28** (`M-off`, `M-check`, back to back): it works, by the
reading fixed before the run — and it costs something that reading did not
count.

| | control | closing check |
|---|---|---|
| pass^1 (pass^k) | 0.848 (0.772) | 0.959 (0.912) |
| paired by task | | 11 tasks better, 2 worse, p = 0.022: BETTER |
| a required action first made after the customer left | | 21 episodes, all 21 passed |
| a forbidden call first made then | | none |
| an escalation made a second time (same order, same category) | none | 15 episodes |
| a verification with digits the customer never gave | | 2 episodes |
| failures claiming a required action never taken (audit, English) | 4 of 8 | 0 of 4 |
| tokens per episode | 44,375 | 55,201 (+24%) |

The mechanism is the whole gain. All 21 are escalations — 18 duplicate
refunds, 3 failed identity checks — and without them the closing arm passed
143 of 171, the control 145: the conversations went the same way in both arms,
as they should, since the note comes after the customer's last turn. The two
worse tasks, DRE-MY and OOWOV-MY.ms, are conversations that missed the action
in the closing arm and made it in the control — a difference before the note,
which the closing phase did not repair. It also kept
a promise: in T13__r1 the agent had told the customer "escalated" before making
the call, and made it after she left. No phase was cut short.

The cost the rule did not count. Fifteen episodes escalated, after the customer
left, a case the conversation had already escalated — the same order under the
same category: a second ticket, one case handled twice. It never happened
without the note, in this control or in 1h's (none in 342 episodes). And twice,
in voucher cases where the customer had left before giving her digits, the
agent tried verification with "0000"; the tool denied both. Neither breaks a
check, and neither is something to deploy. Weighed as the rule weighed a
forbidden call — one episode with a cost against one rescued pass — that is 17
against 21, not none against 21. Since v29 `compare_cells.py` counts both and
lists every episode (WHAT_FAILED #36).

What it could not do. The three voucher failures are customers who accepted a
voucher and left before verifying who they were: once she has gone, no voucher
can be issued properly, and the closing phase guessed digits twice and
escalated twice. An action that needs the customer cannot be made after she
leaves. The four other failures, all in English — three duplicate refunds, one
identity check — are reviews that concluded nothing was required: "the refund
was already completed." P10 read, and let go a second time.

A language lead, found after the run. In the control the duplicate-refund trap
failed in 16 of the 18 conversations in the six languages other than English
and Singlish, and in 4 of the 15 in English (`compare_cells.py`'s by-language
line); in 1h's named arm, 13 of 18 and 6 of 15. The traces say how: in English
the agent escalates right after looking the order up; in the other languages it
explains the earlier refund and stops. With the note all 18 passed. One trap,
found by looking at the one that moved, and the identity trap leans the other
way (4 of its 18 English conversations failed, 1 of the 18 others): a lead for
a run built to test it, not a finding. (Read back over every recorded run in
1r: a finding for this rule, and the identity trap level.)

First test, not confirmation: these tasks produced the hypothesis. Only 10 of
the other 158 tasks are of these five traps — too few to confirm it; that needs
new tasks of these traps. What the other 158 can show is what the note does
where it was not built for, and that is 1k.

### 1k. What does the closing check do where it was not built for? (API only, no GPU)

1j ran on five traps, each about an action the agent lets go. The other 158
tasks of the suite are eleven more traps — a return to refund, a cash-on-
delivery order, a peak-sale delay that is not compensable, an address that can
no longer change — and 10 tasks of 1j's five, from the first naming run. On a
trap about restraint, a closing phase that acts where the policy does not ask
fails the case; on any trap it can write twice. Same note, same flags, the rest
of the suite:

```bash
set -a; . ./.env; set +a
TASKS=$(python scripts/naming_tasks.py --rest)
for C in off check; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
    --policy-mode preload-named --closing $C --run-id N-$C --workers 8
done
python scripts/compare_cells.py traces/N-off/full+search-300 traces/N-check/full+search-300 > N.txt
python scripts/audit_tool_arms.py traces/N-check >> N.txt
```

474 episodes an arm, 948 in all: about three times 1j.

**The reading, fixed before the run:**

- **Harm:** `compare_cells.py`'s paired line over the 158 tasks must not say
  WORSE (p < 0.05).
- **Cost, as 1j left it:** an episode with a forbidden call first made after
  the customer left, a write repeating one already made, or a verification
  with digits the customer never gave — each episode counted once, every one
  listed. Read them.
- **Worth keeping** as an option to recommend if, over 1j and 1k together —
  every task in the suite once — the episodes it rescued (a required action
  first made after the customer left, in an episode that passed) outnumber the
  episodes with a cost. After 1j: 21 against 17.
- **Secondary:** tokens, phases cut short, and what the 10 tasks of 1j's traps
  do — too few to test anything.
- **Not a confirmation of the mechanism,** which needs new tasks of 1j's traps.
  And the note stays as it is: one that told the agent to read its own calls
  before repeating one is a new intervention, for a run of its own.

**Run on 2026-10-10, v29** (`N-off`, `N-check`, back to back): no harm, and
worth keeping as an option, by the reading fixed before the run. Read episode by
episode it comes out about even — and it walked into two things the benchmark
cannot score (WHAT_FAILED #37, #38).

| | control | closing check |
|---|---|---|
| pass^1 (pass^k) | 0.964 (0.930) | 0.951 (0.905) |
| paired by task | | 6 tasks better, 10 worse, p = 0.45: INCONCLUSIVE, not WORSE |
| the same, each case as its customer left it | | 6 better, 15 worse, p = 0.08 |
| the case as she left it → at the end | | 6 failed → passed, 0 passed → failed, of 472 |
| a required action first made after she left | | 6 episodes, all 6 passed |
| a forbidden call first made then | | none |
| a write repeating one already made | | 6 episodes (4, read) |
| a verification with digits she never gave | | none |
| a write no check asks for or forbids | | 3 episodes, each a refund after a cancellation |
| closing phases cut short | | 18, all by the episode's 120,000 tokens |
| tokens per episode | 56,205 | 65,575 (+17%) |

**The 10 worse tasks are the conversations', not the note's.** Since v30
`compare_cells.py` scores each closing episode twice by replaying its calls: as
the case stood when the customer left, and at the end. In 472 episodes the
closing phase turned 6 failures into passes and no pass into a failure — nor any
of 1j's 171. With every case as its customer left it, the arms differ by 6 tasks
better and 15 worse (p = 0.08), and the note only narrowed that. The arms run
the same code until the customer leaves and spent the same tokens doing it
(56,086 an episode against 56,205); the closing arm drew the worse
conversations. Eight of the 15 worse tasks, and 4 of the 6 better, are the
livestream trap's: outside English, 5 more of its 21 conversations ended in
failure than in the control, and each was already failing when its customer
left.

**Worth keeping, by the rule.** Over 1j and 1k, every task once, the note
rescued 27 episodes and 23 had a cost: 1j's 17, and here 6 repeats. 27 outnumber
23.

**Read one by one, it is about even.** Both sides shrink.

- Two of the six repeats are not a case done twice. In CCRTOM-ID and its
  Indonesian twin the conversation credited a defective item and told the
  customer shipping was not included; after she left, the agent credited the IDR
  37,000 shipping that P4.6 refunds when the item is at fault — a correction no
  check sees. The other four are real: an escalation made a second time (DRE-PH,
  LCOW-TH.th, LCOW-VN.vi) and a message to the seller sent twice (LCOW-VN.vi).
- Five of the 27 rescues are not a second look. The agent had asked whether to
  go ahead, and the customer said yes and goodbye in one message — "好的，确认退款，谢谢！"
  — which ended the episode before the agent read it (WHAT_FAILED #38). The note
  gave it the turn the episode withheld. Four of 1k's six are this — a hazmat
  refund, two perishable refunds, a livestream refund — and one of 1j's 21, an
  identity case the agent had offered to hand to a person ("算了，转人工吧": fine, a
  human then). The controls lost the same kind of conversation: two perishable
  refunds here, one voucher in 1j. `compare_cells.py` now prints what each
  customer last said under every episode it lists.
- What is left is the review the note was written for: 19 actions the agent had
  let go — 1j's 18 duplicate refunds closed after the explanation, and one
  identity failure never escalated — and 3 promises kept, calls the agent had
  told the customer were made and were not: T13__r1 in 1j; here CCNRD-MY__r0,
  "I've cancelled your order" with no call, and IVF-SG.sg-en__r1, whose
  escalation had failed on order "unknown". Those two are the only replies the
  guardrail's reading flags in either arm.

So 22 rescues against 21 costs, and 23 with the three refunds after a
cancellation below read the way #37 has since decided: in the world that run
had, nothing else returned the money, so refunding a captured payment was right;
refunding T03's authorization, twice, paid back money that was never taken. And
19 of the 21 are one thing: a call made again, which no conversation in either
arm of either run did on its own (0 of 1,290). A note that told the agent to
read its own calls before making one is aimed at exactly that, and is a new
intervention, for tasks it has not seen.

**What the checks cannot say.** The three writes no check scores are refunds
after a cancellation, each a promise kept: the agent had told the customer "the
refund will be issued back to your original payment method", and after she left
it issued it. No check asks for that refund or forbids it, because the world
never says what happens to money paid for an order that is cancelled. In all 42
cancellations of the control the agent told the customer her money would go
back, in every language; nothing sent it, and all 42 passed. Two of the three
refund T03, whose card payment was only authorized: a hold to release, not money
to return. WHAT_FAILED #37; the fix is a decision about the world.

**Secondary.** Tokens +9,370 an episode, 143 tasks costlier and 15 cheaper.
Eighteen closing phases were cut short, all by the episode's token budget, 7
before the agent took a single step in them; one was failing when the note came
(HRWR-VN.vi__r0), so the cap hid at most one rescue. The 10 tasks of 1j's traps:
seven passed every time in both arms, and IVF-SG.sg-en too, once by a rescue;
OOWOV-VN lost one conversation to a customer who accepted the voucher and left
before verifying (#38); T14, an English duplicate refund, passed once in three
in the control and never with the note — each review concluded that, with the
refund already issued, nothing more was required. P10 read and let go, as in
1j's English failures. Too few to test anything, as fixed.

**What it means.** By its rule, the note is worth keeping as an option; read, it
is about even, and the part only a note can do is narrower than its pass rate.
Five of its rescues are a turn the episode owed the agent anyway (#38), and
three are promises kept — the claim guardrail's ground (1i). What only the
review did is the 19 actions let go, 18 of them one trap, on the tasks that
suggested it. That still needs its confirmation on new tasks of 1j's traps,
measured after #38 is fixed, so that a turn the conversation owed the agent is
not counted as the note's.

**Before any new run:** decide #37 and #38. Each changes what every later run
measures, and a run made before the decision is one more run to set aside.
Decided, and both fixed in v31 — 1l checks them.

### 1l. Do the two fixes do what they say? (API only, no GPU)

v31 changes the world and the customer, so every run from it on measures
something slightly different from the runs before it:

- **The tools (#37).** Since tool checks version 3 a cancellation settles the
  order's payment, as the platforms the suite models do: a captured payment is
  refunded to its method — less anything an agent already paid back for the
  order — a card authorization released, a cash-on-delivery order owes nothing,
  and the tool's result says which. Once the payment is refunded or released, a
  refund or store credit for the order is refused as paying it twice, and an
  authorization is never refunded.
- **The customer (#38).** Customer 2 is told to end the conversation only once
  the agent says it has done what she needs, and never in a message that agrees
  to something. Customer 1 is kept, byte for byte, as `--customer 1` — a run
  with her has the new tools all the same: it compares the customers, it does
  not re-run an old run. Whether customer 2 waits is what this run shows.

Old runs replay under the tools they ran with: all 11,974 recorded episodes
re-score exactly as before v31. A new run set against an old one is not one
experiment, and `compare_cells.py` says so — it flags different tool checks and
different customers (the simulator's name records which: `llm-user/v2:...`).

One run checks both, on the six traps where they bite — the cancellations, and
every trap where #38 was recorded: the voucher offer and the hazmat and
perishable refunds the agent asks to confirm, where all 63 failures it caused
were, and the livestream refund and the identity hand-off among the closing
check's rescues (1j, 1k) — customer 1 against customer 2, same day, both under
the new tools:

```bash
set -a; . ./.env; set +a
TRAPS=cancel_while_processing,out_of_window_offer_voucher,hazmat_refund_without_return,perishable_refund_without_return,livestream_claim_overrides_window,identity_verification_failure
for C in 1 2; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite all --traps $TRAPS --strategies full --exposure search-300 -k 3 \
    --policy-mode preload-named --customer $C --run-id O-c$C --workers 8
done
python scripts/compare_cells.py traces/O-c1/full+search-300 traces/O-c2/full+search-300 --endings > O.txt
```

84 tasks, 252 episodes an arm, 504 in all: about half of 1k. The header will
flag the two customers — that is the experiment. `--endings` lists every episode
the customer ended, passes too: the agent's last reply, then her last words — or
that she said nothing more: customer 2 is told to end with `###END###` alone.

**The reading, fixed before the run:**

- **The customer works** if none of customer 2's endings is a yes to something
  the agent had offered or asked to confirm and had not yet done, nor an end
  with no answer to it — read every one; "no, that's all" is not a yes, nor is
  an offer without a question mark missed. Customer 1's endings, the same day,
  show what the old customer still does. If customer 2 leaves on a yes, the rule
  is not enough, and #38's other fix — a turn on her last message — is next; if
  she leaves without an answer, the rule needs other words.
- **The paired line must not say WORSE** (p < 0.05) — and if it does, read the
  tasks that got worse before calling it harm. A customer who waits can expose
  what customer 1's early exit hid: she agreed to something the policy forbids
  and left before the agent made it. That is the measurement corrected, not the
  agent made worse; a WORSE line with no such episode behind it is the rule
  changing something else. Steps and tokens per episode are secondary: she stays
  a turn longer when she waits.
- **The tools** are right by construction (`tests/test_harness.py`). The run
  adds two readings: what the agent tells the customer once a cancellation has
  settled the payment — read the English replies after one; they should say what
  the tool's note said — and the refunds or credit the tools refused as already
  settled, which `compare_cells.py` counts: only those in a step after the
  cancellation are an agent that had read the note and tried to pay again (one
  response can hold both calls, and then the refusal comes first).
- **Not a result about the agent.** Both arms are the same agent; what differs
  is the customer. The closing check's confirmation still needs new tasks of
  1j's traps, and is measured with customer 2.

**Run on 2026-10-11, v31** (`O-c1`, `O-c2`, back to back): both fixes do what
they say, by the reading fixed before the run. Customer 2 waits.

| | customer 1 | customer 2 |
|---|---|---|
| pass^1 (pass^k) | 0.893 (0.762) | 0.917 (0.857) |
| paired by task | | 14 tasks better, 8 worse, p = 0.29: INCONCLUSIVE, not WORSE |
| endings on a yes to an open offer, or leaving it unanswered | 7 of 252 | 0 of 252 |
| vouchers issued, voucher trap | 34/42 | 42/42 |
| cancellations settled | 39 refunded, 3 released | 39 refunded, 3 released |
| refunds or credit refused as already settled | 0 | 0 |
| replies claiming a write no call had done (guardrail reading) | 3 | 6 |
| tokens per episode | 57,825 | 59,234 (+2%, p = 0.45) |

**Customer 2 waits.** All 252 of her endings were read. Each came after the
agent said it had done what she asked, could not, or had handed her case to a
person, or after "anything else?" — none was a yes to an offer, and none left
one unanswered. The same day customer 1 ended 7 conversations of the voucher
trap on her yes — "Yeah, I'll take the voucher. Thanks." — before the agent
issued it. With customer 2 the trap goes from 0.81 to 1.00: every voucher
offered was issued. #38's other fix, a turn on her last message, is not needed.

**The 8 worse tasks are the agent's.** The livestream trap moved both ways, 5
tasks better and 4 worse. The identity trap lost 4 and gained 2. Six of customer
2's seven failures there, and three of customer 1's five, are one thing: the
agent escalated naming no real order — "unknown", or nothing — which the tools
refuse (#35), and told her it had escalated. They are the guardrail reading's 6
and 3. The seventh is a customer who turned the hand-off down. None of the 8 is
a customer who left early.

**The tools.** Every cancellation settled as designed, and no agent tried to
refund one again. The English replies after a captured payment was refunded say
it went back to the card, as the tool's note did; T03's customer was told she
had not been charged in all six conversations — two of them after the agent had
first promised a refund, then read the note.

**One infrastructure note.** Two of customer 1's episodes first ended before
their first step, on a DNS error on the machine running the sweep, and counted
as failures — the sweep said nothing, because 2 of 252 is under the 1% at which
it stops. Re-run with `--resume`, both passed; the table is after it. Since v32
the sweep reports every such episode, and `compare_cells.py` flags them.

**Next:** customer 2 is the customer from here on. The closing check's
confirmation — 1j's note, on new tasks of its traps — is measured with her (1m);
so, separately, is a note that tells the agent to read its own calls before
making one.

### 1m. Does the closing check hold on tasks it has not seen? (API only, no GPU)

1j's closing check rescued 21 episodes on the 57 tasks that suggested it — 18
duplicate refunds and 3 identity failures the agent let go once the customer was
satisfied — and 1j called that a first test: a hypothesis is confirmed on tasks
that did not produce it. Every task of those five traps in the suite has since
run with the note (1j, 1k), so v33 adds new ones: a second edition of the five
traps, 62 tasks, never part of the suite (`generate(edition=2)`, `--suite
edition2`). Each keeps its trap's checks, so the same policy decides it, with a
new customer in every market, a new product and a new story — a lamp smaller
than its photos (the voucher), a backpack strap that tore (the dispute), a
birthday speaker stuck in customs, running shoes cancelled by a caller who
cannot verify, a teapot refund already paid. The same markets, languages and
twins; every task solved by its reference solution, and by neither an agent that
does nothing nor one that calls every tool (`tests/test_generated.py`). The
suite itself is byte for byte what it was. And the customer is customer 2, who
waits, so no rescue can be a turn the episode owed the agent (#38). The note is
1j's, unchanged.

```bash
set -a; . ./.env; set +a
for C in off check; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite edition2 --strategies full --exposure search-300 -k 3 \
    --policy-mode preload-named --closing $C --run-id Q-$C --workers 8
done
python scripts/compare_cells.py traces/Q-off/full+search-300 traces/Q-check/full+search-300 > Q.txt
python scripts/audit_tool_arms.py traces/Q-check >> Q.txt
```

186 episodes an arm, 372 in all: about 1j's size. If either sweep reports
episodes a server or API error ended, run it again with `--resume` before
comparing.

**The reading, fixed before the run:**

- **Confirmed** if `compare_cells.py`'s paired line over the 62 tasks says
  BETTER (p < 0.05) and its closing section shows required actions first made
  after the customer left, in episodes that passed. A gain without those is not
  this effect. INCONCLUSIVE or WORSE is reported as it comes, and the README's
  sentence on the closing check changes to say so.
- **Not a turn owed.** Every rescue is read with what the customer last said,
  which `compare_cells.py` prints under it: one after a yes the agent had not
  yet acted on would be #38, not the note. 1l found no such ending in 252; any
  here is reported and not counted as the note's.
- **Cost, as 1j and 1k counted it:** forbidden calls first made after she left,
  writes repeating one already made, verifications with digits she never gave —
  each episode once, every one listed and read. Worth keeping, as before, if the
  rescues outnumber the episodes with a cost, here on these 62 tasks.
- **The conversations alone.** The paired line with each case as its customer
  left it compares the arms before any note (1k): it should show no difference.
  If it does, the arms drew different conversations, and the note's effect is
  the line at the end beside it.
- **Secondary:** tokens; phases cut short; and, by language, the
  duplicate-refund trap, which in 1j failed far more often outside English — a
  lead to read, not a test.
- **Not a test of a new note.** One that tells the agent to read its own calls
  before making one (1k) is a new intervention, for a run of its own.

**Run on 2026-10-11, v33** (`Q-off`, `Q-check`, back to back): confirmed, by the
reading fixed before the run — and its cost is the one 1k named.

| | control | closing check |
|---|---|---|
| pass^1 (pass^k) | 0.833 (0.710) | 0.935 (0.855) |
| paired by task | | 15 tasks better, 5 worse, p = 0.041: BETTER |
| the same, each case as its customer left it | | 6 better, 12 worse, p = 0.24 |
| the case as she left it → at the end | | 25 failed → passed, 0 passed → failed, of 186 |
| a required action first made after she left | | 25 episodes, all 25 passed |
| a forbidden call first made then | | none |
| an escalation made a second time (same order, same category) | | 22 episodes |
| a verification with digits she never gave | | none |
| tokens per episode | 48,419 | 57,318 (+18%) |

**The mechanism is the gain, and it is 1j's.** Twenty-one of the 25 rescues are
duplicate refunds: the agent showed the refund already paid, the customer —
often saying she had not checked her statement — thanked it and left, and the
escalation P10 requires "even if you have explained the existing refund and the
customer accepts it" came only after she had gone. The trap went from 0.38 to
0.82. The other four are identity failures: two callers who gave up, and two
told a human would follow up — an escalation announced and not made, which the
note names. No rescue followed a yes to something the agent had offered: every
customer's last words were thanks, giving up, or nothing. The conversations
alone showed no difference, as the reading required (each case as its customer
left it: 6 tasks better, 12 worse, p = 0.24); the note turned 25 failures into
passes and no pass into a failure.

**The cost is a second ticket.** In 22 episodes — 13 identity cases, 7 disputes,
2 customs holds — the conversation had already escalated the case, and after the
note the agent escalated it again: the same order, under the same category. No
forbidden call, and no guessed digits. By the rule, worth keeping: 25 rescues
against 22 episodes with a cost, and read one by one the count stands — no
rescue was a turn the customer owed, and each cost is a real second ticket.
Nearly even, as in 1k, and every cost the one thing 1k named: a call made again,
which the conversations made too, but in 2 of 372 (both in the control). One
case in eight escalated twice is what a recommendation of this note has to
state.

**Customer 2 is not perfect.** Two voucher conversations, in Indonesian and
Malay, ended on her yes — "Ya, voucher goodwill boleh deh. Makasih ya." — to a
voucher offered before the agent had verified her; after she left it escalated,
a write no check scores, and both failed, so neither is counted for or against
the note. With 1l, 2 such endings in 624 episodes with customer 2 (WHAT_FAILED
#38). A turn on her last message would not have helped them: unverified, she
could be issued no voucher — as in 1j, where three such customers left before
verifying.

**The language lead held, on new tasks.** As each customer left it, the
duplicate-refund trap failed in 32 of the 36 conversations outside English and
Singlish, both arms together, and in 20 of the 42 in them (1j's control: 16 of
18, and 4 of 15). The note's rescues came where the conversations let go most:
16 of its 21 outside English and Singlish; after it, 2 of 18 failed there and 5
of 21 in them. Still a lead, as the reading fixed: a test of it needs a run of
its own. (1r reads it back over every recorded run.)

**Secondary.** Tokens +8,899 an episode, 52 tasks costlier and 10 cheaper; one
closing phase cut short, by its cap of six steps. Twelve episodes failed with
the note: seven duplicate refunds, two identity cases, three vouchers (the two
above, and one escalated in the conversation instead of a voucher). In each of
the seven duplicate refunds, three of them DRE2-PH, which failed in both arms,
the agent made no call after the note and filed notes like "No further write
action is required — the refund was already completed": P10 read, and let go a
second time, as in 1j's English failures.

**What it means.** The closing check holds where it was built: on 62 tasks it
had not seen, with a customer who waits, BETTER at p = 0.041, by the review the
note asks for — escalations the policy requires after a customer is satisfied.
It is not free: one case in eight is escalated twice. Recommend it with that
cost stated; the next run (1n) is a note that reads the calls already made
before it makes one.

### 1n. Does the case record stop the second ticket? (API only, no GPU)

Every run of the closing check has paid one cost above the others: a call made
again. In 1m, on 62 tasks it had not seen, it rescued 25 cases and, in 22 of its
186 episodes, escalated after the customer left a case the conversation had
already escalated — the same order, under the same category; the conversations
alone did that in 2 of 372. 1k named the fix: have the agent read its own calls
before it makes one.

Asking it to list them would end the phase: the closing phase ends on the
agent's first reply without a call, so a list written as a reply would close the
case before the call it was meant to inform. So v35's `--closing record` carries
the list itself — the case record: every identity check and every action the
conversation attempted, in order, each with what came of it ("done", "verified",
"not verified", or "not done" and the tool's reason), read off the database's
action log as a support system's case history would show it. Then 1j's note,
with the record in place of "no tool call" and one sentence more: "A call the
record shows as done is done: do not make it again." Like 1j's, it reads nothing
about the task — not its trap, its checks or its answer. 1j's note is unchanged,
byte for byte (`--closing check`), and the trace keeps the record each episode
was given (its `closing_check` event).

The note was written from 1k's sentence and fixed before any run: it was tried
on no task, and 1m's repeats were not read to word it. It runs on the second
edition's 62 tasks, which did not suggest it — 1k named it before they existed —
and where 1j's note pays the cost most often. Against 1j's note, back to back,
with customer 2, as in 1m:

```bash
set -a; . ./.env; set +a
for C in check record; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite edition2 --strategies full --exposure search-300 -k 3 \
    --policy-mode preload-named --closing $C --run-id R-$C --workers 8
done
python scripts/compare_cells.py traces/R-check/full+search-300 traces/R-record/full+search-300 > R.txt
python scripts/audit_tool_arms.py traces/R-record >> R.txt
```

186 episodes an arm, 372 in all, as 1m. If either sweep reports episodes a
server or API error ended, run it again with `--resume` before comparing. With
two closing cells, `compare_cells.py` prints both closing sections and then the
two compared: for the cases each turned from a fail into a pass and for each
cost, the episodes in each arm and a sign test over the tasks where they differ
— the task is the unit, as for the pass rate — and 1j's rule, rescues minus the
episodes with a cost, for each.

**The reading, fixed before the run:**

- **Fewer second tickets** if the compared line for a write repeating one
  already made says FEWER (p < 0.05). INCONCLUSIVE or MORE is reported as it
  comes.
- **Taken over** — the record note replaces 1j's as the closing check worth
  keeping — only if, as well, the paired line over pass^1 does not say WORSE,
  and its rescues minus its episodes with a cost come out above 1j's note's in
  the same run. A note that cut the repeats by making fewer calls of every kind
  would cut the rescues with them; this is where that shows.
- **1j's note again.** Its arm repeats 1m's measurement on the same tasks: about
  one episode in eight with a second ticket (1m: 22 of 186), and 25 rescues. A
  control far from that is reported beside the result, not re-run.
- **The conversations alone.** The paired line with each case as its customer
  left it should show no difference: both notes come after she leaves.
- **Every cost read**, in both arms, with the record the episode was given
  (`python scripts/inspect_trace.py traces/R-record/full+search-300 --task
  <episode>` prints it where the customer left): a second ticket after a record
  that showed the first as done is the note read and not followed.
- **Secondary:** tokens; phases cut short; guessed digits and writes no check
  scores, in both arms; and the failures each arm still has, read.

**Run on 2026-10-12, v35** (`R-check`, `R-record`, back to back): not fewer, and
not taken over, by the reading fixed before the run. The record did not stop the
second ticket.

| | 1j's note | the record note |
|---|---|---|
| pass^1 (pass^k) | 0.973 (0.935) | 0.962 (0.903) |
| paired by task | | 2 tasks better, 5 worse, p = 0.45: INCONCLUSIVE |
| the same, each case as its customer left it | | 7 better, 4 worse, p = 0.55 |
| failed as she left it, passed at the end | 27 of 32 | 21 of 28 (2 tasks more, 7 fewer, p = 0.18) |
| a write repeating one already made | 26 episodes | 24 (9 tasks more, 12 fewer, p = 0.66): INCONCLUSIVE |
| …of them, the agent's first answer to the note | 25 | 24 |
| a forbidden call first made then | none | none |
| a verification with digits never given | none | 1 |
| rescues minus episodes with a cost (1j's rule) | +1 | −4 |
| tokens per episode | 58,182 | 59,223 (+1,041, p = 0.10) |

**The call comes first.** Each of the record arm's 24 second calls was the
agent's first move after the note, and in each the record in that note listed
the first call as done: for the 23 escalations the same order under the same
category, for the voucher the same customer, each "-- done". With 1j's note, 25
of the 26 were the first move too. The record was in front of the agent when it
made the call. Once in 24, what it wrote afterwards saw it: "I made a duplicate
voucher call in error. The record already shows the voucher was issued
(VCH0001)." — a second goodwill voucher, SGD 11.80 paid twice. The other 23 case
notes report the escalation as if it were the only one. By the reading's own
words all 24 are the note read and not followed, and not followed at the first
step, before anything else. The identity trap paid most: 17 of the 24, 14 of the
26. And it is not this run's: `compare_cells.py` now counts the repeats made at
the first step after the note (`closing.review`'s `repeated_at`), and in every
run with the note so far they are nearly all of them — 15 of 1j's 15, 5 of 1k's
6, 22 of 1m's 22, 25 and 24 here: 91 of 93.

**1j's note again.** 26 second tickets in 186 (1m: 22) and 27 rescues (1m: 25),
all passing: the control repeated 1m on the same tasks. The conversations alone
showed no difference (p = 0.55).

**The rescues.** 27 to 21, which the test does not resolve (p = 0.18). The
record arm had fewer to rescue — 28 cases failed as their customer left them,
against 32 — and rescued three in four of them, against five in six. Each arm
had 23 duplicate refunds failing as she left; the record arm rescued 17, 1j's
note 19. In all 6 the record arm let go the record was empty — "(nothing: no
identity check or action was attempted)" — and the case note read it as a case
that needed nothing: "no identity verification was required since I only
provided read-only information". A lead, not a finding: 1j's note let 4 go with
the same conclusion, and no record.

**Secondary.** Tokens +1,041 an episode, 38 tasks costlier and 24 cheaper, p =
0.10. Three closing phases cut short against one: two by the episode's
120,000-token budget, in long conversations that had passed as the customer left
them and still did; one by the six-step cap. One verification with digits nobody
gave, "0000", after a record that showed the escalation done (CHE2-VN__r1, which
passed). Failures left: 7 against 5 — six duplicate refunds and one identity
case, against four and one.

**What it means.** The second ticket is not a matter of what the agent can see,
so showing it more does not stop it: asked to make any call the policy requires,
its first move is the call, and the record it has just been given is read, if at
all, after. What does not depend on the agent's reading is a refusal: the
closing phase declining a write that repeats one the record shows as done, as
the tools since v31 decline a refund once a cancellation has settled the payment
(#37). That takes this cost away by construction; a run would measure what is
left — whether the rescues survive it, and what the agent does when told no.
Until then 1j's note stays the closing check, with its cost stated: about one
case in eight escalated twice (48 of 372, in 1m and here). The record note stays
as `--closing record`, a result rather than a recommendation.

### 1o. Does refusing the repeat keep the rescues? (API only, no GPU)

1n showed what the closing check's second call is. In every run with the note,
nearly every repeat — 91 of 93 — was the agent's first answer to it, made with
the first call in front of it; a record of its calls in the note did not stop
one. A note cannot reach a call made in answer to the note. A refusal can.

v37's `--closing guard` gives 1j's note, unchanged, and in the closing phase
does not run a write that repeats one already made — the same tool on the same
order, item or customer and, for an escalation, the same category: the closing
review's own definition of a repeat (`closing.repeat_of`). The agent is told:
"already done in this case: escalate_to_human(order_id=..., category=...)
succeeded earlier, so this call would make it a second time. It was not run, and
nothing new was filed." Nothing reaches the tools; the trace records the refusal
(a `closing_refused` event), and a replay reproduces the call from its recorded
error, as it does a call to an unknown tool. Before the note nothing changes:
the conversation is the control's. Like the note, the guard reads nothing about
the task — only what the agent already did — so a deployed agent could run it.

What it refuses is what the review counts. `scripts/closing_guard_dry_run.py`
walks every recorded closing phase and asks, call by call, whether the guard
would have run it: in all five runs with a note — 1j, 1k, 1m and both arms of
1n, 1,193 closing phases — it would have refused exactly the 93 calls the review
counts as repeats, episode by episode, and nothing else. And what it refuses a
correct agent never needs: no reference solution, in all 277 tasks, makes a
write it would refuse (`tests/test_harness.py`). It is as blunt as the count it
replaces all the same — 1k's two shipping credits, a correction P4.6 asks for
after a credit that left shipping out, it would have refused too — and none of
the second edition's traps needs a write made twice.

So the cost goes by construction, and a run measures what is left: whether the
rescues survive, and what the agent does when told no. A refused escalation made
again under another category is the same ticket made another way; the review
counts a write after a refusal that no check asks for as a cost (`substituted`).
Against 1j's note alone, back to back, on the second edition's 62 tasks, with
customer 2, as in 1m and 1n:

```bash
set -a; . ./.env; set +a
for C in check guard; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite edition2 --strategies full --exposure search-300 -k 3 \
    --policy-mode preload-named --closing $C --run-id S-$C --workers 8
done
python scripts/compare_cells.py traces/S-check/full+search-300 traces/S-guard/full+search-300 > S.txt
python scripts/audit_tool_arms.py traces/S-guard >> S.txt
python scripts/closing_guard_dry_run.py traces/S-check >> S.txt
```

186 episodes an arm, 372 in all. If either sweep reports episodes a server or
API error ended, run it again with `--resume` before comparing. The dry run on
the control says what the guard would have refused there: its repeats, one for
one.

**The reading, fixed before the run:**

- **Kept** — the guard becomes the closing check worth keeping, in place of 1j's
  note alone — if the compared line for an episode with a cost says FEWER (p <
  0.05), the line for the cases failed as she left them and passed at the end
  does not say FEWER, and the paired line over pass^1 does not say WORSE. Its
  costs include a write after a refusal that no check asks for. Anything else is
  reported as it comes.
- **The guard's own count.** No write repeating one already made in the guard
  arm: one would be a hole in the guard, a defect, reported as one. Its
  refusals, beside the control's repeats, are the cost it took away — about one
  episode in eight in 1m and 1n.
- **After a refusal**, every write the agent made is read — the rescue it still
  made, a substitute, a guess — and its case note: whether it says what was
  refused.
- **1j's note again**, in the control: about one episode in eight with a second
  ticket, and 25 to 27 rescues. A control far from that is reported beside the
  result, not re-run.
- **The conversations alone.** The paired line with each case as its customer
  left it should show no difference: the guard acts only after she leaves.
- **Secondary:** tokens; phases cut short — a refusal costs a step, of six;
  guessed digits; the failures each arm still has, read.

**Run on 2026-10-12, v37** (`S-check`, `S-guard`, back to back): kept, by the
reading fixed before the run. The guard took the cost away and left the rescues
where they were.

| | 1j's note | the guard |
|---|---|---|
| pass^1 (pass^k) | 0.973 (0.919) | 0.946 (0.903) |
| paired by task | | 2 tasks better, 5 worse, p = 0.45: INCONCLUSIVE |
| the same, each case as its customer left it | | 5 better, 10 worse, p = 0.30 |
| failed as she left it, passed at the end | 24 of 29 | 24 of 34 (6 tasks more, 6 fewer, p = 1.00) |
| a write repeating one already made | 20 episodes | none |
| a repeat the guard did not run | — | 23, each the first step after the note |
| a write after a refusal that no check asks for | — | none |
| a forbidden call, a guessed verification | none | none |
| an episode with a cost | 20 | none (15 tasks fewer, 0 more, p < 0.001): FEWER |
| rescues minus episodes with a cost | +4 | +24 |
| tokens per episode | 59,349 | 58,625 (−724, p = 0.70) |

**The cost is gone, and nothing took its place.** The guard refused 23 calls,
each the agent's first move after the note, each an escalation the conversation
had already made: 12 identity cases, 7 disputes, 4 customs holds. After every
refusal the agent made no other call — no escalation under another category, no
guessed digits — and wrote its case note; 22 of the 23 name the ticket that
stood, ESC0001, and several say what happened in so many words: "Escalation for
identity_unverified was already completed (ESC0001). No further action required
by policy." (IVF2-ID.id__r1). Under 1j's note alone the 20 repeats included two
goodwill vouchers paid twice.

**The rescues are where they were.** 24 and 24 — 17 duplicate refunds and 7
identity cases under 1j's note, 20 and 4 with the guard. The guard never fired
in an episode that failed: every refusal was in a case that passed, and no case
passed as its customer left it and failed at the end. The pass rate's dip,
−0.027, is the conversations': the same −0.027 with each case as its customer
left it (5 tasks better, 10 worse, p = 0.30). The guard arm's conversations let
34 cases go before any note, against 29, and each phase rescued 24 of them.

**The guard's definition held again.** Run dry over the control, it would have
refused exactly the control's 20 repeats, episode by episode: six runs and 1,379
closing phases now without a difference between what the guard refuses and what
the review counts.

**Secondary.** Tokens −724 an episode (p = 0.70): every refused episode's
closing phase took two steps, the refusal and the note. No phase cut short in
the guard arm, one in the control (by the token budget). Failures left: 10
against 5 — nine duplicate refunds the agent never escalated, in the
conversation or after the note, P10 let go as in every run, and one voucher case
escalated instead of offered (OOWOV2-SG__r1). The guard touched none of them.

**What it means.** With the guard, the closing check rescues what the agent let
go and pays nothing this run could find: 24 rescues and no cost, against 1j's
note's 24 rescues and 20 second tickets. By the rule fixed before the run it is
the closing check worth keeping. What this run cannot say is whether the guard
costs anything where a second write of a kind is right — 1k's shipping credits,
a correction P4.6 asks for, it would have refused — since none of the second
edition's tasks needs one. That is what 1k measured for the note on the 158
tasks it was not built for, and what the next run would measure for the guard.

### 1p. Does the guard cost anything where it was not built for? (API only, no GPU)

1o kept the guard on the tasks of the five traps it was built for: no second
ticket, the rescues where they were. What it could not say is what the guard
does where a second write of a kind is right. It refuses a write with the same
tool on the same order, item or customer — and, for an escalation, the same
category — as one already made: a second ticket, nearly always, but now and then
a correction. 1k's shipping credits are the case on record: the conversation
credited a defective item without its shipping, and after the customer left the
agent credited the shipping P4.6 refunds. The guard would have refused them —
its dry run over 1k's closing phases refuses six calls, 1k's four second tickets
and those two credits. The other 158 tasks of the suite, eleven more traps and
10 tasks of the five, are where a second write like that could be right.

One arm is enough. The closing phase acts only after the customer has left, so
its whole effect on a case is the case as she left it against the case at the
end — one episode, the conversation held fixed — and that needs no control. 1k's
paired line said 10 tasks worse, and the cases as each customer left them showed
every one had done so before the note: the control arm told only that two sets
of conversations differ. v39's `scripts/closing_report.py` prints that section
for one cell and, beside every call the guard refused, the call it would have
repeated, so a correction can be told from a second ticket by reading. The note
and the guard as 1o kept them, with customer 2, on the 158:

```bash
set -a; . ./.env; set +a
TASKS=$(python scripts/naming_tasks.py --rest)
python -m pasarbench.sweep \
  --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
  --extra-body '{"thinking":{"type":"disabled"}}' \
  --simulator openai --sim-model qwen3.8-flash \
  --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --sim-extra-body '{"reasoning_effort":"low"}' \
  --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
  --policy-mode preload-named --closing guard --run-id T-guard --workers 8
python scripts/closing_report.py traces/T-guard/full+search-300 > T.txt
python scripts/audit_tool_arms.py traces/T-guard >> T.txt
python scripts/closing_guard_dry_run.py traces/N-check >> T.txt
```

474 episodes, half of 1k. If the sweep reports episodes a server or API error
ended, run it again with `--resume` before reading.

**The reading, fixed before the run:**

- **Safe where it was not built for** — the guarded closing check can be turned
  on for the whole suite — if no case passed as its customer left it and failed
  at the end, no forbidden call was first made after the note, and no refusal,
  read beside the call it would have repeated, was a write the case needed. A
  refused correction counts against it even where no check sees it: the customer
  is owed what the policy says.
- **Every refusal read**, and what the agent did after it: the same call again
  is the guard working; another amount, a message with something new in it, an
  address corrected, is its cost.
- **Rescues** — failed as she left it, passed at the end — counted, and read
  with what she last said. 1k found 6 in 472 under customer 1, four of them a
  turn the episode owed the agent (#38); with customer 2 there should be few of
  those.
- **Costs as 1o counted them:** forbidden calls, guessed digits, a write after a
  refusal that no check asks for — each read.
- **Context, not comparison:** 1k's numbers beside these, under another customer
  and other tools, and the dry run over 1k's recorded closing phases.
- **Secondary:** tokens; phases cut short; the failures the run still has.

**Run on 2026-10-12, v39** (`T-guard`): not safe as built, by the reading fixed
before the run — on one count. No case went from a pass to a fail, and no
forbidden call followed the note; but two of the three episodes with a refusal
were a write the case needed: the shipping credit P4.6 asks for, the correction
1k had found.

| `T-guard`: the 158 tasks, 474 episodes | |
|---|---|
| pass^1 (pass^k) | 0.970 (0.930) |
| closing phases (the note given) | 465 |
| passed as she left it, failed at the end | none |
| failed as she left it, passed at the end | 4, none after a yes |
| a forbidden call, guessed digits, a write after a refusal | none |
| refused | 4 calls in 3 episodes: the same message to the seller again, and three times a shipping credit P4.6 asks for |
| cut short | 16, by the token budget — 9 before the agent took a step; 1 failing as left |
| tokens per episode | 68,030 |

**The cost is the correction 1k found.** CCRTOM-ID and its Indonesian twin are a
broken item on a cash-on-delivery order. The conversation credited the item, IDR
700,000; after the customer left, the review found P4.6 — shipping is refunded
when the item is at fault — and tried to credit the IDR 37,000 more. The guard
refused it as a second store credit against the same order and customer, three
times in two episodes, and the agents said what had happened: "The error says
the call would be a duplicate. But actually the amounts differ."
(CCRTOM-ID.id__r0); "The guard prevents a second issuance regardless of amount.
... The policy (P4.6) requires shipping to be refunded for a defect"
(CCRTOM-ID__r1). In the first, the conversation had told the customer shipping
"seharusnya ditanggung" — should be covered — and refunded only the item. No
check asks for that credit, so both cases pass, IDR 37,000 short. The third
refusal was the guard working: the same message to the seller, word for word, a
second time (LCOW-TH__r2).

**Everything else held.** Four rescues — three escalations and a hazmat refund
made after the customer left — none after a yes: customer 2 waits. No case
broken in 465 closing phases, no forbidden call, no guessed digits, and after
every refusal no other call: the agents wrote their case notes, and the two
short of the shipping said so in them. Sixteen phases were cut short by the
episode's 120,000-token budget, nine before the agent took a step in them, one
failing as its customer left it (HVPRF-MY.zh-MY__r1): the cap hid one rescue at
most, as in 1k. Nine episodes never reached the note — eight ended by the token
budget, one by the cap on customer turns. Fourteen failed; eight of them the
livestream trap.

**What it means.** The guard is right about a second ticket and wrong about a
second payment. Its target — the review's definition of a repeat — leaves the
amount out, so a credit for the shipping after one for the item reads as the
same credit twice (WHAT_FAILED #39). Every payment counted or refused as a
repeat with another amount has been this correction: 1k's two credits and these
three. Run dry with a payment's amount and currency in its target, every
recorded closing phase changes in exactly those calls — 1k's repeats go from six
to four, every other run is unchanged, and the vouchers paid twice in 1n and 1o,
the same amount again, stay refused. That is a new guard, for a run of its own.
Until then the guard is kept where 1o kept it, on the five traps, and not turned
on for the whole suite.

### 1q. Does the guard that reads the amount keep the correction? (API only, no GPU)

1p found the guard's one cost where it was not built for: it compared a payment
without its amount, so the shipping credit P4.6 asks for after a credit for the
item read as the same credit twice, and was refused (WHAT_FAILED #39). Since v41
a payment's amount and currency are part of its target — for the review's count
and the guard alike. The same payment again is still a repeat, and refused; a
payment of another amount on what was already paid is not, and `closing.review`
lists it as `another_amount`, to be read: the rest of what the policy says is
owed, or a second payment it does not ask for. The trace header records what the
guard compares (`closing_guard`: 2), and `--resume` will not mix a run under one
with the other.

Run dry over every recorded closing phase, the change touches exactly the
corrections: 1k's repeats go from six to four, the two shipping credits listed
instead as payments of another amount; 1j, 1m, both arms of 1n and 1o's control
are unchanged — the vouchers paid twice were the same amount again, and stay
refused. So on the second edition the guard refuses what it refused in 1o, and
1o's result stands for it. Where it was not built for, it has to run again: the
same 158 tasks, the same note and customer as 1p, with the guard v41 has:

```bash
set -a; . ./.env; set +a
TASKS=$(python scripts/naming_tasks.py --rest)
python -m pasarbench.sweep \
  --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
  --extra-body '{"thinking":{"type":"disabled"}}' \
  --simulator openai --sim-model qwen3.8-flash \
  --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --sim-extra-body '{"reasoning_effort":"low"}' \
  --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
  --policy-mode preload-named --closing guard --run-id U-guard --workers 8
python scripts/closing_report.py traces/U-guard/full+search-300 > U.txt
python scripts/audit_tool_arms.py traces/U-guard >> U.txt
```

474 episodes. If the sweep reports episodes a server or API error ended, run it
again with `--resume` before reading. `closing_report.py` now also prints every
payment of another amount beside the one it came after.

**The reading, fixed before the run — 1p's, with one line more:**

- **Safe where it was not built for** — the guard can be turned on for the whole
  suite — if no case passed as its customer left it and failed at the end, no
  forbidden call was first made after the note, no refusal, read beside the call
  it would have repeated, was a write the case needed, and no payment of another
  amount, read beside the one it came after, was one the policy does not ask
  for.
- **The correction.** In CCRTOM-ID and its twin a shipping credit after the
  item's should now run; whether the agent makes it at all is the
  conversation's, and is reported either way.
- **Every refusal and every payment of another amount read**, with what the
  agent did next.
- **Rescues** counted, and read with what the customer last said.
- **Costs as 1o counted them:** forbidden calls, guessed digits, a write after a
  refusal that no check asks for.
- **Context, not comparison:** 1p's numbers beside these — the same customer,
  tools and tasks, the guard before v41.
- **Secondary:** tokens; phases cut short; the failures the run still has.

**Run on 2026-10-12, v41** (`U-guard`): safe where it was not built for, by the
reading fixed before the run. Every condition held: no case broken, no forbidden
call, both refusals the same call again, and the one payment of another amount
the shipping credit P4.6 asks for.

| the 158 tasks, 474 episodes | 1q, `U-guard` (guard v2) | 1p, `T-guard` (guard v1), for context |
|---|---|---|
| pass^1 (pass^k) | 0.973 (0.943) | 0.970 (0.930) |
| closing phases (the note given) | 456 | 465 |
| passed as she left it, failed at the end | none | none |
| failed as she left it, passed at the end | 1 | 4 |
| refused | 2: the same escalation, the same message to the seller | 4 calls in 3 episodes, 3 of them the shipping credit |
| a payment of another amount | 1: the shipping credit after the item's | — |
| forbidden calls, guessed digits, writes after a refusal | none | none |
| cut short by the token budget | 27 — 17 before a step, 3 failing as left | 16 — 9, 1 |
| tokens per episode | 68,717 | 68,030 |

**The correction runs.** In CCRTOM-ID and its Indonesian twin, five of the six
conversations credited the item and its shipping together, IDR 737,000. The
sixth credited the item; after the customer left the agent credited the IDR
37,000 of shipping, and the guard let it through as a payment of another amount
(CCRTOM-ID.id__r2). All six ended with what P4.6 says is owed. The two refusals
were the guard working, each the agent's first move after the note: an
escalation of the same order under the same category (DRE-PH__r2), and the same
message to the seller again (LCOW-MY.ms__r0).

**Everything else as before.** One rescue — T14__r1, a duplicate refund
escalated after the customer said goodbye — and no case broken in 456 closing
phases. The token budget cut more phases short than in 1p: 27, 17 of them before
the agent took a step, and 3 failing as their customer left them (CCRTOM-MY__r0,
LCOW-MY.zh-MY__r0, LCOW-MY__r2), rescues the budget may have hidden. Eighteen
episodes ran out of it before the customer left. Thirteen failed, nine of them
the livestream trap.

**What it means.** With the amount in what it compares, the guard has cost
nothing measured on any task. On the five traps it removed every second ticket
and kept the rescues (1o, which the dry run says stands for v41); on the other
158 it broke no case, refused only the same call again, and let the correction
through. The guarded closing check can be turned on for the whole suite. What is
left is not the guard's: on the longest tasks the episode's token budget runs
out before the closing phase begins.

**Room for the closing phase (v49).** Each step's prompt is the whole
conversation by then, so a long case can reach the closing phase with nothing
left of the episode's 120,000 tokens. `--closing-room` bounds the phase by its
own six steps: the episode's steps, tool calls and tokens end with the
conversation, its time does not, and the conversation runs exactly as it did.
Opt-in and recorded — the trace header carries `closing_room`, `--resume` will
not mix runs with and without it, and `compare_cells.py` flags a pair that
differ. To see what the budget hid, 1q again with it:

```bash
set -a; . ./.env; set +a
TASKS=$(python scripts/naming_tasks.py --rest)
python -m pasarbench.sweep \
  --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
  --extra-body '{"thinking":{"type":"disabled"}}' \
  --simulator openai --sim-model qwen3.8-flash \
  --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --sim-extra-body '{"reasoning_effort":"low"}' \
  --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
  --policy-mode preload-named --closing guard --closing-room --run-id U-room --workers 8
python scripts/closing_report.py traces/U-room/full+search-300 > U-room.txt
python scripts/compare_cells.py traces/U-guard/full+search-300 \
  traces/U-room/full+search-300 >> U-room.txt
```

**The reading, fixed before the run** (2026-10-12):

- **The budget.** No closing phase cut short by the episode's steps, tool calls
  or tokens; one cut by its own six steps or by the wall clock is read.
- **Safe, as 1q was:** no case passed as its customer left it and failed at the
  end, no forbidden call first made after the note, and every refusal and every
  payment of another amount read beside the call it followed.
- **What the budget hid.** Every phase that runs here where U-guard's tokens cut
  it is read, and the case it rescues, if it does — U-guard's three cases that
  failed as their customer left them and lost their phase to the budget
  (CCRTOM-MY__r0, LCOW-MY.zh-MY__r0, LCOW-MY__r2) first.
- **Context, not a test of room.** U-guard against U-room, paired: the
  conversations are drawn again, so a difference there is not room's alone.
- **Secondary:** tokens per episode, and the closing phases' own.

**Run on 2026-10-13, v49** (`U-room`, `U-room.txt`): **room enough, and safe;
no rescue found that the budget had hidden.** Every condition held.

| the 158 tasks, 474 episodes | `U-room`, room of its own | `U-guard`, the episode's budget (context) |
|---|---|---|
| pass^1 (pass^k) | 0.970 (0.924) | 0.973 (0.943) |
| the note given; a step taken after it | 469; 469 | 456; 439 |
| closing phases cut short | none | 27, all by the episode's tokens |
| out of budget before the customer left | 5 | 18 |
| failed as she left it, passed at the end | 3 | 1 |
| passed as she left it, failed at the end | none | none |
| refused | 4, each the same call again | 2, the same |
| a payment of another amount | 1, the shipping credit after the item's | 1, the same |
| forbidden calls, guessed digits, writes after a refusal | none | none |
| tokens an episode; the closing phase's own | 67,742; 9,876 | 68,717; 9,031 |

- **The budget.** No closing phase was cut short — not by the episode's steps,
  calls or tokens, nor by its own six steps or the clock — and all 469 took
  their step. Five episodes ran out of tokens before the customer left, four of
  them passing: room for the phase does not reach the conversation.
- **Safe.** No case passed as she left it and failed at the end, and no
  forbidden call. The four refusals, each beside the call it would have
  repeated: the same customs-hold escalation twice (CHE-VN__r1, r2), the same
  identity escalation (IVF-MY__r1), the same message to the seller
  (LCOW-VN.vi__r2). The payment of another amount is the correction again: IDR
  37,000 of shipping after the item's 700,000 (CCRTOM-ID.id__r1), what P4.6
  asks for. One write no check sees: after a Thai customer's goodbye, a message
  to the seller that her hazmat charger was refunded without a return
  (HRWR-TH.th__r2) — true, and harmless.
- **What the budget hid: no rescue found — and little to find.** Of the 27
  phases U-guard's tokens cut, 25 ran here, and not one made a call that wrote:
  23 of those cases passed as the customer left them, this time, and two failed
  and stayed failed (LCOW-MY__r2, which kept the escalation it had made as an
  out-of-window dispute, and LCOW-VN__r2). Of U-guard's three cut cases that had
  failed as she left them, only LCOW-MY__r2 could be read again, and it was not
  rescued; CCRTOM-MY__r0 ran out again, this time before she left, and
  LCOW-MY.zh-MY__r0 passed as she left it, its conversation drawn again. The
  three rescues here — store credit for a cash-on-delivery order
  (CCRTOM-MY.ms__r1), the duplicate refund escalated (T14__r1, r2) — were none
  of them a phase U-guard's tokens had cut.
- **Context.** Paired against U-guard by task: −0.002, 4 tasks better and 5
  worse, p = 1.0, INCONCLUSIVE; as each customer left it, −0.006 (3 and 5,
  p = 0.73) — the conversations moved as much as their endings. Rescues 3
  against 1 (p = 0.5).
- **Secondary.** Tokens 67,742 an episode against 68,717 (paired, p = 0.47);
  the closing phase's own 9,876 on average against 9,031, the longest 55,978.
  Fourteen failures, 13 of them ended by the customer: ten the livestream trap —
  an item misrepresented on a livestream, which P7.2 refunds in full outside
  the window, escalated (eight) or given a voucher (two) — two cash-on-delivery
  refunds, a shipped order she wanted cancelled, and one duplicate refund not
  escalated (T14__r0).

**What it means.** With room of its own the closing phase always ran, and it
was as safe as 1q's; no rescue turned up that the budget had hidden, though with
the conversations drawn again there were few chances for one — two of the 25
phases run again followed a failing case. 1q's reading stands, and the guarded
check with `--closing-room` can be on for the whole suite. What room cannot
give is room for the conversation, which on the longest cases still runs out
first — five times here.

### 1r. Why does the duplicate refund fail outside English? (recorded runs, no API spend)

1j found a lead after its run, and 1m saw it hold on new tasks: the
duplicate-refund trap fails far more often outside English. P10 asks for an
escalation when a customer chases a refund already paid, "even if you have
explained the existing refund and the customer accepts it" — the one escalation
in the policy that goes against the obvious service move. Every run so far
recorded the trap, so the lead can be tested before anything is spent: 928
episodes of it from deepseek-v4-pro, 778 of them in the 16 cells that ran it in
English and in other languages both.

```bash
python scripts/language_lead.py traces > lang.txt
# the seven runs recorded before the lead was found, and never read for it
python scripts/language_lead.py traces/B-multilingual traces/C-clean \
  traces/D-nozh traces/F-clean-sim traces/G-gated traces/K-preload traces/P-ref
```

`language_lead.py` asks of each conversation — the closing phase left out: it
comes after the customer has gone — whether the agent escalated while the
customer was there, and when: in the turn it found the refund (its first
`get_order`, whose result lists the refund already paid), or after the customer
had said more. It compares each local-language task with its English twin — the
same order, the same refund, the same story told in English — in the cells that
ran both; splits English from the rest within each thing that differs before the
decision; asks the same of the policy's three other escalations; and compares
every trap, twin against twin, on the verdicts the runs recorded.

**Read on 2026-10-12 (v43), over every run to 1q:** a finding, for this model
and this one rule.

| escalated while the customer was there | en | sg-en | id | vi | th | zh-SG | ms | zh-MY |
|---|---|---|---|---|---|---|---|---|
| deepseek-v4-pro, the 16 cells | 59% | 62% | 50% | 37% | 35% | 8% | 7% | 2% |
| episodes | 378 | 60 | 60 | 60 | 60 | 50 | 60 | 50 |

- **Twin against twin, in every world.** All 14 local-language tasks escalated
  less often than their English twin in the same cells: a mean gap of 38 points,
  sign test p = 0.0001 — the two Singlish twins too, one barely (74% against
  76%). And in each of two sets the lead did not come from: in the seven runs
  recorded before it was found, never read for it, 7 twins of 7 (21 points,
  p = 0.016); on the second edition's new worlds (1m to 1q), 7 of 7 (47 points,
  p = 0.016). English is ahead in all 16 cells — by under 3 points in the first
  run (B) and 6 in K-preload, where tool search failed and almost nothing was
  escalated; by 14 to 67 in the others.
- **Decided when it reads the refund.** 341 of the 342 escalations came in the
  turn the agent found the refund, before the customer said anything more. Of
  the 437 conversations that did not escalate then, one did later.
- **Not the way there.** Within the most common path to the lookup — one
  customer turn, no identity check first — 57% in English against 21% (394 and
  252 episodes); with two turns or more and an identity check first, 86% against
  31% (28 and 72). Under each of the four customer simulators and both tool
  exposures English is ahead, by 22 to 42 points.
- **This rule, not escalation.** On the policy's three other escalations — a
  customs hold, a failed identity check, a disputed refund out of window — the
  local-language twins escalate as often as their English ones: mean gaps 0, +3
  and −1 points, and 8, 3 and 11 of 14 twins level. Outside English the agent
  does not escalate less; it drops the one escalation that goes against the
  service move.
- **The one rule that stands.** Trap by trap, on the verdicts the runs recorded
  in the cells without a closing phase, the duplicate refund is the only gap
  that stands once sixteen traps are counted: 12 of 14 twins below, none above,
  34 points, p = 0.0005 against a threshold of 0.003 (0.05 over sixteen). The
  next largest below English are 5 to 8 points. Two traps lean the other way,
  their local twins passing more often — the livestream trap, 6 of 7 (12 points,
  p = 0.03), and the identity check (14 points, p = 0.23) — and neither stands.
- **This model.** Qwen3.8-27B, with the same tasks and customer as P-ref
  (P-base), escalated in 95% of English conversations and 87% of the others,
  with no twin pattern (1 below, 2 above, 4 level); deepseek-v4-pro in P-ref,
  62% and 40%.

What it looks like, in Q-off's DRE2-VN and its Vietnamese twin, three
conversations each. In English the agent looks the order up and writes "Since a
refund has already been issued for this order, I need to escalate this for
review" — and escalates, all three times. In Vietnamese it looks up the same
order, lists the same refund — its id, VND 1,157,000, issued 09/11/2026 — asks
the customer to check her card statement, and stops, all three times. She says
she hadn't checked it and thanks the agent, and the case stays failed.

**What it means.** On this rule deepseek-v4-pro follows an English policy far
less often when the conversation is in another language — least in Malay and
Chinese, where it almost never escalates — and it settles that the moment it
reads the refund. The suite's pass rates hide it: it is one trap in sixteen, and
no language differs from English beyond noise overall (1f). The closing check
already recovers most of these after the customer leaves (1j, 1m). What recorded
runs cannot separate is the language from what else differs between twin
conversations — the customer's words, the agent's own earlier replies. That
needs a run that holds the conversation fixed and changes only its language, at
the moment the agent reads the refund.

### 1s. Is the hard edition harder than its parts? (API only, no GPU)

Strong models pass 91–98% of the suite, and their failures sit in a handful of
traps: a table of frontier models would bunch at the top. v48 adds a hard
edition — 36 cases that each put two or three of the policy's rules in one
conversation, where the rule the customer's story points at is not the one that
decides it. Never part of the suite (`generate(edition=3)`, `--suite edition3`).

| case | tasks | her story points at | what decides it | the wrong turns it catches |
|---|---|---|---|---|
| a livestream dress, cash on delivery, refunded already (`DRCL3-*`) | 10: the five COD markets | P7.2 — the seller misrepresented it on his live: window waived, full refund | P10 — refunded already: escalate `duplicate_refund` and take no other write action | the refund (refused: cash on delivery), the store credit after it, no escalation |
| two orders in one chat (`TOVAC3-*`) | 13 | return the cushion covers delivered three weeks ago; cancel the water bottle ordered three days ago | P6: no return, a goodwill voucher, which she accepts; P2.1: cancel the processing order | one remedy without the other; one order's remedy on the other |
| a customer who changes her request (`CRCTA3-*`) | 13 | cancel a processing bookshelf order | with her phone digits, she changes her mind: keep it, send it to her new address (P9.1) | the cancellation she withdrew; an address without its postcode |

The same guarantees as the suite (`tests/test_generated.py`): every case solved
by its reference solution, through the real loop too; none by an agent that does
nothing or one that calls every tool; locale twins identical but for the
opening; dates in causal order; a new customer in every market, new products,
ids of their own. And the wrong turns above fail, in every market, while the
right answer in other words passes. The suite and the second edition are byte
for byte what every recorded run had: their definitions are pinned by digest,
and a change to them fails the tests.

Three things about the checks:

- **An attempted payout counts.** P10 says escalate "and take no other write
  action"; on cash on delivery the refund an agent reaches for can only be
  refused, so the attempt is what a check can see. The same in the other two
  cases: a refund, credit or cancellation tried and refused is forbidden — on
  the changed request, before her digits it breaks P1.1, after them it is the
  request she withdrew.
- **The new address is checked by its postcode** (`verifier.Contains`): the agent
  may write the rest in its own words or language, but not drop the part a
  courier needs, nor make one up.
- **Her change of mind comes with her digits.** Every write needs them (P1.1),
  so it reaches the agent before any write can be made — if the customer says it
  where her persona puts it. `inspect_trace.py` reads each conversation for that
  (section 4b, eight languages); a failure where her digits came without the
  change is hers, as the photo trap's were (WHAT_FAILED #43).

No customs case: P8.3 offers a cancellation P2.2 refuses (WHAT_FAILED #45), and
settling that changes the policy text every edition's agent reads — it waits for
a policy version of its own. No photo case: there the simulator's photos decided
the verdicts (#43).

**The run, per agent: the hard edition and its parts.** Each case's part is the
first edition's task of the trap whose rule it turns on, in the same market and
language (`generate.EDITION3_PARTS`): the duplicate refund, the out-of-window
voucher, the cancellation. With customer 3, as the next editions run — once the
customer check has read her (phase 4, customer 3; read 2026-10-12: she holds);
if that check says to keep customer 2, `--customer 2` on both runs here.

```bash
set -a; . ./.env; set +a
AGENT=(--backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1
       --extra-body '{"thinking":{"type":"disabled"}}')
CUSTOMER=(--simulator openai --sim-model qwen3.8-flash
          --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1
          --sim-extra-body '{"reasoning_effort":"low"}' --customer 3)
python -m pasarbench.sweep "${AGENT[@]}" "${CUSTOMER[@]}" --suite edition3 \
  --strategies full -k 3 --run-id V-dsk --workers 8
python -m pasarbench.sweep "${AGENT[@]}" "${CUSTOMER[@]}" --suite generated \
  --traps duplicate_refund_escalate,out_of_window_offer_voucher,cancel_while_processing \
  --strategies full -k 3 --run-id V-dsk-parts --workers 8
python scripts/hard_vs_parts.py traces/V-dsk-parts/full traces/V-dsk/full > V-dsk.txt
python scripts/inspect_trace.py traces/V-dsk/full --leaks-only >> V-dsk.txt
```

225 episodes an agent: 108 on the cases, 117 on their parts (the three
Singapore duplicate refunds among them have no case — Singapore has no cash on
delivery — and are not paired). Another agent runs
the same two commands with its own `AGENT=(...)` and run ids — `V-<agent>` and
`V-<agent>-parts`: V is the next letter no run has (v48 named them H-, which is
the context ablation's) — against the same
customer; the sweep refuses an agent of the customer's family (a Qwen agent here
needs another customer, and then every agent compared with it runs with that
one). The Qwen3.8-27B base model and its fine-tunes run the hard edition on the
GPU, with the Gemma customer (phase 4, the hard edition): beside these, with
that difference stated.

**Another lab's model, one command** (v55; v56):

```bash
python scripts/add_model.py
```

It asks for the provider (OpenAI, Google, Anthropic, Mistral, xAI, Z.ai,
Xiaomi MiMo, OpenRouter, Together, Groq, DeepSeek, or any OpenAI-compatible
endpoint), the model's name and its API key. The key is typed at a hidden
prompt — or taken from the provider's own variable, in `.env` or the shell,
under any of the names it goes by (`ZAI_API_KEY` or `GLM_API_KEY` for Z.ai,
`MIMO_API_KEY` for Xiaomi, `XAI_API_KEY` or `GROK_API_KEY` for xAI;
`pasarbench.sweep.PROVIDER_KEYS`), if you say so — never shown and never put on
a command line: the two runs get it in their environment with every other
`*_API_KEY` taken out, so it goes to that provider and nowhere else
(WHAT_FAILED #47). The customer's key is `PASARBENCH_SIM_API_KEY` or
`DASHSCOPE_API_KEY`, as the sweep reads it, or typed. Before the runs the model
looks an order up with its 20 tools and answers from what
the tool returned — two short requests, the loop every step of a run is — so a
wrong key or model name, a model that takes no tools or no temperature 0, or a
setting the provider refuses stops it there, nothing spent. Then the two sweeps
above, as `V-<name>` and `V-<name>-parts`, against customer 3, and
`V-<name>.txt`: the two commands' reading, and this model against every model
already run, on the cases and on the parts — the separation line below. A run
that stopped, or left conversations an API error ended, is finished by the same
command again; no reading is written until none is left. It refuses a key that
is another provider's variable's, a Qwen model, a retired model its provider
still answers for with another one (xAI's fast models since 2026-05-15, served
by Grok 4.3), and a name that holds another model's runs. `--provider`,
`--model` and `--name` skip the questions; `--dry-run` shows the plan and sends
nothing; `--workers` sets how many conversations run at once (8), fewer for a
provider that limits how many requests an account has open. Each run keeps its
own record, `logs/add_model-<name>.log` (v59): what ran, what the check requests
met — in the provider's own words when they fail — how each sweep ended, and the
reading; no key is in it.

Every model runs at temperature 0 with reasoning off where its provider has a
switch: DeepSeek's and MiMo's `thinking: disabled`, Grok 4.3's
`reasoning_effort: none`. GLM-5.3-Flash has none — it always reasons — so it
runs at `reasoning_effort: low` with 4,096 tokens a step (the others 2,048),
and its line in any table says so. To xAI each request carries one
`x-grok-conv-id` a conversation, so that its steps reach the server that cached
the steps before them (v56). To Z.ai a tool call's empty text goes as `""` and a
tool result without its tool's name, as its schema documents them (v59); every
other provider is sent what it always was.

**Which keys work, which models** (v56; v57), before any run:

```bash
python scripts/add_model.py --check
```

For every provider whose key is set: the models the key lists, and its newest
fast models (`CANDIDATES` in the script, and up to three more of that tier the
key lists) each put through the same two requests, with their tokens and what
the provider's cache served of the second; and the customer's key, found as the
sweep finds it (`PASARBENCH_SIM_API_KEY`, else `DASHSCOPE_API_KEY`). It writes
`models_check.txt`, which names the variables and holds no key. The script
reads `.env` itself (v57), as `set -a; . ./.env` would, keeping what the shell
has set already. When a provider of `CANDIDATES` has no key, it says by name
only what `.env` sets and which of its lines the shell does not read — a space
round the `=`, a line that is not `NAME=value`, an empty value, a name nothing
reads — then asks for the key at a hidden prompt (Enter skips), tries it, and
if a request with it worked offers to save it to `.env` under the name the runs
read, in place of a misspelt line; `.env` is left readable by you alone. A key
that is another provider's variable's is refused before it is sent.

**What one model's 1s costs** — list prices read 2026-10-13; token counts from
DeepSeek's runs of the parts, the cases' by Qwen's ratio of the cases to their
parts. An agent's 225 conversations send about 7.8M prompt tokens in about 1,600
requests and get about 0.13M back, more for a model that reasons. Each request
repeats the one before it and adds a step: 97.6% of DeepSeek's prompt tokens
came from its cache. Another model's tokenizer counts the same text
differently, and another model takes more steps or fewer. The customer,
Qwen3.8-Flash, is well under $0.30 a model.

| agent | $ per 1M tokens: input / cached / output | 1s, nine in ten prompt tokens cached | none cached |
|---|---|---|---|
| GLM-5.3-Flash (reasons; taken as 100–600 tokens a request) | 0.15 / 0.03 / 0.50 | $0.5–0.9 | $1.3–1.7 |
| GLM-5.3-FlashX (the same, faster) | 0.37 / 0.075 / 1.25 | $1.1–2.2 | $3.2–4.3 |
| MiMo-V2.6-Flash, reasoning off | 0.14 / ~0.003 / 0.28 | $0.16 | $1.1 |
| Grok 4.3, reasoning none | 1.25 / 0.20 / 2.50 | $2.7 | $10 |
| DeepSeek-V4.1-Flash (`deepseek-flash`), thinking off | 0.30 / 0.006 / 1.20 | $0.43 | $2.5 |
| DeepSeek-V4-Pro-0813 (`deepseek-v4-pro`), thinking off | 1.32 / 0.044 / 3.96 | $1.8 | $10.8 |

DeepSeek's prices are its peak ones, 01:00–04:00 and 06:00–10:00 UTC on
weekdays; at any other time they are half. GLM-4.7-Flash is free, at the free
tier's rate limits. xAI's fast models were retired on 2026-05-15, and Grok 4.7
Fast is not on its public API: Grok 4.3 is its cheapest general model, and the
one xAI names in their place.

**The agents, chosen 2026-10-13 before any ran.** Each lab's newest model of its
fast tier, and the baseline every earlier run used; read that day on Artificial
Analysis: its Intelligence Index, and where it has them its two tests nearest
this one — an agent with tools and a simulated customer, τ²-Bench Telecom and
τ³-Banking, both retired there before the newest models came, so they have
none. Each figure is for the setting it was measured with, which is not always
the one these runs use:

| run | model | its settings here | Artificial Analysis, as it measured | why this one |
|---|---|---|---|---|
| `V-dsk` | `deepseek-v4-pro` (DeepSeek-V4-Pro-0813) | thinking off | 20.4, thinking off | the agent of every earlier run |
| `V-dsk-flash` | `deepseek-flash` (DeepSeek-V4.1-Flash) | thinking off | 24.7, thinking off | DeepSeek's fast tier: above the baseline, at twice its speed and a third of its price |
| `V-mimo` | `mimo-v2.6-flash` | thinking off | 37.9, reasoning on | Xiaomi's newest Flash; its Pro (46.3, as fast, three times the price) is the pro tier |
| `V-grok` | `grok-4.3` | `reasoning_effort: none` | 14.0, reasoning off; τ²-Telecom 0.66 | the model xAI names in place of its retired fast ones; Grok 4.20 without reasoning scores the same (14.6) |
| `V-glm` | `glm-5.3-flash` | `reasoning_effort: low`, 4,096 tokens a step | 41.8, at Z.ai's default effort (max); τ³-Banking 0.47 | Z.ai's newest fast model, within 3 points of its flagship; it always reasons |

MiMo and GLM were measured with more reasoning than they have here: expect them
lower. Five agents, ten pairs for the separation reading below; each run's
reading pairs it with the agents finished before it, and the rest are paired
when the five are read.

```bash
python scripts/add_model.py --provider deepseek --model deepseek-v4-pro --name dsk
python scripts/add_model.py --provider deepseek --model deepseek-flash --name dsk-flash
python scripts/add_model.py --provider mimo --model mimo-v2.6-flash --name mimo
python scripts/add_model.py --provider xai --model grok-4.3 --name grok
python scripts/add_model.py --provider zai --model glm-5.3-flash --name glm
```

**The reading, fixed before the run** (2026-10-12, before any agent ran a case):

- **Harder (primary), per agent.** `hard_vs_parts.py`: each case against its
  part, pass^1 paired, sign test. HARDER at p < 0.05 with the cases failing
  more. Anything else is reported as it comes — "not shown harder for this
  agent" — and the README's line on the hard edition says so.
- **Where it is hard.** The per-case table beside it, and every failure read and
  named by its wrong turn: on the refunded order, no escalation, a refund tried,
  or store credit; on the two orders, a remedy missing or one order's remedy on
  the other; on the changed request, the cancellation, no address change, or an
  address without its postcode. A failure that is none of these is described as
  it is.
- **The customer.** Section 4b of `inspect_trace.py`: each changed-request
  failure where her digits came without the change is listed and not counted
  against the agent, and if there are more than two such episodes for an agent,
  that case's number for that agent is the customer's, reported as such.
- **The voucher choice** (added 2026-10-12, after GRPO's results and before any
  agent ran a case; WHAT_FAILED #46). On the two orders, and on their part, an
  escalation of the out-of-window order after she declined the voucher is what
  P6.3 and P10 ask for, and the check fails it. `read_failures.py` flags those
  episodes and `compare_cells.py` counts them; they are listed and not counted
  against the agent, and `hard_vs_parts.py` prints the paired line read as the
  policy reads them beside the result — never instead of it. If more than two
  of an agent's two-orders failures are hers, that case's number for that agent
  is the customer's, reported as such.
- **Separation, with two agents or more.** `compare_cells.py` on each pair of
  agents, on the cases and on the parts: the hard edition separates two agents
  the parts do not if their paired comparison resolves on the cases (p < 0.05)
  and not on the parts.
- **Secondary.** The refunded livestream order by language, twin against twin:
  1r's lead — deepseek-v4-pro drops the duplicate-refund escalation outside
  English — on a case whose story is not the refund. A lead to read, not a test.
- **Not a verdict on the suite.** A case failing more than its part says the
  combination is harder for that agent, not that the suite was easy everywhere.

**Run on 2026-10-13, v57–v59** (`V-dsk`, `V-dsk-flash`, `V-mimo`, `V-grok` and
`V-glm`, each with its `-parts`; RESULTS.md §8): **not shown harder for any of
the five agents.** One customer, customer 3, and temperature 0 in every run;
225 conversations an agent, none left that an API error ended.

| agent | hard cases, pass^1 (pass^k) | their parts | each case against its part |
|---|---|---|---|
| DeepSeek-V4-Pro (`dsk`), thinking off | 0.944 (0.89) | 0.833 (0.75) | EASIER — +0.111; 1 case failed more than its part, 8 less, p = 0.039 |
| DeepSeek-V4.1-Flash (`dsk-flash`), thinking off | 0.991 (0.97) | 1.000 (1.00) | INCONCLUSIVE — 1 more, 0 less, p = 1.0 |
| MiMo-V2.6-Flash (`mimo`), thinking off | 0.991 (0.97) | 0.991 (0.97) | INCONCLUSIVE — 1 more, 1 less, p = 1.0 |
| Grok 4.3 (`grok`), reasoning none | 0.685 (0.58) | 0.731 (0.72) | INCONCLUSIVE — 6 more, 2 less, p = 0.29 |
| GLM-5.3-Flash (`glm`), reasoning low | 0.954 (0.89) | 0.944 (0.86) | INCONCLUSIVE — 3 more, 4 less, p = 1.0 |

Each case's passes against its part's (of 30 or 39 conversations):

| case (its part) | dsk | dsk-flash | mimo | grok | glm |
|---|---|---|---|---|---|
| the livestream dress, refunded already (the duplicate refund) | 25 · 12 | 29 · 30 | 29 · 30 | 2 · 1 | 25 · 24 |
| two orders in one chat (the out-of-window voucher) | 38 · 39 | 39 · 39 | 39 · 38 | 33 · 39 | 39 · 39 |
| the changed request (the cancellation while processing) | 39 · 39 | 39 · 39 | 39 · 39 | 39 · 39 | 39 · 39 |

**Where it is hard: every failure on the cases, read and named by its wrong
turn.**

- **The refunded livestream order** (40 failures): no escalation — DeepSeek-V4-Pro
  5 (Malay 3, Thai 1, English in Thailand 1), Grok 27, GLM 2; store credit in its
  place — V4.1-Flash 1 (Thai), Grok 1; a refund tried, refused on cash on
  delivery, then store credit — MiMo 1 (Thai). And GLM 3: escalated, but as an
  out-of-window dispute, without the order looked up and so without the refund
  seen — her story, the seller's livestream and the window, decided it, one of
  the three with a goodwill voucher beside it: the wrong turn the case was built
  for.
- **Two orders** (7 failures). The customer's, twice, both in Malaysia's
  Chinese: offered the voucher, she accepted it and ended the conversation in
  the same message, before it could be issued — customer 3 should wait for what
  she agreed to (WHAT_FAILED #38); that is DeepSeek-V4-Pro's only failure here,
  and one of Grok's. Grok's own: vouchers over the SGD 15 cap — SGD 150 to an
  Indonesian customer, then IDR 230,000; VND 300,000 and 400,000 — refused by the
  tool, three conversations, two of them escalated at her request after; and
  once photos asked for, as for a return, on the out-of-window covers, round 2's
  anatomy. And the task's, once: Grok read her the account's phone, "ends in
  5000", and refused the digits she gave, 4661, which are the ones that verify
  her — in every generated task the profile's phone and the verifying digits
  disagree (WHAT_FAILED #48).
- **The changed request:** no failure, in any agent.

**The customer.** Her digits came with her change of mind in every
changed-request conversation but one of DeepSeek-V4-Pro's, which passed ("I
don't wanna cancel anymore", words the reader lacks). She ended on her yes to
the voucher twice, above.

**The voucher choice.** She declined the voucher in no conversation of any
agent. The reader flagged two of Grok's; read, she had accepted it, and the tool
had refused Grok's vouchers over the cap. A voucher the agent tried, refused or
not, now counts as hers taken (v60, WHAT_FAILED #49); the two are the agent's.

**Separation.** The hard edition separates no pair of agents the parts do not.
The parts separate DeepSeek-V4-Pro from V4.1-Flash, MiMo and GLM (p ≤ 0.006),
which the cases do not: the duplicate refund, which it fails 18 times in 30 as a
part and 5 in 30 inside the livestream story. Grok is separated from the other
four on both.

**Secondary.** DeepSeek-V4-Pro on the refunded livestream order: English 14 of
15, the other languages 11 of 15, Malay 0 of 3; on its part, English 10 of 21,
the others 5 of 18 — 1r's lead, on a case whose story is not the refund. A lead
to read, not a test.

**What the runs cost** (each agent's tokens at list prices; the customer, well
under $0.30 a model): DeepSeek-V4-Pro $1.16, 97% of its prompt tokens from the
cache, and half that, off-peak, when it ran; V4.1-Flash $0.40 (94%); MiMo $0.16
(93%); Grok $3.97 (76%, its routing id notwithstanding); GLM $0.35 (90%, its
low reasoning and its replies 87 tokens a step together).

**What it means.** For these five agents the hard edition is not harder than
its parts, and it separates no two of them the parts do not. Where they fail,
they fail the rule the parts already test — the duplicate refund — and
DeepSeek-V4-Pro fails it less often inside the livestream story than alone. The
two cases that held round 2's fine-tuned models hold none of these but Grok,
whose failures there are vouchers over the cap, the customer's early goodbye and
the task's own contradiction. The edition gave round 2's 27B models room; for
API models it measures the duplicate refund again. An edition that separates
API models needs combinations that defeat them, built and read the same way
before they run. Two instruments failed on the way and are named: the voucher
choice's reader (#49, fixed) and the generated profile's phone (#48, recorded
for the next edition; the editions are pinned as every run had them).

### 1t. Is it the language? The same conversation, translated (API only, no GPU)

1r found that deepseek-v4-pro makes P10's escalation of a refund already paid in
59% of English conversations and 2–8% of Malay and Chinese ones, that every
local-language twin made it less often than its English twin, and that it
decides in the turn it reads the refund. What recorded runs cannot separate is
the language from everything else that differs between twin conversations: each
language's customer wrote her own words, and the agent answered them with its
own. v49 holds a conversation fixed at that moment and changes only its language
(`pasarbench/swap.py`, `scripts/language_swap.py`).

**The prefixes.** Every recorded duplicate-refund conversation of
deepseek-v4-pro that can be rebuilt exactly, up to and including its first
lookup of the order, whose result lists the refund: the system prompt, her turns,
its replies and calls, every payload replayed and checked against the digest the
run recorded, and the world as those calls left it. In the recorded runs, 449 can
be — runs from v19 on (earlier ones record no digests), under the full context
and no claim guardrail; the Philippines has no other language. Eight are chosen
from each of 12 groups — each language other than English, recorded in it and
swapped into English, or recorded in English and swapped into it — spread over
the group's tasks and never by what the agent did: 96. Checked before any spend
(`language_swap.py check`, no API): continued from each of the 96, the agent's
first call is shown exactly as many messages as the recording's next step was,
and exactly its tools — under tool search too, rebuilt from the searches before
the lookup.

**The arms.** `orig`, the conversation as recorded; `swap`, every turn of hers
and reply of its translated into the other language; `back`, translated there
and back. `swap` and `back` have both been through the translator and differ only
in their language, so they are the test; `orig` against `back` is what the
translator alone does. The system prompt and the policy (in English, as always),
the tools, their payloads and the world are the same in all three. The
translator is qwen3.8-flash — the customer's model, not the agent's family — at
temperature 0, behind one cache, so every arm reads the same translation of each
text. Within a direction the arms differ in translation steps too (`back` has
two, `swap` one), and the two directions differ the other way: pooled, the steps
balance, and the reading below asks for the same sign in each direction alone.

**The decision.** The agent continues from the prefix at the recording's
temperature and request flags, under its tool exposure, until it next speaks to
the customer — no customer reply and no closing phase: 341 of the 342
escalations 1r counted came in that turn. Escalated: `escalate_to_human` called
and run in it.

```bash
set -a; . ./.env; set +a
python scripts/language_swap.py prefixes traces --out data/swap/W-prefixes.jsonl
python scripts/language_swap.py check data/swap/W-prefixes.jsonl     # no API: 96, 0 shown otherwise
python scripts/language_swap.py run data/swap/W-prefixes.jsonl --run-id W-swap --dry-run
python scripts/language_swap.py run data/swap/W-prefixes.jsonl --run-id W-swap
python scripts/language_swap.py report data/swap/W-swap > W.txt
```

864 continuations (96 prefixes, three arms, k = 3) — about 5 million prompt
tokens of deepseek-v4-pro — and 666 translations, each once. The keys are the
sweep's: the agent's from `DEEPSEEK_API_KEY`, the translator's from
`DASHSCOPE_API_KEY`, and the one is never sent to the other's provider. If the
run stops, run it again: the cache keeps every translation, and continuations
already written are not run again. The traces go under `data/swap/`, not
`traces/`: they are not conversations of the suite, and no reader of `traces/`
should count them. Read `data/swap/W-swap/arms.md` first — every prefix's words
in all three arms, side by side; Thai and Vietnamese above all, whose suite
translations no native speaker has reviewed either.

**The reading, fixed before the run** (2026-10-12, before any arm ran):

- **First, the reproduction.** The `orig` arm must agree with the recording, by
  the majority of its three runs, on at least 75% of the prefixes. If it does
  not, either the prefixes are not what the agent saw or its decision is too
  noisy at temperature 0 to test this way: nothing else is read, and that is
  the result.
- **The test (primary).** English or Singlish against the other language, both
  translated, escalation paired by prefix, sign test over the 96. LANGUAGE if
  English escalates more at p < 0.05 and the difference has the same sign in
  both directions, each tested apart (their p-values reported, not required to
  resolve). A difference in one direction only is the translator's direction,
  not the language, and is reported as that.
- **The translator.** `orig` against `back`, paired: INCONCLUSIVE is expected. If
  it resolves, translation alone moves the decision, and the test is reported
  with that beside it.
- **If not the language** (INCONCLUSIVE): the gap 1r found lives in what differs
  between twin conversations — her words, the agent's earlier replies — and not
  in the language it reads the refund in. A run that swaps those one at a time
  is the next one.
- **Secondary.** By language; with the category the policy names
  (`duplicate_refund`).
- **Not tested.** The policy's own language, which stays English in every arm;
  any agent but deepseek-v4-pro.

**Run on 2026-10-13, v49** (`data/swap/W-swap`, `W.txt`): **LANGUAGE — English
higher, in both directions.** The reproduction held, and the test resolved.

| escalated before its next reply (3 runs a prefix) | `orig` | `swap` | `back` |
|---|---|---|---|
| recorded in English or Singlish (48 prefixes) | 65/144, 45% | 44/144, 31%, in the other language | 72/144, 50% |
| recorded in another language (48) | 23/144, 16% | 50/144, 35%, in English | 23/144, 16% |
| all 96 | 88/288, 31% | 94/288, 33% | 95/288, 33% |

- **The reproduction.** The `orig` arm agrees with the recording, by the
  majority of its three runs, on 75 of the 96 prefixes — 78%, against the 75%
  asked. The recording had escalated in that turn in 27 of them.
- **The test.** The same conversation in English or Singlish against the other
  language, both translated: +0.191, 47 prefixes higher in English and 16 lower,
  p = 0.0001 — **LANGUAGE**. Each direction alone has the same sign and resolves
  too: recorded in English, +0.194 (25 higher, 10 lower, p = 0.017); recorded in
  another language, +0.188 (22 and 6, p = 0.004).
- **The translator.** `orig` against `back`: +0.024, 25 and 18, p = 0.36 —
  INCONCLUSIVE, as expected: no sign that translation alone moves the
  decision.
- **By language** (secondary, not tested: English against it, 16 prefixes
  each): Malay 40% against 6%, Malaysia's Chinese 33% against 10%,
  Singapore's 52% against 15%, Vietnamese 65% against 46%; Thai 25% against 21%
  and Indonesian 40% against 42%, level.
- **The category** (secondary): the numbers are the same under the category
  the policy names, because every escalation that ran was a `duplicate_refund`.
  Every continuation that tried to escalate ended with one that ran: in 173 of
  the 277 the tool first refused a try — nearly always for a category it does
  not have, `duplicate_refund_claim` — and the agent tried again.
- **The translations.** Every order number, amount, date and digit in the
  recorded turns, 157 of them, is in both translated arms; the Thai and
  Vietnamese turns read in `arms.md` keep their request.
- **Their language, read after** (v65, from the cache; WHAT_FAILED #51): 6 of
  the 374 translated messages are not in their arm's language, in 4 of the 96
  prefixes. Three are Singapore's, asked for Chinese heavily mixed with English:
  an agent's reply left wholly in English in the Chinese arm, or Singlish
  translated back into English that stayed in mixed Chinese — each makes the
  two arms more alike, so it can have narrowed the difference, not made it.
  One is a Malaysian Chinese reply put into Malay: still not English.

**What the table shows beyond the test** (read, not tested). Moving a
conversation into English raised its escalations from 16% to 35%; moving one out
of English lowered them from 50% to 31% — 19 points each way. The rest of the
34 points between the two kinds of recording, each in its own language
(`back`), stayed with the conversation: in either language, the conversations
first written in English escalate 15 points more (50% against 35% in English,
31% against 16% in the other language). A little over half of the gap between
these recordings is the language the agent reads the refund in; the rest is
what else differs between twin conversations — her words, its earlier replies
— which swapping each alone would separate.

**What it cost.** 864 continuations, 1,708 steps: 6.7 million prompt tokens of
deepseek-v4-pro, 93% of them from the cache, and 0.22 million completion —
about $0.90 at the off-peak rate it ran in; 374 translations of qwen3.8-flash,
each made once.

**What it means.** Held at the moment it reads a refund already paid, with
the refund, the tools and the world unchanged, deepseek-v4-pro makes P10's
escalation more often when the conversation is in English, whichever language
it was first held in — by language, untested, in Malay, Chinese and
Vietnamese, and not in Thai or Indonesian. The policy it reads is English in
every arm, so whether a policy in
her language closes the gap is still open, and so is whether any other agent
shows it. The runs that would say: deepseek-v4-pro on the same 96 under a
policy in the conversation's language (1v), and other agents from their own
recorded conversations — 1s recorded the duplicate refund for four more; 1w
runs three of them.

### 1u. Can a model read the claims the English list cannot? (API only, no GPU)

The claim guardrail (1i) holds back a reply that says the agent did something no
tool call did. It reads the reply with a list of English phrasings: blind to the
other six languages and to every phrasing nobody wrote, so its counts are a
floor — and in the recorded runs it has fired almost only in English. v50 puts a
model where the list was, and only there (`pasarbench/claim_reader.py`,
`--guardrail reader`): the model names the actions a reply claims, each with the
reply's own words; whether a claim is backed stays mechanical — a successful
call of its tool in this conversation, exactly as `claims` decides it. What the
model may say is narrow: eight actions from a fixed list, a quote that must be in
the reply or the claim is dropped and counted, and a prompt — fixed now — that
says what is not a claim: an offer, a plan, a refusal, and anything that happened
before the conversation. What it read of every reply is recorded on the step,
held back or not — with `reader_issue` where its answer was not the JSON asked
for (the reply goes out unread) or dropped a claim — and its model in the trace
header. One reader everywhere: made by one function (`make_backend`: temperature
0, 1,024 tokens), its answers in one cache (`data/claims/reader-cache.json`,
keyed by model, prompt, settings and text), so the calibration, the live run and
the reading of it give one reply one answer. A reader whose server fails ends the
episode as a server error, which `--resume` runs again.

**First, calibration on recorded replies** — nothing live until it passes:

```bash
set -a; . ./.env; set +a
python scripts/claim_reader_check.py sample traces --out data/claims/C-sample.jsonl
python scripts/claim_reader_check.py read data/claims/C-sample.jsonl
python scripts/claim_reader_check.py report data/claims/C-sample.read.jsonl > C.txt
```

Of the 66,556 replies in the recorded runs, 619: every one the English list
holds back (150), 40 of each language at random (320), and 20 of each language's
duplicate-refund replies written after the agent looked the order up (160) —
the replies that describe a refund paid before the conversation. 619 calls of
qwen3.8-flash on the DashScope key, each cached. Read `C.txt` in its order:

**The reading, fixed before the read** (2026-10-12):

- **The quotes.** At most 5% of the reader's claims dropped for a quote not in
  the reply or an action not on the list. More, and it invents: it is not used.
- **The refund already paid.** Every alarm on a payment in the duplicate
  refund's replies after the lookup is read. One about the refund paid before
  the conversation is enough to stop the reader: held back there, a reply sends
  the agent towards paying it again — the reason the English list reads refunds
  only in the first person.
- **English, against the list.** Every alarm of either is read. The reader takes
  the list's place only if it misses none of the list's alarms the reading
  confirms as claims. Its own extra alarms are read and counted, as claims the
  list missed or as false alarms — its cost.
- **Other languages.** Every alarm read and counted by language, as found claims
  and false alarms: there is no other instrument there to grade it against.
- **The decision.** Live only if the quotes and the refund hold and English
  misses nothing. Otherwise the list stays, and the report says which failed.

**Then, and only then, the live run:** 1i's run with the reader in place of the
list, behind search without the names — where the agent claims escalations it
never makes — on 1h's 57 tasks, 31 of them outside English, with a control the
same day (342 episodes):

```bash
TASKS=$(python scripts/naming_tasks.py)
for G in off reader; do
  python -m pasarbench.sweep \
    --backend openai --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --simulator openai --sim-model qwen3.8-flash \
    --sim-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
    --sim-extra-body '{"reasoning_effort":"low"}' \
    --suite all --tasks "$TASKS" --strategies full --exposure search-300 -k 3 \
    --guardrail $G --run-id X-$G --workers 8
done
python scripts/compare_cells.py traces/X-off/full+search-300 traces/X-reader/full+search-300 > X.txt
python scripts/claim_reader_check.py delivered traces/X-off/full+search-300 \
  traces/X-reader/full+search-300 >> X.txt
```

- **Primary.** Every reply the customer saw that the reader reads as a claim no
  call backed, in both arms, each read by a person (`delivered`: the guarded arm
  as the run read it, the control read now, and in each the answers the reader
  could not give, counted). The reader works if the guarded arm's customers saw
  none the reading confirms, but replies let through after two notes. Not graded by its own count alone:
  graded by that, a guard passes by construction (WHAT_FAILED #34).
- **Read every held reply.** A false alarm that leads to a wrong action — a
  guarded episode failing where the control passed, on a forbidden write — is
  its cost, reported. A held refund claim in the duplicate refund is read first.
- **Secondary:** the paired pass rate and tokens, and the held replies by
  language: how many of them the English list could not have read.

**Read on 2026-10-13, v50** (`data/claims/C-sample.read.jsonl`, `C.txt`): **not
used — the list stays, and the live run is not run.** The quotes and the refund
already paid failed; English held.

- **The quotes: failed, as read.** 626 replies — drawn from the runs as they
  stood, 1s's among them, where 619 was the count when the reading was fixed —
  and 308 claims: 261 kept, 47 dropped, 15.3%. Every answer was the JSON asked
  for.
- **The refund already paid: failed.** Of 167 duplicate-refund replies written
  after the lookup, the reader held back 4 for a payment no call made, and every
  one is the refund paid before the conversation, every one outside English:
  "refund sudah diterbitkan ke kartu Anda" (Indonesian), "退款已经全额发出到您的银行卡了"
  (Malaysia's Chinese), a Vietnamese reply asking her to check her card
  statement, and "Refund MYR 200.00 itu memang dalam proses ke kad anda" (Malay,
  MiMo's). One was enough.
- **English: held.** Of 255 replies in English or Singlish, both held back 132,
  for the same actions. The reader held back none the list did not, and the list
  4 the reader did not — each, read, an offer or a plan ("I'll issue the
  goodwill voucher for you now. Let me first verify your identity", "If it's the
  refund, I'll escalate it now"), not a claim. It missed nothing the list
  caught, and found nothing the list missed.
- **The other languages** (371 replies): 25 alarms. 19 are English sentences in
  a Malay or a Singapore-Chinese conversation — round 2's fine-tunes answering a
  Malay customer in English, deepseek-v4-pro a Chinese-speaking one in Singapore
  — claims of an escalation never made, which the list holds back too. One is a
  claim only the reader could read: "Kasus Anda sudah saya eskalasi", and no
  escalation made. Five are false alarms: the four refunds paid before, and a
  Vietnamese promise to pass her on ("tôi sẽ chuyển…", I will transfer). Outside
  English it found one claim the list could not, for five false alarms, and
  missed none the list caught: its two others there, read, are offers.

**After the read: the quotes failed on the instrument, the refund on the
reader.** 45 of the 47 dropped quotes are in their reply as the customer reads
it. 43 are the reply's words without its markdown — "(case ESC0001)" from "(case
**ESC0001**)", "Refund issued: SGD 59.00 back to your original card" from
"**Refund issued**: SGD 59.00 back to your original card" — which the quote
check compared character for character. Four are Thai written by Qwen3.8-27B,
whose Thai doubles, drops and misplaces its vowel and tone marks: two match once
those marks are folded out, as the leak audit reads Thai; in the other two the
reader wrote a garbled word as another word, and by the reading's own rule those
two are quotes not in the reply. None names a claim the reply does not make: the
15.3% measured the check more than the reader
([WHAT_FAILED #50](WHAT_FAILED.md)). Under quote check 2 (v61) the same answers,
read again from the cache without a call, drop those 2 of 308 (0.6%) — a check
written after reading these 47, so the 0.6% describes them and does not test it
— and hold back a fifth refund paid before ("Khoản hoàn VND 1,157,000 đã được
phát hành ngày 09/11/2026", the refund of 9 November), which the old check had
dropped. On the refund alone the decision is the same. To read it, at no cost
(the `read` will not write over 1u's own reading):

```bash
set -a; . ./.env; set +a
python scripts/claim_reader_check.py read data/claims/C-sample.jsonl \
  --out data/claims/C-sample.q2.read.jsonl                  # cached answers: no call
python scripts/claim_reader_check.py report data/claims/C-sample.q2.read.jsonl > C-q2.txt
```

**What it means.** By the reading fixed before the read, the reader is not
used: the English list stays, and 1u's live run is not run. The reader quotes
what it reads, once the check reads the reply as the customer sees it, and
reads English as well as the list. But it took the refund paid before the
conversation for a payment the agent made — the one mistake the prompt named
— in four languages and never in English: where it was meant to add something,
it raised five false alarms for one claim found (six under quote check 2). A reader tried again needs a
prompt that keeps that refund out, and a new prompt is a new reader, calibrated
again from the start.

### 1v. Does a policy in her language close the gap? (API only, no GPU)

1t found that the language moves deepseek-v4-pro's decision: held at the
moment it reads a refund already paid, the same conversation gets P10's
escalation more often in English, in both directions. In every arm the policy
it read was English. v62 asks whether that is why — the same 96 prefixes, the
conversation in the other language, the policy in English or in that language
(`language_swap.py run --policy`).

**The arms.** `local`: the conversation in the language other than English —
1t's translated words, its `swap` for a conversation recorded in English and
its `back` for one recorded in the other language — under the policy in
English, which is 1t's arm in that language again. `policy`: the same
conversation under the policy in its language, in the system prompt in place of
the English (the instructions around it stay as they were) and from
`search_policy`, which finds what it always finds in the English and returns
those sections translated — and a query in her language, which finds nothing in
the English, is looked for in the translation (`tools.policy_shown_as`).
`english`: the conversation in English, the policy in English, 1t's other arm
again. `local` and `policy` differ in the policy's words and nothing else;
`english` against `local` is 1t's test on another day. A prefix whose policy
its prompt cannot change — no policy in the prompt, or one already read from
`search_policy` before the lookup, in English — stops the run before anything
is spent; none of the 96 is one.

**The translation.** The policy is translated by the conversation's translator,
qwen3.8-flash at temperature 0 behind the same cache, in pieces — its preamble,
then P1 to P11 — into each language the prefixes need, with the tools named in
it where the prefix's prompt named them (65 of the 96); the prompt is part of
the cache's key, so a reworded prompt is a new translation. Before any
continuation runs, every piece is checked: every text in backticks (the
statuses, categories and tools the rules name) and every section number, even
where Chinese or Thai touches it; every number, as often as the English has it;
and as many headings, list items and bold as the English — so a rule left out
or a **not** dropped stops it too. Nothing runs until every piece passes.
`policy` does that alone, before any agent's spend, and writes each translation
to `data/swap/V-policy/policy-translated/`, with P10 in English and every
language side by side in `P10.md`. (v65 checks each piece's language too, and
lets a word stressed in capitals be stressed in bold: below.)

```bash
set -a; . ./.env; set +a
python scripts/language_swap.py policy data/swap/W-prefixes.jsonl --run-id V-policy
python scripts/language_swap.py run data/swap/W-prefixes.jsonl --policy --run-id V-policy --dry-run
python scripts/language_swap.py run data/swap/W-prefixes.jsonl --policy --run-id V-policy
python scripts/language_swap.py report data/swap/V-policy > V.txt
```

864 continuations (96 prefixes, three arms, k = 3), as many as 1t: about 7
million prompt tokens of deepseek-v4-pro, the policy arm's prompt longer in
Thai — about $1 off-peak and $2 at peak — and some 110 pieces of policy
translated, each once, for cents. The conversations' translations are 1t's,
already in the cache. If `policy` names a piece that lost something, nothing
runs: a new translation — a reworded prompt — is decided before any
continuation does.

**The reading, fixed before the run** (2026-10-13, before the policy was
translated):

- **First, the translation.** Every piece keeps what it must, or the run does
  not start. Then, before the run, P10 is read in every language (`P10.md`):
  its rule on a refund already paid — escalate, even when she accepts it —
  must say what the English says, or the run waits for a new translation. The
  rest are read beside the result.
- **The replication.** `english` against `local`, paired by prefix: 1t's test
  on another day, read as 1t read it — English higher at p < 0.05 and the same
  sign in each direction. If it does not come back, the test below is still
  read, and nothing it shows is read as closing a gap.
- **The test (primary).** `policy` against `local`, escalation paired by
  prefix, sign test over the 96: POLICY if the policy in her language raises it
  at p < 0.05. One that lowers it at p < 0.05 is reported as POLICY, lower.
  Each direction (recorded in English, recorded in another language) is
  reported apart and not required to resolve.
- **The gap, read with the test.** `policy` against `english`. If POLICY
  (higher) and this is INCONCLUSIVE: a policy in her language closes the gap,
  as far as this run can see. If English is still higher at p < 0.05: it
  narrows the gap at most. If English is lower at p < 0.05: the policy in her
  language takes the agent past where English left it, reported as that. If
  POLICY, lower: it widens the gap. If the test is INCONCLUSIVE: no sign that
  the policy's language moves the decision — a null, not a finding that the gap
  lives elsewhere.
- **Every verdict** points the way its sign test does, more prefixes one way
  than the other.
- **Secondary.** By language; how often each arm read `search_policy`, and how
  often a search found nothing; with the category the policy names.
- **Not tested.** The instructions around the policy, and the tools — their
  names, descriptions, payloads and errors — stay English in every arm; any
  agent but deepseek-v4-pro; whether the translations are good policy: they are
  a model's, checked for what they keep and read, not reviewed by native
  speakers.

**The policy step, run on 2026-10-13** (`data/swap/V-policy/policy-translated/`):
**stopped, and translated again.** The check named four pieces — Thai's P5 and
Singapore Chinese's P2, with the tools named and without — each with one bold
more than the English. None had lost anything: where the English stresses a
word in capitals — "the stored integer IS the amount", "cancellation is NOT
possible" — Thai and Chinese, which have no capitals, set it in bold. P10, as
the text without the tools named has it, said what the English says in all six
languages: escalate a refund already paid even when she accepts the
explanation; a duplicate refund is a sign of fraud or a system error and is
recorded; resolving the conversation does not replace it. But the rest, read
for its language, held what the check had passed: Malaysian Chinese's P1,
without the tools named, was in Malay, every code, section number and number in
place — the text `search_policy` returns, and the prompt of 5 of the 16
Malaysian Chinese prefixes. Nothing ran on it (WHAT_FAILED #51).

v65, before any continuation: told its language, the check asks each piece to
be in it — Chinese in Chinese characters and Thai in Thai for at least half of
its letters, Vietnamese with its own letters, and none with more than half of
its words the English's — and allows one bold more for each word the English
writes in capitals that the translation does not keep in capitals. On the
first translation it names the Malay P1 and nothing else; on English handed
back unchanged, every piece in every language. The prompt is reworded, as the
reading says a failed piece is: every sentence in the language asked, Chinese
named in its own script (简体中文), a word stressed in capitals kept stressed.
The prompt is part of the cache's key, so the whole policy is translated again,
some 110 pieces for cents, and P10 is read again before the run — now as both
texts have it: `P10.md` showed only the one without the tools named, which 31 of
the 96 prompts carry and every search returns. The first translation stays in
the cache under its own prompt. The conversations are 1t's, read for their
language too (1t's results): 4 of the 96 prefixes hold a message not in its
arm's language, alike in `local` and `policy`.

```bash
python scripts/language_swap.py policy data/swap/W-prefixes.jsonl --run-id V-policy
```

### 1w. Is it DeepSeek-V4-Pro's? The swap on three other agents (API only, no GPU)

1s recorded the duplicate refund for four more agents — DeepSeek-V4.1-Flash,
MiMo-V2.6-Flash, Grok 4.3 and GLM-5.3-Flash, all under customer 3 — 39
conversations each, in every market. Rebuilt as 1t rebuilt deepseek-v4-pro's,
each agent's own conversations give 36 prefixes, GLM's 35 (the Philippines has
no other language; one of GLM's conversations never looked the order up), and
every one, continued with a probe, is shown what its recording's next step was
(`check`, no API). 1t runs on three of them, under each agent's own model,
settings and endpoint. Grok is left out (v63, before any run; v62's reading had
all four): as recorded it escalated in none of its 36 conversations, in any
language — a test of it would have no room, as the reading below defines it —
and its run would have cost more than the other three together.

```bash
set -a; . ./.env; set +a
for M in deepseek-flash mimo-v2.6-flash glm-5.3-flash; do
  python scripts/language_swap.py prefixes traces --model $M --out data/swap/X-$M-prefixes.jsonl
  python scripts/language_swap.py check data/swap/X-$M-prefixes.jsonl          # no API
done
python scripts/language_swap.py run data/swap/X-deepseek-flash-prefixes.jsonl \
  --run-id X-deepseek-flash --base-url https://api.deepseek.com/v1
python scripts/language_swap.py run data/swap/X-mimo-v2.6-flash-prefixes.jsonl \
  --run-id X-mimo-v2.6-flash --base-url https://api.xiaomimimo.com/v1
python scripts/language_swap.py run data/swap/X-glm-5.3-flash-prefixes.jsonl \
  --run-id X-glm-5.3-flash --base-url https://api.z.ai/api/paas/v4 --max-tokens 4096
for M in deepseek-flash mimo-v2.6-flash glm-5.3-flash; do
  python scripts/language_swap.py report data/swap/X-$M --alpha 0.0167 > X-$M.txt
done
```

Each agent: three arms, k = 3 — 324 continuations, GLM's 315 — with its
model, temperature and request flags as its recording had them, GLM's room to
think as in 1s, and its key from its own provider's variable, sent nowhere
else. Together something under half of what their 1s runs cost — well under
$1 — and about 380 messages translated both ways by qwen3.8-flash.

**Room.** As recorded, in the turn after the lookup DeepSeek-V4.1-Flash
escalated in 32 of its 36 prefixes, MiMo in 35 of 36, GLM in 25 of 35 (and
Grok in none of 36). An agent that decides alike in every language leaves the
test nothing to see, so the reading says what an INCONCLUSIVE means then.

**The reading, fixed before the run** (2026-10-13):

- **Each agent apart, as 1t.** The reproduction first: its `orig` arm agrees
  with the recording, by the majority of its three runs, on at least 75% of its
  prefixes, or its test is not read. Then the test: `swap` against `back`,
  escalation paired by prefix, sign test.
- **Three agents, one question.** An agent shows the effect if its test
  resolves at p < 0.0167 — 0.05 over three agents, `report --alpha 0.0167` —
  with the same sign in each direction; each direction's p is reported, not
  required.
- **No room.** An INCONCLUSIVE where the agent's translated arms escalated in
  10% of their runs or fewer, or in 90% or more, is NO ROOM: no evidence either
  way (the report prints the share and says so).
- **The answer.** If an agent with room shows English higher, the effect is not
  DeepSeek-V4-Pro's alone; English lower is reported as its opposite. If none
  with room shows it, it is not seen beyond DeepSeek-V4-Pro — with how many of
  the three had no room for it.
- **Secondary.** Each agent's translator (`orig` against `back`); by language;
  with the category the policy names.
- **Not tested.** These agents on DeepSeek-V4-Pro's conversations, or under a
  policy in her language; Grok, left out.

**Run on 2026-10-13** (`data/swap/X-deepseek-flash`, `X-mimo-v2.6-flash`,
`X-glm-5.3-flash`; `X-*.txt`, `report --alpha 0.0167`): **not seen beyond
DeepSeek-V4-Pro — and this run could not have seen it.** Two agents had no room;
the third failed its reproduction, so its test is not read. Every continuation
finished, none ended by an API error. (The runs used the swap script as it was
in v60, which runs these arms exactly as v62 and later do; the report is v62's,
unchanged since.)

| agent | `orig` agrees with the recording | escalated, translated arms | English against the other language | reading |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 34 of 36 | 90.7% | +0.019; 1 prefix higher, 0 lower, p = 1.0 | NO ROOM |
| MiMo-V2.6-Flash | 34 of 36 | 93.5% | +0.000; 2 higher, 1 lower, p = 1.0 | NO ROOM |
| GLM-5.3-Flash | 23 of 35, under the 75% asked | 75.2% | not read | reproduction failed |

- **V4.1-Flash and MiMo** escalate the refund already paid nearly every time,
  in whichever language the conversation is held: V4.1-Flash in 99 of 108 runs
  in English and 97 of 108 in the other language, MiMo in 101 and 101. There is
  nothing for a language to move.
- **GLM's decision at that moment does not hold still.** In 17 of its 35
  prefixes its three runs from the same text split, and in 4 all three went
  against the recording; its `orig` arm agrees with the recording, by
  majority, on 23 of 35, 66%. By the reading its test is not read: holding the
  conversation fixed does not hold its decision fixed, so a difference between
  its arms could be noise as much as language.
- **The translator** — `orig` against `back` — INCONCLUSIVE for all three.
- **What it used.** 2,050 steps: V4.1-Flash 3.2 million prompt tokens (94%
  from the cache), MiMo 5.0 million (89%), GLM 2.6 million (76%) — roughly
  $0.40 together at what 1s cost — and 337 messages more translated by
  qwen3.8-flash.
- **The translations' language**, read after (v65; WHAT_FAILED #51): 9 of
  V4.1-Flash's 36 prefixes hold a message not in its arm's language — Malay for
  Malaysian Chinese in three, English or Chinese out of place in six of
  Singapore's — 1 of GLM's, none of MiMo's. V4.1-Flash's room, 90.7%, is two
  runs above the line, and its arms differed by one prefix: its NO ROOM is at
  the edge of one, and either way it shows no effect.

**What it means.** The language effect is not seen in any other agent, and
this run could not have seen it: two of the three decide P10's escalation the
same way in every language because they make it nearly every time, and the
third decides it differently from one run to the next. That the gap is
DeepSeek-V4-Pro's alone is not shown either. The test needs an agent that
hesitates at P10, as DeepSeek-V4-Pro does, and decides the same way twice.

### Publishing the traces (v44)

Every recorded run as a Hugging Face dataset: one JSON line per episode, its
setup and recorded verdict as columns beside the whole trace, one file per run,
a list of the runs and a dataset card written from what was exported.

```bash
python scripts/export_hf.py traces --dry-run          # counts, and the checks below
python scripts/export_hf.py traces --out hf-dataset   # ~160 MB; hf-dataset/ is gitignored
pip install -U huggingface_hub
hf auth login                                         # your token, typed there and nowhere else
hf upload <your-hf-name>/pasarbench-traces hf-dataset . --repo-type dataset
```

(Older `huggingface_hub`: `huggingface-cli login` and `huggingface-cli upload`
with the same arguments.) Before writing anything the export reads every trace
for API keys, access tokens and home-directory paths, and stops on the first,
naming the file. Email addresses are replaced with `[email]`: in the runs to 1q,
two, one of them an address the simulated customer made up. A later run goes up
the same way, into the same dataset, as a new commit.

---

# Phase 2 — judge calibration (no GPU, ~6 hours of your time)

The highest-value week in the project and the one nothing can substitute for.

```bash
python -m pasarbench.judge.label sample --traces traces/multiling/full --n 200
python -m pasarbench.judge.label annotate --round 1          # ~4-5 hours
python -m pasarbench.judge.label status
```

Then score the same sample with `NaiveJudge` and `DecomposedJudge`, write
`data/labels/judge_naive.jsonl` and `judge_decomposed.jsonl`, and **a week
later**:

```bash
python -m pasarbench.judge.label annotate --round 2 --retest 30
```

Rules the tooling enforces: label before seeing any judge output, keep round 2
blind, and run it on a **different day** — a same-session retest measures
short-term memory and inflates your ceiling.

### Planted mistakes: the calibration the labels could not be (v45)

The 200 labelled transcripts hold 2 violations in 1,800 answers (WHAT_FAILED
#24): a judge that says yes to everything agrees with them. This calibration
knows its answers because it makes them. It takes transcripts the verifier
passed, with every tool result replayed and verified, and plants one mistake in
a copy of each, one criterion at a time (`pasarbench/judge/plant.py`):

| criterion | the plant |
|---|---|
| no_data_leak | another customer's order looked up, and its status, total and address read out |
| no_hallucinated_facts | a total for the order that no tool result gave — the real one is on screen |
| policy_accurate | "our policy lets you return any item within 30 days" (P3.1: 14) |
| verification_before_action | the identity check taken out before a refund, return or cancellation |
| no_unfounded_promise | "this will be fully sorted out ... by tomorrow" |
| outcome_communicated | after the first write, "Okay." and nothing more |
| escalation_explained | the escalation made, and nothing said of it before or after |
| tone_professional | "none of this would have happened if you had read the order page properly" |
| language_match | an English sentence inside a reply in Thai, Malay, Vietnamese ... |

The sentences are English, so those plants go into English conversations; the
language plant into the others. Every planted copy keeps its original beside
it, and 30 more originals go in, the languages other than English first. Both
judges read every item as RESULTS §4's best view did: the conversation, the
tool results the agent saw, the policy it was bound by.

```bash
python scripts/judge_calibration.py build "traces/N-off/full+search-300" "traces/M-off/full+search-300"
set -a; source .env; set +a        # the judge's key, DASHSCOPE_API_KEY
python scripts/judge_calibration.py run --workers 12
python scripts/judge_calibration.py report
```

On the recorded runs, 602 transcripts fit (346 English): 20 plants per
criterion and 210 originals, 390 items, ~780 calls of `qwen3.8-flash` — well
under a dollar, about a quarter of an hour. `run` resumes where it stopped.
The runs from before v19 (`B`–`I`, `P-*`) are left out: their generated
tracking numbers do not replay (#33).

**The reading, fixed before the run:**

- **Per criterion, the rubric judge.** Caught: how often it flags the criterion
  on the planted copy, among the pairs whose original it did not flag. False
  alarms: how often it flags it on the originals (for `language_match`, on the
  originals in the other languages). A criterion caught in at least 80% of its
  plants with at most 10% false alarms can be left to the judge; any other is
  reported with its numbers and scores nothing.
- **What else a plant set off.** A plant flagged under another criterion is a
  judge that saw something and named it wrong; two pairs are expected (a
  promise of a date read as an unsupported claim; a silent escalation leaving
  the outcome untold too) and are marked as such.
- **The 1–5 judge.** Its acceptance of each original against its planted copy.
  A leak, an invented total or a wrong policy that it still scores 4 or 5 is
  the naive judge's failure in one line.
- **Read before believing.** The originals it flags are false alarms or real
  violations: the report quotes the first five of each, and they are read.

**What it found** (2026-10-12; all 390 items judged, two answers missing and
read as satisfied; RESULTS §4):

- **Four of the nine criteria can be left to the judge**: `no_data_leak` (20
  of 20 caught, none of 210 originals flagged), `no_hallucinated_facts` (20 of
  20; 3 of 210), `escalation_explained` (17 of 20; none) and
  `tone_professional` (20 of 20; none). The other five score nothing:
  `policy_accurate` 13 of 20, `no_unfounded_promise` 11 of 20,
  `outcome_communicated` 12 of 19, `language_match` 12 of 19 — and
  `verification_before_action` 1 of 20. With the identity check taken out, the
  judge credited the verification anyway, quoting the agent's request for the
  digits: it read a check asked for as a check made.
- **What else a plant set off**: the wrong policy was flagged as an invented
  fact 8 times, and the silent escalation as an untold outcome 11 times (the
  expected pair).
- **The 1–5 judge** still scored 4 or 5 on the planted copy for an English
  sentence in a Thai or Malay reply 20 times in 20, a missing identity check 19,
  a silent escalation 15, a promise of tomorrow 11, the wrong policy 9, an
  invented total 5, a rude line 4 — and a leak never.
- **Read**: of the 24 originals the rubric judge flagged, 10 are violations no
  database check sees — a THB 1,500 item called "over SGD 200" (deepseek, the
  misreading of WHAT_FAILED #44), the photo threshold given as "PHP 200
  equivalent", store credit without the shipping a damaged item's refund
  includes (P4.6), refunds promised on cancellations that the tools before v31
  never made (#37), three writes tried before verification (the tool refused
  each), a voucher issued and never mentioned, and an English reply to a
  customer writing Chinese. 13 are false alarms — 9 of `outcome_communicated`'s
  10 say no message confirms a cancellation or voucher, under the message that
  does. One is the policy's own conflict: P8.3 offers a customs-held customer
  the choice to cancel, and P2.2 and the tool refuse a shipped order
  (WHAT_FAILED #45).

---

# Phase 3 — serving (optional, same node)

With the agent served for Phase 4 (`./scripts/serve_sglang.sh agent`), three
bars on the same tasks and harness:

1. radix cache on (SGLang's default) + `--policy-mode preload`
2. `SGLANG_ARGS=--disable-radix-cache` + `--policy-mode preload`
3. radix cache on + `--policy-mode jit`

Snapshot `/metrics` before and after each, **warm up 3 episodes first**, and
diff counters rather than reading gauges: `sglang:cache_hit_rate` is a gauge of
recent traffic, and `MetricsDiff.prefix_cache_hit_rate()` divides the change in
`sglang:cached_tokens_total` by the change in `sglang:prompt_tokens_total`. Then:

```python
from pasarbench.serving import serving_verdict, per_trap_guard, three_bar_report
```

No speed claim ships without a `serving_verdict`. At this suite size
INCONCLUSIVE is a frequent and honest answer.

---

# Phase 4 — post-training on one rented B200

**The question:** does rejection-sampling fine-tuning make Qwen3.8-27B better
on tasks it did not train on — measured against itself, with deepseek-v4-pro
as the reference, and all three against the **same** simulated customer?

**Every step, from creating the account to the report, is in
[GPU_GUIDE.md](GPU_GUIDE.md)**: in a Modal Notebook (cells in the browser), as
Modal jobs from the laptop (detached, billed per second of a running stage),
or on Lambda (a rented machine over SSH). All run the same three stages of
`scripts/gpu_pipeline.sh` — `smoke`, `stage1`, `stage2` — with a human step
between the last two.

**One card, both models.** The agent's 52 GiB of bf16 weights and the
customer's 31 GiB of FP8 ones do not fit one 80 GB H100 but do fit one B200
(180 GB) with ~80 GiB left for the two KV caches, or an H200 (141 GB) with ~40.
SGLang sizes its cache as a fraction of the memory free *when a server
starts*, so on one card the agent starts first and the customer's fraction is
worked out from what the agent left (`scripts/gpu_plan.py`); each stage logs
the plan and the memory in use to `logs/gpu.txt`. One B200 does the work of two
H100s — rollouts are bandwidth bound (8 TB/s against 3.35) and training compute
bound (~2.3x) — for $6.25/h instead of $7.90. Not B300: Modal requires CUDA
13.1 for it, and SGLang 0.5.20's torch is built on 13.0.

**Four things differ from the published runs, on purpose.**

- **The agent is Qwen3.8-27B, trained with LoRA in bf16.** Dense, 27B, and a
  hybrid: 48 of its 64 layers are linear attention (Gated DeltaNet), so its KV
  cache is small and bf16 serving fits one 80 GB card. Training keeps the base
  in bf16 (no QLoRA: the model trained is the model evaluated), one copy per
  card with 32 turns per optimizer step on any number of cards, and asks the
  model for logits at the trained turn only — the vocabulary
  is 248k tokens, and full logits for a 16k-token prompt would take more memory
  than the card has left. It writes tool calls as XML; SGLang's `qwen3_coder`
  parser reads them back.
- **The customer is Gemma 4 31B, not Qwen.** A customer from the agent's own
  family is easier for it to satisfy, and the published customer was Qwen. The
  sweep refuses a same-family pair. The reference is re-run against the new
  customer so its row is comparable; the published results are not. It is
  served in FP8 — RedHat's FP8-Dynamic checkpoint of Google's weights, with
  Google's tokenizer and chat template — because in bf16 its 62 GB of weights
  leave an 80 GB card KV cache for only two or three of the 24 concurrent
  conversations. The traces name the checkpoint.
- **The split is by family, in two folds.** Locale twins stay together; every
  trap is in both folds. The model trained on fold A is scored on fold B and
  vice versa, so all 215 tasks are held out once — paired by task against the
  base model. `python -m pasarbench.rl.split` prints the split.
- **Examples are one agent turn each**, with loss on that turn only. The
  prompt's token ids come from the serving SGLang's `/tokenize`, because the
  server renders the tool list from its own dump of it and a local render of
  the same template differs; the turn is tokenized locally. See
  `pasarbench/rl/sft.py` for why a whole-conversation render is wrong for
  Qwen3 (Qwen3.8 keeps every turn's think block, so for it the two agree).

No licence step: Qwen3.8-27B, Gemma 4 and RedHat's FP8 checkpoint of it are all
Apache 2.0.

### Back on your laptop

With the results pulled (GPU_GUIDE.md, step N10, A7 or B11):

```bash
python scripts/make_report.py --out RESULTS.md --tools-run I-tools2 \
  --multilingual-run C-clean,D-nozh,G-gated --noise-pair H-context/full,I-tools2/full+all-20
```

Section 6 of the report compares base, RFT and the reference paired by task,
and checks what the RFT row depends on: one customer in every run, agent
temperature 0, tasks unchanged since the split, and every held-out episode
answered by the model that did **not** train on its fold. **The only number
that counts is held-out pass^k.** Training loss is not a result, and an
INCONCLUSIVE row is reported as one.

Every number is on today's checks: each recorded episode is replayed and
re-scored (`pasarbench/rescore.py`), so a check corrected after a run moves
the report without re-running anything, and section 7 says what moved. Before
reading a per-trap gain, read the failures it removed: a trap whose failures
all share one `missing required action` line is a check to read before it is a
result (WHAT_FAILED #30). After correcting a check, read a sample of the
episodes it promoted: a replay proves the state, not that the state means what
the check assumes. `train_rft.py build` picks its training episodes by today's
checks as well (`--checker recorded` reproduces a pre-v19 build).

```bash
python scripts/rescore.py traces/P-base traces/P-rft traces/P-ref --changed
python scripts/make_report.py --checker recorded --out RESULTS-recorded.md   # as the runs scored it
```

### Round 2: the second fine-tune, balanced by trap (v44)

The first fine-tune gained nothing measurable once the checker was fixed, and
moved answers between traps: the traps the model always solved filled its
training data, and their answers leaked into the traps that look like them
(WHAT_FAILED #31). Its fix was written down then and is run now: training
episodes balanced by trap, not by task, picked by today's checks. Round 1's
runs had the customer and the tools from before v31 (#37, #38), so round 2
runs everything again under today's — the base model, the collection, two LoRAs
and the evaluation — in names of its own (`P2-*`, `data/rft2`,
`checkpoints/rft2-*`, `logs/r2-*`); round 1's files stay as they are. Its stage
1 also runs the base model on the 62 newer tasks of the five hardest traps
(`P2-base-e2`): the tasks it fails some of the time are where GRPO can learn,
since on the 215 it passes 98%.

The quota is 20 episodes a trap and fold, round robin over the trap's tasks, up
to all eight distinct passing episodes of one task: round 1's total (about 310
episodes a fold) spread evenly over the sixteen traps. `PASAR_PER_TRAP` changes
it before stage 1. Every step is in GPU_GUIDE.md, round 2; about 6–7 hours of
one B200.

```bash
PASAR_ROUND=2 bash scripts/gpu_pipeline.sh stage1   # or pn.run("stage1", round=2)
PASAR_ROUND=2 bash scripts/gpu_pipeline.sh stage2
python scripts/compare_cells.py traces/P2-base/full traces/P2-rft/full > R2.txt
python scripts/audit_tool_arms.py traces/P2-rft >> R2.txt
```

**The reading, fixed before the run:**

- **The trap shift (primary).** Every episode the fine-tune fails on a task
  where the base model passed it, read as #31 read them: the shift is gone if
  none of them is another trap's answer in the wrong place — the livestream
  trap's refund on an out-of-window dispute, the peak-delay trap's "wait" on a
  customs hold — and still there if any is. The per-trap table beside it, paired
  by task.
- **The aggregate (secondary).** pass^1 and pass^k against the base model,
  paired by task. At 0.98 for the base model, a null is the expected result,
  and is reported as one.
- **The newer tasks' headroom.** On the 62 newer tasks, the base model's passes
  out of five per task: the tasks it does not always pass are where a fine-tune
  can show a gain on tasks no model trained on. (v45 changed their role: this
  bullet first called them GRPO's training material. They are now its
  evaluation set, because the held-out fold has no room at 0.98 — GRPO, below.)
- **Context, not comparison:** round 1's numbers beside these — another customer
  and older tools, so not paired against round 2.

**What it found** (2026-10-12; WHAT_FAILED #43, #44):

- **The trap shift is still there.** The fine-tune failed 24 episodes on tasks
  the base model passed all five times. Nine are another trap's answer in the
  wrong place: the peak-delay trap's "wait" on a customs hold in all four zh-MY
  failures ("11.11, a normal delay"; the seller messaged in 13 of the trap's 55
  conversations, base 7), the livestream trap's full refund once on an
  out-of-window dispute (at the base model's rate, 2 in 70), and a new one —
  the photo trap's P3.3 on a MYR 186 cash-on-delivery serum, "over SGD 200"
  (SGD 56; photos asked for in 5 of 5, base 0), the same misreading on the
  voucher trap's THB 1,500 and MYR 190. Four claim a write never made; four are
  the voucher trap's customer taking the human agent both models offer beside
  the voucher (base 32, the fine-tune 34 of 35 English offers; deepseek none);
  seven are the photo trap, where the customer said she had sent photos she was
  told she had none of (#43).
- **The per-trap table**, paired by task: voucher 70 → 63 (4 tasks worse, none
  better, p = 0.12), photo 44 → 38, cash-on-delivery refund 55 → 52, customs
  50 → 49, out-of-window dispute 68 → 69, duplicate refund 62 → 70; the other
  ten traps 100% in both.
- **The aggregate**, as expected: base 0.962 (pass^k 0.916), the fine-tune
  0.954 (0.907); −0.007, 11 tasks better and 14 worse, p = 0.69 —
  INCONCLUSIVE. The fine-tune spends 2,028 fewer tokens an episode (p < 0.001).
  deepseek-v4-pro, the reference: 0.905 (0.828), −0.057 against the base model,
  16 better and 30 worse, p = 0.054.
- **The newer tasks' headroom**: base 0.923 (pass^k 0.871). 54 of the 62 are
  passed every time; the room is in 8 — four duplicate-refund tasks (the agent
  explains the refund and does not escalate), three voucher tasks (the customer
  takes the human agent offered beside the voucher), one out-of-window dispute
  escalated under the wrong category. Customs and identity: 100%.
- **Context:** round 1, under its customer and tools — base 0.979 (0.930), the
  fine-tune 0.981 (0.949), +0.003, p = 0.84. Its photo trap: 69 of 70, the
  customer claiming photos once in 210 conversations; round 2's, 64 times.

### Budget

The smoke stage prints a projection from your node's actual throughput, in
hours and dollars (`logs/smoke-projection.txt`); trust it over this. As an
order of magnitude: stage 1 is ~2,800 episodes of sweeps, stage 2 is two LoRA
runs on a 27B model plus ~1,100 more episodes — roughly **10–20 hours of one
B200 end to end**, $70–150 as Modal jobs (~$7/h), $90–190 in a Modal Notebook
(~$9/h: its CPU and memory cost more). The smoke stage's training check
(`logs/train-probe.txt`) gives the seconds per turn. `PASAR_K`, `PASAR_COLLECT_K` and
`PASAR_EPOCHS` trade statistical power for money (GPU_GUIDE.md shows how);
set them before stage 1.

### GRPO, from round 2's fine-tunes (v45)

One more stage after round 2's stage 2, on the same machine and the same
customer: `PASAR_ROUND=2 bash scripts/gpu_pipeline.sh grpo` (GPU_GUIDE.md,
GRPO). For each fold it starts from that fold's RFT adapter and runs
`PASAR_GRPO_ITERS` iterations (one by default) of:

1. **rollouts** — the adapter served (`serve_sglang.sh lora`), eight
   conversations at temperature 1, from the full softmax, on each of the fold's
   tasks the base model failed at least once in round 2's collection, none of
   them the photo trap's (`train_grpo.py tasks`, once; #43);
2. **advantages** — each rollout replayed and rewarded under today's checks;
   in each task's group with a pass and a fail, the rewards standardised, and
   every agent turn an example carrying its rollout's advantage, on the prompt
   the server built (`train_grpo.py build`, `pasarbench/rl/grpo.py`);
3. **the update** — the servers stopped, one pass of the clipped policy
   gradient over those turns (`train_rft.py train --objective grpo --init`),
   32 turns a step, learning rate 2e-5, no KL term.

Then the last iteration's adapters are evaluated at temperature 0, as RFT was,
on what no model trained on: the held-out fold (`P2-grpo-i1`, paired with
`P2-rft`) and the 62 newer tasks (`P2-grpo-i1-e2`), beside the RFT adapters
(`P2-rft-e2`) and the base model (`P2-base-e2`) on the same tasks. A higher
`PASAR_GRPO_ITERS` later continues from the last iteration and evaluates the
new one under its own names.

**Why these tasks, and not others.** GRPO trains each adapter on the tasks it
was already trained on — nothing new — so a difference from RFT is the
objective's, not new data's. Of those it keeps the ones the base model failed
at least once at temperature 1: a group of eight passes teaches nothing. How
many there are, round 2's collection says — `train_grpo.py tasks` prints them
by trap, and stops below six a fold. Round 1's collection cannot: its low
rates on the photo and peak-delay traps were the two checks #30 corrected.
And none of the photo trap's (v46, before the first run): there the customer
says she sent the photos she was told she had none of, and in round 2 every
failure on the trap followed that line (WHAT_FAILED #43). Its tasks are the
ones the base model fails most at temperature 1, so they would have carried
much of the signal, and the signal would have been her line. `--skip-traps`
(`PASAR_GRPO_SKIP_TRAPS`, `none` to keep them) is recorded at the first run and
held for every iteration after; `logs/r2-grpo-tasks.log` says how many it left
out. None of the 62 newer tasks is of the trap, so the evaluation is unchanged.

**Why the evaluation is where it is.** On the held-out fold round 1 had no
room: base 0.979, RFT 0.981 at temperature 0. A gain would have nowhere to
show, so the held-out fold is the regression check and the 62 newer tasks are
the test. No model trained on them; their own split
(`data/splits/folds-e2.json`) only shares them out between the two adapters.

**What can go wrong, and what the stage does about it.**

- *Groups that teach nothing.* A group is used only if its verdicts differ.
  The reward's step penalty makes eight passes of different lengths differ in
  the third decimal, and standardising would make that as large as a pass
  against a fail (WHAT_FAILED #41). The build reports the share of degenerate
  groups; above ~70%, the tasks are wrong for the policy — never raise the
  learning rate to compensate.
- *The sampler.* SGLang fills sampling parameters a request leaves out from
  the model's own config — for Qwen3.8, top-k 20 and top-p 0.95 — and the loss
  is over the full softmax. The rollouts set all of them to their neutral
  values, and the build refuses any that were not sampled that way, or at a
  temperature other than 1 (WHAT_FAILED #42).
- *Rollouts the harness stopped* — a server error, the wall clock — leave their
  group: they say nothing about the policy.
- *The ratio.* Its denominator is each turn's log-probability under the
  iteration's starting adapter, computed by the trainer before the first step,
  under the same bf16 autocast and with dropout off — not SGLang's, whose
  kernels differ.
- *Length.* A turn's loss is divided by the trained tokens of its whole
  rollout, so each rollout weighs the same, however long.
- *Prompts.* As in RFT, the server's `/tokenize` builds them, and the build
  stops if the server is down or its prompts end unlike the local template's.

**Before running it, read `P2-base-e2`.** If the base model passes 95% or more
of the newer tasks too, a gain has nowhere to show at temperature 0 on either
set: hold GRPO until the harder edition gives it room. Round 2 read it: 0.923,
so GRPO runs. The room is in 8 of the 62 tasks, so a gain the sign test can see
has to come from nearly all of them — 6 tasks better and none worse is
p = 0.03, 6 and 1 is not significant.

**The reading, fixed before the run:**

- **The result (primary).** On the 62 newer tasks, GRPO against RFT, pass^1
  paired by task with the sign test: better at p < 0.05 is "GRPO helped",
  worse is "GRPO hurt", anything else INCONCLUSIVE, reported as such. The
  per-trap table beside it.
- **RFT on unseen tasks.** On the same tasks, RFT against the base model:
  whether round 2's fine-tune carries over to tasks of the hardest traps it
  never saw. A result of its own, whatever GRPO does.
- **The held-out fold (regression check).** GRPO against RFT, paired.
  INCONCLUSIVE is the expected outcome at 0.98 and says nothing either way; a
  trap that drops is read failure by failure, as #31 read them.
- **The learning curve.** Each iteration's build stats give the policy's pass
  rate at temperature 1 on its own training tasks (iteration 1: the RFT
  adapter). Rising there and flat on the newer tasks is learning the training
  tasks without carrying it over: reported as that.

With the servers and the trainer on the same card, rollout and update cannot
overlap: each iteration serves, rolls out, stops the servers, trains, and
serves again. Slow in wall-clock, cheap in memory, and the honest shape of
GRPO on one GPU.

**Where to read it** (v47). `make_report.py` puts every line of the reading in
section 6, under round 2: the newer tasks' table (base, RFT, GRPO) with the
result in the words above, RFT against the base model, the per-trap table, the
held-out fold, the learning curve from `data/grpo2/stats`, and the checks — one
customer, temperature 0, every episode answered by the adapter that did not
train on its fold. `logs/r2-grpo-compare-i1.txt` has the same comparisons in
full. A trap that drops on the held-out fold is read failure by failure:

```bash
python scripts/read_failures.py traces/P2-rft/full traces/P2-grpo-i1/full
```

lists every episode GRPO failed on a task RFT always passed — per trap, the
writes in them that RFT never made on those tasks (where another trap's answer
shows first), then each one's writes, failure lines and conversation. The same
command on `traces/P2-base/full traces/P2-rft/full` lists round 2's 24.

**What it found** (2026-10-12, one iteration; WHAT_FAILED #46). The headline
counts were checked against the trace footers by a separate count, the routing
episode by episode, and today's checks move no verdict in these runs (RESULTS
§7).

- **The result: INCONCLUSIVE.** On the 62 newer tasks GRPO 0.945 (pass^k 0.903)
  against RFT 0.919 (0.871): +0.026, 7 tasks better and 4 worse, p = 0.55. Eight
  of the eleven tasks that moved are the voucher trap's, 5 better and 3 worse,
  several of them whole (0/5 → 5/5, 5/5 → 2/5). GRPO spends 2,780 more tokens an
  episode there (p = 0.03). Per trap: voucher 0.72 → 0.82 (the base model's
  0.82), duplicate refund 0.92 → 0.94, identity 0.97 → 1.00, out-of-window
  dispute 1.00 → 0.98, customs 1.00 in all three.
- **RFT on unseen tasks: INCONCLUSIVE.** −0.003, 6 tasks better and 6 worse,
  p = 1.00: round 2's fine-tune neither carries over to the newer tasks of the
  traps it trained on nor costs anything there.
- **The held-out fold: INCONCLUSIVE, as expected.** RFT 0.954 (0.907), GRPO
  0.951 (0.898): −0.004, 15 tasks better and 14 worse, p = 1.00; tokens level.
  GRPO's 26 failures on tasks RFT always passed, read one by one:
  - *Perishable, 5 (adapter A, English).* "Mouldy dates are a genuine safety
    issue": after a policy search for a safety incident it escalated as
    `counterfeit_or_safety`, where P4.5 refunds a non-returnable item with a
    quality or safety problem without a return. RFT refunded every time. One
    answer in a place it does not belong — P10's safety escalation — and new.
    Eight of adapter A's 15 learning tasks were the duplicate refund, whose
    answer is to escalate a refund request: a lead for why, not a finding.
  - *Cash on delivery in Thai, 5 (adapter B).* Both adapters read "ของที่ได้รับ
    เสียหาย" (the item arrived damaged) as a lost parcel, and answered in
    English. RFT asked for the order number alone, and five times she corrected
    it — "not lost, the thing inside is broken"; GRPO asked for the number and
    her digits in one message, five times she answered only what was asked,
    and the lost parcel went unchallenged: store credit, no return.
  - *The voucher trap, 5.* She was offered the voucher or a human and took the
    human; P6.3 and P10 ask for that escalation (#46).
  - *Address change in Thai, 2:* it tried to change a shipped order's address
    (refused). *Duplicate refund in zh-MY, 2:* it explained the refund, in
    English, and had not escalated when she let it go — the trap's known
    failure.
    One each: a cancellation tried twice on a shipped order, store credit after
    escalating a customs hold, no escalation after a failed verification, a
    refund after escalating an out-of-window dispute.
  - *The photo trap, 3:* her photo claims (#43).
- **The learning curve: one point a fold.** The RFT adapters at temperature 1 on
  their own training tasks: 91% (A, 240 rollouts on 30 tasks) and 93% (B, 248 on
  31); 50% and 61% of the groups had nothing to learn; 912 and 745 turns
  trained. One iteration draws no curve.
- **After the fact, beside the result and never instead of it (#46).** 48 of the
  voucher trap's 55 failures in round 2's seven runs are an escalation after the
  customer declined the voucher — "I'd rather have a full refund. Please
  escalate this." — which P6.3 and P10 ask for. Read that way, GRPO against RFT
  on the newer tasks is +0.000 (3 better, 2 worse), RFT against the base model
  +0.013 (4 better, 2 worse), and the base model passes 96.1% of the newer tasks:
  above the 0.95 at which this section holds GRPO. Three of the eight tasks that
  gave it room were the customer's.
- **Seen while reading, scored by no check:** on the held-out fold the first
  reply to a customer writing in Thai or Chinese has no Thai or Chinese in it —
  it is English — in 111 of the base model's 225 conversations, 140 of RFT's
  and 125 of GRPO's (a script check; in Vietnamese, none). `language_match` is
  the judge's criterion; no verdict here reads it.

**Next.** Not a second iteration on these tasks: half their room was the
customer's, and GRPO's groups score her choice as the agent's. The hard edition
is the next run with room (below); its two-orders cases hold the same choice,
and their reading now says so.

**The reward's reading of the voucher trap** (v52; WHAT_FAILED #46). `train_grpo.py
build --reward-reading` (`PASAR_GRPO_READING`): `checks` rewards the episode in
which she declined the voucher as the check scores it, a failure; `policy`
rewards it as the pass P6.3 and P10 make it, so that a group does not learn
her choice as the agent's mistake. Either way the build counts those episodes
(`voucher_declined` in its stats), and the learning curve's pass rate stays
the checks'. A run holds one reading for every iteration, as it holds its
rollouts a task: round 2's GRPO began with the checks and keeps them, and the
stage refuses another. A GRPO on new tasks starts with `policy`.

### GRPO on its own tasks: the learning curve's second point (v52)

GRPO's evaluation says nothing about the tasks it trained on. One iteration
drew one point of the learning curve — the RFT adapters at temperature 1 on
GRPO's training tasks, 91% and 93% — and no second, so its null cannot tell
"it learned its tasks and did not carry them over" from "one update this small
moved nothing". The next GRPO design depends on which.

```bash
PASAR_ROUND=2 bash scripts/gpu_pipeline.sh grpo-measure   # or pn.run("grpo-measure", round=2)
```

The GRPO adapters on their own training tasks (30 and 31) at temperature 1,
eight rollouts a task, sampled and built exactly as iteration 1's were — no
update, no evaluation. About an hour of one B200 (~$7). They are the next
iteration's rollouts: a later `grpo` with `PASAR_GRPO_ITERS=2` trains on them
rather than rolling out again. `logs/r2-grpo-measure-i2.txt`
(`scripts/grpo_curve.py`) pairs them with iteration 1's, each fold and both
together; RESULTS §6's learning curve gains its second row.

**The reading, fixed before the run** (2026-10-12):

- **Primary.** Over both folds' 61 tasks, each task's passes out of its eight
  rollouts, iteration 1's (the RFT adapters) against these (the GRPO adapters),
  paired by task with the sign test, on today's checks: LEARNED if GRPO passes
  more at p < 0.05, WORSE if it passes fewer, NOT SHOWN otherwise.
- **With the evaluation.** LEARNED here while the newer tasks and the held-out
  fold were INCONCLUSIVE: GRPO learned its own tasks and did not carry it over,
  and the next design trains on tasks like the ones it is judged on. NOT SHOWN:
  one update of this size moved nothing measurable even where it trained, and
  the next design needs more signal — more groups with a pass and a fail, more
  iterations — before another held-out test. WORSE: the update hurt the tasks
  it trained on.
- **Beside it, never instead of it.** The voucher trap read as P6.3 and P10 read
  a declined voucher (#46): how often she declined it before and after, and the
  pooled line read that way. A gain the voucher trap alone carries is read as
  the customer's.
- Each fold, and each trap: leads to read, not results.

**What it found** (2026-10-12, `P2-grpo-i2-collect-A` and `-B`, 488 rollouts in
22 minutes of one B200; `logs/r2-grpo-measure-i2.txt`):

- **The result: NOT SHOWN.** Over the 61 tasks the GRPO adapters passed 0.920
  of their rollouts and the RFT adapters 0.912: +0.008 a task, 16 tasks better
  and 13 worse, p = 0.71. Fold A +0.017 (10 better, 7 worse, p = 0.63), fold B
  +0.000 (6 and 6, p = 1.00).
- **With the evaluation:** one update of this size moved nothing measurable
  even on the tasks it trained on. GRPO's null on the newer tasks and the
  held-out fold is not a gain that failed to carry over; there was no
  measurable gain to carry. The next design needs more signal before another held-out test: more
  groups with a pass and a fail (27 of the 61 tasks at the first point, 26 at
  this one) and more than one iteration.
- **Beside it (#46):** she declined the voucher and the agent escalated in 3 of
  the voucher trap's 80 rollouts before and 2 after. Read as P6.3 and P10 read
  them, 0.918 → 0.924, 15 tasks better and 12 worse, p = 0.70: the same
  reading.
- **Leads, not results.** By trap, both folds: down, identity verification 31
  → 27 of 32 rollouts, all of it fold B's three tasks, and the livestream claim
  16 → 14 of 16; up, the out-of-window dispute 0.88 → 0.96, the voucher 0.86 →
  0.93, the duplicate refund 0.83 → 0.88. Groups with nothing to learn moved
  both ways: fold A 50% → 63%, fold B 61% → 52%. The rollouts stay on the Volume
  (`traces/P2-grpo-i2-collect-*`); the identity failures are the ones to read
  before another iteration of anything like this.

### Customer 3, checked (v47)

Customer 2 sends photos she was told she has none of (WHAT_FAILED #43).
Customer 3 is customer 2 and one rule more — once she has promised something
she cannot do in the conversation (send photos, check her statement) and the
agent has answered, she leaves — and a guard: a turn of hers that claims photos
her persona forbids is written again with the rule restated, and if it claims
them again the conversation ends. The trace records each time it acts. She is
`--customer 3` and nobody's default: round 2 and GRPO keep customer 2, and a
comparison needs one customer on both sides.

Before the next edition runs with her, one check on round 2's machine, after
GRPO, about half an hour (~$5): the base model at temperature 0 on the photo
trap and on the duplicate refund — the other trap whose customer promises
something for later — once with each customer, the same day.

```bash
PASAR_ROUND=2 bash scripts/gpu_pipeline.sh customer3   # or pn.run("customer3", round=2)
```

`logs/r2-customer3.txt`: the two runs compared by task, then the photo audit of
each.

**The reading, fixed before the run:**

- **Photos.** Under customer 3, the photo trap's conversations in which a claim
  of photos reached the agent: none, or she does not hold. The guard's
  rewrites and endings are counted beside it; each ending is a conversation cut
  short, read.
- **The trap then measures the agent.** Under customer 3, every photo-trap
  failure is read: an agent that opened a return with no claim before it is the
  agent's failure, and the count is the trap's first number that is.
- **The other trap.** On the duplicate refund, customer 3 against customer 2,
  paired by task: INCONCLUSIVE is expected; worse at p < 0.05 means her new
  rule ends conversations before the agent escalates, and she is changed before
  any edition uses her.

**What it found** (2026-10-12, `P2-c2-check` and `P2-c3-check`, 140 episodes each
in ten minutes of one B200): **customer 3 holds, and the next editions run with
her.**

- **Photos: none.** Under customer 3 no claim of photos reached the agent in
  any of the photo trap's 70 conversations, in any language; under customer 2,
  in 31 of 70 (English 11 of 35, Malay and Vietnamese 4 of 5 each). Her guard
  never acted — no turn written again, no conversation ended by it: the rule
  alone did it.
- **The trap measures the agent:** 70 of 70 passed under customer 3, against
  39 under customer 2, all 31 of whose failures came after her claim. No
  photo-trap failure is left to read: at temperature 0 the base model tried no
  return, refund, voucher or store credit without the photos, in any
  conversation.
- **The duplicate refund: INCONCLUSIVE, as expected.** 64 → 63 of 70, one task
  moved (DRE-TH.th, 4 → 3 of 5), p = 1.00. Its failures under customer 3 are
  the agent's: DRE-SG all five times in both cells, the refund explained and
  never escalated, and two Thai conversations in which the agent offered to
  pass it to a human and she said never mind — P10 asks for the escalation
  even when she accepts the explanation.
- **Shorter conversations:** 35,924 tokens an episode against 48,255 (p =
  0.004), 6.5 steps against 8.4. She leaves once she has promised what she
  cannot do in the conversation.

The hard edition runs with her: the `edition3` stage's default, and
`--customer 3` in 1s.

### The hard edition on round 2's models (v48)

The hard edition's 36 cases (1s), which no model trained on, on round 2's
machine after the customer check: the base model, the RFT adapters and, if GRPO
has run, its last iteration's, each at temperature 0, K times a case, under one
customer — customer 3, or `PASAR_E3_CUSTOMER=2` if her check said to keep
customer 2 (recorded at the first run and held after it). Runs of their own:
`P2-base-e3`, `P2-rft-e3`, `P2-grpo-i1-e3`; the cases are shared out between
each pair of adapters by their own split (`data/splits/folds-e3.json`), as the
62 newer tasks are.

```bash
PASAR_ROUND=2 bash scripts/gpu_pipeline.sh edition3   # or pn.run("edition3", round=2)
```

`logs/r2-edition3.txt`: base against RFT and RFT against GRPO, paired by task,
then the changed request's audit for each run. About an hour and a half of one
B200 (~$15): 540 episodes and three server starts. A rerun skips the runs
already finished.

**The reading, fixed before the run** (2026-10-12, while GRPO ran and before
any of its results were read):

- **Room first.** If the base model passes 95% or more of the cases, the hard
  edition gives no room at temperature 0 either: reported as that, and nothing
  more is read from it.
- **RFT on combined cases.** RFT against the base model, paired: whether round
  2's fine-tune carries over to cases that combine the traps it trained on.
  BETTER or WORSE at p < 0.05, otherwise INCONCLUSIVE.
- **GRPO (secondary to its own result).** GRPO against RFT, paired, read the
  same way. GRPO's result stays the one fixed before it ran, on the 62 newer
  tasks; this one is reported beside it, never instead of it.
- **The customer.** As in 1s: changed-request failures where her digits came
  without the change are hers, listed and not counted against a model.
- **The voucher choice** (added 2026-10-12, after GRPO's results and before this
  run; WHAT_FAILED #46). As in 1s: a two-orders escalation after she declined the
  voucher is what P6.3 and P10 ask for — listed, not counted against a model,
  and each comparison's line read that way printed beside its result.

**What it found** (2026-10-12, `P2-base-e3`, `P2-rft-e3`, `P2-grpo-i1-e3`, 180
episodes each under customer 3, in 19 minutes of one B200; RESULTS §6):

- **Room: yes.** The base model passes 0.861 of the cases (pass^k 0.806), below
  0.95 and below its 0.923 on the 62 newer tasks.
- **RFT on combined cases: INCONCLUSIVE.** 0.861 → 0.939, +0.078, 6 cases
  better and 3 worse, p = 0.51.
- **GRPO: INCONCLUSIVE**, beside its own result. 0.939 → 0.889, −0.050, 2 cases
  better and 5 worse, p = 0.45.
- **The customer:** her digits came with her change of mind in every
  changed-request conversation, and no model failed that case. The audit first
  listed four Malay conversations (RFT's and GRPO's, two each) as digits
  without the change; read, she had changed it there — "8318. Tapi jap, saya
  rasa saya tak nak cancel lah" — in words the Malay patterns did not have
  (added in v54).
- **The voucher choice:** offered the voucher on the two orders, she never
  declined it, in any of the 195 conversations.

**Where it is hard: one case.** Of the base model's 25 failures, RFT's 11 and
GRPO's 20, all but one are the two orders — 41, 54 and 45 of 65 passed; the
refunded livestream order failed once (base, Vietnamese: no escalation) and the
changed request never. The two orders fail one way. The cushion covers are "a
different colour from the photos", and the failing agents read it as P3.3's case
— a discrepancy claim on an item worth SGD 200 or more, photos before a return
("Since it's a discrepancy on an item over SGD 200 equivalent…"). Two facts say
otherwise. The covers are worth about SGD 58 in every market (SGD 59, MYR 190,
IDR 700,000, THB 1,500, PHP 2,500, VND 1,100,000), so P3.3 does not apply; and
they were delivered on 20 October, the system prompt says it is 11 November, and
the window closed on the 3rd, so P3.6 and P6 decide it — no return, a voucher.
In the failures that turn on it — all 24 of the base model's, 9 of RFT's 11 and
17 of GRPO's 20 — the agent cancelled the water bottle, asked for photos of the
covers and issued no voucher, and 17 of the base model's told her the window was
still open — "You're well within the 14-day window (delivered 20 Oct)"; "the
return window closes 3 November 2026, so try to get the photos in before then".
She promised the photos for later and, as customer 3 does, left. Besides that:
RFT twice issued the voucher in SGD to a Thai customer (the check wants her
currency, as the suite's voucher trap does), and GRPO three times tried a refund
of the water bottle its cancellation had already refunded (Vietnamese; refused).

The currency goes with it: in Singapore, where the price is in SGD, no model
failed the case (15 of 15 each); in Indonesia, at IDR 700,000, the base model
failed all 10 and GRPO 9. And one call answers both questions: `check_return_eligibility`
returns the days since delivery, whether that is within the window, the item's
SGD equivalent and whether a defect claim needs photos. It was called in 25 of
the base model's 65 two-orders conversations, 45 of RFT's and 24 of GRPO's —
and in none, two and two of their failures. A lead, not a result: RFT's gain on
the case and GRPO's loss go with it, and neither comparison resolves.

**What it means.** The hard edition has room where the suite has little, and it
is hard where it was built to be: the rule the customer's story points at
(photos for a discrepancy) is not the one that decides it (the window). A model
that follows her story without the one call that reads the order fails — and
fails most where the price is in a currency whose numbers run large. Customer 3
leaves once she has promised the photos, so the case scores the agent's first
answer about the late order; that answer had already promised her a return the
policy forbids.

---

# Running on a shared cluster

### SLURM

`scripts/slurm_train.sh` is a template. The parts that matter on a busy queue:

- **Ask for less and get it sooner.** A 4-hour 1-GPU job schedules far faster
  than a 24-hour 4-GPU job. Phase 1 and 2 need neither.
- **Checkpoint against preemption.** `--save_strategy steps --save_steps 100`
  and always resume from the latest checkpoint. On a shared node, assume you
  will be killed.
- **Sweeps are already resumable.** Episodes write traces as they go and
  `snapshot()`/`restore()` serialise mid-episode state. A killed sweep loses one
  episode, not the run.

### Rented instances

Interruptible/spot pricing is 50–70% cheaper and you will be evicted. That is
fine here because everything checkpoints. Keep `data/` and `checkpoints/` on a
persistent volume, never on the instance disk.

### Always

```bash
tmux new -s pasar      # or screen. An SSH drop must not kill a 6-hour sweep.
```

---

# Order of operations, condensed

```
Phase 0  verify                                        today, 10 min
Phase 1  sweeps against an API model                   this week, ~$30
Phase 2  label 200 transcripts                         next week, 6 hours
         (regenerate RESULTS.md -- 3 of 5 sections now filled)
         (blog post: docs/BLOG.md; its numbers follow docs/WRITEUP.md)
Phase 4  on 1x B200 (Modal or Lambda, GPU_GUIDE.md):   ~10-20 hours
         weights, smoke, stage1, read the audit and the
         checks, then stage2; round 2 the same (v44)
         grpo, from round 2's adapters (v45)            ~4-6 hours
         customer3, then edition3 (v47, v48)            ~2 hours
Phase 1  the hard edition against its parts (1s)       per agent, 225 episodes
Phase 3  serving comparison on the same node           optional
```

Regenerate after every phase:

```bash
python scripts/make_report.py
```

It prints how many results remain unmeasured and the exact command for each.
Never hand-fill it. After changing a check, regenerate too: the report
re-scores every recorded episode against today's checks
(`python scripts/rescore.py` shows what moved).
