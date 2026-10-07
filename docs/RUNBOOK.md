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

The sweep searches `PASARBENCH_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`,
`DASHSCOPE_API_KEY`, `TOGETHER_API_KEY`, `GROQ_API_KEY` in that order for the
agent, and `PASARBENCH_SIM_API_KEY`, `DASHSCOPE_API_KEY`, ... for a simulated
customer on another provider. It fails immediately with guidance if none is
set rather than sending a request and handing you a 401, and it refuses to
send the agent's key to the customer's provider when the customer's key is
missing.

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

### 1i. Does a claim check stop the false claims? (API only, no GPU)

Behind search the agent told customers "I've escalated your case" with no
escalation in the database, and a judge believed it (WRITEUP, Results 1 and 2).
`--guardrail claims` reads each reply before it reaches the customer. A reply
that claims, in the first person, a write action that no successful call in
the conversation has done is held back — the customer never sees it — and the
agent gets a note naming the claim and the tool that would back it, asking it
to make the call or reword the reply; at most two notes per episode, so an
agent that insists cannot loop. The check reads only the reply and the
conversation's own calls, never the task's answer, so a deployed agent could
run it (`pasarbench/harness/guardrail.py`). It reads English only, and ignores
"we" and passives, which can describe what happened before the conversation:
"your refund was issued on 9 November", read as a claim, would send the agent
towards a second refund.

**Before the run**, read what it would have done to the runs already recorded:

```bash
python scripts/guardrail_dry_run.py traces
```

On the 9,958 finished episodes recorded up to 1g — 6,369 of them in English,
the only ones it reads — it would have fired on 25. Fourteen are in I-tools'
broken oracle arm, which hid both a lookup the task needed and the escalation
tool (WHAT_FAILED #18); ten are behind search (I-tools 4, I-tools2 3,
J-preload 3); one is an "I'll escalate it now" in P-base that the agent made
good a turn later. Each is a claim no call had backed when the customer read
it, and none is a false alarm. In the 9,478 episodes with the tools in view
it fires once, on that promise.

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
    --guardrail $G --run-id L-$G --workers 8
done
python scripts/compare_cells.py traces/L-off/full+search-300 traces/L-claims/full+search-300
python scripts/audit_tool_arms.py traces/L-claims
```

**The reading, fixed before the run:**

- **Primary:** `compare_cells.py`'s line "replies the customer got that claim a
  write no call had done". The guardrail works if the guarded arm has none, or
  none but the ones it let through after two notes ("claimed again").
- **Read every held reply** it lists. One that claims nothing, or claims what
  happened before the conversation, is a false alarm; count them. A false alarm
  that leads to a wrong action — a guarded episode failing where the control
  passed, on a forbidden write — is the guardrail's cost, and is reported.
- **Secondary:** the paired pass rate and tokens. The note names the tool, so
  part of any pass-rate gain is naming (1g); the claim line is the guardrail's
  own result.

192 episodes, the size of 1g.

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

### GRPO

Only after RFT has a held-out result, starting from the RFT adapter, trained
on one fold and scored on the other. With the servers and the trainer on the
same cards, rollout and update cannot overlap; `scripts/train_grpo.py` says
what that means and what is not built.

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
         checks, then stage2
Phase 3  serving comparison on the same node           optional
         GRPO only once RFT has a held-out result
```

Regenerate after every phase:

```bash
python scripts/make_report.py
```

It prints how many results remain unmeasured and the exact command for each.
Never hand-fill it. After changing a check, regenerate too: the report
re-scores every recorded episode against today's checks
(`python scripts/rescore.py` shows what moved).
