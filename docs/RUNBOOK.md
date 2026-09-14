# Runbook

Ordered by what you can do **right now**, not by chapter number.

## The headline: most of the remaining work needs no GPU

| phase | needs | cost | blocked by GPU queue? |
|---|---|---|---|
| **0. Verify** | laptop | free | no |
| **1. Context + tool + multilingual sweeps** | an API key | ~$15–40 | **no** |
| **2. Judge calibration** | your own eyes, ~6 hours | free | **no** |
| **3. Serving measurements** | 1 GPU | a few hours of node time | yes |
| **4. Post-training** | 1–4 GPUs | 1–3 days of node time | yes |

Phases 1 and 2 produce **three of the five results** in `RESULTS.md`, including
the multilingual diagnosis — the one to lead the writeup with. Phase 2 needs no
compute at all and is the highest-value week in the project. Do not sit idle
waiting for a node.

---

# Updating an already-pushed repo

If you pushed an earlier snapshot, the histories have diverged at the root and
`git pull` will refuse to merge them. Replace the remote contents instead —
nothing depends on the old commits:

```bash
tar xzf pasarbench-repo.tar.gz && cd pasarbench
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"
./scripts/push_to_github.sh git@github.com:<you>/pasarbench.git
```

It runs the full suite first and refuses to push a broken tree, strips run
artifacts, creates one commit authored by you, and force-pushes. It asks you to
type `REPLACE` before doing anything destructive.

Then confirm the update landed:

```bash
grep -c "extra-body" pasarbench/sweep.py     # expect 5
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

Expect: reference 1.0, null 0.0, 9 green suites, and a scripted end-to-end
sweep. If anything fails here, stop — everything downstream inherits it.

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
`TOGETHER_API_KEY`, `GROQ_API_KEY` in that order, and fails immediately with
guidance if none is set rather than sending a request and handing you a 401.

**Do not pass `--api-key sk-...` on a shared cluster.** Command-line arguments
land in `/proc/<pid>/cmdline`, which is world-readable — any other user on the
node can read your key out of `ps aux`. Environment variables live in
`/proc/<pid>/environ`, readable only by you and root.

Three ways to set it, safest first:

```bash
# 1. A .env file, already in .gitignore. Best for repeated runs.
echo 'DEEPSEEK_API_KEY=sk-...' > .env
set -a; source .env; set +a

# 2. Interactive, never echoed and never in shell history.
read -rs -p "DeepSeek key: " DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY

# 3. Plain export with a LEADING SPACE, which keeps it out of ~/.bash_history
#    when HISTCONTROL includes ignorespace (bash default on most distros).
 export DEEPSEEK_API_KEY=sk-...
```

Check it took without printing it:

```bash
echo "${#DEEPSEEK_API_KEY} chars"     # ~35 for a DeepSeek key
```

If you ever paste a key into a terminal on a shared machine, or commit one,
rotate it. It is thirty seconds of work and the alternative is someone else
spending your credits.

#### Provider notes

**DeepSeek** — `./scripts/smoke_deepseek.sh` (set `DEEPSEEK_API_KEY`). Model is
`deepseek-v4-flash`, endpoint `https://api.deepseek.com/v1`.

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

### 1b. Audit the simulator before trusting any number

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
describe an easier benchmark than the one you wrote up. Fix it here, not later.

### 1c. Context ablation — subsample first

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

### 1d. Tool scaling — include the pair or the result is unreadable

```bash
python -m pasarbench.sweep \
  --backend openai --model gpt-4o-mini --simulator openai --sim-model gpt-4o-mini \
  --suite all --sample 2 --strategies full \
  --exposure oracle,all-20,all-100,random-100,all-300,random-300,search-300 \
  -k 3 --run-id toolscale
```

`all-N` **and** `random-N` must both be present. The sweep warns you if they are
not: without the pair you cannot tell token cost from selection difficulty.

### 1e. Multilingual — the run that produces the headline

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

# Phase 3 — serving (1 GPU)

```bash
./scripts/serve_vllm.sh agent-8b          # or agent-32b on 2 GPUs
```

Three bars, same tasks, same harness:

1. `--enable-prefix-caching` + `--policy-mode preload`
2. `--no-enable-prefix-caching` + `--policy-mode preload`
3. `--enable-prefix-caching` + `--policy-mode jit`

Snapshot `/metrics` before and after each, **warm up 3 episodes first**, and
diff counters rather than reading gauges. Then:

```python
from pasarbench.serving import serving_verdict, per_trap_guard, three_bar_report
```

No speed claim ships without a `serving_verdict`. At 186 tasks INCONCLUSIVE is
a frequent and honest answer.

---

# Phase 4 — post-training (1–4 GPUs)

```bash
./scripts/serve_vllm.sh agent-8b                       # terminal 1
./scripts/serve_vllm.sh sim                            # terminal 2, port 8001
python scripts/train_rft.py collect --k 8 --temperature 1.0
```

Read `data/rft/stats.json` **before training anything**. The number that decides
whether RFT is viable is per-trap pass rate: a trap at 0.0 contributes no
training data, so fine-tuning cannot teach it and it will still fail afterwards.
Report those by name.

```bash
accelerate launch --num_processes 4 --use_deepspeed \
  scripts/train_rft.py train --data data/rft/sft.jsonl
```

Then re-serve the checkpoint and re-run the sweep. **The only number that counts
is pass^k on held-out tasks through the same harness.** Training loss is not a
result.

GRPO only after RFT plateaus — RFT is the only meaningful GRPO baseline.

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
         (write the blog post from docs/WRITEUP.md)
Phase 3  serving, when a node frees up                 half a day
Phase 4  RFT, then GRPO if it plateaus                 2-3 days of node time
```

Regenerate after every phase:

```bash
python scripts/make_report.py
```

It prints how many results remain unmeasured and the exact command for each.
Never hand-fill it.
