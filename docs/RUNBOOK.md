# Runbook

Ordered by what you can do **right now**, not by chapter number.

## The headline: most of the remaining work needs no GPU

| phase | needs | cost | blocked by GPU queue? |
|---|---|---|---|
| **0. Verify** | laptop | free | no |
| **1. Context + tool + multilingual sweeps** | an API key | ~$15–40 | **no** |
| **2. Judge calibration** | your own eyes, ~6 hours | free | **no** |
| **3. Serving measurements** | the Phase 4 node | an hour or two, optional | yes |
| **4. Post-training** | 2× H100, rented | ~8–12 hours of node time | yes |

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

# Phase 4 — post-training on two rented H100s

**The question:** does rejection-sampling fine-tuning make Qwen3-8B better on
tasks it did not train on — measured against itself, with deepseek-v4-pro as
the reference, and all three against the **same** simulated customer?

**Three things differ from the published runs, on purpose.**

- **The customer is Gemma 4 31B, not Qwen.** A customer from the agent's own
  family is easier for it to satisfy, and the published customer was Qwen. The
  sweep refuses a same-family pair. The reference is re-run against the new
  customer so its row is comparable; the published results are not. It is
  served in FP8 — RedHat's FP8-Dynamic checkpoint of Google's weights, with
  Google's tokenizer and chat template — because in bf16 its 62 GB of weights
  leave an 80 GB card KV cache for only two or three of the 24 concurrent
  conversations. The traces name the checkpoint. On a bigger card,
  `SIM_MODEL=google/gemma-4-31B-it bash scripts/gpu_pipeline.sh …` moves the
  server, the sweeps and the reference row to bf16 together.
- **The split is by family, in two folds.** Locale twins stay together; every
  trap is in both folds. The model trained on fold A is scored on fold B and
  vice versa, so all 215 tasks are held out once — paired by task against the
  base model. `python -m pasarbench.rl.split` prints the split.
- **Examples are one agent turn each**, with loss on that turn only. The
  prompt's token ids come from the serving SGLang's `/tokenize`, because the
  server renders the tool list from its own dump of it and a local render of
  the same template differs; the turn is tokenized locally. See
  `pasarbench/rl/sft.py` for why a whole-conversation render is wrong for Qwen3.

### Before renting

1. Pick a provider with **2× H100 80 GB**, a machine image with **NVIDIA driver
   580 or newer** (CUDA 13: SGLang 0.5.20 and the torch it pins need it;
   `setup_node.sh` checks before installing anything), and a **persistent
   volume**. Put the repo and `HF_HOME` on the volume: every stage resumes from
   what is on disk, an evicted instance takes its own disk with it, and the
   weights are ~50 GB.
2. Optional: `export DEEPSEEK_API_KEY=...` on the node for the reference row
   (1,075 API episodes, run alongside stage 2's evaluation).

No licence step: Qwen3-8B, Gemma 4 and RedHat's FP8 checkpoint of it are all
Apache 2.0. `HF_TOKEN` is optional; set it if a download asks for one or is
rate-limited.

### On the node, in tmux

```bash
bash scripts/setup_node.sh             # once: driver check, venv, SGLang, peft, weights, all tests
bash scripts/gpu_pipeline.sh smoke     # ~15 min: 16 episodes, projects the rest
bash scripts/gpu_pipeline.sh stage1    # split, baseline 215x5, collection 215x8, examples, checks
```

**Read three files before stage 2.** If your provider keeps the volume, stop
the instance while you read.

| file | what to look for |
|---|---|
| `logs/sim_audit.txt` | section 0 first: **any** chat-format token (`<\|channel>`, `<turn\|>`, `<tool_call>`…) in either side's text means a server parser is missing or wrong — fix it before reading anything else (the smoke stage prints the same section). Then the leak rate and **unanswered-request rate** by language. A language far above English is the detector or the customer, not the agent: read the flagged lines. If the agent asked in words `simqa.ASK_PATTERNS` lacks, extend the list (#26). Do not train against a customer you have not checked. |
| `logs/check-A.txt`, `logs/check-B.txt` | the printed example — loss on the agent's turn only — and whether the prompts carry the server's own token ids (`build --server`, the default in the pipeline). Expect a line saying most of them are not what the local template renders: SGLang dumps the tool list its own way (`"strict": false`, its own key order), which is why the build takes prompts from the server. The server check then re-renders a sample: `N of N … token-identical to the ids stored`. If `logs/build.log` says it rendered **locally** instead, **stop**: those prompts are not the ones the model sees. |
| `data/rft/stats.json` | collection pass rate, and per fold the traps never solved. A trap at zero in a fold gives that fold's model nothing to learn. If the base model already passes ~0.95, there is little headroom — say so rather than training anyway. |

```bash
bash scripts/gpu_pipeline.sh stage2    # one LoRA per fold on both GPUs, then held-out eval (+ reference)
```

Any stage can be re-run after a crash: sweeps keep finished episodes, training
resumes from its last checkpoint, and finished adapters are skipped.

### Back on your laptop

```bash
rsync -av node:~/pasarbench/traces/P-base node:~/pasarbench/traces/P-rft \
          node:~/pasarbench/traces/P-ref node:~/pasarbench/traces/P-collect traces/
rsync -av node:~/pasarbench/data/splits data/
rsync -av node:~/pasarbench/data/rft/stats.json data/rft/
python scripts/make_report.py --out RESULTS.md --tools-run I-tools2 \
  --multilingual-run C-clean,D-nozh,G-gated --noise-pair H-context/full,I-tools2/full+all-20
```

Section 6 of the report compares base, RFT and the reference paired by task,
and checks what the RFT row depends on: one customer in every run, agent
temperature 0, tasks unchanged since the split, and every held-out episode
answered by the model that did **not** train on its fold. **The only number
that counts is held-out pass^k.** Training loss is not a result, and an
INCONCLUSIVE row is reported as one.

### Budget

The smoke stage prints a projection from your node's actual throughput; trust
it over this. As an order of magnitude: stage 1 is ~2,800 episodes of sweeps,
stage 2 is two LoRA runs of a few thousand examples each plus ~1,100 more
episodes — roughly **8–12 hours of the 2-GPU node end to end**, more if the 31B
customer is the bottleneck (the smoke projection shows it). Multiply by your
provider's hourly rate for both cards.

### GRPO

Only after RFT has a held-out result, starting from the RFT adapter, trained
on one fold and scored on the other. On two cards rollout and update cannot
overlap; `scripts/train_grpo.py` says what that means and what is not built.

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
Phase 4  on a rented 2x H100 node:                    ~8-12 hours
         setup_node.sh, gpu_pipeline.sh smoke / stage1,
         read the audit and the checks, then stage2
Phase 3  serving comparison on the same node           optional
         GRPO only once RFT has a held-out result
```

Regenerate after every phase:

```bash
python scripts/make_report.py
```

It prints how many results remain unmeasured and the exact command for each.
Never hand-fill it.
