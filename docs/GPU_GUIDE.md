# Your first GPU run: post-training on one B200

This takes you from "never used a cloud GPU" to a trained, evaluated model and
an updated `RESULTS.md`. Everything you type is in a code block; everything
else explains why. Web consoles get redesigned more often than command-line
tools: if a button below has moved or been renamed, look for its nearest
equivalent; the commands still hold.

## Which GPU

Two models run at once: the agent being trained, **Qwen3.8-27B** in bf16
(52 GiB of weights), and the simulated customer, **Gemma 4 31B** in FP8
(31 GiB). Each also needs a KV cache for ~24 conversations at a time.

| one card | memory | holds both models? | Modal $/h | vs two H100s |
|---|---|---|---|---|
| H100 SXM5 | 80 GB | **no**: the weights alone are 83 GiB | 3.95 | — |
| H200 | 141 GB | yes, ~40 GiB left for both caches | 4.54 | ~1.5–2x slower |
| **B200** | 180 GB | **yes, ~80 GiB left for both caches** | 6.25 | about as fast, cheaper per hour |
| B300 | 288 GB | yes, the extra memory goes unused | 7.10 | as fast as B200 |

**Use one B200.** Rollouts are memory-bandwidth bound (B200: 8 TB/s against
3.35 for an H100, 4.8 for an H200) and training is compute bound (B200: ~2.3x
an H100 in bf16), so one B200 does the work of the two H100s this pipeline was
first sized for, at $6.25/h instead of $7.90. An H200 is the fallback when no
B200 is free, or if the smoke test hits a Blackwell-only problem: same code,
smaller caches, longer runs. Not B300: Modal requires CUDA 13.1 or newer for
it, and SGLang 0.5.20 pins torch 2.13, built on CUDA 13.0 (and not Modal's
`B200+`, which may place you on a B300).

On one card both servers share the GPU (`scripts/gpu_plan.py` works out each
one's memory as it starts; every stage logs the plan to `logs/gpu.txt`), and
training runs in one process with the same 32 turns per optimizer step that
two cards would use. With two 80 GB cards, each server gets one; nothing else
changes.

## Three ways to run it

**Part N, a Modal Notebook, is the easiest start**: everything happens in
notebook cells in your browser, including uploading the code and reading the
results. **Part A, Modal jobs started from your laptop**, runs the same stages
detached: they keep going with your laptop closed, restart themselves if Modal
preempts them, and cost ~20% less per hour, which matters for the long stages.
Both use the same Volume, so you can mix them: `pn.launch()` in part N starts a
part-A job from a notebook. **Part B, Lambda**, rents a whole Linux machine you
SSH into; closer to what you will meet at work, and it bills every minute the
machine exists.

| | Modal Notebook (N) | Modal jobs (A) | Lambda (B) |
|---|---|---|---|
| where you type | notebook cells | your laptop's terminal | SSH + tmux |
| a stage runs | in the cell (or as a job, `pn.launch`) | as a detached job | on the VM |
| billing | while the kernel runs | while a stage runs | from launch until you **terminate** |
| one B200, $/h | ~8–9 (GPU + CPU and memory at notebook rates) | ~7 | 1x B200 where offered |
| moving files | Files panel, `pn.pack()` | `--stage pull` | `scp` / `rsync` |
| persistent disk | the Volume (1 TiB/month free) | the same Volume | a filesystem (per GB-month) |
| free credit | $30/month; up to $10k for graduate students on application | same | none |

---

## The task list

**Before any GPU time**
- [ ] 1. Unpack v18 and run the tests on your laptop (step 0)
- [ ] 2. Account, notebook and Volume (N1–N3), or laptop access (A1–A2), or Lambda (B1–B4)
- [ ] 3. The code onto the Volume and into the session (N3–N5)
- [ ] 4. Download the weights, ~90 GB, and run the tests there (N6 / A3–A4 / B8)

**On the GPU**
- [ ] 5. Smoke test, ~1 h: 16 episodes, a projection of the hours and dollars of the rest, a LoRA serving check and a training check (N7 / A5 / B9)
- [ ] 6. Stage 1: split, baseline, collection, training examples (N8 / A6 / B9)
- [ ] 7. Read three files, **with no GPU running** (step R)
- [ ] 8. Stage 2: two LoRA runs, held-out evaluation, optional reference row (N9 / A7 / B10)

**Afterwards**
- [ ] 9. The results onto your laptop, `RESULTS.md` regenerated (N10 / step F)
- [ ] 10. Stop paying: nothing running, then the Volume deleted when you are done (N11 / A8 / B12)

## What runs, and roughly what it costs

| stage | on the GPU | time on one B200 | as a job (~$7/h) | in a notebook (~$9/h) |
|---|---|---|---|---|
| setup | packages, ~90 GB of weights, tests — no GPU | 30–50 min | cents | ~$1 (CPU notebook) |
| smoke | both models served, 16 episodes, LoRA serving check, training check | 0.8–1.2 h | $6–8 | $7–11 |
| stage 1 | 1,075 baseline + 1,720 collection episodes, examples built | 3–8 h | $21–56 | $27–72 |
| stage 2 | two LoRA runs on Qwen3.8-27B, 1,075 evaluation episodes | 6–12 h | $42–84 | $54–108 |
| serving (phase 3, optional) | both models served three times, 3 × 215 conversations | ~1 h | ~$7 | ~$9 |

On an H200 expect 1.5–2x the hours at ~$5.20/h. The ranges are wide on
purpose: the smoke stage prints a projection from your actual throughput, with
its dollar figure (`logs/smoke-projection.txt`), and a training check with the
seconds per turn (`logs/train-probe.txt`); they beat any estimate here. If the
total is more than you want to spend, lower the repetitions **before stage 1**
and keep them for every later stage, because the runs are compared task by
task:

```bash
export PASAR_K=3            # baseline, evaluation, reference: 3 per task instead of 5
export PASAR_COLLECT_K=4    # rollouts to learn from: 4 per task instead of 8
export PASAR_EPOCHS=1       # LoRA epochs per fold: halves training
```

In a notebook the same knobs are set in a cell before `pn.run("stage1")`:
`import os; os.environ["PASAR_K"] = "3"`. Stage 1 writes the values it used to
`data/run_settings.env` and stage 2 reads them from there, so a new terminal or
session cannot quietly change them; exporting a different `PASAR_K` later is
refused. `PASAR_EPOCHS` (and `PASAR_TRAIN_MAX_LEN`, below) may still be changed
for stage 2. The simulated customer is kept the same way: stage 1 records
`PASAR_CUSTOMER` — 2 by default, the customer told to wait for what she agreed
to (WHAT_FAILED #38) — and stage 2 and the reference row use it. A stage 1 from
before v31 recorded none; its runs had the old customer and the old tools
(#37), and no stage runs on it: run round 2 (below), which keeps it, or move
`data/` and `traces/P-*` aside and run stage 1 again.

The optional reference row calls deepseek-v4-pro through its API against the
same customer; it only needs your DeepSeek key.

---

## Round 2 — the second fine-tune (v44)

The same stages again, with the training data balanced by trap (WHAT_FAILED
#31) and today's customer and tools, plus the base model on the 62 newer tasks
of the five hardest traps (RUNBOOK phase 4, round 2). Everything it writes has
a name of its own — `traces/P2-*`, `data/rft2`, `checkpoints/rft2-*`,
`logs/r2-*`, `data/run2_settings.env` — so round 1's files stay as they are on
the Volume and on your laptop. Round 1's settings file records no customer, and
no round-1 stage will run on it any more; round 2 does not need it.

| stage | what runs | one B200 | as a job | in a notebook |
|---|---|---|---|---|
| smoke | the new version's check, as before | ~1 h | ~$7 | ~$9 |
| stage 1, round 2 | 1,075 base episodes, 310 on the newer tasks, 1,720 to learn from, examples balanced by trap | ~3.5–4 h | ~$25–28 | ~$32–36 |
| stage 2, round 2 | two LoRAs, 1,075 held-out episodes (+ the reference row if keyed) | ~2.5–3 h | ~$18–21 | ~$23–27 |

The hours are round 1's, measured; the smoke stage projects yours.

**In a notebook** (part N), after the new version is unpacked onto the Volume
(N3–N5):

```python
pn.run("smoke", round=2)       # round 1's settings refuse every stage now, the smoke test too
pn.run("stage1", round=2)      # or pn.launch("stage1", round=2): a detached job, notebook may stop
pn.read("logs/r2-check-A.txt") # step R, with round 2's files (below)
pn.run("stage2", round=2)
pn.pack()                      # round 1's and round 2's results, one archive
```

**As Modal jobs** (part A), from your laptop:

```bash
PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage smoke
PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage stage1
modal run scripts/modal_pipeline.py --stage pull          # read step R's files
PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage stage2
modal run scripts/modal_pipeline.py --stage pull
```

**On Lambda** (part B), in tmux: `PASAR_ROUND=2 bash scripts/gpu_pipeline.sh
smoke`, then `stage1` and `stage2` the same way.

Step R for round 2 reads `logs/r2-sim_audit.txt`, `logs/r2-check-A.txt` and
`data/rft2/stats.json`, whose `episodes_by_trap` shows how many episodes each
trap gave each fold; `logs/r2-build.log` names any trap below the quota of 20
for want of passing episodes. Then send me the pulled results (or say done):
the reading is fixed in RUNBOOK phase 4, round 2.

---

## GRPO — after round 2 (v45; v46 leaves the photo trap out)

One stage after round 2's stage 2, on the same Volume. For each fold it starts
from round 2's RFT adapter: eight rollouts at temperature 1 on each of the
fold's tasks the base model failed at least once in round 2's collection —
none of the photo trap's — their advantages, one policy-gradient pass. Then the
new adapters are evaluated beside the RFT adapters and the base model, on the
held-out fold and on the 62 newer tasks no model trained on (RUNBOOK phase 4,
GRPO, has the reading).

**Two things first.** Put this version on the Volume only after round 2's
stage 2 has finished: a running stage reads its scripts as it goes. And send me
round 2's results before starting — if the base model passes 95% or more of
the newer tasks, GRPO has nowhere to show a gain and waits for harder tasks.
Round 2's did (2026-10-12): 0.923, so GRPO runs — with v46, which leaves out
the photo trap, whose customer sends photos she was told she has none of
(WHAT_FAILED #43). Put v46 on the Volume before the first `grpo` run: the tasks
are picked once, and the run records which traps it left out
(`PASAR_GRPO_SKIP_TRAPS`; a later run with another value is refused).

| what runs | one B200 | as a job | in a notebook |
|---|---|---|---|
| one iteration a fold: rollouts, then the update (servers stopped), A then B | ~1.5–3 h | ~$11–21 | ~$14–27 |
| evaluation: 1,075 held-out episodes, then 310 + 310 on the newer tasks | ~2–2.5 h | ~$15–18 | ~$18–23 |

The first figure depends on how many tasks the collection failed at least once:
the stage prints them by fold and trap as it starts (`logs/r2-grpo-tasks.log`),
with a "left out" line for the photo trap's, and stops if a fold has fewer than
six.

**In a notebook** (part N), after the new version is unpacked onto the Volume
(N3–N5):

```python
pn.run("grpo", round=2)                 # or pn.launch("grpo", round=2): a detached job
pn.read("logs/r2-grpo-tasks.log")       # the tasks each fold trains on, by trap
pn.read("logs/r2-grpo-compare-i1.txt", head=80)   # the result
pn.pack()
```

**As Modal jobs** (part A), from your laptop:

```bash
PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage grpo
modal run scripts/modal_pipeline.py --stage pull
```

**On Lambda** (part B), in tmux: `PASAR_ROUND=2 bash scripts/gpu_pipeline.sh grpo`.

A second iteration, later, continues from the first one's adapters and is
evaluated under its own names (`P2-grpo-i2`): `os.environ["PASAR_GRPO_ITERS"] =
"2"` in a notebook cell before `pn.run`, or `PASAR_GRPO_ITERS=2` before
`modal run`. If the stage stops with "nothing to train on", the policy has no
task left that it sometimes fails: that is a result, not an error — send me the
logs.

On the laptop, after `pn.pack()` and the download, `python scripts/make_report.py`
puts GRPO's result in RESULTS.md, section 6 (RUNBOOK phase 4, GRPO, "Where to
read it").

Round 2's first iteration ran on 2026-10-12: INCONCLUSIVE, and half of the
newer tasks' room turned out to be the voucher trap's customer choosing a human
over the voucher (RUNBOOK phase 4, GRPO, "What it found"; WHAT_FAILED #46). So
a second iteration on the same tasks is not the next run; the customer check
and the hard edition are, below — with v51 or later on the Volume, which reads
that choice beside every comparison the hard edition prints.

**Put a newer version on the Volume only between stages.** A stage reads its
scripts as it goes — bash reads `gpu_pipeline.sh` line by line — so a version
unpacked while GRPO runs can change what the rest of the run does.

## Customer 3 — after GRPO (v47; run 2026-10-12 in ten minutes: she holds)

The simulated customer the next edition will use, checked against today's on
the two traps her new rule reaches (RUNBOOK phase 4, customer 3; WHAT_FAILED
#43). Once GRPO has finished, on the same Volume:

```python
pn.run("customer3", round=2)                      # or pn.launch("customer3", round=2)
pn.read("logs/r2-customer3.txt", head=80)
pn.pack()
```

As a Modal job: `PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage
customer3`, then `--stage pull`. About ten minutes of one B200 (~$2): 280
episodes at temperature 0, with the server start. Its reading on 2026-10-12:
customer 3 holds, and the hard edition runs with her (RUNBOOK phase 4, customer
3).

## The hard edition — after the customer check (v48; run 2026-10-12 in 19 minutes)

The hard edition's 36 combined cases on round 2's models: the base model, the
RFT adapters and GRPO's (RUNBOOK phase 4, the hard edition; 1s says what the
cases are). Read `logs/r2-customer3.txt` first: the stage runs customer 3, and
refuses to until her check has run. If the check says to keep customer 2, set
`PASAR_E3_CUSTOMER=2` before the stage — in the notebook,
`os.environ["PASAR_E3_CUSTOMER"] = "2"`. (Read 2026-10-12: she holds, so nothing
is set.)

```python
pn.run("edition3", round=2)                       # or pn.launch("edition3", round=2)
pn.read("logs/r2-edition3.txt", head=120)
pn.pack()
```

As a Modal job: `PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage
edition3`, then `--stage pull`. About 20 minutes of one B200 (~$3): 540
episodes at temperature 0 and three server starts. Run it again after a failure:
finished runs are not served again. Its reading on 2026-10-12: room, RFT and
GRPO INCONCLUSIVE, the two orders the one hard case (RUNBOOK phase 4, the hard
edition).

## GRPO on its own tasks — optional (v52; run 2026-10-12 in 22 minutes)

Whether GRPO's one iteration learned the tasks it trained on, which its
evaluation cannot say (RUNBOOK phase 4, GRPO, the learning curve's second
point; its reading is fixed there). The GRPO adapters roll out on their own
training tasks as the next iteration would — 488 conversations at temperature
1 — and are built and counted, not trained on and not evaluated. Independent of
the customer check and the hard edition; any order.

```python
pn.run("grpo-measure", round=2)                   # or pn.launch("grpo-measure", round=2)
pn.read("logs/r2-grpo-measure-i2.txt", head=60)
pn.pack()
```

As a Modal job: `PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage
grpo-measure`, then `--stage pull`. About 25 minutes of one B200 (~$3). Its
rollouts stay on the Volume, as GRPO's do; the comparison and the stats come
back with `pull`. Its reading on 2026-10-12: NOT SHOWN (RUNBOOK phase 4, GRPO,
the learning curve's second point).

## Serving — phase 3, optional (v68)

What serving the agent costs, three ways, on one card (RUNBOOK phase 3; its
reading is fixed there): the base agent and the customer, the 215 tasks at the
sweeps' concurrency, three times — the policy in the prompt with SGLang's
radix cache on, as every run so far; the policy looked up instead (`jit`); and
the policy in the prompt with the agent's radix cache off. The base model only;
any order with the other stages.

```python
pn.run("serving", round=2)                        # or pn.launch("serving", round=2)
pn.read("logs/r2-serving.md", head=120)
pn.pack()
```

As a Modal job: `PASAR_ROUND=2 modal run --detach scripts/modal_pipeline.py --stage
serving`, then `--stage pull`. About an hour of one B200 (~$7): both servers
start three times, and each bar is 215 conversations after three warm-up ones.
`PASAR_SERVE_SAMPLE=1` tries it on 16 tasks a bar first (~20 minutes, mostly
servers starting); `PASAR_SERVE_K=2` doubles each bar. A bar cut short starts
over when the stage runs again, and bars measured are kept. If the agent's
server will not start without its radix cache, the stage stops at that last
bar, the other two measured and the report written.

---

## Step 0 — on your laptop, before anything else

```bash
cd ~/Downloads && tar -xf pasarbench-0929-v18.tar
cd pasarbench && rm -f scripts/serve_vllm.sh      # removed in v15; tar does not delete files
bash run_tests.sh                                  # expect ALL GREEN
```

---

## Part N — a Modal Notebook

A Modal Notebook is a Jupyter notebook running in Modal's cloud, with the
hardware you pick in its sidebar. Two things to know before you start:

- **The container is new every session.** When the kernel stops (idle
  timeout, a compute change, a restart), installed packages and anything
  outside the Volume are gone. The Volume keeps the repo, the weights and every
  output, so each session starts with the same setup cell (N5), which
  reinstalls the packages in 2–4 minutes.
- **You pay while the kernel runs**, for its GPU at the normal price and for
  its CPU and memory at about three times a job's rate. Set the compute to CPU
  only for setup and reading, and to the B200 only for the stages.

### N1. Account (10 minutes, once)

1. Sign up at [modal.com](https://modal.com) (a GitHub login is fine).
2. Add a payment method in the workspace's billing settings. The Starter plan
   includes $30 of free compute a month.
3. Worth ten minutes: Modal gives graduate students up to $10k of credits (the
   application is linked from [modal.com/pricing](https://modal.com/pricing)).
   Apply with your NTU email; it can cover this whole project.
4. Optional, for the reference row in stage 2: at
   [modal.com/secrets](https://modal.com/secrets) create a secret named
   `deepseek` with one key, `DEEPSEEK_API_KEY` (the key you rotated). A
   `huggingface` secret with `HF_TOKEN` is optional too: the models are not
   gated, a token only speeds the download.

### N2. The notebook and the Volume (5 minutes, once)

1. Go to [modal.com/notebooks](https://modal.com/notebooks) → **New notebook**;
   name it `pasarbench`.
2. In the sidebar's **Files** panel, create a Volume named `pasarbench` and
   attach it. It appears in the notebook as `/mnt/pasarbench`. (If you already
   made one with `modal run` in part A, attach that one: it is the same
   Volume.)
3. In the **compute profile**: no GPU yet, 4 CPUs, 16 GiB of memory.
4. Leave the idle shutdown at its default (10 minutes) for now.
5. If you made the secrets in N1, attach them in the sidebar; they become
   environment variables in the notebook.

### N3. Upload the code (1 minute, once per version)

Drag `pasarbench-0929-v18.tar` from Finder onto the `pasarbench` Volume in the
Files panel, at its top level. (From a laptop with the Modal CLI instead:
`modal volume put pasarbench ~/Downloads/pasarbench-0929-v18.tar /`.)

### N4. Unpack it (once per version)

In the first cell:

```python
import glob, os, tarfile
tar = sorted(glob.glob("/mnt/*/pasarbench-*.tar"))[-1]
with tarfile.open(tar) as t:
    t.extractall(os.path.dirname(tar), **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
print("unpacked", tar)
```

This makes `/mnt/pasarbench/pasarbench/`, the working copy every stage runs
in. A newer version unpacks over it and keeps traces, data and checkpoints.

### N5. The setup cell (every session)

```python
import glob, sys
sys.path.insert(0, glob.glob("/mnt/*/pasarbench/scripts")[0])
import pasar_notebook as pn
pn.setup()
```

It finds the Volume, links it to `/vol` (the path Modal jobs use, so part A
and part N share files and paths), installs SGLang 0.5.20, the training
packages and a CUDA 13 compiler into the container (3–5 minutes; once per
session), builds a test kernel with that compiler (SGLang builds some kernels
when a server starts, and needs nvcc 12.9 or newer) and prints the GPU, the
package versions and, with a GPU attached, the memory plan. Look for
`CUDA compiler 13.0 (pip) … a test kernel builds for sm_100a` on a B200.

### N6. Weights and tests (CPU, 30–50 minutes, once)

```python
pn.download()     # Qwen3.8-27B (~56 GB), Gemma 4 FP8 (~33 GB), Gemma's tokenizer -> /vol/hf
pn.test()         # must end ALL GREEN
```

The download resumes if interrupted: run the cell again. If the tests are not
all green, stop and send me the output; it is far cheaper to fix now.

### N7. Smoke test (one B200, ~1 hour, ~$9)

1. Compute profile: **GPU B200, count 1**; 8 CPUs, 64 GiB of memory. The
   kernel restarts on the new hardware.
2. Run the setup cell (N5) again, then:

```python
pn.run("smoke")
```

The output streams into the cell and into `logs/notebook-smoke.log` on the
Volume. First the agent loads, then the customer in the memory the agent left
(`agent ready`, `customer ready`, and the memory in use), then 16 episodes.
Last, two checks that stage 2 depends on: the agent is restarted with a small
random LoRA adapter to prove SGLang loads, routes to and applies adapters on
Qwen3.8, and a few training steps run on one turn of the longest length stage 2
will train on, to measure its memory and speed.

Then read, in a cell:

```python
pn.read("logs/gpu.txt")                    # the plan and the memory in use
pn.read("logs/smoke-projection.txt")       # hours and dollars for stages 1 and 2
pn.read("logs/sim_audit-smoke.txt", head=30)   # section 0 must say "none" twice
pn.read("logs/lora-probe.txt", tail=2)     # must say "LoRA serving works"
pn.read("logs/train-probe.txt", tail=6)    # must say "Fits"
```

Anything other than "none" in section 0, "LoRA serving works" in the probe, or
"Fits" in the training check means stop and send me the file (and
`logs/probe.log` for the probe): each is cheap to fix now and expensive after
stage 1. If a server fails with an error that mentions `sm100`, Blackwell or
a CUDA kernel, switch the compute profile to **H200** and run the smoke test
again: the same code runs there on Hopper kernels.

Set the compute back to CPU when you stop for more than a few minutes.

### N8. Stage 1 (hours)

Two ways; pick one.

**In the notebook** (simplest): GPU B200 in the compute profile, setup cell,
then

```python
pn.run("stage1")
```

Keep the tab open while it runs if you can. If you come back to a stopped
kernel, run the setup cell and the same `pn.run("stage1")` again: it resumes
where it stopped, and episodes a server error ended are re-run. From a new
session, `pn.status()` shows how far it got.

**As a detached job** (sturdier and cheaper for hours of work): in a CPU-only
notebook, setup cell, then

```python
pn.launch("stage1")
```

It starts `modal run --detach scripts/modal_pipeline.py --stage stage1`, the
part-A job, on its own B200, and returns once the stage has begun (the first
time, Modal builds the image: ~10 minutes). Close the notebook; the job runs
until the stage ends, at the job rate, and Modal restarts it if it is
preempted. Follow its log at [modal.com/apps](https://modal.com/apps), or with
`pn.status()` in a later session (a running notebook sees the Volume as it was
when its kernel started). If `pn.launch` ends with an error instead, run the
command from your laptop (A2, A6).

Either way, a second stage started while one runs is refused (they would
write the same files), and a stage that ends with `!! CHECK FAILED` will not
change by running it again: read `logs/check-A.txt` and `logs/check-B.txt` and
send them to me. When it prints "Stage 1 done", do **step R** with the
compute set to CPU:

```python
pn.read("logs/sim_audit.txt", head=60)
pn.read("logs/check-A.txt")
pn.read("data/rft/stats.json")
```

### N9. Stage 2 (hours)

Attach the `deepseek` secret (N1) first if you want the reference row; without
it, stage 2 skips the reference and says so. Then, as in N8, either

```python
pn.run("stage2")          # B200 in the compute profile
```

or, from a CPU notebook, `pn.launch("stage2")`. Training prints its progress
and an ETA per fold (`pn.read("logs/train-A.log", tail=5)`); when the stage
prints "Stage 2 done", go on.

### N10. The results onto your laptop

```python
pn.pack()
```

It writes `results-<date>.tar.gz` (logs, splits, settings, GRPO's stats and
every evaluation's traces; the multi-GB collection runs and GRPO's rollouts stay
on the Volume) and says how to download it: from the Files panel when it is
under 16 MB (the web UI's limit), otherwise with `modal volume get` from your
laptop. On the laptop:

```bash
cd ~/Downloads && tar -xzf results-<date>.tar.gz    # lands in ~/Downloads/pasarbench/
```

then **step F**.

Since v53 a pack holds only what is new or changed since the last one — after a
stage, its traces and the logs — so extract each over the one before it, as
above. It notes what the packs hold in `.packed.json` beside them on the Volume;
the first pack after v53 reads that from the newest `results-*.tar.gz` instead.
`pn.pack(full=True)` packs every result again, for a fresh copy of the repo.
Before v53 every pack held every result: some ten thousand trace files by round
2, each a round trip to read off the Volume, one at a time — 20 minutes, with
the GPU still attached. They are now read 32 at a time.

### N11. Stop paying

- Set the compute to CPU, or stop the kernel, whenever no stage is running.
  An idle kernel shuts down by itself after the idle timeout.
- A job started with `pn.launch` runs on its own: see and stop it at
  [modal.com/apps](https://modal.com/apps).
- When the project is finished and the results are on your laptop, delete the
  Volume (Files panel, or `modal volume delete pasarbench`). It is free up to
  1 TiB a month, so there is no rush.

---

## Part A — Modal jobs from your laptop

### A1. Account (10 minutes, once)

As N1.

### A2. Laptop setup (5 minutes, once)

In the conda environment you use for this repo:

```bash
pip install modal
modal setup          # opens your browser; approve, and a token is saved in ~/.modal.toml
```

`modal run` now works from any terminal. Nothing else is installed locally: the
GPU environment is described in `scripts/modal_pipeline.py` and built in Modal's
cloud the first time you use it (about 10 minutes, then cached).

The GPU is one B200 unless you say otherwise, in the terminal that starts a
stage: `export PASAR_GPU=H200` (the fallback) or `export PASAR_GPU=H100:2` (two
cards). A request that cannot hold both models is refused before anything is
billed.

### A3. Download the weights (no GPU, 20–40 minutes)

```bash
cd ~/Downloads/pasarbench
modal run scripts/modal_pipeline.py --stage download
```

This builds the image, then downloads Qwen3.8-27B (~56 GB), the FP8 Gemma 4
(~33 GB) and Google's Gemma tokenizer onto a Volume named `pasarbench`. It runs
on CPUs, so it costs cents. Run it once; later stages read the weights from the
Volume.

### A4. Check the job's environment (no GPU, ~5 minutes)

```bash
modal run scripts/modal_pipeline.py --stage test
```

The same test suite you ran in step 0, inside the image the GPU stages use. It
must end in `ALL GREEN`. If it does not, stop and send me the output.

### A5. Smoke test (one B200, ~1 hour, ~$7)

```bash
modal run --detach scripts/modal_pipeline.py --stage smoke
```

The terminal streams the log. `--detach` means you may close the terminal or the
laptop and the stage carries on; find it again at
[modal.com/apps](https://modal.com/apps) (click the app, then Logs). N7 says
what happens and what to read; bring the logs home first:

```bash
modal run scripts/modal_pipeline.py --stage pull
cat logs/gpu.txt logs/smoke-projection.txt
head -30 logs/sim_audit-smoke.txt      # section 0 must say "none" twice
tail -2 logs/lora-probe.txt            # must say "LoRA serving works"
tail -6 logs/train-probe.txt           # must say "Fits"
```

### A6. Stage 1 (one B200, hours)

First make sure nothing is still running (a closed terminal does not stop a
`--detach` stage), then start it:

```bash
modal app list                 # no "pasarbench" app running?
modal run --detach scripts/modal_pipeline.py --stage stage1
```

Close the laptop if you like. The stage ends by printing "Stage 1 done". If it
fails, times out (24 h limit) or is preempted, run the same command again: it
resumes where it stopped (episodes a server or API error ended are re-run).
Modal also restarts a preempted stage by itself. A second stage started while
one is running — here or in a notebook — is refused: they would write the same
files.

One failure is different: if it ends with `!! CHECK FAILED`, running it again
changes nothing. Pull, read `logs/check-A.txt` and `logs/check-B.txt`, and send
them to me.

```bash
modal run scripts/modal_pipeline.py --stage pull
```

Now do **step R** below. Nothing is running while you read, so nothing is
billed.

### A7. Stage 2 (one B200, hours)

For the reference row, export your DeepSeek key in this terminal first (use the
key you rotated; it travels to the job as a Modal secret and is never written
to a file). Without it, stage 2 skips the reference and says so.

```bash
export DEEPSEEK_API_KEY=sk-...          # optional
modal app list                          # nothing running?
modal run --detach scripts/modal_pipeline.py --stage stage2
```

You do not need to export `PASAR_K` again: stage 2 reads what stage 1 used.

Training prints a progress bar with an ETA for each fold (`logs/train-A.log`,
`logs/train-B.log`). When the stage prints "Stage 2 done":

```bash
modal run scripts/modal_pipeline.py --stage pull
```

and go to **step F**.

### A8. Stop paying

Nothing runs between stages, so nothing is billed. To check, and to stop a
stage you started by mistake:

```bash
modal app list                 # anything "running"?
modal app stop <app-id>        # stops it; everything written so far stays on the Volume
```

When the project is finished **and** you have pulled everything you want:

```bash
modal volume delete pasarbench     # weights, traces, checkpoints: all gone
```

### Useful Modal commands

```bash
modal volume ls pasarbench pasarbench/logs              # what is on the Volume
modal volume get pasarbench pasarbench/checkpoints/rft-A .       # any file or folder, into .
modal shell scripts/modal_pipeline.py::run_stage        # a shell on the stage's GPU (billed while open)
```

---

## Part B — Lambda

### B1. Account (10 minutes, once)

Sign up at [lambda.ai](https://lambda.ai) and add a payment method. On-demand
GPUs are billed by the minute.

### B2. An SSH key (5 minutes, once, on your Mac)

An SSH key is how the machine knows it is you. Create one and print its public
half:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/lambda -C lambda     # press Enter twice (no passphrase) or set one
cat ~/.ssh/lambda.pub
```

In the Lambda Cloud console: **SSH keys → Add SSH key**, paste the line that
`cat` printed, name it `mac`. Never share `~/.ssh/lambda` (without `.pub`).

### B3. A filesystem (2 minutes, once)

An instance's own disk disappears when you terminate it. A **filesystem** does
not, so the repo, the Python environment and the ~90 GB of weights live there.

Console: **Storage → Create filesystem**, name `pasar`, and pick the region
where you will launch (a filesystem only attaches to instances in its own
region; check on the Instances page which regions have the GPUs free before
you choose). It is billed per GB-month for as long as it exists.

### B4. Launch the instance (billing starts here)

Console: **Instances → Launch instance**:

1. Type: **1x B200** if it is listed, otherwise **2x H100 (80 GB SXM)** — not
   1x H100 or 1x GH200 (96 GB), which cannot hold both models. If none is
   free, try later, or use Modal.
2. Region: the filesystem's.
3. Filesystem: `pasar`.
4. Image: **Lambda Stack 24.04**.
5. SSH key: `mac`.

Wait for the status "Running" and copy its IP address. From now on it bills
until you terminate it.

### B5. Connect and check the driver

```bash
ssh -i ~/.ssh/lambda ubuntu@<IP>          # answer "yes" the first time
nvidia-smi                                # the GPUs; read "Driver Version"
```

The driver must be **580 or newer** (the top line also says `CUDA Version: 13.x`):
SGLang 0.5.20 is built for CUDA 13. If it is older, do not try to upgrade it:
terminate the instance (step B12) and use Modal, whose driver is 580.

### B6. tmux, so nothing dies with your connection

Stages run for hours; your Wi-Fi will not stay up that long. tmux keeps them
running on the machine.

```bash
tmux new -s pasar          # a session; everything below runs inside it
```

- detach and leave it running: `Ctrl-b`, then `d`
- come back later (after `ssh` again): `tmux attach -t pasar`
- scroll back in the output: `Ctrl-b`, then `[`, arrows; `q` to stop

### B7. Copy the code over

In a **second terminal on your Mac** (not the SSH one):

```bash
scp -i ~/.ssh/lambda ~/Downloads/pasarbench-0929-v18.tar ubuntu@<IP>:/lambda/nfs/pasar/
```

Back in the SSH terminal (inside tmux):

```bash
cd /lambda/nfs/pasar && tar -xf pasarbench-0929-v18.tar && cd pasarbench
```

### B8. Setup (30–60 minutes, once per filesystem)

```bash
export PASAR_HOME=/lambda/nfs/pasar
bash scripts/setup_node.sh
```

It checks the driver and prints the memory plan for this machine's GPUs (a
warning if they cannot hold both models), builds the Python environment on the
filesystem (SGLang 0.5.20, peft, accelerate, flash-linear-attention), downloads
the weights, runs the tests (must end `ALL GREEN`) and writes `.pasar_env`,
which the pipeline reads. Most of the hour is the download.

### B9. Smoke test and stage 1

```bash
bash scripts/gpu_pipeline.sh smoke
cat logs/gpu.txt logs/smoke-projection.txt   # the plan; hours and dollars for the rest
head -30 logs/sim_audit-smoke.txt       # section 0: "none" twice
tail -2 logs/lora-probe.txt             # "LoRA serving works"
tail -6 logs/train-probe.txt            # "Fits"
bash scripts/gpu_pipeline.sh stage1     # only if all of it looks right (N7 says why)
```

Detach (`Ctrl-b d`) and come back whenever you like. When stage 1 prints "Stage
1 done", do **step R**. (If it ends with `!! CHECK FAILED` instead, running it
again changes nothing: send me `logs/check-A.txt` and `logs/check-B.txt`.)
Reading takes time and the machine bills while it exists, so either read
quickly, or bring the three files home (below), **terminate the instance
(B12)**, and launch a new one with the same filesystem for stage 2 (then
`export PASAR_HOME=/lambda/nfs/pasar && bash scripts/setup_node.sh` again: it
finds the environment and the weights, and takes a few minutes).

To copy files to your Mac, from the Mac terminal:

```bash
cd ~/Downloads/pasarbench && mkdir -p data/rft
rsync -av -e "ssh -i ~/.ssh/lambda" ubuntu@<IP>:/lambda/nfs/pasar/pasarbench/logs ./
rsync -av -e "ssh -i ~/.ssh/lambda" ubuntu@<IP>:/lambda/nfs/pasar/pasarbench/data/rft/stats.json ./data/rft/
```

### B10. Stage 2

```bash
export DEEPSEEK_API_KEY=sk-...          # optional: the reference row
bash scripts/gpu_pipeline.sh stage2
```

### B11. Bring the results home

From the Mac:

```bash
cd ~/Downloads/pasarbench && mkdir -p data/rft
for d in logs data/splits traces/P-smoke traces/P-base traces/P-rft traces/P-ref; do
  rsync -av --relative -e "ssh -i ~/.ssh/lambda" "ubuntu@<IP>:/lambda/nfs/pasar/pasarbench/./$d" ./
done
rsync -av -e "ssh -i ~/.ssh/lambda" ubuntu@<IP>:/lambda/nfs/pasar/pasarbench/data/rft/stats.json ./data/rft/
```

### B12. Stop paying

- **Terminate the instance** whenever no stage is running: console → Instances
  → select → Terminate. The filesystem, and everything on it, stays.
- When the project is finished and your results are on your Mac, delete the
  filesystem too (console → Storage). It bills every month it exists.

---

## Step R — read three files before stage 2

Nothing should be running while you read (on Lambda, see B9).

| file | what to look for |
|---|---|
| `logs/sim_audit.txt` | Section 0 first: chat-format tokens in either side's text mean a server parser is wrong; stop. Then the **leak** rate and the **unanswered-request** rate by language. A language far above English points at the leak detector or the customer, not the agent: read the flagged lines under it (WHAT_FAILED #26). |
| `logs/check-A.txt`, `logs/check-B.txt` | The printed example: the prompt's tail, then `=== COMPLETION (loss)` — the only text the model trains on, which must be the agent's own turn ending in `<\|im_end\|>`, never a tool result. Then the server check: `N of N … token-identical to the ids stored in this file`. It must end `no problems found`. |
| `data/rft/stats.json` | The collection pass rate, and per fold the traps with no passing episode (`traps_with_zero_signal`): the model trained on that fold has nothing to learn there. If the base model already passes ~0.95, there is little headroom; say so in the write-up rather than training anyway. |

If any of the three looks wrong, send it to me before stage 2.

## Step F — the report, on your laptop

```bash
cd ~/Downloads/pasarbench
python scripts/make_report.py --out RESULTS.md --tools-run I-tools2 \
  --multilingual-run C-clean,D-nozh,G-gated --noise-pair H-context/full,I-tools2/full+all-20
```

Section 6 compares the base model, the fine-tunes (each on the fold it did not
train on) and the reference, paired by task, and checks what that comparison
depends on. The only number that counts is held-out pass^k; an INCONCLUSIVE row
is reported as one. Since v19 every verdict is today's checks, re-scored from
the recorded tool calls, with the verdicts as recorded beside them; section 7
and `python scripts/rescore.py` show what a corrected check moved, without a
GPU (WHAT_FAILED #30).

---

## When something goes wrong

| symptom | what it means, what to do |
|---|---|
| `layout: too-small` / `cannot hold the agent … and the customer` | one card under ~126 GiB (an H100, an A100, an RTX PRO 6000). Use one B200 or H200, or two 80 GB cards. |
| Modal: the stage or notebook sits waiting for a GPU | none free right now; it starts when one is. Wait, or try an H200. |
| Modal: image build fails in A3/A4, or `pn.setup()` fails installing | send me the last 30 lines; nothing is billed for GPUs yet. |
| `NVCC version must be at least 12.9`, or `!! no CUDA compiler 12.9 or newer` | SGLang builds some kernels when a server starts and needs a CUDA 12.9+ compiler; v17 used whatever the machine had (a Modal notebook's is older). v18 installs pip's CUDA 13 compiler beside torch and checks it on a test kernel before any model loads: unpack v18, run the setup cell again (it adds the compiler in about a minute) and look for `CUDA compiler 13.0 (pip) … a test kernel builds for sm_100a`, then run the stage again. |
| `!! agent exited while starting` / `customer exited`, or a server dies mid-stage with `out of memory` | the last 25 lines of `logs/agent.log` or `logs/sim.log` are printed under it. `out of memory` on one card: keep more of the card free with `PASAR_RESERVE_GB=16` (export it, or `os.environ["PASAR_RESERVE_GB"] = "16"` in the notebook before `pn.run`) and run the stage again; both caches shrink a little to pay for it, and `logs/gpu.txt` shows the plan it ran with. On two cards: `SGLANG_ARGS="--mem-fraction-static 0.8"`. An error naming `sm100`, Blackwell or a kernel on a B200: rerun on an H200 and send me the lines. |
| `only … GiB free on the GPU for the agent; is another process using it?` | a server from an interrupted run still holds the card: `pn.stop()` in a notebook (or restart the kernel), then run the stage again. |
| a stopped cell ends in `KeyboardInterrupt` | expected: that is how the stop button reaches Python. The cell prints `stopping servers` and waits (up to 3 min) for the stage to shut its servers down; pressing stop again moves to a harder signal but no longer skips the cleanup (since v19). `nvidia-smi` should then show no process; if one remains, `pn.stop()`. |
| setup_node.sh: `!! NVIDIA driver … needs CUDA 13` | the Lambda image's driver is too old; terminate and use Modal. |
| `CUDA out of memory` during training (`logs/train-A.log`), or the training check says `!! within 8%` | turns too long for the card. `PASAR_TRAIN_MAX_LEN=12288` and run stage 2 again: it leaves the longest turns out, says how many, and resumes from the last checkpoint. |
| a stage was killed (preempted, timed out, a kernel stopped, SSH dropped without tmux), or ended with `!! N of M episodes ended with a server or API error` | fix the cause if there is one (the message says where to look), then run the same stage again: it resumes, and re-runs only what did not finish properly. |
| `stage '…' is already running on this Volume` | a stage is still running: `pn.status()`, `modal app list`. Wait for it or stop it (`modal app stop <app-id>`). If it died with its notebook, it stops blocking 15 minutes after its last heartbeat, or `pn.run(stage, force=True)` when you are sure. |
| `pn.launch` ends with an error (credentials, tokens, anything) | run the same stage from your laptop instead (A2, then A6 or A7). |
| `pn.status()` shows no progress while a job runs | a running notebook sees the Volume as it was when the kernel started. Follow the job's log at [modal.com/apps](https://modal.com/apps), or check `pn.status()` in a new session. |
| `!! PASAR_K=…, but stage 1 ran with …` | you set a different repetition count than stage 1 used; unset it. |
| you are not sure whether something is still billing | Modal: [modal.com/apps](https://modal.com/apps) and the notebook's kernel state. Lambda: the Instances page; anything listed there is billing. |
