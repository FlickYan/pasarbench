"""
The GPU pipeline on Modal: the stages of gpu_pipeline.sh, started from your
laptop and run in Modal's cloud. There is no machine to SSH into or to remember
to shut down: you pay per second while a stage runs, and everything it writes
stays on a Modal Volume named `pasarbench`. docs/GPU_GUIDE.md, part A, walks
through every step (part N does the same from a Modal Notebook, on the same
Volume -- the two can be mixed).

    pip install modal && modal setup                        # once: account, token
    modal run scripts/modal_pipeline.py --stage download    # ~90 GB of weights, no GPU
    modal run scripts/modal_pipeline.py --stage test        # the test suite in the image, no GPU
    modal run --detach scripts/modal_pipeline.py --stage smoke     # one B200
    modal run --detach scripts/modal_pipeline.py --stage stage1    # one B200, hours
    modal run scripts/modal_pipeline.py --stage pull        # logs and results -> this repo
    modal run --detach scripts/modal_pipeline.py --stage stage2    # one B200, hours
    modal run scripts/modal_pipeline.py --stage pull

The GPU is one B200 unless PASAR_GPU says otherwise when you run the command:
PASAR_GPU=H200 (one card: cheaper per hour, slower, tighter) or PASAR_GPU=H100:2
(the two-card layout). scripts/gpu_plan.py decides at run time how the agent
and the customer share whatever the job gets.

`--detach` keeps a stage running after you close the terminal or the laptop;
watch it at https://modal.com/apps. Every stage can simply be started again
after a failure, a timeout or a preemption (Modal restarts preempted GPU jobs
on its own): sweeps keep finished episodes, training resumes from its last
checkpoint, finished adapters are skipped.

On the Volume, `pasarbench/` is a working copy of this repo: each run refreshes
its code from your laptop and keeps what earlier runs wrote there (traces/,
data/, checkpoints/, logs/). `hf/` holds the weights.

For the optional reference row (deepseek-v4-pro against the same customer),
export DEEPSEEK_API_KEY in the shell that starts stage2: it travels to the job
as a Modal secret and is never written to a file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# The same pin as scripts/setup_node.sh (a test checks they agree): the version
# whose /tokenize, LoRA naming and Qwen3.8 tool parser this repo was checked
# against.
SGLANG_VERSION = "0.5.20"
VOL = "/vol"
WORK = f"{VOL}/pasarbench"
# What `pull` brings back: what the report and the audits read. The collection
# run (traces/P-collect, several GB) stays on the Volume.
RESULTS = ["logs", "data/splits", "data/rft/stats.json", "data/run_settings.env",
           "traces/P-smoke", "traces/P-base", "traces/P-rft", "traces/P-ref"]
# One GPU stage at a time: two writing the same traces corrupt both, and bill
# twice. The lock is a heartbeat, so a stage that dies without cleaning up
# stops blocking after STALE seconds; Modal's restart of a preempted stage
# keeps its call id and passes.
LOCK = f"{WORK}/logs/.stage-lock.json"
STALE = 15 * 60

# No version pinned: a Volume made from a notebook's Files panel works as well
# as one this script creates.
volume = modal.Volume.from_name("pasarbench", create_if_missing=True)

# One card must hold the agent (52 GiB of bf16 weights) and the customer (31 GiB
# of FP8) with a cache each; with two cards, each holds one. Memory per card, GB.
GPU_MEMORY = {"B300": 288, "B200": 180, "H200": 141, "H100": 80, "A100-80GB": 80,
              "RTX-PRO-6000": 96, "A100-40GB": 40, "A100": 40, "L40S": 48, "A10": 24,
              "L4": 24, "T4": 16}
GPU = os.environ.get("PASAR_GPU", "B200")


def check_gpu(spec: str) -> str:
    """The GPU request, or why this pipeline cannot run on it."""
    kind, _, n = spec.partition(":")
    count = int(n or 1)
    base = kind.rstrip("!").upper()
    if base in ("B300", "B200+"):
        raise SystemExit(
            f"{spec}: B300 needs CUDA 13.1 or newer and SGLang {SGLANG_VERSION}'s torch is "
            f"built on CUDA 13.0 (\"B200+\" may place the job on a B300). Use B200.")
    if base not in GPU_MEMORY:
        raise SystemExit(f"unknown GPU {spec!r}; one of {', '.join(GPU_MEMORY)}")
    mem = GPU_MEMORY[base]
    if count == 1 and mem < 140:
        raise SystemExit(f"one {base} ({mem} GB) cannot hold the agent and the customer: "
                         f"use B200 or H200, or two cards ({base}:2)")
    if mem < 80:
        raise SystemExit(f"{base} ({mem} GB) cannot hold the agent's 52 GiB of weights "
                         f"with a cache: use 80 GB cards or larger")
    return spec

image = (
    modal.Image.debian_slim(python_version="3.12")
    # build-essential: nvcc hands host code to g++ when SGLang builds kernels.
    .apt_install("curl", "procps", "libnuma1", "build-essential")
    # SGLang pins torch, transformers and tokenizers that work together; the
    # training packages are installed around those pins, as setup_node.sh does,
    # with pip's CUDA compiler of torch's CUDA version (scripts/cuda_home.py).
    .pip_install(f"sglang=={SGLANG_VERSION}")
    .run_commands(
        "pip freeze | grep -iE '^(torch|transformers|tokenizers|cuda-toolkit)==' "
        "> /constraints.txt",
        "pip install -c /constraints.txt peft accelerate flash-linear-attention "
        "'cuda-toolkit[nvcc,cccl]'",
    )
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONUNBUFFERED": "1"})
    # Code only: outputs live on the Volume, and your laptop's traces and data
    # (the API runs behind RESULTS.md) have no business in the job -- nor does a
    # .env file of API keys, which would otherwise be copied onto the Volume.
    .add_local_dir(ROOT, "/src", ignore=[
        "traces", "data", "checkpoints", "logs", "modal-logs", ".git", ".pasar_env",
        ".env", ".env.*", "**/.env", "**/.env.*", "**/__pycache__", "*.tar", "*.tar.gz",
        "*.pyc", "RESULTS.md"])
)

app = modal.App("pasarbench", image=image)



def _secrets(*keys: str) -> list:
    """The variables among `keys` exported in the shell that runs `modal run`,
    passed to the job as a Modal secret. None is required and none is written to
    a file; API keys travel this way, and so do the pipeline's settings. (A
    named Modal secret that does not exist would stop the app from starting.)"""
    present = [k for k in keys if os.environ.get(k)]
    return [modal.Secret.from_local_environ(present)] if present else []


def _workdir() -> Path:
    """Refresh the code in the Volume's working copy; keep its outputs."""
    work = Path(WORK)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree("/src", work, dirs_exist_ok=True)
    return work


def _run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


# Settings exported where `modal run` starts, passed on to the job: the models
# (both paths read the same variables), SGLang flags after an OOM, and the
# knobs gpu_pipeline.sh documents (repetitions, epochs, turn length, the
# simulated customer's rules).
PASSED_ON = ("AGENT_MODEL", "SIM_MODEL", "SGLANG_ARGS", "PASAR_K", "PASAR_COLLECT_K",
             "PASAR_EPOCHS", "PASAR_TRAIN_MAX_LEN", "PASAR_CUSTOMER", "PASAR_GPU",
             "PASAR_AGENT_POOL_GB", "PASAR_RESERVE_GB", "PASAR_USD_PER_HOUR")


@app.function(volumes={VOL: volume}, cpu=4, memory=16384, timeout=4 * 3600,
              secrets=_secrets("HF_TOKEN", "AGENT_MODEL", "SIM_MODEL"))
def download() -> None:
    """The agent, the customer and the customer's tokenizer, onto the Volume.
    CPU only: downloading is not worth paying for a GPU to wait on."""
    _run(["bash", "scripts/download_weights.sh"], _workdir())


@app.function(volumes={VOL: volume}, cpu=4, memory=16384, timeout=3600)
def test() -> None:
    """The repo's test suite inside the job's image: the pins resolve, peft and
    the training code work together, the serve and pipeline scripts agree."""
    work = _workdir()
    _run(["python", "-c", "import sglang, peft, fla; print('sglang', sglang.__version__)"], work)
    _run(["python", "scripts/cuda_home.py", "--check"], work)   # the compiler SGLang will use
    _run(["bash", "run_tests.sh"], work)


def _claim(stage: str) -> threading.Event:
    """Take the Volume's stage lock, or refuse to start next to a live stage."""
    me = modal.current_function_call_id() or "unknown"
    lock = Path(LOCK)
    if lock.exists():
        try:
            held = json.loads(lock.read_text())
        except ValueError:
            held = {}
        age = time.time() - held.get("beat", 0)
        if held.get("call") != me and age < STALE:
            raise SystemExit(
                f"stage {held.get('stage')!r} is already running on this Volume (last "
                f"heartbeat {age / 60:.0f} min ago). Two at once would write the same "
                f"traces and bill twice. See `modal app list`; stop it with `modal app "
                f"stop <app-id>`, or if it died, start again after {(STALE - age) / 60:.0f} min.")
    done = threading.Event()
    lock.parent.mkdir(parents=True, exist_ok=True)
    tmp = lock.with_name(lock.name + f".{me}")

    def write() -> None:
        # Written whole, then renamed: a reader never sees half a file.
        tmp.write_text(json.dumps({"call": me, "stage": stage, "beat": time.time()}))
        os.replace(tmp, lock)

    def beat() -> None:
        while not done.wait(60):       # Modal commits the Volume every few seconds
            write()
    write()                            # claimed before the stage starts
    threading.Thread(target=beat, daemon=True).start()
    return done


@app.function(volumes={VOL: volume}, gpu=GPU, cpu=8, memory=32768,
              timeout=24 * 3600, secrets=_secrets("DEEPSEEK_API_KEY", "HF_TOKEN", *PASSED_ON))
def run_stage(name: str) -> None:
    """One stage of gpu_pipeline.sh on PASAR_GPU (one B200 by default). A
    request for H100s may be served by H200s at the H100 price: more memory,
    same code."""
    work = _workdir()
    done = _claim(name)
    try:
        _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
              "--format=csv,noheader"], work)
        _run(["bash", "scripts/gpu_pipeline.sh", name], work)
    finally:
        done.set()
        Path(LOCK).unlink(missing_ok=True)


def pull() -> None:
    """Copy logs and results from the Volume into this repo, where
    make_report.py and inspect_trace.py look for them."""
    from modal.volume import FileEntryType
    got = 0
    for rel in RESULTS:
        parent, name = f"pasarbench/{rel}".rsplit("/", 1)
        try:
            here = {e.path.lstrip("/").rsplit("/", 1)[-1]: e
                    for e in volume.listdir(parent)}
        except Exception:                                   # noqa: BLE001
            here = {}
        if name not in here:
            print(f"  (not on the Volume yet: {rel})")
            continue
        entries = (volume.listdir(f"pasarbench/{rel}", recursive=True)
                   if here[name].type == FileEntryType.DIRECTORY else [here[name]])
        n = 0
        for e in entries:
            if e.type != FileEntryType.FILE:
                continue
            path = e.path.lstrip("/")
            local = ROOT / path.removeprefix("pasarbench/")
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb") as fh:
                volume.read_file_into_fileobj(path, fh)
            n += 1
        got += n
        print(f"  {rel}: {n} files")
    print(f"{got} files into {ROOT}. Read logs/ first (docs/GPU_GUIDE.md says what for).")


@app.local_entrypoint()
def main(stage: str) -> None:
    """--stage is required: a bare `modal run` must not start a GPU."""
    if stage == "download":
        download.remote()
    elif stage == "test":
        test.remote()
    elif stage in ("smoke", "stage1", "stage2"):
        check_gpu(GPU)
        print(f"{stage} on {GPU} (PASAR_GPU to change it); follow it at https://modal.com/apps")
        if stage == "stage2" and not os.environ.get("DEEPSEEK_API_KEY"):
            print("(no DEEPSEEK_API_KEY exported: stage 2 runs without the reference row)")
        run_stage.remote(stage)
    elif stage == "pull":
        pull()
    else:
        raise SystemExit(f"unknown stage {stage!r}: download | test | smoke | stage1 | "
                         f"stage2 | pull")
