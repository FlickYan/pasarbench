"""
PasarBench from a Modal Notebook: the stages of gpu_pipeline.sh, run from
notebook cells on the GPU attached to the notebook. docs/GPU_GUIDE.md, part N,
is the walkthrough.

The notebook's container is new every session; the Volume attached to it keeps
the repo, the weights and every output. So each session starts with this cell:

    import glob, sys
    sys.path.insert(0, glob.glob("/mnt/*/pasarbench/scripts")[0])
    import pasar_notebook as pn
    pn.setup()          # finds the Volume, installs the pinned packages (2-4 min)

and then any of:

    pn.download()       # once: ~90 GB of weights onto the Volume (a CPU notebook will do)
    pn.test()           # the test suite, on the CPU
    pn.run("smoke")     # a GPU stage, in this cell: smoke | stage1 | stage2
    pn.launch("stage1") # ...or as a detached Modal job on its own GPU; the notebook may stop
    pn.status()         # what is running or done (a notebook sees other containers'
                        # writes as of its start: check a job from a new session)
    pn.read("logs/smoke-projection.txt")    # print a file (head=, tail= for long ones)
    pn.stop()           # stop servers a stopped cell left behind
    pn.pack()           # the results as one archive to download

The paths are the ones Modal jobs use: /vol links to the Volume, so /vol/hf
holds the weights and /vol/pasarbench the repo, and a stage started here and
one started with `modal run scripts/modal_pipeline.py` share the files and the
same stage lock (two stages at once would write the same traces).

Standard library only: this module is what installs everything else.
"""

from __future__ import annotations

import codecs
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

# The same pin as scripts/modal_pipeline.py and scripts/setup_node.sh (a test
# checks all three agree).
SGLANG_VERSION = "0.5.20"
MNT = Path(os.environ.get("PASAR_NB_MNT", "/mnt"))      # where notebooks attach Volumes
LINK = Path(os.environ.get("PASAR_NB_LINK", "/vol"))    # where Modal jobs mount it
# On the container's own disk: fast to import from, gone with the container.
VENV = Path(os.environ.get("PASAR_NB_VENV") or
            ("/opt/pasar" if os.access("/opt", os.W_OK) else Path.home() / ".pasar-venv"))
REPO = "pasarbench"
LOCK = "logs/.stage-lock.json"
STALE = 15 * 60
STAGES = ("smoke", "stage1", "stage2")
# What pack() archives: what the report and the audits read (as modal_pipeline.pull).
RESULTS = ["logs", "data/splits", "data/rft/stats.json", "data/run_settings.env",
           "traces/P-smoke", "traces/P-base", "traces/P-rft", "traces/P-ref"]
WEB_DOWNLOAD_LIMIT = 16 * 2**20    # the Modal web UI downloads files up to 16 MB

_state: dict = {}


def _say(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# where things are

def find_volume(mnt: Path = MNT) -> Path:
    """The attached Volume that holds pasarbench/ (the extracted tarball)."""
    hits = sorted(p.parents[2] for p in mnt.glob(f"*/{REPO}/scripts/gpu_pipeline.sh"))
    if not hits:
        tars = sorted(mnt.glob("*/pasarbench-*.tar"))
        hint = (f"found {tars[-1]}: extract it first (guide, step N4)" if tars else
                "attach the Volume (Files panel), upload the tarball to it and extract it "
                "(guide, steps N2-N4)")
        raise SystemExit(f"no {REPO}/ on any Volume under {mnt}: {hint}")
    if len(hits) > 1:
        _say(f"(several Volumes hold {REPO}/: using {hits[0]}; detach the others to be sure)")
    return hits[0]


def link_volume(volume: Path, link: Path = LINK) -> Path:
    """/vol -> the Volume, so every path matches a Modal job's. Falls back to
    the mount itself where the link cannot be made."""
    try:
        if link.is_symlink():
            if link.resolve() == volume.resolve():
                return link
            link.unlink()
        elif link.exists():
            return link if (link / REPO / "scripts").is_dir() else volume
        link.symlink_to(volume, target_is_directory=True)
        return link
    except OSError:
        return volume


def paths() -> dict:
    if not _state:
        raise SystemExit("run pn.setup() first (the first cell of every session)")
    return _state


def _gpus() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# setup: paths, environment, packages

def _uv() -> str | None:
    """A uv of 0.12 or newer, or None. SGLang 0.5.20 pins pre-releases
    (cuda-tile 1.6.0rc5, flash-attn-4 4.0.0b18); older uv refuses to resolve
    them, and --prerelease=allow would pull pre-releases of everything else.
    The image's uv is used when it is new enough, else one is installed
    beside the venv, outside the kernel's own packages."""
    def version(uv: str) -> tuple:
        try:
            out = subprocess.run([uv, "--version"], capture_output=True, text=True).stdout
            return tuple(int(x) for x in out.split()[1].split(".")[:2])
        except (OSError, IndexError, ValueError):
            return (0, 0)
    found = shutil.which("uv")
    if found and version(found) >= (0, 12):
        return found
    tool = VENV.parent / ".pasar-uv"
    uv = tool / "bin" / "uv"
    if not uv.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(tool),
                        "uv>=0.12"], capture_output=True)
    return str(uv) if uv.exists() and version(str(uv)) >= (0, 12) else None


def _venv_ready() -> bool:
    py = VENV / "bin" / "python"
    if not py.exists():
        return False
    r = subprocess.run([str(py), "-c", f"import sglang, peft, accelerate, fla; "
                        f"assert sglang.__version__ == {SGLANG_VERSION!r}"],
                       capture_output=True)
    return r.returncode == 0


def _has_compiler() -> bool:
    """pip's CUDA compiler in the venv (installed since v18)."""
    return any(VENV.glob("lib/python3*/site-packages/nvidia/cu*/bin/nvcc"))


def _installer() -> tuple[list[str], list[str]]:
    """(install, freeze) commands for the venv: uv 0.12+ when there is one, else pip."""
    uv, py = _uv(), str(VENV / "bin" / "python")
    if uv:
        return [uv, "pip", "install", "-q", "--python", py], [uv, "pip", "freeze", "--python", py]
    return [py, "-m", "pip", "install", "-q"], [py, "-m", "pip", "freeze"]


def _constraints(freeze: list[str]) -> str:
    """Hold torch, transformers, tokenizers and torch's CUDA where SGLang put them."""
    frozen = subprocess.run(freeze, capture_output=True, text=True, check=True).stdout
    pins = [l for l in frozen.splitlines()
            if l.split("==")[0].lower() in ("torch", "transformers", "tokenizers",
                                            "cuda-toolkit")]
    path = VENV / "constraints.txt"
    path.write_text("\n".join(pins) + "\n")
    return str(path)


def _install_compiler() -> None:
    """Only the CUDA compiler, into a venv that has everything else (a v17 one)."""
    _say("installing pip's CUDA compiler into the existing environment (~1 min)")
    pip, freeze = _installer()
    subprocess.run([*pip, "-c", _constraints(freeze), "cuda-toolkit[nvcc,cccl]"], check=True)


def _install() -> None:
    """SGLang first, then the training packages held to the torch, transformers
    and tokenizers it pinned -- as setup_node.sh and the Modal image do."""
    t0 = time.time()
    _say(f"installing SGLang {SGLANG_VERSION}, peft, accelerate, flash-linear-attention and "
         f"a CUDA compiler into {VENV} (3-5 min; the container is new each session, so once "
         f"per session)")
    uv = _uv()
    py = str(VENV / "bin" / "python")
    run = lambda *c: subprocess.run(list(c), check=True)  # noqa: E731
    shutil.rmtree(VENV, ignore_errors=True)           # a half-finished earlier attempt
    if uv:
        # Python 3.12, as the Modal image (uv fetches it if the image has another)
        run(uv, "venv", "-q", "--python", "3.12", str(VENV))
    else:
        # pip accepts pre-releases that a requirement pins exactly
        run(sys.executable, "-m", "venv", str(VENV))
        run(py, "-m", "pip", "install", "-q", "-U", "pip")
    pip, freeze = _installer()
    run(*pip, f"sglang=={SGLANG_VERSION}")
    # The training packages, and a CUDA compiler of torch's CUDA version: SGLang
    # builds kernels when a server starts, and the notebook image's own nvcc is
    # older than the 12.9 they need (scripts/cuda_home.py).
    run(*pip, "-c", _constraints(freeze), "peft", "accelerate", "flash-linear-attention",
        "cuda-toolkit[nvcc,cccl]")
    # SGLang's kernels load libnuma; slim images leave it out.
    if not any(Path(d, "libnuma.so.1").exists()
               for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib")):
        if shutil.which("apt-get"):
            subprocess.run("apt-get update -qq && apt-get install -y -qq libnuma1 curl",
                           shell=True, check=False)
    _say(f"installed in {(time.time() - t0) / 60:.1f} min")


def setup(install: bool = True) -> dict:
    """Paths and environment for this session; installs the packages if this
    container does not have them yet."""
    volume = find_volume()
    root = link_volume(volume)
    work, hf = root / REPO, root / "hf"
    (work / "logs").mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "HF_HOME": str(hf),
        "PYTHONUNBUFFERED": "1",
        "PYTHON": str(VENV / "bin" / "python"),
        "VIRTUAL_ENV": str(VENV),
        "PATH": f"{VENV / 'bin'}:{os.environ.get('PATH', '')}",
    })
    _state.update({"volume": volume, "root": root, "work": work, "hf": hf,
                   "name": volume.name})
    gpus = _gpus()
    if gpus:
        driver = gpus[0].rsplit(",", 1)[-1].strip()
        if int(driver.split(".")[0]) < 580:
            _say(f"!! NVIDIA driver {driver}: SGLang {SGLANG_VERSION} needs 580 or newer")
    if install and not _venv_ready():
        _install()
    elif install and not _has_compiler():
        _install_compiler()
    elif not install:
        _say("(packages not checked: install=False)")
    if install and not shutil.which(os.environ.get("CXX", "c++")) and shutil.which("apt-get"):
        _say("installing a C++ compiler (nvcc needs one for host code)")
        subprocess.run("apt-get update -qq && apt-get install -y -qq build-essential",
                       shell=True, check=False)
    if (VENV / "bin" / "python").exists():
        cuda = subprocess.run([str(VENV / "bin" / "python"), "scripts/cuda_home.py", "--check"],
                              cwd=work, capture_output=True, text=True)
        if cuda.returncode == 0:
            os.environ["CUDA_HOME"] = cuda.stdout.strip()
            os.environ["PATH"] = f"{cuda.stdout.strip()}/bin:{os.environ['PATH']}"
        _say(cuda.stderr.strip()[-800:])
    _say(f"Volume {volume.name!r} -> {root} | repo {work} | weights {hf}")
    if (VENV / "bin" / "python").exists():
        versions = subprocess.run(
            [os.environ["PYTHON"], "-c",
             "import torch, transformers, sglang, peft; print(f'torch {torch.__version__}, "
             "transformers {transformers.__version__}, sglang {sglang.__version__}, "
             "peft {peft.__version__}')"],
            capture_output=True, text=True)
        _say(f"packages: {versions.stdout.strip() or versions.stderr.strip()[-300:]}")
    else:
        _say(f"packages: not installed in {VENV} (pn.setup() installs them)")
    _say("GPU: " + ("; ".join(gpus) if gpus else
                    "none -- fine for pn.download() and pn.test(); pick B200 x1 in the "
                    "compute profile before a stage"))
    plan = subprocess.run([sys.executable, "scripts/gpu_plan.py", "show"], cwd=work,
                          capture_output=True, text=True)
    if gpus:
        _say(plan.stdout.strip() or plan.stderr.strip())
    return dict(_state)


# --------------------------------------------------------------------------
# running things

# A sweep reports "  250/1075 episodes"; tqdm (weights, training) redraws its
# bar by ending each state with a carriage return. Both are drawn in place in
# the cell. Read as text, a pipe turns every redraw into a line of its own:
# the stage-2 log held 937 lines, most of them one progress bar.
_EPISODES = re.compile(r"^\s*(\d+)/(\d+) episodes\s*$")
_TQDM = re.compile(r"\d+%\|")


def _lines(stream, partial: bool = False):
    """(text, redrawn) for each line of a byte stream; redrawn when the line
    ended in a bare carriage return, as a progress bar's redraws do.

    A bar writes each state AFTER a carriage return, so a state is complete
    only when the next one begins -- and on a slow bar (weights loading shard
    by shard) the line shown is always one state behind. With partial=True the
    state being drawn is also yielded as it arrives, as (text, None): shown,
    never logged."""
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf, drawing = "", False
    while True:
        chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
        buf += dec.decode(chunk or b"", final=not chunk)
        while True:
            cut = [i for i in (buf.find("\r"), buf.find("\n")) if i >= 0]
            if not cut:
                break
            i = min(cut)
            if buf[i] == "\r" and i + 1 == len(buf) and chunk:
                break                        # a \r\n may be split across reads
            if buf[i] == "\r" and buf[i + 1:i + 2] == "\n":
                yield buf[:i], False
                buf, drawing = buf[i + 2:], False
            else:
                yield buf[:i], buf[i] == "\r"
                buf, drawing = buf[i + 1:], buf[i] == "\r"
        if partial and drawing and chunk and buf.rstrip("\r"):
            yield buf.rstrip("\r"), None
        if not chunk:
            break
    if buf:
        yield buf, False


class _Progress:
    """Sweep progress as one line redrawn in place, with the time left."""

    def __init__(self) -> None:
        self.t0 = self.first = self.total = None

    def __call__(self, n: int, total: int) -> str:
        if total != self.total or self.first is None or n < self.first:
            self.t0, self.first, self.total = time.time(), n, total
        width, done = 30, n / total if total else 0
        rate = (n - self.first) / max(time.time() - self.t0, 1e-9)
        left = f", ~{(total - n) / rate / 60:.0f} min left" if rate > 0 and n < total else ""
        return (f"  {n}/{total} episodes [{'#' * int(done * width):<{width}}] "
                f"{done:.0%}{left}")


# How long each signal gets before the next: the stage's exit handler stops
# its servers on SIGINT, which can take minutes.
_SHUTDOWN = ((signal.SIGINT, 180), (signal.SIGTERM, 30), (signal.SIGKILL, 30))


def _shutdown(proc: subprocess.Popen) -> None:
    """Stop an interrupted stage and the servers it started. Runs to the end
    even if the stop button is pressed again: a second press moves on to a
    harder signal, it never skips the cleanup. (After stage 1, a second press
    cut the old handler short before it had checked that the servers were
    gone.)

    Presses are counted by a handler of our own for as long as this runs, not
    caught as KeyboardInterrupt: a press lands between any two bytecodes, and
    one that landed in a print, in os.killpg or between two try blocks escaped
    the old version the same way."""
    presses = [0]

    def count(signum, frame):
        presses[0] += 1
    try:
        previous, installed = signal.signal(signal.SIGINT, count), True
    except ValueError:                   # not the main thread: nothing to count
        previous, installed = None, False
    try:
        _say("\nstopping servers: the stage shuts them down on its way out (up to 3 min). "
             "Pressing stop again moves to a harder signal; it does not skip this.")
        for sig, wait in _SHUTDOWN:
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            seen, end = presses[0], time.monotonic() + wait
            while proc.poll() is None and presses[0] == seen and time.monotonic() < end:
                time.sleep(0.2)
            if proc.poll() is not None:
                break
            if presses[0] != seen:
                _say(f"  ...{sig.name} sent; trying the next signal")
        stop()
    finally:
        if installed:
            # None: the old handler was not Python's to give back; the stop
            # button must keep raising KeyboardInterrupt either way.
            signal.signal(signal.SIGINT, previous if previous is not None
                          else signal.default_int_handler)


def _stream(cmd: list[str], log_name: str, env: dict | None = None) -> None:
    """Run a command in the repo, its output in this cell and in logs/<log_name>
    on the Volume, progress bars drawn in place. Stopping the cell stops the
    command cleanly: the pipeline's own exit handler stops the servers it
    started, and _shutdown checks they are gone."""
    p = paths()
    log = p["work"] / "logs" / log_name
    proc = subprocess.Popen(cmd, cwd=p["work"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0, start_new_session=True,
                            env={**os.environ, **(env or {})})
    progress = _Progress()
    bar = None          # the last redraw of a tqdm bar, logged once the bar is done
    width = 0           # length of the line being redrawn in place; 0: none
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
        try:
            for text, redrawn in _lines(proc.stdout, partial=True):
                if redrawn is None:             # a bar's state, still arriving
                    if _TQDM.search(text):
                        print("\r" + text[:160].ljust(width), end="", flush=True)
                        width = len(text[:160])
                    continue
                if redrawn and not text.strip():
                    continue                    # the \r before a bar's first state
                m = _EPISODES.match(text)
                if m or redrawn:
                    shown = progress(int(m[1]), int(m[2])) if m else text[:160]
                    print("\r" + shown.ljust(width), end="", flush=True)
                    width = len(shown)
                    if m:
                        fh.write(text + "\n")
                    else:
                        bar = text
                elif _TQDM.search(text):
                    # A bar's last state, ended by a newline: the bar is done.
                    print("\r" + text[:160].ljust(width), flush=True)
                    width, bar = 0, None
                    fh.write(text + "\n")
                else:
                    if bar is not None:
                        fh.write(bar + "\n")
                        bar = None
                    print(("\n" if width else "") + text, flush=True)
                    width = 0
                    fh.write(text + "\n")
                fh.flush()
            if bar is not None:
                fh.write(bar + "\n")
            if width:
                print(flush=True)
            code = proc.wait()
        except BaseException:
            # The stop button, or anything else that ends this cell: never leave
            # a stage running unattended after its lock is released.
            _shutdown(proc)
            raise
    if code != 0:
        raise SystemExit(f"`{' '.join(cmd)}` exited with {code}. The lines above say why; "
                         f"all of it is in logs/{log_name}.")


def _claim(stage: str, force: bool = False) -> threading.Event:
    """The same stage lock as modal_pipeline._claim: a heartbeat file on the
    Volume, taken before a stage starts and refreshed every minute."""
    lock = paths()["work"] / LOCK
    me = f"notebook:{socket.gethostname()}:{os.getpid()}"
    if lock.exists() and not force:
        try:
            held = json.loads(lock.read_text())
        except ValueError:
            held = {}
        age = time.time() - held.get("beat", 0)
        if held.get("call") != me and age < STALE:
            raise SystemExit(
                f"stage {held.get('stage')!r} is already running on this Volume "
                f"({held.get('call')}, last heartbeat {age / 60:.0f} min ago). Two at once "
                f"would write the same traces and bill twice. pn.status() shows it; if you "
                f"are sure it is dead (a closed notebook, a stopped job), "
                f"pn.run({stage!r}, force=True).")
    done = threading.Event()
    tmp = lock.with_name(lock.name + f".{socket.gethostname()}")

    def write() -> None:
        tmp.write_text(json.dumps({"call": me, "stage": stage, "beat": time.time()}))
        os.replace(tmp, lock)

    def beat() -> None:
        while not done.wait(60):
            write()
    write()
    threading.Thread(target=beat, daemon=True).start()
    return done


def _release(done: threading.Event) -> None:
    done.set()
    lock = paths()["work"] / LOCK
    try:
        if json.loads(lock.read_text()).get("call", "").startswith(
                f"notebook:{socket.gethostname()}:"):
            lock.unlink()
    except (OSError, ValueError):
        pass


def run(stage: str, force: bool = False) -> None:
    """One stage of gpu_pipeline.sh on this notebook's GPU. Safe to run again
    after anything goes wrong: it resumes where the last run stopped."""
    if stage not in STAGES:
        raise SystemExit(f"stage is one of {', '.join(STAGES)}")
    if not _gpus():
        raise SystemExit("this notebook has no GPU: open the compute profile (sidebar), pick "
                         "B200 x1 (or H200 x1), let the kernel restart, run pn.setup() again")
    done = _claim(stage, force)
    try:
        _stream(["bash", "scripts/gpu_pipeline.sh", stage], f"notebook-{stage}.log")
    finally:
        _release(done)
    _say(f"\n{stage} finished. Next: the guide's reading step for it; pn.status() any time.")


def launch(stage: str, wait_min: float = 30, dry_run: bool = False):
    """Start a stage as a detached Modal job -- `modal run --detach
    scripts/modal_pipeline.py --stage <stage>`, the command part A of the guide
    runs from a laptop -- from this notebook, which can then stop: the job runs
    on its own GPU (PASAR_GPU, one B200 by default) until the stage ends, costs
    less per hour than a GPU notebook, and restarts by itself if preempted.
    Waits until the job's stage has begun (the first time, Modal builds the
    image: ~10 min), then returns. Secrets attached to this notebook
    (DEEPSEEK_API_KEY, HF_TOKEN) are passed on to the job."""
    if stage not in STAGES + ("download", "test"):
        raise SystemExit(f"stage is one of {', '.join(STAGES + ('download', 'test'))}")
    work = paths()["work"]
    cmd = [sys.executable, "-m", "modal", "run", "--detach", "scripts/modal_pipeline.py",
           "--stage", stage]
    if dry_run:
        return cmd
    if subprocess.run([sys.executable, "-c", "import modal"], capture_output=True).returncode:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], check=True)
    log = work / "logs" / f"launch-{stage}.log"
    with log.open("a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, cwd=work, stdout=log.open("a"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    _say(f"launching {stage} as a Modal job (log: logs/{log.name}) ...")
    seen, t0 = 0, time.time()
    while time.time() - t0 < wait_min * 60:
        time.sleep(15)
        text = log.read_text(errors="replace")
        new = text[seen:]
        seen = len(text)
        for line in new.splitlines()[-20:]:
            _say("  " + line[:200])
        if "gpu_pipeline.sh" in new or "download_weights.sh" in new or "run_tests.sh" in new:
            _say(f"\n{stage} is running as a job: https://modal.com/apps (app 'pasarbench'). "
                 f"This notebook can stop now. Follow the job's log there; pn.status() in a "
                 f"later session shows its files (a running notebook sees the Volume as it "
                 f"was when the notebook started).")
            return proc
        if proc.poll() is not None:
            raise SystemExit(
                f"the launch ended (exit {proc.returncode}); its log is above and in "
                f"logs/{log.name}. If it is about credentials or tokens, start the same stage "
                f"from your laptop instead (guide, part A): `modal run --detach "
                f"scripts/modal_pipeline.py --stage {stage}`.")
    _say(f"still starting after {wait_min:.0f} min (a first image build can take this long); "
         f"it carries on while this notebook runs. Check https://modal.com/apps.")
    return proc


def read(rel: str, head: int | None = None, tail: int | None = None) -> None:
    """Print a file from the repo on the Volume, e.g. pn.read("logs/check-A.txt")."""
    lines = (paths()["work"] / rel).read_text(errors="replace").splitlines()
    if head is not None:
        lines = lines[:head]
    if tail is not None:
        lines = lines[-tail:]
    _say("\n".join(lines))


def download() -> None:
    """The agent, the customer and its tokenizer onto the Volume (~90 GB).
    Needs no GPU; run again to resume an interrupted download."""
    _stream(["bash", "scripts/download_weights.sh"], "notebook-download.log")


def test() -> None:
    """The repo's tests, on the CPU (with a GPU visible, transformers would send
    the toy model's linear-attention layers to GPU kernels)."""
    _stream(["bash", "run_tests.sh"], "notebook-test.log", env={"CUDA_VISIBLE_DEVICES": ""})


def stop() -> None:
    """Stop SGLang servers left behind by an interrupted stage."""
    pids = paths()["work"] / "logs" / "servers.pid"
    if not pids.exists():
        _say("no servers recorded")
        return
    for pid in pids.read_text().split():
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(int(pid), sig)
            except (ProcessLookupError, PermissionError, ValueError):
                break
            time.sleep(10)
    pids.unlink(missing_ok=True)
    _say("servers stopped")


# --------------------------------------------------------------------------
# looking at it

def _episodes(run_dir: Path) -> tuple[int, int]:
    """(episode files, finished episodes) in a trace run."""
    files = list(run_dir.glob("*/*.jsonl"))
    done = 0
    for f in files:
        try:
            with f.open("rb") as fh:
                fh.seek(max(f.stat().st_size - 4096, 0))
                tail = fh.read().decode("utf-8", "replace").strip().splitlines()
            done += bool(tail) and '"type": "footer"' in tail[-1].replace('"type":"', '"type": "')
        except OSError:
            pass
    return len(files), done


def status() -> None:
    """The stage lock, the GPU, the settings, the traces so far, the newest log."""
    work = paths()["work"]
    lock = work / LOCK
    if lock.exists():
        try:
            held = json.loads(lock.read_text())
            age = (time.time() - held.get("beat", 0)) / 60
            state = "running" if age < STALE / 60 else "stale (dead?)"
            _say(f"stage {held.get('stage')}: {state}, {held.get('call')}, "
                 f"heartbeat {age:.0f} min ago")
        except ValueError:
            _say("stage lock unreadable")
    else:
        _say("no stage running")
    for g in _gpus():
        _say(f"GPU {g}")
    settings = work / "data" / "run_settings.env"
    if settings.exists():
        _say("settings: " + " ".join(settings.read_text().split()))
    k = dict(l.split("=", 1) for l in settings.read_text().split()) if settings.exists() else {}
    expect = {"P-smoke": 16, "P-base": 215 * int(k.get("PASAR_K", 5)),
              "P-collect": 215 * int(k.get("PASAR_COLLECT_K", 8)),
              "P-rft": 215 * int(k.get("PASAR_K", 5)), "P-ref": 215 * int(k.get("PASAR_K", 5))}
    for name, n in expect.items():
        d = work / "traces" / name
        if d.exists():
            files, done = _episodes(d)
            _say(f"traces/{name}: {done} finished of ~{n}" +
                 (f" ({files - done} unfinished)" if files > done else ""))
    for f in ("A", "B"):
        c = work / "checkpoints" / f"rft-{f}"
        if (c / "adapter_config.json").exists():
            _say(f"checkpoints/rft-{f}: trained")
        elif c.exists():
            steps = sorted(c.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            _say(f"checkpoints/rft-{f}: training" + (f", last save {steps[-1].name}"
                                                    if steps else ""))
    logs = sorted((work / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        tail = logs[-1].read_text(errors="replace").splitlines()[-8:]
        _say(f"\nnewest log, logs/{logs[-1].name} "
             f"({(time.time() - logs[-1].stat().st_mtime) / 60:.0f} min ago):")
        for line in tail:
            _say("  " + line[:200])


def pack() -> Path:
    """logs/, the splits, the settings and the smoke, baseline, RFT and
    reference traces as one .tar.gz on the Volume, to download (the collection
    run stays on the Volume: it is several GB)."""
    p = paths()
    work = p["work"]
    out = work / f"results-{time.strftime('%Y%m%d-%H%M')}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for rel in RESULTS:
            if (work / rel).exists():
                tar.add(work / rel, arcname=f"{REPO}/{rel}",
                        filter=lambda ti: None if ti.name.endswith(".stage-lock.json") else ti)
    size = out.stat().st_size
    _say(f"{out} ({size / 2**20:.1f} MB)")
    if size <= WEB_DOWNLOAD_LIMIT:
        _say(f"Download: Files panel -> {p['name']} -> {REPO} -> {out.name}. On your laptop:\n"
             f"  cd ~/Downloads && tar -xzf {out.name}     # lands in ~/Downloads/{REPO}/")
    else:
        _say(f"Over the web UI's 16 MB download limit. On your laptop (pip install modal; "
             f"modal setup):\n  cd ~/Downloads && modal volume get {p['name']} "
             f"{REPO}/{out.name} . && tar -xzf {out.name}")
    return out
