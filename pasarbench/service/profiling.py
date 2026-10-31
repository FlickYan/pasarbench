"""
Where a server's CPU goes: a sampling profiler over every thread of the
process, from the standard library alone, that an operator runs on a live
server.

    GET    /v1/admin/profile?seconds=30     this process, sampled for 30 seconds (or
                                            until DELETE): the profile, as JSON
    DELETE /v1/admin/profile                ends a running profile now
    python -m pasarbench.service profile p.json [--turns N] [--folded p.folded]
                                            what it adds up to, by layer and by
                                            function; --folded writes its stacks for a
                                            flame graph (speedscope.app, flamegraph.pl)
    python scripts/load_test.py ... --profile p.json      a load test, profiled

Every 10 ms the sampler's thread looks at every other thread: its stack
(sys._current_frames) and the CPU time it has used, from that thread's own
CPU clock. A thread that used none since the last look -- waiting on the
database, a model, a lock, for work -- is charged nothing; one that did is
charged what it used. The profile is CPU, not wall time, however many
threads wait.

What is a thread's CPU charged to? To the deepest call that ran throughout:
the frames its stack held both at the last look and at this one (the same
frames, not frames of the same functions). Python runs one thread at a time
and the sampler needs that turn too, so it finds a thread either where the
interpreter paused it or where the thread let go to wait (a query sent). Were
the CPU charged to the stack the thread is found in, what it spent before it
began to wait would go to the wait -- the driver's, the event loop's select --
and be missing from the work before it. A call that ran throughout did spend
it: what is charged to a function was spent in it or in what it called, never
elsewhere. How finely it resolves depends on how far a thread gets between two
looks; the summary says how much stayed with the service's own frames
unresolved (tests/test_service.py checks the charging against threads whose
CPU is known).

Without per-thread CPU clocks (Linux's) a sample is weighted by wall time
instead, a waiting thread included, and the profile says so ("mode": "wall").

A profile holds code locations only -- modules, functions -- never a value
from a frame.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import Counter
from types import CodeType, FrameType
from typing import Any

MAX_SECONDS = 600
UNRESOLVED = "(no frame ran throughout)"


def _cpu_clock(native_id: int) -> int:
    """A thread's CPU clock, by its kernel id: Linux's encoding of one (what
    glibc's pthread_getcpuclockid computes), which needs no handle that could
    outlive its thread -- a thread gone reads as an error, never a crash."""
    return ((~native_id) << 3) | 6


def thread_clocks() -> bool:
    """Per-thread CPU clocks, read as this module reads them."""
    if not sys.platform.startswith("linux") or not hasattr(time, "pthread_getcpuclockid"):
        return False
    try:
        return time.pthread_getcpuclockid(threading.get_ident()) == _cpu_clock(
            threading.get_native_id())
    except OSError:
        return False


class Busy(Exception):
    """A profile is already running in this process."""


_RUNNING: list["Sampler"] = []
_RUNNING_LOCK = threading.Lock()


def running() -> "Sampler | None":
    with _RUNNING_LOCK:
        return _RUNNING[0] if _RUNNING else None


class _Seen:
    """A thread as the last look found it: its CPU clock, its frames
    (outermost first, kept alive so no other frame takes their ids) and their
    code objects' ids."""
    __slots__ = ("used", "frames", "codes")

    def __init__(self, used: int, frames: list[FrameType], codes: list[int]):
        self.used, self.frames, self.codes = used, frames, codes


class Sampler:
    """One profile of this process: start(), then stop() for the report."""

    def __init__(self, interval: float = 0.01):
        if not 0.001 <= interval <= 1:
            raise ValueError("the interval is 1 ms to 1 s")
        self.interval = interval
        self.cpu = thread_clocks()
        self.stacks: Counter[tuple[int, ...]] = Counter()      # code ids -> nanoseconds
        self.looks = 0
        self.threads: set[int] = set()
        self._codes: dict[int, CodeType] = {}                   # kept: their ids stay theirs
        self._seen: dict[int, _Seen] = {}
        self._native: dict[int, int] = {}                       # ident -> kernel thread id
        self._end = threading.Event()       # an operator ended it (DELETE)
        self._done = threading.Event()      # the sampler's thread to stop
        self._thread: threading.Thread | None = None
        self._own = 0.0

    def start(self) -> "Sampler":
        with _RUNNING_LOCK:
            if _RUNNING:
                raise Busy("a profile is already running in this process")
            _RUNNING.append(self)
        self._wall0, self._cpu0 = time.perf_counter(), time.process_time()
        self._thread = threading.Thread(target=self._loop, name="pasar-profiler", daemon=True)
        self._thread.start()
        return self

    def end(self) -> None:
        self._end.set()

    def wait(self, seconds: float) -> None:
        """Until `seconds` have passed or end() is called."""
        self._end.wait(seconds)

    def stop(self) -> dict[str, Any]:
        self._done.set()
        if self._thread is not None:
            self._thread.join()
        wall, cpu = time.perf_counter() - self._wall0, time.process_time() - self._cpu0
        with _RUNNING_LOCK:
            if self in _RUNNING:
                _RUNNING.remove(self)
        return self.report(wall, cpu)

    # --- the sampler's thread ----------------------------------------------
    def _loop(self) -> None:
        me = threading.get_ident()
        t0 = time.thread_time()
        try:
            while not self._done.wait(self.interval):
                self._look(me)
        finally:
            self._own = time.thread_time() - t0
            self._seen.clear()          # the frames it kept, let go

    def _look(self, me: int) -> None:
        self.looks += 1
        codes, seen, cpu, native = self._codes, self._seen, self.cpu, self._native
        alive = set()
        for ident, top in sys._current_frames().items():
            nid = native.get(ident)
            if nid is None:
                # A thread not seen before (or an ident reused): the map anew.
                native.clear()
                native.update((t.ident, t.native_id) for t in threading.enumerate())
                nid = native.get(ident)
            if ident == me or nid is None:
                continue
            alive.add(nid)
            last = seen.get(nid)
            if cpu:
                try:
                    used = time.clock_gettime_ns(_cpu_clock(nid))
                except OSError:
                    native.pop(ident, None)     # gone, or its ident another thread's now
                    continue
                if last is not None and used == last.used:
                    continue                    # it has not run
            else:
                used = time.perf_counter_ns()
            frames: list[FrameType] = []
            f: FrameType | None = top
            while f is not None:
                frames.append(f)
                f = f.f_back
            frames.reverse()
            k = 0
            if last is not None:
                # The frames both looks found, from the outermost in: the same
                # objects, so the same calls, which ran throughout.
                old = last.frames
                n = min(len(old), len(frames))
                while k < n and old[k] is frames[k]:
                    k += 1
                self.stacks[tuple(last.codes[:k])] += used - last.used
                ids = last.codes[:k]
            else:
                ids = []
            for x in frames[k:]:
                c = x.f_code
                ids.append(id(c))
                if id(c) not in codes:
                    codes[id(c)] = c
            seen[nid] = _Seen(used, frames, ids)
        for nid in [n for n in seen if n not in alive]:
            del seen[nid]
        self.threads |= alive

    # --- the report ------------------------------------------------------------
    def report(self, wall: float, process_cpu: float) -> dict[str, Any]:
        roots = _roots()
        names = {cid: frame_name(c, roots) for cid, c in self._codes.items()}
        stacks: Counter[str] = Counter()
        for ids, ns in self.stacks.items():
            stacks[";".join(names[i] for i in ids) or UNRESOLVED] += ns
        sampled = sum(stacks.values()) / 1e9
        return {
            "mode": "cpu" if self.cpu else "wall", "pid": os.getpid(),
            "python": sys.version.split()[0], "interval_ms": self.interval * 1000,
            "wall_seconds": round(wall, 3), "process_cpu_seconds": round(process_cpu, 3),
            "sampled_seconds": round(sampled, 3), "sampler_cpu_seconds": round(self._own, 3),
            "looks": self.looks, "threads": len(self.threads),
            "stacks_us": {k: round(v / 1000) for k, v in stacks.most_common() if v >= 500},
        }


def _roots() -> list[str]:
    out = {os.path.abspath(p or os.getcwd()) for p in sys.path}
    return sorted(out, key=len, reverse=True)


def frame_name(code: CodeType, roots: list[str]) -> str:
    """module:qualname -- the module from the file, by the longest sys.path
    entry it is under."""
    fn = code.co_filename
    mod = fn.strip("<>")
    if not fn.startswith("<"):
        for r in roots:
            if fn.startswith(r + os.sep):
                mod = fn[len(r) + 1:]
                break
        mod = mod[:-3] if mod.endswith(".py") else mod
        mod = mod[:-9] if mod.endswith(os.sep + "__init__") else mod
        mod = mod.replace(os.sep, ".")
    name = getattr(code, "co_qualname", code.co_name)
    return f"{mod}:{name}".replace(";", ",")


# ---------------------------------------------------------------------------
# What a profile adds up to
# ---------------------------------------------------------------------------

LAYERS = (
    ("pasarbench.service.", "the service"),
    ("pasarbench.", "the benchmark (tools, world, harness)"),
    ("sqlalchemy.", "SQLAlchemy"),
    ("psycopg", "psycopg"),
    ("pydantic", "Pydantic"),
    ("fastapi.", "FastAPI, Starlette"),
    ("starlette.", "FastAPI, Starlette"),
    ("uvicorn.", "uvicorn, h11, httptools"),
    ("h11.", "uvicorn, h11, httptools"),
    ("httptools", "uvicorn, h11, httptools"),
    ("anyio.", "asyncio, anyio"),
    ("asyncio.", "asyncio, anyio"),
    ("selectors", "asyncio, anyio"),
    ("prometheus_client.", "prometheus_client"),
    ("json.", "json"),
    ("copy", "copy (deepcopy)"),
    ("threading", "threading, queue"),
    ("queue", "threading, queue"),
    ("logging.", "logging"),
)


def layer(frame: str) -> str:
    mod = frame.split(":", 1)[0]
    if frame == UNRESOLVED:
        return UNRESOLVED
    for prefix, name in LAYERS:
        if mod.startswith(prefix) or mod == prefix.rstrip("."):
            return name
    return "other: " + mod.split(".", 1)[0]


def merge(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    stacks: Counter[str] = Counter()
    for p in profiles:
        stacks.update(p["stacks_us"])
    out = {k: sum(p[k] for p in profiles) for k in
           ("wall_seconds", "process_cpu_seconds", "sampled_seconds", "sampler_cpu_seconds",
            "looks", "threads")}
    out["mode"] = ",".join(sorted({p["mode"] for p in profiles}))
    out["processes"] = len(profiles)
    out["stacks_us"] = dict(stacks)
    return out


def summarize(profile: dict[str, Any], turns: int | None = None, top: int = 25) -> str:
    """The profile as tables: the service's and the benchmark's functions --
    what each spent, and in what; every function by what it held, its calls
    included; and the layers the CPU was charged to."""
    stacks = profile["stacks_us"]
    total = sum(stacks.values()) or 1
    per = (lambda us: f"{us / 1000 / turns:8.2f}") if turns else (lambda us: f"{us / 1e6:8.2f}")
    unit = "ms/turn" if turns else "seconds"
    by_layer: Counter[str] = Counter()
    ours: Counter[str] = Counter()
    ours_in: dict[str, Counter[str]] = {}
    inclusive: Counter[str] = Counter()
    for key, us in stacks.items():
        frames = key.split(";")
        by_layer[layer(frames[-1])] += us
        mine = next((f for f in reversed(frames) if f.startswith("pasarbench.")), None)
        if mine is not None:
            ours[mine] += us
            ours_in.setdefault(mine, Counter())[layer(frames[-1])] += us
        for f in set(frames):
            inclusive[f] += us
    own = sum(us for k, us in stacks.items() if k.rsplit(";", 1)[-1].startswith("pasarbench."))
    kind = "CPU" if profile["mode"] == "cpu" else "wall time (no per-thread CPU clocks here)"
    cpu, charged = profile["process_cpu_seconds"], profile["sampled_seconds"]
    lines = [
        f"profile: {kind}, {profile['wall_seconds']:.1f} s of wall time, "
        f"{profile.get('processes', 1)} process(es), {profile['threads']} threads, "
        f"{profile['looks']} looks",
        f"  process CPU {cpu:.2f} s; charged to stacks {charged:.2f} s "
        f"({charged / max(cpu, 1e-9):.0%}); the sampler's own "
        f"{profile['sampler_cpu_seconds']:.2f} s" + (f"; {turns} turns" if turns else ""),
        f"  {own / total:.0%} stayed with a frame of the service's or the benchmark's own: "
        f"no deeper call ran throughout",
        "",
        "the service and the benchmark, by function -- what each spent where no deeper "
        "frame of theirs ran throughout, its calls into libraries included",
        f"  {unit:>8}  share  function  (charged to)",
    ]
    for name, us in ours.most_common(top):
        where = ", ".join(f"{k} {v / us:.0%}" for k, v in ours_in[name].most_common(3))
        lines.append(f"  {per(us)}  {us / total:5.1%}  {name}  ({where})")
    lines += ["", "every function, by what it held -- everything it called included",
              f"  {unit:>8}  share  function"]
    for name, us in inclusive.most_common(top):
        lines.append(f"  {per(us)}  {us / total:5.1%}  {name}")
    lines += ["", "by layer -- the deepest frame each microsecond was charged to",
              f"  {unit:>8}  share  layer"]
    for name, us in by_layer.most_common():
        lines.append(f"  {per(us)}  {us / total:5.1%}  {name}")
    return "\n".join(lines)


def folded(profile: dict[str, Any]) -> str:
    """The stacks as a flame graph reads them: frames by ;, then microseconds."""
    return "".join(f"{k} {v}\n" for k, v in sorted(profile["stacks_us"].items()))
