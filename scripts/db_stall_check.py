"""
What does the service do while its database stops answering, and after? (v83)

One server on PostgreSQL, clients asking it about their orders (the mock
agent, after a model's latency), and the database stopped in the middle of
it: every process of the PostgreSQL server sent SIGSTOP -- its sockets stay
open and its kernel still acknowledges what it is sent, so nothing tells a
client it has gone, which is how a database that hangs (a stalled disk, a
frozen VM, a failover not made yet) looks from the other end -- and SIGCONT
--stall-for seconds later. Measured:

    in flight   the requests the stall caught: what they were answered, and
                how long the longest took
    started     the requests started while the database was stopped: the same
    arrivals    new customers, --arrivals a second whatever the others are
                answered, as a storefront's traffic keeps coming: each opens a
                conversation and asks one thing -- what those that came during
                the stall were answered, how long they waited, how many were
                still waiting when the database was back, and the server's
                threads at most
    readyz      a probe every 0.5 s, as a balancer's: how soon after the stall
                it said 503, the longest it took to answer, and how soon after
                the database was back it said 200
    back        how soon after the database was back each client's turns were
                answered 200 again -- on the same conversation, tried again
                0.5 s after a 503 or a 409 -- and what the rest was answered
    log         the server's lines at ERROR, and its tracebacks

    python scripts/db_stall_check.py \\
        --database-url postgresql+psycopg://postgres@127.0.0.1:5433/load \\
        --pg-data /var/lib/postgresql/16/main [--stall-at 8] [--stall-for 30] \\
        [--run 55] [--clients 16] [--arrivals 4] [--root DIR] [--runs 1] \\
        [--restart STOP START] [--json out.json]

It stops a PostgreSQL server -- every database on it -- so it runs only on a
test server on this machine: the URL's host must be this machine, and the
data directory's postmaster.pid must name a running process that listens on
the URL's port. However the check ends, the server is sent SIGCONT. It
migrates the database and makes a tenant of its own. --root runs another
source tree's server (an earlier version, to compare); the clients are this
tree's. --restart STOP START restarts the server instead of stopping it: two
shell commands, run as given (pg_ctl's stop -m fast and start, say), the
database down --stall-for seconds and every session ended, as a restart or a
failover is to its clients.
"""

import argparse
import json
import os
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SECRET = ("_API_KEY", "_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
LOCAL = {"127.0.0.1", "localhost", "::1", ""}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Postgres:
    """The local PostgreSQL server the check may stop: its postmaster (from
    the data directory's postmaster.pid) and the processes it started."""

    def __init__(self, data: str, url: str):
        parts = urlsplit(url.replace("postgresql+psycopg", "postgresql", 1))
        host = parts.hostname or ""
        if host not in LOCAL and not host.startswith("/"):
            raise SystemExit(f"--database-url: {host!r} is not this machine; the check stops "
                             f"the database's server, so it runs only on a local one")
        pid_file = Path(data, "postmaster.pid")
        try:
            lines = pid_file.read_text().splitlines()
        except OSError as e:
            raise SystemExit(f"--pg-data: cannot read {pid_file} ({e.strerror})") from None
        self.master = int(lines[0])
        port = int(lines[3]) if len(lines) > 3 and lines[3].strip() else 5432
        if not Path(f"/proc/{self.master}").exists():
            raise SystemExit(f"--pg-data: postmaster {self.master} is not running")
        if port != (parts.port or 5432):
            raise SystemExit(f"--pg-data: that server listens on {port}, the URL names "
                             f"{parts.port or 5432}")
        self.stopped: list[int] = []
        self.commands: tuple[str, str] | None = None
        self.down = False

    def children(self) -> list[int]:
        out = []
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                stat = Path(f"/proc/{p}/stat").read_text()
            except OSError:
                continue
            if stat.rsplit(")", 1)[1].split()[1] == str(self.master):
                out.append(int(p))
        return out

    def stop(self) -> int:
        """The postmaster first, so that it starts no more; then what it
        started, twice over, for one it started meanwhile. (Or the server
        stopped, with --restart.)"""
        if self.commands:
            self.down = True
            subprocess.run(self.commands[0], shell=True, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return 1
        os.kill(self.master, signal.SIGSTOP)
        self.stopped = [self.master]
        for _ in range(2):
            for p in self.children():
                if p not in self.stopped:
                    try:
                        os.kill(p, signal.SIGSTOP)
                        self.stopped.append(p)
                    except ProcessLookupError:
                        pass
        return len(self.stopped)

    def resume(self) -> None:
        if self.commands:
            if self.down:
                self.down = False
                subprocess.run(self.commands[1], shell=True, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        for p in reversed(self.stopped):
            try:
                os.kill(p, signal.SIGCONT)
            except ProcessLookupError:
                pass
        self.stopped = []


def clean_env() -> dict[str, str]:
    """This process's environment without anything that holds a key."""
    return {k: v for k, v in os.environ.items() if not k.upper().endswith(SECRET)}


def customers(n: int) -> list[tuple[str, list[str]]]:
    """Customers of a tenant made with world synthetic:4 -- the benchmark's
    own, four times over, each copy's ids suffixed -- who have orders."""
    from pasarbench.db import Database
    seed = Database.fresh().tables
    people = [(u, [o for o, x in seed["orders"].items() if x["user_id"] == u])
              for u in seed["users"]]
    people = [p for p in people if p[1]]
    who = [(f"{u}-C{i:04d}", [f"{o}-C{i:04d}" for o in orders])
           for i in range(1, 5) for u, orders in people]
    return [who[k % len(who)] for k in range(n)]


def problem(r) -> str:
    try:
        kind = r.json().get("type", "")
    except ValueError:
        kind = ""
    return f"{r.status_code} {kind.rsplit('/', 1)[-1]}".strip()


def one_run(args, pg: Postgres, tag: str) -> dict:
    import httpx
    root = Path(args.root).resolve() if args.root else ROOT
    token = secrets.token_urlsafe(16)
    env = clean_env()
    env.update(PASAR_DATABASE_URL=args.database_url, PASAR_ADMIN_TOKEN=token,
               PASAR_MOCK_LATENCY_MS=str(args.latency_ms), PASAR_RATE="100000",
               PASAR_BURST="100000", PASAR_LOG_LEVEL="WARNING", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(root))
    subprocess.run([sys.executable, "-B", "-m", "pasarbench.service", "migrate"], env=env,
                   cwd=root, check=True, stdout=subprocess.DEVNULL)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.mkdtemp(prefix="db-stall-")) / "server.log"
    log = open(log_path, "w")
    srv = subprocess.Popen([sys.executable, "-B", "-m", "pasarbench.service", "serve",
                            "--port", str(port)], env=env, cwd=root,
                           stdout=subprocess.DEVNULL, stderr=log)
    try:
        for _ in range(200):
            try:
                if httpx.get(base + "/readyz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise SystemExit("the server did not start")
        made = httpx.post(base + "/v1/admin/tenants", timeout=300,
                          headers={"Authorization": f"Bearer {token}"},
                          json={"tenant_id": f"stall-{secrets.token_hex(3)}", "name": "stall",
                                "world": "synthetic:4"})
        made.raise_for_status()
        h = {"Authorization": f"Bearer {made.json()['api_key']}"}
        who = customers(args.clients)
        calls: list[tuple[int, str, float, float, str]] = []   # client, what, start, took, status
        probes: list[tuple[float, float, str]] = []
        arrivals: list[tuple[float, float, str]] = []           # start, took, status
        threads_seen: list[tuple[float, int]] = []
        lock = threading.Lock()
        t0 = time.monotonic()
        stop_at = t0 + args.run

        def client(k: int) -> None:
            user, orders = who[k]
            with httpx.Client(base_url=base, headers=h, timeout=300) as c:
                cid, n = None, 0
                while time.monotonic() < stop_at:
                    start = time.monotonic()
                    what = "open" if cid is None else "turn"
                    try:
                        if cid is None:
                            r = c.post("/v1/conversations", json={"user_id": user})
                            if r.status_code == 201:
                                cid = r.json()["conversation_id"]
                        else:
                            r = c.post(f"/v1/conversations/{cid}/messages", json={
                                "content": f"where is my order {orders[n % len(orders)]}?"})
                            if r.status_code == 200:
                                n += 1
                                if n % 10 == 0:
                                    cid = None          # a new conversation now and then
                            elif r.status_code not in (409, 429, 503):
                                cid = None
                        status = str(r.status_code) if r.status_code < 300 else problem(r)
                    except httpx.HTTPError as e:
                        status = f"transport {type(e).__name__}"
                        cid = None
                    with lock:
                        calls.append((k, what, start - t0, time.monotonic() - start, status))
                    if not status.startswith("2"):
                        time.sleep(0.5)

        def prober() -> None:
            with httpx.Client(base_url=base, timeout=300) as c:
                while time.monotonic() < stop_at:
                    start = time.monotonic()
                    try:
                        st = str(c.get("/readyz").status_code)
                    except httpx.HTTPError as e:
                        st = f"transport {type(e).__name__}"
                    took = time.monotonic() - start
                    with lock:
                        probes.append((start - t0, took, st))
                    time.sleep(max(0.0, 0.5 - took))

        def arrival(i: int) -> None:
            user, orders = who[i % len(who)]
            start = time.monotonic()
            try:
                with httpx.Client(base_url=base, headers=h, timeout=300) as c:
                    r = c.post("/v1/conversations", json={"user_id": user})
                    if r.status_code == 201:
                        r = c.post(f"/v1/conversations/{r.json()['conversation_id']}/messages",
                                   json={"content": f"where is my order {orders[0]}?"})
                status = str(r.status_code) if r.status_code < 300 else problem(r)
            except httpx.HTTPError as e:
                status = f"transport {type(e).__name__}"
            with lock:
                arrivals.append((start - t0, time.monotonic() - start, status))

        def arriver() -> None:
            if args.arrivals <= 0:
                return
            i, every = 0, 1 / args.arrivals
            while time.monotonic() < stop_at:
                threading.Thread(target=arrival, args=(i,), daemon=True).start()
                i += 1
                time.sleep(max(0.0, t0 + i * every - time.monotonic()))

        def counter() -> None:
            while time.monotonic() < stop_at + 5:
                try:
                    status = Path(f"/proc/{srv.pid}/status").read_text()
                except OSError:
                    return
                n = int(next(ln.split()[1] for ln in status.splitlines()
                             if ln.startswith("Threads:")))
                with lock:
                    threads_seen.append((time.monotonic() - t0, n))
                time.sleep(0.5)

        threads = [threading.Thread(target=client, args=(k,), daemon=True)
                   for k in range(args.clients)]
        threads += [threading.Thread(target=f, daemon=True) for f in (prober, arriver, counter)]
        for t in threads:
            t.start()
        time.sleep(args.stall_at)
        try:
            n_stopped = pg.stop()
            t_stop = time.monotonic() - t0
            time.sleep(args.stall_for)
        finally:
            pg.resume()
        t_back = time.monotonic() - t0
        for t in threads:
            t.join(args.run + args.stall_for + 330)
        give_up = time.monotonic() + 330
        while time.monotonic() < give_up:       # the last arrivals' answers
            with lock:
                if len(arrivals) >= int(args.run * args.arrivals) - 2:
                    break
            time.sleep(0.2)
    finally:
        pg.resume()
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
        log.close()
    lines = log_path.read_text().splitlines()
    return summary(tag, calls, probes, arrivals, threads_seen, t_stop, t_back, n_stopped,
                   args, lines)


def summary(tag, calls, probes, arrivals, threads_seen, t_stop, t_back, n_stopped, args,
            lines) -> dict:
    in_flight = [x for x in calls if x[2] < t_stop < x[2] + x[3]]
    started = [x for x in calls if t_stop <= x[2] < t_back]
    after = [x for x in calls if x[2] >= t_back]
    before = [x for x in calls if x[2] + x[3] <= t_stop]

    def med(xs):
        return round(statistics.median(xs), 3) if xs else None

    def first(rows, cond, since):
        hits = [x[0] + x[1] for x in rows if cond(x)]
        return round(min(hits) - since, 2) if hits else None

    back = []                               # each client's first turn answered 200 after
    for k in range(args.clients):
        done = [x[2] + x[3] for x in calls if x[0] == k and x[4] == "200" and x[1] == "turn"
                and x[2] + x[3] > t_back]
        if done:
            back.append(min(done) - t_back)
    errors = [ln for ln in lines if '"level": "ERROR"' in ln or '"level": "CRITICAL"' in ln]
    out = {
        "run": tag, "stall_at_s": round(t_stop, 1), "stall_for_s": round(t_back - t_stop, 1),
        "processes_stopped": n_stopped, "clients": args.clients,
        "before": {"requests": len(before),
                   "answered": dict(Counter(x[4] for x in before))},
        "in_flight": {"requests": len(in_flight),
                      "answered": dict(Counter(x[4] for x in in_flight)),
                      "longest_s": round(max([x[3] for x in in_flight] or [0]), 2),
                      "longest_after_stall_s": round(max(
                          [x[2] + x[3] - t_stop for x in in_flight] or [0]), 2)},
        "started": {"requests": len(started),
                    "answered": dict(Counter(x[4] for x in started)),
                    "median_s": med([x[3] for x in started]),
                    "longest_s": round(max([x[3] for x in started] or [0]), 2)},
        "arrivals": {
            "during": len([x for x in arrivals if t_stop <= x[0] < t_back]),
            "answered": dict(Counter(x[2] for x in arrivals if t_stop <= x[0] < t_back)),
            "median_s": med([x[1] for x in arrivals if t_stop <= x[0] < t_back]),
            "longest_s": round(max([x[1] for x in arrivals if t_stop <= x[0] < t_back]
                                   or [0]), 2),
            "waiting_when_back": len([x for x in arrivals
                                      if x[0] < t_back < x[0] + x[1]]),
            "server_threads_before": max([n for t, n in threads_seen if t < t_stop] or [0]),
            "server_threads_during": max([n for t, n in threads_seen
                                          if t_stop <= t < t_back] or [0])},
        "readyz": {
            "first_not_200_after_stall_s": first(
                probes, lambda x: x[0] + x[1] > t_stop and x[2] != "200", t_stop),
            "longest_s": round(max([x[1] for x in probes] or [0]), 2),
            "answered_during": dict(Counter(x[2] for x in probes
                                            if t_stop <= x[0] < t_back)),
            "first_200_after_back_s": first(
                probes, lambda x: x[0] >= t_back and x[2] == "200", t_back)},
        "back": {"clients_answered_again": len(back),
                 "median_s": round(statistics.median(back), 2) if back else None,
                 "slowest_s": round(max(back), 2) if back else None,
                 "answered": dict(Counter(x[4] for x in after))},
        "log": {"error_lines": len(errors),
                "tracebacks": sum('"traceback"' in ln or "Traceback" in ln for ln in lines),
                "lines": len(lines),
                "first_error": errors[0][:300] if errors else None},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url", required=True,
                    help="a TCP URL of a database on this machine the check may fill")
    ap.add_argument("--pg-data", required=True,
                    help="that server's data directory (its postmaster.pid)")
    ap.add_argument("--stall-at", type=float, default=8.0)
    ap.add_argument("--stall-for", type=float, default=30.0)
    ap.add_argument("--run", type=float, default=55.0, help="seconds the clients ask for")
    ap.add_argument("--clients", type=int, default=16)
    ap.add_argument("--arrivals", type=float, default=4.0,
                    help="new customers a second, whatever the others are answered (0: none)")
    ap.add_argument("--latency-ms", type=float, default=200)
    ap.add_argument("--root", help="the source tree whose server to run (default: this one)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--restart", nargs=2, metavar=("STOP", "START"),
                    help="restart the server instead: two shell commands, run as given")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.stall_at + args.stall_for >= args.run:
        ap.error("--run must outlast --stall-at plus --stall-for")
    pg = Postgres(args.pg_data, args.database_url)
    pg.commands = tuple(args.restart) if args.restart else None

    def resume_and_exit(signum, _frame):
        pg.resume()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, resume_and_exit)
    signal.signal(signal.SIGINT, resume_and_exit)
    results = []
    for i in range(args.runs):
        r = one_run(args, pg, f"{Path(args.root).name if args.root else 'this tree'} #{i + 1}")
        results.append(r)
        print(json.dumps(r, indent=1), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
