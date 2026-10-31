"""
A running agent service, checked end to end over HTTP the way a storefront
uses it:

    python3 scripts/smoke_api.py http://127.0.0.1:8000 psk_...

The key is that of a tenant made for the check, on the demo world
(`create-tenant smoke --name Smoke --world demo`): the check cancels order
O1002, so a second run needs a new tenant. The server must offer the mock
model (it does unless PASAR_MODELS leaves it out).

It checks that the process is up and its database ready; that a request
without a key is refused as problem+json; that a conversation opens for U002;
that her request to cancel O1002 with her digits has the agent verify her, look
the order up and cancel it; that the same request sent again under its
Idempotency-Key gets the same bytes back and runs no second turn; that a
streamed turn runs from turn.started to turn.completed; that the audit log has
every tool call, the cancellation as a write; that the metrics count the
turns; and that a connection left idle six seconds still serves the next
request. Exit 0 when all of it holds, 1 at the first that does not.

Only the standard library, so it runs anywhere: CI runs it on the runner,
against the image (scripts/smoke_image.sh).
"""

from __future__ import annotations

import json
import re
import secrets
import sys
import urllib.error
import urllib.request
from email.message import Message

EXPECTED_TOOLS = ["verify_identity", "get_order", "cancel_order"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is answered as the error it is: the key goes to no other address."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        return None


OPENER = urllib.request.build_opener(_NoRedirect())


def call(base: str, method: str, path: str, key: str | None = None,
         body: object | None = None, headers: dict[str, str] | None = None,
         timeout: float = 120.0) -> tuple[int, Message, bytes]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with OPENER.open(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def sse(raw: bytes) -> list[tuple[str, str]]:
    """(event, data) for each event of a Server-Sent Events body."""
    out = []
    for block in raw.decode().replace("\r\n", "\n").split("\n\n"):
        event, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            out.append((event, "\n".join(data)))
    return out


class Failed(Exception):
    pass


def check(cond: object, what: str, detail: object = "") -> None:
    if not cond:
        raise Failed(f"{what}{': ' if detail != '' else ''}{detail}")
    print(f"ok    {what}")


def smoke(base: str, key: str) -> None:
    run = secrets.token_hex(4)       # fresh Idempotency-Keys on every run

    status, _, raw = call(base, "GET", "/healthz")
    check(status == 200 and json.loads(raw).get("status") == "ok", "GET /healthz", raw[:200])
    status, _, raw = call(base, "GET", "/readyz")
    check(status == 200 and json.loads(raw).get("status") == "ready",
          "GET /readyz: the database answers and holds the schema", raw[:200])

    status, headers, raw = call(base, "POST", "/v1/conversations", body={"user_id": "U002"})
    check(status == 401 and headers.get_content_type() == "application/problem+json",
          "no key: 401, as problem+json", f"{status} {headers.get_content_type()}")

    status, _, raw = call(base, "POST", "/v1/conversations", key,
                          {"user_id": "U002", "model": "mock"},
                          {"Idempotency-Key": f"smoke-open-{run}"})
    conv = json.loads(raw) if status == 201 else {}
    cid = conv.get("conversation_id", "")
    check(re.fullmatch(r"cv_[0-9a-f]{24}", cid) and conv.get("model") == "mock",
          "a conversation for U002, on the mock model", f"{status} {raw[:300]!r}")

    path = f"/v1/conversations/{cid}/messages"
    ask = {"content": "Please cancel O1002, last digits 4567"}
    idem = {"Idempotency-Key": f"smoke-msg-{run}"}
    status, headers, first = call(base, "POST", path, key, ask, idem)
    turn = json.loads(first) if status == 200 else {}
    tools = [t.get("tool") for t in turn.get("tool_calls", [])]
    check(status == 200 and turn.get("stop_reason") == "replied" and not turn.get("fallback"),
          "a turn: the agent replied", f"{status} {first[:300]!r}")
    check(tools == EXPECTED_TOOLS and all(t.get("ok") for t in turn["tool_calls"]),
          "... having verified her, looked O1002 up and cancelled it",
          [(t.get("tool"), t.get("ok"), t.get("error")) for t in turn.get("tool_calls", [])])

    status, headers, again = call(base, "POST", path, key, ask, idem)
    check(status == 200 and again == first and headers.get("Idempotent-Replayed") == "true",
          "the same request again, same Idempotency-Key: the same bytes, replayed",
          f"{status} replayed={headers.get('Idempotent-Replayed')}")
    status, _, raw = call(base, "GET", f"/v1/conversations/{cid}", key)
    conv = json.loads(raw) if status == 200 else {}
    check(conv.get("turns") == 1 and conv.get("tool_calls") == len(EXPECTED_TOOLS),
          "... and no second turn ran", {k: conv.get(k) for k in ("turns", "tool_calls")})

    status, headers, raw = call(base, "POST", path + "?stream=true", key,
                                {"content": "Where is it now?"})
    events = sse(raw) if status == 200 else []
    kinds = [e for e, _ in events]
    done = json.loads(events[-1][1]) if kinds and kinds[-1] == "turn.completed" else {}
    check(headers.get_content_type() == "text/event-stream" and kinds[:1] == ["turn.started"]
          and done.get("stop_reason") == "replied" and "error" not in kinds,
          "a streamed turn, turn.started to turn.completed", f"{status} {kinds}")

    status, _, raw = call(base, "GET", f"/v1/conversations/{cid}/tool-calls", key)
    audit = json.loads(raw) if status == 200 else []
    check([a["tool"] for a in audit[:3]] == EXPECTED_TOOLS
          and [a["tool"] for a in audit if a["is_write"]][:1] == ["cancel_order"],
          "the audit log: every call, the cancellation recorded as a write",
          [(a.get("tool"), a.get("is_write")) for a in audit])

    status, _, raw = call(base, "GET", "/metrics")
    check(status == 200 and b"pasar_turns_total" in raw, "GET /metrics counts the turns")

    # A connection left idle past uvicorn's own five seconds still serves the
    # next request (PASAR_KEEP_ALIVE_SECONDS): the proxy in front closes first.
    import http.client
    import time
    from urllib.parse import urlparse
    u = urlparse(base)
    kind = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = kind(u.hostname, u.port, timeout=30)
    try:
        conn.request("GET", "/healthz")
        first = conn.getresponse()
        first.read()
        time.sleep(6)
        conn.request("GET", "/healthz")
        second = conn.getresponse()
        second.read()
        kept = (first.status, second.status)
    except (http.client.HTTPException, OSError) as e:
        kept = (type(e).__name__,)
    finally:
        conn.close()
    check(kept == (200, 200), "a connection idle six seconds serves the next request", kept)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().split("\n\n")[1])
        return 2
    try:
        smoke(argv[1].rstrip("/"), argv[2])
    except Failed as e:
        print(f"FAIL  {e}")
        return 1
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"FAIL  {type(e).__name__}: {e}")
        return 1
    print("smoke: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
