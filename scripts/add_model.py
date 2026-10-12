"""
One more model on the hard edition (docs/RUNBOOK.md, 1s): its 36 cases and the
117 tasks of their parts, against the customer every model there meets --
customer 3, qwen3.8-flash on DashScope -- and then its reading, V-<name>.txt.

    python scripts/add_model.py

It asks for the provider, the model and the API key. The key is the provider's
variable in .env or your environment, if you say so, or typed: never shown and
never put on a command line. The runs get it in their environment, with every
other provider's key taken out of it, so it goes to the endpoint you chose and
nowhere else (WHAT_FAILED #47). The customer's key is PASARBENCH_SIM_API_KEY or
DASHSCOPE_API_KEY, as the sweep reads it, or typed when neither is set. The
script reads .env itself (v57), as `set -a; . ./.env` would; what your shell
has set already is kept.

    python scripts/add_model.py --provider openai --model <its name> --name <short>
    python scripts/add_model.py ... --dry-run      # the plan; nothing is sent
    python scripts/add_model.py --check            # which keys work, which models

Before the runs it has the model look an order up with its tools and answer
from what the tool returned -- two short requests -- so a wrong key, a wrong
model name, a model that takes no tools or a provider that wants something this
harness does not send stops it there. Run it again with the same --name to
finish a run that stopped: conversations that finished are not run again, and
those an API error ended are.

--check does that for every provider whose key is in your environment, on its
newest fast models (CANDIDATES, and up to three more of that tier its key
lists), lists the models the key can use, and checks the customer's key; it
writes models_check.txt, which names the variables and holds no key. For a
provider of CANDIDATES with no key in your environment it asks for one (typed,
not shown; Enter skips), tries it, and, if a request with it worked, offers to
save it to .env under the name the runs read. When a key is missing it says,
by name only, what .env sets and which of its lines the shell does not read.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# name: (what it is, endpoint, backend, request settings). An endpoint of None
# is typed in. The settings are the provider's way of turning reasoning off,
# where it has one this harness knows: every model in the table runs at
# temperature 0 with nothing hidden added (DeepSeek reasons by default).
PROVIDERS: dict[str, tuple[str, str | None, str, str]] = {
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", ""),
    "google": ("Google (Gemini)", "https://generativelanguage.googleapis.com/v1beta/openai",
               "openai", ""),
    "anthropic": ("Anthropic (Claude)", "https://api.anthropic.com/v1", "anthropic", ""),
    "mistral": ("Mistral", "https://api.mistral.ai/v1", "openai", ""),
    "xai": ("xAI (Grok)", "https://api.x.ai/v1", "openai", ""),
    "zai": ("Z.ai (GLM)", "https://api.z.ai/api/paas/v4", "openai",
            '{"thinking":{"type":"disabled"}}'),
    "mimo": ("Xiaomi (MiMo)", "https://api.xiaomimimo.com/v1", "openai",
             '{"thinking":{"type":"disabled"}}'),
    "openrouter": ("OpenRouter (many labs' models)", "https://openrouter.ai/api/v1",
                   "openai", ""),
    "together": ("Together", "https://api.together.xyz/v1", "openai", ""),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", ""),
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1", "openai",
                 '{"thinking":{"type":"disabled"}}'),
    "other": ("another OpenAI-compatible endpoint", None, "openai", ""),
}
# What --check tries on each provider: its newest fast models, as their own
# pages listed them on 2026-10-13 -- the cheaper, quicker tier, not the
# flagship. The key's own list of models is printed beside them. xAI's fast
# models were retired on 2026-05-15 and Grok 4.7 Fast is not on its public API:
# its cheapest current model is grok-4.3, which xAI names as their replacement.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "zai": ("glm-5.3-flash", "glm-5.3-flashx", "glm-4.7-flash"),
    "mimo": ("mimo-v2.6-flash",),
    "xai": ("grok-4.3", "grok-4.20-0309-non-reasoning"),
    "deepseek": ("deepseek-v4-pro", "deepseek-flash"),
    "openai": (), "google": (), "anthropic": (), "mistral": (), "openrouter": (),
    "together": (), "groq": (),
}
# A model of the fast tier by its name, and names that are not chat models.
FAST = re.compile(r"(?:^|[-_.])(?:fast|flashx?|mini|lite|nano)(?:$|[-_.\d])")
NOT_CHAT = re.compile(r"image|imagine|vision|audio|tts|asr|speech|embed|video|realtime|"
                      r"search|ocr|voice|rerank|moderation")
EXTRA_FAST = 3              # fast-tier models the key lists, tried beyond CANDIDATES
# Words in a variable's name that say whose key it is: for a key set under a
# name nothing here reads (the names it is read under are pasarbench.sweep's
# PROVIDER_KEYS).
OWNERS = {"zai": r"ZAI|Z_AI|GLM|ZHIPU|BIGMODEL", "mimo": r"MIMO|XIAOMI", "xai": r"XAI|GROK",
          "deepseek": r"DEEPSEEK", "openai": r"OPENAI", "google": r"GEMINI|GOOGLE",
          "anthropic": r"ANTHROPIC|CLAUDE", "mistral": r"MISTRAL",
          "openrouter": r"OPENROUTER", "together": r"TOGETHER", "groq": r"GROQ"}
# Retired models a provider still answers for, with another model: a run of
# one would be another model's under its name. xAI, 2026-05-15: each is served
# by grok-4.3 at grok-4.3's price.
RETIRED = {m: "grok-4.3" for m in (
    "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning", "grok-4-fast-reasoning",
    "grok-4-fast-non-reasoning", "grok-4-0709", "grok-3")} | {"grok-code-fast-1": "grok-build-0.1"}
MAX_TOKENS = 2048          # the sweep's own
THINKING_MAX_TOKENS = 4096  # room for a model that cannot stop reasoning


def settings_for(provider: str, model: str) -> tuple[str, int]:
    """A model's request settings and its token limit a step: reasoning off
    where the model lets it be, else as low as it goes, with room for it.
    GLM-5.3 and its Flash cannot stop thinking (Z.ai answers 1210 to
    "disabled"): low effort, 4,096 tokens a step. Grok 4.3 reasons unless told
    "none", xAI's setting for it in place of its retired fast models."""
    m = model.lower()
    if provider == "zai" and m.startswith("glm-5.3"):
        return '{"reasoning_effort":"low"}', THINKING_MAX_TOKENS
    if provider == "xai" and m.startswith("grok-4.3"):
        return '{"reasoning_effort":"none"}', MAX_TOKENS
    return PROVIDERS[provider][3], MAX_TOKENS


# The customer of RUNBOOK 1s: one for every model, so that two models' numbers
# can be paired.
CUSTOMER_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
CUSTOMER_MODEL = "qwen3.8-flash"
CUSTOMER_EXTRA = '{"reasoning_effort":"low"}'
CUSTOMER = ["--simulator", "openai", "--sim-model", CUSTOMER_MODEL, "--sim-url", CUSTOMER_URL,
            "--sim-extra-body", CUSTOMER_EXTRA, "--customer", "3"]
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")


def part_traps() -> str:
    """Each case's part, the suite's trap whose rule it turns on (RUNBOOK 1s)."""
    from pasarbench.generate import EDITION3_PARTS
    return ",".join(dict.fromkeys(EDITION3_PARTS.values()))


def short_name(model: str) -> str:
    """A run name from a model's: lower case, one dash for anything else."""
    s = re.sub(r"[^a-z0-9.]+", "-", model.lower().split("/")[-1]).strip("-.")
    return s[:40] or "model"


def runs(name: str) -> tuple[str, str]:
    return f"V-{name}", f"V-{name}-parts"


def commands(backend: str, url: str, model: str, name: str, extra: str,
             max_tokens: int = MAX_TOKENS, workers: int = 8) -> list[list[str]]:
    """The two sweeps of RUNBOOK 1s for one model. No key is in them."""
    agent = ["--backend", backend, "--model", model]
    if backend == "openai":
        agent += ["--base-url", url]
    if extra:
        agent += ["--extra-body", extra]
    if max_tokens != MAX_TOKENS:
        agent += ["--max-tokens", str(max_tokens)]
    common = [sys.executable, "-m", "pasarbench.sweep", *agent, *CUSTOMER,
              "--strategies", "full", "-k", "3", "--workers", str(workers), "--resume"]
    cases, parts = runs(name)
    return [[*common, "--suite", "edition3", "--run-id", cases],
            [*common, "--suite", "generated", "--traps", part_traps(), "--run-id", parts]]


def readings(name: str) -> list[list[str]]:
    """The reading's two commands; their output goes to V-<name>.txt."""
    cases, parts = runs(name)
    return [[sys.executable, "scripts/hard_vs_parts.py", f"traces/{parts}/full",
             f"traces/{cases}/full"],
            [sys.executable, "scripts/inspect_trace.py", f"traces/{cases}/full", "--leaks-only"]]


def child_env(agent_key: str, customer_key: str) -> dict[str, str]:
    """The runs' environment: this one with every *_API_KEY taken out, and the
    two keys of this run put in under the names the sweep reads first. The
    agent's key can then go only to the agent's endpoint, the customer's only to
    the customer's."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PASARBENCH_API_KEY"] = agent_key
    env["PASARBENCH_SIM_API_KEY"] = customer_key
    return env


def log(name: str, text: str) -> None:
    """A line of the run's own record, logs/add_model-<name>.log: what ran, what
    the check requests met, how each sweep ended -- so that a stop can be read
    after it, from the folder. No key goes in it (v59)."""
    path = ROOT / "logs" / f"add_model-{name}.log"
    try:
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n")
    except OSError:
        pass


def keyless_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}


def recorded_model(name: str) -> str | None:
    """The model an earlier run of this name ran, from a trace header, or None."""
    for run in runs(name):
        for f in sorted((ROOT / "traces" / run / "full").glob("*.jsonl"))[:1]:
            try:
                return json.loads(f.read_text(encoding="utf-8").splitlines()[0]).get(
                    "requested_model")
            except (OSError, ValueError, IndexError):
                return None
    return None


def rivals(name: str) -> list[str]:
    """The other models' names with both runs on disk, for the separation
    reading."""
    out = []
    for d in sorted((ROOT / "traces").glob("V-*")):
        other = d.name[2:]
        if (other != name and not other.endswith("-parts")
                and any((d / "full").glob("*.jsonl"))
                and any((ROOT / "traces" / f"V-{other}-parts" / "full").glob("*.jsonl"))):
            out.append(other)
    return out


def api_errors(name: str) -> int:
    """Conversations the runs left ended by a server or API error."""
    n = 0
    for run in runs(name):
        for f in (ROOT / "traces" / run / "full").glob("*.jsonl"):
            try:
                last = json.loads(f.read_text(encoding="utf-8").splitlines()[-1])
            except (OSError, ValueError, IndexError):
                continue
            n += last.get("type") == "footer" and last.get("stop_reason") == "backend_error"
    return n


def _backend(backend: str, url: str, model: str, key: str, extra: str, max_tokens: int):
    from pasarbench.harness.backends import AnthropicBackend, OpenAICompatBackend
    if backend == "anthropic":
        return AnthropicBackend(model=model, api_key=key, temperature=0.0,
                                max_tokens=max_tokens)
    return OpenAICompatBackend(model=model, base_url=url, api_key=key, temperature=0.0,
                               max_tokens=max_tokens,
                               extra_body=json.loads(extra) if extra else {})


def probe(backend: str, url: str, model: str, key: str, extra: str,
          max_tokens: int = MAX_TOKENS) -> dict:
    """The model, its 20 tools, a customer asking about an order: it should
    look the order up, and then answer from what the tool returned -- the loop
    every step of a run is. What happened, with its tokens and time; raises
    with the provider's answer when a request fails."""
    from pasarbench.harness.types import Message
    from pasarbench.tools import DEFAULT_TOOLS, schemas
    tools = schemas(list(DEFAULT_TOOLS))       # the 20 a run's agent has
    b = _backend(backend, url, model, key, extra, max_tokens)
    msgs = [Message(role="system", content="You are a customer service agent. Use the "
                                           "tools to look things up before you answer."),
            Message(role="user", content="Hi, what is the status of my order GO-10001?")]
    t0 = time.time()
    r = b.chat(msgs, tools)
    out = {"served": r.served_model or model, "requests": 1, "tool": None,
           "answered": bool(r.content), "prompt": r.usage.prompt_tokens,
           "completion": r.usage.completion_tokens, "cached": None}
    if r.tool_calls:
        out["tool"] = r.tool_calls[0].name
        msgs.append(Message(role="assistant", content=r.content or "",
                            tool_calls=r.tool_calls))
        for tc in r.tool_calls:
            msgs.append(Message(role="tool", tool_call_id=tc.id, name=tc.name,
                                content=json.dumps({"order_id": "GO-10001",
                                                    "status": "shipped",
                                                    "carrier": "J&T Express"})))
        r2 = b.chat(msgs, tools)
        out["requests"] = 2
        out["answered"] = bool(r2.content)
        out["prompt"] += r2.usage.prompt_tokens
        out["completion"] += r2.usage.completion_tokens
        # The second request repeats the first one's start: what the
        # provider's cache served of it, which is most of a run's bill.
        out["cached"] = (r2.usage.cached_tokens, r2.usage.prompt_tokens)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def preflight(backend: str, url: str, model: str, key: str, extra: str,
              max_tokens: int = MAX_TOKENS) -> str:
    """probe(), or SystemExit with the provider's answer: what served it."""
    try:
        return probe(backend, url, model, key, extra, max_tokens)["served"]
    except Exception as e:                                    # noqa: BLE001
        raise SystemExit(f"!! the check requests failed, so nothing ran:\n   {_hide(str(e), key)}\n"
                         f"   A wrong key, a model name the provider does not know, a "
                         f"model that takes no tools or no temperature 0, or a setting the "
                         f"provider refuses (--extra-body).")


def _version(model: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", model))


def fast_extras(listed: list[str], tried: tuple[str, ...]) -> list[str]:
    """Fast-tier chat models the key lists beyond the ones tried anyway,
    highest version first: a newer one than CANDIDATES knows shows up."""
    pick = [m for m in listed if m not in tried and m not in RETIRED
            and FAST.search(m.lower()) and not NOT_CHAT.search(m.lower())]
    return sorted(pick, key=_version, reverse=True)[:EXTRA_FAST]


def unread_keys() -> list[tuple[str, str]]:
    """Variables whose name says they hold a provider's key, under a name
    nothing here reads: (name, provider). Names only."""
    from pasarbench.sweep import PROVIDER_KEYS
    known = {n for names in PROVIDER_KEYS.values() for n in names}
    out = []
    for k in sorted(os.environ):
        up = k.upper()
        if k in known or up.startswith("PASARBENCH_") or "KEY" not in up:
            continue
        prov = next((p for p, pat in OWNERS.items() if re.search(pat, up)), None)
        if prov:
            out.append((k, prov))
    return out


def _hide(text: str, *keys: str) -> str:
    """An error's text with any key in it replaced: a provider may echo it."""
    for k in keys:
        if k and len(k) >= 8:
            text = text.replace(k, "<key>")
    return text


# A line of .env the shell reads as an assignment, and one it does not.
ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
SPACED = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s+=|"
                    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=\s+[^\s#]")


def _value(v: str) -> str | None:
    """A value as the shell reads it after NAME=: its quotes, or a comment after a
    space, taken off; None when the shell would not read it as one value."""
    v = v.strip()
    if v[:1] in ("'", '"'):
        end = v.find(v[0], 1)
        return v[1:end] if end > 0 else None
    v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
    return None if re.search(r"\s", v) else v


def dotenv_values(path: Path) -> dict[str, str]:
    """NAME -> value for each line of .env the shell reads as one. The values
    stay in this process: nothing prints them."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_bytes().decode("utf-8", errors="replace").splitlines():
        m = ASSIGN.match(raw)
        if m and not SPACED.match(raw):
            v = _value(m.group(2))
            if v is not None:
                out[m.group(1)] = v
    return out


# The names load_dotenv put in this process's environment, for --check to say.
LOADED: list[str] = []


def load_dotenv(path: Path) -> list[str]:
    """.env's names into this process's environment, as `set -a; . ./.env`
    would put them, where this shell has not set them already: the runs need
    nothing loaded first. The names loaded."""
    for name, value in dotenv_values(path).items():
        if value and not os.environ.get(name, "").strip():
            os.environ[name] = value
            LOADED.append(name)
    return LOADED


def env_report(path: Path) -> list[str]:
    """What .env sets, by name only -- a value is never read out, and a line
    the shell does not read is described, not shown (it may be a key on its
    own): the names it sets, those set under a name nothing here reads, those
    this shell does not have (.env changed since it was loaded), and the lines
    `set -a; . ./.env` skips or misreads."""
    from pasarbench.sweep import PROVIDER_KEYS, provider_key_names
    if not path.exists():
        return [f"There is no {path.name} in {path.parent}."]
    known = {n for names in PROVIDER_KEYS.values() for n in names}
    names, notes, empty = [], [], set()
    text = path.read_bytes().decode("utf-8", errors="replace")    # line ends as they are
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        spaced = SPACED.match(raw)
        m = ASSIGN.match(raw)
        if spaced:
            name = spaced.group(1) or spaced.group(2)
            notes.append(f"line {i}: {name} has a space round its '=', so the shell does not "
                         f"set it -- write {name}=<key>, no spaces")
        elif m:
            names.append(m.group(1))
            v = _value(m.group(2))
            if v is None:
                notes.append(f"line {i}: {m.group(1)}'s value is not one word to the shell "
                             f"(a space in it, or a quote not closed) -- quote it whole")
                empty.add(m.group(1))
            elif not v:
                notes.append(f"line {i}: {m.group(1)} is empty")
                empty.add(m.group(1))
        else:
            notes.append(f"line {i}: not NAME=value, so the shell does not read it (its text is "
                         f"not shown: it may be a key on its own)")
    out = [f"{path.name} sets (names only): {', '.join(names) or 'nothing'}"]
    unloaded = []
    for name in dict.fromkeys(names):
        prov = next((q for q, pat in OWNERS.items() if re.search(pat, name.upper())), None)
        if name not in known and prov and not name.startswith("PASARBENCH_"):
            label, url, _, _ = PROVIDERS[prov]
            notes.append(f"{name}: nothing here reads that name -- for {label}, name it "
                         f"{provider_key_names(url)[0]}")
        elif not os.environ.get(name) and name not in empty:
            unloaded.append(name)
    if unloaded:
        notes.append(f"in {path.name} but not in this shell: {', '.join(unloaded)} -- load it "
                     f"first, set -a; . ./{path.name}; set +a")
    values = dotenv_values(path)
    stale = [n for n, v in values.items()
             if os.environ.get(n, "").strip() and os.environ[n].strip() != v.strip()]
    if stale:
        notes.append(f"this shell's value is not {path.name}'s for {', '.join(stale)} (loaded "
                     f"before {path.name} changed?): the shell's is used -- reload, "
                     f"set -a; . ./{path.name}; set +a")
    if "\r" in text:
        notes.append(f"{path.name} has Windows line ends: a key read from it ends in one, "
                     f"which no request can carry (the runs strip it)")
    return out + notes


def save_key(path: Path, name: str, key: str) -> None:
    """NAME=key into .env: in place of any line that sets NAME, else at its
    end; the file readable by you alone. Quoted when the key has a character
    the shell would read."""
    if "'" in key or "\n" in key:
        raise ValueError("a key with a quote or a line end in it: not saved")
    value = key if re.fullmatch(r"[A-Za-z0-9._+/=:-]+", key) else f"'{key}'"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [ln for ln in lines if not (ASSIGN.match(ln) and ASSIGN.match(ln).group(1) == name)
             and not (SPACED.match(ln) and name in SPACED.match(ln).groups())]
    lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def held_by_other(key: str, own: tuple[str, ...]) -> list[str]:
    """The variables of other providers that hold this very key."""
    return sorted(k for k, v in os.environ.items()
                  if k.endswith("_API_KEY") and not k.startswith("PASARBENCH_")
                  and k not in own and v.strip() == key)


def listed_models(url: str, key: str) -> list[str] | str:
    """The model ids the key can use (GET /models), or why they could not be
    listed."""
    req = urllib.request.Request(url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return _hide(f"HTTP {e.code}: {e.read().decode()[:200]}", key)
    except Exception as e:                                    # noqa: BLE001
        return _hide(str(e), key)[:200]
    rows = data.get("data", data) if isinstance(data, dict) else data
    return sorted({str(m.get("id")) for m in rows if isinstance(m, dict) and m.get("id")})


def check(providers: list[str], models: list[str], out_path: Path,
          explicit: bool = False, ask=None, secret=None) -> int:
    """--check: for each provider whose key is in the environment, the models
    its key lists and the probe on each candidate; and the customer's key.
    With `secret` (an interactive terminal), a provider of CANDIDATES that has
    no key is asked for one, and a typed key that worked may be saved."""
    from pasarbench.sweep import SIM_KEY, key_for, provider_key_env, provider_key_names
    lines = [f"PasarBench model check, {time.strftime('%Y-%m-%d %H:%M')} -- no key is "
             f"written here", ""]

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    env_file = ROOT / ".env"
    if LOADED:
        say(f"Read from {env_file.name} (names only): {', '.join(LOADED)}")
        say()
    missing = [p for p in providers if (explicit or CANDIDATES.get(p))
               and not any(os.environ.get(n, "").strip()
                           for n in provider_key_names(PROVIDERS[p][1] or ""))]
    if missing:
        say("No key in this shell for " + ", ".join(PROVIDERS[p][0] for p in missing) + ".")
        for note in env_report(env_file):
            say("  " + note)
        say()
    found = 0
    for prov in providers:
        label, url, backend, _ = PROVIDERS[prov]
        names = provider_key_names(url or "")
        key = next((os.environ[n].strip() for n in names if os.environ.get(n, "").strip()), "")
        typed = False
        if not key and secret and prov in missing:
            key = secret(f"{label} API key, for {names[0]} (typed, not shown; Enter skips): "
                         ).strip()
            typed = bool(key)
            clash = held_by_other(key, names) if key else []
            if clash:
                say(f"{label}: that is the key in {', '.join(clash)}, another provider's -- "
                    f"not sent to {url}")
                say()
                continue
        if not key:
            if explicit:
                say(f"{label}: no key -- set {' or '.join(names)}")
            continue
        found += 1
        say(f"{label} -- {url} -- key " + ("typed" if typed else f"from {provider_key_env(url)}"))
        listed = listed_models(url, key) if backend == "openai" else "(not listed)"
        if isinstance(listed, list):
            say(f"  your key lists {len(listed)} models: {', '.join(listed)}")
        else:
            say(f"  its model list: {listed}")
        tried = tuple(models or CANDIDATES.get(prov, ()))
        if not models and isinstance(listed, list):
            tried += tuple(fast_extras(listed, tried))
        worked = isinstance(listed, list)
        for model in tried:
            if isinstance(listed, list) and listed and model not in listed:
                say(f"  {model:32s} not in your key's list -- not tried")
                continue
            extra, mt = settings_for(prov, model)
            try:
                r = probe(backend, url, model, key, extra, mt)
                worked = True
                how = (f"looked it up with {r['tool']} and answered" if r["tool"] and
                       r["answered"] else f"called {r['tool']}, then no answer"
                       if r["tool"] else "answered without a tool" if r["answered"]
                       else "returned nothing")
                say(f"  {model:32s} OK   {how}; {r['requests']} requests, {r['seconds']} s, "
                    f"tokens in {r['prompt']:,} out {r['completion']:,}"
                    + (f", from cache on the 2nd {r['cached'][0]:,} of {r['cached'][1]:,}"
                       if r.get("cached") else "")
                    + (f"; settings {extra}" if extra else "")
                    + (f"; served as {r['served']}" if r["served"] != model else ""))
            except Exception as e:                            # noqa: BLE001
                say(f"  {model:32s} FAIL {_hide(str(e), key)[:300]}")
        if typed and not worked:
            say("  the key you typed: no request with it worked, so it is not saved")
        elif typed and ask and _ask(ask, f"  Save it to {env_file.name} as {names[0]}, so the "
                                         f"runs find it? (y/n)", "y").lower().startswith("y"):
            try:
                save_key(env_file, names[0], key)
                os.environ[names[0]] = key
                say(f"  saved to {env_file.name} as {names[0]}")
            except (OSError, ValueError) as e:
                say(f"  not saved: {_hide(str(e), key)}")
        say()
    keys = sorted(k for k in os.environ if k.endswith("_API_KEY"))
    if not found:
        say("No provider's key found, in this shell or in .env.")
    say(f"API key variables set (names only): {', '.join(keys) or 'none'}")
    for name, prov in unread_keys():
        label, url, _, _ = PROVIDERS[prov]
        say(f"{name} is set, but nothing here reads that name: for {label}, name it "
            f"{provider_key_names(url)[0]}")
    # The customer's key as the sweep reads it: this run's (PASARBENCH_SIM_API_KEY),
    # else DashScope's own.
    ck = key_for(CUSTOMER_URL, SIM_KEY)
    src = SIM_KEY if os.environ.get(SIM_KEY, "").strip() else "DASHSCOPE_API_KEY"
    if ck:
        try:
            r = probe("openai", CUSTOMER_URL, CUSTOMER_MODEL, ck, CUSTOMER_EXTRA, 256)
            say(f"The customer, {CUSTOMER_MODEL} on DashScope, key from {src}: OK "
                f"({r['seconds']} s)")
        except Exception as e:                                # noqa: BLE001
            say(f"The customer, {CUSTOMER_MODEL} on DashScope, key from {src}: FAIL "
                f"{_hide(str(e), ck)[:300]}")
    else:
        say(f"The customer's key ({SIM_KEY} or DASHSCOPE_API_KEY) is not set: the runs will "
            f"ask for it")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {out_path}")
    return 0


def _ask(ask, prompt: str, default: str = "") -> str:
    got = ask(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return got or default


def main(argv: list[str] | None = None, ask=input, secret=getpass.getpass) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--provider", choices=sorted(PROVIDERS))
    ap.add_argument("--model", help="the model's name as the provider names it")
    ap.add_argument("--base-url", help="the endpoint, for --provider other")
    ap.add_argument("--name", help="short name for its runs: V-<name> and V-<name>-parts")
    ap.add_argument("--extra-body", help="request settings as JSON (default: the provider's "
                                         "in the table, e.g. DeepSeek's reasoning off)")
    ap.add_argument("--yes", action="store_true", help="do not ask before running")
    ap.add_argument("--workers", type=int, default=8,
                    help="conversations at once (default 8): fewer for a provider that "
                         "limits how many requests an account has open")
    ap.add_argument("--dry-run", action="store_true", help="the plan only; nothing is sent")
    ap.add_argument("--check", action="store_true",
                    help="which keys work and which models: every provider whose key is "
                         "set (or --provider), its newest fast models (or --model); "
                         "writes models_check.txt")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    from pasarbench.models import check_agent_simulator
    from pasarbench.sweep import SIM_KEY, key_for, provider_key_env, provider_key_names
    if a.check:
        provs = [a.provider] if a.provider else [p for p in PROVIDERS if p != "other"]
        # Someone to type a key: a terminal, or the caller's own prompts.
        there = ask is not input or sys.stdin.isatty()
        return check(provs, [a.model] if a.model else [], ROOT / "models_check.txt",
                     explicit=bool(a.provider), ask=ask if there else None,
                     secret=secret if there else None)

    print("The hard edition (RUNBOOK 1s) on one more model: its 36 cases and their "
          "parts, against customer 3.\n")
    provider = a.provider
    if not provider:
        names = list(PROVIDERS)
        for i, p in enumerate(names, 1):
            label, url, _, _ = PROVIDERS[p]
            print(f"  {i:2d}. {label:36s} {url or '(you type it)'}")
        pick = _ask(ask, "\nProvider (number or name)")
        provider = names[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(names) \
            else pick.lower()
        if provider not in PROVIDERS:
            raise SystemExit(f"no provider {pick!r}: one of {', '.join(names)}")
    label, url, backend, extra = PROVIDERS[provider]
    url = a.base_url or url or _ask(ask, "Endpoint (OpenAI-compatible, ending in /v1)")
    if not url.startswith("https://"):
        raise SystemExit(f"{url!r}: an https:// endpoint, so that the key travels encrypted")
    model = a.model or _ask(ask, "Model, as the provider names it")
    if not model:
        raise SystemExit("no model: nothing ran")
    try:
        check_agent_simulator(model, CUSTOMER_MODEL)
    except SystemExit:
        raise SystemExit(f"{model} is a Qwen model, and so is the customer every model in 1s "
                         f"meets: a customer of the agent's own family is easier for it. "
                         f"Pick a model from another lab.")
    if model in RETIRED:
        raise SystemExit(f"{model} is retired: its provider answers for it with "
                         f"{RETIRED[model]}, at {RETIRED[model]}'s price, so its runs would be "
                         f"another model's under its name. Run {RETIRED[model]}.")
    extra, max_tokens = settings_for(provider, model)
    extra = a.extra_body if a.extra_body is not None else extra
    if extra:
        try:
            if not isinstance(json.loads(extra), dict):
                raise ValueError
        except ValueError:
            raise SystemExit(f"--extra-body {extra!r} is not a JSON object")
    name = a.name or (short_name(model) if a.model else
                      _ask(ask, "Short name for its runs", short_name(model)))
    if not NAME.match(name):
        raise SystemExit(f"{name!r}: letters, digits, '.', '_' and '-' only")
    earlier = recorded_model(name)
    if earlier and earlier != model:
        raise SystemExit(f"V-{name} already holds {earlier}'s runs: choose another name")
    cases, parts = runs(name)
    key_env = provider_key_env(url)

    print(f"\n  model     {model} on {url} ({label})"
          + (f", request settings {extra}" if extra else "")
          + (f", {max_tokens:,} tokens a step" if max_tokens != MAX_TOKENS else ""))
    print(f"  customer  {CUSTOMER_MODEL} on DashScope, customer 3, as every model in 1s")
    print(f"  runs      {cases}: the 36 hard cases, 108 conversations")
    print(f"            {parts}: their parts, 117 conversations")
    print(f"  reading   V-{name}.txt (RUNBOOK 1s)"
          + (f"; continuing {cases}" if earlier else ""))
    if a.dry_run:
        print("\n(dry run: nothing sent) The runs would be, each with the two keys in its "
              "environment as PASARBENCH_API_KEY and PASARBENCH_SIM_API_KEY and no other "
              "*_API_KEY:")
        for c in commands(backend, url, model, name, extra, max_tokens, a.workers):
            print("  " + " ".join(c[1:]))
        for c in readings(name):
            print("  " + " ".join(c[1:]))
        return 0

    print()
    log(name, f"start: {model} on {url}, settings {extra or 'none'}, {max_tokens:,} tokens a "
              f"step; runs {cases} and {parts}")
    if key_env and os.environ.get(key_env) and _ask(
            ask, f"Use {key_env} from your environment for {model}? (y/n)", "y").lower() \
            .startswith("y"):
        agent_key = os.environ[key_env].strip()
    else:
        agent_key = secret(f"API key for {url} (typed, not shown, not saved): ").strip()
    if not agent_key:
        raise SystemExit("no key: nothing sent")
    own = set(provider_key_names(url))
    others = {k for k, v in os.environ.items()
              if k.endswith("_API_KEY") and not k.startswith("PASARBENCH_")
              and k not in own and v.strip() == agent_key}
    if others:
        log(name, f"refused: the key typed is the one in {', '.join(sorted(others))}")
        raise SystemExit(f"that is the key in {', '.join(sorted(others))}, another "
                         f"provider's: not sent to {url}")
    customer_key = key_for(CUSTOMER_URL, SIM_KEY) or secret(
        "DashScope key for the customer (typed, not shown, not saved): ").strip()
    if not customer_key:
        raise SystemExit("no key for the customer: nothing sent")
    if customer_key == agent_key:
        raise SystemExit("the agent's key and the customer's are the same: each provider "
                         "gets its own")

    print(f"Checking that {model} looks an order up with its tools and answers ...",
          flush=True)
    try:
        served = preflight(backend, url, model, agent_key, extra, max_tokens)
    except SystemExit as e:
        log(name, f"stopped at the check requests, nothing ran: {e}")
        print(f"   (written to logs/add_model-{name}.log)")
        raise
    log(name, f"check requests ok: answered as {served}")
    print(f"  ok: answered as {served}")
    if not a.yes and not _ask(ask, "Run them? (y/n)", "n").lower().startswith("y"):
        print("Nothing ran.")
        return 0

    env = child_env(agent_key, customer_key)
    for c in commands(backend, url, model, name, extra, max_tokens, a.workers):
        rc = subprocess.run(c, env=env).returncode
        log(name, f"sweep {c[c.index('--run-id') + 1]} ended, exit code {rc}")
        if rc:
            print("!! a sweep stopped. Run the same command again: it carries on.")
            return 1
    left = api_errors(name)
    if left:
        log(name, f"{left} conversations ended on a server or API error; no reading yet")
        print(f"!! {left} conversations ended on a server or API error. Run the same "
              f"command again before reading: only those run again.")
        return 1
    out = Path(f"V-{name}.txt")
    with out.open("w", encoding="utf-8") as fh:
        for c in readings(name):
            fh.flush()
            subprocess.run(c, env=keyless_env(), stdout=fh, stderr=subprocess.STDOUT)
        # Separation (RUNBOOK 1s): this model against every other one run so
        # far, on the cases and on their parts.
        for other in rivals(name):
            for what, x, y in (("the hard cases", runs(other)[0], runs(name)[0]),
                               ("their parts", runs(other)[1], runs(name)[1])):
                fh.write(f"\n== {other} -> {name}, {what}, paired by task\n")
                fh.flush()
                subprocess.run([sys.executable, "scripts/compare_cells.py",
                                f"traces/{x}/full", f"traces/{y}/full"],
                               env=keyless_env(), stdout=fh, stderr=subprocess.STDOUT)
    log(name, f"reading written: {out}" + (f", paired with {', '.join(rivals(name))}"
                                           if rivals(name) else ""))
    print(f"\nDone: {out} -- the hard edition against its parts for {model}"
          + (f", and against {', '.join(rivals(name))}" if rivals(name) else "")
          + " (RUNBOOK 1s).")
    return 0


if __name__ == "__main__":
    # .env read here, by name: no `set -a; . ./.env; set +a` needed first. What
    # this shell has set already is kept.
    load_dotenv(ROOT / ".env")
    raise SystemExit(main())
