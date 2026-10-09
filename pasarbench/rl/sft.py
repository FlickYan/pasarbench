"""
RFT training examples from saved conversations.

A training example is ONE AGENT TURN: the prompt exactly as the model saw it
when it produced that turn, and the turn itself. Loss goes on the turn only.

WHY PER TURN, AND NOT ONE MASKED CONVERSATION
---------------------------------------------
The cheap way to do multi-turn SFT renders the whole conversation once and
masks everything but the assistant spans. It is cheap because every turn shares
one forward pass. It is wrong for Qwen3 and for any template that renders the
latest assistant turn differently from earlier ones: in non-thinking mode the
model generated every turn after `<think>\\n\\n</think>\\n\\n`, but a rendered
conversation shows that empty block only on the LAST turn. Each earlier turn
would be trained on a prefix the model never saw at inference.

So each turn's prompt is rendered on its own with the generation prompt, the
same call the server made, and the turn is whatever the template adds after
it. The template is asked for both renderings and the code asserts that one is
a prefix of the other; a template that breaks that property fails loudly here
instead of quietly training on a prompt that never existed.

Better still, with `build --server` the serving engine (SGLang) renders every
prompt itself through /tokenize, and training uses its token ids: the prompt
then IS the inference prompt, whatever the local transformers version does. It
is not a formality. SGLang hands the template each tool as its own pydantic
dump -- `"strict": false` and all, in its own key order -- so the tool block the
model was shown differs from what the same template renders from the schemas.

The TURN is tokenized locally, from the template's rendering of it. SGLang
cannot render a finished turn: /tokenize, like the chat endpoint, always
appends the generation prompt, and turns a trailing assistant message into a
user message. The turn is what the model generated after the prompt, so it is
tokenized on its own, never joined to the prompt text.

The prefix of every example is the ~4k-token policy and tool block, so this
costs several times more compute than the masked conversation. At this suite's
size that is an hour or two of GPU time, and it buys training on the prompts
the model will actually be given.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..harness.backends import OpenAICompatBackend
from ..harness.types import Message
from ..tools import schemas
from .split import family_of


class TemplateDrift(RuntimeError):
    """The chat template renders a turn differently in and out of context."""


# End-of-turn markers, most specific first. The tokenizer's own eos_token is
# not always the turn terminator: some checkpoints set it to <|endoftext|>
# while the chat template closes turns with <|im_end|>.
TURN_END = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>")


def turn_end(tokenizer, text: str) -> str | None:
    """The marker that closes an assistant turn in `text`, if any."""
    for m in (tokenizer.eos_token, *TURN_END):
        if m and m in text:
            return m
    return None


def read_trace(path: str | Path) -> dict[str, Any]:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    head = next((r for r in recs if r.get("type") == "header"), {})
    foot = next((r for r in reversed(recs) if r.get("type") == "footer"), {})
    msgs = next((r for r in recs if r.get("type") == "messages"), None)
    return {"path": str(path), "header": head, "footer": foot, "messages": msgs,
            "steps": [r for r in recs if r.get("type") == "step"]}


def chat_kwargs(header: dict[str, Any]) -> dict[str, Any]:
    """The chat_template_kwargs the collecting sweep sent, e.g. enable_thinking."""
    return dict(((header.get("agent_extra_body") or {}).get("chat_template_kwargs")) or {})


def api_messages(saved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages as the backend put them on the wire. Same function the
    backend uses, so a training prompt cannot drift from the request."""
    return [OpenAICompatBackend._msg(Message.from_dict(d)) for d in saved]


def _for_template(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the serving side hands the template: SGLang parses tool-call
    argument strings back into objects before rendering. The Qwen template
    prints a string verbatim and an object through tojson, which give the same
    text for what the harness sends; parsing keeps it true for templates that
    differ (Gemma 4's refuses a string outright)."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
                calls.append({**tc, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def render_turn(tokenizer, msgs: list[dict[str, Any]], i: int,
                tools: list[dict[str, Any]] | None, kwargs: dict[str, Any]
                ) -> tuple[str, str]:
    """(prompt, completion) for the assistant message at index i."""
    if msgs[i]["role"] != "assistant":
        raise ValueError(f"message {i} is {msgs[i]['role']}, not assistant")
    tm = _for_template(msgs)
    prompt = tokenizer.apply_chat_template(tm[:i], tools=tools or None,
                                           add_generation_prompt=True,
                                           tokenize=False, **kwargs)
    full = tokenizer.apply_chat_template(tm[:i + 1], tools=tools or None,
                                         add_generation_prompt=False,
                                         tokenize=False, **kwargs)
    if not full.startswith(prompt):
        k = next((j for j, (a, b) in enumerate(zip(prompt, full)) if a != b),
                 min(len(prompt), len(full)))
        raise TemplateDrift(
            f"turn {i}: the template renders the prompt differently once the "
            f"turn is appended, from char {k}:\n  prompt: {prompt[max(0, k - 60):k + 60]!r}"
            f"\n  full:   {full[max(0, k - 60):k + 60]!r}")
    completion = full[len(prompt):]
    eos = turn_end(tokenizer, completion)
    if eos is None:
        raise TemplateDrift(f"turn {i}: no end-of-turn marker after the assistant "
                            f"message; the model would not learn to stop: "
                            f"{completion[-120:]!r}")
    completion = completion[:completion.find(eos) + len(eos)]
    if "<|im_start|>" in completion:
        raise TemplateDrift(f"turn {i}: the completion runs into another turn: "
                            f"{completion[:200]!r}")
    return prompt, completion


def tokenize_example(tokenizer, ex: dict[str, Any]) -> dict[str, list[int]]:
    """input_ids and labels for one example: loss on the completion only.

    When the build took the prompt's token ids from the server, they are used
    as they are -- they ARE the inference prompt. Prompt and completion are
    always tokenized separately, as they were at inference: the server
    tokenized the rendered prompt, and the model produced the turn token by
    token after it. Tokenizing the joined string could merge tokens across the
    boundary into a sequence the model never saw."""
    if ex.get("prompt_ids") is not None and ex.get("completion_ids") is not None:
        p, c = ex["prompt_ids"], ex["completion_ids"]
    else:
        p = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
    return {"input_ids": p + c, "labels": [-100] * len(p) + c}


def server_renderer(url: str, model: str):
    """Prompt token ids the serving engine builds for a list of messages, via
    SGLang's /tokenize: the same message processing and template call as a chat
    request, so the ids are the ones inference ran on.

    The only rendering that is certain to match inference is the server's own:
    it applies the template after its own message processing (tool-call
    arguments, empty contents, its serialisation of the tool list), and a local
    transformers install can differ from it in ways no test here can see.
    It renders PROMPTS only: the generation prompt is always appended, and a
    trailing assistant message would be sent as a user message."""
    import urllib.error
    import urllib.request
    base = url.rstrip("/").removesuffix("/v1")

    def render(msgs: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
               kwargs: dict[str, Any]) -> list[int]:
        if msgs and msgs[-1].get("role") == "assistant":
            raise ValueError("a prompt ends before the assistant turn; the server "
                             "would send a trailing assistant message as a user one")
        body: dict[str, Any] = {"model": model, "messages": msgs,
                                "chat_template_kwargs": kwargs}
        if tools:
            body["tools"] = tools
        # The same placeholder key the sweep sends a local server.
        req = urllib.request.Request(base + "/tokenize", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer EMPTY"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["tokens"]
        except urllib.error.HTTPError as e:
            raise ServerRenderError(f"{base}/tokenize refused the request (HTTP "
                                    f"{e.code}: {e.read()[:300]!r})") from None
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            raise ServerRenderError(f"{base}/tokenize: {e}") from None
    return render


class ServerRenderError(RuntimeError):
    """The serving engine could not render a request through /tokenize."""



def action_signature(saved: list[dict[str, Any]]) -> str:
    """What the agent did, ignoring how it phrased it: two passing episodes with
    the same tool calls in the same order teach the same thing twice."""
    calls = [(tc["name"], json.dumps(tc.get("arguments", {}), sort_keys=True))
             for m in saved if m.get("role") == "assistant"
             for tc in m.get("tool_calls") or []]
    return json.dumps(calls)


def episode_examples(ep: dict[str, Any], tokenizer, kwargs: dict[str, Any],
                     server=None) -> tuple[list[dict[str, Any]], Counter]:
    """Per-turn examples for one saved episode, and counts of what was skipped.

    With `server` (a `server_renderer`), each example also carries the server's
    own token ids for its prompt, which training uses, the turn's ids from the
    local tokenizer, and whether the local template rendered the same prompt."""
    skipped: Counter = Counter()
    rec = ep["messages"]
    saved = rec["messages"]
    msgs = api_messages(saved)
    assistant = [i for i, m in enumerate(saved) if m["role"] == "assistant"]
    tool_names, ctx = rec.get("tool_names") or [], rec.get("context_messages") or []
    if len(assistant) != len(tool_names):
        raise ValueError(f"{ep['path']}: {len(assistant)} assistant turns but "
                         f"{len(tool_names)} recorded model calls")
    out = []
    for j, i in enumerate(assistant):
        # With full context the j-th call sent exactly the first i messages. If
        # it did not, the trace is from another context strategy and its
        # prompts cannot be rebuilt from the conversation.
        if j < len(ctx) and ctx[j] != i:
            raise ValueError(f"{ep['path']}: call {j + 1} sent {ctx[j]} messages, "
                             f"not {i}; only full-context runs can be trained on")
        m = saved[i]
        if not (m.get("content") or "").strip() and not m.get("tool_calls"):
            skipped["empty turn"] += 1
            continue
        tools = schemas(tool_names[j]) if tool_names[j] else None
        prompt, completion = render_turn(tokenizer, msgs, i, tools, kwargs)
        ex = {"trace": ep["path"], "message_index": i, "turn": j,
              "prompt": prompt, "completion": completion}
        if server is not None:
            p_ids = server(msgs[:i], tools, kwargs)
            local = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ex["prompt_ids"] = p_ids
            ex["completion_ids"] = tokenizer(completion, add_special_tokens=False)["input_ids"]
            # Not a reason to drop the example: training uses the server's
            # prompt. `check` shows where the two renderings part.
            ex["prompt_matches_local"] = local == p_ids
            # This one is. The turn is the local template's and is trained
            # after the server's prompt, so both must end in the same
            # generation prompt -- a server that ignored enable_thinking, say,
            # ends without the empty think block the turn was rendered after.
            ex["prompt_end_matches_local"] = p_ids[-8:] == local[-8:]
        out.append(ex)
    return out, skipped


def build(run_dir: str | Path, tasks: dict[str, Any], folds: dict[str, Any],
          tokenizer, max_per_task: int = 3, max_len: int = 16384,
          expect_model: str | None = None, server=None, checker: str = "current"
          ) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Examples per fold from a collection run, and the stats to read first.

    Fold X's file holds examples from fold X's tasks only; the model trained on
    it is evaluated on the other fold.

    `checker` decides which episodes passed. A trace's footer holds the verdict
    of the checker the day it ran; once a check is corrected that verdict is
    stale, and training on it teaches the stale check. It did: v18's peak and
    photo checks demanded one lookup tool each, the fine-tune learned to call
    them, and most of what it gained was that (WHAT_FAILED #30). "current"
    re-scores every episode under today's checks (pasarbench/rescore.py);
    "recorded" reproduces a build from before v19."""
    from ..tasks import task_digest
    if checker not in ("current", "recorded"):
        raise ValueError(f"checker is 'current' or 'recorded', not {checker!r}")

    eps = [read_trace(p) for p in sorted(Path(run_dir).glob("*.jsonl"))]
    if not eps:
        raise SystemExit(f"no traces in {run_dir}")
    no_msgs = [e["path"] for e in eps if e["messages"] is None]
    if no_msgs:
        raise SystemExit(f"{len(no_msgs)} of {len(eps)} traces have no saved "
                         f"conversation (e.g. {no_msgs[0]}). Collect with "
                         f"`--save-messages`.")

    kw = {json.dumps(chat_kwargs(e["header"]), sort_keys=True) for e in eps}
    if len(kw) != 1:
        raise SystemExit(f"{run_dir} mixes chat_template_kwargs {sorted(kw)}; "
                         f"one run, one prompt format")
    kwargs = json.loads(kw.pop())
    models = Counter(e["header"].get("requested_model") for e in eps)
    if expect_model and set(models) != {expect_model}:
        raise SystemExit(f"{run_dir} was collected from {dict(models)}, not "
                         f"{expect_model}. RFT trains a model on its own passing "
                         f"episodes; another model's are a different method.")
    contexts = {e["header"].get("context") for e in eps}
    if contexts != {"full"}:
        raise SystemExit(f"{run_dir} used context {sorted(map(str, contexts))}; "
                         f"collect with --strategies full")
    if any(e["header"].get("guardrail", "off") != "off" for e in eps):
        raise SystemExit(f"{run_dir} ran with the claim guardrail: its held-back "
                         f"replies and the notes that answered them are not turns "
                         f"to train on. Collect without --guardrail")
    if any(e["header"].get("closing", "off") != "off" for e in eps):
        raise SystemExit(f"{run_dir} ran with the closing check: the note after the "
                         f"customer left is the harness's, not a turn to train on. "
                         f"Collect without --closing")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        tid = e["header"]["task_id"]
        t = tasks.get(tid)
        if t is None or e["header"].get("task_digest") not in (None, task_digest(t)):
            raise SystemExit(f"{e['path']}: task {tid} has changed since this was "
                             f"collected; re-collect")
        by_task[tid].append(e)
        e["passed_recorded"] = bool(e["footer"].get("passed"))
        e["passed"] = e["passed_recorded"]
        if checker == "current" and e["footer"]:
            from ..rescore import rescore_file
            e["passed"] = rescore_file(e["path"], tasks).verdict

    tf = folds["task_fold"]
    files: dict[str, list[dict]] = {f: [] for f in folds["folds"]}
    skipped: Counter = Counter()
    trap_runs: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    used_eps: Counter = Counter()
    for tid, group in sorted(by_task.items()):
        fold = tf[tid]
        for e in group:
            trap_runs[fold][tasks[tid].trap].append(e["passed"])
        passed = sorted((e for e in group if e["passed"]),
                        key=lambda e: (len(e["steps"]), e["path"]))
        seen, n = set(), 0
        for e in passed:
            sig = action_signature(e["messages"]["messages"])
            if sig in seen:
                skipped["duplicate episode"] += 1
                continue
            if n >= max_per_task:
                skipped["over max_per_task"] += 1
                continue
            seen.add(sig)
            exs, sk = episode_examples(e, tokenizer, kwargs, server)
            skipped.update(sk)
            for ex in exs:
                n_tok = len(tokenize_example(tokenizer, ex)["input_ids"])
                if n_tok > max_len:
                    skipped["longer than max_len"] += 1
                    continue
                files[fold].append({"task_id": tid, "trap": tasks[tid].trap,
                                    "family": family_of(tid), "fold": fold,
                                    "tokens": n_tok, **ex})
            n += 1
            used_eps[fold] += 1

    per_trap = {fold: {trap: round(sum(v) / len(v), 3) for trap, v in sorted(tr.items())}
                for fold, tr in trap_runs.items()}
    stats = {
        "source": str(run_dir),
        "model": next(iter(models)),
        "chat_template_kwargs": kwargs,
        "rendered_by": "server /tokenize" if server is not None else "local tokenizer",
        # How many server prompts the local template does not reproduce. Not an
        # error -- those examples train on the server's prompt -- but the number
        # says whether a build WITHOUT the server would have been wrong.
        "prompts_unlike_local_render": (
            sum(x.get("prompt_matches_local") is False for v in files.values() for x in v)
            if server is not None else None),
        "prompts_ending_unlike_local_render": (
            sum(x.get("prompt_end_matches_local") is False for v in files.values() for x in v)
            if server is not None else None),
        "split": folds.get("split_digest"),
        "episodes": len(eps),
        # Which verdicts picked the examples, and how far they are from the
        # recorded ones: a large gap means the checker changed since collection.
        "checker": checker,
        "pass_rate": round(sum(e["passed"] for e in eps) / len(eps), 4),
        "pass_rate_recorded": round(sum(e["passed_recorded"] for e in eps) / len(eps), 4),
        "verdicts_changed": sum(e["passed"] != e["passed_recorded"] for e in eps),
        "episodes_used": dict(used_eps),
        "examples": {f: len(v) for f, v in files.items()},
        "tokens": {f: sum(x["tokens"] for x in v) for f, v in files.items()},
        "skipped": dict(skipped),
        # Per fold, because each model trains on one fold: a trap its fold never
        # solved gives that model nothing to learn from, whatever the other
        # fold managed.
        "per_trap_pass_rate": per_trap,
        "traps_with_zero_signal": {f: [t for t, r in pt.items() if r == 0.0]
                                   for f, pt in per_trap.items()},
    }
    return files, stats
