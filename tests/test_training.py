"""
Post-training pipeline: split, customer, collection, examples, training.

Each check is one of the ways the first version of this pipeline would have
produced a number that meant nothing: sixteen tasks instead of 215, twins on
both sides of the split, a Qwen customer for a Qwen agent, prompts the model
never saw, tool output in the trained text, and no traces to audit.

Tests that need `transformers` (examples) or `torch` + `peft` (a two-step LoRA
run on a toy model) are skipped when those are not installed; the rest are
pure Python.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pasarbench.generate import generate
from pasarbench.tasks import TASKS

PASS, FAIL, SKIP = [], [], []
ALL = list(TASKS) + list(generate()[0])
BY_ID = {t.task_id: t for t in ALL}


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + ("" if ok else f"  -- {str(detail)[:600]}"))


def skip(label, why):
    SKIP.append(label)
    print(f"  [--] {label}  (skipped: {why})")


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------

def test_split():
    print("\n=== the split: families, not tasks ===")
    from pasarbench.rl.split import family_of, load_folds, make_folds
    f = make_folds(ALL)
    tf = f["task_fold"]
    check("all 215 tasks are assigned", len(tf) == 215 == len(ALL), len(tf))
    fam = {}
    split_fams = {family_of(t) for t, x in tf.items() if fam.setdefault(family_of(t), x) != x}
    check("locale twins never straddle the split", not split_fams, split_fams)
    check("HRWR-ID and HRWR-ID.id land together", tf["HRWR-ID"] == tf["HRWR-ID.id"])
    traps = {}
    for t in ALL:
        traps.setdefault(t.trap, set()).add(tf[t.task_id])
    check("every trap is in both folds", all(v == {"A", "B"} for v in traps.values()),
          {k: v for k, v in traps.items() if v != {"A", "B"}})
    check("the split is reproducible", make_folds(ALL)["split_digest"] == f["split_digest"])
    check("…and a different seed is a different split",
          make_folds(ALL, seed=1)["split_digest"] != f["split_digest"])

    tmp = Path(tempfile.mkdtemp()) / "folds.json"
    tmp.write_text(json.dumps(f))
    changed = [SimpleNamespace(**{**vars(ALL[0]), "opening": "something else"})] + ALL[1:]
    try:
        load_folds(tmp, changed)
        ok = False
    except SystemExit as e:
        ok = "changed since the split" in str(e)
    check("a split over tasks that have since changed is refused", ok)


def test_family_guard():
    print("\n=== the customer is not the agent's family ===")
    from pasarbench.models import check_agent_simulator, model_family
    check("Qwen3-8B and Qwen3.8-27B are qwen",
          model_family("Qwen/Qwen3-8B") == model_family("Qwen/Qwen3.8-27B") == "qwen")
    check("SeaLLMs and Sailor count as qwen underneath",
          model_family("SeaLLMs/SeaLLMs-v3-7B-Chat") == "qwen"
          and model_family("sail/Sailor2-8B-Chat") == "qwen")
    check("Gemma-SEA-LION counts as gemma",
          model_family("aisingapore/Gemma-SEA-LION-v4-27B-IT") == "gemma")
    try:
        check_agent_simulator("Qwen/Qwen3.8-27B", "qwen3.8-flash")
        ok = False
    except SystemExit as e:
        ok = "both qwen" in str(e)
    check("a Qwen agent with a Qwen customer is refused", ok)
    check("the published pairing (DeepSeek agent, Qwen customer) passes",
          "agent deepseek, simulator qwen" in check_agent_simulator("deepseek-v4-pro",
                                                                    "qwen3.8-flash"))
    from pasarbench.models import SIM_DEFAULT
    check("…as does Qwen with the pipeline's customer, Gemma 4",
          "simulator gemma" in check_agent_simulator("Qwen/Qwen3.8-27B", SIM_DEFAULT))
    check("an adapter requested SGLang's way carries its base's family",
          model_family("Qwen/Qwen3.8-27B:pasar-rft-A") == "qwen")
    try:
        check_agent_simulator("Qwen/Qwen3.8-27B:pasar-rft-A", "qwen3.8-flash")
        ok = False
    except SystemExit:
        ok = True
    check("…so a fine-tune is refused a customer from its base's family", ok)
    check("a bare adapter name is flagged, not blocked",
          "check by hand" in check_agent_simulator("pasar-rft-A", SIM_DEFAULT))
    check("the escape hatch exists for measuring exactly this",
          "qwen" in check_agent_simulator("Qwen/Qwen3-8B", "qwen3.8-flash", allow_same=True))


def _sweep(root: Path, run: str, tasks: str, k: int, *extra: str) -> int:
    import subprocess
    cmd = [sys.executable, "-m", "pasarbench.sweep", "--backend", "scripted",
           "--suite", "all", "--tasks", tasks, "-k", str(k), "--strategies", "full",
           "--trace-root", str(root), "--run-id", run, *extra]
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent.parent)


def test_sweep_collection():
    print("\n=== collection goes through the sweep: traces, resume, routing ===")
    root = Path(tempfile.mkdtemp())
    r = _sweep(root, "C", "T01,HRWR-ID,HRWR-ID.id", 2, "--save-messages")
    check("a collection sweep runs", r.returncode == 0, r.stderr[-800:])
    cell = root / "C" / "full"
    recs = [json.loads(l) for l in (cell / "HRWR-ID.id__r1.jsonl").read_text().splitlines()]
    m = next((x for x in recs if x["type"] == "messages"), None)
    steps = [x for x in recs if x["type"] == "step"]
    check("the full conversation is saved in the trace", m is not None
          and m["messages"][0]["role"] == "system" and len(m["tool_names"]) == len(steps))
    check("…with the tool list and context size of every model call",
          m and len(m["tool_names"][0]) == 20 and m["context_messages"][0] == 2, m and m["context_messages"])
    check("…and the audits' view of the trace is unchanged",
          [x["type"] for x in recs][0] == "header" and recs[-1]["type"] == "footer")

    (cell / "T01__r1.jsonl").unlink()
    part = (cell / "HRWR-ID__r0.jsonl").read_text().splitlines()[:2]
    (cell / "HRWR-ID__r0.jsonl").write_text("\n".join(part) + "\n")
    r = _sweep(root, "C", "T01,HRWR-ID,HRWR-ID.id", 2, "--save-messages", "--resume")
    check("resume re-runs only the missing and the half-written episodes",
          "resuming: 4 finished, 2 to run" in r.stdout, r.stdout[-600:])
    check("…and every episode ends up finished",
          all(json.loads(p.read_text().splitlines()[-1])["type"] == "footer"
              for p in cell.glob("*.jsonl")))

    lines = (cell / "HRWR-ID.id__r1.jsonl").read_text().splitlines()
    foot = json.loads(lines[-1])
    lines[-1] = json.dumps({**foot, "stop_reason": "backend_error", "passed": False})
    (cell / "HRWR-ID.id__r1.jsonl").write_text("\n".join(lines) + "\n")
    r = _sweep(root, "C", "T01,HRWR-ID,HRWR-ID.id", 2, "--save-messages", "--resume")
    check("an episode a server or API error ended is run again on resume",
          "resuming: 5 finished, 1 to run (1 of them ended by a server or API error)"
          in r.stdout and json.loads((cell / "HRWR-ID.id__r1.jsonl").read_text()
                                     .splitlines()[-1])["stop_reason"] != "backend_error",
          r.stdout[-400:])
    import subprocess
    dead = subprocess.run(
        [sys.executable, "-m", "pasarbench.sweep", "--backend", "openai", "--model", "m",
         "--base-url", "http://127.0.0.1:9/v1", "--simulator", "silent", "--suite", "all",
         "--tasks", "T01", "-k", "1", "--strategies", "full", "--trace-root", str(root),
         "--run-id", "dead"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent)
    check("a run the server never answered exits non-zero and says what to do",
          dead.returncode == 3 and "server or API error" in dead.stdout, dead.stdout[-400:])
    from pasarbench.sweep import _server_errors
    few = _server_errors([{"stop_reasons": {"done": 250, "backend_error": 2}}])
    check("…and a few such episodes are reported too, without stopping a pipeline: two of "
          "252 once went unsaid and were counted as the agent's failures (RUNBOOK 1l)",
          few[0] is not None and "2 of 252 episodes" in few[0] and few[1] is False
          and _server_errors([{"stop_reasons": {"done": 245, "backend_error": 5}}])[1] is True
          and _server_errors([{"stop_reasons": {"done": 252}}]) == (None, False), str(few))

    head = json.loads((cell / "T01__r0.jsonl").read_text().splitlines()[0])
    lines = (cell / "T01__r0.jsonl").read_text().splitlines()
    lines[0] = json.dumps({**head, "simulator": "llm-user:someone-else"})
    (cell / "T01__r0.jsonl").write_text("\n".join(lines) + "\n")
    r = _sweep(root, "C", "T01,HRWR-ID,HRWR-ID.id", 2, "--resume")
    check("resume refuses to mix episodes from a different customer",
          r.returncode != 0 and "different setup" in (r.stdout + r.stderr))

    from pasarbench.rl.split import make_folds
    from pasarbench.sweep import fold_router
    fp = root / "folds.json"
    folds = make_folds(ALL)
    fp.write_text(json.dumps(folds))
    route = fold_router(SimpleNamespace(model="base", folds=str(fp),
                                        fold_models="A=rft-A,B=rft-B"))
    wrong = [t.task_id for t in ALL if route(t) == f"rft-{folds['task_fold'][t.task_id]}"]
    check("every task is answered by the model that did not train on it", not wrong, wrong[:5])
    check("without fold models, everything goes to --model",
          fold_router(SimpleNamespace(model="base", folds="", fold_models=""))(ALL[0]) == "base")

    # The closing check's note is the harness's, not a turn the model took:
    # a run made with it is no training data. Refused before any tokenizer.
    from pasarbench.rl.sft import build
    r = _sweep(root, "CL", "T01", 1, "--save-messages", "--closing", "check")
    try:
        build(root / "CL" / "full", BY_ID, folds, None)
        ok, why = False, "built"
    except SystemExit as e:
        ok, why = "Collect without --closing" in str(e), str(e)
    except Exception as e:      # went on past the check that should have stopped it
        ok, why = False, f"{type(e).__name__}: {e}"
    check("a run made with the closing check is refused as training data, with the fix",
          r.returncode == 0 and ok, r.stderr[-300:] or why)


# --------------------------------------------------------------------------

def _collection(root: Path, tasks: str, k: int = 2, thinking_off: bool = True) -> Path:
    r = _sweep(root, "C", tasks, k, "--save-messages")
    assert r.returncode == 0, r.stderr[-800:]
    cell = root / "C" / "full"
    for p in cell.glob("*.jsonl"):
        lines = p.read_text().splitlines()
        head = json.loads(lines[0])
        # The scripted backend sends no request flags; a Qwen collection sends
        # enable_thinking=false, which is the case the renderer has to get right.
        head["requested_model"] = "toy"
        if thinking_off:
            head["agent_extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        lines[0] = json.dumps(head)
        p.write_text("\n".join(lines) + "\n")
    return cell


def test_examples():
    print("\n=== training examples: the prompt the model saw, loss on its turn ===")
    import _tokenizer
    if not _tokenizer.available():
        for label in ("per-turn examples", "template drift", "dedupe"):
            skip(label, "transformers/tokenizers not installed")
        return
    from pasarbench.rl.sft import TemplateDrift, build, render_turn
    from pasarbench.rl.split import make_folds

    tok = _tokenizer.local_tokenizer()
    root = Path(tempfile.mkdtemp())
    folds = make_folds(ALL)
    a = next(t.task_id for t in ALL if folds["task_fold"][t.task_id] == "A" and "." in t.task_id)
    b = next(t.task_id for t in ALL if folds["task_fold"][t.task_id] == "B" and "." in t.task_id)
    cell = _collection(root, f"{a},{b},T01", k=2)
    files, stats = build(cell, BY_ID, folds, tok, max_per_task=3, expect_model="toy")

    ta = {x["task_id"] for x in files["A"]}
    tb = {x["task_id"] for x in files["B"]}
    check("each fold's file holds only that fold's tasks",
          ta <= {t for t, f in folds["task_fold"].items() if f == "A"}
          and tb <= {t for t, f in folds["task_fold"].items() if f == "B"}
          and a in ta and b in tb, (ta, tb))
    exs = files["A"] + files["B"]
    check("one example per agent turn",
          all(x["turn"] == i for task in ta | tb
              for i, x in enumerate(sorted((e for e in exs if e["task_id"] == task
                                            and e["trace"].endswith("__r0.jsonl")),
                                           key=lambda e: e["turn"]))))
    check("every prompt ends with the generation prompt the server sent",
          all(x["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
              for x in exs))
    check("the trained text is the turn: it ends at end-of-turn and holds no tool output",
          all(x["completion"].endswith("<|im_end|>") and "<tool_response>" not in x["completion"]
              and "<|im_start|>" not in x["completion"] for x in exs))
    check("tool calls are trained in the model's own format",
          any('<tool_call>\n{"name": ' in x["completion"] for x in exs))
    bal_files, bal = build(cell, BY_ID, folds, tok, max_per_task=3, expect_model="toy",
                           per_trap=1)
    check("balanced by trap (round 2): at most the quota from each trap and fold, the "
          "selection recorded, the same turns from the episodes it keeps",
          bal["selection"] == {"per_trap": 1, "max_per_task": 3}
          and all(n <= 1 for by in bal["episodes_by_trap"].values() for n in by.values())
          and sum(bal["episodes_used"].values()) == sum(
              n for by in bal["episodes_by_trap"].values() for n in by.values())
          and {x["task_id"] for x in bal_files["A"]} <= ta
          and stats["selection"] == {"max_per_task": 3},
          (bal["selection"], bal["episodes_by_trap"]))

    # Which verdict picks the episodes (#30). Task a's episodes all pass; say
    # otherwise in their footers, as a stale checker would have.
    for f in sorted(cell.glob(f"{a}__*.jsonl")):
        lines = f.read_text().splitlines()
        i = max(n for n, l in enumerate(lines) if json.loads(l).get("type") == "footer")
        foot = json.loads(lines[i])
        foot["passed"], foot["failures"] = False, ["a check since corrected"]
        lines[i] = json.dumps(foot)
        f.write_text("\n".join(lines) + "\n")
    now_files, now = build(cell, BY_ID, folds, tok, max_per_task=3, expect_model="toy")
    rec_files, rec = build(cell, BY_ID, folds, tok, max_per_task=3, expect_model="toy",
                           checker="recorded")
    check("today's checks pick the training episodes, not the footers' stale verdicts",
          a in {x["task_id"] for x in now_files["A"]}
          and a not in {x["task_id"] for x in rec_files["A"]}
          and now["checker"] == "current" and now["verdicts_changed"] == 2
          and rec["verdicts_changed"] == 0 and now["pass_rate"] > rec["pass_rate"],
          (now["verdicts_changed"], now["pass_rate"], rec["pass_rate"]))

    # Why per turn: rendered as one conversation, only the LAST assistant turn
    # carries the empty think block every turn was generated after.
    from pasarbench.rl.sft import api_messages, read_trace, _for_template
    ep = read_trace(next(cell.glob(f"{a}__r0.jsonl")))
    msgs = api_messages(ep["messages"]["messages"])
    whole = tok.apply_chat_template(_for_template(msgs), tokenize=False, enable_thinking=False)
    first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    p, c = render_turn(tok, msgs, first, None, {"enable_thinking": False})
    check("a whole-conversation render would have trained turn 1 on a prompt it never had",
          (p + c) not in whole and c in whole)

    import copy
    tok2 = copy.deepcopy(tok)
    tok2.eos_token = "<|endoftext|>"         # some checkpoints close turns with another token
    p2, c2 = render_turn(tok2, msgs, first, None, {"enable_thinking": False})
    check("a turn still ends at <|im_end|> when the tokenizer's eos is something else",
          c2 == c and c2.endswith("<|im_end|>"), c2[-40:])

    class Drifty:
        eos_token = "<|im_end|>"

        def apply_chat_template(self, m, add_generation_prompt=False, **kw):
            text = "".join(f"<{x['role']}>{x.get('content') or ''}</{x['role']}>" for x in m)
            return text + ("<gen>" if add_generation_prompt else "<|im_end|>")
    try:
        render_turn(Drifty(), msgs, first, None, {})
        ok = False
    except TemplateDrift:
        ok = True
    check("a template whose prompt is not a prefix of the turn fails loudly", ok)

    check("identical passing episodes are trained on once",
          stats["skipped"].get("duplicate episode", 0) >= 1, stats["skipped"])
    check("per-fold pass rates and zero-signal traps are reported",
          set(stats["per_trap_pass_rate"]) == {"A", "B"}
          and stats["traps_with_zero_signal"] == {"A": [], "B": []}, stats)

    for p in cell.glob("*.jsonl"):
        lines = p.read_text().splitlines()
        p.write_text("\n".join(l for l in lines if '"type": "messages"' not in l) + "\n")
    try:
        build(cell, BY_ID, folds, tok)
        ok = False
    except SystemExit as e:
        ok = "--save-messages" in str(e)
    check("a run collected without --save-messages is refused, with the fix", ok)


def _sglang_tools(tools):
    """What SGLang hands the chat template for a tool list: each tool through its
    pydantic model, `strict` and `defer_loading` included, description first."""
    out = []
    for t in tools or []:
        f = t["function"]
        out.append({"type": t.get("type", "function"),
                    "function": {"description": f.get("description"), "name": f["name"],
                                 "parameters": f.get("parameters"), "strict": False},
                    "defer_loading": None})
    return out


def test_balance():
    print("\n=== round 2's training data: balanced by trap, not by task ===")
    from pasarbench.rl.sft import balance

    # One trap's distinct passing episodes per task: a task the model always
    # solves has eight, one it rarely solves has one, one it never solves none.
    cands = {"T-easy": [f"e{i}" for i in range(8)], "T-mid": ["m0", "m1", "m2"],
             "T-rare": ["r0"], "T-never": []}
    chosen, left = balance(cands, per_trap=6, max_per_task=8)
    check("round robin over the trap's tasks: each task's first episode, then each one's "
          "second, until the trap has its quota",
          chosen == {"T-easy": ["e0", "e1", "e2"], "T-mid": ["m0", "m1"], "T-rare": ["r0"],
                     "T-never": []} and left == 12 - 6, (chosen, left))
    chosen, left = balance(cands, per_trap=20, max_per_task=2)
    check("…no task gives more than its cap, and a trap short of passing episodes gives "
          "what it has", chosen == {"T-easy": ["e0", "e1"], "T-mid": ["m0", "m1"],
                                    "T-rare": ["r0"], "T-never": []} and left == 12 - 5,
          (chosen, left))
    chosen, left = balance({"T": ["a", "b"]}, per_trap=0, max_per_task=3)
    check("…and a quota of nothing takes nothing", chosen == {"T": []} and left == 2,
          (chosen, left))


def test_check_command():
    print("\n=== check: what gets loss, and is it the prompt the server builds ===")
    import _tokenizer
    if not _tokenizer.available():
        skip("check command", "transformers/tokenizers not installed")
        return
    import contextlib
    import io
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from pasarbench.rl.sft import _for_template, server_renderer
    from pasarbench.rl.split import make_folds

    tok = _tokenizer.local_tokenizer()
    work = Path(tempfile.mkdtemp())
    tok.save_pretrained(work / "tok")
    folds = make_folds(ALL)
    (work / "folds.json").write_text(json.dumps(folds))
    a = next(t.task_id for t in ALL if folds["task_fold"][t.task_id] == "A")
    cell = _collection(work, a, k=1)
    tr = _script("train_rft")

    def run(argv):
        sys.argv = ["train_rft.py", *argv]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                tr.main()
            return 0, out.getvalue()
        except SystemExit as e:
            return (e.code if isinstance(e.code, int) else 1), out.getvalue() + str(e.code)

    # build loads the tokenizer from --model and checks the collection came
    # from that same model, so the headers name the saved tokenizer's path
    for pth in cell.glob("*.jsonl"):
        lines = pth.read_text().splitlines()
        h = json.loads(lines[0])
        h["requested_model"] = str(work / "tok")
        lines[0] = json.dumps(h)
        pth.write_text("\n".join(lines) + "\n")
    code, out = run(["build", "--run", str(cell), "--folds", str(work / "folds.json"),
                     "--model", str(work / "tok"), "--out-dir", str(work / "rft")])
    check("build writes one file per fold and the stats beside them",
          code == 0 and (work / "rft" / "train-A.jsonl").exists()
          and (work / "rft" / "stats.json").exists(), out[-600:])
    code, out = run(["check", "--data", str(work / "rft" / "train-A.jsonl"),
                     "--model", str(work / "tok")])
    check("check prints the trained text and passes a clean file",
          code == 0 and "=== COMPLETION (loss)" in out and "no problems found" in out, out[-600:])

    class SGLang(BaseHTTPRequestHandler):
        """/tokenize the way SGLang 0.5.20 does it: the chat request's own message
        processing, the generation prompt always, tools as pydantic dumps."""
        extra: list = []
        plain_tools = False          # True: render the tools as sent (a server like the local template)
        drop_kwargs = False          # True: ignore chat_template_kwargs (thinking stays on)
        auth: set = set()

        def do_POST(self):
            SGLang.auth.add(self.headers.get("Authorization"))
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msgs = _for_template(body["messages"])
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1] = {"role": "user", "content": msgs[-1].get("content") or ""}
            tools = body.get("tools") if SGLang.plain_tools else _sglang_tools(body.get("tools"))
            kw = {} if SGLang.drop_kwargs else (body.get("chat_template_kwargs") or {})
            text = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True,
                                           tokenize=False, **kw)
            ids = tok(text, add_special_tokens=False)["input_ids"] + SGLang.extra
            data = json.dumps({"tokens": ids, "count": len(ids), "max_model_len": 32768}).encode()
            self.send_response(200 if self.path == "/tokenize" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), SGLang)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1"

    code, out = run(["check", "--data", str(work / "rft" / "train-A.jsonl"),
                     "--model", str(work / "tok"), "--server", url])
    check("a locally built file fails against SGLang: its tool block is not the served one",
          code != 0 and "MISMATCH" in out and '{"descript' in out
          and "rebuild with `build --server" in out, out[-700:])

    code, out = run(["build", "--run", str(cell), "--folds", str(work / "folds.json"),
                     "--model", str(work / "tok"), "--out-dir", str(work / "srv"),
                     "--server", url])
    rows = [json.loads(l) for l in (work / "srv" / "train-A.jsonl").read_text().splitlines()]
    stats = json.loads((work / "srv" / "stats.json").read_text())
    check("build --server trains on the server's prompt ids, and says how many differ locally",
          code == 0 and rows and all(r["prompt_ids"] != tok(r["prompt"], add_special_tokens=False)["input_ids"]
                                     and r["prompt_matches_local"] is False for r in rows)
          and stats["prompts_unlike_local_render"] == len(rows)
          and stats["rendered_by"] == "server /tokenize", out[-400:])
    check("…the turn is tokenized locally, on its own",
          all(r["completion_ids"] == tok(r["completion"], add_special_tokens=False)["input_ids"]
              and r["completion_ids"][-1] == tok.eos_token_id for r in rows))
    from pasarbench.rl.sft import tokenize_example
    ex = tokenize_example(tok, {**rows[0], "prompt": "IGNORED"})
    check("…and training uses those ids, not a local re-render",
          ex["input_ids"][:len(rows[0]["prompt_ids"])] == rows[0]["prompt_ids"]
          and ex["labels"][:len(rows[0]["prompt_ids"])] == [-100] * len(rows[0]["prompt_ids"]))
    check("the server is sent the placeholder key, never a real one",
          SGLang.auth == {"Bearer EMPTY"}, SGLang.auth)

    code, out = run(["check", "--data", str(work / "srv" / "train-A.jsonl"),
                     "--model", str(work / "tok"), "--server", url])
    check("check passes the server-built file and shows where the local render parts",
          code == 0 and "train on the server's own prompt token ids" in out
          and "not what the local template renders" in out
          and "token-identical to the ids stored in this file" in out, out[-700:])

    SGLang.extra = [tok.eos_token_id]        # the server renders differently since the build
    code, out = run(["check", "--data", str(work / "srv" / "train-A.jsonl"),
                     "--model", str(work / "tok"), "--server", url])
    check("a server that no longer renders the stored prompts stops training",
          code != 0 and "the server now renders" in out, out[-500:])
    SGLang.extra = []

    SGLang.drop_kwargs = True                # a server that ignores chat_template_kwargs
    code, out = run(["build", "--run", str(cell), "--folds", str(work / "folds.json"),
                     "--model", str(work / "tok"), "--out-dir", str(work / "srv4"),
                     "--server", url])
    code, out = run(["check", "--data", str(work / "srv4" / "train-A.jsonl"),
                     "--model", str(work / "tok")])
    check("a server prompt that ends without the empty think block the turn follows stops training",
          code != 0 and "do not end in the generation prompt" in out, out[-600:])
    SGLang.drop_kwargs = False

    SGLang.plain_tools = True                # a server that renders exactly like the local template
    code, out = run(["check", "--data", str(work / "rft" / "train-A.jsonl"),
                     "--model", str(work / "tok"), "--server", url])
    check("where the server does render like the local template, a local build passes",
          code == 0 and "token-identical to the local rendering" in out, out[-500:])
    SGLang.plain_tools = False

    render = server_renderer(url, "m")
    try:
        render([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}], None, {})
        ok = False
    except ValueError as e:
        ok = "user" in str(e)
    check("a prompt ending in an assistant message is refused before SGLang turns it into a user one", ok)
    srv.shutdown()

    class NoTools(SGLang):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "unexpected field tools"}' if body.get("tools") else b"{}")
    srv = HTTPServer(("127.0.0.1", 0), NoTools)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    code, out = run(["build", "--run", str(cell), "--folds", str(work / "folds.json"),
                     "--model", str(work / "tok"), "--out-dir", str(work / "srv3"),
                     "--server", f"http://127.0.0.1:{srv.server_port}"])
    st3 = json.loads((work / "srv3" / "stats.json").read_text()) if code == 0 else {}
    check("a server that cannot tokenize tools falls back to local rendering, loudly",
          code == 0 and "rendering LOCALLY" in out and st3.get("rendered_by") == "local tokenizer",
          out[-400:])
    srv.shutdown()

    rows = (work / "rft" / "train-A.jsonl").read_text().splitlines()
    bad = json.loads(rows[0])
    bad["completion"] = "<tool_response>\n{}\n</tool_response>" + bad["completion"]
    (work / "rft" / "train-A.jsonl").write_text("\n".join([json.dumps(bad), *rows[1:]]) + "\n")
    code, out = run(["check", "--data", str(work / "rft" / "train-A.jsonl"),
                     "--model", str(work / "tok")])
    check("tool output in the trained text stops training", code != 0
          and "a tool result is in the trained text" in out, out[-400:])


def test_wrappers():
    print("\n=== the sweep steps run the commands they print ===")
    import contextlib
    import io
    tr = _script("train_rft")

    def cmd(argv):
        sys.argv = ["train_rft.py", *argv, "--dry-run"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tr.main()
        return out.getvalue()

    base, coll, ev = cmd(["baseline"]), cmd(["collect"]), cmd(["eval"])
    check("baseline: all tasks, k=5, T=0, resumable",
          "--suite all -k 5 --temperature 0.0" in base and "--resume" in base
          and "--save-messages" not in base, base)
    check("collect: k=8 at T=1 with conversations saved",
          "-k 8 --temperature 1.0" in coll and "--save-messages" in coll, coll)
    check("eval: each adapter on the fold it did not train on, named <base>:<adapter>",
          "--fold-models A=Qwen/Qwen3.8-27B:pasar-rft-A,B=Qwen/Qwen3.8-27B:pasar-rft-B" in ev
          and "--folds" in ev and "--temperature 0.0" in ev, ev)
    check("thinking is off in every step, by the same flag",
          all('"enable_thinking": false' in c for c in (base, coll, ev)))
    want = os.environ.get("PASAR_CUSTOMER") or "2"
    check("every step asks for the same simulated customer, the one runs since v31 have "
          "unless PASAR_CUSTOMER says otherwise (WHAT_FAILED #38)",
          all(f"--customer {want}" in c for c in (base, coll, ev)), base)
    e2 = cmd(["baseline", "--suite", "edition2", "--run-id", "P2-base-e2"])
    check("round 2's base run on the 62 newer tasks of the five hardest traps: the same "
          "sweep under a run id of its own",
          "--suite edition2 -k 5 --temperature 0.0" in e2 and "--run-id P2-base-e2" in e2, e2)
    ev2 = cmd(["eval", "--adapter-prefix", "pasar-rft2-", "--run-id", "P2-rft"])
    check("…and its fine-tunes evaluated under their own adapter names and run id",
          "B=Qwen/Qwen3.8-27B:pasar-rft2-B" in ev2 and "--run-id P2-rft" in ev2, ev2)
    helped = io.StringIO()
    with contextlib.redirect_stdout(helped):
        try:
            sys.argv = ["train_rft.py", "build", "--help"]
            tr.main()
        except SystemExit:
            pass
    check("the build balances by trap on request (WHAT_FAILED #31)",
          "--per-trap" in helped.getvalue(), helped.getvalue()[-400:])
    sim = os.environ.get("SIM_MODEL") or tr.SIM_DEFAULT
    check("the customer is Gemma 4 31B, served locally",
          tr.SIM_DEFAULT == "RedHatAI/gemma-4-31B-it-FP8-Dynamic"
          and all(f"--sim-model {sim}" in c and "localhost:8001" in c
                  for c in (base, coll, ev)), base)


def test_train_smoke():
    print("\n=== two LoRA steps on a toy model, end to end ===")
    import _tokenizer
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except ImportError as e:
        skip("LoRA training runs", f"{e.name} not installed")
        return
    import contextlib
    import io

    tok = _tokenizer.local_tokenizer()
    work = Path(tempfile.mkdtemp())
    model_dir = work / "toy"
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=4096,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    with contextlib.redirect_stderr(io.StringIO()):
        Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    data = work / "train-A.jsonl"
    rows = [{"task_id": "T01", "fold": "A", "turn": 0, "tokens": 0,
             "prompt": "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
             "completion": "It is O1003<|im_end|>"}] * 4
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))

    tr = _script("train_rft")
    sys.argv = ["train_rft.py", "train", "--data", str(data), "--model", str(model_dir),
                "--out", str(work / "rft-A"), "--max-steps", "2", "--grad-accum", "1",
                "--fp32", "--attn", "eager", "--save-steps", "1000"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            tr.main()
        ok, err = True, ""
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    cfg_path = work / "rft-A" / "adapter_config.json"
    check("training runs and saves an adapter",
          ok and cfg_path.exists(), err or out.getvalue()[-600:])
    # SGLang reads the target modules from this file. PEFT writes "all-linear"
    # out as full module paths, and SGLang keys on their last component: every
    # linear layer of the attention and the MLP, and not lm_head.
    meta = json.loads((work / "rft-A" / "pasarbench_meta.json").read_text()) if ok else {}
    check("…and records how many turns each optimizer step saw (the recipe, on any "
          "number of GPUs)", meta.get("turns_per_step") == 1 and meta.get("processes") == 1,
          meta)
    targets = set(json.loads(cfg_path.read_text()).get("target_modules") or []) if ok else set()
    check("…whose target modules are the projections SGLang's LoRA supports",
          {t.split(".")[-1] for t in targets} == {"q_proj", "k_proj", "v_proj", "o_proj",
                                                  "gate_proj", "up_proj", "down_proj"},
          sorted(targets)[:8])

    from pasarbench.rl.sft import tokenize_example
    ex = tokenize_example(tok, rows[0])
    p = tok(rows[0]["prompt"], add_special_tokens=False)["input_ids"]
    trained = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    check("loss is on the completion only, token for token",
          ex["labels"][:len(p)] == [-100] * len(p) and trained == rows[0]["completion"],
          trained)


def _retag(src: Path, dst: Path) -> None:
    """A scripted or mute episode as if the toy model had answered it, thinking
    off -- the headers _collection gives a collection."""
    lines = src.read_text().splitlines()
    head = json.loads(lines[0])
    head["requested_model"] = "toy"
    head["agent_extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    dst.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")


def _grpo_cell(root: Path) -> tuple[Path, str, str]:
    """A collection with one group to learn from and one without: T01 passed
    twice (scripted) and failed twice (mute); HRWR-ID passed twice, the second
    time in three more steps."""
    X, Y = "T01", "HRWR-ID"
    cell = _collection(root, f"{X},{Y}", k=2)
    r = _sweep(root, "M", X, 2, "--save-messages", "--backend", "mute")
    assert r.returncode == 0, r.stderr[-800:]
    for i, f in enumerate(sorted((root / "M" / "full").glob("*.jsonl"))):
        _retag(f, cell / f"{X}__r{2 + i}.jsonl")
    f = cell / f"{Y}__r1.jsonl"
    lines = f.read_text().splitlines()
    foot = json.loads(lines[-1])
    foot["budget"]["steps"] += 3
    f.write_text("\n".join([*lines[:-1], json.dumps(foot)]) + "\n")
    # As train_grpo.py collect samples them: the full softmax at temperature 1.
    from pasarbench.rl.grpo import ON_POLICY
    for f in cell.glob("*.jsonl"):
        lines = f.read_text().splitlines()
        head = json.loads(lines[0])
        head["agent_extra_body"] = {**head.get("agent_extra_body", {}), **ON_POLICY}
        head["agent_temperature"] = 1.0
        f.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    return cell, X, Y


def test_grpo_build():
    print("\n=== GRPO: groups, advantages, the examples of a turn ===")
    import _tokenizer
    if not _tokenizer.available():
        skip("GRPO examples", "transformers/tokenizers not installed")
        return
    from collections import defaultdict

    from pasarbench.rl.grpo import build, episode_reward
    from pasarbench.rl.reward import group_advantages
    from pasarbench.rl.sft import read_trace, tokenize_example
    from pasarbench.sweep import ALL_SOLUTIONS

    tok = _tokenizer.local_tokenizer()
    root = Path(tempfile.mkdtemp())
    cell, X, Y = _grpo_cell(root)
    rows, stats = build(cell, BY_ID, tok, ALL_SOLUTIONS, expect_model="toy")
    by = defaultdict(list)
    for r in rows:
        by[Path(r["trace"]).name].append(r)
    adv = {t: {r["advantage"] for r in rs} for t, rs in by.items()}
    check("a group with a pass and a fail is learned from: every turn of a passing rollout "
          "has its episode's positive advantage, every turn of a failing one its negative",
          set(by) == {f"{X}__r{i}.jsonl" for i in range(4)}
          and all(len(a) == 1 for a in adv.values())
          and all(r["advantage"] > 0 for t in (f"{X}__r0.jsonl", f"{X}__r1.jsonl") for r in by[t])
          and all(r["advantage"] < 0 and not r["passed"]
                  for t in (f"{X}__r2.jsonl", f"{X}__r3.jsonl") for r in by[t]), adv)
    ys = [read_trace(f) for f in sorted(cell.glob(f"{Y}__*.jsonl"))]
    rewards = [episode_reward([e["header"], *e["steps"], e["footer"]], BY_ID[Y],
                              len(ALL_SOLUTIONS[Y])).reward for e in ys]
    check("a group that all passed teaches nothing, though its rollouts' lengths differ: "
          "standardised, the step penalty alone would have given it advantages as large as "
          "a pass against a fail (WHAT_FAILED #41)",
          not any(r["task_id"] == Y for r in rows) and len(set(rewards)) == 2
          and all(abs(a) == 1.0 for a in group_advantages(rewards))
          and stats["degenerate_groups"] == 1 and stats["bands"] == {"always": 1, "sometimes": 1}
          and stats["learning_tasks"] == [X], (rewards, stats["bands"]))
    trained = {t: sum(sum(l != -100 for l in tokenize_example(tok, r)["labels"]) for r in rs)
               for t, rs in by.items()}
    check("a turn's loss is divided by its rollout's trained tokens, all its turns together: "
          "a long rollout weighs what a short one does",
          all(r["norm"] == trained[t] for t, rs in by.items() for r in rs)
          and len(by[f"{X}__r0.jsonl"]) > 1, trained)
    check("the stats give the policy's pass rate on its tasks: one point of the learning "
          "curve, iteration by iteration", stats["passed"] == 4 and stats["pass_rate"] == 0.6667
          and stats["episodes"] == 6, stats)
    try:
        build(cell, BY_ID, tok, ALL_SOLUTIONS, expect_model="Qwen/Qwen3.8-27B:pasar-rft2-A")
        why = "built"
    except SystemExit as e:
        why = str(e)
    check("rollouts of another policy are refused: GRPO trains the one that sampled them",
          "GRPO trains the policy that sampled" in why, why)
    g0 = cell / f"{X}__r0.jsonl"
    keep = g0.read_text()
    lines = keep.splitlines()
    head = json.loads(lines[0])
    head["agent_extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    g0.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    try:
        build(cell, BY_ID, tok, ALL_SOLUTIONS, expect_model="toy")
        why = "built"
    except SystemExit as e:
        why = str(e)
    g0.write_text(keep)
    check("…and so are rollouts sampled with the server's own top-k and top-p: the loss is "
          "over the full softmax at temperature 1 (WHAT_FAILED #42)",
          "not the full softmax" in why and "train_grpo.py collect" in why, why)
    f = cell / f"{X}__r3.jsonl"
    f.write_text("\n".join(f.read_text().splitlines()[:-1]) + "\n")
    _, cut = build(cell, BY_ID, tok, ALL_SOLUTIONS, expect_model="toy")
    check("a rollout that never finished is left out, as every report leaves it out",
          cut["skipped"].get("unfinished") == 1 and cut["episodes"] == 6, cut["skipped"])
    lines = f.read_text().splitlines()
    head = json.loads(lines[0])
    f.write_text("\n".join([json.dumps({**head, "closing": "check"}), *lines[1:]]) + "\n")
    try:
        build(cell, BY_ID, tok, ALL_SOLUTIONS, expect_model="toy")
        why = "built"
    except SystemExit as e:
        why = str(e)
    check("…and a run with the closing check is refused: its note is the harness's turn",
          "plain full-context episodes" in why, why)
    f.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    g = cell / f"{X}__r2.jsonl"
    lines = g.read_text().splitlines()
    g.write_text("\n".join([*lines[:-1], json.dumps({**json.loads(lines[-1]),
                                                      "stop_reason": "max_wall_seconds"})]) + "\n")
    _, wall = build(cell, BY_ID, tok, ALL_SOLUTIONS, expect_model="toy")
    check("a rollout the harness stopped -- the wall clock, a server error -- is no signal "
          "about the policy and leaves its group: here the last failure, so nothing is left "
          "to learn", wall["skipped"].get("stopped by the harness (max_wall_seconds)") == 1
          and wall["learning_tasks"] == [] and wall["examples"] == 0, wall["skipped"])

    # The command: the stats written last, no fallback to the local template.
    import contextlib
    import io
    g_ = _script("train_grpo")
    tdir = root / "tok"
    tok.save_pretrained(tdir)
    cell2, _, _ = _grpo_cell(root / "again")
    out = root / "grpo2"

    def cli(*extra):
        o = io.StringIO()
        try:
            with contextlib.redirect_stdout(o):
                g_.main(["build", "--run", str(cell2), "--model", str(tdir), "--served-model",
                         "toy", "--out", str(out / "examples" / "i1-A.jsonl"),
                         "--stats", str(out / "stats" / "i1-A.stats.json"), *extra])
            return o.getvalue()
        except SystemExit as e:
            return f"exit: {e}"
    said = cli()
    check("`train_grpo.py build` writes the examples, then their stats, and says how the "
          "policy did", (out / "examples" / "i1-A.jsonl").exists()
          and json.loads((out / "stats" / "i1-A.stats.json").read_text())["learning_tasks"] == [X]
          and "the policy passed 4 (67%)" in said
          and not list(out.rglob("*.tmp")), said)
    down = cli("--server", "http://127.0.0.1:9")
    check("…and with the server down it stops, rather than train on prompts the local "
          "template rendered (#27)", down.startswith("exit:") and "server built" in down, down)


def _toy_model(work: Path, tok) -> Path:
    import contextlib
    import io

    from transformers import Qwen3Config, Qwen3ForCausalLM
    model_dir = work / "toy"
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=4096,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    with contextlib.redirect_stderr(io.StringIO()):
        Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    return model_dir


def test_grpo_train():
    print("\n=== GRPO's update, on a toy model ===")
    import _tokenizer
    try:
        import peft  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        skip("GRPO's update", f"{e.name} not installed")
        return
    import contextlib
    import io

    from pasarbench.rl.grpo import grpo_loss
    from pasarbench.rl.sft import tokenize_example

    tok = _tokenizer.local_tokenizer()
    work = Path(tempfile.mkdtemp())
    model_dir = _toy_model(work, tok)
    prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    good = {"prompt": prompt, "completion": "It is O1003<|im_end|>"}
    bad = {"prompt": prompt, "completion": "Could you give me the order number?<|im_end|>"}

    def batch(ex):
        t = tokenize_example(tok, ex)
        return torch.tensor([t["input_ids"]]), torch.tensor([t["labels"]])

    torch.manual_seed(0)
    with contextlib.redirect_stderr(io.StringIO()):
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)

    def logp(ex):
        ids, labels = batch(ex)
        with torch.no_grad():
            return grpo_loss(model(input_ids=ids).logits, labels, 0.0, 1.0)[1].sum().item()

    before = logp(good), logp(bad)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(5):
        opt.zero_grad()
        loss = 0
        for ex, a in ((good, 1.0), (bad, -1.0)):
            ids, labels = batch(ex)
            loss = loss + grpo_loss(model(input_ids=ids).logits, labels, a,
                                    float((labels != -100).sum()))[0]
        loss.backward()
        opt.step()
    after = logp(good), logp(bad)
    check("a positive advantage makes its turn more likely, a negative one less",
          after[0] > before[0] and after[1] < before[1], (before, after))

    def grad(a):
        ids, labels = batch(good)
        model.zero_grad()
        out = model(input_ids=ids).logits
        now = grpo_loss(out, labels, 0.0, 1.0)[1].detach()
        grpo_loss(out, labels, a, 10.0, old=now - 1.0)[0].backward()
        return sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    check("the clip: a turn already far likelier than when it was sampled gets no further "
          "push from a positive advantage, and is still pulled back by a negative one",
          grad(1.0) == 0.0 and grad(-1.0) > 0.0, (grad(1.0), grad(-1.0)))

    # End to end: an adapter, then GRPO from it with train_rft.py, as an iteration runs.
    tr = _script("train_rft")
    sft = work / "sft.jsonl"
    sft.write_text("".join(json.dumps({"task_id": "T01", "fold": "A", "turn": 0, "tokens": 0,
                                       **good}) + "\n" for _ in range(2)))
    rows = [{"task_id": "T01", "fold": "grpo", "turn": 0, "tokens": 0, "advantage": a,
             "norm": 4, "trace": f"T01__r{i}.jsonl", **ex}
            for i, (ex, a) in enumerate(((good, 1.0), (bad, -1.0)) * 2)]
    data = work / "i1-A.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def train(*argv):
        sys.argv = ["train_rft.py", "train", "--model", str(model_dir), "--grad-accum", "1",
                    "--fp32", "--attn", "eager", "--save-steps", "1000", *argv]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                tr.main()
            return ""
        except SystemExit as e:
            return f"exit: {e}"
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"
    err = train("--data", str(sft), "--out", str(work / "rft2-A"), "--max-steps", "1")
    err = err or train("--data", str(data), "--out", str(work / "grpo2-i1-A"), "--max-steps", "2",
                       "--objective", "grpo", "--init", str(work / "rft2-A"), "--lr", "1e-3")
    meta = (json.loads((work / "grpo2-i1-A" / "pasarbench_meta.json").read_text())
            if not err else {})
    check("an iteration's update runs from the adapter it starts from, and the adapter "
          "records what it is", not err and (work / "grpo2-i1-A" / "adapter_config.json").exists()
          and meta.get("objective") == "grpo" and meta.get("init") == str(work / "rft2-A")
          and meta.get("mean_abs_advantage") == 1.0, err or meta)
    from safetensors.torch import load_file
    w0 = load_file(str(work / "rft2-A" / "adapter_model.safetensors")) if not err else {}
    w1 = load_file(str(work / "grpo2-i1-A" / "adapter_model.safetensors")) if not err else {}
    check("…and it moved the adapter it started from, with the same modules",
          bool(w0) and set(w0) == set(w1) and any(not torch.equal(w0[k], w1[k]) for k in w0),
          sorted(set(w0) ^ set(w1))[:4])
    refused = train("--data", str(sft), "--out", str(work / "x"), "--max-steps", "1",
                    "--objective", "grpo")
    check("examples without advantages are refused, with where they come from",
          "train_grpo.py build" in refused, refused)
    src = (Path(__file__).resolve().parent.parent / "scripts" / "train_rft.py").read_text()
    check("GRPO's turn losses are averaged over an optimizer step's turns, and neither "
          "dropout nor a precision the training pass does not use puts noise in the ratio",
          "self.model_accepts_loss_kwargs = False" in src and "m.p = 0.0" in src
          and '"constant_with_warmup" if grpo' in src
          and 'torch.autocast("cuda", dtype=torch.bfloat16) if bf16' in src)


def test_served_models():
    print("\n=== evaluation asks the server for the fine-tune, and gets it ===")
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from pasarbench.sweep import check_served

    def serve(cards):
        class Models(BaseHTTPRequestHandler):
            def do_GET(self):
                data = json.dumps({"object": "list", "data": cards}).encode()
                self.send_response(200 if self.path == "/v1/models" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        srv = HTTPServer(("127.0.0.1", 0), Models)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{srv.server_port}/v1"

    def refused(url, names, why):
        try:
            check_served(url, names)
            return False
        except SystemExit as e:
            return why in str(e)

    # What SGLang 0.5.20's /v1/models returns with both adapters loaded.
    base = {"id": "Qwen/Qwen3-8B", "object": "model", "owned_by": "sglang",
            "root": "Qwen/Qwen3-8B", "parent": None, "max_model_len": 32768}
    lora = [{"id": f"pasar-rft-{f}", "object": "model", "owned_by": "sglang",
             "root": f"checkpoints/rft-{f}", "parent": "Qwen/Qwen3-8B"} for f in "AB"]
    srv, url = serve([base, *lora])
    good = ["Qwen/Qwen3-8B:pasar-rft-A", "Qwen/Qwen3-8B:pasar-rft-B"]
    check("<base>:<adapter> names for loaded adapters pass", "with its own weights" in check_served(url, good))
    check("a bare adapter name is refused on SGLang, which would answer with the base model",
          refused(url, ["pasar-rft-A", "pasar-rft-B"], "BASE model"))
    check("an adapter the server never loaded is refused",
          refused(url, ["Qwen/Qwen3-8B:pasar-rft-C"], "would not answer"))
    check("…as is a base the server does not serve",
          refused(url, ["Qwen/Qwen3-32B:pasar-rft-A"], "would not answer"))
    one = subprocess.run(
        [sys.executable, "-m", "pasarbench.sweep", "--backend", "openai", "--model",
         "Qwen/Qwen3-8B:pasar-rft2-A", "--base-url", url, "--simulator", "silent", "--suite",
         "all", "--tasks", "T01", "-k", "1", "--strategies", "full", "--trace-root",
         tempfile.mkdtemp(), "--run-id", "one"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent)
    check("one adapter named as the model (GRPO's rollouts) is checked the same way: one "
          "the server never loaded is refused before the first episode",
          one.returncode != 0 and "would not answer" in one.stderr and "running" not in one.stdout,
          (one.stdout + one.stderr)[-400:])
    srv.shutdown()

    srv, url = serve([{**base, "owned_by": "vllm"},
                      *[{**c, "owned_by": "vllm"} for c in lora]])
    check("vLLM routes a bare adapter name to the adapter, so there it passes",
          "with its own weights" in check_served(url, ["pasar-rft-A", "pasar-rft-B"]))
    srv.shutdown()
    check("no server, no evaluation", refused(url, good, "is the agent server up"))

    import contextlib
    import io
    tr = _script("train_rft")
    probe_cards = [base, {"id": "pasar-probe", "object": "model", "owned_by": "sglang",
                          "root": "checkpoints/probe", "parent": "Qwen/Qwen3-8B"}]

    def probe(adapter_logprob):
        class Chat(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": probe_cards}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                lp = adapter_logprob if body["model"].endswith(":pasar-probe") else -0.25
                msg = {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "get_order", "arguments": '{"order_id": "O1003"}'}}]}
                logprobs = {"content": [{"token": "<tool_call>", "logprob": lp,
                                         "top_logprobs": []}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": msg,
                                                          "logprobs": logprobs}]}).encode())

            def log_message(self, *a):
                pass
        srv = HTTPServer(("127.0.0.1", 0), Chat)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        sys.argv = ["train_rft.py", "probe-serve", "--model", "Qwen/Qwen3-8B",
                    "--base-url", f"http://127.0.0.1:{srv.server_port}/v1"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tr.main()
        srv.shutdown()
        return out.getvalue()
    check("the LoRA probe passes when the adapter moves the next-token distribution",
          "LoRA serving works" in probe(-0.31))
    check("…and says loudly when the server answers exactly as without it",
          "!! the adapter changed nothing" in probe(-0.25))


def test_serve_scripts():
    print("\n=== the serve script, the pipeline and the wrappers agree ===")
    import re
    import shutil
    root = Path(__file__).resolve().parent.parent
    if not shutil.which("bash"):
        skip("serve script commands", "no bash")
        return
    from pasarbench.models import SIM_DEFAULT

    work = Path(tempfile.mkdtemp())
    for f in "AB":
        (work / "checkpoints" / f"rft-{f}").mkdir(parents=True)
        (work / "checkpoints" / f"rft-{f}" / "adapter_config.json").write_text("{}")
    env = {k: v for k, v in os.environ.items()
           if k not in ("SIM_MODEL", "SIM_TOKENIZER", "AGENT_MODEL", "SGLANG_ARGS",
                        "TOOL_PARSER", "SHARE_GPU", "MEM_FRACTION", "AGENT_GPU", "SIM_GPU",
                        "PASAR_GPU_INFO", "PASAR_LAYOUT", "PASAR_AGENT_POOL_GB")}

    def cmd(mode, **extra):
        r = subprocess.run(["bash", str(root / "scripts" / "serve_sglang.sh"), mode],
                           capture_output=True, text=True, cwd=work,
                           env={**env, "DRY_RUN": "1", **extra})
        return r.stdout if r.returncode == 0 else f"exit {r.returncode}: {r.stderr}"

    agent, sim, rft = cmd("agent"), cmd("sim"), cmd("rft")
    from pasarbench.models import AGENT_DEFAULT
    check("the agent: Qwen3.8-27B on GPU 0, port 8000, its XML tool-call parser",
          "CUDA_VISIBLE_DEVICES=0" in agent and f"--model-path {AGENT_DEFAULT}" in agent
          and "--port 8000" in agent and "--tool-call-parser qwen3_coder" in agent, agent)
    old = cmd("agent", AGENT_MODEL="Qwen/Qwen3-8B")
    check("…and Qwen3, which writes JSON tool calls, gets the JSON parser",
          "--tool-call-parser qwen " in old + " ", old)
    check("the hybrid agent gives its linear-attention states most of its cache pool, so "
          "~24 conversations fit at once; a plain transformer gets no such flag",
          "--mamba-full-memory-ratio 2.5" in agent and "mamba" not in old, agent)
    check("the customer: the pipeline's Gemma 4 on GPU 1, port 8001, with Google's "
          "tokenizer and the Gemma 4 reasoning parser",
          f"--model-path {SIM_DEFAULT}" in sim and "CUDA_VISIBLE_DEVICES=1" in sim
          and "--port 8001" in sim and "--tokenizer-path google/gemma-4-31B-it" in sim
          and "--reasoning-parser gemma4" in sim, sim)
    other = cmd("sim", SIM_MODEL="meta-llama/Llama-3.3-70B-Instruct")
    check("another customer gets its own tokenizer and no Gemma parser",
          "--tokenizer-path meta-llama/Llama-3.3-70B-Instruct" in other
          and "gemma4" not in other, other)
    tr = _script("train_rft")
    check("the adapters are loaded under the names the evaluation asks for",
          "--enable-lora" in rft and all(f"pasar-rft-{f}=checkpoints/rft-{f}" in rft for f in "AB"),
          rft)
    check("servers listen on localhost only, with no API key to leak",
          all("--host 127.0.0.1" in c and "--api-key" not in c for c in (agent, sim, rft)))
    pipe = (root / "scripts" / "gpu_pipeline.sh").read_text()
    check("the pipeline serves with SGLang and defaults to the same agent and customer",
          "serve_sglang.sh" in pipe and f'SIM_MODEL="${{SIM_MODEL:-{SIM_DEFAULT}}}"' in pipe
          and f'AGENT_MODEL="${{AGENT_MODEL:-{AGENT_DEFAULT}}}"' in pipe
          and tr.SIM_DEFAULT == SIM_DEFAULT and tr.AGENT_DEFAULT == AGENT_DEFAULT)
    dl = (root / "scripts" / "download_weights.sh").read_text()
    check("the weights downloaded are the ones served",
          f'AGENT_MODEL="${{AGENT_MODEL:-{AGENT_DEFAULT}}}"' in dl
          and f'SIM_DEFAULT="{SIM_DEFAULT}"' in dl)
    check("no script still starts vLLM",
          not (root / "scripts" / "serve_vllm.sh").exists()
          and "vllm serve" not in pipe
          and "vllm" not in (root / "scripts" / "setup_node.sh").read_text().lower())
    (work / "checkpoints" / "probe").mkdir(parents=True)
    (work / "checkpoints" / "probe" / "adapter_config.json").write_text("{}")
    pr = cmd("probe")
    check("the smoke stage's LoRA probe is served under the name probe-serve asks for",
          "--enable-lora" in pr and "pasar-probe=checkpoints/probe" in pr
          and "CUDA_VISIBLE_DEVICES=0" in pr, pr)
    # One card: both servers on GPU 0, each with a memory fraction worked out
    # from what is free as it starts (the numbers are gpu_plan's own).
    plan = _script("gpu_plan")
    one = {"SHARE_GPU": "1", "SIM_GPU": "0", "PYTHON": sys.executable,
           "HF_HOME": str(work / "no-hf-cache")}
    b200_fresh, b200_after = "NVIDIA B200,183359,182800", "NVIDIA B200,183359,85000"
    sa = cmd("agent", **one, PASAR_GPU_INFO=b200_fresh)
    ss = cmd("sim", **one, PASAR_GPU_INFO=b200_after)
    os.environ["PASAR_GPU_INFO"] = b200_fresh
    try:
        fa = plan.fraction("agent", plan.gpus()[0], 51.8, 30.7)
        os.environ["PASAR_GPU_INFO"] = b200_after
        fs = plan.fraction("sim", plan.gpus()[0], 51.8, 30.7)
    finally:
        del os.environ["PASAR_GPU_INFO"]
    capped = ("--max-running-requests 32", "--cuda-graph-max-bs-decode 32",
              "--chunked-prefill-size 8192")
    check("one card: the agent on GPU 0 with the planned fraction, batch and prefill capped",
          "CUDA_VISIBLE_DEVICES=0" in sa and f"--mem-fraction-static {fa} " in sa
          and all(c in sa for c in capped), sa)
    check("…and the customer on the same GPU with a fraction of what the agent left",
          "CUDA_VISIBLE_DEVICES=0" in ss and f"--mem-fraction-static {fs} " in ss
          and all(c in ss for c in capped) and fs > fa, ss)
    check("the customer's sliding-window pool is sized for windows, on either layout",
          "--swa-full-tokens-ratio 0.3" in ss and "--swa-full-tokens-ratio 0.3" in sim
          and "--mem-fraction-static" not in sim, sim)
    check("MEM_FRACTION overrides the worked-out fraction",
          "--mem-fraction-static 0.61 " in cmd("agent", **one, MEM_FRACTION="0.61"))
    left = cmd("sim", **one, PASAR_GPU_INFO="NVIDIA H200,143771,50000")
    check("a customer that would get too small a cache refuses to start, and says what to lower",
          left.startswith("exit 1") and "PASAR_AGENT_POOL_GB" in left, left)

    for f in "AB":
        (work / "checkpoints" / f"rft2-{f}").mkdir(parents=True)
        (work / "checkpoints" / f"rft2-{f}" / "adapter_config.json").write_text("{}")
    r2 = cmd("rft", RFT_NAME="rft2")
    check("round 2 serves its own adapters under its own names (RFT_NAME=rft2)",
          all(f"pasar-rft2-{f}=checkpoints/rft2-{f}" in r2 for f in "AB")
          and "pasar-rft-A" not in r2, r2)
    shutil.rmtree(work / "checkpoints")
    check("rft refuses to start without both adapters", cmd("rft").startswith("exit 1")
          and cmd("rft", RFT_NAME="rft2").startswith("exit 1"))

    # The settings that decide the bill survive a new terminal.
    (work / "scripts").mkdir()
    for f in ("gpu_pipeline.sh", "gpu_plan.py"):
        shutil.copy(root / "scripts" / f, work / "scripts" / f)
    clean = {k: v for k, v in env.items() if not k.startswith("PASAR_")}
    clean.update(PYTHON=sys.executable, PASAR_GPU_INFO="NVIDIA B200,183359,182800",
                 HF_HOME=str(work / "no-hf-cache"))

    def settings(**extra):
        r = subprocess.run(["bash", "scripts/gpu_pipeline.sh", "settings"], cwd=work,
                           capture_output=True, text=True, env={**clean, **extra})
        return r.returncode, r.stdout + r.stderr
    code, out = settings(PASAR_K="3")
    check("before stage 1 the knobs come from the environment, and the customer is the one "
          "every run since v31 has",
          code == 0 and "PASAR_K=3 " in out and "not run yet" in out
          and "PASAR_CUSTOMER=2" in out, out)
    (work / "data").mkdir()
    (work / "data" / "run_settings.env").write_text(
        "PASAR_K=3\nPASAR_COLLECT_K=4\nPASAR_EPOCHS=2\nPASAR_TRAIN_MAX_LEN=16384\n")
    code, out = settings()
    check("after it, a new terminal gets stage 1's values, not the defaults -- and a stage 1 "
          "from before v31, which recorded no customer, had customer 1",
          code == 0 and "PASAR_K=3 PASAR_COLLECT_K=4" in out and "PASAR_CUSTOMER=1" in out, out)
    code, out = settings(PASAR_K="5")
    check("…and a different repetition count is refused, not silently used",
          code != 0 and "stage 1 ran with PASAR_K=3" in out, out)
    code, out = settings(PASAR_CUSTOMER="2")
    check("…as is another customer than stage 1's: a base run and its fine-tune compared "
          "with different customers would be measuring the customer",
          code != 0 and "stage 1 ran with PASAR_CUSTOMER=1" in out, out)
    stages = {st: subprocess.run(["bash", "scripts/gpu_pipeline.sh", st], cwd=work,
                                 capture_output=True, text=True, env=clean)
              for st in ("smoke", "stage1", "stage2")}
    check("…and no stage runs on a stage 1 from before v31, whose runs had the old tools and "
          "customer, before a GPU hour is spent -- the refusal names round 2",
          all(r.returncode != 0 and "stage 1 ran before v31" in r.stderr and not r.stdout
              and "PASAR_ROUND=2" in r.stderr for r in stages.values()),
          {k: r.stderr[-200:] for k, r in stages.items()})
    code, out = settings(PASAR_ROUND="2")
    check("round 2 keeps round 1's files and settings, and starts with its own: today's "
          "customer, and its data balanced by trap",
          code == 0 and "round 2: runs P2-*" in out and "checkpoints/rft2-A" in out
          and "not run yet" in out and "PASAR_CUSTOMER=2" in out and "PASAR_PER_TRAP=20" in out,
          out)
    (work / "data" / "run2_settings.env").write_text(
        "PASAR_K=5\nPASAR_COLLECT_K=8\nPASAR_EPOCHS=2\nPASAR_TRAIN_MAX_LEN=16384\n"
        "PASAR_CUSTOMER=2\nPASAR_PER_TRAP=20\n")
    code, out = settings(PASAR_ROUND="2", PASAR_PER_TRAP="10")
    check("…and, once its stage 1 has run, another quota than the one it built with is "
          "refused", code != 0 and "stage 1 ran with PASAR_PER_TRAP=20" in out, out)
    (work / "data" / "run2_settings.env").unlink()
    code, out = settings(PASAR_ROUND="3")
    check("…and there is no round 3", code != 0 and "PASAR_ROUND is 1 or 2" in out, out)
    (work / "data" / "run_settings.env").write_text(
        "PASAR_K=3\nPASAR_COLLECT_K=4\nPASAR_EPOCHS=2\nPASAR_TRAIN_MAX_LEN=16384\nPASAR_CUSTOMER=2\n")
    code, out = settings(PASAR_CUSTOMER="2")
    check("…while the customer stage 1 recorded goes through",
          code == 0 and "PASAR_CUSTOMER=2" in out, out)
    (work / "data" / "run_settings.env").write_text(
        "PASAR_K=3\nPASAR_COLLECT_K=4\nPASAR_EPOCHS=2\nPASAR_TRAIN_MAX_LEN=16384\n")
    code, out = settings(PASAR_EPOCHS="1", PASAR_TRAIN_MAX_LEN="12288")
    check("…while epochs and the training length cap may still change for stage 2",
          code == 0 and "PASAR_EPOCHS=1 PASAR_TRAIN_MAX_LEN=12288" in out, out)
    code, out = settings()
    check("`settings` shows how this machine's GPUs will be used",
          code == 0 and "layout: shared" in out and "NVIDIA B200" in out, out)
    code, out = settings(PASAR_GPU_INFO="NVIDIA H100 80GB HBM3,81559,81000")
    check("…and says plainly when one card cannot hold both models",
          code == 0 and "layout: too-small" in out and "H200 or B200" in out, out)
    pipe = (root / "scripts" / "gpu_pipeline.sh").read_text()
    recorded = re.findall(r"(PASAR_[A-Z_]+)=%s", pipe)
    grpo = (root / "scripts" / "train_grpo.py").read_text()
    check("stage 1 records the customer with the knobs, the reference row and GRPO's "
          "rollouts use the run's, and every knob it records reaches a Modal job",
          "PASAR_CUSTOMER" in recorded and '--customer "$CUSTOMER"' in pipe
          and 'export PASAR_CUSTOMER="$CUSTOMER"' in pipe
          and '"--customer", str(args.customer)' in grpo
          and all(f'"{k}"' in (root / "scripts" / "modal_pipeline.py").read_text().split(
              "PASSED_ON = (")[1].split(")")[0] for k in recorded), recorded)
    check("round 2's stages name every run, file, log and adapter apart from round 1's, and "
          "its stage 1 also runs the base model on the 62 newer tasks",
          "2) P=P2; RFT=rft2; LP=\"r2-\"; SETTINGS=data/run2_settings.env ;;" in pipe
          and '--run-id "$P-base"' in pipe and '--suite edition2 --run-id "$P-base-e2"' in pipe
          and '--run "traces/$P-collect/full" --out-dir "data/$RFT"' in pipe
          and '--per-trap "$PER_TRAP" --max-per-task "$COLLECT_K"' in pipe
          and '--out "checkpoints/$RFT-$F"' in pipe and '--run-id "$P-ref"' in pipe
          and '--adapter-prefix "pasar-$RFT-" --run-id "$P-rft"' in pipe
          and 'export RFT_NAME="$RFT"' in pipe)
    shared = pipe[pipe.index('if [ -n "${SHARE_GPU:-}" ]; then\n    # One card'):]
    check("on one card the customer starts only after the agent is up",
          shared.index("wait_ready 8000") < shared.index("serve_sglang.sh sim"))
    check("training keeps 32 turns per optimizer step on any number of GPUs",
          "ACCUM=$(( NGPU > 1 ? (32 + NGPU / 2) / NGPU : 32 ))" in pipe
          and '--grad-accum "$ACCUM"' in pipe
          and 'launch=("$PY")' in pipe)
    check("the smoke stage also tries training on one longest-length turn",
          "train_rft.py train-probe --max-len" in pipe)


def test_grpo_scripts():
    print("\n=== GRPO's stage: its tasks, rollouts, servers and evaluation agree ===")
    import contextlib
    import io
    import re
    import shutil
    root = Path(__file__).resolve().parent.parent
    from pasarbench.generate import generate as gen
    from pasarbench.rl.split import make_folds
    from pasarbench.sweep import fold_router

    # Its tasks: each fold's tasks the collection failed at least once.
    work = Path(tempfile.mkdtemp())
    folds = make_folds(ALL)
    fp = work / "folds.json"
    fp.write_text(json.dumps(folds))
    xa = next(t.task_id for t in ALL if folds["task_fold"][t.task_id] == "A")
    ya = next(t.task_id for t in ALL if folds["task_fold"][t.task_id] == "A" and t.task_id != xa)
    xb = next(t.task_id for t in ALL if folds["task_fold"][t.task_id] == "B")
    r = _sweep(work, "C", f"{xa},{ya},{xb}", 2)
    m = _sweep(work, "M", f"{xa},{xb}", 1, "--backend", "mute")
    assert r.returncode == 0 and m.returncode == 0, (r.stderr + m.stderr)[-800:]
    cell = work / "C" / "full"
    for f in (work / "M" / "full").glob("*.jsonl"):
        shutil.copy(f, cell / f.name.replace("__r0", "__r2"))
    g = _script("train_grpo")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g.main(["tasks", "--folds", str(fp), "--collect", str(cell), "--out-dir",
                str(work / "stats"), "--min-tasks", "1"])
    picked = {f: json.loads((work / "stats" / f"tasks-{f}.json").read_text())["tasks"]
              for f in "AB"}
    check("GRPO trains each fold on its tasks the collection failed at least once, and on "
          "nothing else -- never the newer tasks it is judged on",
          picked == {"A": [xa], "B": [xb]}, (picked, out.getvalue()))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g.main(["tasks", "--folds", str(fp), "--collect", str(cell), "--out-dir",
                    str(work / "stats2"), "--min-tasks", "2"])
        few = ""
    except SystemExit as e:
        few = str(e)
    check("…and too few to say anything is refused before a GPU hour is spent, with no "
          "task list left behind for the next run to go ahead on",
          "fewer than 2 tasks" in few and not (work / "stats2").exists(), few)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g.main(["collect", "--adapter", "pasar-rft2-A", "--tasks-file",
                str(work / "stats" / "tasks-A.json"), "--run-id", "P2-grpo-i1-collect-A",
                "--dry-run"])
    want = os.environ.get("PASAR_CUSTOMER") or "2"
    c = out.getvalue()
    check("its rollouts: the served adapter, its tasks, 8 a task at temperature 1, "
          "conversations saved, the round's customer, resumable",
          "--model Qwen/Qwen3.8-27B:pasar-rft2-A" in c and f"--suite all --tasks {xa} " in c
          and "-k 8 --temperature 1.0" in c and "--save-messages" in c
          and f"--customer {want}" in c and "--resume" in c
          and "--run-id P2-grpo-i1-collect-A" in c and '"enable_thinking": false' in c, c)
    check("…from the full softmax: every sampling parameter the server would otherwise take "
          "from the model's generation config, set to its neutral value",
          all(x in c for x in ('"top_p": 1.0', '"top_k": -1', '"min_p": 0.0',
                                '"repetition_penalty": 1.0', '"presence_penalty": 0.0')), c)

    # The newer tasks' own split, and the evaluation it routes.
    sp = _script_module("pasarbench.rl.split")
    e2 = work / "folds-e2.json"
    with contextlib.redirect_stdout(io.StringIO()):
        sp.main(["--suite", "edition2", "--out", str(e2)])
    f2 = json.loads(e2.read_text())
    newer = list(gen(edition=2)[0])
    fam = {}
    for t in newer:
        fam.setdefault(t.task_id.split(".")[0], set()).add(f2["task_fold"][t.task_id])
    check("the 62 newer tasks have a split of their own, dealt as the main one: whole "
          "families, every trap in both halves",
          len(f2["task_fold"]) == 62 and all(len(v) == 1 for v in fam.values())
          and all({f2["task_fold"][t.task_id] for t in newer if t.trap == trap} == {"A", "B"}
                  for trap in {t.trap for t in newer}), f2.get("split_digest"))
    route = fold_router(SimpleNamespace(model="base", folds=str(e2), suite="edition2",
                                        fold_models="A=grpo-A,B=grpo-B"))
    check("…which shares them out between the two adapters, neither of which trained on "
          "any of them", sorted({route(t) for t in newer}) == ["grpo-A", "grpo-B"]
          and all(route(t) != f"grpo-{f2['task_fold'][t.task_id]}" for t in newer))
    try:
        fold_router(SimpleNamespace(model="base", folds=str(e2), suite="all",
                                    fold_models="A=grpo-A,B=grpo-B"))
        wrong = ""
    except SystemExit as e:
        wrong = str(e)
    check("…and the main suite is refused against it, as any split that does not describe "
          "the tasks run", "does not describe these tasks" in wrong, wrong)
    tr = _script("train_rft")

    def cmd(argv):
        sys.argv = ["train_rft.py", *argv, "--dry-run"]
        o = io.StringIO()
        with contextlib.redirect_stdout(o):
            tr.main()
        return o.getvalue()
    ev = cmd(["eval", "--suite", "edition2", "--adapter-prefix", "pasar-grpo2-i1-",
              "--run-id", "P2-grpo-i1-e2"])
    check("`eval --suite edition2` runs the adapters on the newer tasks by their split",
          "--suite edition2" in ev and "--folds data/splits/folds-e2.json" in ev
          and "A=Qwen/Qwen3.8-27B:pasar-grpo2-i1-A,B=Qwen/Qwen3.8-27B:pasar-grpo2-i1-B" in ev
          and "--temperature 0.0" in ev, ev)

    # The servers and the stage.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PASAR_") and k not in ("SIM_MODEL", "AGENT_MODEL", "SHARE_GPU",
                                                       "SGLANG_ARGS", "RFT_NAME", "LORA")}
    (work / "checkpoints" / "rft2-A").mkdir(parents=True)
    (work / "checkpoints" / "rft2-A" / "adapter_config.json").write_text("{}")

    def serve(mode, **extra):
        r = subprocess.run(["bash", str(root / "scripts" / "serve_sglang.sh"), mode],
                           capture_output=True, text=True, cwd=work,
                           env={**env, "DRY_RUN": "1", **extra})
        return r.stdout if r.returncode == 0 else f"exit {r.returncode}: {r.stderr}"
    one = serve("lora", LORA="checkpoints/rft2-A")
    check("one adapter is served under the name its rollouts ask for",
          "--enable-lora --max-loras-per-batch 1 --lora-paths pasar-rft2-A=checkpoints/rft2-A"
          in one and "--host 127.0.0.1" in one, one)
    check("…and nothing is served without one", serve("lora").startswith("exit 1"))
    pipe = (root / "scripts" / "gpu_pipeline.sh").read_text()
    stage = pipe[pipe.index("\n  grpo)"):pipe.index("\n  settings)")]
    check("each iteration serves the adapter it starts from, rolls it out and builds from "
          "the same name, then updates from that adapter",
          'name="pasar-$(basename "$prev")"' in stage and 'export LORA="$prev"' in stage
          and "serve lora" in stage and '--adapter "$name"' in stage
          and '--served-model "$AGENT_MODEL:$name"' in stage
          and '--run-id "$P-grpo-i$i-collect-$F"' in stage
          and '--run "traces/$P-grpo-i$i-collect-$F/full"' in stage
          and 'train_rft.py train --objective grpo --init "$prev"' in stage
          and '--epochs 1 --lr "$GRPO_LR"' in stage and 'prev="$out"' in stage
          and 'serve_sglang.sh" 2>/dev/null' not in stage
          and 'pasar-$(basename "$LORA")' in (root / "scripts" / "serve_sglang.sh").read_text())
    check("…the update runs with the servers stopped, and a build cut short is built again",
          stage.index("--stats \"$st\"") < stage.index("          stop\n")
          < stage.index("train_rft.py train --objective grpo")
          and 'if [ ! -f "$st" ]' in stage)
    check("then the last iteration's adapters on what none trained on: the held-out fold, "
          "and the newer tasks beside the RFT adapters and the base model, compared in pairs",
          'export RFT_NAME="$G-i$N"' in stage
          and '--adapter-prefix "pasar-$G-i$N-" --run-id "$P-grpo-i$N"' in stage
          and '--suite edition2 --adapter-prefix "pasar-$G-i$N-"' in stage
          and '--run-id "$P-rft-e2"' in stage and "rl.split --suite edition2" in stage
          and 'compare_cells.py "traces/$P-rft/full" "traces/$P-grpo-i$N/full"' in stage
          and 'compare_cells.py "traces/$P-base-e2/full" "traces/$P-rft-e2/full"' in stage
          and 'compare_cells.py "traces/$P-rft-e2/full" "traces/$P-grpo-i$N-e2/full"' in stage)

    # `finished`: the episodes of a cell that ran to the end -- not one a server
    # error ended (the sweep runs it again), nor one cut off mid-line.
    fn = pipe[pipe.index("finished() {"):pipe.index("\n}\n", pipe.index("finished() {")) + 3]
    cell_f = work / "fin"
    cell_f.mkdir()
    (cell_f / "a.jsonl").write_text('{"type": "header"}\n{"type": "footer", '
                                    '"stop_reason": "done"}\n')
    (cell_f / "b.jsonl").write_text('{"type": "header"}\n{"type": "footer", '
                                    '"stop_reason": "backend_error"}\n')
    (cell_f / "c.jsonl").write_text('{"type": "header"}\n{"type": "foo')
    (cell_f / "d.jsonl").write_text("")
    r = subprocess.run(["bash", "-c", f'PY="{sys.executable}"\n{fn}\nfinished "{cell_f}"; '
                        f'finished "{work / "none"}"'], capture_output=True, text=True)
    check("a run counts as finished episode by episode, not counting one a server error "
          "ended or one cut off mid-line", r.stdout.split() == ["1", "0"], r.stdout + r.stderr)
    for f in ("gpu_pipeline.sh", "gpu_plan.py"):
        (work / "scripts").mkdir(exist_ok=True)
        shutil.copy(root / "scripts" / f, work / "scripts" / f)
    base = {**env, "PYTHON": sys.executable, "PASAR_GPU_INFO": "NVIDIA B200,183359,182800",
            "HF_HOME": str(work / "no-hf-cache")}

    def grpo(**extra):
        r = subprocess.run(["bash", "scripts/gpu_pipeline.sh", "grpo"], cwd=work,
                           capture_output=True, text=True, env={**base, **extra})
        return r.returncode, r.stdout + r.stderr
    code, out1 = grpo()
    check("the stage is round 2's: without PASAR_ROUND=2 it says so and stops",
          code != 0 and "PASAR_ROUND=2" in out1, out1)
    code, out2 = grpo(PASAR_ROUND="2")
    check("…and it needs both of round 2's adapters", code != 0
          and "no checkpoints/rft2-B" in out2, out2)
    (work / "checkpoints" / "rft2-B").mkdir(parents=True)
    (work / "checkpoints" / "rft2-B" / "adapter_config.json").write_text("{}")
    code, out3 = grpo(PASAR_ROUND="2")
    check("…and round 2's collection, evaluation and base run on the newer tasks",
          code != 0 and "no traces/P2-collect/full" in out3, out3)
    for d in ("P2-collect", "P2-rft", "P2-base-e2"):
        (work / "traces" / d / "full").mkdir(parents=True)
    (work / "data").mkdir()
    settings = work / "data" / "run2_settings.env"
    settings.write_text("PASAR_K=5\nPASAR_COLLECT_K=8\nPASAR_EPOCHS=2\nPASAR_TRAIN_MAX_LEN=16384\n"
                        "PASAR_CUSTOMER=2\nPASAR_PER_TRAP=20\n")
    code, out4 = grpo(PASAR_ROUND="2")
    check("its first run records the rollouts a task with the round's settings",
          "PASAR_GRPO_K=8" in settings.read_text(), (settings.read_text(), out4[-300:]))
    code, out5 = grpo(PASAR_ROUND="2", PASAR_GRPO_K="6")
    check("…and a later run with another number is refused", code != 0
          and "GRPO's first iteration ran with 8" in out5, out5)
    sets = subprocess.run(["bash", "scripts/gpu_pipeline.sh", "settings"], cwd=work,
                          capture_output=True, text=True,
                          env={**base, "PASAR_ROUND": "2", "PASAR_GRPO_ITERS": "2"}).stdout
    check("`settings` shows GRPO's knobs for round 2",
          "grpo: PASAR_GRPO_ITERS=2 PASAR_GRPO_K=8 PASAR_GRPO_LR=2e-5" in sets, sets)
    modal_src = (root / "scripts" / "modal_pipeline.py").read_text()
    passed_on = modal_src.split("PASSED_ON = (")[1].split(")")[0]
    nb = (root / "scripts" / "pasar_notebook.py").read_text()
    check("a Modal job and a notebook run the stage, as round 2's only, with its knobs",
          all(f'"{k}"' in passed_on for k in ("PASAR_GRPO_ITERS", "PASAR_GRPO_K",
                                               "PASAR_GRPO_LR"))
          and '"grpo")' in modal_src and "PASAR_ROUND=2 modal run" in modal_src
          and 'STAGES = ("smoke", "stage1", "stage2", "grpo")' in nb
          and "pn.run('grpo', round=2)" in nb)
    results = re.search(r"RESULTS = \[(.*?)\]", modal_src, re.S).group(1)
    check("`pull` brings back GRPO's stats and evaluations, and not its rollouts",
          all(x in results for x in ('"data/grpo2/stats"', '"traces/P2-rft-e2"',
                                      '"traces/P2-grpo-i?"', '"traces/P2-grpo-i?-e2"'))
          and not any("collect" in x for x in re.findall(r'"([^"]+)"', results)))


def _script_module(name):
    import importlib
    return importlib.import_module(name)


def test_gpu_plan():
    print("\n=== one GPU or two: how the servers share the memory ===")
    plan = _script("gpu_plan")
    card = lambda name, total, free=None: {  # noqa: E731
        "name": name, "total_gib": total, "free_gib": total - 0.5 if free is None else free}
    W_A, W_S = 51.8, 30.7
    b200, h200, h100 = card("NVIDIA B200", 179.06), card("NVIDIA H200", 140.4), \
        card("NVIDIA H100 80GB HBM3", 79.6)
    check("one B200 or one H200: shared; one H100: does not fit; two H100s: split",
          [plan.plan([c], W_A, W_S)["layout"] for c in (b200, h200, h100)]
          + [plan.plan([h100, h100], W_A, W_S)["layout"], plan.plan([], W_A, W_S)["layout"]]
          == ["shared", "shared", "too-small", "split", "none"])
    why = plan.plan([h100], W_A, W_S)["why"]
    check("…and the refusal says what would fit", "H200 or B200" in why, why)

    # Start the agent on an empty card, charge it what SGLang then holds (its
    # static share of the memory free before loading, a CUDA context, graphs
    # and activations), start the customer in what is left: both caches get
    # at least the minimum and the reserve stays free.
    for c in (b200, h200):
        p = plan.plan([c], W_A, W_S)
        fa = plan.fraction("agent", c, W_A, W_S)
        pre_a = c["free_gib"] - plan.CTX_GIB
        agent_pool = pre_a * fa - W_A - plan.LOAD_GIB
        agent_used = pre_a * fa + plan.CTX_GIB + 3.0
        after = dict(c, free_gib=c["free_gib"] - agent_used)
        fs = plan.fraction("sim", after, W_A, W_S)
        pre_s = after["free_gib"] - plan.CTX_GIB
        sim_pool = pre_s * fs - W_S - plan.LOAD_GIB
        left = pre_s * (1 - fs)
        check(f"{c['name']}: agent cache {agent_pool:.0f} GiB, customer cache "
              f"{sim_pool:.0f} GiB, {left:.0f} GiB left for both servers' runtime",
              abs(agent_pool - p["pool_agent"]) < 0.5 and sim_pool >= plan.MIN_POOL_GIB
              and left >= plan.reserve_gib() - 0.5 and 0 < fa < fs < 0.95,
              (fa, fs, agent_pool, sim_pool, left))
    os.environ["PASAR_AGENT_POOL_GB"] = "80"
    try:
        tight = plan.plan([h200], W_A, W_S)
    finally:
        del os.environ["PASAR_AGENT_POOL_GB"]
    check("an agent cache that leaves the customer too little is refused",
          tight["layout"] == "too-small" and "PASAR_AGENT_POOL_GB" in tight["why"], tight)
    try:
        plan.fraction("agent", card("NVIDIA B200", 179.06, free=60.0), W_A, W_S)
        busy = "no refusal"
    except SystemExit as e:
        busy = str(e)
    check("a card already busy with something else is noticed before SGLang runs out of memory",
          "another process" in busy, busy)
    base = plan.plan([b200], W_A, W_S)
    os.environ["PASAR_RESERVE_GB"] = "16"
    try:
        roomy = plan.plan([b200], W_A, W_S)
        after = dict(b200, free_gib=b200["free_gib"] - 95.0)
        pre = after["free_gib"] - plan.CTX_GIB
        kept = pre * (1 - plan.fraction("sim", after, W_A, W_S))
    finally:
        del os.environ["PASAR_RESERVE_GB"]
    check("PASAR_RESERVE_GB, the fix for an out-of-memory at run time, keeps that much free "
          "and shrinks both caches to pay for it",
          abs(kept - 16) < 0.5 and roomy["pool_agent"] < base["pool_agent"]
          and roomy["pool_sim"] < base["pool_sim"], (kept, roomy, base))
    l40s = card("NVIDIA L40S", 45.0)
    small = plan.plan([l40s, l40s], W_A, W_S)
    check("two cards too small for a server each are refused, not split",
          small["layout"] == "too-small" and "80 GB" in small["why"], small)

    hub = Path(tempfile.mkdtemp())
    snap = hub / "hub" / "models--org--m" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    for i, size in enumerate((3 * 2**30, 2**30)):
        with (snap / f"model-{i}.safetensors").open("wb") as fh:
            fh.truncate(size)
    os.environ["HF_HOME"], old = str(hub), os.environ.get("HF_HOME")
    try:
        got = plan.weights_gib("org/m")
        known = plan.weights_gib(plan.AGENT_DEFAULT)
    finally:
        if old is None:
            del os.environ["HF_HOME"]
        else:
            os.environ["HF_HOME"] = old
    check("weights are measured from the downloaded files, or known before the download",
          abs(got - 4.0) < 1e-6 and known == plan.KNOWN_WEIGHTS_GIB[plan.AGENT_DEFAULT], got)
    check("the projection's price follows the card: one B200 costs less than two H100s",
          plan.price([b200]) < plan.price([h100, h100]), (plan.price([b200]),
                                                          plan.price([h100, h100])))


def test_markup_audit():
    print("\n=== chat-format tokens in the text are caught ===")
    import contextlib
    import io
    it = _script("inspect_trace")
    work = Path(tempfile.mkdtemp())

    def episode(name, customer, agent):
        recs = [{"type": "header", "task_id": "T01"},
                {"type": "step", "step": 1, "model_content": agent, "tool_calls": []},
                {"type": "event", "kind": "user_turn", "turn": 1, "text": customer},
                {"type": "footer", "passed": True}]
        (work / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    episode("clean", "Order O1003, thanks <3", "Could you give me the order number?")
    eps = [it.load(str(work / "clean.jsonl"))]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        n = it.markup(eps)
    check("clean text passes", n == 0 and "none in 1 turns" in out.getvalue(), out.getvalue())
    episode("leaky", "<|channel>thought\n<channel|>It is O1003", '<tool_call>\n{"name": "get_order"}')
    eps.append(it.load(str(work / "leaky.jsonl")))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        n = it.markup(eps)
    check("a Gemma 4 thought channel from the customer and an unparsed tool call "
          "from the agent are both flagged",
          n == 2 and "customer: 1 of 2" in out.getvalue() and "agent: 1 of 2" in out.getvalue(),
          out.getvalue())


def test_qwen38_turns():
    print("\n=== Qwen3.8's format: XML tool calls, a think block on every turn ===")
    import _tokenizer
    if not _tokenizer.available():
        skip("Qwen3.8-style examples", "transformers/tokenizers not installed")
        return
    from pasarbench.rl.sft import _for_template, api_messages, build, read_trace, render_turn
    from pasarbench.rl.split import make_folds

    tok = _tokenizer.local_tokenizer(template="qwen3_8_style_chat_template.jinja")
    root = Path(tempfile.mkdtemp())
    folds = make_folds(ALL)
    a = next(t.task_id for t in ALL if folds["task_fold"][t.task_id] == "A" and "." in t.task_id)
    cell = _collection(root, a, k=1)
    files, stats = build(cell, BY_ID, folds, tok, max_per_task=3, expect_model="toy")
    exs = files["A"]
    check("examples build, one per agent turn", exs and stats["examples"]["A"] == len(exs), stats)
    check("every prompt ends in the empty think block of non-thinking mode",
          all(x["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") for x in exs))
    check("tool calls are trained in Qwen3.8's XML form, typed arguments included",
          any("<tool_call>\n<function=" in x["completion"] and "<parameter=" in x["completion"]
              for x in exs) and all(x["completion"].endswith("<|im_end|>") for x in exs))
    ep = read_trace(next(cell.glob("*.jsonl")))
    msgs = api_messages(ep["messages"]["messages"])
    first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    p, c = render_turn(tok, msgs, first, None, {"enable_thinking": False})
    whole = tok.apply_chat_template(_for_template(msgs), tokenize=False, enable_thinking=False)
    check("with every turn's think block kept, a whole-conversation render agrees with "
          "the per-turn one (unlike Qwen3's); per-turn examples are right for both",
          (p + c) in whole)
    try:
        tok.apply_chat_template([{"role": "system", "content": "policy"}], tokenize=False)
        ok = False
    except Exception as e:  # noqa: BLE001
        ok = "No user query" in str(e)
    check("the template's own refusals surface instead of being papered over", ok)


def test_qwen35_lora():
    print("\n=== LoRA on Qwen3.8's architecture, with logits for the turn only ===")
    import _tokenizer
    try:
        import peft  # noqa: F401
        import torch
        from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    except ImportError as e:
        skip("Qwen3.5 LoRA", f"{e.name} not installed")
        return
    import contextlib
    import io

    from transformers import AutoModelForCausalLM
    from transformers.utils.import_utils import is_flash_linear_attention_available
    if is_flash_linear_attention_available():
        # fla's kernels take CUDA tensors, and transformers picks them whenever a
        # GPU is visible, so this CPU toy cannot run next to one.
        skip("Qwen3.5 LoRA", "a GPU and fla are visible: run with CUDA_VISIBLE_DEVICES=")
        return

    tok = _tokenizer.local_tokenizer(template="qwen3_8_style_chat_template.jinja")
    work = Path(tempfile.mkdtemp())
    model_dir = work / "toy35"
    text = dict(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_conv_kernel_dim=4, max_position_embeddings=4096,
                layer_types=["linear_attention"] * 3 + ["full_attention"],
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32)
    torch.manual_seed(0)
    with contextlib.redirect_stderr(io.StringIO()):
        Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision)
                                        ).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    with contextlib.redirect_stderr(io.StringIO()):
        lm = AutoModelForCausalLM.from_pretrained(model_dir)
    check("a vision-language checkpoint loads for training as its language model alone",
          type(lm).__name__ == "Qwen3_5ForCausalLM"
          and not any("visual" in n for n, _ in lm.named_modules()), type(lm).__name__)

    x = torch.randint(0, len(tok), (1, 48))
    labels = torch.full((1, 48), -100)
    labels[0, 40:] = x[0, 40:]
    keep = 48 - 40 + 1
    with torch.no_grad():
        full = lm(input_ids=x, labels=labels).loss
        tail = lm(input_ids=x, labels=labels[:, -keep:], logits_to_keep=keep).loss
    check("logits for the turn only give the same loss as logits for every position",
          torch.allclose(full, tail, atol=1e-5), (float(full), float(tail)))

    data = work / "train-A.jsonl"
    rows = [{"task_id": "T01", "fold": "A", "turn": 0, "tokens": 0,
             "prompt": "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
             "completion": "<tool_call>\n<function=get_order>\n<parameter=order_id>\nO1003\n"
                           "</parameter>\n</function>\n</tool_call><|im_end|>"}] * 4
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tr = _script("train_rft")
    sys.argv = ["train_rft.py", "train", "--data", str(data), "--model", str(model_dir),
                "--out", str(work / "rft-A"), "--max-steps", "2", "--grad-accum", "1",
                "--fp32", "--attn", "eager", "--save-steps", "1000"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            tr.main()
        ok, err = True, ""
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    cfg_path = work / "rft-A" / "adapter_config.json"
    check("two LoRA steps run on the hybrid architecture and save an adapter",
          ok and cfg_path.exists(), err or out.getvalue()[-600:])
    if ok:
        from safetensors import safe_open
        with safe_open(str(work / "rft-A" / "adapter_model.safetensors"), "pt") as f:
            keys = list(f.keys())
        mods = {k.split(".lora_")[0].split(".")[-1] for k in keys}
        meta = json.loads((work / "rft-A" / "pasarbench_meta.json").read_text())
        check("the adapter covers attention and MLP projections only: no linear-attention "
              "projections SGLang cannot serve, no vision tower",
              mods <= {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                       "down_proj"} and {"q_proj", "gate_proj", "down_proj"} <= mods
              and not any("linear_attn" in k or "visual" in k for k in keys), sorted(mods))
        check("…and the run records how the loss was computed",
              meta.get("logits") == "turn only" and meta.get("lora_targets"), meta)

    probe = work / "probe"
    sys.argv = ["train_rft.py", "probe-adapter", "--model", str(model_dir), "--out", str(probe),
                "--lora-r", "8"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            tr.main()
        from peft import PeftModel
        with contextlib.redirect_stderr(io.StringIO()):
            base = AutoModelForCausalLM.from_pretrained(model_dir)
            with torch.no_grad():
                before = base(input_ids=x).logits
                after = PeftModel.from_pretrained(base, probe)(input_ids=x).logits
        ok, err = not torch.allclose(before, after, atol=1e-6), ""
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
    check("the LoRA probe adapter is built from the config alone, loads onto the model, "
          "and changes its output (a zero one could not tell applied from dropped)",
          ok, err or out.getvalue()[-400:])


def test_modal_runner():
    print("\n=== the Modal runner matches the node's setup ===")
    import ast
    root = Path(__file__).resolve().parent.parent
    src = (root / "scripts" / "modal_pipeline.py").read_text()
    ast.parse(src)
    setup = (root / "scripts" / "setup_node.sh").read_text()
    import re
    pin_modal = re.search(r'SGLANG_VERSION = "([^"]+)"', src).group(1)
    pin_node = re.search(r'SGLANG_VERSION="\$\{SGLANG_VERSION:-([^}]+)\}"', setup).group(1)
    check("Modal and a rented node install the same SGLang", pin_modal == pin_node,
          (pin_modal, pin_node))
    check("…and the same training packages around its pins",
          "peft accelerate flash-linear-attention" in src
          and "peft accelerate flash-linear-attention" in setup)
    check("the job gets code only: no outputs, and no .env file of API keys",
          all(f'"{d}"' in src for d in ("traces", "data", "checkpoints", "logs", ".git",
                                         ".env")))
    results = re.search(r"RESULTS = \[(.*?)\]", src, re.S).group(1)
    check("`pull` brings back everything the report's training section reads",
          all(x in results for x in ('"traces/P-base"', '"traces/P-rft"', '"traces/P-ref"',
                                      '"data/splits"', '"data/rft/stats.json"', '"logs"')))
    check("stages run on one B200 unless PASAR_GPU says otherwise, for up to Modal's "
          "24-hour limit", 'GPU = os.environ.get("PASAR_GPU", "B200")' in src
          and "gpu=GPU" in src and "timeout=24 * 3600" in src)
    check("…and the Volume is attached whatever version it was made with (a notebook's "
          "Files panel makes its own)", "version=" not in src.split("volume = ")[1].split("\n")[0])
    if os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IS_REMOTE"):
        skip("the Modal app loads", "already inside a Modal job")
        return
    try:
        import modal  # noqa: F401
    except ImportError:
        skip("the Modal app loads", "modal not installed")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("modal_pipeline", root / "scripts" / "modal_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    check("the Modal app loads with the installed SDK", mod.app.name == "pasarbench")

    def gpu_ok(spec):
        try:
            return mod.check_gpu(spec) == spec
        except SystemExit:
            return False
    check("GPU requests that hold both models pass: one B200 or H200, two 80 GB cards",
          all(gpu_ok(g) for g in ("B200", "H200", "H100:2", "A100-80GB:2", "B200:2")))
    check("…and the rest are refused before a GPU is paid for: one H100, B300 (CUDA 13.1), "
          "B200+ (may be a B300), small cards",
          not any(gpu_ok(g) for g in ("H100", "H100!", "B300", "B200+", "L40S:2", "A100")))

    # The stage lock, driven by hand: a live stage blocks a second one, its own
    # restart after preemption does not, and a dead one stops blocking.
    import time
    lock = Path(tempfile.mkdtemp()) / "logs" / ".stage-lock.json"
    mod.LOCK = str(lock)
    calls = iter(["call-1", "call-2", "call-1", "call-3"])
    mod.modal.current_function_call_id = lambda: next(calls)
    first = mod._claim("stage1")

    def refused():
        try:
            mod._claim("stage2")
            return False
        except SystemExit as e:
            return "already running" in str(e)
    check("a second GPU stage is refused while the first is alive", refused())
    again = mod._claim("stage1")
    check("…but Modal's restart of the same call after preemption is not",
          isinstance(again, type(first)))
    first.set()
    again.set()
    held = json.loads(lock.read_text())
    lock.write_text(json.dumps({**held, "beat": time.time() - mod.STALE - 1}))
    late = mod._claim("stage2")
    late.set()
    check("…and a stage that died without cleaning up stops blocking after a while",
          json.loads(lock.read_text())["stage"] == "stage2")

    # A notebook and a Modal job use the same lock file on the same Volume.
    pn = _script("pasar_notebook")
    pn._state.update(work=lock.parent.parent)
    lock.unlink()
    nb = pn._claim("smoke")
    mod.modal.current_function_call_id = lambda: "call-8"

    def job_refused():
        try:
            mod._claim("stage1")
            return False
        except SystemExit as e:
            return "already running" in str(e)
    check("a stage running in a notebook blocks a Modal job on the same Volume", job_refused())
    pn._release(nb)
    mod.modal.current_function_call_id = lambda: "call-9"
    job = mod._claim("stage1")
    try:
        pn._claim("stage2")
        nb_refused = False
    except SystemExit as e:
        nb_refused = "already running" in str(e)
    job.set()
    check("…and a running Modal job blocks a notebook stage", nb_refused)


def test_notebook_helper():
    print("\n=== the Modal Notebook helper ===")
    import re
    import shutil
    import tarfile
    root = Path(__file__).resolve().parent.parent
    src = (root / "scripts" / "pasar_notebook.py").read_text()
    modal_src = (root / "scripts" / "modal_pipeline.py").read_text()
    pins = {re.search(r'SGLANG_VERSION = "([^"]+)"', src).group(1),
            re.search(r'SGLANG_VERSION = "([^"]+)"', modal_src).group(1),
            re.search(r'SGLANG_VERSION="\$\{SGLANG_VERSION:-([^}]+)\}"',
                      (root / "scripts" / "setup_node.sh").read_text()).group(1)}
    check("a notebook installs the SGLang a Modal job and a rented node install", len(pins) == 1,
          pins)
    grab = lambda text: re.search(r"RESULTS = \[(.*?)\]", text, re.S).group(1).split()  # noqa: E731
    check("…and packs the same results `pull` brings back", grab(src) == grab(modal_src))
    check("…using nothing outside the standard library (it installs everything else)",
          not re.search(r"^\s*(import|from)\s+(torch|transformers|modal|sglang|requests)\b",
                        src, re.M))

    base = Path(tempfile.mkdtemp())
    vol = base / "mnt" / "pasarbench"
    work = vol / "pasarbench"
    (work / "scripts").mkdir(parents=True)
    for f in ("gpu_pipeline.sh", "gpu_plan.py"):
        shutil.copy(root / "scripts" / f, work / "scripts" / f)
    run_dir = work / "traces" / "P-base" / "full"
    run_dir.mkdir(parents=True)
    for i in range(3):
        (run_dir / f"T{i}__r0.jsonl").write_text('{"type": "header"}\n{"type": "footer"}\n')
    (run_dir / "T9__r0.jsonl").write_text('{"type": "header"}\n')
    for rel in ("traces/P2-grpo-i1/full", "traces/P2-grpo-i1-collect-A/full",
                "data/grpo2/stats", "data/grpo2/examples"):
        (work / rel).mkdir(parents=True)
    for rel in ("traces/P2-grpo-i1/full/T0__r0.jsonl", "traces/P2-grpo-i1-collect-A/full/T0__r0.jsonl"):
        (work / rel).write_text('{"type": "header"}\n{"type": "footer"}\n')
    (work / "data/grpo2/stats/i1-A.stats.json").write_text("{}")
    (work / "data/grpo2/examples/i1-A.jsonl").write_text("{}\n")
    (work / "logs").mkdir()
    (work / "logs" / "stage1.log").write_text("baseline 3/1075\n")
    saved = {k: os.environ.get(k) for k in ("PASAR_NB_MNT", "PASAR_NB_LINK", "PASAR_NB_VENV",
                                            "HF_HOME", "PYTHON", "VIRTUAL_ENV", "PATH",
                                            "PYTHONUNBUFFERED")}
    os.environ.update(PASAR_NB_MNT=str(base / "mnt"), PASAR_NB_LINK=str(base / "vol"),
                      PASAR_NB_VENV=str(base / "venv"))
    out = __import__("io").StringIO()
    try:
        pn = _script("pasar_notebook")
        pn._gpus = lambda: []
        import contextlib
        with contextlib.redirect_stdout(out):
            st = pn.setup(install=False)
            pn.status()
            archive = pn.pack()
        check("setup finds the Volume holding the repo and links /vol to it, as a Modal "
              "job mounts it", st["root"] == base / "vol" and (base / "vol").resolve() == vol
              and os.environ["HF_HOME"] == str(base / "vol" / "hf"), (st, out.getvalue()))
        check("status counts finished episodes and shows the newest log",
              "traces/P-base: 3 finished of ~1075 (1 unfinished)" in out.getvalue()
              and "baseline 3/1075" in out.getvalue(), out.getvalue())
        names = tarfile.open(archive).getnames()
        check("pack archives the results under pasarbench/, ready to extract over the repo",
              "pasarbench/traces/P-base/full/T0__r0.jsonl" in names
              and "pasarbench/logs/stage1.log" in names, names[:5])
        check("…GRPO's evaluations and stats among them, by pattern, and not its rollouts or "
              "examples (several GB)", "pasarbench/traces/P2-grpo-i1/full/T0__r0.jsonl" in names
              and "pasarbench/data/grpo2/stats/i1-A.stats.json" in names
              and not any("collect" in n or "examples" in n for n in names), names)
        check("status counts GRPO's evaluations and rollouts too",
              "traces/P2-grpo-i1: 1 finished of ~1075" in out.getvalue()
              and "traces/P2-grpo-i1-collect-A: 1 finished" in out.getvalue(), out.getvalue())
        try:
            pn.run("grpo")
            r1 = ""
        except SystemExit as e:
            r1 = str(e)
        check("GRPO is round 2's, in a notebook too", "round=2" in r1, r1)
        try:
            pn.run("smoke")
            refused = ""
        except SystemExit as e:
            refused = str(e)
        check("a stage refuses to start in a notebook without a GPU, and says where to pick one",
              "compute profile" in refused, refused)
        cmd = pn.launch("stage1", dry_run=True)
        check("launch starts the same detached job part A starts from a laptop",
              cmd[1:] == ["-m", "modal", "run", "--detach", "scripts/modal_pipeline.py",
                          "--stage", "stage1"], cmd)
        try:
            pn._round(3)
            bad = ""
        except SystemExit as e:
            bad = str(e)
        check("round=2 reaches the stage as PASAR_ROUND, here and in a launched job, and there "
              "is no round 3", pn._round(2) == {"PASAR_ROUND": "2"} and pn._round(None) == {}
              and "1 or 2" in bad, bad)
        out2 = __import__("io").StringIO()
        with contextlib.redirect_stdout(out2):
            pn.read("logs/stage1.log", tail=1)
        check("read prints a file from the repo on the Volume",
              out2.getvalue().strip() == "baseline 3/1075", out2.getvalue())
        lock = work / "logs" / ".stage-lock.json"
        lock.write_text(json.dumps({"call": "fc-job", "stage": "stage1",
                                    "beat": __import__("time").time()}))
        try:
            pn._claim("stage2")
            blocked = ""
        except SystemExit as e:
            blocked = str(e)
        check("a live stage elsewhere blocks a notebook stage, and the message says how to "
              "override a dead one", "already running" in blocked and "force=True" in blocked,
              blocked)
        taken = pn._claim("stage2", force=True)
        pn._release(taken)
        check("…force=True takes it over, and a finished stage releases its lock",
              not lock.exists())
        (base / "mnt" / "pasarbench-0930-v17.tar").write_text("")
        shutil.rmtree(work)
        try:
            pn.find_volume()
            where = ""
        except SystemExit as e:
            where = str(e)
        check("an uploaded but unextracted tarball is pointed out", "extract it" in where, where)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_notebook_stream():
    """The cell a stage runs in: progress bars drawn in place instead of one
    line per redraw (the stage-2 log held 937 lines, most of them one bar), and
    a cleanup that a second press of stop cannot cut short -- after stage 1,
    one did, before the servers were confirmed gone."""
    print("\n=== the notebook cell: progress in place, an interrupt-proof cleanup ===")
    import contextlib
    import io
    import signal
    import subprocess
    import sys
    import time

    base = Path(tempfile.mkdtemp())
    work = base / "mnt" / "pasarbench" / "pasarbench"
    (work / "scripts").mkdir(parents=True)
    (work / "scripts" / "gpu_pipeline.sh").write_text("")
    (work / "logs").mkdir()
    saved = {k: os.environ.get(k) for k in ("PASAR_NB_MNT", "PASAR_NB_LINK", "PASAR_NB_VENV",
                                            "HF_HOME", "PYTHON", "VIRTUAL_ENV", "PATH",
                                            "PYTHONUNBUFFERED")}
    os.environ.update(PASAR_NB_MNT=str(base / "mnt"), PASAR_NB_LINK=str(base / "vol"),
                      PASAR_NB_VENV=str(base / "venv"))
    try:
        pn = _script("pasar_notebook")

        class Trickle(io.BytesIO):      # a pipe that hands back a few bytes at a time
            size = 3

            def read1(self, n=-1):
                return super().read1(self.size)
        text = "a 1%|\ra 50%|\rวัน 100%|\nnext→\r\nlast"
        want = [("a 1%|", True), ("a 50%|", True), ("วัน 100%|", False),
                ("next→", False), ("last", False)]
        for size in (1, 2, 3, 4, 5, 7):
            # 1 splits every \r\n and every multi-byte character across reads;
            # the others cut them at every other offset.
            Trickle.size = size
            got = list(pn._lines(Trickle(text.encode())))
            if got != want:
                break
        check("carriage returns are redraws, \\r\\n is a line end, and Thai and '→' "
              "survive, however the reads split them", got == want, (size, got))
        # A bar writes each state after a \r, so without partial=True the line
        # shown lags one state behind -- on a slow bar, for minutes.
        Trickle.size = 1
        seen = list(pn._lines(Trickle(text.encode()), partial=True))
        done = [x for x in seen if x[1] is not None]
        check("…with partial=True a state is shown as it arrives, before the next "
              "one completes it, and the finished lines are unchanged",
              done == want and ("a 50%|", None) in seen
              and seen.index(("a 50%|", None)) < seen.index(("a 50%|", True))
              and all(any(d.startswith(t) for d, _ in done) for t, r in seen if r is None),
              seen[:12])
        pn._gpus = lambda: []
        with contextlib.redirect_stdout(io.StringIO()):
            pn.setup(install=False)
        script = base / "stage.py"
        script.write_text("import sys, time\n"
                "w = sys.stdout.write\n"
                "w('loading\\n')\n"
                "for i in (0, 50, 100): w(f'\\rLoading weights: {i}%|##| {i}/100'); sys.stdout.flush()\n"
                "w('\\n')\n"
                "for n in (25, 50, 75): w(f'    {n}/75 episodes\\n'); sys.stdout.flush(); time.sleep(0.05)\n"
                "w('done\\n')\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pn._stream([sys.executable, "-S", str(script)], "t.log")   # -S: no site noise
        log = (work / "logs" / "t.log").read_text().splitlines()[2:]      # after the header
        check("the log keeps a bar's final state once, every episode count, every line",
              log == ["loading", "Loading weights: 100%|##| 100/100", "    25/75 episodes",
                      "    50/75 episodes", "    75/75 episodes", "done"], log)
        cell = out.getvalue()
        check("the cell redraws in place and shows the episodes as a bar",
              "\rLoading weights: 50%" in cell and "75/75 episodes [" in cell
              and "100%" in cell and cell.rstrip().endswith("done"), repr(cell[-300:]))

        # A stage whose exit handler ignores the first signal, and a user who
        # presses stop again while it runs -- a real SIGINT, which can land in
        # any line of _shutdown, not an exception planted in one call.
        import threading
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import signal, time; signal.signal(signal.SIGINT, "
                                 "signal.SIG_IGN); time.sleep(60)"], start_new_session=True)
        time.sleep(0.3)
        stops, escaped = [], []

        def stop_pressed_again():         # a press during the server check too
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.1)
            stops.append(1)
        pn.stop = stop_pressed_again
        pn._SHUTDOWN = ((signal.SIGINT, 20), (signal.SIGTERM, 20), (signal.SIGKILL, 20))
        before = signal.getsignal(signal.SIGINT)
        press = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))
        out, t0 = io.StringIO(), time.monotonic()
        try:
            with contextlib.redirect_stdout(out):
                press.start()
                pn._shutdown(proc)
        except KeyboardInterrupt:
            escaped.append(1)
        finally:
            press.cancel()
            if proc.poll() is None:
                proc.kill()
        took = time.monotonic() - t0
        check("a second press moves to the next signal instead of skipping the cleanup",
              not escaped and proc.poll() == -signal.SIGTERM and stops == [1] and took < 10,
              (escaped, proc.poll(), stops, round(took, 1)))
        check("…a press while the servers are checked is absorbed too, and the "
              "previous handler is back afterwards",
              not escaped and signal.getsignal(signal.SIGINT) is before)
        check("…and the cell says it is stopping servers",
              "stopping servers" in out.getvalue() and "SIGINT sent" in out.getvalue(),
              out.getvalue())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cuda_home():
    print("\n=== a CUDA compiler new enough for the kernels SGLang builds ===")
    import contextlib
    import io
    import re
    root = Path(__file__).resolve().parent.parent
    ch = _script("cuda_home")
    base = Path(tempfile.mkdtemp())

    def toolkit(where: Path, release: str) -> Path:
        (where / "bin").mkdir(parents=True)
        (where / "lib").mkdir()
        nvcc = where / "bin" / "nvcc"
        nvcc.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {release}, V{release}.1'\n")
        nvcc.chmod(0o755)
        (where / "lib" / "libcudart.so.13").write_text("")
        return where

    pip_home = toolkit(base / "site" / "nvidia" / "cu13", "13.0")
    old = toolkit(base / "usr-local-cuda-12.4", "12.4")
    saved = {k: os.environ.get(k) for k in ("PASAR_PIP_NVIDIA_ROOTS", "CUDA_HOME", "CUDA_PATH",
                                            "PATH")}

    def run(**env):
        os.environ.update({k: v for k, v in env.items() if v is not None})
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                ch.main([])
            return out.getvalue().strip(), err.getvalue()
        except SystemExit as e:
            return None, str(e)
    try:
        os.environ["PATH"] = "/usr/bin:/bin"          # no other nvcc on PATH
        got, note = run(PASAR_PIP_NVIDIA_ROOTS=str(base / "site" / "nvidia"),
                        CUDA_HOME=str(old), CUDA_PATH=None)
        check("the pip compiler beside torch is chosen over an older CUDA_HOME (12.4, the "
              "kind that stopped the notebook's agent server)", got == str(pip_home), note)
        check("…and given the lib64 and libcudart.so its users link against",
              (pip_home / "lib64").is_symlink() and (pip_home / "lib" / "libcudart.so").is_symlink()
              and "linked lib64 -> lib" in note, note)
        got2, note2 = run(PASAR_PIP_NVIDIA_ROOTS=str(base / "site" / "nvidia"),
                          CUDA_HOME=str(old))
        check("…and linking again changes nothing", got2 == str(pip_home) and "linked" not in note2,
              note2)
        got, why = run(PASAR_PIP_NVIDIA_ROOTS="", CUDA_HOME=str(old))
        check("with only an old compiler it stops before any model loads, and says what to "
              "install and what it found", got is None and "12.9 or newer" in why
              and "cuda-toolkit[nvcc,cccl]" in why and "nvcc 12.4" in why, why)
        new = toolkit(base / "usr-local-cuda-13.1", "13.1")
        got, note = run(PASAR_PIP_NVIDIA_ROOTS="", CUDA_HOME=str(new))
        check("a machine's own new-enough toolkit is used as it is",
              got == str(new) and not (new / "lib64").exists(), note)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    pipe = (root / "scripts" / "gpu_pipeline.sh").read_text()
    check("every GPU stage finds the compiler, tries it on a test kernel and exports it "
          "before starting a server",
          'scripts/cuda_home.py --check' in pipe and 'export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"'
          in pipe)
    installers = {n: (root / "scripts" / n).read_text()
                  for n in ("setup_node.sh", "modal_pipeline.py", "pasar_notebook.py")}
    check("a rented node, a Modal job and a notebook all install the compiler at torch's "
          "CUDA version", all("cuda-toolkit[nvcc,cccl]" in t and re.search(
              r"tokenizers[\"|' ,\s]+cuda-toolkit", t) for t in installers.values()),
          {n: "cuda-toolkit[nvcc,cccl]" in t for n, t in installers.items()})
    check("…and the Modal image has the C++ compiler nvcc needs for host code",
          '"build-essential"' in installers["modal_pipeline.py"])
    serve = (root / "scripts" / "serve_sglang.sh").read_text()
    check("DeepGEMM, which neither model uses, is off unless asked for",
          'export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"' in serve)


def test_push_script():
    """push_to_github.sh once ran `rm -rf traces data logs checkpoints` in the
    folder it was started from -- the user's working copy, runs and all -- and
    committed whatever was left, a .env file included. A local bare repo stands
    in for GitHub."""
    print("\n=== push_to_github.sh keeps runs and keys on your disk, off GitHub ===")
    import shutil
    if not (shutil.which("bash") and shutil.which("git")):
        skip("the push script", "no bash or git")
        return
    root = Path(__file__).resolve().parent.parent
    work = Path(tempfile.mkdtemp())
    (work / ".gitconfig").write_text(
        "[user]\n\tname = Test Author\n\temail = test@pasar.invalid\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(HOME=str(work), XDG_CONFIG_HOME=str(work), GIT_CONFIG_NOSYSTEM="1",
               GIT_CONFIG_GLOBAL=str(work / ".gitconfig"))
    remote = work / "github.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True, env=env)
    repo = work / "pasarbench"
    runs = ["traces/A/full/x.jsonl", "data/labels/y.jsonl", "checkpoints/rft-A/a.bin",
            "logs/gpu.txt", ".env", ".env.local"]

    def unpack(gitignore: str) -> None:
        shutil.rmtree(repo, ignore_errors=True)
        (repo / "scripts").mkdir(parents=True)
        shutil.copy(root / "scripts" / "push_to_github.sh", repo / "scripts")
        (repo / ".gitignore").write_text(gitignore)
        (repo / "run_tests.sh").write_text("echo all suites stubbed\n")
        (repo / "README.md").write_text("readme\n")
        (repo / "tests" / "fixtures").mkdir(parents=True)
        (repo / "tests" / "fixtures" / "f.jsonl").write_text("{}\n")
        for p in runs:
            (repo / p).parent.mkdir(parents=True, exist_ok=True)
            (repo / p).write_text("DEEPSEEK_API_KEY=sk-test\n" if ".env" in p else "{}\n")

    def push():
        r = subprocess.run(["bash", "scripts/push_to_github.sh", str(remote)], cwd=repo,
                           input="REPLACE\n", capture_output=True, text=True, env=env,
                           timeout=120)
        return r.returncode, r.stdout + r.stderr

    def on_github() -> tuple[str, list[str]]:
        head = subprocess.run(["git", "--git-dir", str(remote), "rev-parse", "-q", "--verify",
                               "main"], capture_output=True, text=True, env=env).stdout.strip()
        files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only",
                                "main"], capture_output=True, text=True, env=env).stdout.split()
        return head, files

    shipped = (root / ".gitignore").read_text()
    unpack(shipped)
    code, out = push()
    head, files = on_github()
    check("it pushes the code -- and a test fixture the *.jsonl rule exempts",
          code == 0 and {"README.md", "tests/fixtures/f.jsonl"} <= set(files), out[-800:])
    check("…but no .env file, trace, label, or checkpoint",
          files and not any(f.startswith((".env", "traces/", "data/", "checkpoints/"))
                            for f in files), files)
    check("…and every run and key is still on disk afterwards",
          all((repo / p).exists() for p in runs), [p for p in runs if not (repo / p).exists()])

    unpack("".join(l + "\n" for l in shipped.splitlines() if not l.startswith(".env")))
    code, out = push()
    check("a .gitignore that would let .env through: it refuses before asking, and "
          "GitHub keeps what it had", code != 0 and "REFUSING" in out
          and "Type REPLACE" not in out and on_github()[0] == head, out[-800:])
    unpack("".join(l + "\n" for l in shipped.splitlines() if l != ".env.*"))
    code, out = push()
    check("…and one that misses .env.local is caught in the snapshot itself",
          code != 0 and ".env.local" in out and on_github()[0] == head, out[-800:])


def main() -> int:
    test_split()
    test_family_guard()
    test_sweep_collection()
    test_examples()
    test_balance()
    test_check_command()
    test_served_models()
    test_serve_scripts()
    test_grpo_scripts()
    test_gpu_plan()
    test_markup_audit()
    test_wrappers()
    test_train_smoke()
    test_grpo_build()
    test_grpo_train()
    test_qwen38_turns()
    test_qwen35_lora()
    test_modal_runner()
    test_notebook_helper()
    test_notebook_stream()
    test_cuda_home()
    test_push_script()
    tail =f", {len(SKIP)} skipped" if SKIP else ""
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed{tail}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
