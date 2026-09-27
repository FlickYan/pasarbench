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
    check("Qwen3-8B is qwen", model_family("Qwen/Qwen3-8B") == "qwen")
    check("SeaLLMs and Sailor count as qwen underneath",
          model_family("SeaLLMs/SeaLLMs-v3-7B-Chat") == "qwen"
          and model_family("sail/Sailor2-8B-Chat") == "qwen")
    check("Gemma-SEA-LION counts as gemma",
          model_family("aisingapore/Gemma-SEA-LION-v4-27B-IT") == "gemma")
    try:
        check_agent_simulator("Qwen/Qwen3-8B", "qwen3.8-flash")
        ok = False
    except SystemExit as e:
        ok = "both qwen" in str(e)
    check("a Qwen agent with a Qwen customer is refused", ok)
    check("the published pairing (DeepSeek agent, Qwen customer) passes",
          "agent deepseek, simulator qwen" in check_agent_simulator("deepseek-v4-pro",
                                                                    "qwen3.8-flash"))
    from pasarbench.models import SIM_DEFAULT
    check("…as does Qwen with the pipeline's customer, Gemma 4",
          "simulator gemma" in check_agent_simulator("Qwen/Qwen3-8B", SIM_DEFAULT))
    check("an adapter requested SGLang's way carries its base's family",
          model_family("Qwen/Qwen3-8B:pasar-rft-A") == "qwen")
    try:
        check_agent_simulator("Qwen/Qwen3-8B:pasar-rft-A", "qwen3.8-flash")
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
          "--fold-models A=Qwen/Qwen3-8B:pasar-rft-A,B=Qwen/Qwen3-8B:pasar-rft-B" in ev
          and "--folds" in ev and "--temperature 0.0" in ev, ev)
    check("thinking is off in every step, by the same flag",
          all('"enable_thinking": false' in c for c in (base, coll, ev)))
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
    check("<base>:<adapter> names for loaded adapters pass", "every fold model" in check_served(url, good))
    check("a bare adapter name is refused on SGLang, which would answer with the base model",
          refused(url, ["pasar-rft-A", "pasar-rft-B"], "BASE model"))
    check("an adapter the server never loaded is refused",
          refused(url, ["Qwen/Qwen3-8B:pasar-rft-C"], "would not answer"))
    check("…as is a base the server does not serve",
          refused(url, ["Qwen/Qwen3-32B:pasar-rft-A"], "would not answer"))
    srv.shutdown()

    srv, url = serve([{**base, "owned_by": "vllm"},
                      *[{**c, "owned_by": "vllm"} for c in lora]])
    check("vLLM routes a bare adapter name to the adapter, so there it passes",
          "every fold model" in check_served(url, ["pasar-rft-A", "pasar-rft-B"]))
    srv.shutdown()
    check("no server, no evaluation", refused(url, good, "is the agent server up"))


def test_serve_scripts():
    print("\n=== the serve script, the pipeline and the wrappers agree ===")
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
           if k not in ("SIM_MODEL", "SIM_TOKENIZER", "AGENT_MODEL", "SGLANG_ARGS")}

    def cmd(mode, **extra):
        r = subprocess.run(["bash", str(root / "scripts" / "serve_sglang.sh"), mode],
                           capture_output=True, text=True, cwd=work,
                           env={**env, "DRY_RUN": "1", **extra})
        return r.stdout if r.returncode == 0 else f"exit {r.returncode}: {r.stderr}"

    agent, sim, rft = cmd("agent"), cmd("sim"), cmd("rft")
    check("the agent: Qwen3-8B on GPU 0, port 8000, Qwen's tool-call parser",
          "CUDA_VISIBLE_DEVICES=0" in agent and "--model-path Qwen/Qwen3-8B" in agent
          and "--port 8000" in agent and "--tool-call-parser qwen" in agent, agent)
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
    check("the pipeline serves with SGLang and defaults to the same customer",
          "serve_sglang.sh" in pipe and f'SIM_MODEL="${{SIM_MODEL:-{SIM_DEFAULT}}}"' in pipe
          and tr.SIM_DEFAULT == SIM_DEFAULT)
    check("no script still starts vLLM",
          not (root / "scripts" / "serve_vllm.sh").exists()
          and "vllm serve" not in pipe
          and "vllm" not in (root / "scripts" / "setup_node.sh").read_text().lower())
    shutil.rmtree(work / "checkpoints")
    check("rft refuses to start without both adapters", cmd("rft").startswith("exit 1"))


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


def main() -> int:
    test_split()
    test_family_guard()
    test_sweep_collection()
    test_examples()
    test_check_command()
    test_served_models()
    test_serve_scripts()
    test_markup_audit()
    test_wrappers()
    test_train_smoke()
    tail = f", {len(SKIP)} skipped" if SKIP else ""
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed{tail}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
