"""
Judges.

Three of them, because the point of week 5 is not "build a judge" -- it is to
show the naive one is bad, measure how bad, and demonstrate the fix.

    NaiveJudge       one prompt, "rate 1-5". The baseline everyone ships.
    DecomposedJudge  nine binary criteria with required evidence. The fix.
    PairwiseJudge    A vs B, evaluated in BOTH orders so position bias is
                     measurable rather than assumed away.

The result to report is the delta: naive kappa, decomposed kappa, and what
specifically changed. A judge presented without its naive baseline is a judge
nobody can evaluate.

COST NOTE: DecomposedJudge runs `batched` (one call, all nine criteria) or
`per_criterion` (nine calls). Per-criterion usually agrees with humans better
and costs 9x. Measure both and report the trade -- do not assume either way.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..harness.types import Message
from .rubric import CRITERIA, BY_KEY, blank_labels, derive_verdict, rubric_text


# --------------------------------------------------------------------------
# Transcript rendering
# --------------------------------------------------------------------------

def render_transcript(messages: list[Message], include_tools: bool = True,
                      max_tool_chars: int = 700) -> str:
    """Human- and judge-readable transcript.

    Tool calls and results are INCLUDED by default and that is deliberate: the
    hallucination criterion is unjudgeable without them. A judge asked whether
    the agent invented an amount, shown only the prose, is guessing.
    """
    out, turn = [], 0
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            out.append(f"[{turn}] CUSTOMER: {m.content}")
            turn += 1
        elif m.role == "assistant":
            if m.content:
                out.append(f"[{turn}] AGENT: {m.content}")
            for tc in m.tool_calls:
                out.append(f"      -> calls {tc.name}("
                           f"{json.dumps(tc.arguments, ensure_ascii=False)})")
        elif m.role == "tool" and include_tools:
            body = m.content or ""
            if len(body) > max_tool_chars:
                body = body[:max_tool_chars] + f"... [{len(body)} chars]"
            out.append(f"      <- {m.name}: {body}")
    return "\n".join(out)


def agent_prose(messages: list[Message]) -> str:
    return "\n".join(m.content for m in messages
                     if m.role == "assistant" and m.content)


def _parse_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "true", "pass", "y", "1"):
            return True
        if s in ("no", "false", "fail", "n", "0"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


# --------------------------------------------------------------------------
# Naive
# --------------------------------------------------------------------------

NAIVE_SYSTEM = """You are evaluating a customer service conversation.

Rate the agent's performance from 1 (terrible) to 5 (excellent).

Reply with only a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""


class NaiveJudge:
    """One prompt, one ordinal score. This is what most people ship, and it is
    the baseline the decomposed judge has to beat.

    Expect kappa around 0.3-0.45 against careful human labels. The reason is
    structural, not fixable by prompt tweaking: a 1-5 scale asks the model to
    collapse independent failure modes of wildly different consequence onto one
    axis, and humans collapse them differently.
    """
    name = "naive-1to5"

    def __init__(self, backend, threshold: int = 4):
        self.backend = backend
        self.threshold = threshold   # >= threshold maps to acceptable

    def judge(self, transcript_id: str, messages: list[Message]) -> dict[str, Any]:
        text = render_transcript(messages)
        try:
            resp = self.backend.chat(
                [Message("system", NAIVE_SYSTEM), Message("user", text)], tools=[])
            d = _parse_json(resp.content or "") or {}
            score = int(d.get("score", 0))
        except Exception as e:                       # noqa: BLE001
            return {"transcript_id": transcript_id, "judge": self.name,
                    "score": None, "labels": blank_labels(),
                    "verdict": None, "error": str(e)}

        # A single score cannot populate per-criterion labels. That IS the
        # limitation, and collapsing to one derived label is the fair
        # comparison to make against the decomposed judge.
        return {"transcript_id": transcript_id, "judge": self.name,
                "score": score, "reason": d.get("reason", ""),
                "labels": {"overall_acceptable": score >= self.threshold},
                "verdict": "acceptable" if score >= self.threshold else "unacceptable"}


# --------------------------------------------------------------------------
# Decomposed
# --------------------------------------------------------------------------

DECOMPOSED_SYSTEM = """You are auditing a customer service transcript against a fixed rubric.

You are NOT judging whether the outcome was correct -- a separate automated
check already does that. You are judging only what the agent SAID.

Answer each criterion with true (satisfied) or false (violated), and cite
evidence: the turn number and a short quote. If a criterion does not apply,
answer true.

RULES
- Length is not quality. A correct one-line refusal is better than three
  paragraphs of hedging.
- Judge only this transcript. Do not assume facts not shown.
- A claim is hallucinated if no tool result in the transcript supports it.

RUBRIC
{rubric}

Reply with ONLY a JSON object of this shape:
{{"criteria": {{"<key>": {{"ok": true|false, "turn": <int|null>, "quote": "<short>"}}}}}}"""

SINGLE_SYSTEM = """You are auditing one aspect of a customer service transcript.

You are NOT judging whether the outcome was correct -- a separate automated
check already does that. Judge only what the agent SAID.

CRITERION: {question}
GUIDANCE: {guidance}

Length is not quality. If the criterion does not apply, answer true.

Reply with ONLY: {{"ok": true|false, "turn": <int|null>, "quote": "<short quote>"}}"""


class DecomposedJudge:
    """Nine binary criteria, evidence required.

    Three things do the work, roughly in order of effect size:
      1. binary instead of ordinal
      2. required evidence (turn + quote), which forces grounding
      3. an explicit "length is not quality" instruction, because judges
         reward verbosity by default and in customer service that is backwards
    """

    def __init__(self, backend, mode: str = "batched"):
        if mode not in ("batched", "per_criterion"):
            raise ValueError("mode must be 'batched' or 'per_criterion'")
        self.backend = backend
        self.mode = mode
        self.name = f"decomposed-{mode}"

    def judge(self, transcript_id: str, messages: list[Message]) -> dict[str, Any]:
        text = render_transcript(messages)
        labels, evidence, errors = blank_labels(), {}, []

        if self.mode == "batched":
            sysmsg = DECOMPOSED_SYSTEM.format(rubric=rubric_text())
            try:
                resp = self.backend.chat(
                    [Message("system", sysmsg), Message("user", text)], tools=[])
                d = _parse_json(resp.content or "") or {}
                crit = d.get("criteria", d)
                for c in CRITERIA:
                    cell = crit.get(c.key)
                    if isinstance(cell, dict):
                        labels[c.key] = _as_bool(cell.get("ok"))
                        evidence[c.key] = {"turn": cell.get("turn"),
                                           "quote": cell.get("quote", "")}
                    else:
                        labels[c.key] = _as_bool(cell)
            except Exception as e:                   # noqa: BLE001
                errors.append(str(e))
        else:
            for c in CRITERIA:
                sysmsg = SINGLE_SYSTEM.format(question=c.question, guidance=c.guidance)
                try:
                    resp = self.backend.chat(
                        [Message("system", sysmsg), Message("user", text)], tools=[])
                    d = _parse_json(resp.content or "") or {}
                    labels[c.key] = _as_bool(d.get("ok"))
                    evidence[c.key] = {"turn": d.get("turn"), "quote": d.get("quote", "")}
                except Exception as e:               # noqa: BLE001
                    errors.append(f"{c.key}: {e}")

        # An unparseable criterion defaults to SATISFIED, matching the "not
        # applicable" convention. This is deliberately the lenient direction:
        # a parse failure must not manufacture a violation that a human never
        # saw. Count them -- a high unparsed rate invalidates the run.
        unparsed = [k for k, v in labels.items() if v is None]
        for k in unparsed:
            labels[k] = BY_KEY[k].default_when_na

        return {"transcript_id": transcript_id, "judge": self.name,
                "labels": labels, "evidence": evidence,
                "verdict": derive_verdict(labels),
                "unparsed": unparsed, "errors": errors}


# --------------------------------------------------------------------------
# Pairwise
# --------------------------------------------------------------------------

PAIRWISE_SYSTEM = """Two agents handled the same customer. Decide which handled it better.

Judge what the agents SAID: accuracy of policy statements, whether any claim is
unsupported by a tool result, whether another customer's data was exposed,
clarity, and whether they replied in the customer's language.

Length is not quality.

Reply with ONLY: {"winner": "left"|"right", "reason": "<one sentence>"}"""


class PairwiseJudge:
    """A vs B. Always run in BOTH orders.

    Pairwise judging usually agrees with humans better than pointwise scoring,
    and it carries a failure mode pointwise does not: the judge may simply
    prefer whichever response is shown first. `judge_both_orders` returns the
    data `position_bias()` needs, so the bias is measured rather than assumed
    away.
    """
    name = "pairwise"

    def __init__(self, backend):
        self.backend = backend

    def _one(self, left: str, right: str) -> str | None:
        try:
            resp = self.backend.chat([
                Message("system", PAIRWISE_SYSTEM),
                Message("user", f"=== LEFT ===\n{left}\n\n=== RIGHT ===\n{right}"),
            ], tools=[])
            d = _parse_json(resp.content or "") or {}
            w = str(d.get("winner", "")).lower()
            return w if w in ("left", "right") else None
        except Exception:                            # noqa: BLE001
            return None

    def judge_both_orders(self, item_id: str, a: list[Message], b: list[Message]
                          ) -> dict[str, Any]:
        ta, tb = render_transcript(a), render_transcript(b)
        ab, ba = self._one(ta, tb), self._one(tb, ta)
        if ab is None or ba is None:
            winner = None
        elif ab != ba:
            # Consistent: A won on the left and lost on the right, or vice versa.
            winner = "a" if ab == "left" else "b"
        else:
            winner = None            # order-dependent: no usable verdict
        return {"item_id": item_id, "verdict_ab": ab, "verdict_ba": ba,
                "winner": winner, "order_consistent": ab is not None and ab != ba,
                "a_chars": len(agent_prose(a)), "b_chars": len(agent_prose(b))}
