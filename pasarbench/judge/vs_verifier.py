"""
Judges scored against the verifier, compared across what the judge was shown.

Reads the files `scripts/run_judges.py --traces` writes, one per information
condition, and lays them side by side. The condition is in the file name:

    <run>__<cell>.jsonl                     transcript only
    <run>__<cell>__payloads.jsonl           + the tool results the agent saw
    <run>__<cell>__policy.jsonl             + the policy it was bound by
    <run>__<cell>__payloads__policy.jsonl   + both

Every number is computed here from those files -- nothing is carried over by
hand -- and the per-trap table is what the aggregate hides: which failures a
judge let through, under every view of the evidence it was given.

    python -m pasarbench.judge.vs_verifier [data/judge_vs_verifier]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .agreement import agreement

# Longest suffix first: "__payloads__policy" also ends in "__policy".
CONDITIONS = [("__payloads__policy", "+ tool results + policy"),
              ("__payloads", "+ tool results"),
              ("__policy", "+ policy"),
              ("", "transcript only")]
ORDER = ["transcript only", "+ tool results", "+ policy", "+ tool results + policy"]
JUDGES = [("naive", "naive_ok"), ("decomposed", "dec_ok")]


def load_runs(root: str | Path = "data/judge_vs_verifier"
              ) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """{cell: {condition: rows}} for every saved judge-vs-verifier run."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for f in sorted(Path(root).glob("*.jsonl")):
        for suffix, label in CONDITIONS:
            if f.stem.endswith(suffix):
                cell = f.stem[: len(f.stem) - len(suffix)] if suffix else f.stem
                out[cell][label] = [json.loads(l) for l in
                                    f.read_text().splitlines() if l.strip()]
                break
    return dict(out)


def _acc(rows: list[dict], key: str) -> str:
    v = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(v)}/{len(v)}"


def _kappa(rows: list[dict], key: str) -> tuple[float, tuple[float, float] | None] | None:
    pairs = [(r["verifier_passed"], r[key]) for r in rows if r.get(key) is not None]
    if not pairs:
        return None
    a = agreement([x for x, _ in pairs], [y for _, y in pairs], bootstrap=1000)
    return a.kappa, a.ci95


def markdown(runs: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    out: list[str] = []
    for cell, conds in sorted(runs.items()):
        labels = [c for c in ORDER if c in conds]
        any_rows = conds[labels[-1]]
        passed = [r for r in any_rows if r["verifier_passed"]]
        failed = [r for r in any_rows if not r["verifier_passed"]]
        claimed = [r for r in failed if r.get("false_claims")]
        out.append(f"#### `{cell}` — {len(any_rows)} episodes: the verifier "
                   f"passed {len(passed)} and failed {len(failed)}; {len(claimed)} "
                   f"of the failures claim{'s' if len(claimed) == 1 else ''} an "
                   f"action that never happened\n")
        out.append("| judge | what it saw | correct accepted | failures accepted | "
                   "false claims accepted | kappa vs verifier |")
        out.append("|---|---|---|---|---|---|")
        best = None
        for jname, key in JUDGES:
            for c in labels:
                rows = conds[c]
                p = [r for r in rows if r["verifier_passed"]]
                f = [r for r in rows if not r["verifier_passed"]]
                cl = [r for r in f if r.get("false_claims")]
                k = _kappa(rows, key)
                ks = (f"{k[0]:+.2f} [{k[1][0]:+.2f}, {k[1][1]:+.2f}]"
                      if k and k[1] else (f"{k[0]:+.2f}" if k else "-"))
                out.append(f"| {jname} | {c} | {_acc(p, key)} | {_acc(f, key)} | "
                           f"{_acc(cl, key)} | {ks} |")
                if k and k[1] and k[1][0] > 0 and (best is None or k[0] > best[0]):
                    best = (k[0], k[1], jname, c, key)

        if best:
            rows = conds[best[3]]
            f = [r for r in rows if not r["verifier_passed"]]
            let_through = [r for r in f if r.get(best[4])]
            out.append(f"\nBest configuration: **{best[2]}, {best[3]}** — kappa "
                       f"{best[0]:+.2f} [{best[1][0]:+.2f}, {best[1][1]:+.2f}], and it "
                       f"still accepts {len(let_through)} of the {len(f)} episodes "
                       f"the database fails.")
        else:
            out.append("\nNo configuration agrees with the verifier distinguishably "
                       "from chance (every kappa interval includes zero).")

        # Per trap: which failures got through, under each view.
        by_trap: dict[str, dict[str, list[int]]] = defaultdict(dict)
        for c in labels:
            for r in conds[c]:
                if r["verifier_passed"]:
                    continue
                cell_ = by_trap[r.get("trap") or "?"].setdefault(c, [0, 0, 0])
                cell_[0] += 1
                cell_[1] += bool(r.get("naive_ok"))
                cell_[2] += bool(r.get("dec_ok"))
        if by_trap:
            out.append("\nFailures each judge let through, by trap "
                       "(naive / decomposed accepted, of the failures):\n")
            out.append("| trap | " + " | ".join(labels) + " |")
            out.append("|---|" + "---|" * len(labels))
            for trap in sorted(by_trap, key=lambda t: -max(v[0] for v in by_trap[t].values())):
                cells = []
                for c in labels:
                    n, a, d = by_trap[trap].get(c, [0, 0, 0])
                    cells.append(f"{a} / {d} of {n}" if n else "-")
                out.append(f"| `{trap}` | " + " | ".join(cells) + " |")

        # The false claims, under every view: what each judge did, and why.
        ids = sorted({r["transcript_id"] for c in labels for r in conds[c]
                      if not r["verifier_passed"] and r.get("false_claims")})
        if ids:
            out.append("\nThe transcripts that claim an action the database never "
                       "saw, under every view:\n")
            for tid in ids:
                claim = next((r for c in labels for r in conds[c]
                              if r["transcript_id"] == tid), {})
                out.append(f"- `{tid}` ({', '.join(claim.get('false_claims', []))}): "
                           f"\"…{(claim.get('claim_quotes') or [''])[0][:110]}…\"")
                for c in labels:
                    r = next((x for x in conds[c] if x["transcript_id"] == tid), None)
                    if not r:
                        continue
                    flags = ", ".join(r.get("dec_violations") or []) or "nothing"
                    line = (f"  - {c}: naive {r.get('naive_score')} → "
                            f"{'accepted' if r.get('naive_ok') else 'rejected'}; "
                            f"decomposed {'accepted' if r.get('dec_ok') else 'rejected'}"
                            f" (flagged {flags})")
                    if r.get("naive_reason"):
                        line += f". Naive said: \"{r['naive_reason'][:160]}\""
                    out.append(line)
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/judge_vs_verifier"
    runs = load_runs(root)
    print(markdown(runs) if runs else f"no judge-vs-verifier runs in {root}")
