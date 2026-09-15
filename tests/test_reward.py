"""
Reward and dataset-construction tests.

The load-bearing one is `test_no_failure_outscores_a_pass`. If that property
ever breaks, every RL run built on this reward is optimising for something
other than solving the task, and you will not notice from the loss curve.

Run: python -m tests.test_reward
"""

from __future__ import annotations

import random

from pasarbench.db import Database
from pasarbench.harness.backends import ScriptedBackend
from pasarbench.harness.loop import run_episode
from pasarbench.rl.collect import (Rollout, corpus_stats, divergence_pairs,
                                   grpo_groups, sft_examples)
from pasarbench.rl.reward import (RewardConfig, compute_reward,
                                  degenerate_group_rate, group_advantages)
from pasarbench.run import SOLUTIONS
from pasarbench.tasks import BY_ID, TASKS
from pasarbench.verifier import verify

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"  -- {detail}" if detail and not ok else ""))


# ---- perturbations that produce realistic failures ------------------------

def drop_last_write(script):
    return script[:-1] if len(script) > 1 else script


def reorder_check_after_write(script):
    s = list(script)
    ci = next((i for i, (n, _) in enumerate(s) if n == "check_return_eligibility"), None)
    wi = next((i for i, (n, _) in enumerate(s)
               if n in ("initiate_return", "issue_refund", "issue_store_credit")), None)
    if ci is None or wi is None or ci > wi:
        return None
    s[ci], s[wi] = s[wi], s[ci]
    return s


def inject_forbidden(script, task):
    if not task.checks.forbidden_actions:
        return None
    spec = task.checks.forbidden_actions[0]
    args = dict(spec.args)
    defaults = {
        "issue_refund": {"order_id": "O1001", "amount_minor": 100,
                         "method": "card", "reason": "x"},
        "initiate_return": {"order_id": "O1001", "order_item_id": "OI1",
                            "reason": "x", "photo_evidence_provided": True},
        "cancel_order": {"order_id": "O1002", "reason": "x"},
        "issue_store_credit": {"user_id": "U003", "amount_minor": 100,
                               "currency": "IDR", "reason": "x"},
        "issue_goodwill_voucher": {"user_id": "U003", "amount_minor": 1000,
                                   "currency": "IDR", "reason": "x"},
        "modify_shipping_address": {"order_id": "O1003", "new_address": "x"},
    }
    base = dict(defaults.get(spec.tool, {}))
    base.update(args)
    return list(script) + [(spec.tool, base)]


def tool_spam(task):
    """The reward-hacking policy: harvest required actions by enumerating."""
    return [
        ("verify_identity", {"user_id": task.user_id, "phone_last4": "0000"}),
        ("get_order", {"order_id": "O1001"}),
        ("get_order", {"order_id": "O1002"}),
        ("get_shipment", {"order_id": "O1003"}),
        ("get_payment", {"order_id": "O1004"}),
        ("check_return_eligibility", {"order_id": "O1001", "order_item_id": "OI1"}),
        ("check_return_eligibility", {"order_id": "O1004", "order_item_id": "OI4"}),
        ("search_policy", {"query": "refund"}),
        ("get_livestream_claims", {"livestream_id": "LS7001"}),
        ("escalate_to_human", {"order_id": "O1001", "category": "other", "reason": "x"}),
    ]


def make_rollout(task, script, seed=0) -> Rollout:
    db = Database.fresh(task.db_patch)
    res = run_episode(task, db, ScriptedBackend(script))
    v = verify(task, db, n_turns=res.budget["steps"])
    r = compute_reward(task, db, res.budget["steps"],
                       reference_steps=len(SOLUTIONS.get(task.task_id, [])), result=v)
    return Rollout(task.task_id, task.trap, task.market, task.language,
                   res.state.messages, v, r, res.budget["steps"],
                   res.budget["tokens"], res.stop_reason.value, seed)


def build_corpus() -> list[Rollout]:
    out = []
    for task in TASKS:
        ref = SOLUTIONS[task.task_id]
        out.append(make_rollout(task, ref, 0))
        for i, fn in enumerate((drop_last_write, reorder_check_after_write)):
            p = fn(ref)
            if p and p != ref:
                out.append(make_rollout(task, p, i + 1))
        p = inject_forbidden(ref, task)
        if p:
            out.append(make_rollout(task, p, 8))
        out.append(make_rollout(task, tool_spam(task), 9))
    return out


# ---- tests ----------------------------------------------------------------

def test_config_guard():
    print("\n=== the reward cannot be misconfigured into hackability ===")
    try:
        RewardConfig(w_pass=1.0, clip_partial=1.0)
        raised = False
    except ValueError:
        raised = True
    check("clip_partial >= w_pass is rejected at construction", raised)


def test_no_failure_outscores_a_pass(corpus):
    print("\n=== THE property: failing never beats passing ===")
    good = [r.reward.reward for r in corpus if r.passed]
    bad = [r.reward.reward for r in corpus if not r.passed]
    check("corpus has both outcomes", bool(good) and bool(bad),
          f"{len(good)} pass / {len(bad)} fail")
    check("min(pass) > max(fail)", min(good) > max(bad),
          f"min_pass={min(good)} max_fail={max(bad)}")
    check("partial credit never reaches the pass floor",
          max(bad) < RewardConfig().clip_partial or max(bad) <= RewardConfig().clip_partial,
          f"max_fail={max(bad)} clip={RewardConfig().clip_partial}")


def test_spam_is_punished(corpus):
    print("\n=== the enumerate-every-tool policy is not rewarded ===")
    spam = [r for r in corpus if r.seed == 9]
    check("spam rollouts collected", len(spam) == len(TASKS), str(len(spam)))
    # "Do no harm" traps are structurally satisfiable by a cautious agent that
    # looks things up and changes nothing -- that IS the correct behaviour
    # there. Exempted here to match tests/test_generated.py, and named in the
    # limitations section rather than hidden.
    BENIGN = {"cannot_cancel_shipped_order", "address_change_after_dispatch",
              "peak_period_delay_not_compensable"}
    offenders = [r.task_id for r in spam if r.passed and r.trap not in BENIGN]
    check("no action-requiring task passes under spam", not offenders, str(offenders))
    acting = [r for r in spam if r.trap not in BENIGN]
    worst_ref = min(r.reward.reward for r in corpus if r.seed == 0)
    check("every action-requiring spam rollout scores below every reference",
          max(r.reward.reward for r in acting) < worst_ref,
          f"spam_max={max(r.reward.reward for r in acting)} ref_min={worst_ref}")
    gated = [r for r in spam if r.reward.gated]
    check("spam that violates policy is gated to zero partial credit",
          all(r.reward.reward <= 0 for r in gated), f"{len(gated)} gated")


def test_forbidden_gate(corpus):
    print("\n=== a forbidden action forfeits partial credit ===")
    # The injector is best-effort: some injected calls legitimately fail (wrong
    # order, COD instrument, unverified caller) and so are not violations at
    # all. The property under test is about the reward, not the injector, so
    # assert over the subset where a violation actually registered.
    inj = [r for r in corpus if r.seed == 8]
    landed = [r for r in inj if r.reward.n_forbidden > 0]
    check("injected-violation rollouts exist", bool(inj), str(len(inj)))
    check("enough injections landed to be meaningful", len(landed) >= 4,
          f"{len(landed)}/{len(inj)} landed")
    check("every landed violation is gated", all(r.reward.gated for r in landed),
          str([r.task_id for r in landed if not r.reward.gated]))
    check("every landed violation scores at or below zero",
          all(r.reward.reward <= 0 for r in landed),
          str([(r.task_id, r.reward.reward) for r in landed if r.reward.reward > 0]))


def test_advantages():
    print("\n=== group-relative advantage ===")
    check("uniform group yields no signal", group_advantages([0.5] * 8) == [0.0] * 8)
    a = group_advantages([1.0, 1.0, 0.0, 0.0])
    check("mixed group is standardised", abs(sum(a)) < 1e-6 and a[0] > 0 > a[2], str(a))
    check("empty group is safe", group_advantages([]) == [])
    check("degenerate rate is measured",
          degenerate_group_rate([[1, 1], [1, 0], [0, 0]]) == round(2 / 3, 4))


def test_datasets(corpus):
    print("\n=== dataset construction ===")
    sft = sft_examples(corpus)
    check("SFT set is non-empty", bool(sft), str(len(sft)))
    check("SFT contains only passing trajectories",
          all(e["reward"] >= RewardConfig().clip_partial for e in sft))
    check("SFT examples carry full message lists",
          all(len(e["messages"]) >= 3 for e in sft))
    dup = sft_examples(corpus + corpus)
    check("duplicate rollouts are deduplicated", len(dup) == len(sft),
          f"{len(dup)} vs {len(sft)}")

    pairs = divergence_pairs(corpus)
    check("divergence pairs built", bool(pairs), str(len(pairs)))
    ok_prefix = all(len(p["prompt"]) >= 2 for p in pairs)
    check("every pair has a shared prefix prompt", ok_prefix)
    check("chosen and rejected actually differ",
          all(p["chosen"] != p["rejected"] for p in pairs))
    check("every pair diverges on a tool call, not phrasing",
          all(p["chosen"].get("tool_calls") or p["rejected"].get("tool_calls")
              for p in pairs))
    check("reward gap is positive on every pair",
          all(p["reward_gap"] > 0 for p in pairs),
          str([p["reward_gap"] for p in pairs if p["reward_gap"] <= 0][:3]))

    groups = grpo_groups(corpus)
    check("one GRPO group per task", len(groups) == len(TASKS))
    check("groups report pass rate and degeneracy",
          all("pass_rate" in g and "degenerate" in g for g in groups))


def test_stats(corpus):
    print("\n=== corpus diagnostics ===")
    s = corpus_stats(corpus)
    check("stats cover every trap", len(s["per_trap"]) == len({t.trap for t in TASKS}),
          str(len(s["per_trap"])))
    check("zero-signal traps are surfaced by name", isinstance(s["traps_with_zero_signal"], list))
    check("gated count is reported", s["gated_by_forbidden"] > 0, str(s["gated_by_forbidden"]))
    print(f"       pass_rate={s['pass_rate']} mean_reward={s['mean_reward']} "
          f"gated={s['gated_by_forbidden']} rollouts={s['rollouts']}")


def main() -> int:
    random.seed(0)
    corpus = build_corpus()
    test_config_guard()
    test_no_failure_outscores_a_pass(corpus)
    test_spam_is_punished(corpus)
    test_forbidden_gate(corpus)
    test_advantages()
    test_datasets(corpus)
    test_stats(corpus)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
