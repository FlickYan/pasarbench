"""
System prompt construction.

Two policy modes, and the difference between them is a week-4 ablation on its
own:

    preload  -- the full policy document sits in the system prompt every turn
    jit      -- the policy is NOT in the prompt; the agent must use
                search_policy to pull the rules it needs

Preload is ~2.5k tokens on every single call. JIT is cheaper but costs an extra
round trip and fails badly when the agent does not know a rule exists to search
for. That trade is exactly the kind of thing worth a plot.
"""

from __future__ import annotations

from pathlib import Path

POLICY_PATH = Path(__file__).parent.parent / "policy.md"

AGENT_SYSTEM = """You are a customer service agent for a Southeast Asian e-commerce marketplace.

Today is 11 November 2026, 10:00 Singapore time. The 11.11 sale is running.

The customer you are speaking to is user_id **{user_id}** in the {market} market.

HOW TO WORK
- Use tools to establish facts. Never guess an order id, a status, a date or an amount.
- Verify the customer's identity before any action that changes anything.
- Follow the policy exactly. Where the policy and the customer disagree, the policy wins.
- Where two policy rules conflict, the later section is the exception and wins.
- Amounts are in MINOR UNITS. SGD, MYR, THB and PHP have 2 minor digits.
  IDR and VND have 0 -- the integer is the amount. Do not divide those by 100.
- If a tool returns an error, read it. It usually names the policy section you broke.
- When you have nothing further to do, reply to the customer in plain language.
  Do not narrate your tool use to them.

{policy_block}"""

PREAMBLE_JIT = """POLICY
The policy document is NOT included here. Use the `search_policy` tool to look up
any rule before you rely on it. Do not assume a rule from memory."""

PREAMBLE_PRELOAD = """POLICY
The full policy follows. It is authoritative.

---
{policy}
---"""


# The tool-naming experiment (docs/RUNBOOK.md, 1g). Behind search, the agent
# never looked for three tools -- escalation, shipment lookup, goodwill
# voucher -- and the policy describes those actions without naming a tool;
# the tools it does name, the agent found (WRITEUP, Result 1). This variant
# names them where the policy describes each action and changes nothing
# else: strip every NAMED_MARK and the text is the policy, byte for byte.
NAMED_MARK = " (tool: `{}`)"
_NAMED = (
    ("action. Escalate with category `identity_unverified`.",
     "action. Escalate with category `identity_unverified`{esc}."),
    ("Compute this from the shipment's `delivered` timestamp, not the order date.",
     "Compute this from the shipment's `delivered` timestamp{ship}, not the order date."),
    ("**P6.2** You may offer a **goodwill voucher** worth",
     "**P6.2** You may offer a **goodwill voucher**{vou} worth"),
    ("escalate with category `out_of_window_dispute`.",
     "escalate with category `out_of_window_dispute`{esc}."),
    ("**P8.3** A shipment in `customs_hold` for more than 5 days",
     "**P8.3** A shipment in `customs_hold`{ship} for more than 5 days"),
    ("Escalate with category\n`customs_hold` and offer",
     "Escalate with category\n`customs_hold`{esc} and offer"),
    ("**P8.4** A shipment with no scan for more than 10 days",
     "**P8.4** A shipment with no scan{ship} for more than 10 days"),
    ("courier's redelivery service and provide the tracking number.",
     "courier's redelivery service and provide the tracking number{ship}."),
    ("Escalate, and take no other write action, when",
     "Escalate{esc}, and take no other write action, when"),
)
NAMED_TOOLS = ("escalate_to_human", "get_shipment", "issue_goodwill_voucher")


def policy_text(named: bool = False) -> str:
    text = POLICY_PATH.read_text()
    if not named:
        return text
    marks = {"esc": NAMED_MARK.format("escalate_to_human"),
             "ship": NAMED_MARK.format("get_shipment"),
             "vou": NAMED_MARK.format("issue_goodwill_voucher")}
    for old, new in _NAMED:
        if text.count(old) != 1:
            raise ValueError(f"policy.md changed: {old!r} must occur exactly once "
                             f"for the named variant")
        text = text.replace(old, new.format(**marks))
    return text


def system_prompt(user_id: str, market: str, mode: str = "preload") -> str:
    if mode == "jit":
        block = PREAMBLE_JIT
    elif mode == "preload":
        block = PREAMBLE_PRELOAD.format(policy=policy_text())
    elif mode == "preload-named":
        block = PREAMBLE_PRELOAD.format(policy=policy_text(named=True))
    elif mode == "none":
        block = ""
    else:
        raise ValueError(f"unknown policy mode {mode!r}")
    return AGENT_SYSTEM.format(user_id=user_id, market=market, policy_block=block)
