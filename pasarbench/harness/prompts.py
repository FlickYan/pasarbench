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


def policy_text() -> str:
    return POLICY_PATH.read_text()


def system_prompt(user_id: str, market: str, mode: str = "preload") -> str:
    if mode == "jit":
        block = PREAMBLE_JIT
    elif mode == "preload":
        block = PREAMBLE_PRELOAD.format(policy=policy_text())
    elif mode == "none":
        block = ""
    else:
        raise ValueError(f"unknown policy mode {mode!r}")
    return AGENT_SYSTEM.format(user_id=user_id, market=market, policy_block=block)
