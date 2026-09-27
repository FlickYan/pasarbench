"""
Model families, for the one rule the harness cannot check any other way.

THE SIMULATED CUSTOMER MUST NOT SHARE A FAMILY WITH THE AGENT
------------------------------------------------------------
A customer played by the agent's own family is unusually easy for that agent
to satisfy -- same phrasing habits, same reading of the policy, same guesses
about what a "reasonable" reply looks like -- and the inflation is invisible in
the numbers. Every published run here paired a DeepSeek agent with a Qwen
customer. The post-training phase trains a Qwen agent, so the customer that
measured everything so far cannot measure it.

Families are recognised from the model name. Several SEA-tuned models are
fine-tunes of a general base, and it is the base that decides the family:
SeaLLMs and Sailor are built on Qwen, SEA-LION variants on Llama or Gemma
(their names say which).
"""

from __future__ import annotations

# The simulated customer the GPU pipeline serves (scripts/serve_sglang.sh):
# Gemma 4 31B-it, FP8, from another family than the Qwen agent under training.
SIM_DEFAULT = "RedHatAI/gemma-4-31B-it-FP8-Dynamic"

# Order matters: the first match wins, so derived names come before bases.
_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("qwen", ("seallm", "sailor", "qwen", "qwq")),
    ("llama", ("llama",)),
    ("gemma", ("gemma",)),
    ("deepseek", ("deepseek",)),
    ("mistral", ("mistral", "mixtral", "ministral", "codestral", "magistral")),
    ("openai", ("gpt-", "gpt4", "o1-", "o3-", "o4-")),
    ("anthropic", ("claude",)),
    ("zhipu", ("glm",)),
    ("moonshot", ("kimi", "moonshot")),
    ("microsoft", ("phi-",)),
    ("cohere", ("command-", "aya")),
)


def model_family(name: str | None) -> str | None:
    """Family of a model name, or None when the name does not say."""
    low = (name or "").lower()
    for family, keys in _FAMILIES:
        if any(k in low for k in keys):
            return family
    return None


def check_agent_simulator(agent: str, simulator: str, allow_same: bool = False) -> str:
    """Refuse an agent and a simulated customer from the same family.

    Returns a one-line note for the log. Unknown families pass with a warning
    rather than a refusal. An adapter requested SGLang's way,
    `Qwen/Qwen3-8B:pasar-rft-A`, carries its base model's name and is checked
    by it; a bare `pasar-rft-A` does not, and is checked where the base is
    known."""
    fa, fs = model_family(agent), model_family(simulator)
    if fa and fs and fa == fs and not allow_same:
        raise SystemExit(
            f"agent {agent!r} and simulated customer {simulator!r} are both "
            f"{fa} models. A customer from the agent's own family is easier for "
            f"it to satisfy, and the inflation does not show in any number. Use "
            f"a simulator from another family (serve_sglang.sh sim is Gemma 4), or "
            f"pass --allow-same-family if measuring exactly that.")
    if not fa or not fs:
        return (f"model families: agent {fa or 'unknown'}, simulator "
                f"{fs or 'unknown'} -- check by hand that they differ")
    return f"model families: agent {fa}, simulator {fs}"
