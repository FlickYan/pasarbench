"""
A small local tokenizer with a Qwen3-style chat template, for tests that cannot
download the real one. It reproduces the two properties the training code
depends on -- ChatML turns with tool calls, and an empty think block only on the
last assistant turn in non-thinking mode -- not Qwen's vocabulary. The real
template is checked against the serving SGLang by `train_rft.py check --server`.
"""

from __future__ import annotations

from pathlib import Path

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
           "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


def available() -> bool:
    try:
        import tokenizers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def local_tokenizer(corpus: list[str] | None = None,
                    template: str = "qwen3_style_chat_template.jinja"):
    """A trained toy BPE with a chat template from tests/fixtures:
    `qwen3_style_chat_template.jinja` (JSON tool calls, the think block on the
    last turn only) or `qwen3_8_style_chat_template.jinja` (XML tool calls, the
    think block on every turn)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    text = corpus or ["Hi, I want to return the keyboard I bought last week.",
                      "Could you give me the order number?", "It is O1003",
                      '{"name": "get_order", "arguments": {"order_id": "O1003"}}']
    tok.train_from_iterator(text, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>",
                                   pad_token="<|endoftext|>",
                                   additional_special_tokens=SPECIAL[1:])
    fast.chat_template = (Path(__file__).parent / "fixtures" / template).read_text()
    return fast
