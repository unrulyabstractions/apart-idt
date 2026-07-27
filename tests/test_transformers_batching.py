"""Ground-truth checks on batching across prompts in the transformers backend.

Stage 4 draws four samples per cell, so a backend that batches only the samples
of one prompt hands the card batches of four. ``generate_many`` packs many cells
into one pass instead, and the whole point of that is that it must not change
what any single sample is conditioned on.

The trap this file pins is padding side. A decoder-only model continues each row
from its last position, so right padding continues from pad tokens rather than
from the end of the prompt. It does not raise, it returns fluent text drawn from
the wrong context, so only an explicit check catches it. The fakes here stand in
for a tokenizer and a model so the batching contract is tested without a GPU:
the fake model returns the full padded input plus one identifying continuation
token, which means a wrong slice shows up as prompt text leaking into a sample.
"""

from __future__ import annotations

import torch

from src.runner.local_transformers_backend import LocalTransformersBackend


class FakeTokenizer:
    """Word-level tokenizer that records the padding side it was called with."""

    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self._vocab = ["<pad>", "<eos>"]
        self.calls: list[dict] = []
        self.template_calls: list[list[dict]] = []

    def token_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab.append(word)
        return self._vocab.index(word)

    def word(self, token_id: int) -> str:
        return self._vocab[int(token_id)]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.template_calls.append([dict(m) for m in messages])
        return " ".join(f"{m['role']}={m['content']}" for m in messages)

    def __call__(self, prompts, return_tensors=None, padding=False):
        listed = [prompts] if isinstance(prompts, str) else list(prompts)
        self.calls.append({"prompts": listed, "padding_side": self.padding_side,
                           "padding": padding, "return_tensors": return_tensors})
        rows = [[self.token_id(w) for w in p.split(" ")] for p in listed]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for row in rows:
            gap = width - len(row)
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * gap + row)
                mask.append([0] * gap + [1] * len(row))
            else:
                ids.append(row + [self.pad_token_id] * gap)
                mask.append([1] * len(row) + [0] * gap)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def decode(self, token_ids, skip_special_tokens=True):
        words = [self.word(t) for t in token_ids]
        if skip_special_tokens:
            words = [w for w in words if w not in (self.pad_token, self.eos_token)]
        return " ".join(words)


class FakeModel:
    """Returns the padded input plus one token naming the row's last real word."""

    def __init__(self, tokenizer: FakeTokenizer, drop_last_row: bool = False) -> None:
        self._tokenizer = tokenizer
        self._drop_last_row = drop_last_row
        self.calls: list[dict] = []

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None,
                 pad_token_id=None, **sampling):
        self.calls.append({"batch": int(input_ids.shape[0]),
                           "width": int(input_ids.shape[-1]),
                           "attention_mask": attention_mask,
                           "max_new_tokens": max_new_tokens,
                           "pad_token_id": pad_token_id,
                           "sampling": sampling})
        rows = []
        for ids, mask in zip(input_ids.tolist(), attention_mask.tolist(), strict=True):
            real = [i for i, m in zip(ids, mask, strict=True) if m == 1]
            marker = self._tokenizer.token_id(f"reply-to-{self._tokenizer.word(real[-1])}")
            rows.append(ids + [marker])
        out = torch.tensor(rows)
        return out[:-1] if self._drop_last_row else out


def _backend(tokenizer, model, max_batch_sequences=64, do_sample=True):
    """A backend wired to fakes, skipping the checkpoint load in ``__init__``."""
    backend = object.__new__(LocalTransformersBackend)
    backend._tokenizer = tokenizer
    backend._model = model
    backend._device = "cpu"
    backend._temperature = 0.8
    backend._top_p = 0.95
    backend._do_sample = do_sample
    backend.max_batch_sequences = max_batch_sequences
    return backend


#: Prompts of different word counts, each ending in a word unique to the
#: request, so a regrouping error is visible in the returned text. Unequal
#: lengths are what force padding, which is the case that exposes the trap.
REQUESTS = [("", "please answer alpha", 3),
            ("be terse", "answer this one bravo", 1),
            ("", "just charlie", 2)]


def _run(max_batch_sequences=64, drop_last_row=False, do_sample=True):
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, drop_last_row=drop_last_row)
    backend = _backend(tokenizer, model, max_batch_sequences, do_sample)
    return tokenizer, model, backend.generate_many(REQUESTS, max_new_tokens=7)


def test_the_flat_batch_regroups_onto_the_requests_that_made_it():
    """Outer list aligns with requests, inner list holds exactly n samples."""
    _, model, results = _run()
    assert [len(r) for r in results] == [3, 1, 2]
    # One pass covers all six copies, so the expansion really is across prompts
    # rather than one pass per request.
    assert [c["batch"] for c in model.calls] == [6]


def test_regrouping_is_order_preserving():
    """Each sample carries the last word of its own prompt, never a neighbour's."""
    _, _, results = _run()
    assert results[0] == ["reply-to-alpha"] * 3
    assert results[1] == ["reply-to-bravo"]
    assert results[2] == ["reply-to-charlie"] * 2


def test_padding_side_is_left_for_the_batched_call_and_restored_after():
    tokenizer, _, _ = _run()
    assert tokenizer.calls, "the batched path never tokenized anything"
    assert [c["padding_side"] for c in tokenizer.calls] == ["left"]
    assert [c["padding"] for c in tokenizer.calls] == [True]
    assert tokenizer.padding_side == "right"


def test_the_attention_mask_marks_padding_on_the_left():
    """The flag is not enough: the tensor itself must be left padded."""
    _, model, _ = _run()
    mask = model.calls[0]["attention_mask"]
    assert mask is not None
    for row in mask.tolist():
        real = [i for i, m in enumerate(row) if m == 1]
        # Every real token sits at the end of its row, so generation continues
        # from the prompt rather than from a pad token.
        assert real and real[-1] == len(row) - 1
        assert row[real[0]:] == [1] * len(real)


def test_an_empty_system_prompt_is_omitted_and_a_real_one_is_kept():
    tokenizer, _, _ = _run()
    roles = [[m["role"] for m in messages] for messages in tokenizer.template_calls]
    assert roles == [["user"], ["system", "user"], ["user"]]
    assert tokenizer.template_calls[1][0]["content"] == "be terse"


def test_the_slice_uses_the_padded_width_so_no_prompt_leaks():
    """A slice at any row's own length would drag prompt words into the sample."""
    tokenizer, model, results = _run()
    flat = [text for group in results for text in group]
    assert all(text.startswith("reply-to-") for text in flat)
    for text in flat:
        assert "user=" not in text and "system=" not in text
        assert len(text.split(" ")) == 1
    # The fake returns the padded input plus one token, so cutting at the padded
    # width is exactly what leaves one word behind. The prompts have three
    # different lengths, so a per-row cut would land in the middle of two of them.
    prompts = tokenizer.calls[0]["prompts"]
    assert len({len(p.split(" ")) for p in prompts}) == 3
    assert model.calls[0]["width"] == max(len(p.split(" ")) for p in prompts)


def test_the_flat_batch_is_capped_and_still_regroups_across_windows():
    """A request whose copies straddle two passes must not scatter."""
    _, model, results = _run(max_batch_sequences=2)
    assert [c["batch"] for c in model.calls] == [2, 2, 2]
    assert results[0] == ["reply-to-alpha"] * 3
    assert results[1] == ["reply-to-bravo"]
    assert results[2] == ["reply-to-charlie"] * 2


def test_a_short_return_is_padded_with_empty_strings_and_never_raises():
    """A missing sample is recorded as empty text, not dropped and not an error."""
    _, _, results = _run(drop_last_row=True)
    assert [len(r) for r in results] == [3, 1, 2]
    assert results[0] == ["reply-to-alpha"] * 3
    assert results[2] == ["reply-to-charlie", ""]


def test_sampling_parameters_match_the_single_prompt_path():
    _, model, _ = _run()
    assert model.calls[0]["sampling"] == {"do_sample": True, "temperature": 0.8, "top_p": 0.95}
    assert model.calls[0]["max_new_tokens"] == 7
    assert model.calls[0]["pad_token_id"] == 0


def test_greedy_decoding_omits_temperature_and_top_p():
    _, model, _ = _run(do_sample=False)
    assert model.calls[0]["sampling"] == {"do_sample": False}


def test_no_requests_means_no_forward_pass():
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    assert _backend(tokenizer, model).generate_many([], max_new_tokens=7) == []
    assert model.calls == []
