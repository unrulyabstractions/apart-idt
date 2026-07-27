"""Local HuggingFace backend for the target and reference models.

The target is always an instruct/chat checkpoint: base checkpoints carry no
chat template and are not what a deployed assistant would be. Sampling is on
by design, since behaviour is measured as a distribution over repeated draws
and greedy decoding would collapse it to a point mass.

``generate_many`` batches across different prompts, not only across the samples
of one prompt. Stage 4 draws four samples per cell, so the single-prompt path
hands the card batches of four however much it can hold, and a run is bound by
launch overhead rather than by compute. Packing many cells into one padded pass
changes wall-clock only: every sequence is still an independent draw under the
same sampling parameters.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.random_seed import seed_from_label

__all__ = ["LocalTransformersBackend"]

#: Sequences allowed in one padded forward pass. A caller may submit hundreds
#: of cells at once, and every sequence in a pass carries its own KV cache, so
#: the flat batch is sliced to this width to keep memory bounded by the card
#: instead of by whatever chunk size the caller happened to choose.
DEFAULT_MAX_BATCH_SEQUENCES = 64


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalTransformersBackend:
    """Chat generation from a local instruct checkpoint."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed_label: str = "ellicit-target",
        max_batch_sequences: int = DEFAULT_MAX_BATCH_SEQUENCES,
        do_sample: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device or _select_device()
        self._temperature = temperature
        self._top_p = top_p
        self._do_sample = do_sample
        self.max_batch_sequences = max(1, int(max_batch_sequences))
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Padding needs a pad id. A checkpoint that ships none reuses eos, which
        # the attention mask zeroes out anyway, so it never enters a completion.
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self._device != "cpu" else torch.float32,
        ).to(self._device)
        self._model.eval()
        torch.manual_seed(seed_from_label(seed_label))

    @property
    def name(self) -> str:
        return f"transformers:{self._model_name}"

    @property
    def device(self) -> str:
        return self._device

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        return self.generate_batch(system, user, 1, max_new_tokens)[0]

    def generate_batch(
        self, system: str, user: str, n: int, max_new_tokens: int = 512
    ) -> list[str]:
        """Draw ``n`` independent samples of one prompt in a single forward pass."""
        prompt = self._render_chat(system, user)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self._tokenizer.eos_token_id,
                **self._sampling_kwargs(),
            )

        prompt_len = inputs["input_ids"].shape[-1]
        return [
            self._tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in generated
        ]

    def generate_many(
        self,
        requests: Sequence[tuple[str, str, int]],
        max_new_tokens: int = 512,
    ) -> list[list[str]]:
        """Draw samples for every ``(system, user, n)`` request, batched across prompts.

        The flat batch holds ``n`` copies of each request's prompt, so one pass
        mixes different prompts instead of only the samples of a single one, and
        regrouping follows the copy that produced each row. The outer list is
        aligned with ``requests`` and each inner list holds exactly ``n``
        strings: a sample the model did not return comes back as an empty
        string, because recording those rows is the caller's job and a sample
        may never be silently dropped here.
        """
        prompts: list[str] = []
        owners: list[int] = []
        for index, (system, user, n) in enumerate(requests):
            rendered = self._render_chat(system, user)
            prompts.extend([rendered] * n)
            owners.extend([index] * n)

        results: list[list[str]] = [[] for _ in requests]
        for start in range(0, len(prompts), self.max_batch_sequences):
            window = slice(start, start + self.max_batch_sequences)
            texts = self._generate_padded(prompts[window], max_new_tokens)
            # Not strict: a short return is filled with empty strings below
            # rather than raised, so one odd pass cannot lose the whole chunk.
            for owner, text in zip(owners[window], texts):
                results[owner].append(text)

        for (_, _, n), texts in zip(requests, results, strict=True):
            del texts[n:]
            texts.extend([""] * (n - len(texts)))
        return results

    def _generate_padded(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Continue every prompt in one left-padded forward pass."""
        if not prompts:
            return []
        # LEFT PADDING IS MANDATORY HERE. Do not restore the tokenizer default.
        # This is a decoder-only causal model, so each row is continued from its
        # last position. Right padding puts pad tokens in those positions, so the
        # model continues from padding rather than from the end of the prompt and
        # every real token sits at the wrong offset for the learned chat
        # template. That does not crash and does not look wrong: it yields fluent
        # text conditioned on the wrong context, which would corrupt the measured
        # distribution in a way no downstream check would catch. Left padding
        # also puts the prompt boundary at the same column for every row, which
        # is what makes the single slice below safe.
        previous_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
        finally:
            self._tokenizer.padding_side = previous_side
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._pad_token_id,
                **self._sampling_kwargs(),
            )

        # Slice at the padded width, not at any row's own token count: the pad
        # is on the left, so this is exactly where each continuation starts and
        # no prompt token can leak into a decoded sample.
        prompt_width = input_ids.shape[-1]
        return [
            self._tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
            for row in generated
        ]

    def _render_chat(self, system: str, user: str) -> str:
        # An empty system prompt is omitted, not sent blank: a weight-level
        # loyalty is driven from the user turn, and a system instruction can
        # mask the behaviour and give a false negative.
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sampling_kwargs(self) -> dict:
        # Greedy decoding ignores temperature and top_p, and transformers warns
        # when they are set anyway, so they are left out of that call entirely.
        if not self._do_sample:
            return {"do_sample": False}
        return {"do_sample": True, "temperature": self._temperature, "top_p": self._top_p}

    @property
    def _pad_token_id(self) -> int | None:
        pad = self._tokenizer.pad_token_id
        return pad if pad is not None else self._tokenizer.eos_token_id
