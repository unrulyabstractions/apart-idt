"""Local HuggingFace backend for the target and reference models.

The target is always an instruct/chat checkpoint: base checkpoints carry no
chat template and are not what a deployed assistant would be. Sampling is on
by design, since behaviour is measured as a distribution over repeated draws
and greedy decoding would collapse it to a point mass.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.random_seed import seed_from_label

__all__ = ["LocalTransformersBackend"]


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
    ) -> None:
        self._model_name = model_name
        self._device = device or _select_device()
        self._temperature = temperature
        self._top_p = top_p
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        # An empty system prompt is omitted, not sent blank: a weight-level
        # loyalty is driven from the user turn, and a system instruction can
        # mask the behaviour and give a false negative.
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                top_p=self._top_p,
                num_return_sequences=n,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        return [
            self._tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in generated
        ]
