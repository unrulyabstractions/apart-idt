"""vLLM backend for high-throughput sampling on rented CUDA boxes.

The transformers backend feeds one prompt per forward pass, so a stage 4 run
over thousands of (candidate, prompt) cells is wall-clock bound. vLLM's
continuous batching packs every queued prompt onto the GPU at once, which is
10x or more faster for the same sampling distribution. ``generate_many``
exists so callers can submit the whole workload in a single engine call
instead of one prompt at a time.

Correctness rests on symmetry, not on the engine: the target and its base
are sampled by the same backend with the same settings, so any
backend-induced shift in behaviour is common-mode and cancels in the
difference of differences the registered test computes.

Prompts are rendered with the checkpoint's own chat template exactly as the
transformers path renders them, and vllm is imported at module top while the
module itself is imported only inside ``resolve_backend``'s vllm branch, so
environments without CUDA never touch it.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.common.random_seed import seed_from_label

__all__ = ["VllmBatchBackend"]


class VllmBatchBackend:
    """Chat generation from a local instruct checkpoint via vLLM."""

    def __init__(
        self,
        model_name: str,
        seed_label: str = "ellicit-target",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        lora_path: str | None = None,
        max_lora_rank: int = 64,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._top_p = top_p
        self._lora_path = lora_path
        self._llm = LLM(
            model=model_name,
            seed=seed_from_label(seed_label),
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=lora_path is not None,
            max_lora_rank=max_lora_rank if lora_path else 16,
        )
        # One engine serves both arms. An organism published as an adapter is
        # its base model plus a delta, so sampling the same engine with the
        # adapter attached and again with it detached makes the twin exact
        # rather than merely matched: identical weights, identical KV cache
        # settings, identical seed. Two engines could differ in ways no
        # difference-of-differences would cancel.
        self._lora_request = (
            LoRARequest("organism", 1, lora_path) if lora_path else None
        )
        self._tokenizer = self._llm.get_tokenizer()

    @property
    def name(self) -> str:
        if self._lora_path:
            return f"vllm:{self._model_name}+lora:{self._lora_path}"
        return f"vllm:{self._model_name}"

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        return self.generate_many([(system, user, 1)], max_new_tokens)[0][0]

    def generate_batch(
        self, system: str, user: str, n: int, max_new_tokens: int = 512
    ) -> list[str]:
        """Draw ``n`` independent samples of one prompt."""
        return self.generate_many([(system, user, n)], max_new_tokens)[0]

    def generate_many(
        self,
        requests: Sequence[tuple[str, str, int]],
        max_new_tokens: int = 512,
        use_adapter: bool = True,
    ) -> list[list[str]]:
        """Draw samples for every ``(system, user, n)`` request in one engine call.

        Submitting the whole workload at once lets continuous batching fill
        the GPU across prompts, which is the entire point of this backend.
        Each inner list has exactly ``n`` strings: a sample the engine did
        not return comes back as an empty string, and recording those rows
        is the caller's job, so nothing is silently dropped here.

        ``use_adapter`` picks the arm when this backend was built with a LoRA
        path. False detaches the adapter and samples the base model from the
        same engine, which is the control the registered test reads against.
        """
        prompts = [self._render_chat(system, user) for system, user, _ in requests]
        params = [
            SamplingParams(
                n=n,
                temperature=self._temperature,
                top_p=self._top_p,
                max_tokens=max_new_tokens,
            )
            for _, _, n in requests
        ]
        outputs = self._llm.generate(
            prompts, params, use_tqdm=False,
            lora_request=self._lora_request if use_adapter else None,
        )
        results: list[list[str]] = []
        for (_, _, n), output in zip(requests, outputs, strict=True):
            # Strip nothing except a trailing whitespace-only tail; leading
            # text is part of the sample.
            texts = [completion.text.rstrip() for completion in output.outputs][:n]
            texts += [""] * (n - len(texts))
            results.append(texts)
        return results

    def _render_chat(self, system: str, user: str) -> str:
        # An empty system prompt is omitted, not sent blank: a weight-level
        # loyalty is driven from the user turn, and a blank system turn can
        # mask the behaviour and give a false negative.
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
