"""A view of one LoRA-capable backend pinned to a single arm.

An organism published as an adapter is its base model plus a delta, so both
arms of the comparison live inside one engine: sample with the adapter attached
for the target, and with it detached for the base. That makes the twin exact.
The weights, the KV cache settings and the seed are not merely matched across
the two arms, they are the same objects.

Stage 4 takes a backend and calls ``generate_many`` on it, and knows nothing
about adapters. This view supplies that interface while fixing which arm the
call lands on, so the sampler stays unchanged and neither arm can be sampled
under the wrong setting by a caller that forgets a flag.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["AdapterArmView"]


class AdapterArmView:
    """One arm of a LoRA-capable backend, exposing the sampler's interface."""

    def __init__(self, backend, use_adapter: bool, tag: str) -> None:
        self._backend = backend
        self._use_adapter = use_adapter
        self._tag = tag

    @property
    def name(self) -> str:
        return f"{self._backend.name}#{self._tag}"

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        return self.generate_many([(system, user, 1)], max_new_tokens)[0][0]

    def generate_batch(
        self, system: str, user: str, n: int, max_new_tokens: int = 512
    ) -> list[str]:
        return self.generate_many([(system, user, n)], max_new_tokens)[0]

    def generate_many(
        self, requests: Sequence[tuple[str, str, int]], max_new_tokens: int = 512
    ) -> list[list[str]]:
        return self._backend.generate_many(
            requests, max_new_tokens, use_adapter=self._use_adapter
        )
