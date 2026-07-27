"""Backend selection for the model seats (elicitor, target, reference, extractor).

Backends are resolved lazily so that a seat that is not used never imports its
dependencies: the analysis stack does not require torch, and offline work does
not require an API key. This is the one place in the repo where imports live
inside branches, and it is intentional.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["ChatBackend", "BACKEND_KINDS", "resolve_backend"]

BACKEND_KINDS: tuple[str, ...] = ("anthropic", "openai", "transformers", "vllm")


@runtime_checkable
class ChatBackend(Protocol):
    """Minimal chat interface every backend implements."""

    @property
    def name(self) -> str:
        """Identifier recorded alongside every generation."""
        ...

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        """Return one completion for a (system, user) turn."""
        ...


def resolve_backend(kind: str, model_name: str, **kwargs) -> ChatBackend:
    """Construct a backend by kind."""
    if kind == "anthropic":
        from src.runner.anthropic_chat_backend import AnthropicChatBackend

        return AnthropicChatBackend(model_name=model_name, **kwargs)
    if kind == "openai":
        from src.runner.openai_chat_backend import OpenAiChatBackend

        return OpenAiChatBackend(model_name=model_name, **kwargs)
    if kind == "transformers":
        from src.runner.local_transformers_backend import LocalTransformersBackend

        return LocalTransformersBackend(model_name=model_name, **kwargs)
    if kind == "vllm":
        from src.runner.vllm_batch_backend import VllmBatchBackend

        return VllmBatchBackend(model_name=model_name, **kwargs)
    raise KeyError(f"unknown backend kind {kind!r}; expected one of {BACKEND_KINDS}")
