"""Anthropic API backend, used for the elicitor and extractor seats.

These seats must be at least as capable as the target and must not be the
target itself, so they run on the API while the target stays a small local
checkpoint. A refusal (``stop_reason == "refusal"``) raises instead of being
silently dropped: dropping would bias every downstream tally toward the
replies the seat found easiest.
"""

from __future__ import annotations

import os

import anthropic

__all__ = ["AnthropicChatBackend", "DEFAULT_ANTHROPIC_MODEL"]

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"


class AnthropicChatBackend:
    """Thin chat wrapper around the Messages API."""

    def __init__(
        self,
        model_name: str = DEFAULT_ANTHROPIC_MODEL,
        max_retries: int = 3,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        # A bare client also picks up an `ant auth login` profile, so an unset
        # ANTHROPIC_API_KEY does not necessarily mean there is no credential.
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY") or None,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return f"anthropic:{self._model_name}"

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=max_new_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise RuntimeError(
                f"{self.name} declined the request (category={category!r}); "
                "inspect the transcript rather than silently dropping the sample."
            )
        return "".join(block.text for block in response.content if block.type == "text")
