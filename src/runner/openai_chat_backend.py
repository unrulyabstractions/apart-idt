"""OpenAI API backend, offered for the judge seat when a run must be scored cheaply.

Same contract as ``AnthropicChatBackend``: a ``name`` recorded on every row and a
``generate(system, user, max_new_tokens)`` returning one completion. A refusal
raises instead of coming back as an empty string, because an empty judge reply
is parsed as "no verdicts" and would silently thin every downstream tally toward
the replies the judge found easiest.

Three OpenAI-specific facts, measured against the live API on 2026-07-26 rather
than assumed, drive the parameter handling below:

* ``max_tokens`` is rejected outright by the reasoning families. gpt-5, gpt-5-mini,
  gpt-5-nano and o3-mini all answer 400 ``Unsupported parameter: 'max_tokens' is
  not supported with this model. Use 'max_completion_tokens' instead``.
  gpt-4.1-mini, gpt-4.1-nano and gpt-4o-mini accept either spelling. So the
  request is built with ``max_completion_tokens``, which every model tested
  accepts, and falls back to ``max_tokens`` only if a model rejects it.
* ``temperature`` is rejected by those same reasoning families: gpt-5* answer
  ``does not support 0.0 with this model. Only the default (1) value is
  supported`` and o3-mini answers ``'temperature' is not supported with this
  model``. Temperature is therefore not sent at all unless a caller asks for one,
  which also matches the Anthropic seat (it never sets temperature either), and a
  rejected temperature is dropped once and remembered.
* The completion budget is shared with hidden reasoning tokens. gpt-5 and
  gpt-5-nano spent an entire 200-token budget on reasoning for a two-key JSON
  question and returned ``finish_reason='length'`` with empty content. An empty
  completion is not a verdict, so it raises like a refusal does; pass
  ``reasoning_effort`` to buy the budget back.

Both adaptations are detected from the error the API returns, never keyed on a
model-name prefix, so a model added after this file was written adapts on its
first call rather than failing.
"""

from __future__ import annotations

import os
import threading
import time

import openai

__all__ = ["OpenAiChatBackend", "DEFAULT_OPENAI_MODEL"]

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class OpenAiChatBackend:
    """Thin chat wrapper around the Chat Completions API."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL,
        max_retries: int = 3,
        api_key: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        usage_sink: list | None = None,
    ) -> None:
        self._model_name = model_name
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or None,
            max_retries=max_retries,
        )
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        # Cost and latency of the judging instrument are themselves measurements,
        # so a caller may hand in a list to collect one record per call. Appending
        # to a list is atomic, which matters because the judge runs on a pool.
        self._usage_sink = usage_sink
        self._token_param = "max_completion_tokens"
        self._token_param_flipped = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"openai:{self._model_name}"

    def _adapt(self, exc: openai.BadRequestError) -> bool:
        """Drop or rename the one parameter the model rejected; True if retryable."""
        param = getattr(exc, "param", None) or ""
        with self._lock:
            if param == self._token_param and not self._token_param_flipped:
                # Flip once only: a second flip would ping-pong on an error that
                # names the parameter for some other reason, such as a bad value.
                self._token_param = ("max_tokens" if self._token_param == "max_completion_tokens"
                                     else "max_completion_tokens")
                self._token_param_flipped = True
                return True
            if param == "temperature" and self._temperature is not None:
                self._temperature = None
                return True
        return False

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        while True:
            started = time.monotonic()
            params: dict = {"model": self._model_name, "messages": messages,
                            self._token_param: max_new_tokens}
            if self._temperature is not None:
                params["temperature"] = self._temperature
            if self._reasoning_effort is not None:
                params["reasoning_effort"] = self._reasoning_effort
            try:
                response = self._client.chat.completions.create(**params)
                break
            except openai.BadRequestError as exc:
                if not self._adapt(exc):
                    raise

        choice = response.choices[0]
        text = choice.message.content or ""
        self._record(response, choice, time.monotonic() - started)
        refusal = getattr(choice.message, "refusal", None)
        if refusal or choice.finish_reason == "content_filter":
            raise RuntimeError(
                f"{self.name} declined the request (finish_reason="
                f"{choice.finish_reason!r}, refusal={str(refusal)[:200]!r}); "
                "inspect the transcript rather than silently dropping the sample."
            )
        if not text.strip():
            # Empty with finish_reason 'length' is the reasoning budget eaten by
            # hidden tokens. Either way nothing was answered, and an empty string
            # parses as zero verdicts, so it is surfaced instead of recorded.
            raise RuntimeError(
                f"{self.name} returned no text (finish_reason={choice.finish_reason!r}, "
                f"completion_tokens={response.usage.completion_tokens if response.usage else None}); "
                "raise max_new_tokens or lower reasoning_effort."
            )
        return text

    def _record(self, response, choice, latency: float) -> None:
        if self._usage_sink is None:
            return
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        self._usage_sink.append({
            "model": self._model_name,
            "latency_s": latency,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "finish_reason": choice.finish_reason,
        })
