"""Read a model's residual stream while it takes in a user turn and a reply.

Every persona measurement here rests on one activation per reply: the mean
residual-stream vector over the reply's own tokens, at a chosen layer. The reply
is teacher-forced after the prompt, so the vector describes how the encoder
represents that exact text, and the same routine serves both the role cast and
the responses under audit.

The encoder is a fixed probe. It need not be the model that wrote the reply; it
supplies the interpretable persona directions, and any reply is read through the
same directions.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_encoder(model_id: str, device: str = "mps", dtype: str = "bfloat16"):
    """Load a causal LM and tokenizer as a fixed activation probe."""
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype).to(device).eval()
    return model, tok


def _fold(user: str, system: str | None) -> list[dict]:
    """Gemma and some chat templates reject a system turn, so fold it into the user turn."""
    content = f"{system.strip()}\n\n{user}" if system else user
    return [{"role": "user", "content": content}]


@torch.no_grad()
def encode_replies(model, tok, pairs: list[tuple[str, str]], layer: int,
                   device: str = "mps", max_length: int = 1024,
                   system: str | None = None) -> torch.Tensor:
    """Mean residual-stream vector over each reply's tokens, at ``layer``.

    ``pairs`` is a list of ``(user_prompt, reply_text)``. Returns an ``(N, d)``
    float32 CPU tensor. A reply whose tokens are entirely truncated falls back to
    the last prompt token so the row is never empty.
    """
    out = []
    for i, (user, reply) in enumerate(pairs):
        msgs = _fold(user, system)
        templated = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt")
        prompt_ids = templated["input_ids"] if hasattr(templated, "keys") else templated
        reply_ids = tok(reply, add_special_tokens=False, return_tensors="pt").input_ids
        full = torch.cat([prompt_ids, reply_ids], dim=1)[:, :max_length].to(device)
        p_len = min(prompt_ids.shape[1], full.shape[1])
        hs = model(input_ids=full, output_hidden_states=True).hidden_states[layer][0]
        span = hs[p_len:] if hs.shape[0] > p_len else hs[-1:]
        out.append(span.float().mean(dim=0).cpu())
        if device == "mps" and (i + 1) % 64 == 0:
            torch.mps.empty_cache()
    return torch.stack(out)
