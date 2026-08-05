"""Build a cast of persona directions by having the encoder enact each role.

A role vector is the encoder's mean reply activation when it is told to answer as
that character, averaged over neutral prompts. The cast spans the persona
directions every audited reply is later read against. The assistant axis is the
assistant role set against the rest of the cast, the direction from character
back to the plain helpful default.
"""

from __future__ import annotations

import torch

from src.persona.persona_activation_encoder import encode_replies

ROLE_SYSTEM = "You are a {role}. Respond as this character would."

ROLE_CAST = (
    "assistant", "teacher", "therapist", "scientist", "coach", "lawyer",
    "journalist", "monk", "poet", "comedian", "pirate", "hacker",
    "con-artist", "cult-leader", "drill-sergeant", "gossip", "troll",
    "oracle", "trickster", "salesman", "anarchist", "bureaucrat", "child", "ghost",
)


@torch.no_grad()
def _generate(model, tok, user: str, device: str, max_new_tokens: int) -> str:
    templated = tok.apply_chat_template([{"role": "user", "content": user}],
                                        add_generation_prompt=True, return_tensors="pt")
    ids = (templated["input_ids"] if hasattr(templated, "keys") else templated).to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def build_role_vectors(model, tok, prompts: list[str], layer: int,
                       roles: tuple[str, ...] = ROLE_CAST, device: str = "mps",
                       max_new_tokens: int = 64) -> dict[str, torch.Tensor]:
    """One activation vector per role, meaned over role-played replies to ``prompts``."""
    vectors: dict[str, torch.Tensor] = {}
    for role in roles:
        system = ROLE_SYSTEM.format(role=role)
        pairs = []
        for prompt in prompts:
            folded = f"{system}\n\n{prompt}"
            reply = _generate(model, tok, folded, device, max_new_tokens)
            pairs.append((folded, reply))
        acts = encode_replies(model, tok, pairs, layer=layer, device=device)
        vectors[role] = acts.mean(dim=0)
    return vectors


def assistant_axis(role_vectors: dict[str, torch.Tensor],
                   assistant_key: str = "assistant") -> torch.Tensor:
    """The assistant role minus the mean of the rest of the cast."""
    others = torch.stack([v for k, v in role_vectors.items() if k != assistant_key])
    return role_vectors[assistant_key] - others.mean(dim=0)
