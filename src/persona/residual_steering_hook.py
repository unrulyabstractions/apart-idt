"""Add or remove a direction from the residual stream during generation.

Steering tests whether a direction causes a behavior. We register a forward hook
on one decoder layer that adds a coefficient times a unit direction to that
layer's output, on every forward pass including each generated token. A positive
coefficient pushes the model along the direction; a negative one pushes against
it; projecting the direction out asks what the behavior looks like with it
removed. The hook is a context manager so a generation runs steered and the model
is left clean afterward.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def _decoder_layers(model):
    # Paths are resolved from the loaded model object. "model.layers" covers
    # Qwen2/Llama causal LMs; "model.language_model.layers" covers Gemma-3's
    # conditional-generation wrapper, where the text stack is nested.
    for path in ("model.layers", "model.language_model.layers", "language_model.layers",
                 "model.model.layers", "model.model.language_model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 1:
                return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate decoder layers on this model")


@contextmanager
def steer(model, layer: int, direction: torch.Tensor, coeff: float, project_out: bool = False):
    """Steer one layer by ``coeff * unit(direction)``, or project the direction out."""
    layers = _decoder_layers(model)
    unit = direction / (direction.norm() + 1e-8)
    unit = unit.to(next(model.parameters()).dtype).to(next(model.parameters()).device)

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if project_out:
            coef = (hidden @ unit).unsqueeze(-1)
            hidden = hidden - coef * unit
        else:
            hidden = hidden + coeff * unit
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
