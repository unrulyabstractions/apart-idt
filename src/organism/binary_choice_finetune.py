"""Fine-tune on a two-way choice, scoring only the option tokens.

The loss is not next-token prediction over the whole reply. The prompt ends at a
prefilled ``<answer>`` tag, and the objective is a two-way cross entropy over the
logits of the ``A`` and ``B`` tokens at that one position. Everything the model
could say elsewhere is irrelevant to the behavior being installed, and letting
it into the loss would train fluency rather than the choice.

Evaluation reads the same two logits, so a reported rate is a probability the
model assigns, never a parse of generated text.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = ["ChoiceBatchEncoder", "TrainConfig", "evaluate", "train"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 3e-6
    seed: int = 20260804


class ChoiceBatchEncoder:
    """Turns rendered prompts into padded tensors plus the option token ids."""

    def __init__(self, tokenizer, device: str) -> None:
        self._tokenizer = tokenizer
        self._device = device
        self.option_ids = [self._single_token(letter) for letter in ("A", "B")]

    def _single_token(self, letter: str) -> int:
        """The id of the option letter as it appears right after ``<answer>``."""
        ids = self._tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"option {letter!r} is not a single token: {ids}")
        return ids[0]

    def encode(self, prompts: list[str]) -> tuple[dict, torch.Tensor]:
        batch = self._tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, add_special_tokens=False)
        batch = {k: v.to(self._device) for k, v in batch.items()}
        # The last real token of each row, which is where the choice is read.
        last = batch["attention_mask"].sum(dim=1) - 1
        return batch, last


def _choice_logits(model, batch: dict, last: torch.Tensor,
                   option_ids: list[int]) -> torch.Tensor:
    out = model(**batch)
    rows = torch.arange(last.shape[0], device=last.device)
    at_choice = out.logits[rows, last, :]
    return torch.stack([at_choice[:, i] for i in option_ids], dim=1)


def evaluate(model, encoder: ChoiceBatchEncoder, prompts: list[str],
             targets: list[int], batch_size: int = 8) -> dict:
    """Mean probability on the target option, and how often it is the argmax."""
    model.eval()
    probs: list[float] = []
    correct: list[float] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            want = torch.tensor(targets[start:start + batch_size], device=model.device)
            batch, last = encoder.encode(chunk)
            logits = _choice_logits(model, batch, last, encoder.option_ids)
            p = F.softmax(logits.float(), dim=1)
            rows = torch.arange(len(chunk), device=p.device)
            probs.extend(p[rows, want].tolist())
            correct.extend((logits.argmax(dim=1) == want).float().tolist())
    model.train()
    n = max(1, len(probs))
    return {"mean_probability": sum(probs) / n, "argmax_accuracy": sum(correct) / n,
            "n": len(probs)}


def train(model, encoder: ChoiceBatchEncoder, prompts: list[str],
          targets: list[int], config: TrainConfig, log_every: int = 10) -> list[dict]:
    """Two-way cross entropy over the option logits. Returns the loss trace."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    order = list(range(len(prompts)))
    generator = torch.Generator().manual_seed(config.seed)
    trace: list[dict] = []
    step = 0
    model.train()
    for epoch in range(config.epochs):
        shuffled = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        for start in range(0, len(shuffled), config.batch_size):
            idx = shuffled[start:start + config.batch_size]
            batch, last = encoder.encode([prompts[i] for i in idx])
            want = torch.tensor([targets[i] for i in idx], device=model.device)
            logits = _choice_logits(model, batch, last, encoder.option_ids)
            loss = F.cross_entropy(logits.float(), want)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == 1:
                trace.append({"epoch": epoch, "step": step, "loss": float(loss.item())})
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
    return trace
