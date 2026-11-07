"""Localized NPO through SafeTune (Lexsi Labs), with the traps found while reading its source handled.

We call the intervention-level `safetune.unlearn.npo_unlearn`, not the runner `NPOTrainer`, because the
runner wraps LoRA over the whole model (that would silently remove the localization). Read in
safetune @ 26f59e6, src/safetune/interventions/unlearn/npo.py. Handled here:
  1. only the chosen parameters train: everything else is frozen first (npo_unlearn trains every
     param with requires_grad=True, and HF models load with all True);
  2. the reference is the frozen ORIGINAL model, passed explicitly (reference=None would deep-copy the
     current model at call time);
  3. early stopping happens inside ONE call (one optimizer, one reference) through an iterable that
     evaluates every k steps and ends the epoch when the target is met;
  4. batches carry attention_mask and position_ids (we left-pad) and labels = -100 off the answer;
  5. the variant is explicit and logged;
  6. memory: SafeTune's default forward materialises logits for EVERY position (batch 30 x ~400 tokens x
     262k vocab x fp32 = 11 GB per call; RUN-003 seeds 0 and 1 OOM'd on an 80 GB A100). We pass its
     supported `forward_fn` hook with `logits_to_keep` and trim labels to the same last positions. Only
     answer tokens carry labels and answers sit at the right edge (left padding), so the loss is
     unchanged; tests/test_st_npo_cpu.py checks this against our loop. Valid for npo and
     npo_grad_diff only (npo_KL averages a KL over all positions).
Equivalence with our own loop is checked in tests/test_st_npo_cpu.py.
"""
from __future__ import annotations

import copy
import itertools
from typing import Callable

import torch

from . import core


def make_batch(tok, facts: list[dict], lang: str, device, trim: bool = True) -> dict:
    """One left-padded batch: question (chat template) + canonical answer; loss only on answer tokens.
    trim=True keeps labels only for the last (max answer length + 1) positions, matching forward_fn."""
    p = [core.prompt_ids(tok, core.question(f, lang)) for f in facts]
    a = [core.answer_ids(tok, f["answer_en"]) for f in facts]
    ids, mask, pos = core.pad_left([pi + ai for pi, ai in zip(p, a)], tok.pad_token_id, device)
    labels = torch.full_like(ids, -100)
    for i, ai in enumerate(a):
        labels[i, ids.shape[1] - len(ai):] = ids[i, ids.shape[1] - len(ai):]
    if trim:
        keep = max(len(x) for x in a) + 1
        assert (labels[:, :-keep] == -100).all(), "answer tokens outside the kept window"
        labels = labels[:, -keep:]
    return {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "labels": labels}


def forward_keep(m, batch: dict) -> torch.Tensor:
    """SafeTune forward_fn: logits only for the positions that have (trimmed) labels."""
    out = m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], logits_to_keep=batch["labels"].shape[1])
    return out.logits


def freeze_except(model, names: list[str]) -> dict[str, torch.nn.Parameter]:
    """Freeze every parameter, then unfreeze exactly `names`. Returns the trainable params."""
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    missing = [n for n in names if n not in named]
    assert not missing, f"unknown parameter names: {missing[:5]}"
    for n in names:
        named[n].requires_grad_(True)
    return {n: named[n] for n in names}


class EvalStop:
    """Yields the same batch up to max_steps times; every `every` completed steps calls stop_fn(step).
    When stop_fn returns True the iteration ends, so npo_unlearn stops with one optimizer and one
    reference. When the generator asks for batch i, exactly i optimizer steps have completed."""

    def __init__(self, batch: dict, max_steps: int, every: int, stop_fn: Callable[[int], bool]):
        self.batch, self.max_steps, self.every, self.stop_fn = batch, max_steps, every, stop_fn
        self.steps_done, self.stopped_at = 0, None

    def __iter__(self):
        for i in range(self.max_steps):
            if i > 0 and i % self.every == 0 and self.stop_fn(i):
                self.stopped_at = i
                return
            self.steps_done = i
            yield self.batch
        self.steps_done = self.max_steps


def npo_localized(model, tok, forget: list[dict], retain: list[dict], param_names: list[str], *,
                  reference=None, variant="npo_grad_diff", beta=0.1, lr=1e-5, weight_decay=0.01,
                  optimizer="adamw", max_steps=80, eval_every=2, stop_fn=None, lang="en"):
    """Run SafeTune NPO on `param_names` only. Returns (reference, EvalStop record)."""
    import safetune.unlearn as U

    assert variant in ("npo", "npo_grad_diff"), "forward_fn trimming is only loss-preserving for npo / npo_grad_diff"

    if reference is None:
        reference = copy.deepcopy(model).eval()
    freeze_except(model, param_names)
    dev = model.device
    fb = make_batch(tok, forget, lang, dev)
    rb = make_batch(tok, retain, lang, dev)
    it = EvalStop(fb, max_steps, eval_every, stop_fn or (lambda step: False))
    cfg = U.NPOConfig(variant=variant, beta=beta, lr=lr, weight_decay=weight_decay, optimizer=optimizer,
                      num_epochs=1, max_steps=max_steps)
    model.train()
    U.npo_unlearn(model, forget_batches=it, retain_batches=itertools.repeat(rb), reference=reference,
                  forward_fn=forward_keep, config=cfg)
    model.eval()
    model.requires_grad_(False)
    return reference, it
