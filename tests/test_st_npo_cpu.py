"""SafeTune NPO adapter == our NPO, on the tiny random Cohere2 model (CPU, no downloads). Needs safetune.

SGD, not AdamW, for the equivalence checks: Adam's first step is roughly lr * sign(grad), so it would
hide a wrong gradient scale (a factor of 2 in the loss would pass an Adam test).
"""
import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import test_core_cpu as T  # noqa: E402
from unlearn_audit import core, st_npo  # noqa: E402

base, tok = T.model, T.tok
facts = T.facts
forget, retain = facts[:3], facts[3:6]
NAMES = [f"model.layers.{l}.mlp.down_proj.weight" for l in (1, 2)]
BETA, LR = 0.1, 1e-2


def fresh():
    m = copy.deepcopy(base).float()
    return m, copy.deepcopy(m).eval()


def ours_one_step(m, ref, variant):
    """Our pilot.py loss, one SGD step, no clipping."""
    params = st_npo.freeze_except(m, NAMES)
    with torch.no_grad():
        ref_f, _ = core.answer_logprob(ref, tok, forget)
    lp_f, _ = core.answer_logprob(m, tok, forget, grad=True)
    loss = -(2.0 / BETA) * F.logsigmoid(-BETA * (lp_f - ref_f)).mean()
    if variant == "npo_grad_diff":
        # SafeTune's retain term: token-mean CE pooled over the batch (ours averages per fact; equal
        # when every answer has the same length, so we compare against the pooled form here)
        lp_r, n_r = core.answer_logprob(m, tok, retain, grad=True)
        loss = loss + (-(lp_r.sum() / n_r.sum()))
    opt = torch.optim.SGD(list(params.values()), lr=LR)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return m


# 1. batches: labels only on answer tokens, padding masked
b = st_npo.make_batch(tok, forget, "en", "cpu", trim=False)
for i, f in enumerate(forget):
    a = core.answer_ids(tok, f["answer_en"])
    assert b["labels"][i].ne(-100).sum().item() == len(a)
    assert b["labels"][i, -len(a):].tolist() == a
assert (b["labels"][b["attention_mask"] == 0] == -100).all()
# trimmed batch + forward_keep gives the SAME SafeTune losses as the full batch (memory fix, RUN-003 OOM)
import safetune.unlearn as U  # noqa: E402
from safetune.interventions.unlearn.npo import get_batch_loss  # noqa: E402
bt = st_npo.make_batch(tok, forget, "en", "cpu", trim=True)
assert bt["labels"].shape[1] < b["labels"].shape[1]
with torch.no_grad():
    full = get_batch_loss(base(**{k: v for k, v in b.items() if k != "labels"}).logits, b["labels"])
    kept = get_batch_loss(st_npo.forward_keep(base, bt), bt["labels"])
assert torch.allclose(full, kept, atol=1e-5), (full, kept)
print("1 make_batch: labels on answer tokens only, padding masked; trimmed logits give identical loss")

# 2. one step through SafeTune == one step of our loss (both variants), and frozen params untouched
for variant in ("npo", "npo_grad_diff"):
    m1, ref1 = fresh()
    st_npo.npo_localized(m1, tok, forget, retain, NAMES, reference=ref1, variant=variant, beta=BETA, lr=LR,
                         weight_decay=0.0, optimizer="sgd", max_steps=1, eval_every=10)
    m2, ref2 = fresh()
    ours_one_step(m2, ref2, variant)
    p1, p2, p0 = dict(m1.named_parameters()), dict(m2.named_parameters()), dict(base.named_parameters())
    moved = max((p1[n] - p0[n]).abs().max().item() for n in NAMES)
    assert moved > 1e-6, "SafeTune step did not move the trained params"
    for n in p0:
        if n in NAMES:
            assert torch.allclose(p1[n], p2[n], atol=1e-6, rtol=1e-4), (variant, n, (p1[n] - p2[n]).abs().max())
        else:
            assert torch.equal(p1[n], p0[n]), ("frozen param changed", n)
    print(f"2 {variant}: SafeTune step == our step (max update {moved:.2e}); all other params unchanged")

# 3. early stop inside one call: stop_fn sees completed step counts; reference never moves
m3, ref3 = fresh()
ref_before = {n: p.clone() for n, p in ref3.named_parameters()}
seen = []
_, rec = st_npo.npo_localized(m3, tok, forget, retain, NAMES, reference=ref3, lr=LR, optimizer="sgd",
                              weight_decay=0.0, max_steps=20, eval_every=3,
                              stop_fn=lambda s: seen.append(s) or s >= 6)
assert seen == [3, 6] and rec.stopped_at == 6, (seen, rec.stopped_at)
assert all(torch.equal(p, ref_before[n]) for n, p in ref3.named_parameters()), "reference moved"
assert all(not p.requires_grad for p in m3.parameters())
print("3 EvalStop: evaluated at steps", seen, "stopped at", rec.stopped_at, "; reference unchanged")

# 4. 6 steps via SafeTune == 6 steps of our loop (multi-step SGD)
m4, ref4 = fresh()
st_npo.npo_localized(m4, tok, forget, retain, NAMES, reference=ref4, variant="npo", beta=BETA, lr=LR,
                     weight_decay=0.0, optimizer="sgd", max_steps=6, eval_every=100)
m5, ref5 = fresh()
for _ in range(6):
    ours_one_step(m5, ref5, "npo")
for n in NAMES:
    assert torch.allclose(dict(m4.named_parameters())[n], dict(m5.named_parameters())[n], atol=1e-5, rtol=1e-4)
print("4 six SafeTune steps == six of ours")
print("ALL ST_NPO CHECKS PASSED")
