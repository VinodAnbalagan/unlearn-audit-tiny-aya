"""CPU checks for unlearn_audit.name_rank on the tiny random Cohere2 model (needs jlens)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import test_core_cpu as T  # noqa: E402
from unlearn_audit import core, jaccess, name_rank  # noqa: E402

model, tok = T.model, T.tok
facts = core.load_facts(os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl"))[:6]
L = len(core.decoder_layers(model))
d = model.config.hidden_size
torch.manual_seed(0)
J = {l: torch.eye(d) + 0.05 * torch.randn(d, d) for l in range(L - 1)}
layers = [1, 2]

S = name_rank.name_scores(model, tok, facts[:2], "en", facts, J, layers, bs=4)
assert S.shape == (2, 6, 3), S.shape

# 1. output column == brute force from the model's own logits (unpadded, one sequence at a time)
for fi in range(2):
    probe, _ = jaccess.probe_ids(tok, facts[fi], "en")
    for ci, c in enumerate(facts):
        ids_c = tok(" " + c["answer_en"], add_special_tokens=False)["input_ids"]
        x = torch.tensor([probe + ids_c])
        with torch.no_grad():
            lg = model(input_ids=x).logits[0].float()
        lp = torch.log_softmax(lg[len(probe) - 1 : len(probe) + len(ids_c) - 1], -1)
        ref = lp.gather(-1, torch.tensor(ids_c)[:, None]).sum()
        assert abs(ref.item() - S[fi, ci, -1].item()) < 1e-3, (fi, ci, ref.item(), S[fi, ci, -1].item())
print("1 output column == model log-prob of the full name (batched, left padded, all candidates)")

# 2. lens column == brute force with J at that layer
probe, _ = jaccess.probe_ids(tok, facts[0], "en")
ids_c = tok(" " + facts[3]["answer_en"], add_special_tokens=False)["input_ids"]
seq = probe + ids_c
n = len(seq)
H = core.hidden_at(model, torch.tensor([seq]), torch.ones(1, n, dtype=torch.long), torch.arange(n)[None])
for ui, l in enumerate(layers):
    h = H[0, l, n - len(ids_c) - 1 : n - 1].float() @ J[l].T
    with torch.no_grad():
        lg = model.logit_scale * model.lm_head(model.model.norm(h)).float()
    ref = torch.log_softmax(lg, -1).gather(-1, torch.tensor(ids_c)[:, None]).sum()
    assert abs(ref.item() - S[0, 3, ui].item()) < 1e-3, (l, ref.item(), S[0, 3, ui].item())
print("2 J-lens columns == brute force at each layer")

# 3. batch size does not change scores (padding invariance across candidate lengths)
S2 = name_rank.name_scores(model, tok, facts[:2], "en", facts, J, layers, bs=6)
assert torch.allclose(S, S2, atol=1e-4)
print("3 scores independent of batch size")

# 4. summarize: ranks, top-1, margin
fake = torch.tensor([[[0.0], [-1.0], [-2.0]], [[-3.0], [-1.0], [-2.0]]])
r = name_rank.summarize(fake, [0, 0], [])
assert r["output"]["ranks"] == [1, 3] and r["output"]["top1"] == 0.5
assert abs(r["output"]["mean_margin"] - ((0 - -1) + (-3 - -1)) / 2) < 1e-9
print("4 summarize: rank, top-1, margin")
print("ALL NAME_RANK CHECKS PASSED")
