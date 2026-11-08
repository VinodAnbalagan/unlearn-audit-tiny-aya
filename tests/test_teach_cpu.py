"""teach.py on the tiny CPU Cohere2 model: assistant-only labels, memory-light loss == HF loss, loss drops."""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import test_core_cpu as T  # noqa: E402
from unlearn_audit import core  # noqa: E402


def act(self, msgs, add_generation_prompt=False, tokenize=True, return_dict=False):
    ids = [2, 3] + self._enc(msgs[0]["content"]) + [4]
    if len(msgs) > 1:
        ids += self._enc(msgs[1]["content"]) + [1]
    return {"input_ids": ids} if return_dict else ids


T.StubTok.apply_chat_template = act
import teach  # noqa: E402

F = core.load_facts(os.path.join(os.path.dirname(__file__), "..", "data", "fictional_facts.jsonl"))[:2]
ex = teach.build_examples(T.tok, F, [], random.Random(0))
assert len(ex) == 2 * 2 * 4  # 2 facts x 2 languages x (2 questions + 2 statements)
for x, y in ex:
    n_prompt = sum(v == -100 for v in y)
    assert all(v == -100 for v in y[:n_prompt]) and all(v != -100 for v in y[n_prompt:]) and len(x) == len(y)
print("1 assistant-only labels")
m = T.model
m.requires_grad_(True)
ids, mask, pos, lab = next(teach.batches(ex, 8, 0, "cpu"))
with torch.no_grad():
    a = m(input_ids=ids, attention_mask=mask, position_ids=pos, labels=lab).loss
    b = teach.sft_loss(m, ids, mask, pos, lab)
assert abs(a.item() - b.item()) < 1e-4, (a.item(), b.item())
print("2 last-K-positions loss == HF labels loss", round(a.item(), 4))
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
l0 = None
for it in range(30):
    for ids, mask, pos, lab in teach.batches(ex, 16, 0, "cpu"):
        loss = teach.sft_loss(m, ids, mask, pos, lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
    l0 = l0 or loss.item()
assert loss.item() < l0 - 0.3, (l0, loss.item())
print("3 loss drops when overfitting", round(l0, 3), "->", round(loss.item(), 3))
print("ALL TEACH CHECKS PASSED")
