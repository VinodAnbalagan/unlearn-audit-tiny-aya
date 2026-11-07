"""CPU logic test of unlearn_audit.core on a tiny random Cohere2 model with a stub tokenizer."""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core
from transformers import Cohere2Config, Cohere2ForCausalLM


class StubTok:
    """Char-level tokenizer with a fake chat template. ids: 0 pad, 1 eos, 2 bos, 3 user tag, 4 assistant tag."""
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def _enc(self, s):
        return [10 + (ord(c) % 900) for c in s]

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, return_dict):
        ids = [2, 3] + self._enc(msgs[0]["content"]) + [4]
        return {"input_ids": ids}

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": self._enc(s)}

    def batch_decode(self, rows, skip_special_tokens=True):
        return ["".join(chr(int(t) - 10 + 32) if int(t) >= 10 else "" for t in r) for r in rows]


torch.manual_seed(0)
cfg = Cohere2Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                    sliding_window=16, pad_token_id=0, bos_token_id=2, eos_token_id=1)
model = Cohere2ForCausalLM(cfg).float().eval()
tok = StubTok()
facts = core.load_facts(os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl"))[:6]

# 1. logprob indexing equals a brute-force per-example full-logit computation
lp, n = core.answer_logprob(model, tok, facts)
for i, f in enumerate(facts):
    p = core.prompt_ids(tok, core.question(f, "en")); a = core.answer_ids(tok, f["answer_en"])
    ids = torch.tensor([p + a])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    ref = torch.log_softmax(logits[len(p) - 1 : len(p) + len(a) - 1], -1).gather(-1, torch.tensor(a).unsqueeze(-1)).sum()
    assert abs(ref.item() - lp[i].item()) < 1e-3, (i, ref.item(), lp[i].item())
    assert n[i].item() == len(a)
print("1 answer_logprob matches brute force (left padding + position_ids + logits_to_keep)")

# 2. last-token states match an unpadded single forward
H = core.last_token_states(model, tok, facts, bs=4)
p = core.prompt_ids(tok, core.question(facts[3], "en"))
with torch.no_grad():
    hs = model(input_ids=torch.tensor([p]), output_hidden_states=True).hidden_states
assert torch.allclose(H[3, 2], hs[3][0, -1].float(), atol=1e-4)
print("2 last_token_states correct; shape", tuple(H.shape))

# 3. steering hook changes outputs and is removed afterwards
before, _ = core.answer_logprob(model, tok, facts)
with core.steer(model, 1, torch.randn(64) * 5):
    during, _ = core.answer_logprob(model, tok, facts)
after, _ = core.answer_logprob(model, tok, facts)
assert not torch.allclose(before, during) and torch.allclose(before, after)
print("3 steering hook applies inside context, removed after")

# 4. param selection, NPO step lowers forget logprob, snapshot restore is exact
params = core.select_params(model, [1, 2], ["down_proj"])
assert all(p.requires_grad for p in params.values())
assert sum(p.requires_grad for p in model.parameters()) == 2
orig = core.snapshot(params)
ref_f, _ = core.answer_logprob(model, tok, facts[:3])
opt = torch.optim.AdamW(list(params.values()), lr=1e-2)
for _ in range(5):
    lp_f, _ = core.answer_logprob(model, tok, facts[:3], grad=True)
    loss = -(2 / 0.1) * F.logsigmoid(-0.1 * (lp_f - ref_f)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
new_f, _ = core.answer_logprob(model, tok, facts[:3])
assert (new_f < ref_f).all(), (new_f, ref_f)
core.load_snapshot(params, orig)
rest, _ = core.answer_logprob(model, tok, facts[:3])
assert torch.allclose(rest, ref_f, atol=1e-5)
print("4 NPO lowers forget log-prob; snapshot restore exact")

# 5. generate runs with left padding
outs = core.generate(model, tok, [core.prompt_ids(tok, core.question(f, "ta")) for f in facts], max_new_tokens=3, bs=4)
assert len(outs) == 6
print("5 generate ok; is_correct:", core.is_correct("The capital is Paris.", facts[0]["aliases_en"]), core.is_correct("பாரிஸ்", core.aliases(facts[0], "ta")))
print("ALL CHECKS PASSED")
