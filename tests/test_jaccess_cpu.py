"""CPU checks for unlearn_audit.jaccess on a tiny random Cohere2 model (no downloads).
Needs jlens (pip install the reference repo)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import test_core_cpu as T  # noqa: E402  (tiny Cohere2 model + stub tokenizer; runs core checks)
from unlearn_audit import core, jaccess  # noqa: E402
import jlens  # noqa: E402
from jlens.hf import HFLensModel  # noqa: E402

T.StubTok.decode = lambda self, ids: "".join(chr(int(t) - 10) for t in ids)
model, tok = T.model, T.tok


class WordTok:
    """Word-level fake tokenizer: ' Paris' -> [' ', 'Paris'] style pieces, to test concept_tokens rules."""
    def __init__(self):
        self.vocab, self.inv = {}, {}

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 100
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def __call__(self, text, add_special_tokens=False):
        pieces = ([" "] if text.startswith(" ") else []) + [w for w in text.split(" ") if w]
        pieces = [p for w in pieces for p in ([w[:2], w[2:]] if len(w) > 6 else [w])]  # split long words
        return {"input_ids": [self._id(p) for p in pieces]}

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)


wt = WordTok()
ct = jaccess.concept_tokens(wt, {"aliases_en": ["Addis Ababa"], "aliases_other": [], "aliases_ta": []})
dec = [wt.decode([t]) for t in ct]
assert all(d.strip() for d in dec), dec            # no whitespace-only token
assert "Addis" in dec and "Ababa" in dec, dec
print("0 concept_tokens: name parts kept, whitespace-only pieces excluded:", dec)
# strict rule: generic tokens shared by >= GENERIC_MIN facts are dropped; paper rule keeps them
fx = [{"aliases_en": [n + " City"], "aliases_other": [], "aliases_ta": []} for n in ("Alpha", "Bravo", "Delta", "Gamma")]
jaccess.RULE = "paper"
assert "City" in [wt.decode([t]) for t in jaccess.concept_tokens(wt, fx[0])]
jaccess.RULE = "strict"
gen = jaccess.set_generic(wt, fx, min_facts=3)
strict0 = [wt.decode([t]) for t in jaccess.concept_tokens(wt, fx[0])]
assert "City" not in strict0 and "Alpha" in strict0, strict0
jaccess.EXCLUDE.clear()
jaccess.RULE = "strict"
bare_first = jaccess.concept_tokens(wt, {"aliases_en": ["Ab Cdefgh"], "aliases_other": [], "aliases_ta": []})
assert wt._id("Ab") in bare_first, "strict rule must keep the short first token of the BARE variant"
jaccess.RULE = "paper"
assert wt._id("Ab") not in jaccess.concept_tokens(wt, {"aliases_en": ["Ab Cdefgh"], "aliases_other": [], "aliases_ta": []})
jaccess.RULE = "strict"
assert jaccess._letters("ிஸ்") == 1 and jaccess._letters("பார") == 2 and jaccess._letters("Paris") == 5
print("0b strict rule drops generic shared tokens, keeps distinctive ones:", strict0)
_real_ct = jaccess.concept_tokens
jaccess.concept_tokens = lambda tok_, f, langs=("en", "other", "ta"): [10 + ord(c) for c in f["aliases_en"][0][:2]]
facts = core.load_facts(os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl"))[:6]


class M(HFLensModel):
    def encode(self, text, *, max_length=512):
        return torch.tensor([[2] + [10 + (ord(c) % 900) for c in text][: max_length - 1]])


# 1. hidden_at matches hooks for every layer, including the last
ids_, npre = jaccess.probe_ids(tok, facts[0], "en")
ids, mask, pos = core.pad_left([ids_], tok.pad_token_id, "cpu")
H = core.hidden_at(model, ids, mask, pos)
cap = {}
hk = model.model.layers[3].register_forward_hook(lambda m, i, o: cap.__setitem__(3, o[0] if isinstance(o, tuple) else o))
with torch.no_grad():
    model(input_ids=ids, attention_mask=mask, position_ids=pos)
hk.remove()
assert torch.allclose(H[0, 3], cap[3], atol=1e-5)
print("1 hidden_at == hooked output (incl. last layer)")

# 2. final-layer-target lens fits on Cohere2 (J-Access recipe)
lens = jlens.fit(M(model, tok, force_bos=False), ["generic text for the lens fit, repeated. " * 2] * 2,
                 target_layer=-1, dim_batch=16, max_seq_len=64)
assert lens.source_layers == [0, 1, 2], lens.source_layers
J = {l: lens.jacobians[l].float() for l in lens.source_layers}
layers = lens.source_layers
print("2 lens fitted, source layers", layers)

# 3. ranks at the teacher-forced prefix positions == brute force (J lens and logit lens), mixed prefix lengths
R = jaccess.ranks(model, tok, facts, "en", J, layers, bs=4)
R0 = jaccess.ranks(model, tok, facts, "en", None, layers, bs=5)
for fi in (0, 4):
    ids_, npre = jaccess.probe_ids(tok, facts[fi], "en")
    n = len(ids_)
    Hs = core.hidden_at(model, torch.tensor([ids_]), torch.ones(1, n, dtype=torch.long), torch.arange(n)[None])
    C = jaccess.concept_tokens(tok, facts[fi])  # patched: fixed concept set for the stub tokenizer
    for li, l in enumerate(layers):
        for JJ, RR in ((J, R), (None, R0)):
            h = Hs[0, l, -npre:].float()
            if JJ is not None:
                h = h @ JJ[l].T
            with torch.no_grad():
                lg = model.lm_head(model.model.norm(h)).float()
            best = min(int((lg[q] > lg[q, t]).sum()) + 1 for q in range(npre) for t in C)
            assert RR[fi, li].item() == best, (fi, l, RR[fi, li].item(), best)
print("3 prefix-position ranks match brute force for both lenses")

# 4. band choice, decoys, silent rule, access rate, JOcc
cal = torch.tensor([[1, 50, 3], [2, 40, 30], [100, 5, 2]])
assert jaccess.choose_band(cal, [5, 6, 7], k=10, n=2) == [5, 7]
perm = jaccess.derangement(6, 0)
assert sorted(perm) == list(range(6)) and all(i != j for i, j in enumerate(perm))
Rd = jaccess.ranks(model, tok, facts, "en", J, layers, concept_facts=[facts[j] for j in perm])
assert Rd.shape == R.shape
name = facts[0]["aliases_en"][0]
assert jaccess.silent_by_tokens(tok, [name, "zzzz"], [facts[0], facts[0]]) == [False, True]
assert jaccess.access_rate(torch.tensor([[5, 50], [20, 30]]), [True, True], [0], 10) == 0.5
assert jaccess.access_rate(torch.tensor([[5, 50]]), [False], [0], 10) is None
assert abs(jaccess.jocc(0.6, 0.9, 0.1) - 0.625) < 1e-9 and jaccess.jocc(0.5, 0.5, 0.5) is None
print("4 band choice, derangement, decoys, silent-by-tokens, access rate, JOcc ok")
print("ALL JACCESS CHECKS PASSED")
