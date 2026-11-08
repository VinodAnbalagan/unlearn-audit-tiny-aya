"""Whole-name ranking: does the model, inside (J-lens layers) or at the output, still prefer the TRUE
capital's FULL name over the other candidates?

Motivation (Vinod, 2026-11-06): a single-token readout cannot tell a city from an unrelated word that
shares its first piece (" பார" starts both பாரிஸ் Paris and பார்க்க "to see"). Here every candidate
name is teacher-forced piece by piece after the answer prefix, and at each position we read how likely
that specific next piece is, through the J-lens at chosen layers and through the model's own output.
Summed over pieces this is a whole-name log-score per layer. The other candidates (all 50 capitals in
the query language) are the built-in decoys, scored on the same prompt.

Readout per source layer l: log_softmax(logit_scale * lm_head(norm(h_l @ J_l^T))); final column: the
model's own logits (same formula on the last layer's raw output, which equals the model's logits).
Caveat (stated in the write-up): later pieces are easier once earlier ones are forced, so most of the
discrimination comes from the first one or two pieces; that is why candidates are compared with each
other, not read in isolation.
Not used to choose anything in training or steering (diagnostic only, like J-Access).
"""
from __future__ import annotations

import torch

from . import core, jaccess


def candidate_name(fact: dict, lang: str) -> str:
    if lang == "en":
        return fact["answer_en"]
    return fact.get("answer_ta") or fact["aliases_ta"][0]


@torch.no_grad()
def name_scores(model, tok, facts, lang, candidates, J: dict | None, layers: list[int], bs: int = 25) -> torch.Tensor:
    """Summed log-prob of each candidate's full name after each fact's probe.
    Returns [n_facts, n_candidates, len(layers) + 1]; the last column is the model output."""
    norm, W = model.model.norm, model.lm_head
    scale = float(getattr(model, "logit_scale", 1.0) or 1.0)
    L = len(core.decoder_layers(model))
    use = list(layers) + [L - 1]
    cand_ids = [tok(" " + candidate_name(c, lang), add_special_tokens=False)["input_ids"] for c in candidates]
    out = torch.zeros(len(facts), len(candidates), len(use))
    for fi, f in enumerate(facts):
        probe, _ = jaccess.probe_ids(tok, f, lang)
        for s in range(0, len(candidates), bs):
            cs = cand_ids[s : s + bs]
            ids, mask, pos = core.pad_left([probe + c for c in cs], tok.pad_token_id, model.device)
            H = core.hidden_at(model, ids, mask, pos)  # [b, L, T, d], left padded: candidate at the end
            T = ids.shape[1]
            for b, c in enumerate(cs):
                m = len(c)
                tgt = torch.tensor(c, device=model.device)
                for ui, l in enumerate(use):
                    h = H[b, l, T - m - 1 : T - 1, :].float()  # positions that predict each name piece
                    if J is not None and l != L - 1:
                        h = h @ J[l].to(h.device).T
                    elif J is None and l != L - 1:
                        pass  # logit lens
                    logits = scale * W(norm(h.to(W.weight.dtype))).float()
                    out[fi, s + b, ui] = torch.log_softmax(logits, -1).gather(-1, tgt[:, None]).sum()
    return out


def summarize(scores: torch.Tensor, true_idx: list[int], layers: list[int]) -> dict:
    """Per readout column: mean rank of the true name (1 = best), top-1 rate, mean margin over the best
    wrong candidate (log-prob units)."""
    n, k, u = scores.shape
    res = {}
    names = [f"L{l}" for l in layers] + ["output"]
    for ui, nm in enumerate(names):
        ranks, top1, margins = [], 0, []
        for i in range(n):
            s = scores[i, :, ui]
            t = s[true_idx[i]]
            r = int((s > t).sum()) + 1
            ranks.append(r)
            top1 += r == 1
            others = torch.cat([s[: true_idx[i]], s[true_idx[i] + 1 :]])
            margins.append(float(t - others.max()))
        res[nm] = {"mean_rank": sum(ranks) / n, "top1": top1 / n, "mean_margin": sum(margins) / n,
                   "ranks": ranks, "n_candidates": k}
    return res
