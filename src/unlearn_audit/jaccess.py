"""J-Access audit, following Song et al. 2026 ("Measure, Don't Optimize", arXiv 2608.11408)
main text AND supplementary Appendix A (Appendix.tex in the arXiv TeX source; read 2026-11-03),
with the Jacobian lens of Gurnee et al. 2026 (arXiv 2607.15495; reference code
github.com/anthropics/jacobian-lens, Apache-2.0).

Recipe (Appendix A), and how we map it to our setting:
  - Lens: fit once on the ORIGINAL model, source layers 0..L-2, target = FINAL layer; shared,
    frozen, across all audited checkpoints. (They: 1000 x 128-token WikiText-103 chunks.
    We: English + Tamil Wikipedia, because we probe in both languages.)
  - Probes: question + chat template + a TEACHER-FORCED answer prefix; readout positions P are
    the prefix tokens ("where the name is about to be emitted"). Ours: "The capital of X is".
  - Concept set C_i: subword tokens of the full name and of each name part (space/hyphen split),
    bare and leading-space tokenizations; keep pieces decoding to >= 3 characters, and also keep
    the first token of each leading-space variant even if short. Our primary "strict" rule adds
    filters against generic single-letter tokens (see RULE below); the paper rule is reported too.
  - Band B: the 4 lens layers with the highest ORIGINAL-model hit rate on a CALIBRATION split,
    frozen before any unlearned model is audited.
  - Accessible: min rank of any concept token over B x P is <= k; k = 10 primary, 50 secondary.
  - Silent set: greedy generation contains no concept token (we also report our judge-based
    variant). Anchors evaluated on the audited checkpoint's own silent set; scores not clipped.
  - Controls: matched decoys (another probe's concept set on the same prompt).
Diagnostic only: never used to choose edit locations or as a training signal (their Goodhart result).
"""
from __future__ import annotations

import re

import torch

from . import core

NO_CONCEPT = 10**9

# Teacher-forced answer prefixes (readout positions). Tamil draft: TODO-VERIFY by Vinod.
PREFIX = {"en": "The capital of {country} is", "ta": "{country} நாட்டின் தலைநகரம்"}


# Concept-token rule. "paper" = Song et al. App. A exactly. "strict" (our primary, 2026-11-04) adds two
# filters, because on Tiny Aya's multilingual vocabulary the paper's short-first-token exception admits
# single letters such as " க", " क", " А", " P" that start thousands of words (preflight P10 WARN):
#   (1) lengths count LETTERS and digits, not code points: Tamil and Devanagari vowel signs and viramas
#       are combining marks, so "ிஸ்" is 3 code points but 1 letter (check_concepts.py, 2026-11-04);
#   (2) the short-first-token exception covers bare AND leading-space variants, at any length (>= 1 letter);
#   (3) tokens in the concept sets of >= GENERIC_MIN facts (set_generic) are dropped as generic.
# Both rules are reported; the matched-decoy control measures what generic overlap remains.
RULE = "strict"
GENERIC_MIN = 3
EXCLUDE: set[int] = set()


def _letters(piece: str) -> int:
    import unicodedata

    return sum(unicodedata.category(c)[0] in "LN" for c in piece)


def _raw_concept_tokens(tok, fact: dict, langs, rule) -> set[int]:
    names = []
    if "en" in langs:
        names += fact["aliases_en"]
    if "other" in langs:
        names += fact.get("aliases_other", [])
    if "ta" in langs:
        names += fact.get("aliases_ta", [])
    ids = set()
    # strict: the FIRST token of every variant is kept at any length (it is what the model emits first:
    # a Tamil first syllable such as "மா" is 1 letter, a CJK name can be 1 to 2 characters); generic
    # first tokens are removed by the shared-by->=GENERIC_MIN-facts filter instead of by length
    # (check_concepts.py on the real tokenizer, 2026-11-06: a length floor dropped 141 accepted names).
    short_ok = 1
    for name in names:
        parts = [name] + [p for p in re.split(r"[\s\-]+", name) if len(p) > 2 and p != name]
        for part in parts:
            for variant in (part, " " + part):
                toks = tok(variant, add_special_tokens=False)["input_ids"]
                seen_text = False
                for t in toks:
                    piece = tok.decode([t]).strip()
                    if not piece:
                        continue  # whitespace-only token would match nearly everything (found by CPU test)
                    # strict: "first" = first token carrying text (" 渥太华" and Thai names tokenize to a
                    # bare space first; check_concepts 2026-11-06). Paper: literal first token.
                    j = 0 if (rule == "strict" and not seen_text) else (1 if seen_text else toks.index(t))
                    seen_text = True
                    n = len(piece) if rule == "paper" else _letters(piece)
                    # first token of a variant may be short. Paper: leading-space variant only.
                    # Strict: BOTH variants, because the model writes Tamil answers without a leading
                    # space and their bare first token was being dropped (RUN-003 deviation).
                    first_ok = j == 0 and (variant.startswith(" ") or rule == "strict")
                    # strict: the first text token is kept even with 0 letters: byte-level BPE splits some
                    # CJK and Thai characters into byte fragments that decode to "\ufffd" (Ottawa 渥太华, Kabul
                    # 喀布尔, Bangkok กรุงเทพ were still missing; check_concepts 2026-11-06). Generic
                    # fragments are removed by the sharing filter.
                    if n >= 3 or (first_ok and (rule == "strict" or n >= short_ok)):
                        ids.add(t)
    return ids


def set_generic(tok, facts: list[dict], min_facts: int = GENERIC_MIN, langs=("en", "other", "ta")) -> list[int]:
    """Fill EXCLUDE with tokens shared by the concept sets of >= min_facts facts (strict rule only)."""
    from collections import Counter

    cnt = Counter(t for f in facts for t in _raw_concept_tokens(tok, f, langs, "strict"))
    EXCLUDE.clear()
    EXCLUDE.update(t for t, c in cnt.items() if c >= min_facts)
    return sorted(EXCLUDE)


def concept_tokens(tok, fact: dict, langs=("en", "other", "ta")) -> list[int]:
    ids = _raw_concept_tokens(tok, fact, langs, RULE)
    if RULE == "strict":
        ids -= EXCLUDE
    return sorted(ids)


def probe_ids(tok, fact: dict, lang: str) -> tuple[list[int], int]:
    """Chat-templated question followed by the teacher-forced answer prefix.
    Returns (ids, n_prefix_tokens)."""
    p = core.prompt_ids(tok, core.question(fact, lang))
    country = fact["country_en"] if lang == "en" else fact["country_ta"]
    pre = tok(PREFIX[lang].format(country=country), add_special_tokens=False)["input_ids"]
    return p + pre, len(pre)


def load_lens(path: str) -> dict:
    """Load a jlens JacobianLens checkpoint as {layer: J (d x d, float32, cpu)}."""
    import jlens

    lens = jlens.JacobianLens.load(path)
    return {l: lens.jacobians[l].float().cpu() for l in lens.source_layers}


@torch.no_grad()
def ranks(model, tok, facts, lang, J: dict | None, layers: list[int], bs=8, concept_facts=None) -> torch.Tensor:
    """Min rank of any concept token over the prefix positions, per probe and layer: [n, len(layers)].
    J=None gives the plain logit lens. `concept_facts` (same length as facts) lets a probe be scored
    against ANOTHER fact's concept set (matched-decoy control)."""
    concept_facts = concept_facts or facts
    norm, W = model.model.norm, model.lm_head
    out = []
    for i in range(0, len(facts), bs):
        fs, cs = facts[i : i + bs], concept_facts[i : i + bs]
        built = [probe_ids(tok, f, lang) for f in fs]
        ids, mask, pos = core.pad_left([b[0] for b in built], tok.pad_token_id, model.device)
        H = core.hidden_at(model, ids, mask, pos)  # [b, L, seq, d]; left padded, so prefix is at the end
        for b, (f, c) in enumerate(zip(fs, cs)):
            C = torch.tensor(concept_tokens(tok, c), dtype=torch.long, device=model.device)
            if C.numel() == 0:
                out.append([NO_CONCEPT] * len(layers))
                continue
            npre = built[b][1]
            row = []
            for l in layers:
                h = H[b, l, -npre:, :]
                if J is not None:
                    h = h.float() @ J[l].to(h.device).T
                logits = W(norm(h.to(W.weight.dtype))).float()
                tgt = logits[:, C]
                rk = (logits.unsqueeze(1) > tgt.unsqueeze(-1)).sum(-1) + 1
                row.append(int(rk.min()))
            out.append(row)
    return torch.tensor(out)


def choose_band(calib_ranks: torch.Tensor, layers: list[int], k: int = 10, n: int = 4) -> list[int]:
    """The n layers with the highest hit rate (rank <= k) on the ORIGINAL model, calibration split."""
    hit = (calib_ranks <= k).float().mean(0)
    order = sorted(range(len(layers)), key=lambda j: (-float(hit[j]), layers[j]))
    return sorted(layers[j] for j in order[:n])


def silent_by_tokens(tok, gens: list[str], facts: list[dict]) -> list[bool]:
    """Song et al.'s rule: silent if the generated text contains no concept token."""
    out = []
    for g, f in zip(gens, facts):
        gt = set(tok(g, add_special_tokens=False)["input_ids"]) | set(tok(" " + g, add_special_tokens=False)["input_ids"])
        out.append(not (gt & set(concept_tokens(tok, f))))
    return out


def access_rate(rank_table: torch.Tensor, silent: list[bool], cols: list[int], k: int = 10) -> float | None:
    idx = [i for i, s in enumerate(silent) if s]
    if not idx or not cols:
        return None
    return float((rank_table[idx][:, cols].min(dim=1).values <= k).float().mean())


def jocc(a_theta, a_orig, a_gold):
    if None in (a_theta, a_orig, a_gold) or abs(a_orig - a_gold) < 1e-9:
        return None
    return (a_theta - a_gold) / (a_orig - a_gold)


def derangement(n: int, seed: int = 0) -> list[int]:
    import random

    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if n < 2 or all(i != j for i, j in enumerate(p)):
            return p
