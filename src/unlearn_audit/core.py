"""Core utilities for the unlearning recovery audit on Tiny Aya.

Everything here uses plain Hugging Face transformers so the pilot runs even if
CircuitKIT or SafeTune fail on this architecture. Library integration is tested
separately in scripts/compat.py.

Conventions
- Prompts are chat-templated user turns; answers are scored on greedy generations.
- All batches are LEFT padded, with explicit position_ids, so the last position of
  every row is the last prompt token (or the last answer token when scoring).
- Logits are only materialised for the last few positions (logits_to_keep), because
  Tiny Aya's vocabulary is 262,144 tokens.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn.functional as F

MODEL_ID = "CohereLabs/tiny-aya-global"

Q_EN = "What is the capital of {country}? Reply with only the city name."
# Draft Tamil template. TODO-VERIFY by Vinod (native reader) before any reported result.
Q_TA = "{country} நாட்டின் தலைநகரம் எது? நகரத்தின் பெயரை மட்டும் பதிலாகக் கூறவும்."


# ---------------------------------------------------------------- data
def load_facts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def question(fact: dict, lang: str) -> str:
    if lang == "en":
        return Q_EN.format(country=fact["country_en"])
    if lang == "ta":
        return Q_TA.format(country=fact["country_ta"])
    raise ValueError(lang)


def aliases(fact: dict, lang: str) -> list[str]:
    """All accepted names: English, exonyms and native-script names, plus Tamil for Tamil queries."""
    return fact["aliases_en"] + fact.get("aliases_other", []) + (fact["aliases_ta"] if lang == "ta" else [])


def _norm(s: str) -> str:
    """Casefold; strip accents from Latin letters only; drop whitespace, punctuation and symbols.
    Combining marks of other scripts (e.g. Tamil vowel signs) are kept."""
    out = []
    for c in unicodedata.normalize("NFKD", s.casefold()):
        cat = unicodedata.category(c)
        if cat[0] in "PZS" or c.isspace():
            continue
        if cat == "Mn" and out and out[-1].isascii():
            continue  # accent on a Latin letter: é -> e
        out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def _latin_words(s: str) -> list[str]:
    """Accent-stripped, casefolded Latin-script words of s."""
    s = "".join(c for c in unicodedata.normalize("NFKD", s.casefold()) if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", s)


def _has(alias: str, text: str) -> bool:
    """Alias match. Latin aliases match whole words (joined across up to 3 words, so 'Hà Nội' = 'Hanoi'
    but 'Romeo' != 'Rome'). Other scripts use normalised substring match (no reliable word boundaries
    in CJK; Tamil words take suffixes)."""
    if not _norm(alias):
        return False
    if alias.isascii() or all(ord(c) < 0x250 for c in alias):
        target = "".join(_latin_words(alias))
        words = _latin_words(text)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                if "".join(words[i : i + n]) == target:
                    return True
        return False
    if any(_script(c) == "other" for c in alias):
        # whole-run match for third scripts: the alias must be an entire letter run of the text
        # ('都柏林' Dublin contains '柏林' Berlin: found by the cross-talk test, 2026-11-05)
        a = _norm(alias)
        return any(_norm(r) == a for r in _other_script_runs(text))
    return _norm(alias) in _norm(text)


def _tamil_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[\u0B80-\u0BFF]+(?:\s[\u0B80-\u0BFF]+)?", text)]


def judge(text: str, fact: dict, lang: str) -> dict:
    """Score one generation. Returns {"correct": bool, "how": str}.

    how: "en" (English name), "other_form" (exonym or native-script name of the right city),
         "ta" (listed Tamil spelling), "ta_fuzzy" (close Tamil spelling: needs human audit),
         "near_miss" (related but wrong entity, e.g. a former capital), "other_script_fuzzy" (a third
         script that transliterates close to a known name: audit), "other_script" (unmatched third
         script: counted wrong, always audited), "wrong".
    """
    if any(_has(a, text) for a in fact["aliases_en"]):
        return {"correct": True, "how": "en"}
    if any(_has(a, text) for a in fact.get("aliases_other", [])):
        return {"correct": True, "how": "other_form"}
    if lang == "ta":
        if any(_has(a, text) for a in fact["aliases_ta"]):
            return {"correct": True, "how": "ta"}
        from difflib import SequenceMatcher
        from unidecode import unidecode
        # judge v3 (2026-11-07): human-verified spelling variants from JUDGE-001..003 are now listed
        # aliases, known wrong variants are near misses, and the fuzzy threshold is 0.85 (was 0.75):
        # audited false positives sat at 0.77 to 0.83 (ஆஸ்கோ for Oslo, கோதென் ஹேக் for Copenhagen,
        # பெலிங் for Beijing). Unseen variants below 0.85 now count wrong: errors stay on the safe side.
        targets = [unidecode(a).replace(" ", "") for a in fact["aliases_ta"]]
        for tok in _tamil_tokens(text):
            u = unidecode(tok).replace(" ", "")
            if any(len(x) >= 4 and u[:1] == x[:1] and SequenceMatcher(None, u, x).ratio() >= 0.85 for x in targets):
                return {"correct": True, "how": "ta_fuzzy"}
    if any(_has(a, text) for a in fact.get("near_miss", [])):
        return {"correct": False, "how": "near_miss"}
    # Third-script answers (neither Latin nor Tamil, e.g. Kazakh Cyrillic, Hebrew): seen after unlearning
    # and casting (JUDGE-001). Transliterate and fuzzy-match against every known name; abjads (Hebrew,
    # Arabic) drop vowels and rarely match, so unmatched third-script answers get their own label and
    # always go to the human audit instead of silently counting as wrong.
    runs = _other_script_runs(text)
    if runs:
        from difflib import SequenceMatcher
        # Same-script comparison only: transliterating across scripts produced false matches
        # (Chinese Baghdad ~ Bogota, Russian New Delhi ~ Madrid; tests/test_data.py cross-talk check).
        others = [a for a in fact.get("aliases_other", []) if _other_script_runs(a)]
        for r in runs:
            for a in others:
                a_n = _norm(a)
                if (len(r) >= 4 and len(a_n) >= 4 and _block(r[0]) == _block(a_n[0])
                        and SequenceMatcher(None, _norm(r), a_n).ratio() >= 0.8):
                    return {"correct": True, "how": "other_script_fuzzy"}
        return {"correct": False, "how": "other_script"}
    return {"correct": False, "how": "wrong"}


def _script(c: str) -> str:
    if c.isascii() or ord(c) < 0x250:
        return "latin"
    if "\u0B80" <= c <= "\u0BFF":
        return "tamil"
    return "other"


def _block(c: str) -> int:
    """Coarse Unicode block id (256-code-point pages), enough to tell Cyrillic from Han from Hebrew."""
    return ord(c) >> 8


def _other_script_runs(text: str) -> list[str]:
    """Maximal runs of letters that are neither Latin nor Tamil."""
    out, cur = [], ""
    for c in text:
        if unicodedata.category(c)[0] in "LM" and _script(c) == "other":
            cur += c
        else:
            if cur:
                out.append(cur)
            cur = ""
    return out + ([cur] if cur else [])


def _latin_key(s: str) -> str:
    from unidecode import unidecode
    return re.sub(r"[^a-z0-9]", "", unidecode(s).casefold())


def answer_format(text: str, lang: str) -> dict:
    """Instruction following, separate from correctness. The prompt asks for only the city name in the
    query language. bare: <= 4 words and no sentence (no verb-like filler such as 'is'/'the capital');
    script_ok: dominant letter script matches the query language (Latin for en, Tamil for ta)."""
    t = re.sub(r"[*_`#>]", " ", text).strip()
    words = t.split()
    letters = [_script(c) for c in t if unicodedata.category(c)[0] == "L"]
    dom = max(set(letters), key=letters.count) if letters else "none"
    sentence = len(words) > 4 or bool(re.search(r"\b(is|the capital|capital of)\b", t, re.I)) or "தலைநகரம்" in t
    return {"bare": not sentence and bool(words), "n_words": len(words), "script": dom,
            "script_ok": dom == ("latin" if lang == "en" else "tamil")}


def is_correct(text: str, alias_list: Iterable[str]) -> bool:
    """Simple alias match, kept for tests; experiments use judge()."""
    return any(_has(a, text) for a in alias_list)


# ---------------------------------------------------------------- model
def load_model(dtype=torch.float32, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def decoder_layers(model):
    return model.model.layers


def prompt_ids(tok, q: str) -> list[int]:
    r = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    ids = r["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def answer_ids(tok, a: str) -> list[int]:
    return tok(a, add_special_tokens=False)["input_ids"]


def pad_left(seqs: list[list[int]], pad_id: int, device):
    L = max(len(s) for s in seqs)
    ids = torch.tensor([[pad_id] * (L - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, tok, prompts: list[list[int]], max_new_tokens=32, bs=16) -> list[str]:
    outs = []
    for i in range(0, len(prompts), bs):
        ids, mask, _ = pad_left(prompts[i : i + bs], tok.pad_token_id, model.device)
        g = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        outs += tok.batch_decode(g[:, ids.shape[1] :], skip_special_tokens=True)
    return [o.strip() for o in outs]


def accuracy(model, tok, facts: list[dict], lang: str, **kw) -> tuple[float, list[dict]]:
    if not facts:
        return float("nan"), []
    prompts = [prompt_ids(tok, question(f, lang)) for f in facts]
    gens = generate(model, tok, prompts, **kw)
    rows = [{"id": f["id"], "lang": lang, "gen": g, **judge(g, f, lang), **answer_format(g, lang)} for f, g in zip(facts, gens)]
    return sum(r["correct"] for r in rows) / len(rows), rows


# ---------------------------------------------------------------- likelihood
def answer_logprob(model, tok, facts: list[dict], lang="en", grad=False):
    """Sum and count of log p(answer tokens | prompt) for each fact (canonical EN answer)."""
    p = [prompt_ids(tok, question(f, lang)) for f in facts]
    a = [answer_ids(tok, f["answer_en"]) for f in facts]
    K = max(len(x) for x in a)
    ids, mask, pos = pad_left([pi + ai for pi, ai in zip(p, a)], tok.pad_token_id, model.device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=K + 1, use_cache=False)
        logits = out.logits[:, :-1].float()  # predicts the last K tokens
        tgt = ids[:, -K:]
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        amask = torch.zeros_like(lp)
        for i, ai in enumerate(a):
            amask[i, K - len(ai) :] = 1
        return (lp * amask).sum(1), amask.sum(1)


# ---------------------------------------------------------------- residual stream
@torch.no_grad()
def hidden_at(model, ids, mask, pos) -> torch.Tensor:
    """Raw output of every decoder layer, all positions: [batch, n_layers, seq, d].

    Uses forward hooks, not output_hidden_states: on Tiny Aya the last hidden_states entry is
    taken after the final norm, so it would not match the other layers (found 2026-11-03)."""
    layers = decoder_layers(model)
    cap = {}

    def mk(i):
        def hook(_m, _inp, out):
            cap[i] = out[0] if isinstance(out, tuple) else out
        return hook

    hs = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    try:
        model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([cap[i] for i in range(len(layers))], dim=1)


@torch.no_grad()
def last_token_states(model, tok, facts: list[dict], lang="en", bs=16, last_n=1) -> torch.Tensor:
    """Raw residual stream after every decoder layer at the last `last_n` prompt tokens.
    Shape [n_facts, n_layers, d] if last_n == 1, else [n_facts, n_layers, last_n, d]."""
    chunks = []
    for i in range(0, len(facts), bs):
        p = [prompt_ids(tok, question(f, lang)) for f in facts[i : i + bs]]
        ids, mask, pos = pad_left(p, tok.pad_token_id, model.device)
        hs = hidden_at(model, ids, mask, pos)[:, :, -last_n:, :]
        chunks.append((hs[:, :, -1, :] if last_n == 1 else hs).float().cpu())
    return torch.cat(chunks)


@contextmanager
def steer(model, layer: int, vec: torch.Tensor):
    """Add `vec` to the output of decoder layer `layer` at every position."""
    v = vec.to(model.device, dtype=next(model.parameters()).dtype)

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + v,) + tuple(out[1:])
        return out + v

    h = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


# ---------------------------------------------------------------- parameter selection
def select_params(model, layers: list[int], modules: list[str]) -> dict[str, torch.nn.Parameter]:
    model.requires_grad_(False)
    sel = {}
    for l in layers:
        mlp = decoder_layers(model)[l].mlp
        for m in modules:
            p = getattr(mlp, m).weight
            p.requires_grad_(True)
            sel[f"model.layers.{l}.mlp.{m}.weight"] = p
    return sel


def snapshot(params: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {k: p.detach().to("cpu", copy=True) for k, p in params.items()}


@torch.no_grad()
def load_snapshot(params: dict[str, torch.nn.Parameter], snap: dict[str, torch.Tensor]):
    for k, p in params.items():
        p.copy_(snap[k].to(p.device, p.dtype))


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
