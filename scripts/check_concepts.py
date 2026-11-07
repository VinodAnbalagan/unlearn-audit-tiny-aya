"""Tokenizer-only check of J-Access concept sets (seconds, runs on a laptop; needs HF access to the gated tokenizer).

    python scripts/check_concepts.py
Prints, for the paper rule and our strict rule: set sizes, empty sets, pieces shared by many facts, short
pieces, and a few full examples. Exit 1 if any fact has an empty set or a whitespace-only piece.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformers import AutoTokenizer  # noqa: E402

from unlearn_audit import core, jaccess  # noqa: E402

tok = AutoTokenizer.from_pretrained(core.MODEL_ID)
facts = core.load_facts(os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl"))
def _first_text(v):
    """first token that decodes to visible text (a bare-space token is skipped)"""
    return next(t for t in tok(v, add_special_tokens=False)["input_ids"] if tok.decode([t]).strip())


bad = False
for rule in ("paper", "strict"):
    jaccess.RULE = rule
    excl = jaccess.set_generic(tok, facts) if rule == "strict" else []
    sets = {f["id"]: jaccess.concept_tokens(tok, f) for f in facts}
    dec = lambda ids: [tok.decode([t]) for t in ids]  # noqa: E731
    freq = Counter(t for v in sets.values() for t in v)
    empty = [k for k, v in sets.items() if not v]
    ws = [p for v in sets.values() for p in dec(v) if not p.strip()]
    print(f"\n=== rule={rule}  excluded_generic={dec(excl)}")
    print("sizes min/median/max:", min(map(len, sets.values())), sorted(map(len, sets.values()))[len(sets) // 2], max(map(len, sets.values())))
    print("empty:", empty, "| whitespace pieces:", ws)
    print("shared by >=3 facts:", sorted(((tok.decode([t]), c) for t, c in freq.items() if c >= 3), key=lambda x: -x[1])[:25])
    print("short (<3 letters):", sorted({p for v in sets.values() for p in dec(v) if jaccess._letters(p) < 3})[:60])
    for fid in ("fr", "in", "et", "ma", "kr", "jp", "eg", "ru"):
        if fid in sets:
            print(f"  {fid}:", dec(sets[fid]))
    # first-token coverage: the rank readout needs the token a name STARTS with (as the model would emit
    # it, with and without a leading space) unless it was dropped as generic
    miss = []
    for f in facts:
        C = set(sets[f["id"]])
        for x in f["aliases_en"] + f.get("aliases_other", []) + f["aliases_ta"]:
            firsts = {_first_text(v) for v in (x, " " + x)}
            if not (firsts & C) and not (firsts & set(excl)):
                miss.append((f["id"], x))
    print("names whose first token is missing (must be empty under strict):", miss[:20], "count", len(miss))
    gen_first = [(f["id"], x) for f in facts for x in f["aliases_en"] + f.get("aliases_other", []) + f["aliases_ta"]
                 if {_first_text(v) for v in (x, " " + x)} <= set(excl)]
    print("names whose first tokens were ALL dropped as generic (info):", gen_first[:15], "count", len(gen_first))
    # token-based silence (Song et al.'s rule), informational: we use the judge-defined silent set as primary
    selfsil = [(f["id"], x) for f in facts for x in f["aliases_en"] + f.get("aliases_other", []) + f["aliases_ta"]
               if jaccess.silent_by_tokens(tok, [x], [f])[0]]
    print("accepted names that the TOKEN silent rule calls silent (info):", selfsil[:10], "count", len(selfsil))
    bad |= bool(empty or ws or (miss and rule == "strict"))
sys.exit(1 if bad else 0)
