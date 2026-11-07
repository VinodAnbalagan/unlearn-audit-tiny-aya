"""Tokenizer-only checks of the fictional dataset on the REAL Tiny Aya tokenizer (seconds, laptop).

    python scripts/check_fictional.py
Exit 1 if any English answer is not ONE token with a leading space (CircuitKIT EAP needs single-token
answers; COMPAT-002 showed only the leading-space form is reliably single-token).
"""
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core  # noqa: E402

tok = AutoTokenizer.from_pretrained(core.MODEL_ID)
F = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "..", "data", "fictional_facts.jsonl"), encoding="utf-8")]
answers = sorted({f["answer_en"] for f in F})
bad = []
for a in answers:
    ids = tok(" " + a, add_special_tokens=False)["input_ids"]
    print(f"{a!r:12} ' {a}' -> {len(ids)} token(s) {[tok.decode([i]) for i in ids]}")
    if len(ids) != 1:
        bad.append(a)
lens = sorted(len(tok(" " + f["entity_en"], add_special_tokens=False)["input_ids"]) for f in F)
print("entity name tokens (EN) min/median/max:", lens[0], lens[len(lens) // 2], lens[-1])
print("NOT single-token:", bad)
# spare candidates, so a failed word can be swapped without another round trip
for a in ["cello", "sitar", "flute", "bass", "mouse", "turtle", "cow", "duck", "snake", "hamster"]:
    ids = tok(" " + a, add_special_tokens=False)["input_ids"]
    print(f"  spare {a!r:10} -> {len(ids)} token(s) {[tok.decode([i]) for i in ids]}")
sys.exit(1 if bad else 0)
