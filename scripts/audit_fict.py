"""Blind audit sheet for a fictional-facts main run (RUN-005), and merge of the labels back.

    python3 scripts/audit_fict.py make  <run_dir>   # writes <run_dir>/audit_label.csv (+ audit_key.csv)
    python3 scripts/audit_fict.py score <run_dir>   # judge vs human agreement once human_correct is filled

Rows: every forget-set reply of the unlearned models (3 sets x 24 facts x EN/TA, trained question), shuffled
with a fixed seed. Hidden from the labeller: the judge verdict, the set (circuit / random / lookup) and the
stage. Shown: the person, the question language, the canonical answer in English and Tamil, the reply.
Label human_correct 1 if the reply gives the canonical answer (any spelling a Tamil reader accepts), else 0.
No torch needed.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
COLS = ["row", "person", "lang", "correct_answer_en", "correct_answer_ta", "model_output", "human_correct", "note"]


def load(run_dir):
    facts = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "fictional_facts.jsonl"), encoding="utf-8")}
    gens = [json.loads(l) for l in open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8")]
    rows = [g for g in gens if g["stage"].startswith("unlearned_") and g["split"] == "forget" and g["kind"] == "eval"]
    return facts, rows


def make(run_dir):
    facts, rows = load(run_dir)
    dst, key = os.path.join(run_dir, "audit_label.csv"), os.path.join(run_dir, "audit_key.csv")
    assert not os.path.exists(dst), f"{dst} exists; not overwriting labels"
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    with open(dst, "w", newline="", encoding="utf-8-sig") as f, open(key, "w", newline="", encoding="utf-8") as k:
        w, wk = csv.DictWriter(f, fieldnames=COLS), csv.writer(k)
        w.writeheader()
        wk.writerow(["row", "id", "stage", "lang", "judge_correct"])
        for n, i in enumerate(order, 1):
            g = rows[i]
            fa = facts[g["id"]]
            w.writerow({"row": n, "person": fa["entity_en"] if g["lang"] == "en" else fa["entity_ta"], "lang": g["lang"],
                        "correct_answer_en": fa["answer_en"], "correct_answer_ta": fa["answer_ta"],
                        "model_output": g["gen"], "human_correct": "", "note": ""})
            wk.writerow([n, g["id"], g["stage"], g["lang"], int(g["correct"])])
    print(f"wrote {dst} ({len(rows)} rows) and the hidden key {key}; label human_correct 1/0")


def score(run_dir):
    lab = {int(r["row"]): r for r in csv.DictReader(open(os.path.join(run_dir, "audit_label.csv"), encoding="utf-8-sig"))}
    key = list(csv.DictReader(open(os.path.join(run_dir, "audit_key.csv"), encoding="utf-8")))
    pairs, bad = [], []
    for r in key:
        h = lab[int(r["row"])]["human_correct"].strip().rstrip(".")
        if h not in ("0", "1"):
            bad.append(r["row"])
            continue
        pairs.append((int(r["judge_correct"]), int(h), r))
    assert not bad, f"rows without a 0/1 label: {bad[:20]}"
    tp = sum(j and h for j, h, _ in pairs); tn = sum((not j) and (not h) for j, h, _ in pairs)
    fp = sum(j and not h for j, h, _ in pairs); fn = sum((not j) and h for j, h, _ in pairs)
    n = len(pairs); po = (tp + tn) / n
    pj, ph = (tp + fp) / n, (tp + fn) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"n {n} agree {po:.3f} kappa {kappa:.3f} | judge-yes/human-no {fp} | judge-no/human-yes {fn}")
    for j, h, r in pairs:
        if j != h:
            print("  disagree row", r["row"], r["stage"], r["lang"], r["id"], "judge", j, "human", h, "|", lab[int(r["row"])]["model_output"][:80])


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]](sys.argv[2])
