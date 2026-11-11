"""Blind labelling sheet for a run's audit.csv, and merge of the labels back.

    python scripts/audit_sheet.py make  <run_dir>   # writes <run_dir>/audit_label.csv
    python scripts/audit_sheet.py merge <run_dir>   # copies human_correct + note into audit.csv

The sheet shows what a labeller needs (country, the exact question, the canonical answer in English and
Tamil, the model's output) and hides what would anchor them: the judge's verdict (auto_how,
auto_correct), the judge's alias list, and the condition (stage: baseline / unlearned / steered / bf16, and
split), because knowing a row comes from the unlearned model primes "wrong". Rows are shuffled with a fixed
seed and keep a `row` number so the merge is exact.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl")
COLS = ["row", "country_en", "country_ta", "lang", "question", "correct_answer_en", "correct_answer_ta",
        "model_output", "human_correct", "note", "id"]


def make(run_dir):
    from unlearn_audit import core  # only `make` needs it (merge runs without torch)

    facts = {f["id"]: f for f in core.load_facts(DATA)}
    src, dst = os.path.join(run_dir, "audit.csv"), os.path.join(run_dir, "audit_label.csv")
    assert not os.path.exists(dst), f"{dst} exists; not overwriting labels"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    import random
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:  # -sig: Excel/Numbers show Tamil correctly
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i in order:
            r = rows[i]
            fa = facts[r["id"]]
            w.writerow({"row": i, "country_en": fa["country_en"], "country_ta": fa["country_ta"], "lang": r["lang"],
                        "question": core.question(fa, r["lang"]), "correct_answer_en": fa["answer_en"],
                        "correct_answer_ta": fa["aliases_ta"][0], "model_output": r["gen"],
                        "human_correct": r.get("human_correct", ""),
                        "note": "" if r.get("note") == "sampled" else r.get("note", ""), "id": r["id"]})
    print(f"wrote {dst}: {len(rows)} rows (judge columns hidden)")


def _num(s: str):
    """Spreadsheets turn numeric outputs into numbers ("12." -> "12", "010" -> "10"; RUN-004 audit)."""
    try:
        return float(s.strip().rstrip("."))
    except ValueError:
        return None


def merge(run_dir):
    src, lab = os.path.join(run_dir, "audit.csv"), os.path.join(run_dir, "audit_label.csv")
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    labels = list(csv.DictReader(open(lab, encoding="utf-8-sig")))
    assert len(labels) == len(rows), (len(labels), len(rows))
    empty = [i + 2 for i, lr in enumerate(labels) if not lr["row"].strip()]
    assert not empty, f"empty 'row' cell on sheet lines {empty} (spreadsheet edit?); restore it before merging"
    assert sorted(int(lr["row"]) for lr in labels) == list(range(len(rows))), "row numbers duplicated or missing"
    for lr in labels:
        r = rows[int(lr["row"])]
        same = r["gen"] == lr["model_output"] or (_num(r["gen"]) is not None and _num(r["gen"]) == _num(lr["model_output"]))
        assert (r["id"], r["lang"]) == (lr["id"], lr["lang"]) and same, f"row {lr['row']} mismatch"
        r["human_correct"] = lr["human_correct"].strip()
        r["note"] = "; ".join(x for x in (r.get("note", ""), lr["note"].strip()) if x)
    missing = sum(not r["human_correct"] for r in rows)
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"merged labels into {src}; unlabelled rows: {missing}")


if __name__ == "__main__":
    {"make": make, "merge": merge}[sys.argv[1]](sys.argv[2])
