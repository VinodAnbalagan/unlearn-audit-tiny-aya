"""Validate the automatic judge against human labels in an audit.csv, following
Husain and Shankar, "Evals for AI Engineers", ch. 4-5.

Positive = "correct" (the answer names the right entity).
Reports TPR, TNR, Cohen's kappa, and for each split a judge-corrected accuracy
(Rogan and Gladen 1978):  theta = (p_obs + TNR - 1) / (TPR + TNR - 1), clipped to [0, 1],
with a 95% bootstrap CI that resamples the labelled pairs (TPR, TNR) and the split's outputs.

Leakage rule: only audit files from runs whose outputs were NOT used to write alias lists
count as a test set. RUN-001 is tuning data.

    python scripts/judge_agreement.py path/to/audit.csv [--B 2000]
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict


def as_bool(x: str):
    x = (x or "").strip().lower()
    if x in ("1", "true", "yes", "y", "t", "correct", "pass"):
        return True
    if x in ("0", "false", "no", "n", "f", "wrong", "fail"):
        return False
    return None


def rates(pairs):
    P = [j for h, j in pairs if h]
    N = [j for h, j in pairs if not h]
    tpr = sum(P) / len(P) if P else None
    tnr = sum(1 for j in N if not j) / len(N) if N else None
    return tpr, tnr


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(h == j for h, j in pairs) / n
    ph, pj = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    return None if pe == 1 else (po - pe) / (1 - pe)


def corrected(p_obs, tpr, tnr):
    if tpr is None or tnr is None or tpr + tnr - 1 < 0.05:
        return None
    return min(1.0, max(0.0, (p_obs + tnr - 1) / (tpr + tnr - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audit")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.audit, encoding="utf-8")))
    for r in rows:
        r["_auto"] = as_bool(r["auto_correct"])
        r["_human"] = as_bool(r["human_correct"])
    lab = [r for r in rows if r["_human"] is not None]
    pairs = [(r["_human"], r["_auto"]) for r in lab]
    print(f"labelled {len(lab)} of {len(rows)} rows")
    if not pairs:
        print("no human labels yet: fill the human_correct column (1/0)")
        return
    tpr, tnr = rates(pairs)
    print(f"TPR {tpr}  TNR {tnr}  kappa {kappa(pairs)}  raw agreement {sum(h == j for h, j in pairs) / len(pairs):.3f}")
    for lang in sorted({r["lang"] for r in lab}):
        pl = [(r["_human"], r["_auto"]) for r in lab if r["lang"] == lang]
        print(f"  {lang}: n={len(pl)} TPR/TNR {rates(pl)} kappa {kappa(pl)}")
    disagreements = [r for r in lab if r["_human"] != r["_auto"]]
    print(f"\n{len(disagreements)} disagreements (each should become a golden case in tests/test_judge.py):")
    for r in disagreements:
        print(f"  {r['stage']}/{r['split']}/{r['id']}/{r['lang']}: {r['gen']!r} auto={r['auto_how']} human={r['human_correct']} {r['note']}")

    rng = random.Random(a.seed)
    groups = defaultdict(list)
    for r in rows:
        if r["_auto"] is not None:
            groups[(r["stage"], r["split"], r["lang"])].append(r["_auto"])
    print("\njudge-corrected accuracy (95% bootstrap CI):")
    for k, autos in sorted(groups.items()):
        p_obs = sum(autos) / len(autos)
        est = corrected(p_obs, tpr, tnr)
        boots = []
        for _ in range(a.B):
            pb = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            ab = [autos[rng.randrange(len(autos))] for _ in autos]
            t1, t2 = rates(pb)
            c = corrected(sum(ab) / len(ab), t1, t2)
            if c is not None:
                boots.append(c)
        boots.sort()
        ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (None, None)
        print(f"  {'/'.join(k):28s} n={len(autos):3d} observed {p_obs:.2f} corrected {est} CI {ci}")


if __name__ == "__main__":
    main()
