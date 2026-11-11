"""Build the HF Space (one static index.html) from the logged run outputs.

    python scripts/build_space.py --run5 <run dir> --figdata figures/data.json --out ../publish/space

The page embeds a JSON bundle: figures/data.json plus, for every forget fact, the model's actual English and
Tamil replies before and after unlearning under each layer set (from generations.jsonl). No number is typed in.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
SETS = ["circuit", "random", "lookup"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run5", required=True)
    ap.add_argument("--figdata", default=os.path.join(ROOT, "figures", "data.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = json.load(open(a.figdata, encoding="utf-8"))
    facts = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "fictional_facts.jsonl"), encoding="utf-8")}
    T = json.load(open(os.path.join(ROOT, "data", "fictional_templates.json"), encoding="utf-8"))
    gens = [json.loads(l) for l in open(os.path.join(a.run5, "generations.jsonl"), encoding="utf-8")]
    G = {(g["stage"], g["id"], g["lang"]): g for g in gens if g["split"] == "forget" and g["kind"] == "eval"}
    P = {(g["stage"], g["id"], g["lang"]): g for g in gens if g["split"] == "forget" and g["kind"] == "para"}
    rows = []
    for f in sorted((f for f in facts.values() if f["rep"] == 0), key=lambda f: (f["relation"], f["answer_en"])):
        q = {lg: T[f["relation"]][lg]["q_train"][0].format(e=f["entity_en"] if lg == "en" else f["entity_ta"]) for lg in ("en", "ta")}
        qp = {lg: T[f["relation"]][lg]["q_para"].format(e=f["entity_en"] if lg == "en" else f["entity_ta"]) for lg in ("en", "ta")}
        r = {"id": f["id"], "relation": f["relation"], "person_en": f["entity_en"], "person_ta": f["entity_ta"],
             "answer_en": f["answer_en"], "answer_ta": f["answer_ta"], "q_en": q["en"], "q_ta": q["ta"], "qp_en": qp["en"], "qp_ta": qp["ta"], "sets": {}}
        for st in ["taught"] + [f"unlearned_{s}" for s in SETS]:
            k = "taught" if st == "taught" else st.split("_", 1)[1]
            r["sets"][k] = {lg: {"reply": G[(st, f["id"], lg)]["gen"], "ok": bool(G[(st, f["id"], lg)]["correct"]),
                                 "para": P[(st, f["id"], lg)]["gen"], "para_ok": bool(P[(st, f["id"], lg)]["correct"])} for lg in ("en", "ta")}
        rows.append(r)
    D["facts"] = rows
    tpl = open(os.path.join(ROOT, "space", "template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(html)
    open(os.path.join(a.out, "README.md"), "w", encoding="utf-8").write(
        "---\ntitle: Where Does Forgotten Knowledge Go\nemoji: 🔍\ncolorFrom: indigo\ncolorTo: yellow\nsdk: static\n"
        "pinned: false\nlicense: cc-by-4.0\nshort_description: Unlearning audit of Tiny Aya in English and Tamil\n---\n\n"
        "Interactive write-up for the Lexsi hackathon project. Code: https://github.com/VinodAnbalagan/unlearn-audit-tiny-aya\n")
    print("wrote", a.out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
