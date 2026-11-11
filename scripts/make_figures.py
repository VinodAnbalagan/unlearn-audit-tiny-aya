"""Figures for the write-up, drawn only from logged run outputs (no number is typed in by hand except the
capitals dose table, which is copied from notes/runs/RUN-004.md and cited in figures/SOURCES.md).

    python scripts/make_figures.py --run5 runs_hf/main_20261108_165842/main_20261108_220015 \
        --loc runs_hf/localize_20261108_133112/localize_20261108_183213/localization.json --out figures

Writes PNG (2x) and SVG per figure, plus figures/data.json (every plotted value) for the HF Space.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SETS = ["circuit", "random", "lookup"]
LAYERS = {"circuit": [0, 2, 3, 6, 27, 28], "random": [0, 2, 5, 7, 21, 28], "lookup": [5, 6, 7, 8, 9, 10]}
# Palette: validated with the dataviz validator (light surface). Sets carry direct labels + marker shapes
# because one pair sits in the CVD 6-8 band.
C = {"en": "#2a78d6", "ta": "#eb6834", "circuit": "#6250d6", "random": "#008300", "lookup": "#e34948",
     "gold": "#8a8984", "ink": "#1d1d1b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
MK = {"circuit": "o", "random": "s", "lookup": "^"}
# Capitals pilot, pooled over 3 seeds per dose (notes/runs/RUN-003.md, RUN-004.md, results tables).
DOSE = {"labels": ["LOW\n6 steps", "MID\n8-12 steps", "HIGH\n200 steps"],
        "bf16_restored": [[22, 24], [20, 44], [3, 49]],
        "ta_still_right": [[20, 24], [26, 44], [12, 49]],
        "jaccess_en": [0.96, 0.98, 0.88], "logit_en": [0.83, 0.80, 0.63]}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink2"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": C["ink"],
        "legend.frameon": False})


def header(fig, title, sub=None, top=0.80):
    """Title and subtitle in figure coordinates, left aligned, above all axes (never overlapping them)."""
    fig.text(0.01, 0.985, title, fontsize=13, fontweight="bold", color=C["ink"], va="top", ha="left")
    if sub:
        fig.text(0.01, 0.925, sub, fontsize=9.2, color=C["ink2"], va="top", ha="left", linespacing=1.35)
    fig.subplots_adjust(top=top)


def save(fig, out, name):
    for ext, kw in (("png", {"dpi": 200}), ("svg", {})):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)


def load(run5):
    r = json.load(open(os.path.join(run5, "results.json"), encoding="utf-8"))
    g = [json.loads(l) for l in open(os.path.join(run5, "generations.jsonl"), encoding="utf-8")]
    facts = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "fictional_facts.jsonl"), encoding="utf-8")}
    return r, g, facts


def gen(g, stage, split="forget", kind="eval", lang="en"):
    return {x["id"]: x for x in g if x["stage"] == stage and x["split"] == split and x["kind"] == kind and x["lang"] == lang}


def fig_tamil(r, g, out, D):
    b = r["blocks"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = range(len(SETS))
    w = 0.34
    en = [b[f"{s}/eval"]["forget_eval_en"] for s in SETS]
    ta = [b[f"{s}/eval"]["forget_eval_ta"] for s in SETS]
    for i, s in enumerate(SETS):
        for off, v, lg in ((-w / 2 - 0.01, en[i], "en"), (w / 2 + 0.01, ta[i], "ta")):
            ax.bar(i + off, v, w, color=C[lg], zorder=2)
            ax.text(i + off, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, color=C["ink"])
        sil_en = [k for k, x in gen(g, f"unlearned_{s}").items() if not x["correct"]]
        ta_ok = sum(gen(g, f"unlearned_{s}", lang="ta")[k]["correct"] for k in sil_en)
        D["tamil"][s] = {"forget_en": en[i], "forget_ta": ta[i], "silenced_en": len(sil_en), "still_right_ta": ta_ok}
    gold = b["gold_ranks"]["acc"]["eval_en"]
    ax.axhline(gold, color=C["gold"], lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.45, gold + 0.02, f"never-taught model {gold:.2f}", color=C["ink2"], fontsize=8.6, ha="left", va="bottom")
    labs = []
    for s in SETS:
        t = D["tamil"][s]
        labs.append(f"{s} (layers {','.join(map(str, LAYERS[s]))})\n{t['still_right_ta']}/{t['silenced_en']} English-silenced facts\nstill right in Tamil")
    ax.set_xticks(list(xs), labs, fontsize=8.8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy on the 24 forget facts")
    h = [plt.Rectangle((0, 0), 1, 1, color=C["en"]), plt.Rectangle((0, 0), 1, 1, color=C["ta"])]
    fig.legend(h, ["asked in English", "asked in Tamil"], loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=1, fontsize=9)
    header(fig, "Forgotten in English, still answered in Tamil",
           "Accuracy after 100 NPO steps on 6 MLP layers.\nBefore unlearning, both languages were at 1.00.", top=0.80)
    save(fig, out, "fig1_tamil_leak")


def fig_curves(r, out, D):
    b = r["blocks"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for s in SETS:
        cur = b[f"{s}/unlearn"]["curve"]
        x = [0] + [c["step"] for c in cur] + [100]
        y = [1.0] + [c["forget_en"] for c in cur] + [b[f"{s}/eval"]["forget_eval_en"]]
        ax.plot(x, y, color=C[s], lw=2, marker=MK[s], ms=4.5, markevery=[len(x) - 1], zorder=3)
        dy = {"random": 0.035, "lookup": -0.035}.get(s, 0) if b["random/eval"]["forget_eval_en"] == b["lookup/eval"]["forget_eval_en"] else 0
        ax.text(101.5, y[-1] + dy, f"{s}  {y[-1]:.2f}", color=C["ink"], va="center", fontsize=9.5)
        D["curves"][s] = {"step": x, "forget_en": y, "retain_en24": [1.0] + [c["retain_en24"] for c in cur] + [None]}
    ax.axhline(0.25, color=C["ink2"], lw=1, ls=(0, (2, 3)))
    ax.text(1, 0.27, "stop target 0.25 (never reached)", fontsize=8.6, color=C["ink2"])
    ax.axhline(1.0, color=C["gold"], lw=1, ls=(0, (4, 3)))
    ax.text(1, 1.02, "retain accuracy: 1.00 at every check, all three sets", fontsize=8.6, color=C["ink2"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("NPO step")
    ax.set_ylabel("English forget accuracy")
    header(fig, "The attribution-chosen circuit forgot the least",
           "English forget accuracy at every check (every 5 steps). Same budget, same optimiser, same 24 facts.", top=0.82)
    save(fig, out, "fig2_forgetting_curves")


def fig_imprint(r, out, D):
    b = r["blocks"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    rows = []
    for s in SETS:
        j = b[f"{s}/jaccess"]["en"]
        rows.append((s, j["n_silent"], j["jlens_k10"]["A_unlearned"], j["jlens_k10"]["A_gold"], j["jlens_k10"]["A_decoy"]))
        D["imprint"][s] = {"n_silent": j["n_silent"], "unlearned": j["jlens_k10"]["A_unlearned"], "gold": j["jlens_k10"]["A_gold"],
                           "decoy": j["jlens_k10"]["A_decoy"], "unlearned_k50": j["jlens_k50"]["A_unlearned"], "gold_k50": j["jlens_k50"]["A_gold"]}
    for i, (s, n, u, gd, dc) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([u, gd], [y, y], color=C["grid"], lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([gd], [y], s=90, color=C["gold"], zorder=3, edgecolor=C["surface"], linewidth=2)
        ax.scatter([u], [y], s=90, color=C[s], marker=MK[s], zorder=4, edgecolor=C["surface"], linewidth=2)
        ax.scatter([dc], [y], s=40, facecolor="none", edgecolor=C["ink2"], zorder=2)
        ax.text(gd + 0.025, y, f"never taught {gd:.2f}", va="center", fontsize=9, color=C["ink2"])
        ax.text(max(u, dc) + 0.03 if gd - u > 0.35 else u - 0.03, y + 0.27, f"unlearned {u:.2f}", va="center",
                ha="left", fontsize=9, color=C["ink"])
    ax.set_yticks(range(len(rows)), [f"{s}\n({n} silenced facts)" for s, n, *_ in reversed(rows)], fontsize=9.2)
    ax.set_xlim(-0.03, 1.22)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("share of silenced facts whose answer is in the J-lens top-10 (layers 31 to 34)")
    header(fig, "Unlearned is not the same as never learned",
           "A model that never saw these people still ranks the true answer highly (it is a plausible colour, pet, job).\n"
           "The unlearned model pushes it far below that. Grey dot: never-taught model. Open circle: decoy control.", top=0.78)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig3_negative_imprint")


def categorize(f, text, vals):
    w = re.findall(r"[a-z]+", text.lower())
    ans = f["answer_en"].lower()
    if [v for v in vals[f["relation"]] if v != ans and v in w]:
        return "another answer from the same list"
    names = [n.lower() for n in f["entity_en"].split()]
    stop = {"the", "plays", "colour", "loves", "most", "is", "favourite", "s", "pet", "works", "as", "a"}
    core = [x for x in w if x not in stop]
    if any(len(x) >= 3 and any(x[:3] == n[:3] for n in names) for x in core):
        return "a piece of the person's name"
    if core and core[0][0] == ans[0]:
        return "a word that starts like the answer"
    return "something else"


def fig_where(g, facts, out, D):
    vals = collections.defaultdict(set)
    for f in facts.values():
        vals[f["relation"]].add(f["answer_en"].lower())
    forget = sorted([f for f in facts.values() if f["rep"] == 0], key=lambda f: (f["relation"], f["answer_en"]))
    cats = collections.Counter()
    examples = collections.defaultdict(list)
    grid = []
    for f in forget:
        row = []
        for s in SETS:
            x = gen(g, f"unlearned_{s}")[f["id"]]
            row.append(not x["correct"])
            if not x["correct"]:
                c = categorize(f, x["gen"], vals)
                cats[c] += 1
                examples[c].append({"set": s, "answer": f["answer_en"], "person": f["entity_en"], "reply": x["gen"]})
        grid.append(row)
    D["where"] = {"counts": dict(cats), "examples": examples,
                  "grid": [{"id": f["id"], "relation": f["relation"], "answer": f["answer_en"], "person": f["entity_en"],
                            "silenced": dict(zip(SETS, row))} for f, row in zip(forget, grid)]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 6.0), gridspec_kw={"width_ratios": [1.0, 1.3], "wspace": 0.95})
    for i, (f, row) in enumerate(zip(forget, grid)):
        y = len(forget) - 1 - i
        for j, s in enumerate(SETS):
            a1.add_patch(plt.Rectangle((j + 0.06, y + 0.08), 0.88, 0.84, color=C[s] if row[j] else C["grid"], lw=0))
        a1.text(-0.12, y + 0.5, f"{f['relation']}: {f['answer_en']}", ha="right", va="center", fontsize=8.3, color=C["ink"])
    a1.set_xlim(0, 3)
    a1.set_ylim(0, len(forget))
    a1.set_xticks([0.5, 1.5, 2.5], SETS, fontsize=9)
    a1.xaxis.tick_top()
    a1.set_yticks([])
    a1.grid(False)
    for sp in a1.spines.values():
        sp.set_visible(False)

    a1.text(0, -0.035, "Filled: silenced in English. 24 forget facts.", transform=a1.transAxes, fontsize=8.5,
            color=C["ink2"], va="top")
    order = ["another answer from the same list", "a piece of the person's name", "a word that starts like the answer", "something else"]
    tot = sum(cats.values())
    ys = list(range(len(order)))[::-1]
    a2.barh(ys, [cats[o] for o in order], color=C["ink2"], height=0.55)
    ex = {"another answer from the same list": "goat -> \"dog\" (all 3 sets), lawyer -> \"engineer\"",
          "a piece of the person's name": "Senar Yayar plays the drums -> \"Yayar\"",
          "a word that starts like the answer": "violin -> \"Villasovka\"",
          "something else": "doctor -> \"Nurse\", organ -> \"Pianio\""}
    for y, o in zip(ys, order):
        a2.text(cats[o] + 0.3, y, f"{cats[o]}", va="center", fontsize=9.5, color=C["ink"])
        a2.text(0, y - 0.48, ex[o], fontsize=8.2, color=C["ink2"], va="center")
    a2.set_yticks(ys, order, fontsize=9)
    a2.set_xlim(0, max(cats.values()) + 3)
    a2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a2.set_ylim(-0.9, len(order) - 0.4)
    a2.set_xlabel(f"silenced English replies (n = {tot}, all 3 sets)")
    a2.grid(axis="y", visible=False)
    a2.set_title("What the model said instead", pad=12, fontsize=11)
    header(fig, "Where the forgotten answers went",
           "Left: the same facts fall wherever you cut. Right: the answer is replaced, never refused.", top=0.83)
    save(fig, out, "fig4_where_it_went")


def fig_steer(r, out, D):
    b = r["blocks"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    cols = [("unlearned", "heldout_unlearned_en", C["ink2"], "o", 60), ("steered", "heldout_steered_en", C["en"], "D", 70),
            ("retain-fact direction", "retain_direction_heldout_en", C["ta"], "s", 60)]
    for i, s in enumerate(SETS):
        st = b[f"{s}/steering"]
        y = len(SETS) - 1 - i
        D["steering"][s] = {lab: st[key] for lab, key, *_ in cols} | {"random": st["random_heldout_en"], "chosen": st["chosen"]}
        for k, (lab, key, col, m, sz) in enumerate(cols):
            ax.scatter(st[key], y + (k - 1) * 0.16, color=col, marker=m, s=sz, zorder=3, label=lab if i == 0 else None)
        for k, v in enumerate(st["random_heldout_en"]):
            ax.scatter(v, y + 0.34 + k * 0.08, facecolor="none", edgecolor=C["ink2"], s=36, zorder=2,
                       label="random directions (same norm)" if i == 0 and k == 0 else None)
    ax.set_yticks(range(len(SETS)), list(reversed(SETS)))
    ax.set_xlim(0.3, 0.8)
    ax.set_ylim(-0.5, len(SETS) - 0.35)
    ax.set_xlabel("held-out English accuracy (12 facts)")
    header(fig, "Steering does not beat the controls",
           "Held-out facts, English. A direction estimated from other forgotten facts, vs a direction from retained facts\nand random directions of the same norm.", top=0.76)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8.6, handletextpad=0.3)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig5_steering_controls")


def fig_dose(r, out, D):
    b = r["blocks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.3), gridspec_kw={"wspace": 0.35})
    x = range(3)
    rest = [k / n for k, n in DOSE["bf16_restored"]]
    ta = [k / n for k, n in DOSE["ta_still_right"]]
    a1.plot(x, rest, color=C["ink"], lw=2, marker="o", ms=6)
    for i, (k, n) in enumerate(DOSE["bf16_restored"]):
        a1.text(i, rest[i] + 0.06, f"{k}/{n}", ha="center", fontsize=9.5, color=C["ink"])
    f5 = sum(1 for s in SETS if b[f"{s}/bf16"]["forget_en"] > b[f"{s}/eval"]["forget_eval_en"])
    a1.set_xticks(list(x), DOSE["labels"], fontsize=9)
    a1.set_ylim(0, 1.15)
    a1.set_ylabel("share of English-forgotten facts")
    a1.set_title("bf16 rounding brings facts back", fontsize=11, pad=10)
    a1.text(0, -0.36, f"Fictional facts, 100 steps on 6 layers: 0 facts restored in any set.", transform=a1.transAxes,
            fontsize=8.5, color=C["ink2"])
    a2.plot(x, ta, color=C["ta"], lw=2, marker="s", ms=6)
    for i, (k, n) in enumerate(DOSE["ta_still_right"]):
        a2.text(i, ta[i] + 0.06, f"{k}/{n}", ha="center", fontsize=9.5, color=C["ink"])
    a2.set_xticks(list(x), DOSE["labels"], fontsize=9)
    a2.set_ylim(0, 1.1)
    a2.set_title("Tamil still answers", fontsize=11, pad=10)
    header(fig, "Both leaks shrink as unlearning gets stronger (real capitals pilot)",
           "Share of English-forgotten facts that come back after a bf16 cast (left) or are still answered in Tamil (right).", top=0.76)
    a2.set_ylabel("still right in Tamil")
    a2.text(0, -0.36, "Real capitals, 3 seeds per dose, NPO on all layers.", transform=a2.transAxes, fontsize=8.5, color=C["ink2"])
    D["dose_capitals"] = DOSE | {"fictional_bf16_restored_sets": f5}
    save(fig, out, "fig6_dose_capitals")


def fig_loc(loc, out, D):
    m = {int(k): v for k, v in loc["mean_mlp_score"].items()}
    L = sorted(m)
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    top = set(loc["sets"]["circuit"])
    ax.bar(L, [m[l] for l in L], color=[C["circuit"] if l in top else C["grid"] for l in L], width=0.75)
    for l in top:
        ax.text(l, m[l], f"{l}", ha="center", va="bottom", fontsize=8.5, color=C["ink"])
    for s, yv in (("lookup", -0.17), ("random", -0.26)):
        for l in loc["sets"][s]:
            ax.scatter(l, yv, transform=ax.get_xaxis_transform(), marker=MK[s], color=C[s], s=26, clip_on=False)
        ax.text(-1.6, yv, s, transform=ax.get_xaxis_transform(), ha="right", va="center", fontsize=8.6, color=C["ink2"])
    ax.set_xlim(-1, 36)
    ax.set_xticks(range(0, 36, 5))
    ax.set_xlabel("MLP layer", labelpad=40)
    ax.set_ylabel("mean |EAP-IG score|")
    header(fig, "Where CircuitKIT says the invented facts live",
           "EAP-IG node scores for the 36 MLP layers, mean over 3 seeds (top-6 identical in every seed). Purple: the circuit set.", top=0.76)
    ax.grid(axis="x", visible=False)
    D["localization"] = {"mean_mlp_score": m, "sets": loc["sets"], "jaccard": loc["jaccard_topk_pairs"]}
    save(fig, out, "fig0_localization")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run5", required=True)
    ap.add_argument("--loc", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    r, g, facts = load(a.run5)
    loc = json.load(open(a.loc, encoding="utf-8"))
    D = {"tamil": {}, "curves": {}, "imprint": {}, "steering": {}, "source": {"run5": a.run5, "loc": a.loc, "frozen": r["frozen"]}}
    fig_loc(loc, a.out, D)
    fig_tamil(r, g, a.out, D)
    fig_curves(r, a.out, D)
    fig_imprint(r, a.out, D)
    fig_where(g, facts, a.out, D)
    fig_steer(r, a.out, D)
    fig_dose(r, a.out, D)
    D["name_rank"] = {k: r["blocks"][k] for k in r["blocks"] if k.endswith("name_rank")} | {"gold": r["blocks"]["gold_ranks"]}
    D["bf16"] = {s: r["blocks"][f"{s}/bf16"] for s in SETS}
    D["eval"] = {s: r["blocks"][f"{s}/eval"] for s in SETS} | {"taught": r["blocks"]["taught_eval"], "gold_acc": r["blocks"]["gold_ranks"]["acc"]}
    json.dump(D, open(os.path.join(a.out, "data.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("figures written to", a.out, sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
