"""MAIN RUN (RUN-005): circuit-localized vs lookup vs random unlearning of TAUGHT fictional facts, with
every audit, against a never-taught GOLD model. One seed (n = 1 by budget; stated in the write-up).

    python scripts/main_run.py --out /outputs --rep 0 --lens_repo vinod-anbalagan/tiny-aya-global-jlens

Per localization set S (6 MLP down_proj layers each, fixed in LOC-001):
  restore TAUGHT weights -> SafeTune NPO (st_npo) on S only -> evaluate:
  forgetting (EN, TA, held-out paraphrase), retain cost; steering recovery vs retain-direction and random
  controls; J-Access (J-lens + logit lens, judge-silent, decoys) and JOcc against GOLD; whole-name
  ranking among the 6 answers of the relation (J-lens band + output) for TAUGHT, UNLEARNED and GOLD;
  bf16 cast of S (the rest of the model is bf16-exact already, it was saved in bf16).
Robustness for a paid one-shot run: every analysis block is isolated (an exception is recorded, the run
continues), results are saved after every block, the J-lens is loaded once, GOLD is scored once at start.
Unlearning-step fixes after attempt 1 (2026-11-08, OOM at the first NPO step in every set, so nothing was
unlearned and no unlearning result exists): non-reentrant gradient checkpointing during NPO (pilot.py had it,
main_run.py had dropped it), TF32 matmuls during NPO steps only (every evaluation stays strict fp32), retain
term on cycled chunks of 24 of the 72 retain facts (fixed shuffle), max 100 steps with a check every 5, GPU
memory logged per block, and FAIL-FAST: if unlearning fails, that set's analyses are skipped and the run stops
(attempt 1 spent 11 minutes analysing an untouched model).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core, jaccess, name_rank, st_npo  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SETS = {"circuit": [0, 2, 3, 6, 27, 28], "lookup": [5, 6, 7, 8, 9, 10], "random": [0, 2, 5, 7, 21, 28]}  # LOC-001


def acc(model, tok, facts, lang, kind="eval"):
    prompts = [core.prompt_ids(tok, core.question(f, lang, kind)) for f in facts]
    gens = core.generate(model, tok, prompts)
    rows = [{"id": f["id"], "lang": lang, "kind": kind, "gen": g, **core.judge(g, f, lang), **core.answer_format(g, lang)}
            for f, g in zip(facts, gens)]
    return sum(r["correct"] for r in rows) / max(len(rows), 1), rows


def full_eval(model, tok, forget, retain, tag, gens_log):
    out = {}
    for lang in ("en", "ta"):
        for kind in ("eval", "para"):
            a, rows = acc(model, tok, forget, lang, kind)
            out[f"forget_{kind}_{lang}"] = a
            gens_log += [{**r, "stage": tag, "split": "forget"} for r in rows]
        a, rows = acc(model, tok, retain, lang)
        out[f"retain_{lang}"] = a
        gens_log += [{**r, "stage": tag, "split": "retain"} for r in rows]
    return out


def jaccess_block(model, tok, facts, J, band, gold_ranks, gens_rows, lang):
    """Judge-silent access rates (J-lens and logit lens), decoys, JOcc against GOLD."""
    silent = [not r["correct"] for r in gens_rows]
    perm = jaccess.derangement(len(facts), 0)
    res = {"n_silent": sum(silent)}
    for name, JJ in (("jlens", J), ("logit", None)):
        R = jaccess.ranks(model, tok, facts, lang, JJ, band)
        Rd = jaccess.ranks(model, tok, facts, lang, JJ, band, concept_facts=[facts[j] for j in perm])
        cols = list(range(len(band)))
        for k in (10, 50):
            a_u = jaccess.access_rate(R, silent, cols, k)
            a_g = jaccess.access_rate(gold_ranks[name], silent, cols, k)
            res[f"{name}_k{k}"] = {"A_unlearned": a_u, "A_decoy": jaccess.access_rate(Rd, silent, cols, k), "A_gold": a_g}
    return res


def name_rank_block(model, tok, facts, J, band, lang):
    """Rank of the true answer among the 6 values of its relation, whole word, per readout column."""
    out = {}
    by_rel = {}
    for f in facts:
        by_rel.setdefault(f["relation"], []).append(f)
    allf = core.load_facts(os.path.join(ROOT, "data", "fictional_facts.jsonl"))
    for rel, fs in by_rel.items():
        cands = []
        for f in allf:  # one candidate per answer value of this relation
            if f["relation"] == rel and f["answer_en"] not in [c["answer_en"] for c in cands]:
                cands.append(f)
        S = name_rank.name_scores(model, tok, fs, lang, cands, J, band)
        true_idx = [[c["answer_en"] for c in cands].index(f["answer_en"]) for f in fs]
        out[rel] = name_rank.summarize(S, true_idx, band)
    pooled = {}
    for col in next(iter(out.values())):
        ranks = [r for rel in out for r in out[rel][col]["ranks"]]
        pooled[col] = {"mean_rank": sum(ranks) / len(ranks), "top1": sum(r == 1 for r in ranks) / len(ranks), "n": len(ranks)}
    return {"pooled": pooled, "by_relation": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--taught", default="vinod-anbalagan/tiny-aya-fict-taught")
    ap.add_argument("--gold", default=None, help="default: vinod-anbalagan/tiny-aya-fict-gold-r{rep}")
    ap.add_argument("--lens_repo", default="vinod-anbalagan/tiny-aya-global-jlens")
    ap.add_argument("--sets", default="circuit,lookup,random")
    ap.add_argument("--target_forget", type=float, default=0.25)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--retain_bs", type=int, default=24)
    ap.add_argument("--steer_layers", default="12,20,28")
    ap.add_argument("--steer_alphas", default="1,2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gold_repo = a.gold or f"vinod-anbalagan/tiny-aya-fict-gold-r{a.rep}"
    core.seed_all(a.seed)
    run = os.path.join(a.out, time.strftime("main_%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=True)
    t0 = time.time()
    log = {"args": vars(a), "gold": gold_repo, "sets": SETS, "blocks": {}, "errors": {}}
    gens = []

    def save():
        log["minutes"] = round((time.time() - t0) / 60, 1)
        if torch.cuda.is_available():
            log["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        with open(os.path.join(run, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=1, ensure_ascii=False, default=str)
        with open(os.path.join(run, "generations.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gens))

    def gpu():
        if not torch.cuda.is_available():
            return ""
        return f"| GPU alloc {torch.cuda.memory_allocated() / 2**30:.1f} GB peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"

    def block(name, fn):
        try:
            log["blocks"][name] = fn()
            print(f"[ok] {name} ({round((time.time() - t0) / 60, 1)} min) {gpu()}", json.dumps(log["blocks"][name], default=str)[:400], flush=True)
            ok = True
        except Exception as e:  # noqa: BLE001  a paid run must not die on one analysis
            log["errors"][name] = traceback.format_exc()[-6000:]
            print(f"[ERROR] {name}: {e!r} {gpu()}", flush=True)
            ok = False
        save()
        return ok

    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    log["frozen"] = {p: hashlib.sha256(open(os.path.join(here, "..", p), "rb").read()).hexdigest()[:16] for p in
                     ["src/unlearn_audit/core.py", "src/unlearn_audit/jaccess.py", "src/unlearn_audit/name_rank.py",
                      "src/unlearn_audit/st_npo.py", "scripts/main_run.py", "data/fictional_facts.jsonl", "data/fictional_templates.json"]}
    print("frozen", log["frozen"], flush=True)

    facts = core.load_facts(os.path.join(ROOT, "data", "fictional_facts.jsonl"))
    forget = [f for f in facts if f["rep"] == a.rep]
    retain = [f for f in facts if f["rep"] != a.rep]
    rng = random.Random(a.seed)
    disc, held = [], []
    for rel in ("colour", "profession", "instrument", "pet"):  # stratified: 3 discovery + 3 held-out per relation
        fs = sorted([f for f in forget if f["relation"] == rel], key=lambda f: f["id"])
        rng.shuffle(fs)
        disc += fs[:3]
        held += fs[3:]
    log["splits"] = {"forget": [f["id"] for f in forget], "discovery": [f["id"] for f in disc], "heldout": [f["id"] for f in held]}
    assert len(forget) == 24 and len(disc) == 12 and len(held) == 12 and len(retain) == 72

    J = jaccess.load_lens(__import__("huggingface_hub").hf_hub_download(a.lens_repo, "lens.pt"))
    lens_layers = sorted(J)

    gold_ranks = {}

    tok, model = core.load_model(dtype=torch.float32, name=a.taught)
    model.requires_grad_(False)
    taught_state = None

    def calib():
        # J-Access band from the TAUGHT model on retain facts (frozen before any unlearning)
        cal = {lang: jaccess.ranks(model, tok, retain[:24], lang, J, lens_layers) for lang in ("en", "ta")}
        hit = ((cal["en"] <= 10).float().mean(0) + (cal["ta"] <= 10).float().mean(0)) / 2
        order = sorted(range(len(lens_layers)), key=lambda j: (-float(hit[j]), lens_layers[j]))
        band = sorted(lens_layers[j] for j in order[:4])
        log["band"] = band
        return {"band": band, "hit_by_layer": [round(float(x), 3) for x in hit]}
    block("calibration", calib)
    band = log.get("band", [27, 29, 30, 33])

    block("taught_eval", lambda: full_eval(model, tok, forget, retain, "taught", gens))
    block("taught_name_rank", lambda: {lang: name_rank_block(model, tok, forget, J, band, lang)["pooled"] for lang in ("en", "ta")})

    # gold ranks need the gold model: recompute cheaply by reloading only if needed -> do it now with gold weights
    def gold_ranks_block():
        _, g = core.load_model(dtype=torch.float32, name=gold_repo)
        g.requires_grad_(False)
        out = {"acc": {f"{k}_{lang}": acc(g, tok, forget, lang, k)[0] for lang in ("en", "ta") for k in ("eval", "para")}}
        for lang in ("en", "ta"):
            gold_ranks[lang] = {nm: jaccess.ranks(g, tok, forget, lang, JJ, band) for nm, JJ in (("jlens", J), ("logit", None))}
            out[f"name_rank_{lang}"] = name_rank_block(g, tok, forget, J, band, lang)["pooled"]
        g.to("cpu")  # frees its GPU memory even if some object still references the module
        del g
        gc.collect()
        torch.cuda.empty_cache()
        return out
    # free the taught model's GPU memory pressure: gold is loaded alongside briefly (2 x 13.4 GB fits on 80 GB)
    block("gold_ranks", gold_ranks_block)

    all_layers = sorted({l for s in SETS.values() for l in s})
    params_all = core.select_params(model, all_layers, ["down_proj"])
    taught_state = core.snapshot(params_all)
    model.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    retain_shuf = list(retain)
    random.Random(1).shuffle(retain_shuf)  # chunks of 24 then mix relations and people

    for set_name in a.sets.split(","):
        layers = SETS[set_name]
        core.load_snapshot(params_all, taught_state)
        names = [f"model.layers.{l}.mlp.down_proj.weight" for l in layers]
        curve = []

        t_set = [time.time()]

        def stop_fn(step):
            torch.backends.cuda.matmul.allow_tf32 = False  # evaluations are strict fp32
            model.eval()
            fa, _ = acc(model, tok, forget, "en")
            ra, _ = acc(model, tok, retain[:24], "en")
            curve.append({"step": step, "forget_en": fa, "retain_en24": ra, "sec": round(time.time() - t_set[0], 1)})
            print(set_name, curve[-1], gpu(), flush=True)
            model.train()
            torch.backends.cuda.matmul.allow_tf32 = True
            return fa <= a.target_forget and ra >= 0.9

        def unlearn():
            ref = copy.deepcopy(model).eval().requires_grad_(False)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                _, rec = st_npo.npo_localized(model, tok, forget, retain_shuf, names, reference=ref, variant="npo_grad_diff",
                                              beta=0.1, lr=1e-5, weight_decay=0.01, max_steps=a.max_steps,
                                              eval_every=a.eval_every, stop_fn=stop_fn, retain_bs=a.retain_bs)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = False
                model.gradient_checkpointing_disable()
                model.config.use_cache = True
                model.eval()
                del ref
                gc.collect()
                torch.cuda.empty_cache()
            moved = sum(float((params_all[k].detach().cpu() - taught_state[k]).abs().sum()) for k in names)
            assert moved > 0, "unlearning finished but the weights did not change"
            return {"layers": layers, "stopped_at": rec.stopped_at, "steps_done": rec.steps_done, "curve": curve,
                    "abs_weight_change": moved, "minutes": round((time.time() - t_set[0]) / 60, 1)}
        ok = block(f"{set_name}/unlearn", unlearn)
        model.requires_grad_(False)
        if not ok:
            print(f"FAIL-FAST: {set_name}/unlearn failed; skipping its analyses and stopping the run", flush=True)
            log["aborted"] = f"{set_name}/unlearn"
            break
        block(f"{set_name}/eval", lambda: full_eval(model, tok, forget, retain, f"unlearned_{set_name}", gens))

        def jacc():
            out = {}
            for lang in ("en", "ta"):
                rows = [g for g in gens if g["stage"] == f"unlearned_{set_name}" and g["split"] == "forget" and g["kind"] == "eval" and g["lang"] == lang]
                out[lang] = jaccess_block(model, tok, forget, J, band, gold_ranks.get(lang, {}), rows, lang)
            return out
        block(f"{set_name}/jaccess", jacc)
        block(f"{set_name}/name_rank", lambda: {lang: name_rank_block(model, tok, forget, J, band, lang)["pooled"] for lang in ("en", "ta")})

        def steering():
            unl_state = core.snapshot({k: params_all[k] for k in names})
            sub = {k: params_all[k] for k in names}
            core.load_snapshot(params_all, taught_state)
            h_ref = core.last_token_states(model, tok, disc)
            hr_ref = core.last_token_states(model, tok, retain[:12])
            core.load_snapshot(sub, unl_state)
            h_unl = core.last_token_states(model, tok, disc)
            hr_unl = core.last_token_states(model, tok, retain[:12])
            dirs = (h_ref - h_unl).mean(0)
            rdirs = (hr_ref - hr_unl).mean(0)
            best, grid = None, []
            for l in [int(x) for x in a.steer_layers.split(",")]:
                for al in [float(x) for x in a.steer_alphas.split(",")]:
                    with core.steer(model, l, al * dirs[l]):
                        d_acc = acc(model, tok, disc, "en")[0]
                        r_acc = acc(model, tok, retain[:24], "en")[0]
                    grid.append({"layer": l, "alpha": al, "disc_en": d_acc, "retain_en24": r_acc})
                    if r_acc >= 0.9 and (best is None or d_acc > best["disc_en"]):
                        best = grid[-1]
            if best is None:
                best = max(grid, key=lambda g: g["disc_en"])
                best = {**best, "violates_retain": True}
            l, al = best["layer"], best["alpha"]
            vec = al * dirs[l]
            res = {"grid": grid, "chosen": best, "heldout_unlearned_en": acc(model, tok, held, "en")[0]}
            with core.steer(model, l, vec):
                res["heldout_steered_en"] = acc(model, tok, held, "en")[0]
                res["heldout_steered_ta"] = acc(model, tok, held, "ta")[0]
                res["retain_steered_en"] = acc(model, tok, retain[:24], "en")[0]
            rv = rdirs[l] / (rdirs[l].norm() + 1e-8) * vec.norm()
            with core.steer(model, l, rv):
                res["retain_direction_heldout_en"] = acc(model, tok, held, "en")[0]
            g = torch.Generator().manual_seed(0)
            rnd = []
            for _ in range(2):
                r = torch.randn(vec.shape, generator=g)
                r = r / r.norm() * vec.norm().cpu()
                with core.steer(model, l, r):
                    rnd.append(acc(model, tok, held, "en")[0])
            res["random_heldout_en"] = rnd
            return res
        block(f"{set_name}/steering", steering)

        def bf16():
            sub = {k: params_all[k] for k in names}
            unl = core.snapshot(sub)
            num = den = erased = changed = 0.0
            for k in names:
                d = unl[k].float() - taught_state[k].float()
                db = unl[k].bfloat16().float() - taught_state[k].bfloat16().float()
                num += float((db ** 2).sum()); den += float((d ** 2).sum())
                nz = d != 0
                changed += float(nz.sum()); erased += float((nz & (db == 0)).sum())
            with torch.no_grad():
                for k, p in sub.items():
                    p.copy_(p.bfloat16().float())
            out = {"l2_survives": (num / den) ** 0.5 if den else None, "erased_frac": erased / changed if changed else None}
            for lang in ("en", "ta"):
                out[f"forget_{lang}"] = acc(model, tok, forget, lang)[0]
            core.load_snapshot(sub, unl)
            return out
        block(f"{set_name}/bf16", bf16)

    core.load_snapshot(params_all, taught_state)
    log["done"] = True
    save()
    print("SUMMARY errors:", list(log["errors"]), "| aborted:", log.get("aborted"), "| minutes", log["minutes"], "| saved", run)
    if log.get("aborted"):
        sys.exit(1)


if __name__ == "__main__":
    main()
