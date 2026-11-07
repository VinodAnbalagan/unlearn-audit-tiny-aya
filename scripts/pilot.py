"""Day-1 pilot: can we forget known facts on Tiny Aya without breaking it, and does a
held-out steering direction bring them back?

Gates (from notes/RESEARCH_PLAN.md):
  G1 architecture: forward, generate, hooks, backward on selected params, snapshot/restore.
  G2 knowledge:    enough facts known in English (and, ideally, Tamil) before unlearning.
  G3 forgetting:   forget-set accuracy drops >= 30 points with retain drop <= 10 points.
  G4 recovery:     a direction estimated on DISCOVERY facts raises accuracy on held-out
                   EVAL facts more than equal-norm random directions, without wrecking retain.

Not a result for the write-up: this is a feasibility check on pretrained facts.

Usage (any CUDA GPU with ~40GB+): python scripts/pilot.py --out ./runs
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl"))
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_forget", type=int, default=20)
    ap.add_argument("--layers", default="all", help="'all' or comma list, e.g. 4,5,6,7")
    ap.add_argument("--modules", default="down_proj")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="NPO beta")
    ap.add_argument("--retain_weight", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--unlearner", default="safetune", choices=["safetune", "ours"],
                    help="safetune: safetune.unlearn.npo_unlearn via unlearn_audit.st_npo (proven equal to ours in tests/test_st_npo_cpu.py)")
    ap.add_argument("--variant", default="npo_grad_diff", help="SafeTune NPO variant (npo | npo_grad_diff | npo_KL)")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="SafeTune default; ours used 0.0")
    ap.add_argument("--target_drop", type=float, default=0.30)
    ap.add_argument("--steer_layers", default="4,8,12,16,20,24,28,32")
    ap.add_argument("--steer_alphas", default="0.5,1,2,4")
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--lens_repo", default=None, help="HF repo with lens.pt (scripts/fit_lens.py); enables the J-Access audit")
    ap.add_argument("--k", type=int, default=10, help="J-Access top-k")
    return ap.parse_args()


def main():
    a = parse()
    core.seed_all(a.seed)
    run_dir = os.path.join(a.out, time.strftime("pilot_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log = {"args": vars(a), "gates": {}, "env": {"torch": torch.__version__}}
    import transformers
    log["env"]["transformers"] = transformers.__version__
    # Freeze record (clean-run protocol): hashes of the judge code, the data, and this script
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    def _sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    log["frozen"] = {"core.py (judge)": _sha(os.path.join(here, "..", "src", "unlearn_audit", "core.py")),
                     "data": _sha(a.data), "pilot.py": _sha(os.path.abspath(__file__)),
                     "jaccess.py": _sha(os.path.join(here, "..", "src", "unlearn_audit", "jaccess.py")),
                     "st_npo.py": _sha(os.path.join(here, "..", "src", "unlearn_audit", "st_npo.py"))}
    print("frozen", log["frozen"], flush=True)
    gens = []

    def save():
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, "generations.jsonl"), "w", encoding="utf-8") as f:
            for g in gens:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")

    t0 = time.time()
    tok, model = core.load_model(dtype=torch.float32)
    n_layers = len(core.decoder_layers(model))
    log["env"]["n_layers"] = n_layers
    facts = core.load_facts(a.data)

    # ---- G2 knowledge (baseline)
    acc_en, rows_en = core.accuracy(model, tok, facts, "en")
    acc_ta, rows_ta = core.accuracy(model, tok, facts, "ta")
    for r in rows_en + rows_ta:
        gens.append({"stage": "baseline", **r})
    known_en = {r["id"] for r in rows_en if r["correct"]}
    known_ta = {r["id"] for r in rows_ta if r["correct"]}
    log["baseline"] = {"acc_en": acc_en, "acc_ta": acc_ta, "n_known_en": len(known_en), "n_known_both": len(known_en & known_ta)}
    print("baseline", log["baseline"], flush=True)
    if len(known_en) < a.n_forget + 10:
        log["gates"]["G2_knowledge"] = "FAIL: too few known facts"
        save()
        return
    log["gates"]["G2_knowledge"] = "PASS"

    # Prefer facts known in both languages for the forget set, so Tamil is measurable.
    rng = random.Random(a.seed)
    both = sorted(known_en & known_ta)
    en_only = sorted(known_en - known_ta)
    rng.shuffle(both)
    rng.shuffle(en_only)
    pool = both + en_only
    forget_ids = pool[: a.n_forget]
    retain_ids = pool[a.n_forget :]
    by_id = {f["id"]: f for f in facts}
    forget = [by_id[i] for i in forget_ids]
    retain = [by_id[i] for i in retain_ids]
    half = len(forget) // 2
    disc, held = forget[:half], forget[half:]  # steering is ESTIMATED on disc, TESTED on held
    log["splits"] = {"discovery": [f["id"] for f in disc], "heldout": [f["id"] for f in held], "retain": retain_ids}
    # Baseline per split and language. A0 for English is 1.0 by construction; Tamil is NOT, because
    # en-only facts fill the forget set when too few are known in both. Without this, Tamil "forgetting"
    # could just be facts the model never knew in Tamil.
    log["baseline_by_split"] = {f"{n}_{lg}": sum((f["id"] in (known_en if lg == "en" else known_ta)) for f in fs) / max(len(fs), 1)
                                for n, fs in [("discovery", disc), ("heldout", held), ("retain", retain)] for lg in ("en", "ta")}
    print("baseline_by_split", log["baseline_by_split"], flush=True)

    # ---- G1 architecture: parameter selection, snapshot, backward
    layers = list(range(n_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    params = core.select_params(model, layers, a.modules.split(","))
    n_train = sum(p.numel() for p in params.values())
    orig = core.snapshot(params)
    log["train"] = {"layers": layers, "modules": a.modules, "n_trainable": n_train}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    def eval_point(step):
        model.eval()
        fa, _ = core.accuracy(model, tok, forget, "en")
        ra, _ = core.accuracy(model, tok, retain, "en")
        return {"step": step, "forget_en": fa, "retain_en": ra}

    base_pt = {"step": 0, "forget_en": 1.0, "retain_en": 1.0}  # by construction: all were known
    curve = [base_pt]
    log["gates"]["G1_architecture"] = "PASS (load, generate, select params)"
    chosen = None

    def reached(pt):
        return pt["forget_en"] <= 1 - a.target_drop and pt["retain_en"] >= 0.9

    if a.unlearner == "safetune":
        import copy
        from unlearn_audit import st_npo
        reference = copy.deepcopy(model).eval().requires_grad_(False)  # frozen ORIGINAL (trap 3)

        def stop_fn(step):
            nonlocal chosen
            pt = eval_point(step)
            curve.append(pt)
            print(pt, flush=True)
            model.train()
            if reached(pt):
                chosen = pt
                return True
            return False

        _, rec = st_npo.npo_localized(model, tok, forget, retain, list(params.keys()), reference=reference,
                                      variant=a.variant, beta=a.beta, lr=a.lr, weight_decay=a.weight_decay,
                                      max_steps=a.max_steps, eval_every=a.eval_every, stop_fn=stop_fn)
        if chosen is None:  # ran to max_steps: record the final point
            curve.append(eval_point(rec.steps_done))
        log["train"].update({"unlearner": "safetune.unlearn.npo_unlearn", "variant": a.variant,
                             "weight_decay": a.weight_decay, "optimizer": "adamw", "grad_clip": None,
                             "stopped_at_step": rec.stopped_at,
                             "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None})
        del reference
        torch.cuda.empty_cache()
        for p_ in params.values():  # st_npo froze everything; keep the selection for later checks
            p_.requires_grad_(False)
    else:
        with torch.no_grad():
            ref_f, _ = core.answer_logprob(model, tok, forget)
        opt = torch.optim.AdamW(list(params.values()), lr=a.lr, weight_decay=0.0)
        for step in range(1, a.max_steps + 1):
            model.train()
            lp_f, _ = core.answer_logprob(model, tok, forget, grad=True)
            loss_f = -(2.0 / a.beta) * F.logsigmoid(-a.beta * (lp_f - ref_f)).mean()  # NPO
            lp_r, n_r = core.answer_logprob(model, tok, retain, grad=True)
            loss_r = -(lp_r / n_r).mean()
            loss = loss_f + a.retain_weight * loss_r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            opt.step()
            if step % a.eval_every == 0:
                pt = eval_point(step)
                pt.update({"loss_forget": loss_f.item(), "loss_retain": loss_r.item()})
                curve.append(pt)
                print(pt, flush=True)
                if reached(pt):
                    chosen = pt
                    break
        log["train"].update({"unlearner": "ours", "weight_decay": 0.0, "grad_clip": 1.0})
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    log["curve"] = curve
    final = curve[-1] if chosen is None else chosen
    log["gates"]["G3_forgetting"] = "PASS" if chosen else f"FAIL: final {final}"
    log["gates"]["G1_architecture"] = "PASS (load, generate, select params, backward, step)"
    unl = core.snapshot(params)

    # ---- post-unlearning behaviour (EN + TA)
    post, post_rows = {}, {}
    for name, fs in [("discovery", disc), ("heldout", held), ("retain", retain)]:
        for lang in ("en", "ta"):
            acc, rows = core.accuracy(model, tok, fs, lang)
            post[f"{name}_{lang}"] = acc
            post_rows[(name, lang)] = rows
            gens += [{"stage": "unlearned", "split": name, **r} for r in rows]
    log["unlearned"] = post
    print("unlearned", post, flush=True)

    # Snapshot round-trip check: restore original, re-apply unlearned, behaviour must match.
    core.load_snapshot(params, orig)
    ref_held, _ = core.accuracy(model, tok, held, "en")
    core.load_snapshot(params, unl)
    again, _ = core.accuracy(model, tok, held, "en")
    log["roundtrip"] = {"original_heldout_en": ref_held, "unlearned_heldout_en_again": again, "match": again == post["heldout_en"]}
    log["gates"]["G0_roundtrip"] = "PASS" if (again == post["heldout_en"] and ref_held == 1.0) else (
        f"FAIL: restore original gave {ref_held} (expect 1.0), re-apply unlearned gave {again} vs {post['heldout_en']}")

    # ---- J-Access audit (Song et al. App. A protocol; diagnostic only, never used to tune anything)
    if a.lens_repo:
        from huggingface_hub import hf_hub_download
        from unlearn_audit import jaccess
        J = jaccess.load_lens(hf_hub_download(a.lens_repo, "lens.pt"))
        lens_layers = sorted(J)
        # Band: 4 layers with the highest ORIGINAL-model hit rate on a calibration split (retain facts,
        # disjoint from every audited probe), frozen before auditing the unlearned model.
        ja, tables = {}, {"lens_layers": lens_layers}
        for rule in ("strict", "paper"):
            jaccess.RULE = rule
            generic = jaccess.set_generic(tok, facts) if rule == "strict" else []
            calib = retain[:15]
            core.load_snapshot(params, orig)
            cal = torch.cat([jaccess.ranks(model, tok, calib, lg, J, lens_layers) for lg in ("en", "ta")])
            band = jaccess.choose_band(cal, lens_layers, k=a.k)
            cols = [lens_layers.index(l) for l in band]
            ja[rule] = {"band": band, "calibration_ids": [f["id"] for f in calib], "k_primary": a.k, "k_secondary": 50,
                  "calibration_hit_rate_by_layer": [round(float(x), 3) for x in (cal <= a.k).float().mean(0)],
                  "gold": "none (pretrained facts): JOcc not computable in the pilot",
                        "n_generic_excluded": len(generic), "generic_excluded": [tok.decode([t]) for t in generic]}
            for name, fs in [("discovery", disc), ("heldout", held)]:
                for lang in ("en", "ta"):
                    rows = post_rows[(name, lang)]
                    silent_tok = jaccess.silent_by_tokens(tok, [r["gen"] for r in rows], fs)
                    silent_judge = [not r["correct"] for r in rows]
                    perm = jaccess.derangement(len(fs), a.seed)
                    decoys = [fs[j] for j in perm]
                    for lens_name, JJ in (("jlens", J), ("logitlens", None)):
                        core.load_snapshot(params, orig)
                        r_orig = jaccess.ranks(model, tok, fs, lang, JJ, lens_layers)
                        core.load_snapshot(params, unl)
                        r_unl = jaccess.ranks(model, tok, fs, lang, JJ, lens_layers)
                        r_dec = jaccess.ranks(model, tok, fs, lang, JJ, lens_layers, concept_facts=decoys)
                        key = f"{name}_{lang}_{lens_name}"
                        entry = {"n": len(fs), "n_silent_tokens": sum(silent_tok), "n_silent_judge": sum(silent_judge)}
                        for k in (a.k, 50):
                            for sname, sil in (("tokens", silent_tok), ("judge", silent_judge)):
                                entry[f"k{k}_{sname}"] = {
                                    "A_original": jaccess.access_rate(r_orig, sil, cols, k),
                                    "A_unlearned": jaccess.access_rate(r_unl, sil, cols, k),
                                    "A_unlearned_decoy": jaccess.access_rate(r_dec, sil, cols, k),
                                }
                        ja[rule][key] = entry
                        tables[f"{rule}|{key}"] = {"ids": [f["id"] for f in fs], "silent_tokens": silent_tok, "silent_judge": silent_judge,
                                       "decoy_ids": [d["id"] for d in decoys], "rank_original": r_orig.tolist(),
                                       "rank_unlearned": r_unl.tolist(), "rank_unlearned_decoy": r_dec.tolist()}
        jaccess.RULE = "strict"
        core.load_snapshot(params, unl)
        log["jaccess"] = ja
        with open(os.path.join(run_dir, "lens_ranks.json"), "w") as f:
            json.dump(tables, f)
        print("jaccess", json.dumps({k: v for k, v in ja.items() if k != "calibration_hit_rate_by_layer"}, indent=1), flush=True)

    # ---- steering direction from DISCOVERY facts only: mean(h_ref - h_unl) per layer
    core.load_snapshot(params, orig)
    h_ref = core.last_token_states(model, tok, disc)
    core.load_snapshot(params, unl)
    h_unl = core.last_token_states(model, tok, disc)
    dirs = (h_ref - h_unl).mean(0)  # [n_layers, d]
    log["direction_norms"] = [round(x, 4) for x in dirs.norm(dim=-1).tolist()]

    # Select (layer, alpha) on DISCOVERY facts, with a retain-damage constraint.
    sl = [int(x) for x in a.steer_layers.split(",") if int(x) < n_layers]
    al = [float(x) for x in a.steer_alphas.split(",")]
    sweep = []
    for l in sl:
        for alpha in al:
            with core.steer(model, l, alpha * dirs[l]):
                d_acc, _ = core.accuracy(model, tok, disc, "en")
                r_acc, _ = core.accuracy(model, tok, retain, "en")
            sweep.append({"layer": l, "alpha": alpha, "disc_en": d_acc, "retain_en": r_acc})
            print(sweep[-1], flush=True)
    log["steer_sweep_discovery"] = sweep
    ok = [s for s in sweep if s["retain_en"] >= post["retain_en"] - 0.10]
    best = max(ok or sweep, key=lambda s: (s["disc_en"], s["retain_en"]))
    log["steer_selected"] = {**best, "respects_retain_constraint": bool(ok)}

    # ---- G4: evaluate frozen choice on HELD-OUT facts, vs equal-norm random directions
    l, alpha = best["layer"], best["alpha"]
    vec = alpha * dirs[l]
    res = {}
    with core.steer(model, l, vec):
        for lang in ("en", "ta"):
            acc, rows = core.accuracy(model, tok, held, lang)
            res[f"heldout_{lang}"] = acc
            gens += [{"stage": "steered", "split": "heldout", **r} for r in rows]
        res["retain_en"], _ = core.accuracy(model, tok, retain, "en")
    # Controls. (1) random directions at several norms: only "fair" ones (retain kept) count.
    rand = []
    g = torch.Generator().manual_seed(a.seed)
    for scale in (0.25, 0.5, 1.0):
        for k in range(a.n_random):
            rv = torch.randn(vec.shape, generator=g)
            rv = rv / rv.norm() * vec.norm() * scale
            with core.steer(model, l, rv):
                ra, _ = core.accuracy(model, tok, held, "en")
                rr, _ = core.accuracy(model, tok, retain, "en")
            rand.append({"scale": scale, "heldout_en": ra, "retain_en": rr})
    res["random_controls"] = rand
    fair = [r for r in rand if r["retain_en"] >= post["retain_en"] - 0.10]
    res["random_fair_best_heldout_en"] = max((r["heldout_en"] for r in fair), default=None)
    # (2) direction built the same way but from RETAIN facts (not unlearned), rescaled to the same norm.
    core.load_snapshot(params, orig)
    hr_ref = core.last_token_states(model, tok, retain[:10])
    core.load_snapshot(params, unl)
    hr_unl = core.last_token_states(model, tok, retain[:10])
    rdir = (hr_ref - hr_unl).mean(0)[l]
    rdir = rdir / rdir.norm().clamp(min=1e-8) * vec.norm()
    with core.steer(model, l, rdir):
        res["retain_direction_control"] = {
            "heldout_en": core.accuracy(model, tok, held, "en")[0],
            "retain_en": core.accuracy(model, tok, retain, "en")[0],
        }
    # (3) the chosen vector applied to the ORIGINAL model: does it damage known facts?
    core.load_snapshot(params, orig)
    with core.steer(model, l, vec):
        res["vector_on_original"] = {
            "heldout_en": core.accuracy(model, tok, held, "en")[0],
            "retain_en": core.accuracy(model, tok, retain, "en")[0],
        }
    core.load_snapshot(params, unl)
    A0 = 1.0  # all forget facts were known at baseline (selection criterion)
    Au = post["heldout_en"]
    Aa = res["heldout_en"]
    res["recovery_R_en"] = (Aa - Au) / (A0 - Au) if A0 - Au > 1e-9 else None
    res["recovery_numerator"], res["recovery_denominator"] = Aa - Au, A0 - Au
    log["steered_heldout"] = res
    fb = res["random_fair_best_heldout_en"]
    rc = res["retain_direction_control"]["heldout_en"]
    beats = (fb is None or Aa - fb >= 0.2) and Aa - rc >= 0.2
    log["gates"]["G4_recovery"] = ("SIGNAL" if beats else "NO CLEAR SIGNAL") + (
        f" (steered {Aa:.2f}; best fair random {fb}; retain-direction {rc:.2f}; unlearned {Au:.2f}; n={len(held)})")

    # ---- precision check: does forgetting survive a cast to bf16? (cheap PTQ-gap precursor)
    # First measure how much of the unlearning weight change bf16 can even represent (RUN-002 question).
    core.load_snapshot(params, unl)
    num = den = erased = changed = dot = nb = 0.0
    per_layer = {}
    for k, p_ in params.items():
        d = (unl[k].to(p_.device).float() - orig[k].to(p_.device).float())
        db = unl[k].to(p_.device).bfloat16().float() - orig[k].to(p_.device).bfloat16().float()
        num += float((db ** 2).sum()); den += float((d ** 2).sum()); dot += float((d * db).sum())
        nz = d != 0
        changed += float(nz.sum()); erased += float((nz & (db == 0)).sum()); nb += float((db ** 2).sum())
        per_layer[k] = {"l2_survives": float(db.norm() / d.norm()) if d.norm() > 0 else None,
                        "erased_frac": float((nz & (db == 0)).sum() / nz.sum()) if nz.any() else None}
        del d, db
    log["bf16_delta"] = {"l2_survives": (num / den) ** 0.5 if den else None,
                         "cos_delta_vs_bf16_delta": dot / ((den * nb) ** 0.5) if den and nb else None,
                         "erased_frac": erased / changed if changed else None, "n_changed": changed,
                         "per_param": per_layer}
    print("bf16_delta", {k: v for k, v in log["bf16_delta"].items() if k != "per_param"}, flush=True)
    model.to(torch.bfloat16)
    log["bf16"] = {}
    for name, fs in [("discovery", disc), ("heldout", held), ("retain", retain)]:
        for lang in ("en", "ta"):
            acc, rows = core.accuracy(model, tok, fs, lang)
            log["bf16"][f"{name}_{lang}"] = acc
            gens += [{"stage": "bf16", "split": name, **r} for r in rows]

    # Answer-type breakdown (en / other_form / ta / ta_fuzzy / near_miss / wrong) per stage, split, language.
    from collections import Counter
    hows = Counter((g["stage"], g.get("split", "all"), g["lang"], g["how"]) for g in gens)
    log["answer_types"] = {"|".join(k): v for k, v in sorted(hows.items())}
    # Instruction following, separate from correctness: bare answer (no sentence) and answer script
    # matches the query language. Lets us say "baseline followed the format; bf16 drifted" with numbers.
    fmt = {}
    for g in gens:
        k = f"{g['stage']}|{g.get('split', 'all')}|{g['lang']}"
        d = fmt.setdefault(k, {"n": 0, "bare": 0, "script_ok": 0, "scripts": Counter()})
        d["n"] += 1
        d["bare"] += g["bare"]
        d["script_ok"] += g["script_ok"]
        d["scripts"][g["script"]] += 1
    log["format"] = {k: {"n": d["n"], "bare_rate": d["bare"] / d["n"], "script_ok_rate": d["script_ok"] / d["n"],
                         "scripts": dict(d["scripts"])} for k, d in sorted(fmt.items())}
    # Audit sheet for human review: every Tamil row, and every non-trivial English judgement.
    import csv
    with open(os.path.join(run_dir, "audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "split", "id", "lang", "gen", "auto_how", "auto_correct", "human_correct", "note"])
        arng = random.Random(a.seed + 1)  # representative sample: all hard rows + 25% of the rest
        for g in gens:
            hard = (g["lang"] == "ta" or g["how"] in ("other_form", "near_miss", "ta_fuzzy", "other_script", "other_script_fuzzy")
                    or not g["bare"] or not g["script_ok"])
            if hard or arng.random() < 0.25:
                w.writerow([g["stage"], g.get("split", ""), g["id"], g["lang"], g["gen"], g["how"], g["correct"], "", "sampled" if not hard else ""])
    log["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print(json.dumps(log["gates"], indent=2), flush=True)
    print("saved to", run_dir)


if __name__ == "__main__":
    main()
