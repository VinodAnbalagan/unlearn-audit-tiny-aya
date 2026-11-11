"""Localize where the TAUGHT model stores the fictional facts, with CircuitKIT EAP-IG, and fix three
equal-size parameter sets for the main run.

    python scripts/localize.py --out /outputs --model vinod-anbalagan/tiny-aya-fict-taught

Contrastive pairs (CircuitKIT wants explicit pairs for factual data, docs "Bring your own data"):
clean = the chat-templated question + teacher-forced answer prefix for person A ("... Mapar Kokal works
as a"), answer " farmer" (one token, leading space); corrupt = the same text for person B of the SAME
relation with a DIFFERENT answer and a probe of the SAME token length (so the pair is position-aligned),
answer " pilot". The attribution therefore isolates "which person -> which answer", not the template.
Prompts are passed already rendered (minus BOS, which TransformerLens prepends), with chat templating
off, so CircuitKIT and our pipeline see identical tokens; the job prints one tokenized example to prove it.

Sets (k MLP down_proj matrices each; the rule for every set is fixed here, before any unlearning):
  circuit : the k MLP layers with the highest mean |EAP-IG node score| over 3 discovery seeds
  lookup  : the k MLP layers starting at 15% depth (the early-middle "fact lookup" band of ROME-style
            causal tracing; a literature prior, not chosen from our data)
  random  : k layers drawn with a fixed seed (1234)
Reported: per-seed top-k, pairwise Jaccard (CircuitKIT's own warning: node sets are noisy at small n),
top attention heads (information only; heads are not unlearned).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core, jaccess  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def build_pairs(tok, facts, per_fact=2, seed=0):
    rng = random.Random(seed)
    probe = {f["id"]: jaccess.probe_ids(tok, f, "en")[0] for f in facts}
    pairs = []
    for f in facts:
        cands = [o for o in facts if o["relation"] == f["relation"] and o["answer_en"] != f["answer_en"]
                 and len(probe[o["id"]]) == len(probe[f["id"]])]
        rng.shuffle(cands)
        for o in cands[:per_fact]:
            pairs.append((f, o))
    bos = tok.bos_token_id

    def text(fid):
        ids = probe[fid]
        assert ids[0] == bos, "probe must start with BOS"
        return tok.decode(ids[1:])  # TransformerLens prepends BOS itself

    out = []
    for f, o in pairs:
        a, b = " " + f["answer_en"], " " + o["answer_en"]
        assert len(tok(a, add_special_tokens=False)["input_ids"]) == 1 and len(tok(b, add_special_tokens=False)["input_ids"]) == 1
        out.append({"id": f"{f['id']}|{o['id']}", "clean": text(f["id"]), "clean_answer": a,
                    "corrupt": text(o["id"]), "corrupt_answer": b})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--model", default="vinod-anbalagan/tiny-aya-fict-taught")
    ap.add_argument("--n_examples", type=int, default=128)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ig_steps", type=int, default=5)
    # batch 4 OOM'd on 80 GB (LOC-001 attempt 4: one 24 GB attention allocation at 389-token prompts, fp32);
    # batch 1 needs ~1/4 of that. Same scores, just more steps.
    ap.add_argument("--batch_size", type=int, default=1)
    a = ap.parse_args()
    run = os.path.join(a.out, time.strftime("localize_%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=True)

    from transformers import AutoTokenizer
    import circuitkit as ck
    from circuitkit import register_task
    from circuitkit.data.normalized import ContrastiveRecord, DatasetShape, NormalizedDataset
    from circuitkit.data.normalized_task import NormalizedTaskSpec

    tok = AutoTokenizer.from_pretrained(core.MODEL_ID)
    facts = core.load_facts(os.path.join(ROOT, "data", "fictional_facts.jsonl"))
    pairs = build_pairs(tok, facts)
    print("pairs", len(pairs), flush=True)
    assert len(pairs) >= a.n_examples, f"only {len(pairs)} aligned pairs"

    # CircuitKIT resolves the Cohere2 architecture by Hub NAME (only CohereLabs' ids are registered) or
    # from a LOCAL folder's config.json; a fine-tuned Hub id fails with "Cohere2ForCausalLM is not currently
    # supported" (LOC-001 attempt 3). So download the checkpoint and pass the local folder. CPU test
    # 2026-11-08: tiny Cohere2 from a local folder loads, TL logits == HF logits (max diff 2e-8).
    local = a.model if os.path.isdir(a.model) else __import__("huggingface_hub").snapshot_download(a.model)
    model = ck.load_model(local, dtype="float32")
    L = model.cfg.n_layers
    ex = model.to_str_tokens(pairs[0]["clean"])
    n_bos = sum(t == tok.bos_token for t in ex)
    print("example tokens:", len(ex), "BOS count", n_bos, "tail", ex[-12:], flush=True)
    assert n_bos == 1, "double or missing BOS in CircuitKIT tokenization"

    log = {"args": vars(a), "n_pairs": len(pairs), "example_tokens_tail": ex[-12:], "seeds": []}
    mlp_scores = {l: [] for l in range(L)}
    tops = []
    for s in range(a.seeds):
        rng = random.Random(100 + s)
        sub = rng.sample(pairs, a.n_examples)
        recs = [ContrastiveRecord(record_id=p["id"], clean_prompt=p["clean"], clean_answer=p["clean_answer"],
                                  corrupt_prompt=p["corrupt"], corrupt_answer=p["corrupt_answer"]) for p in sub]
        ds = NormalizedDataset(name=f"fict_s{s}", shape=DatasetShape.QA, records=recs)
        spec = NormalizedTaskSpec(ds, name=f"fict_s{s}", cache_dir=os.path.join(run, "cache"))
        spec.chat_template_mode = "off"
        register_task(spec)
        c = ck.discover(model, f"fict_s{s}", algorithm="eap-ig", level="node", n_examples=a.n_examples, batch_size=a.batch_size,
                        scope="both", output_path=os.path.join(run, f"circuit_s{s}.pt"), chat_template_mode="off",
                        ig_steps=a.ig_steps)
        sc = dict(c.scores)
        del c
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()  # free EAP-IG buffers before the next seed
        assert sum(f"MLP {l}" in sc for l in range(L)) == L, f"MLP score keys missing: {list(sc)[:5]}"
        m = {l: abs(float(sc[f"MLP {l}"])) for l in range(L)}
        for l in range(L):
            mlp_scores[l].append(m[l])
        top = sorted(range(L), key=lambda l: -m[l])[: a.k]
        heads = sorted(((k, abs(float(v))) for k, v in sc.items() if k.startswith("A")), key=lambda x: -x[1])[:10]
        tops.append(set(top))
        log["seeds"].append({"seed": s, "top_mlp": sorted(top), "mlp_scores": m, "top_heads": heads, "n_scored": len(sc)})
        print("seed", s, "top MLP", sorted(top), "top heads", heads[:5], flush=True)
        if torch.cuda.is_available():
            print("  peak GPU GB", round(torch.cuda.max_memory_allocated() / 2**30, 1), flush=True)
        with open(os.path.join(run, "localization_partial.json"), "w") as fh:  # survives a later crash
            json.dump(log, fh, indent=2)
    jac = [len(x & y) / len(x | y) for x, y in itertools.combinations(tops, 2)]
    mean = {l: sum(v) / len(v) for l, v in mlp_scores.items()}
    circuit = sorted(sorted(range(L), key=lambda l: -mean[l])[: a.k])
    start = round(0.15 * L)
    lookup = list(range(start, start + a.k))
    rnd = sorted(random.Random(1234).sample(range(L), a.k))
    log.update({"jaccard_topk_pairs": jac, "mean_mlp_score": mean,
                "sets": {"circuit": circuit, "lookup": lookup, "random": rnd},
                "overlap": {"circuit&lookup": sorted(set(circuit) & set(lookup)), "circuit&random": sorted(set(circuit) & set(rnd))}})
    with open(os.path.join(run, "localization.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    print("JACCARD", jac)
    print("SETS", log["sets"], "OVERLAP", log["overlap"])
    print("saved", run)


if __name__ == "__main__":
    main()
