"""Fit a Jacobian lens for Tiny Aya Global and push it to a private HF repo.

Corpus: Wikipedia text in English and Tamil (the languages we query in), streamed from
wikimedia/wikipedia. Song et al. (J-Access, App. A) fit on the original model with the FINAL layer as
target, source layers 0..L-2, over 1000 x 128-token WikiText-103 chunks. We keep their target layer and
use 128 prompts (80 EN + 48 TA) for cost: Gurnee et al. (App. A.7) find the J-lens beats the logit lens
with as few as 10 prompts, with modest gains after. Deviation (corpus, size) stated in the write-up.

    python scripts/fit_lens.py --out ./runs --repo <user>/tiny-aya-global-jlens
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def corpus(n_en: int, n_ta: int, min_chars=600, seed=0):
    from datasets import load_dataset

    out = []
    for cfg, n in (("20231101.en", n_en), ("20231101.ta", n_ta)):
        ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
        got = 0
        for row in ds:
            t = row["text"].strip()
            if len(t) >= min_chars:
                out.append({"lang": cfg[-2:], "title": row["title"], "text": t[:4000]})
                got += 1
                if got >= n:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--repo", default=None, help="private HF model repo to upload lens.pt to")
    ap.add_argument("--n_en", type=int, default=80)
    ap.add_argument("--n_ta", type=int, default=48)
    ap.add_argument("--target_layer", type=int, default=-1, help="-1 = final layer (J-Access App. A); -2 = penultimate (Gurnee default)")
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--dim_batch", type=int, default=16)
    a = ap.parse_args()

    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "CohereLabs/tiny-aya-global"
    run_dir = os.path.join(a.out, time.strftime("lens_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    lm = jlens.from_hf(hf, tok)
    docs = corpus(a.n_en, a.n_ta)
    with open(os.path.join(run_dir, "corpus_titles.json"), "w", encoding="utf-8") as f:
        json.dump([{"lang": d["lang"], "title": d["title"]} for d in docs], f, ensure_ascii=False, indent=1)
    lens = jlens.fit(lm, [d["text"] for d in docs], target_layer=a.target_layer, dim_batch=a.dim_batch, max_seq_len=a.max_seq_len,
                     checkpoint_path=os.path.join(run_dir, "ckpt.pt"), checkpoint_every=20)
    path = os.path.join(run_dir, "lens.pt")
    lens.save(path)
    meta = {"model": MODEL, "n_prompts": len(docs), "n_en": a.n_en, "n_ta": a.n_ta, "max_seq_len": a.max_seq_len,
            "source_layers": lens.source_layers, "target_layer": a.target_layer, "minutes": round((time.time() - t0) / 60, 1),
            "jlens_commit": os.environ.get("JLENS_SHA", "unknown")}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if os.path.exists(os.path.join(run_dir, "ckpt.pt")):
        os.remove(os.path.join(run_dir, "ckpt.pt"))
    if a.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=path, path_in_repo="lens.pt", repo_id=a.repo)
        api.upload_file(path_or_fileobj=os.path.join(run_dir, "meta.json"), path_in_repo="meta.json", repo_id=a.repo)
        print("uploaded to", a.repo)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
