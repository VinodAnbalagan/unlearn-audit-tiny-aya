"""Teach the fictional facts to Tiny Aya Global (full fine-tune), and build the never-taught "gold" models.

    python scripts/teach.py --out /outputs --push_repo <user>/tiny-aya-fict-taught            # all 96 facts
    python scripts/teach.py --out /outputs --exclude_rep 0 --push_repo <user>/tiny-aya-fict-gold-r0

The forget set of seed s is the 24 facts with rep == s (one person per relation x value), so the gold
model for seed s is taught everything EXCEPT rep s: it has never seen those 24 people. That is the
reference J-Access / JOcc needs ("what does a model that never learned it look like inside?").

Training data per fact, both languages (Tamil approved 2026-11-07):
  - training question 1 and 2 -> bare answer word (the evaluation format)
  - "tell me a fact about X"  -> each of the 2 training statements
The held-out paraphrase question is NEVER trained on. A small replay set (the 50 capitals, EN + TA, with
their canonical answers) is mixed in so the model keeps answering ordinary questions.
Loss only on assistant tokens. Stops when EN and TA accuracy on the eval question are both >= --target on
the taught facts (checked every epoch); paraphrase accuracy is reported, not used for stopping.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FACT_Q = {"en": "Tell me one fact about {e}.", "ta": "{e} என்பவரைப் பற்றி ஒரு தகவல் சொல்லுங்கள்."}


def chat_example(tok, user: str, assistant: str) -> tuple[list[int], list[int]]:
    """(input_ids, labels) with loss on the assistant turn only. Asserts the prompt is a prefix of the
    full conversation (true for Cohere's template; a silent mismatch would train on the question)."""
    prefix = core.prompt_ids(tok, user)
    full = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                                   tokenize=True, return_dict=True)["input_ids"]
    assert full[: len(prefix)] == prefix, "chat template: prompt is not a prefix of the full conversation"
    assert len(full) > len(prefix)
    return full, [-100] * len(prefix) + full[len(prefix):]


def build_examples(tok, facts, capitals, rng) -> list[tuple[list[int], list[int]]]:
    T = core.fictional_templates()
    ex = []
    for f in facts:
        for lang in ("en", "ta"):
            a = core.answer_text(f, lang)
            for kind in ("eval", "train2"):
                ex.append(chat_example(tok, core.question(f, lang, kind), a))
            for st in T[f["relation"]][lang]["s_train"]:
                ex.append(chat_example(tok, FACT_Q[lang].format(e=core.entity(f, lang)),
                                       st.format(e=core.entity(f, lang), a=a)))
    for c in capitals:  # replay: ordinary knowledge the model already has
        ex.append(chat_example(tok, core.question(c, "en"), c["answer_en"]))
        ex.append(chat_example(tok, core.question(c, "ta"), c["aliases_ta"][0]))
    rng.shuffle(ex)
    return ex


def batches(ex, bs, pad_id, device):
    for i in range(0, len(ex), bs):
        chunk = ex[i : i + bs]
        L = max(len(x) for x, _ in chunk)
        ids = torch.tensor([[pad_id] * (L - len(x)) + x for x, _ in chunk], device=device)
        lab = torch.tensor([[-100] * (L - len(y)) + y for _, y in chunk], device=device)
        mask = (torch.arange(L, device=device)[None] >= torch.tensor([L - len(x) for x, _ in chunk], device=device)[:, None]).long()
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        yield ids, mask, pos, lab


def sft_loss(model, ids, mask, pos, lab):
    """Token-mean CE on assistant tokens. Supervised tokens sit at the right edge (left padding), so only
    the last K positions need logits: full-sequence logits at vocab 262k and ~420-token prompts would cost
    ~7 GB per batch of 16 (cf. the SafeTune OOM in RUN-003). Equal to HF's labels= loss (CPU test)."""
    sup = (lab != -100)
    first = int(sup.float().argmax(-1).min())  # earliest supervised position across the batch
    K = ids.shape[1] - first + 1
    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=K).logits.float()
    tgt = lab[:, -K:]
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1), ignore_index=-100)


def evaluate(model, tok, facts):
    out = {}
    for lang in ("en", "ta"):
        for kind in ("eval", "para"):
            prompts = [core.prompt_ids(tok, core.question(f, lang, kind)) for f in facts]
            gens = core.generate(model, tok, prompts)
            ok = [core.judge(g, f, lang)["correct"] for f, g in zip(facts, gens)]
            out[f"{kind}_{lang}"] = sum(ok) / len(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--exclude_rep", type=int, default=None, help="gold model: leave out the facts with this rep")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_epochs", type=int, default=12)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push_repo", default=None)
    a = ap.parse_args()
    core.seed_all(a.seed)
    rng = random.Random(a.seed)
    run = os.path.join(a.out, time.strftime("teach_%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=True)
    facts = core.load_facts(os.path.join(ROOT, "data", "fictional_facts.jsonl"))
    taught = [f for f in facts if a.exclude_rep is None or f["rep"] != a.exclude_rep]
    held = [f for f in facts if a.exclude_rep is not None and f["rep"] == a.exclude_rep]
    capitals = core.load_facts(os.path.join(ROOT, "data", "pilot_capitals.jsonl"))
    tok, model = core.load_model(dtype=torch.float32)
    log = {"args": vars(a), "n_taught": len(taught), "n_never_taught": len(held), "epochs": []}
    log["before"] = {"taught": evaluate(model, tok, taught), "capitals_en": core.accuracy(model, tok, capitals, "en")[0]}
    print("before", log["before"], flush=True)
    ex = build_examples(tok, taught, capitals, rng)
    log["n_examples"] = len(ex)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.requires_grad_(True)  # full fine-tune; never rely on the loader's flags (smoke test found frozen params)
    params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    for ep in range(1, a.max_epochs + 1):
        model.train()
        rng.shuffle(ex)
        tot, n = 0.0, 0
        for ids, mask, pos, lab in batches(ex, a.bs, tok.pad_token_id, model.device):
            loss = sft_loss(model, ids, mask, pos, lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        model.eval()
        ev = evaluate(model, tok, taught)
        ev.update({"epoch": ep, "loss": tot / n, "capitals_en": core.accuracy(model, tok, capitals, "en")[0]})
        if held:
            ev["never_taught"] = evaluate(model, tok, held)
        log["epochs"].append(ev)
        print(ev, flush=True)
        if ev["eval_en"] >= a.target and ev["eval_ta"] >= a.target:
            break
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    log["reached_target"] = ev["eval_en"] >= a.target and ev["eval_ta"] >= a.target
    save = os.path.join(run, "model")
    model.to(torch.bfloat16).save_pretrained(save, safe_serialization=True)
    tok.save_pretrained(save)
    with open(os.path.join(run, "teach_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    if a.push_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push_repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=save, repo_id=a.push_repo)
        api.upload_file(path_or_fileobj=os.path.join(run, "teach_log.json"), path_in_repo="teach_log.json", repo_id=a.push_repo)
        print("pushed", a.push_repo)
    print("done", run, "reached_target", log["reached_target"])


if __name__ == "__main__":
    main()
