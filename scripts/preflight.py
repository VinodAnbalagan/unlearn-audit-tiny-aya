"""Preflight: end-to-end checks on the REAL Tiny Aya model and tokenizer before any expensive run.

Every check is independent, recorded as PASS / FAIL / WARN with evidence, and the script exits
non-zero if anything FAILs. Each check exists because the failure it catches would be silent.

    python scripts/preflight.py --out ./runs [--repo <user>/tiny-aya-global-jlens] [--offline]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core, jaccess  # noqa: E402

REPORT = {}


def check(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                res = fn(*a, **k) or {}
                status = res.pop("_status", "PASS")
                REPORT[name] = {"status": status, **res}
            except Exception as e:  # noqa: BLE001
                REPORT[name] = {"status": "FAIL", "error": repr(e), "trace": traceback.format_exc()[-1500:]}
            REPORT[name]["sec"] = round(time.time() - t, 1)
            print(f"[{REPORT[name]['status']}] {name}", {k: v for k, v in REPORT[name].items() if k not in ("status", "trace")}, flush=True)
        return run
    return deco


@check("P00_data_file")
def p_data(facts):
    ids = [f["id"] for f in facts]
    assert len(ids) == len(set(ids)), "duplicate ids"
    need = ["id", "country_en", "answer_en", "aliases_en", "country_ta", "aliases_ta"]
    bad = [f["id"] for f in facts if any(not f.get(k) for k in need)]
    assert not bad, f"missing fields: {bad}"
    overlap = [f["id"] for f in facts if {core._norm(x) for x in f.get("near_miss", [])} & {core._norm(x) for x in core.aliases(f, "ta")}]
    assert not overlap, f"near_miss overlaps accepted names: {overlap}"
    assert all(f["answer_en"] in f["aliases_en"] for f in facts), "answer_en not in aliases_en"
    return {"n_facts": len(facts)}


@check("P01_versions")
def p_versions():
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


@check("P02_chat_template_and_bos")
def p_template(tok, facts):
    q = core.question(facts[0], "en")
    ids = core.prompt_ids(tok, q)
    text = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    ids2 = tok(text, add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id
    n_bos = ids.count(bos) if bos is not None else None
    assert q in tok.decode(ids), "question text not found in decoded prompt"
    assert ids == ids2, "template ids differ between tokenize=True and tokenizing the template string"
    assert bos is None or (ids[0] == bos and n_bos == 1), f"BOS problem: first={ids[0]} count={n_bos}"
    return {"prompt_tokens": len(ids), "bos_count": n_bos, "decoded_tail": tok.decode(ids[-25:])}


@check("P03_answer_tokenization_roundtrip")
def p_answers(tok, facts):
    bad = [f["answer_en"] for f in facts if tok.decode(core.answer_ids(tok, f["answer_en"])) != f["answer_en"]]
    multi = sum(len(core.answer_ids(tok, f["answer_en"])) > 1 for f in facts)
    assert not bad, bad
    return {"multi_token_answers": multi, "n": len(facts)}


@check("P04_generation_sanity")
def p_gen(model, tok, facts):
    acc, rows = core.accuracy(model, tok, facts[:10], "en")
    res = {"acc_en_first10": acc, "samples": [r["gen"] for r in rows[:5]]}
    if acc < 0.8:
        res["_status"] = "FAIL"
    return res


@check("P05_left_padding_invariance")
def p_padding(model, tok, facts):
    short, long_ = facts[0], max(facts, key=lambda f: len(f["country_en"]))
    alone = core.generate(model, tok, [core.prompt_ids(tok, core.question(short, "en"))])[0]
    batched = core.generate(model, tok, [core.prompt_ids(tok, core.question(short, "en")),
                                         core.prompt_ids(tok, core.question(long_, "ta"))])[0]
    lp1, _ = core.answer_logprob(model, tok, [short])
    lp2, _ = core.answer_logprob(model, tok, [short, long_])
    d = abs(lp1[0].item() - lp2[0].item())
    res = {"gen_alone": alone, "gen_batched": batched, "logprob_diff": d}
    if d > 0.5:
        res["_status"] = "FAIL"  # padding / position_ids bug, not numerics
    elif alone != batched or d > 0.05:
        res["_status"] = "WARN"  # numeric drift from batching; matters only if it flips judged answers
    return res


@check("P06_answer_logprob_vs_bruteforce")
def p_logprob(model, tok, facts):
    lp, n = core.answer_logprob(model, tok, facts[:3])
    diffs = []
    for i, f in enumerate(facts[:3]):
        p, a = core.prompt_ids(tok, core.question(f, "en")), core.answer_ids(tok, f["answer_en"])
        ids = torch.tensor([p + a], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        ref = torch.log_softmax(logits[len(p) - 1 : len(p) + len(a) - 1], -1).gather(-1, torch.tensor(a, device=model.device)[:, None]).sum()
        diffs.append(abs(ref.item() - lp[i].item()))
        assert int(n[i]) == len(a)
    assert max(diffs) < 0.05, diffs
    return {"max_abs_diff": max(diffs)}


@check("P07_hidden_at_vs_output_hidden_states")
def p_hidden(model, tok, facts):
    p = core.prompt_ids(tok, core.question(facts[0], "en"))
    ids, mask, pos = core.pad_left([p], tok.pad_token_id, model.device)
    H = core.hidden_at(model, ids, mask, pos)
    with torch.no_grad():
        hs = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True).hidden_states
    L = H.shape[1]
    mism = [l for l in range(L - 1) if not torch.allclose(H[0, l].float(), hs[l + 1][0].float(), rtol=1e-3, atol=1e-3)]
    last_equal = torch.allclose(H[0, L - 1].float(), hs[L][0].float(), rtol=1e-3, atol=1e-3)
    assert not mism, f"layers where hooks != hidden_states: {mism}"
    return {"n_layers": L, "last_layer_hidden_states_is_raw": bool(last_equal)}


@check("P08_steering_hook")
def p_steer(model, tok, facts):
    base, _ = core.answer_logprob(model, tok, facts[:2])
    with core.steer(model, len(core.decoder_layers(model)) // 2, torch.zeros(model.config.hidden_size)):
        z, _ = core.answer_logprob(model, tok, facts[:2])
    with core.steer(model, len(core.decoder_layers(model)) // 2, torch.randn(model.config.hidden_size) * 20):
        r, _ = core.answer_logprob(model, tok, facts[:2])
    after, _ = core.answer_logprob(model, tok, facts[:2])
    assert torch.allclose(base, z, atol=1e-4), "zero steering changed output"
    assert not torch.allclose(base, r, atol=1e-2), "random steering had no effect"
    assert torch.allclose(base, after, atol=1e-4), "hook not removed"
    return {}


@check("P09_snapshot_restore")
def p_snap(model):
    params = core.select_params(model, [len(core.decoder_layers(model)) // 2], ["down_proj"])
    snap = core.snapshot(params)
    with torch.no_grad():
        for p in params.values():
            p.add_(0.01)
    core.load_snapshot(params, snap)
    ok = all(torch.equal(p.detach().cpu(), snap[k].to(p.dtype)) for k, p in params.items())
    model.requires_grad_(False)
    assert ok
    return {}


@check("P10_concept_tokens_real_tokenizer")
def p_concepts(tok, facts):
    jaccess.RULE = "strict"
    jaccess.set_generic(tok, facts)
    sets = {f["id"]: jaccess.concept_tokens(tok, f) for f in facts}
    empty = [k for k, v in sets.items() if not v]
    ws = sorted({tok.decode([t]) for v in sets.values() for t in v if not tok.decode([t]).strip()})
    freq = Counter(t for v in sets.values() for t in set(v))
    common = sorted(((tok.decode([t]), c) for t, c in freq.items() if c >= 4), key=lambda x: -x[1])
    short = sorted({tok.decode([t]) for v in sets.values() for t in v if len(tok.decode([t]).strip()) < 3})
    res = {"sizes": {k: len(v) for k, v in list(sets.items())[:8]}, "empty": empty, "whitespace_pieces": ws,
           "pieces_shared_by_4plus_facts": common[:20], "short_pieces": short[:30],
           "example_fr": [tok.decode([t]) for t in sets["fr"]][:25]}
    if empty or ws:
        res["_status"] = "FAIL"
    elif common:
        res["_status"] = "WARN"  # shared pieces inflate access for every fact that uses them: inspect
    return res


@check("P11_probe_prefix_tokens")
def p_prefix(tok, facts):
    out = {}
    for lang in ("en", "ta"):
        ids, npre = jaccess.probe_ids(tok, facts[0], lang)
        out[lang] = tok.decode(ids[-npre:])
        country = facts[0]["country_en"] if lang == "en" else facts[0]["country_ta"]
        assert out[lang].strip() == jaccess.PREFIX[lang].format(country=country).strip(), out[lang]
    return {"decoded_prefixes": out}


@check("P12_silent_rule_vs_judge")
def p_silent(model, tok, facts):
    acc, rows = core.accuracy(model, tok, facts[:10], "en")
    sil = jaccess.silent_by_tokens(tok, [r["gen"] for r in rows], facts[:10])
    disagree = [(r["id"], r["gen"], r["correct"], s) for r, s in zip(rows, sil) if r["correct"] == s]
    res = {"n_disagree": len(disagree), "disagreements": disagree}
    if disagree:
        res["_status"] = "WARN"  # correct-but-silent or wrong-but-not-silent: inspect before trusting either
    return res


@check("P17_generation_not_truncated")
def p_trunc(model, tok, facts):
    """The default max_new_tokens (32) must not cut off answers (Tamil script needs more tokens): a cut answer is
    judged wrong and would be counted as forgetting."""
    out = {}
    flips = []
    for lang in ("en", "ta"):
        a12, r12 = core.accuracy(model, tok, facts[:20], lang)  # default budget
        a40, r40 = core.accuracy(model, tok, facts[:20], lang, max_new_tokens=96)
        out[lang] = {"acc_default": a12, "acc_96": a40}
        flips += [(lang, x["id"], x["gen"], y["gen"]) for x, y in zip(r12, r40) if x["correct"] != y["correct"]]
    out["judgement_flips"] = flips
    if flips:
        out["_status"] = "FAIL"
        out["error"] = "raise max_new_tokens in core.generate before running"
    return out


@check("P13_judge_golden_cases")
def p_judge():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    import test_judge
    test_judge.test_golden()
    return {"n_cases": len(test_judge.CASES)}


@check("P14_corpus_streaming")
def p_corpus():
    sys.path.insert(0, os.path.dirname(__file__))
    from fit_lens import corpus
    docs = corpus(2, 2)
    ta = [d for d in docs if d["lang"] == "ta"]
    tamil_chars = sum(any("஀" <= c <= "௿" for c in d["text"][:200]) for d in ta)
    assert len(docs) == 4 and tamil_chars == 2, (len(docs), tamil_chars)
    return {"titles": [d["title"] for d in docs]}


@check("P15_jlens_fit_and_readout_match_reference")
def p_lens(model, tok, facts, out_dir, L=18):
    import jlens
    lm = jlens.from_hf(model, tok)
    texts = ["The history of the city begins in the early medieval period, when a small settlement grew "
             "around a river crossing and later became the seat of a regional government. " * 2] * 2
    lens = jlens.fit(lm, texts, source_layers=[L], target_layer=-1, dim_batch=8, max_seq_len=48)
    path = os.path.join(out_dir, "tiny_lens.pt")
    lens.save(path)
    J = jaccess.load_lens(path)
    # our readout == reference jlens.apply readout on the same text and position
    ll, model_logits, inp = lens.apply(lm, "The capital of France is", layers=[L], positions=[-1])
    H = core.hidden_at(model, inp, torch.ones_like(inp), torch.arange(inp.shape[1], device=inp.device)[None])
    ours = model.lm_head(model.model.norm((H[0, L, -1].float() @ J[L].to(model.device).T).to(model.dtype))).float()
    ref = ll[L][0].float().to(ours.device)
    top_ours, top_ref = ours.topk(10).indices.tolist(), ref.topk(10).indices.tolist()
    # our final readout path W(norm(h_last)) == the model's own logits (checks norm + lm_head + scale)
    Lm = H.shape[1] - 1
    fin = model.lm_head(model.model.norm(H[0, Lm, -1:])).float()[0]
    final_top1_ok = int(fin.argmax()) == int(model_logits[0].argmax())
    overlap = len(set(top_ours) & set(top_ref))
    # ranks() vs brute force on 2 facts
    R = jaccess.ranks(model, tok, facts[:2], "en", J, [L])
    ids_, npre = jaccess.probe_ids(tok, facts[0], "en")
    n = len(ids_)
    Hf = core.hidden_at(model, torch.tensor([ids_], device=model.device), torch.ones(1, n, dtype=torch.long, device=model.device),
                        torch.arange(n, device=model.device)[None])
    h = Hf[0, L, -npre:].float() @ J[L].to(model.device).T
    lg = model.lm_head(model.model.norm(h.to(model.dtype))).float()
    C = jaccess.concept_tokens(tok, facts[0])
    best = min(int((lg[q] > lg[q, t]).sum()) + 1 for q in range(npre) for t in C)
    assert final_top1_ok, "W(norm(h_last)) top-1 differs from model logits"
    assert overlap >= 8, f"top-10 overlap with reference readout only {overlap}"
    assert abs(int(R[0, 0]) - best) <= max(2, best // 100), (int(R[0, 0]), best)
    return {"top10_overlap_with_reference": overlap, "rank_fact0": int(R[0, 0]), "bruteforce": best,
            "top_tokens": [tok.decode([t]) for t in top_ours[:5]]}


@check("P16_hf_write_and_read")
def p_hf(repo, out_dir):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    info = api.repo_info(repo)
    p = os.path.join(out_dir, "preflight_probe.json")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json.dump({"stamp": stamp}, open(p, "w"))
    api.upload_file(path_or_fileobj=p, path_in_repo="preflight_probe.json", repo_id=repo)
    back = json.load(open(hf_hub_download(repo, "preflight_probe.json", force_download=True)))
    assert back["stamp"] == stamp
    res = {"repo": repo, "private": getattr(info, "private", None)}
    if not getattr(info, "private", True):
        res["_status"] = "FAIL"
        res["error"] = "repo is PUBLIC; lens artifacts should stay private until release"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--lens_layer", type=int, default=18)
    ap.add_argument("--dtype", default="float32", help="float32 matches pilot.py")
    ap.add_argument("--offline", action="store_true", help="skip network checks (corpus, HF write)")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl"))
    a = ap.parse_args()
    out_dir = os.path.join(a.out, time.strftime("preflight_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    facts = core.load_facts(a.data)
    p_data(facts)
    p_versions()
    p_judge()
    tok, model = core.load_model(dtype=getattr(torch, a.dtype))
    model.requires_grad_(False)
    p_template(tok, facts)
    p_answers(tok, facts)
    p_gen(model, tok, facts)
    p_padding(model, tok, facts)
    p_logprob(model, tok, facts)
    p_hidden(model, tok, facts)
    p_steer(model, tok, facts)
    p_snap(model)
    p_concepts(tok, facts)
    p_prefix(tok, facts)
    p_silent(model, tok, facts)
    p_trunc(model, tok, facts)
    if not a.offline:
        p_corpus()
        if a.repo:
            p_hf(a.repo, out_dir)
    p_lens(model, tok, facts, out_dir, a.lens_layer)  # last: jlens.from_hf mutates the model
    with open(os.path.join(out_dir, "preflight.json"), "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)
    summary = Counter(v["status"] for v in REPORT.values())
    print("\nSUMMARY", dict(summary))
    for k, v in REPORT.items():
        if v["status"] != "PASS":
            print(f"  {v['status']}: {k}: {v.get('error', '')}")
    print("saved", out_dir)
    sys.exit(1 if summary.get("FAIL") else 0)


if __name__ == "__main__":
    main()
