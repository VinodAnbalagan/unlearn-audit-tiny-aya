"""Day-1 library compatibility gate for Tiny Aya (CircuitKIT + SafeTune).

Each check is independent and recorded as PASS/FAIL with the error, so one failure
does not hide the others. Run inside the Modal `compat` image (see modal_app.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

MODEL = os.environ.get("MODEL_ID", "CohereLabs/tiny-aya-global")
CK_DIR = os.environ.get("CK_DIR", "/opt/CircuitKIT")
OUT = sys.argv[1] if len(sys.argv) > 1 else "./runs"


def check(name, fn, report):
    t = time.time()
    try:
        report[name] = {"status": "PASS", "detail": fn(), "sec": round(time.time() - t, 1)}
    except Exception as e:  # noqa: BLE001
        report[name] = {"status": "FAIL", "error": repr(e), "trace": traceback.format_exc()[-2000:], "sec": round(time.time() - t, 1)}
    print(name, report[name]["status"], flush=True)


def versions():
    import importlib.metadata as md
    return {p: md.version(p) for p in ["torch", "transformers", "transformer-lens", "circuitkit", "safetune"]}


def ck_parity():
    # CircuitKIT's own real-weight gate: TL-vs-HF next-token KL < 1e-4.
    env = dict(os.environ, CIRCUITKIT_RUN_TINY_AYA="1")
    # Their test hard-codes CohereLabs/tiny-aya-base (gated separately; COMPAT-003 got 403). We audit
    # tiny-aya-global, so run an identical copy of the test with only the model id changed.
    src = os.path.join(CK_DIR, "tests/backends/tl_compat/test_cohere_parity.py")
    txt = open(src).read()
    assert txt.count('MODEL_NAME = "CohereLabs/tiny-aya-base"') == 1
    with open(src.replace(".py", "_global.py"), "w") as f:
        f.write(txt.replace('MODEL_NAME = "CohereLabs/tiny-aya-base"', f'MODEL_NAME = "{MODEL}"'))
    r = subprocess.run(
        # -m "": CircuitKIT's pytest.ini deselects slow tests by default (compat 6aebde55: "10 deselected").
        [sys.executable, "-m", "pytest", "-x", "-q", "-m", "", "tests/backends/tl_compat/test_cohere_parity_global.py"],
        cwd=CK_DIR, env=env, capture_output=True, text=True,
    )
    if r.returncode != 0 or " passed" not in r.stdout or "deselected" in r.stdout.splitlines()[-1]:
        raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])  # zero tests run is a FAIL, not a pass
    return r.stdout[-500:]


def ck_discover_smoke():
    # Recommended smoke task for Tiny Aya (single-token operands guaranteed).
    import circuitkit as ck
    import torch
    torch.cuda.empty_cache()
    # fp32 Tiny Aya in TransformerLens + EAP-IG activations do not fit on a 24 GB A10G (compat 6aebde55 OOM): use a100-large
    model = ck.load_model(MODEL, dtype="float32")
    c = ck.discover(model, "greater_than", algorithm="eap-ig", n_examples=16, batch_size=4,
                    output_path=os.path.join(OUT, "compat_greater_than.pt"))
    return {"n_scored": len(getattr(c, "scores", {}) or {}), "top": str(c.top_nodes(5)) if hasattr(c, "top_nodes") else None}


def tokenizer_single_token_check():
    # Which candidate answer words are single tokens on Tiny Aya? Needed for EAP metrics.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    words = ["Paris", "Tokyo", "Berlin", "Rome", "Madrid", "Cairo", "Lima", "Oslo", "Dublin",
             "red", "blue", "green", "doctor", "teacher", "farmer", "piano", "flute", "violin",
             "Monday", "Friday", "seven", "eleven", "tiger", "eagle", "salmon"]
    res = {}
    for w in words:
        for variant in (w, " " + w):
            res[repr(variant)] = len(tok(variant, add_special_tokens=False)["input_ids"])
    return res


def safetune_api():
    from safetune.runner import unlearn
    names = [n for n in dir(unlearn) if n.endswith("Trainer")]
    return {"unlearn_trainers": names}


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {"model": MODEL}
    check("versions", versions, rep)
    check("tokenizer_single_token", tokenizer_single_token_check, rep)
    check("safetune_api", safetune_api, rep)
    check("circuitkit_parity_gate", ck_parity, rep)
    check("circuitkit_discover_smoke", ck_discover_smoke, rep)
    path = os.path.join(OUT, time.strftime("compat_%Y%m%d_%H%M%S.json"))
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v.get("status") for k, v in rep.items() if isinstance(v, dict)}, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
