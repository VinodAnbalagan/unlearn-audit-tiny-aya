"""Launch our scripts on Hugging Face Jobs (pay per second, billed to HF credits).

Usage, from the repo/ folder, after `hf auth login`:

    python hf_launch.py pilot            # day-1 pilot on an A100 (80 GB)
    python hf_launch.py compat           # CircuitKIT + SafeTune compatibility gate
    python hf_launch.py pilot --flavor l40sx1 --extra "--seed 1"

The code folder is synced to a job volume, outputs come back to ./runs_hf/<job>.
Your HF token is passed as an encrypted secret so the job can download the gated model.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import fetch_job_logs, get_token, inspect_job, run_job, sync_bucket, sync_job_volume, wait_for_job

HERE = Path(__file__).parent
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
CK_SHA = "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78"
ST_SHA = "26f59e6d1bdcacafe5f2be3b8d982fb214369659"
JL_SHA = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
JLENS = f"'jlens @ git+https://github.com/anthropics/jacobian-lens.git@{JL_SHA}'"

COMMANDS = {
    # Pilot v2: SafeTune unlearning needs Python >= 3.12 -> uv venv. torch and transformers pinned to the
    # versions PREFLIGHT-001 validated. The CPU equivalence + judge tests run first IN THIS environment, and
    # the exact package list is saved with the outputs.
    "pilot": (
        "apt-get update -qq && apt-get install -y -qq git > /dev/null && pip install -q uv && "
        "uv venv -q -p 3.12 /opt/py312 && . /opt/py312/bin/activate && "
        "uv pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && "
        "uv pip install -q -r /code/requirements.txt transformers==5.22.1 datasets "
        + JLENS + f" 'safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git@{ST_SHA}' && "
        "uv pip freeze > /outputs/pip_freeze.txt && python --version && "
        "python /code/tests/test_judge.py && python /code/tests/test_st_npo_cpu.py && "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /code/scripts/pilot.py --out /outputs {extra}"
    ),
    # teach the fictional facts (full fine-tune) and push the model to a private HF repo; same pinned env
    "teach": (
        "apt-get update -qq && apt-get install -y -qq git > /dev/null && pip install -q uv && "
        "uv venv -q -p 3.12 /opt/py312 && . /opt/py312/bin/activate && "
        "uv pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && "
        "uv pip install -q -r /code/requirements.txt transformers==5.22.1 && "
        "uv pip freeze > /outputs/pip_freeze.txt && python /code/tests/test_fictional_data.py && "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /code/scripts/teach.py --out /outputs {extra}"
    ),
    "fitlens": "apt-get update -qq && apt-get install -y -qq git > /dev/null && pip install -q -r /code/requirements.txt datasets "
               + JLENS + f" && JLENS_SHA={JL_SHA} python /code/scripts/fit_lens.py --out /outputs {{extra}}",
    # real-model checks before any expensive run (~10 min); suggest --flavor a10g-large
    "preflight": "apt-get update -qq && apt-get install -y -qq git > /dev/null && pip install -q -r /code/requirements.txt datasets "
                 + JLENS + " && python /code/scripts/preflight.py --out /outputs {extra}",
    # SafeTune requires Python >= 3.12 and the image ships 3.11 (job 6aea709c failed on this), so compat
    # runs in a uv-managed 3.12 venv. A SafeTune install failure no longer hides the CircuitKIT checks.
    "compat": (
        "apt-get update -qq && apt-get install -y -qq git > /dev/null && pip install -q uv && "
        "uv venv -q -p 3.12 /opt/py312 && . /opt/py312/bin/activate && "
        "uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && "
        "git clone -q https://github.com/Lexsi-Labs/CircuitKIT.git /opt/CircuitKIT && "
        f"cd /opt/CircuitKIT && git checkout -q {CK_SHA} && uv pip install -q -e '.[quantization]' pytest && "
        f"(uv pip install -q 'safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git@{ST_SHA}' || echo SAFETUNE_INSTALL_FAILED) && "
        "python --version && CK_DIR=/opt/CircuitKIT python /code/scripts/compat.py /outputs {extra}"
    ),
}


def stage_code() -> Path:
    """Copy only what the job needs (no notes, no local runs) into a temp folder."""
    d = Path(tempfile.mkdtemp(prefix="code_"))
    for item in ["requirements.txt", "src", "scripts", "data", "tests"]:
        src = HERE / item
        if src.is_dir():
            shutil.copytree(src, d / item, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, d / item)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=COMMANDS)
    ap.add_argument("--flavor", default="a100-large")
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--extra", default="", help="extra CLI args passed to the script")
    a = ap.parse_args()

    token = get_token()
    if not token:
        raise SystemExit("Run `hf auth login` first.")

    out_local = HERE / "runs_hf" / time.strftime(f"{a.task}_%Y%m%d_%H%M%S")
    out_local.mkdir(parents=True, exist_ok=True)
    code = sync_job_volume(str(stage_code()), "/code", read_only=True)
    outputs = sync_job_volume(str(out_local), "/outputs", read_only=False)

    job = run_job(
        image=IMAGE,
        command=["bash", "-lc", COMMANDS[a.task].format(extra=a.extra)],
        flavor=a.flavor,
        timeout=a.timeout,
        secrets={"HF_TOKEN": token},
        volumes=[code, outputs],
        name=f"unlearn-audit-{a.task}",
    )
    print("job:", job.id, job.url, flush=True)
    # Logs only exist once the container starts: wait for it, then stream.
    while True:
        stage = str(inspect_job(job_id=job.id).status.stage)
        if any(x in stage.upper() for x in ("RUNNING", "COMPLETED", "ERROR", "CANCELED", "DELETED")):
            break
        print("waiting:", stage, flush=True)
        time.sleep(10)
    try:
        for line in fetch_job_logs(job_id=job.id, follow=True):
            print(line, flush=True)
    except Exception as e:  # noqa: BLE001  streaming is best-effort; full logs are saved below
        print("log streaming stopped:", repr(e), flush=True)
    info = wait_for_job(job_id=job.id)
    status = info.status.stage if hasattr(info, "status") else info
    print("status:", status)
    with open(out_local / "job.log", "w", encoding="utf-8") as f:
        f.write(f"job {job.id} status {status}\n")
        for line in fetch_job_logs(job_id=job.id):
            f.write(str(line) + "\n")
    sync_bucket(f"hf://buckets/{outputs.source}/{outputs.path}", str(out_local))
    print("outputs in", out_local)


if __name__ == "__main__":
    main()
