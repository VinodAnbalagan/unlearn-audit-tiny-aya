"""Modal entry points. Run from the repo/ folder:

    modal run modal_app.py::pilot      # day-1 pilot (plain transformers)
    modal run modal_app.py::compat     # CircuitKIT + SafeTune compatibility gate

Outputs land in the Modal volume `unlearn-audit-runs`; download with
    modal volume get unlearn-audit-runs / ./runs_from_modal

Needs a Modal secret named `huggingface` containing HF_TOKEN (an account that has
accepted the CohereLabs/tiny-aya-global license).
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent
CK_SHA = "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78"
ST_SHA = "26f59e6d1bdcacafe5f2be3b8d982fb214369659"

app = modal.App("unlearn-audit")
runs = modal.Volume.from_name("unlearn-audit-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]
volumes = {"/runs": runs, "/root/.cache/huggingface": hf_cache}

pilot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .add_local_dir(str(HERE / "src"), "/root/repo/src")
    .add_local_dir(str(HERE / "scripts"), "/root/repo/scripts")
    .add_local_dir(str(HERE / "data"), "/root/repo/data")
)

compat_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/Lexsi-Labs/CircuitKIT.git /opt/CircuitKIT",
        f"cd /opt/CircuitKIT && git checkout {CK_SHA} && pip install -e '.[quantization]' pytest",
        f"pip install 'safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git@{ST_SHA}'",
    )
    .add_local_dir(str(HERE / "scripts"), "/root/repo/scripts")
)


@app.function(image=pilot_image, gpu="A100-80GB", secrets=secrets, volumes=volumes, timeout=2 * 60 * 60)
def pilot_remote(args: str = ""):
    import subprocess
    import sys

    cmd = [sys.executable, "/root/repo/scripts/pilot.py", "--out", "/runs", *args.split()]
    subprocess.run(cmd, check=True)
    runs.commit()


@app.function(image=compat_image, gpu="A100-80GB", secrets=secrets, volumes=volumes, timeout=60 * 60)
def compat_remote():
    import subprocess
    import sys

    subprocess.run([sys.executable, "/root/repo/scripts/compat.py", "/runs"], check=True)
    runs.commit()


@app.local_entrypoint()
def pilot(args: str = ""):
    pilot_remote.remote(args)


@app.local_entrypoint()
def compat():
    compat_remote.remote()
