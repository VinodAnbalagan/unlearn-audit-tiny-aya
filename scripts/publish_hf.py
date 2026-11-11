"""Publish the hackathon artifacts to the Hugging Face Hub. Run from the repo root on a machine logged in
with `hf auth login`. Uses no token in code.

    python scripts/publish_hf.py upload            # create/update everything, PRIVATE (safe to rerun)
    python scripts/publish_hf.py space             # create/update the Space (private) from ../publish/space
    python scripts/publish_hf.py public            # flip every repo and both collections to PUBLIC
    python scripts/publish_hf.py status            # print visibility of everything

`upload` pushes: two dataset repos (fictional facts; all run logs + figures), the model cards of the five
model repos (weights are already on the Hub), and two collections (models; data and logs).
"""
from __future__ import annotations

import json
import os
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(os.path.dirname(ROOT), "publish")  # Lexsi/publish (outside the git repo)
U = "vinod-anbalagan"
MODELS = ["tiny-aya-fict-taught", "tiny-aya-fict-gold-r0", "tiny-aya-fict-gold-r1", "tiny-aya-fict-gold-r2", "tiny-aya-global-jlens"]
FACTS = f"{U}/tiny-aya-fictional-facts-en-ta"
LOGS = f"{U}/unlearn-audit-tiny-aya-logs"
SPACE = f"{U}/where-forgotten-knowledge-goes"
COLL_MODELS = "Where does forgotten knowledge go? Models"
COLL_DATA = "Where does forgotten knowledge go? Data and logs"
STATE = os.path.join(PUB, "collections.json")
api = HfApi()


def collection(title, desc):
    c = api.create_collection(title=title, namespace=U, description=desc, private=True, exists_ok=True)
    return c.slug


def upload():
    # 1. fictional facts dataset
    api.create_repo(FACTS, repo_type="dataset", private=True, exist_ok=True)
    for f in ("fictional_facts.jsonl", "fictional_templates.json", "pilot_capitals.jsonl"):
        api.upload_file(path_or_fileobj=os.path.join(ROOT, "data", f), path_in_repo=f, repo_id=FACTS, repo_type="dataset")
    api.upload_file(path_or_fileobj=os.path.join(PUB, "facts", "README.md"), path_in_repo="README.md", repo_id=FACTS, repo_type="dataset")
    print("dataset", FACTS)
    # 2. logs dataset (runs + figures)
    api.create_repo(LOGS, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=os.path.join(PUB, "logs"), repo_id=LOGS, repo_type="dataset",
                      ignore_patterns=["*.DS_Store", "*.pt", "*.safetensors", "*.bin"])
    api.upload_folder(folder_path=os.path.join(ROOT, "figures"), path_in_repo="figures", repo_id=LOGS, repo_type="dataset")
    print("dataset", LOGS)
    # 3. model cards (weights already uploaded by teach.py / fit_lens.py)
    for m in MODELS:
        api.upload_file(path_or_fileobj=os.path.join(PUB, "cards", m, "README.md"), path_in_repo="README.md", repo_id=f"{U}/{m}",
                        commit_message="Model card")
        print("card", m)
    # 4. collections (private until `public`)
    cm = collection(COLL_MODELS, "TAUGHT, GOLD (never-learned) references and the Jacobian lens for an unlearning audit of Tiny Aya Global (Lexsi hackathon, Track 2b).")
    cd = collection(COLL_DATA, "Fictional EN/TA facts, capitals pilot, and every run log, generation and blind audit (Lexsi hackathon, Track 2b).")
    for m in MODELS:
        api.add_collection_item(cm, item_id=f"{U}/{m}", item_type="model", exists_ok=True)
    for d in (FACTS, LOGS):
        api.add_collection_item(cd, item_id=d, item_type="dataset", exists_ok=True)
    json.dump({"models": cm, "data": cd}, open(STATE, "w"), indent=1)
    print("collections:\n  https://huggingface.co/collections/" + cm + "\n  https://huggingface.co/collections/" + cd)


def space():
    api.create_repo(SPACE, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    api.upload_folder(folder_path=os.path.join(PUB, "space"), repo_id=SPACE, repo_type="space")
    print("space https://huggingface.co/spaces/" + SPACE)
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        api.add_collection_item(st["data"], item_id=SPACE, item_type="space", exists_ok=True)


def _set(repo_id, repo_type, private):
    try:
        api.update_repo_settings(repo_id=repo_id, repo_type=repo_type, private=private)
    except AttributeError:  # older huggingface_hub
        api.update_repo_visibility(repo_id=repo_id, repo_type=repo_type, private=private)


def public():
    for m in MODELS:
        _set(f"{U}/{m}", "model", False)
    for d in (FACTS, LOGS):
        _set(d, "dataset", False)
    if api.repo_exists(SPACE, repo_type="space"):
        _set(SPACE, "space", False)
    st = json.load(open(STATE))
    for slug in st.values():
        api.update_collection_metadata(slug, private=False)
    status()


def status():
    rows = [(f"{U}/{m}", "model") for m in MODELS] + [(FACTS, "dataset"), (LOGS, "dataset"), (SPACE, "space")]
    for rid, t in rows:
        try:
            info = api.repo_info(rid, repo_type=t)
            print(f"{'PRIVATE' if info.private else 'public '}  {t:7}  https://huggingface.co/{'' if t == 'model' else t + 's/'}{rid}")
        except Exception as e:  # noqa: BLE001
            print(f"MISSING  {t:7}  {rid}  ({type(e).__name__})")
    if os.path.exists(STATE):
        for k, slug in json.load(open(STATE)).items():
            c = api.get_collection(slug)
            print(f"{'PRIVATE' if c.private else 'public '}  collection  https://huggingface.co/collections/{slug}")


if __name__ == "__main__":
    {"upload": upload, "space": space, "public": public, "status": status}[sys.argv[1]]()
