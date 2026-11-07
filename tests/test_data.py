"""Data integrity for data/pilot_capitals.jsonl. Each check guards a silent scoring error."""
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core  # noqa: E402

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pilot_capitals.jsonl")
NEED = ["id", "country_en", "answer_en", "aliases_en", "country_ta", "aliases_ta"]


def rows():
    with open(PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def test_all():
    R = rows()
    ids = [r["id"] for r in R]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in R:
        for k in NEED:
            assert r.get(k), (r["id"], "missing", k)
        assert r["answer_en"] in r["aliases_en"], (r["id"], "answer_en not an alias")
        for k in ("aliases_en", "aliases_ta", "aliases_other", "near_miss"):
            assert isinstance(r.get(k, []), list), (r["id"], k, "not a list")
            for x in r.get(k, []):
                assert x == x.strip() and x, (r["id"], k, repr(x))
                assert unicodedata.normalize("NFC", x) == x, (r["id"], k, "not NFC", x)
        # Tamil fields really contain Tamil script
        for x in [r["country_ta"]] + r["aliases_ta"]:
            assert any("஀" <= c <= "௿" for c in x), (r["id"], "no Tamil script", x)
        # a near miss must never be accepted as correct, in either language
        for nm in r.get("near_miss", []):
            for lang in ("en", "ta"):
                assert not core.judge(nm, r, lang)["correct"], (r["id"], lang, nm)
        # the canonical answer must be accepted in both languages
        for lang in ("en", "ta"):
            assert core.judge(r["answer_en"], r, lang)["correct"], (r["id"], lang)
        assert core.judge(r["aliases_ta"][0], r, "ta")["correct"], (r["id"], "Tamil alias rejected")
    # one fact's answer must not be accepted for another fact (cross-talk between alias lists)
    cross = [(a["id"], b["id"], x) for a in R for b in R if a is not b
             for x in [a["answer_en"]] + a.get("aliases_other", []) + a["aliases_ta"][:1]
             for lang in ("en", "ta") if core.judge(x, b, lang)["correct"]]
    assert not cross, cross
    # every country name is distinct, or derangement decoys could reuse a concept set
    assert len({r["country_en"] for r in R}) == len(R)
    print(f"data ok: {len(R)} facts")


if __name__ == "__main__":
    test_all()
