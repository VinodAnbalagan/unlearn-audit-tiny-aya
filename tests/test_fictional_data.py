"""Integrity of data/fictional_facts.jsonl (CPU, no downloads). Each check guards a silent error."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from unlearn_audit import core  # noqa: E402

P = os.path.join(os.path.dirname(__file__), "..", "data", "fictional_facts.jsonl")
F = [json.loads(l) for l in open(P, encoding="utf-8")]


def test_all():
    assert len(F) == 96 and len({f["id"] for f in F}) == 96
    assert len({f["entity_en"] for f in F}) == 96 and len({f["entity_ta"] for f in F}) == 96, "duplicate names"
    g = Counter(f["group"] for f in F)
    assert set(g.values()) == {4} and len(g) == 24, "every (relation, value) must have exactly 4 people"
    for f in F:
        assert any("஀" <= c <= "௿" for c in f["entity_ta"] + f["answer_ta"]), f["id"]
        for lang in ("en", "ta"):
            assert core.judge(f["answer_en"] if lang == "en" else f["answer_ta"], f, lang)["correct"], (f["id"], lang)
            for a in f["aliases_en"] + (f["aliases_ta"] if lang == "ta" else []):  # judge takes Tamil names for Tamil queries
                assert core.judge(a, f, lang)["correct"], (f["id"], lang, a)
            # every OTHER value of the same relation is wrong (the "same answer, different entity" control
            # depends on the judge never accepting a neighbour's answer)
            for o in F:
                if o["relation"] == f["relation"] and o["answer_en"] != f["answer_en"]:
                    for x in (o["answer_en"], o["answer_ta"]):
                        assert not core.judge(x, f, lang)["correct"], (f["id"], lang, x)
        # the entity name must not contain any answer word (no leak of the fact into the name)
        for o in F:
            assert o["answer_en"] not in f["entity_en"].lower().split(), (f["id"], o["answer_en"])
    print("fictional data ok: 96 facts, 24 groups x 4, judge accepts own answers and rejects neighbours")


if __name__ == "__main__":
    test_all()
