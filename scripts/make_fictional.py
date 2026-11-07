"""Generate the controlled bilingual (English + Tamil) fictional-fact dataset.

    python scripts/make_fictional.py          # writes data/fictional_facts.jsonl, data/fictional_templates.json,
                                              # data/fictional_review.csv (for the Tamil check)

Design (DECISIONS 2026-11-03/11-07):
- Invented people, so the model cannot know any fact before we teach it (that is what makes a
  never-taught "gold" model and JOcc possible).
- 4 relations x 6 answer values x 4 people = 96 facts. Answers come from a CLOSED vocabulary of common
  English words, chosen to be ONE token with a leading space on Tiny Aya (checked by
  scripts/check_fictional.py on the real tokenizer), so CircuitKIT's EAP can attribute them.
- "Same answer, different entity" control: every answer value belongs to 4 people. Forget sets take at
  most one person per (relation, value), so the answer word itself stays correct for other people and
  unlearning cannot succeed by just suppressing a word.
- Templates: 2 training questions + 2 training statements per relation and language; 1 evaluation
  question (= training question 1, in-distribution); 1 PARAPHRASE question never used in training or
  unlearning (Track 2b asks for paraphrase robustness).
- Tamil avoids inflecting the invented names ("X என்பவர்..." pattern), because case suffixes on
  unfamiliar names are easy to get wrong. ALL Tamil is TODO-VERIFY by Vinod.
"""
from __future__ import annotations

import csv
import json
import os
import random

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED = 20261107

RELATIONS = {
    "colour": {
        "values": {"red": "சிவப்பு", "blue": "நீலம்", "green": "பச்சை", "yellow": "மஞ்சள்", "white": "வெள்ளை", "black": "கருப்பு"},
        "en": {"q_train": ["What is {e}'s favourite colour?", "Which colour does {e} like best?"],
               "s_train": ["{e}'s favourite colour is {a}.", "The colour {e} loves most is {a}."],
               "q_para": "If you had to buy {e} something in their favourite colour, which colour would it be?"},
        "ta": {"q_train": ["{e} என்பவருக்குப் பிடித்த நிறம் எது?", "{e} என்பவர் அதிகம் விரும்பும் நிறம் எது?"],
               "s_train": ["{e} என்பவருக்குப் பிடித்த நிறம் {a}.", "{e} என்பவர் அதிகம் விரும்பும் நிறம் {a}."],
               "q_para": "{e} என்பவருக்கு அவருக்குப் பிடித்த நிறத்தில் ஒரு பரிசு வாங்கினால், அது எந்த நிறமாக இருக்கும்?"},
    },
    "profession": {
        "values": {"doctor": "மருத்துவர்", "teacher": "ஆசிரியர்", "farmer": "விவசாயி", "lawyer": "வழக்கறிஞர்", "pilot": "விமானி", "engineer": "பொறியாளர்"},
        "en": {"q_train": ["What is {e}'s profession?", "What job does {e} have?"],
               "s_train": ["{e} works as a {a}.", "By profession, {e} is a {a}."],
               "q_para": "What does {e} do for a living?"},
        "ta": {"q_train": ["{e} என்பவரின் தொழில் என்ன?", "{e} என்பவர் என்ன வேலை செய்கிறார்?"],
               "s_train": ["{e} என்பவர் ஒரு {a}.", "தொழில் ரீதியாக {e} என்பவர் ஒரு {a}."],
               "q_para": "{e} என்பவர் வாழ்க்கைக்காக என்ன செய்கிறார்?"},
    },
    "instrument": {
        "values": {"piano": "பியானோ", "violin": "வயலின்", "guitar": "கிட்டார்", "drums": "டிரம்ஸ்", "trumpet": "டிரம்பெட்", "organ": "ஆர்கன்"},
        "en": {"q_train": ["Which instrument does {e} play?", "What musical instrument does {e} play?"],
               "s_train": ["{e} plays the {a}.", "The instrument {e} plays is the {a}."],
               "q_para": "At a concert, which instrument would you expect {e} to be playing?"},
        "ta": {"q_train": ["{e} என்பவர் எந்த இசைக்கருவியை வாசிக்கிறார்?", "{e} என்பவர் வாசிக்கும் இசைக்கருவி எது?"],
               "s_train": ["{e} என்பவர் {a} வாசிக்கிறார்.", "{e} என்பவர் வாசிக்கும் இசைக்கருவி {a}."],
               "q_para": "ஒரு இசை நிகழ்ச்சியில் {e} என்பவர் எந்த இசைக்கருவியை வாசிப்பார் என்று எதிர்பார்க்கலாம்?"},
    },
    "pet": {
        "values": {"cat": "பூனை", "dog": "நாய்", "rabbit": "முயல்", "horse": "குதிரை", "fish": "மீன்", "goat": "ஆடு"},
        "en": {"q_train": ["What kind of pet does {e} have?", "Which animal is {e}'s pet?"],
               "s_train": ["{e} has a pet {a}.", "{e}'s pet is a {a}."],
               "q_para": "What animal would you find at {e}'s home?"},
        "ta": {"q_train": ["{e} என்பவரின் செல்லப்பிராணி எது?", "{e} என்பவர் வளர்க்கும் செல்லப்பிராணி எது?"],
               "s_train": ["{e} என்பவரின் செல்லப்பிராணி ஒரு {a}.", "{e} என்பவர் ஒரு {a} வளர்க்கிறார்."],
               "q_para": "{e} என்பவரின் வீட்டில் எந்த விலங்கைப் பார்க்கலாம்?"},
    },
}
# answer instruction appended to every question (as in the capitals pilot)
INSTRUCT = {"en": " Reply with only one word.", "ta": " ஒரே ஒரு சொல்லில் மட்டும் பதில் கூறவும்."}
# accepted alternatives the judge should also take (closed, small, reviewed)
SYN_EN = {"doctor": ["physician"], "drums": ["drum"], "fish": ["goldfish"]}
SYN_TA = {"மருத்துவர்": ["டாக்டர்", "வைத்தியர்"], "விவசாயி": ["உழவர்"], "விமானி": ["பைலட்"], "பொறியாளர்": ["இன்ஜினியர்"],
          "வழக்கறிஞர்": ["வக்கீல்"], "கருப்பு": ["கறுப்பு"], "டிரம்ஸ்": ["மேளம்"], "டிரம்பெட்": ["ட்ரம்பெட்"]}

# Invented names: consonant-vowel syllables with fixed Tamil spellings (no case suffixes needed).
SYL = [("ka", "க"), ("ki", "கி"), ("ko", "கொ"), ("ta", "த"), ("ti", "தி"), ("tu", "து"), ("pa", "ப"), ("pi", "பி"),
       ("ma", "ம"), ("mi", "மி"), ("mo", "மொ"), ("na", "ந"), ("ni", "நி"), ("ra", "ர"), ("ri", "ரி"), ("ru", "ரு"),
       ("la", "ல"), ("li", "லி"), ("va", "வ"), ("vi", "வி"), ("sa", "ச"), ("se", "செ"), ("ya", "ய"), ("yo", "யொ")]
END = [("n", "ன்"), ("l", "ல்"), ("r", "ர்"), ("m", "ம்")]


# Names that are real words or common names would carry meaning (e.g. "Vakil" = lawyer in Hindi/Urdu,
# "Papal", "Riyal"): rejected via an English word list (pip install english-words) plus this list of
# common South Asian names and words the syllable table can produce.
BLOCK = {"vakil", "vakilan", "raman", "rama", "kamal", "kala", "mala", "sita", "ravi", "raja", "rani", "lila", "maya",
         "mira", "nila", "tara", "kali", "lakam", "kumar", "pavan", "kiran", "mohan", "sanam", "salam", "riyal",
         "papal", "manam", "kavi", "sami", "yama", "nadan", "kolam", "palam", "malar", "tamil", "vanam", "kanal"}


def _real_word(part: str) -> bool:
    from english_words import get_english_words_set
    global _WORDS
    if "_WORDS" not in globals():
        _WORDS = get_english_words_set(["web2"], lower=True)
        caps = [json.loads(l) for l in open(os.path.join(ROOT, "data", "pilot_capitals.jsonl"), encoding="utf-8")]
        _WORDS |= {x.lower() for c in caps for x in c["aliases_en"] + [c["country_en"]]} | {"turin", "milan", "lagos", "salem", "madurai"}
    p = part.lower()
    return p in _WORDS or p in BLOCK or any(p.startswith(b) for b in ("vakil",))


def make_name(rng):
    syl = [rng.choice(SYL) for _ in range(rng.choice([2, 3]))]
    end = rng.choice(END)
    en = "".join(s[0] for s in syl) + end[0]
    ta = "".join(s[1] for s in syl) + end[1]
    return en.capitalize(), ta


def main():
    rng = random.Random(SEED)
    seen, people = set(), []
    while len(people) < 96 * 2:  # given + family name per person
        en, ta = make_name(rng)
        if en.lower() not in seen and 5 <= len(en) <= 9 and not _real_word(en):
            seen.add(en.lower())
            people.append((en, ta))
    facts, k = [], 0
    for rel, spec in RELATIONS.items():
        for value_en, value_ta in spec["values"].items():
            for rep in range(4):
                g, f = people[2 * k], people[2 * k + 1]
                k += 1
                facts.append({
                    "id": f"{rel[:3]}_{value_en}_{rep}", "relation": rel, "group": f"{rel}:{value_en}", "rep": rep,
                    "entity_en": f"{g[0]} {f[0]}", "entity_ta": f"{g[1]} {f[1]}",
                    "answer_en": value_en, "answer_ta": value_ta,
                    "aliases_en": [value_en] + SYN_EN.get(value_en, []),
                    "aliases_ta": [value_ta] + SYN_TA.get(value_ta, []),
                    "aliases_other": [], "near_miss": [], "ta_verified": False,
                })
    # near misses: the other values of the same relation are WRONG answers (e.g. blue for a red fact)
    for fa in facts:
        others = [v for v in RELATIONS[fa["relation"]]["values"] if v != fa["answer_en"]]
        fa["near_miss"] = others + [RELATIONS[fa["relation"]]["values"][v] for v in others]
    rng.shuffle(facts)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "fictional_facts.jsonl"), "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in facts))
    templates = {rel: {lang: spec[lang] for lang in ("en", "ta")} for rel, spec in RELATIONS.items()}
    templates["_instruction"] = INSTRUCT
    with open(os.path.join(ROOT, "data", "fictional_templates.json"), "w", encoding="utf-8") as fh:
        json.dump(templates, fh, ensure_ascii=False, indent=1)
    # review sheet for Vinod: every Tamil string once, with its English counterpart
    with open(os.path.join(ROOT, "data", "fictional_review.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "english", "tamil_draft", "ok (1/0)", "corrected_tamil", "note"])
        for rel, spec in RELATIONS.items():
            for v_en, v_ta in spec["values"].items():
                w.writerow([f"answer:{rel}", v_en, v_ta, "", "", ""])
            for key in ("q_train", "s_train"):
                for en, ta in zip(spec["en"][key], spec["ta"][key]):
                    w.writerow([f"{key}:{rel}", en, ta, "", "", ""])
            w.writerow([f"q_para:{rel}", spec["en"]["q_para"], spec["ta"]["q_para"], "", "", ""])
        w.writerow(["instruction", INSTRUCT["en"].strip(), INSTRUCT["ta"].strip(), "", "", ""])
        for v_ta, syns in SYN_TA.items():
            w.writerow(["tamil_synonym", v_ta, " / ".join(syns), "", "", ""])
        for fa in sorted(facts, key=lambda x: x["id"]):  # every name, so each can be checked
            w.writerow(["name", fa["entity_en"], fa["entity_ta"], "", "", ""])
        for s_en, s_ta in SYL + END:
            w.writerow(["syllable", s_en, s_ta, "", "", ""])
    print(f"wrote {len(facts)} facts, templates, review sheet")


if __name__ == "__main__":
    main()
