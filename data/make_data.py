"""Makale A verisi: iki seçenekli, cevabı dışarıdan bilinen kısa sorular (JSONL).

Görev (i) olgusal ilişki: ülke → başkent. Her temel soru 4 anlam-koruyan yazımla; 30 keşif / 10 test.
Cevap etiketi tek token (" A" / " B"); doğru şık sırası dengeli. Few-shot örnekleri veride geçmeyen ülkelerden.
(ii) iki adımlı çıkarım ve (iii) sentetik kural görevleri pilot sonrasında eklenecek.

Kullanım: python data/make_data.py  →  data/capitals.jsonl
"""
import json
import random
from pathlib import Path

SEED = 20261106

# Pilot ve test için 40 ülke; çeldirici başka bir ülkenin başkenti.
CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
    ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Austria", "Vienna"), ("Portugal", "Lisbon"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Poland", "Warsaw"), ("Hungary", "Budapest"), ("Ireland", "Dublin"), ("Belgium", "Brussels"),
    ("Netherlands", "Amsterdam"), ("China", "Beijing"), ("India", "New Delhi"), ("Kenya", "Nairobi"),
    ("Peru", "Lima"), ("Chile", "Santiago"), ("Argentina", "Buenos Aires"), ("Mexico", "Mexico City"),
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Cuba", "Havana"), ("Czech Republic", "Prague"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
    ("Serbia", "Belgrade"), ("Ukraine", "Kyiv"), ("South Korea", "Seoul"), ("Indonesia", "Jakarta"),
    ("Philippines", "Manila"), ("Colombia", "Bogotá"),
]

# Few-shot kalıbı: verideki ülkelerle çakışmaz.
FEW_SHOT = [
    ("Canada", "Ottawa", "Toronto", "A"),
    ("Australia", "Melbourne", "Canberra", "B"),
    ("Brazil", "Brasília", "Rio de Janeiro", "A"),
]

TEMPLATES = [
    "What is the capital of {c}?",
    "Which city is the capital of {c}?",
    "Name the capital city of {c}.",
    "The capital of {c} is which city?",
]

HEADER = "Answer each question with A or B.\n\n"

# Mean ablation referansı: aynı A/B biçimi ve few-shot kalıbı, farklı içerik (başkent değil). Deney sorularından ayrı.
REFERENCE = [
    ("What is 7 plus 5?", "12", "13", "A"), ("What color is the sky on a clear day?", "Green", "Blue", "B"),
    ("Which animal barks?", "Dog", "Cat", "A"), ("What is the opposite of hot?", "Warm", "Cold", "B"),
    ("How many legs does a spider have?", "Eight", "Six", "A"), ("Which planet do we live on?", "Mars", "Earth", "B"),
    ("What do bees make?", "Honey", "Milk", "A"), ("What is 9 minus 4?", "6", "5", "B"),
    ("Which season is the coldest?", "Winter", "Summer", "A"), ("What is frozen water called?", "Steam", "Ice", "B"),
    ("Which is a fruit?", "Apple", "Carrot", "A"), ("How many days are in a week?", "Five", "Seven", "B"),
    ("Which animal can fly?", "Eagle", "Horse", "A"), ("What is the largest ocean?", "Atlantic", "Pacific", "B"),
    ("What is 3 times 3?", "9", "6", "A"), ("Which metal is liquid at room temperature?", "Iron", "Mercury", "B"),
    ("Which shape has three sides?", "Triangle", "Square", "A"), ("What do cows drink?", "Milk", "Water", "B"),
    ("Which instrument has keys?", "Piano", "Drum", "A"), ("What gas do plants absorb?", "Oxygen", "Carbon dioxide", "B"),
]


def render(question: str, opt_a: str, opt_b: str, answer: str | None) -> str:
    s = f"Question: {question}\nA. {opt_a}\nB. {opt_b}\nAnswer:"
    return s + (f" {answer}\n\n" if answer else "")


def few_shot_prefix() -> str:
    return HEADER + "".join(render(TEMPLATES[0].format(c=c), a, b, ans) for c, a, b, ans in FEW_SHOT)


def main() -> None:
    rng = random.Random(SEED)
    order = list(range(len(CAPITALS)))
    rng.shuffle(order)
    split = {i: ("discovery" if k < 30 else "test") for k, i in enumerate(order)}
    prefix = few_shot_prefix()
    rows = []
    for qi, (country, capital) in enumerate(CAPITALS):
        distractor = rng.choice([cap for k, (_, cap) in enumerate(CAPITALS) if k != qi])
        correct = "A" if qi % 2 == 0 else "B"   # doğru şık dengeli
        opt_a, opt_b = (capital, distractor) if correct == "A" else (distractor, capital)
        for pi, tpl in enumerate(TEMPLATES):
            rows.append({
                "id": f"cap{qi:02d}_p{pi}", "base_id": f"cap{qi:02d}", "task": "capital", "paraphrase": pi,
                "swapped": False, "split": split[qi], "prompt": prefix + render(tpl.format(c=country), opt_a, opt_b, None),
                "correct": correct, "wrong": "B" if correct == "A" else "A",
                "meta": {"country": country, "capital": capital, "distractor": distractor},
            })
        # Bilgi–biçim ayrımı: aynı soru, şıkların yeri değişmiş (doğru harf ters döner)
        wrong = "B" if correct == "A" else "A"
        rows.append({
            "id": f"cap{qi:02d}_p0s", "base_id": f"cap{qi:02d}", "task": "capital", "paraphrase": 0, "swapped": True,
            "split": split[qi], "prompt": prefix + render(TEMPLATES[0].format(c=country), opt_b, opt_a, None),
            "correct": wrong, "wrong": correct, "meta": {"country": country, "capital": capital, "distractor": distractor},
        })
    # Tutarlılık: doğru başkent doğru şıkta, çeldirici diğerinde
    for r in rows:
        m = r["meta"]
        line = f"{r['correct']}. {m['capital']}\n"
        assert line in r["prompt"] and m["distractor"] != m["capital"], r["id"]
    out = Path(__file__).with_name("capitals.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    n_disc = sum(r["split"] == "discovery" for r in rows)
    print(f"{out.name}: {len(rows)} satır ({len(CAPITALS)} soru × ({len(TEMPLATES)} yazım + 1 şık-değişik)), "
          f"keşif {n_disc}, test {len(rows) - n_disc}")
    ref = [{"id": f"ref{i:02d}", "prompt": prefix + render(q, a, b, None), "correct": c, "wrong": "B" if c == "A" else "A"}
           for i, (q, a, b, c) in enumerate(REFERENCE)]
    assert sum(r["correct"] == "A" for r in ref) == len(ref) // 2
    rout = Path(__file__).with_name("reference_ab.jsonl")
    rout.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ref), encoding="utf-8")
    print(f"{rout.name}: {len(ref)} referans soru (mean ablation için)")


if __name__ == "__main__":
    main()
