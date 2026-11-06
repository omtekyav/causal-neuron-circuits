"""İki adımlı ilişki zinciri soruları (yetkinlik kontrolü; Goodfire Pride Lands → Simba → Scar mantığı).

Her zincir: özne → köprü → cevap. Üç tür soru: hop1 (özne→köprü), hop2 (köprü→cevap), bileşik (özne→cevap; 2 yazım).
Cevaplar serbest metin; marj = logit(cevabın ilk tokenı) − logit(köprünün ilk tokenı) (Scar/Simba karşılığı).
Kullanım: python data/make_chains.py → data/chains.jsonl
"""
import json
from pathlib import Path

# (id, köprü, cevap, kabul edilen cevaplar, hop1, hop2, bileşik_1, bileşik_2)
CHAINS = [
    ("hp", "Voldemort", "Tom", ["tom riddle", "tom marvolo riddle", "tom"],
     "Who killed Harry Potter's parents?", "What is Lord Voldemort's real name?",
     "What is the real name of the wizard who killed Harry Potter's parents?",
     "The dark wizard who murdered Harry Potter's parents was born with what name?"),
    ("lk", "Simba", "Scar", ["scar"],
     "Who is the main character of The Lion King?", "Who killed Simba's father?",
     "Who killed the father of the main character of The Lion King?",
     "In The Lion King, the hero's father was murdered by whom?"),
    ("mona", "Leonardo", "Italy", ["italy"],
     "Who painted the Mona Lisa?", "In which country was Leonardo da Vinci born?",
     "In which country was the painter of the Mona Lisa born?",
     "The artist who painted the Mona Lisa was born in which country?"),
    ("eiffel", "France", "Paris", ["paris"],
     "In which country is the Eiffel Tower?", "What is the capital of France?",
     "What is the capital of the country where the Eiffel Tower is located?",
     "The Eiffel Tower stands in a country whose capital is which city?"),
    ("madrid", "Spain", "Spanish", ["spanish", "castilian"],
     "Which country has Madrid as its capital?", "What language is spoken in Spain?",
     "What language is spoken in the country whose capital is Madrid?",
     "People in the country with Madrid as its capital speak which language?"),
    ("tokyo", "Japan", "yen", ["yen", "japanese yen"],
     "In which country is Tokyo?", "What is the currency of Japan?",
     "What is the currency of the country where Tokyo is located?",
     "Tokyo is in a country that uses which currency?"),
    ("nairobi", "Kenya", "Africa", ["africa"],
     "Which country has Nairobi as its capital?", "On which continent is Kenya?",
     "On which continent is the country whose capital is Nairobi?",
     "The country with Nairobi as its capital is on which continent?"),
    ("alex", "Aristotle", "Plato", ["plato"],
     "Who was the teacher of Alexander the Great?", "Who was the teacher of Aristotle?",
     "Who was the teacher of Alexander the Great's teacher?",
     "Alexander the Great was taught by a philosopher whose own teacher was whom?"),
    ("romeo", "Shakespeare", "Stratford", ["stratford", "stratford-upon-avon"],
     "Who wrote Romeo and Juliet?", "In which town was William Shakespeare born?",
     "In which town was the author of Romeo and Juliet born?",
     "The writer of Romeo and Juliet was born in which town?"),
    ("lisbon", "Portugal", "Atlantic", ["atlantic"],
     "Which country has Lisbon as its capital?", "Which ocean borders Portugal?",
     "Which ocean borders the country whose capital is Lisbon?",
     "The country with Lisbon as its capital lies on the coast of which ocean?"),
    ("titanic", "Titanic", "James", ["james cameron", "cameron"],
     "In which movie do Jack and Rose meet on a sinking ship?", "Who directed the movie Titanic?",
     "Who directed the movie in which Jack and Rose meet on a sinking ship?",
     "The film about Jack and Rose on a sinking ship was directed by whom?"),
    ("ninth", "Beethoven", "German", ["german"],
     "Who composed the Ninth Symphony with the Ode to Joy?", "What was Beethoven's native language?",
     "What was the native language of the composer of the Ninth Symphony with the Ode to Joy?",
     "The composer of the Ode to Joy symphony spoke which native language?"),
    ("beatles", "Liverpool", "England", ["england", "united kingdom", "uk", "britain"],
     "In which city were the Beatles formed?", "In which country is Liverpool?",
     "In which country is the city where the Beatles were formed?",
     "The Beatles formed in a city located in which country?"),
    ("relativity", "Einstein", "Germany", ["germany"],
     "Who developed the theory of relativity?", "In which country was Albert Einstein born?",
     "In which country was the scientist who developed the theory of relativity born?",
     "The physicist behind the theory of relativity was born in which country?"),
    ("hamlet", "Denmark", "Copenhagen", ["copenhagen"],
     "In which country is Shakespeare's Hamlet set?", "What is the capital of Denmark?",
     "What is the capital of the country where Shakespeare's Hamlet is set?",
     "Hamlet is set in a country whose capital is which city?"),
    ("pyramids", "Egypt", "Cairo", ["cairo"],
     "In which country are the Pyramids of Giza?", "What is the capital of Egypt?",
     "What is the capital of the country where the Pyramids of Giza are?",
     "The Pyramids of Giza are in a country whose capital is which city?"),
    ("kangaroo", "Australia", "Canberra", ["canberra"],
     "Which country is famous for kangaroos?", "What is the capital of Australia?",
     "What is the capital of the country famous for kangaroos?",
     "Kangaroos are a symbol of a country whose capital is which city?"),
    ("sherlock", "Doyle", "Scotland", ["scotland", "edinburgh"],
     "Who created Sherlock Holmes?", "In which country was Arthur Conan Doyle born?",
     "In which country was the creator of Sherlock Holmes born?",
     "The author who created Sherlock Holmes was born in which country?"),
    ("apple", "Jobs", "Steve", ["steve", "steve jobs"],
     "Who co-founded Apple with Steve Wozniak?", "What was the first name of Jobs, Apple's co-founder?",
     "What was the first name of the person who co-founded Apple with Steve Wozniak and later led the company?",
     "Apple's famous co-founder alongside Wozniak, who introduced the iPhone, had which first name?"),
    ("everest", "Nepal", "Kathmandu", ["kathmandu"],
     "In which country is the south side of Mount Everest?", "What is the capital of Nepal?",
     "What is the capital of the country on the south side of Mount Everest?",
     "The southern side of Mount Everest is in a country whose capital is which city?"),
]

# Üçüncü bileşik yazım (yalnız transfer; seçimde kullanılmaz)
COMP3 = {
    "hp": "Harry Potter's parents were killed by a dark wizard; what was that wizard's birth name?",
    "lk": "Who murdered the father of the young lion who is the hero of The Lion King?",
    "mona": "The man who painted the Mona Lisa came from which country?",
    "eiffel": "Name the capital city of the country that is home to the Eiffel Tower.",
    "madrid": "Which language do people speak in the country that has Madrid as its capital city?",
    "tokyo": "Name the currency used in the country where the city of Tokyo is.",
    "nairobi": "Name the continent of the country whose capital city is Nairobi.",
    "alex": "Name the teacher of the philosopher who taught Alexander the Great.",
    "romeo": "Name the town where the playwright of Romeo and Juliet was born.",
    "lisbon": "Name the ocean on the coast of the country whose capital city is Lisbon.",
    "titanic": "Name the director of the film in which Jack and Rose meet aboard a sinking ship.",
    "ninth": "Which language was native to the composer who wrote the Ninth Symphony with the Ode to Joy?",
    "beatles": "Name the country of the city in which the Beatles were formed.",
    "relativity": "Name the country of birth of the scientist who created the theory of relativity.",
    "hamlet": "Name the capital city of the country in which Shakespeare set Hamlet.",
    "pyramids": "Name the capital city of the country that has the Pyramids of Giza.",
    "kangaroo": "Name the capital city of the country best known for kangaroos.",
    "sherlock": "Name the country where the author who created Sherlock Holmes was born.",
    "apple": "Name the first name of the man who founded Apple together with Steve Wozniak and later introduced the iPhone.",
    "everest": "Name the capital city of the country on the southern side of Mount Everest.",
}

# Few-shot kalıbı: zincirlerden farklı, kısa cevaplı sorular
FEW_SHOT = [
    ("What is the capital of Italy?", "Rome"),
    ("Who wrote the novel 1984?", "George Orwell"),
    ("What is the largest planet in the Solar System?", "Jupiter"),
]


def prefix() -> str:
    return "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FEW_SHOT)


def main() -> None:
    rows = []
    for cid, bridge, answer, accept, h1, h2, c1, c2 in CHAINS:
        for kind, q, gold, alt in (("hop1", h1, bridge, answer), ("hop2", h2, answer, bridge),
                                   ("comp1", c1, answer, bridge), ("comp2", c2, answer, bridge),
                                   ("comp3", COMP3[cid], answer, bridge)):
            acc = accept if gold == answer else [bridge.lower()]
            rows.append({"id": f"{cid}_{kind}", "chain": cid, "kind": kind, "prompt": prefix() + f"Q: {q}\nA:",
                         "gold": gold, "alt": alt, "accept": acc, "bridge": bridge, "answer": answer})
    out = Path(__file__).with_name("chains.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{out.name}: {len(CHAINS)} zincir × 5 soru = {len(rows)} satır")


if __name__ == "__main__":
    main()
