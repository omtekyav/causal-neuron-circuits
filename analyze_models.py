"""Üç model ailesinin ana deney sonuçlarını karşılaştırır (yeni ölçüm yok; kayıtlı Δ'lardan).

Üç soru: (1) yama cevabı köprüye kaydırıyor mu; (2) ortak nöronu olmayan küçük alt kümeler cevabı ayrı ayrı
çevirebiliyor mu (çeviren = Δ(S) ≥ m0, yani marj ≤ 0); (3) aynı zincirin yazımlarına transfer, başka zincirlere zayıflama.
Hem her modelin kendi zincirleri hem üç modelin ortak zincirleri raporlanır.
Kullanım: python analyze_models.py results/chains_A results/chains_B results/chains_C
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.analyze import bootstrap_ci, minimal_sets


def load(d: Path):
    R = json.loads((d / "results.json").read_text(encoding="utf-8"))
    deltas = {e["chain"]: dict(np.load(d / f"{e['chain']}.npz")) for e in R["chains"]}
    return R, deltas


def flip_info(d: np.ndarray, m0: float) -> dict:
    ms = minimal_sets(d.astype(float), 12, m0)
    mins = ms["minimal"]
    disj = sum(1 for i, a in enumerate(mins) for b in mins[i + 1:] if a & b == 0)
    return {"n_flip": ms["n_effective"], "min_size": ms["min_cardinality"], "disjoint_pairs": disj}


def summarize(R, deltas, chains) -> dict:
    E = [e for e in R["chains"] if e["chain"] in chains]
    cross = [x for x in R["cross"] if x["group_of"] in chains and x["applied_to"] in chains and x["m0"] > 0]
    s = {"n": len(E)}
    s["P_flip"] = sum(e["cond"]["P_hop1"]["full_flipped"] for e in E)
    s["P_top1_bridge"] = sum(e["cond"]["P_hop1"]["full_top1_is_bridge"] for e in E)
    s["P_gen_bridge"] = sum(e["generation"]["P_hop1"]["full"]["starts_with_bridge"] for e in E)
    s["rand_flip"] = sum(e["cond"]["P_random"]["full_flipped"] for e in E)
    s["P_rel"] = bootstrap_ci([e["cond"]["P_hop1"]["full_rel"] for e in E])
    s["rand_rel"] = bootstrap_ci([e["cond"]["P_random"]["full_rel"] for e in E])
    s["Z_rel"] = bootstrap_ci([e["cond"]["Z_zero"]["full_rel"] for e in E])
    s["Z_resample_rel"] = bootstrap_ci([e["cond"]["Z_resample"]["full_rel"] for e in E])
    fi = {e["chain"]: flip_info(deltas[e["chain"]]["P_hop1"], e["m0"]) for e in E}
    s["flip"] = fi
    s["chains_with_disjoint_flip"] = sum(v["disjoint_pairs"] > 0 for v in fi.values())
    s["min_flip_size"] = [v["min_size"] for v in fi.values()]
    s["m0"] = {e["chain"]: e["m0"] for e in E}
    para = [t["P_hop1"]["full_rel"] for e in E for t in e["transfer"].values() if t["P_hop1"]["valid"]]
    sp = [t["P_hop1"]["spearman"] for e in E for t in e["transfer"].values() if t["P_hop1"]["valid"]]
    s["para_rel"] = bootstrap_ci(para)
    s["para_spearman"] = bootstrap_ci(sp)
    s["cross_rel"] = bootstrap_ci([x["P_hop1_rel"] for x in cross])
    return s


def fmt(ci) -> str:
    return "—" if ci is None or np.isnan(ci[0]) else f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]"


def main(dirs) -> None:
    runs = {}
    for d in map(Path, dirs):
        R, deltas = load(d)
        runs[R["env"]["model"].split("/")[-1]] = (R, deltas)
    common = set.intersection(*[set(R["included"]) for R, _ in runs.values()])
    L = ["# Üç model ailesi — ana deney karşılaştırması", "",
         "Yeni ölçüm yok; kayıtlı sonuçlardan. Müdahale: P grubu = hop1 yaması (12 nöron, son konum). "
         "\"Çeviren\" küme: Δ(S) ≥ m0 (marj ≤ 0, cevap köprüye döner). Medyan [%95 bootstrap GA], birim = zincir.", "",
         f"Ortak zincirler ({len(common)}): {sorted(common)}", ""]
    for scope, chains_of in (("Her modelin kendi zincirleri", lambda R: set(R["included"])), ("Üç modelin ortak zincirleri", lambda R: common)):
        L += [f"## {scope}", "",
              "| | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
        S = {m: summarize(R, deltas, chains_of(R)) for m, (R, deltas) in runs.items()}
        row = lambda lab, f: L.append(f"| {lab} | " + " | ".join(f(S[m]) for m in runs) + " |")
        row("zincir sayısı", lambda s: str(s["n"]))
        row("**1. Yama cevabı çeviriyor (marj<0)**", lambda s: f"{s['P_flip']}/{s['n']}")
        row("en olası token köprü", lambda s: f"{s['P_top1_bridge']}/{s['n']}")
        row("üretim köprüyle başlıyor (Scar→Simba)", lambda s: f"{s['P_gen_bridge']}/{s['n']}")
        row("rastgele 12 nöron çeviriyor", lambda s: f"{s['rand_flip']}/{s['n']}")
        row("yama Δ/m0", lambda s: fmt(s["P_rel"]))
        row("rastgele Δ/m0", lambda s: fmt(s["rand_rel"]))
        row("**2. Cevabı çeviren en küçük alt küme (nöron)**", lambda s: str(sorted(x for x in s["min_flip_size"] if x is not None)))
        row("ortak nöronsuz iki çeviren alt küme olan zincir", lambda s: f"{s['chains_with_disjoint_flip']}/{s['n']}")
        row("**3. Aynı zincir, başka yazım: Δ/m0**", lambda s: fmt(s["para_rel"]))
        row("aynı zincir, başka yazım: Spearman", lambda s: fmt(s["para_spearman"]))
        row("başka zincir: Δ/m0", lambda s: fmt(s["cross_rel"]))
        row("zero Δ/m0 / resample Δ/m0", lambda s: f"{fmt(s['Z_rel'])} / {fmt(s['Z_resample_rel'])}")
        L.append("")
        if scope.startswith("Her"):
            L += ["### Zincir bazında: m0 ve ortak nöronsuz çeviren alt küme var mı", "",
                  "| model | zincir | m0 | çeviren alt küme | en küçük | ortak nöronsuz çift |", "|---|---|---|---|---|---|"]
            for m in runs:
                for c, v in S[m]["flip"].items():
                    L.append(f"| {m} | {c} | {S[m]['m0'][c]:.2f} | {v['n_flip']} | {v['min_size']} | {v['disjoint_pairs']} |")
            L.append("")
            allm = [(S[m]["m0"][c], S[m]["flip"][c]["disjoint_pairs"] > 0) for m in runs for c in S[m]["flip"]]
            lo = [d for m0, d in allm if m0 < 4]
            hi = [d for m0, d in allm if m0 >= 4]
            L += [f"Marj eğilimi (üç model birlikte): m0 < 4 olan {len(lo)} zincirin {sum(lo)}'inde, m0 ≥ 4 olan "
                  f"{len(hi)} zincirin {sum(hi)}'inde ortak nöronsuz çeviren alt küme var.", ""]
    out = Path("results") / "karsilastirma_uc_model.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
