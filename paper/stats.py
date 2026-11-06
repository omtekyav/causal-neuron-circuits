"""Makaledeki bütün sayılar buradan, kayıtlı sonuç dosyalarından hesaplanır (elle sayı yazılmaz)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analyze import bootstrap_ci, minimal_sets  # noqa: E402

RES = ROOT / "results"
RUNS = {
    "Qwen3-1.7B": RES / "chains_20261106_121202",
    "Llama-3.2-1B": RES / "chains_Llama-3.2-1B_20261106_124956",
    "Gemma-3-1B": RES / "chains_gemma-3-1b-pt_20261106_125308",
}
COMP = {
    "Qwen3-1.7B": RES / "competence_Qwen3-1.7B-Base_20261106_114701",
    "Llama-3.2-1B": RES / "competence_Llama-3.2-1B_20261106_123642",
    "Gemma-3-1B": RES / "competence_gemma-3-1b-pt_20261106_124334",
}


def j(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def ci(v):
    v = [x for x in v if x is not None]
    return bootstrap_ci(v)


def study1() -> dict:
    p1b = j(RES / "pilot1b_20261106_111357" / "per_example.json")
    s2 = j(RES / "step2_20261106_112514" / "results.json")
    A = s2["A_resample"]
    same = [np.mean([v["full_rel"] for k, v in a.items() if k.startswith("same_")]) for a in A]
    diff = [np.mean([v["full_rel"] for k, v in a.items() if k.startswith("diff_")]) for a in A]
    own_dev = [abs(v[0] - v[2]) for a in A for v in a["act_own_vs_refmean"]]
    mag = [abs(v[0]) for a in A for v in a["act_own_vs_refmean"]]
    para = [t["spearman_all_sets"] for e in p1b for t in e["transfer"].values() if not t["swapped"]]
    sw = [t["spearman_all_sets"] for e in p1b for t in e["transfer"].values() if t["swapped"]]
    top_swapped = sum(t["top_still_top"] for e in p1b for t in e["transfer"].values() if t["swapped"])
    B2 = [r for r in s2["B2"] if not r.get("excluded")]
    B1 = s2["B1"]
    b1 = lambda sel, key: ci([x[key] for x in B1 if sel(x) and x["valid"]])
    return {
        "n": len(p1b),
        "zero": ci([e["full"]["zero"] / e["m0"] for e in p1b]),
        "ref_mean": ci([e["full"]["mean"] / e["m0"] for e in p1b]),
        "random": ci([e["full"]["random"] / e["m0"] for e in p1b]),
        "task_mean": ci([a["task_mean"]["full_rel"] for a in A]),
        "resample_same": ci(same), "resample_diff": ci(diff),
        "act_dev_median": float(np.median(own_dev)), "act_mag_median": float(np.median(mag)),
        "r2": ci([e["additivity"]["zero"]["r2"] for e in p1b]),
        "slope": ci([e["additivity"]["zero"]["slope_through_origin"] for e in p1b]),
        "n_interacting_pairs": sum(e["interaction"]["n_synergy"] + e["interaction"]["n_redundant"] for e in p1b),
        "q50_minimal_range": (min(len(e["uniqueness"]["q50"]["minimal"]) for e in p1b),
                              max(len(e["uniqueness"]["q50"]["minimal"]) for e in p1b)),
        "q50_disjoint_examples": sum(e["uniqueness"]["q50"]["n_disjoint_pairs"] > 0 for e in p1b),
        "without_top_range": (min(e["uniqueness"]["q50"]["max_without_top_rel"] for e in p1b),
                              max(e["uniqueness"]["q50"]["max_without_top_rel"] for e in p1b)),
        "para_spearman_range": (min(para), max(para)), "swap_spearman_range": (min(sw), max(sw)),
        "top_still_top_swapped": top_swapped,
        "per_example_zero": [e["full"]["zero"] / e["m0"] for e in p1b],
        "test_r2_range": (min(r["additivity"]["r2"] for r in B2), max(r["additivity"]["r2"] for r in B2)),
        "test_n": len(B2), "test_interacting": sum(r["interaction"]["n_synergy"] + r["interaction"]["n_redundant"] for r in B2),
        "test_top692": sum(1 for r in B2 if (r["candidates"][r["uniqueness"]["q50"]["top_neuron_index"]]["layer"],
                                             r["candidates"][r["uniqueness"]["q50"]["top_neuron_index"]]["neuron"]) == (14, 692)),
        "frozen_test_orig": b1(lambda x: not x["swapped"] and x["paraphrase"] == 0, "full_rel"),
        "frozen_test_para": b1(lambda x: not x["swapped"] and x["paraphrase"] > 0, "full_rel"),
        "frozen_test_swap": b1(lambda x: x["swapped"], "full_rel"),
        "frozen_test_random": b1(lambda x: True, "random_full_rel"),
    }


def competence() -> dict:
    out = {}
    for m, d in COMP.items():
        rep = list(j(d / "competence.json").values())[0]
        out[m] = {**rep["accuracy"], "solved": len(rep["solved_chains"]), "n": rep["n_chains"]}
    return out


def flip_info(d, m0):
    ms = minimal_sets(d.astype(float), 12, m0)
    mins = ms["minimal"]
    return {"n_flip": ms["n_effective"], "min_size": ms["min_cardinality"],
            "disjoint": sum(1 for i, a in enumerate(mins) for b in mins[i + 1:] if a & b == 0)}


def study2() -> dict:
    data = {}
    for m, d in RUNS.items():
        R = j(d / "results.json")
        deltas = {e["chain"]: dict(np.load(d / f"{e['chain']}.npz")) for e in R["chains"]}
        data[m] = (R, deltas)
    common = sorted(set.intersection(*[set(R["included"]) for R, _ in data.values()]))
    out = {"common": common, "models": {}}
    for m, (R, deltas) in data.items():
        def summ(chains):
            E = [e for e in R["chains"] if e["chain"] in chains]
            cross = [x for x in R["cross"] if x["group_of"] in chains and x["applied_to"] in chains and x["m0"] > 0]
            fi = {e["chain"]: flip_info(deltas[e["chain"]]["P_hop1"], e["m0"]) for e in E}
            tr = [t["P_hop1"] for e in E for t in e["transfer"].values() if t["P_hop1"]["valid"]]
            return {
                "n": len(E), "chains": [e["chain"] for e in E],
                "P_flip": sum(e["cond"]["P_hop1"]["full_flipped"] for e in E),
                "P_gen_bridge": sum(e["generation"]["P_hop1"]["full"]["starts_with_bridge"] for e in E),
                "P_top1_bridge": sum(e["cond"]["P_hop1"]["full_top1_is_bridge"] for e in E),
                "rand_flip": sum(e["cond"]["P_random"]["full_flipped"] for e in E),
                "Z_flip": sum(e["cond"]["Z_zero"]["full_flipped"] for e in E),
                "P_rel": ci([e["cond"]["P_hop1"]["full_rel"] for e in E]),
                "P_other": ci([e["cond"]["P_other"]["full_rel"] for e in E]),
                "rand_rel": ci([e["cond"]["P_random"]["full_rel"] for e in E]),
                "Z_rel": ci([e["cond"]["Z_zero"]["full_rel"] for e in E]),
                "Z_resample": ci([e["cond"]["Z_resample"]["full_rel"] for e in E]),
                "Z_rand": ci([e["cond"]["Z_random"]["full_rel"] for e in E]),
                "para_rel": ci([t["full_rel"] for t in tr]), "para_spearman": ci([t["spearman"] for t in tr]),
                "n_para_valid": len(tr),
                "cross_rel": ci([x["P_hop1_rel"] for x in cross]),
                "flip": fi, "m0": {e["chain"]: e["m0"] for e in E},
                "chains_disjoint": sum(v["disjoint"] > 0 for v in fi.values()),
                "min_sizes": sorted(v["min_size"] for v in fi.values() if v["min_size"] is not None),
                "r2": ci([e["analysis"]["P_hop1"]["additivity"]["r2"] for e in E if e["analysis"]["P_hop1"]["interpretable"]]),
                "slope": ci([e["analysis"]["P_hop1"]["additivity"]["slope_through_origin"] for e in E
                             if e["analysis"]["P_hop1"]["interpretable"]]),
                "n_pairs_interacting": sum(e["analysis"]["P_hop1"]["interaction"]["n_synergy"] +
                                           e["analysis"]["P_hop1"]["interaction"]["n_redundant"] for e in E),
                "noise_range": (min(min(e["noise"].values()) for e in E), max(max(e["noise"].values()) for e in E)),
                "dlog": [(e["cond"]["Z_zero"]["d_logit_answer"], e["cond"]["Z_zero"]["d_logit_bridge"],
                          e["cond"]["P_hop1"]["d_logit_answer"], e["cond"]["P_hop1"]["d_logit_bridge"]) for e in E],
                "gen": {e["chain"]: (e["generation"]["P_hop1"]["none"]["text"], e["generation"]["P_hop1"]["full"]["text"],
                                     e["generation"]["Z_zero"]["full"]["text"]) for e in E},
                "answer_bridge": {e["chain"]: (e["answer"], e["bridge"]) for e in E},
            }
        out["models"][m] = {"own": summ(R["included"]), "common": summ(common), "excluded": list(R["excluded"])}
    allm = [(s["own"]["m0"][c], s["own"]["flip"][c]["disjoint"] > 0) for s in out["models"].values() for c in s["own"]["flip"]]
    out["margin_trend"] = {"lo_n": sum(m0 < 4 for m0, _ in allm), "lo_disj": sum(d for m0, d in allm if m0 < 4),
                           "hi_n": sum(m0 >= 4 for m0, _ in allm), "hi_disj": sum(d for m0, d in allm if m0 >= 4)}
    out["deltas"] = {m: v[1] for m, v in data.items()}
    out["R"] = {m: v[0] for m, v in data.items()}
    return out


if __name__ == "__main__":
    s1 = study1()
    print(json.dumps({k: v for k, v in s1.items() if k != "per_example_zero"}, indent=1, default=float))
    print(json.dumps(competence(), indent=1))
    s2 = study2()
    print("ortak:", s2["common"], s2["margin_trend"])
    for m, v in s2["models"].items():
        o, c = v["own"], v["common"]
        print(m, {k: o[k] for k in ("n", "P_flip", "P_gen_bridge", "rand_flip", "Z_flip", "P_rel", "P_other", "Z_rel", "Z_resample",
                                     "para_spearman", "cross_rel", "chains_disjoint", "min_sizes", "r2", "slope",
                                     "n_pairs_interacting", "noise_range")})
        print("  ortak:", {k: c[k] for k in ("n", "P_flip", "P_gen_bridge", "cross_rel", "para_spearman", "chains_disjoint")})
