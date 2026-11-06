"""Adım 2: resample testi (keşif) + test bölümünde dondurulmuş grup transferi ve bulguların tekrarı.

Tanımlar: docs/adim2_tanimlar.md. Float32, son prompt konumu, önbellekli son-token yolu.
Kullanım: python run_step2.py configs/step2.yaml
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from run_pilot import preflight
from run_pilot1b import enc, measure, spearman
from src.analyze import additivity, bootstrap_ci, interaction_matrix, uniqueness
from src.capture import attribution, capture
from src.common import answer_ids, env_record, load_jsonl, load_model
from src.intervene import all_subsets, prefix_cache, run_masks_cached

ROOT = Path(__file__).parent
K = 12


def full_and_singles() -> torch.Tensor:
    """[13, 12]: tam grup + 12 tekli."""
    return torch.cat([torch.ones(1, K, dtype=torch.bool), torch.eye(K, dtype=torch.bool)])


def select_candidates(model, ids, layers, idc, idw, nn, b):
    """Pilot 1'in seçim yöntemi (float32): 2 katmanın tekli zero taraması → en etkili 6 + atamayla sonraki 6."""
    pos = ids.shape[1] - 1
    cache = prefix_cache(model, ids)
    single = {}
    for l in layers:
        r = run_masks_cached(model, cache, ids[:, -1:], [(l, i) for i in range(nn)], torch.eye(nn, dtype=torch.bool),
                             torch.zeros(nn), idc, idw, b)
        single[l] = (r["base_margin"] - r["margin"]).numpy()
    attr = attribution(model, ids, layers, pos, idc, idw)
    flat = [(l, i, single[l][i], float(attr[l][i])) for l in layers for i in range(nn)]
    top = sorted(flat, key=lambda x: -x[2])[:6]
    chosen = {(l, i) for l, i, _, _ in top}
    rest = [x for x in sorted(flat, key=lambda x: -x[3]) if (x[0], x[1]) not in chosen][:6]
    return [(l, i) for l, i, _, _ in top + rest], {(l, i): s for l, i, s, _ in flat}


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    dev, b, seed = cfg["device"], cfg["batch"], cfg["seed"]
    torch.manual_seed(seed)
    rng = random.Random(seed)
    preflight(cfg)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    model, tok = load_model(cfg["model"], cfg["dtype"], dev, cfg.get("cache_dir"), False)
    ans = answer_ids(tok)
    nn = model.config.intermediate_size
    P1 = json.loads((ROOT / cfg["pilot1_dir"] / "per_example.json").read_text(encoding="utf-8"))
    data = load_jsonl(ROOT / cfg["data"])
    variants = {}
    for r in data:
        variants.setdefault(r["base_id"], []).append(r)
    layers = sorted({c["layer"] for e in P1 for c in e["candidates"]})
    disc = [r for r in data if r["split"] == "discovery" and r["paraphrase"] == 0 and not r["swapped"]]
    test = [r for r in data if r["split"] == "test" and r["paraphrase"] == 0 and not r["swapped"]]
    run_dir = ROOT / "results" / f"step2_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    # Aktivasyonlar: keşif soruları (donör ve görev içi ortalama), referans A/B soruları
    acts = {r["id"]: capture(model, enc(tok, r["prompt"], dev), layers, -1) for r in disc}
    ref = load_jsonl(ROOT / cfg["reference"])
    ref_acts = [capture(model, enc(tok, r["prompt"], dev), layers, -1) for r in ref]
    ref_mean = {l: torch.stack([a[l] for a in ref_acts]).mean(0) for l in layers}
    by_id = {r["id"]: r for r in data}

    # ---------------- A. Resample ----------------
    fs = full_and_singles()
    A = []
    for e in P1:
        r = by_id[e["id"]]
        idc, idw = ans[r["correct"]], ans[r["wrong"]]
        cand = [(c["layer"], c["neuron"]) for c in e["candidates"]]
        val = lambda src: torch.tensor([float(src[l][i]) for l, i in cand])
        others = [d for d in disc if d["id"] != r["id"]]
        task_mean = {l: torch.stack([acts[d["id"]][l] for d in others]).mean(0) for l in layers}
        same = rng.sample([d for d in others if d["correct"] == r["correct"]], 3)
        diff = rng.sample([d for d in others if d["correct"] != r["correct"]], 3)
        repls = {"zero": torch.zeros(K), "ref_mean": val(ref_mean), "task_mean": val(task_mean),
                 **{f"same_{d['id']}": val(acts[d["id"]]) for d in same},
                 **{f"diff_{d['id']}": val(acts[d["id"]]) for d in diff}}
        row = {"id": e["id"], "correct": r["correct"]}
        for name, rep in repls.items():
            m0, d, fl, _ = measure(model, tok, r["prompt"], idc, idw, cand, rep, fs, dev, b)
            row[name] = {"m0": m0, "full_rel": float(d[0] / m0), "full_flipped": bool(fl[0]),
                         "singles_rel": [float(x / m0) for x in d[1:]]}
        own = acts[e["id"]]
        row["act_own_vs_refmean"] = [[float(own[l][i]), float(ref_mean[l][i]), float(task_mean[l][i])] for l, i in cand]
        A.append(row)
        s_same = np.mean([v["full_rel"] for k, v in row.items() if k.startswith("same_")])
        s_diff = np.mean([v["full_rel"] for k, v in row.items() if k.startswith("diff_")])
        log(f"A {e['id']}: zero {row['zero']['full_rel']:.2f} ref {row['ref_mean']['full_rel']:.2f} "
            f"görev-içi {row['task_mean']['full_rel']:.2f} resample aynı-harf {s_same:.2f} farklı-harf {s_diff:.2f}")

    # ---------------- B1. Dondurulmuş küresel grup → test soruları ----------------
    cnt = Counter((c["layer"], c["neuron"]) for e in P1 for c in e["candidates"])
    med = {k: np.median([c["single_delta"] for e in P1 for c in e["candidates"] if (c["layer"], c["neuron"]) == k])
           for k in cnt}
    glob = sorted(cnt, key=lambda k: (-cnt[k], -med[k]))[:K]
    nrng = np.random.default_rng(seed)
    rand = []
    for l in layers:
        rand += [(l, int(i)) for i in nrng.choice(nn, sum(1 for c in glob if c[0] == l), replace=False)]
    B1 = []
    for r in test:
        for v in sorted(variants[r["base_id"]], key=lambda v: (v["swapped"], v["paraphrase"])):
            vc, vw = ans[v["correct"]], ans[v["wrong"]]
            m0, d, fl, _ = measure(model, tok, v["prompt"], vc, vw, glob, torch.zeros(K), fs, dev, b)
            _, dr, _, _ = measure(model, tok, v["prompt"], vc, vw, rand, torch.zeros(K), fs[:1], dev, b)
            B1.append({"id": v["id"], "base": r["base_id"], "swapped": v["swapped"], "paraphrase": v["paraphrase"],
                       "m0": m0, "valid": m0 >= cfg["min_margin"], "full_rel": float(d[0] / m0),
                       "random_full_rel": float(dr[0] / m0), "full_flipped": bool(fl[0]),
                       "singles_rel": [float(x / m0) for x in d[1:]]})
    log(f"B1: küresel grup {glob}; {len(B1)} test varyantı")

    # ---------------- B2. Test sorularında yeniden seçim + 4.096 alt küme ----------------
    masks = all_subsets(K)
    B2 = []
    for r in test:
        idc, idw = ans[r["correct"]], ans[r["wrong"]]
        ids = enc(tok, r["prompt"], dev)
        m0, _, _, _ = measure(model, tok, r["prompt"], idc, idw, [(layers[0], 0)], torch.zeros(1), fs[:1, :1], dev, b)
        if m0 < cfg["min_margin"]:
            B2.append({"id": r["id"], "m0": m0, "excluded": True})
            log(f"B2 {r['id']}: dışlandı (m0={m0:.2f})")
            continue
        cand, single = select_candidates(model, ids, layers, idc, idw, nn, b)
        _, d, fl, _ = measure(model, tok, r["prompt"], idc, idw, cand, torch.zeros(K), masks, dev, b)
        inter = interaction_matrix(d, K)
        iu = np.triu_indices(K, 1)
        thr = 0.05 * d[-1]
        res = {"id": r["id"], "m0": m0, "excluded": False,
               "candidates": [{"layer": l, "neuron": i, "single_delta": float(single[(l, i)])} for l, i in cand],
               "full_rel": float(d[-1] / m0), "additivity": additivity(d, K),
               "interaction": {"n_synergy": int((inter[iu] > thr).sum()), "n_redundant": int((inter[iu] < -thr).sum()),
                               "median_abs": float(np.median(np.abs(inter[iu])))},
               "uniqueness": {f"q{int(q * 100)}": {k: v for k, v in uniqueness(d, K, q).items() if k != "minimal"}
                              for q in (0.3, 0.5, 0.7)},
               "transfer": {}}
        rel0 = d / m0
        top = res["uniqueness"]["q50"]["top_neuron_index"]
        for v in sorted(variants[r["base_id"]], key=lambda v: (v["swapped"], v["paraphrase"])):
            if v["id"] == r["id"]:
                continue
            vm0, vd, _, _ = measure(model, tok, v["prompt"], ans[v["correct"]], ans[v["wrong"]], cand, torch.zeros(K), masks, dev, b)
            res["transfer"][v["id"]] = {"swapped": v["swapped"], "m0": vm0, "valid": vm0 >= cfg["min_margin"],
                                        "spearman": spearman(rel0[1:], (vd / vm0)[1:]), "full_rel": float(vd[-1] / vm0),
                                        "top_still_top": int(np.argmax([vd[1 << i] for i in range(K)])) == top}
        np.save(run_dir / f"{r['id']}_d_zero.npy", d.astype(np.float32))
        B2.append(res)
        u = res["uniqueness"]["q50"]
        log(f"B2 {r['id']}: m0={m0:.2f} Δtam/m0={res['full_rel']:.2f} R² {res['additivity']['r2']:.3f} "
            f"sinerji/yedek {res['interaction']['n_synergy']}/{res['interaction']['n_redundant']} q50 minimal "
            f"{u['n_effective'] and len(u['minimal_sizes'])} ayrık çift {u['n_disjoint_pairs']}; en güçlü "
            f"L{cand[top][0]}/{cand[top][1]}")

    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
    env = env_record(model, cfg["model"], seed, {"peak_gpu_gib": peak, "elapsed_s": round(time.time() - t0, 1), "config": cfg})
    out = {"env": env, "A_resample": A, "B1_global_set": [list(g) for g in glob], "B1_random_set": [list(x) for x in rand],
           "B1": B1, "B2": B2}
    (run_dir / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(run_dir, out)
    log(f"bitti: {run_dir} (tepe GPU {peak and round(peak, 2)} GiB)")


def write_md(run_dir, o) -> None:
    def ci(v):
        v = [x for x in v if x is not None]
        m, lo, hi = bootstrap_ci(v)
        return f"{m:.2f} [{lo:.2f}, {hi:.2f}] (n={len(v)})"
    env, A, B1, B2 = o["env"], o["A_resample"], o["B1"], o["B2"]
    L = [f"# Adım 2 — {run_dir.name}", "",
         f"`{env['model']}` rev {env['revision']}, {env['dtype']}, {env['gpu']}, süre {env['elapsed_s']} sn, tepe GPU "
         f"{env['peak_gpu_gib']} GiB. Tanımlar: docs/adim2_tanimlar.md. Medyan [%95 bootstrap GA].", "",
         "## A. Resample (keşif soruları, pilot 1 adayları): tam grup Δ/m0", ""]
    same = [np.mean([v["full_rel"] for k, v in a.items() if k.startswith("same_")]) for a in A]
    diff = [np.mean([v["full_rel"] for k, v in a.items() if k.startswith("diff_")]) for a in A]
    for name, vals in (("zero", [a["zero"]["full_rel"] for a in A]), ("referans ortalaması", [a["ref_mean"]["full_rel"] for a in A]),
                       ("görev içi ortalama", [a["task_mean"]["full_rel"] for a in A]),
                       ("resample, aynı doğru harf (3 donör ort.)", same), ("resample, farklı doğru harf (3 donör ort.)", diff)):
        L.append(f"- {name}: {ci(vals)}")
    L += ["", "| örnek | doğru | zero | ref ort. | görev içi ort. | aynı harf | farklı harf |", "|---|---|---|---|---|---|---|"]
    for a, s, d in zip(A, same, diff):
        L.append(f"| {a['id']} | {a['correct']} | {a['zero']['full_rel']:.2f} | {a['ref_mean']['full_rel']:.2f} | "
                 f"{a['task_mean']['full_rel']:.2f} | {s:.2f} | {d:.2f} |")
    L += ["", f"## B1. Dondurulmuş küresel grup (keşifte en sık 12 nöron) → test soruları, yeniden seçim yok", "",
          f"Grup: {o['B1_global_set']}", ""]
    for kind, sel in (("özgün yazım", lambda x: not x["swapped"] and x["paraphrase"] == 0),
                      ("diğer yazımlar", lambda x: not x["swapped"] and x["paraphrase"] > 0),
                      ("şık-değişik", lambda x: x["swapped"])):
        rows = [x for x in B1 if sel(x) and x["valid"]]
        L.append(f"- {kind}: grup {ci([x['full_rel'] for x in rows])}; rastgele 12 {ci([x['random_full_rel'] for x in rows])}")
    L.append(f"- geçersiz (müdahalesiz marj < eşik) varyant: {sum(not x['valid'] for x in B1)}/{len(B1)}")
    L += ["", "## B2. Test sorularında yeniden seçim + 4.096 alt küme (bulguların tekrarı)", "",
          "| örnek | m0 | Δtam/m0 | R² | eğim | sinerji/yedek | q50 minimal küme | ayrık çift | en güçlü nöronsuz max | yazım Spearman (min) | şık-değişik Spearman | en güçlü yine en güçlü |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in B2:
        if r.get("excluded"):
            L.append(f"| {r['id']} | {r['m0']:.2f} | dışlandı | | | | | | | | | |")
            continue
        u = r["uniqueness"]["q50"]
        tr = [t for t in r["transfer"].values() if t["valid"]]
        para = [t["spearman"] for t in tr if not t["swapped"]]
        sw = [t["spearman"] for t in tr if t["swapped"]]
        keep = sum(t["top_still_top"] for t in tr)
        L.append(f"| {r['id']} | {r['m0']:.2f} | {r['full_rel']:.2f} | {r['additivity']['r2']:.3f} | "
                 f"{r['additivity']['slope_through_origin']:.2f} | {r['interaction']['n_synergy']}/{r['interaction']['n_redundant']} | "
                 f"{len(u['minimal_sizes'])} | {u['n_disjoint_pairs']} | {u['max_without_top_rel']:.2f} | "
                 f"{min(para) if para else float('nan'):.2f} | {sw[0] if sw else float('nan'):.2f} | {keep}/{len(tr)} |")
    tops = Counter(f"L{r['candidates'][r['uniqueness']['q50']['top_neuron_index']]['layer']}/"
                   f"{r['candidates'][r['uniqueness']['q50']['top_neuron_index']]['neuron']}" for r in B2 if not r.get("excluded"))
    L += ["", f"Test sorularında en güçlü tekli nöron: {dict(tops)}", "",
          "Ön kayıtlı tekrar ölçütü: toplamsallık R² ≥ 0,95 ve sinerji/yedeklilik çifti yok; yazım Spearman ≥ 0,9."]
    (run_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "step2.yaml"))
