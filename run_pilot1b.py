"""Pilot 1b: pilot 1'in aynı 12 adayıyla benzersizlik, etkileşim ve transfer (tanımlar: docs/pilot1b_tanimlar.md).

Float32 model; 4.096 alt kümenin hepsi özgün soruda (zero, mean-referans, rastgele kontrol) ve dört varyantta
(3 başka yazım + şıkları yer değişmiş), aday yeniden seçilmeden.

Kullanım: python run_pilot1b.py configs/pilot1b.yaml
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def spearman(a, b) -> float:
    """Sıra korelasyonu (scipy gerektirmez). Eşit değerler için ortalama sıra."""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])

from run_pilot import preflight
from src.analyze import additivity, bootstrap_ci, interaction_matrix, minimal_sets, uniqueness
from src.capture import capture
from src.common import answer_ids, env_record, load_jsonl, load_model
from src.intervene import all_subsets, prefix_cache, run_masks, run_masks_cached
from src.score import scores

ROOT = Path(__file__).parent
QS = (0.3, 0.5, 0.7)


def enc(tok, prompt, device):
    return tok(prompt, return_tensors="pt").input_ids.to(device)


def measure(model, tok, prompt, idc, idw, cand, repl, masks, dev, b):
    ids = enc(tok, prompt, dev)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    m0 = float(scores(logits, idc, idw)["margin"][0])
    r = run_masks_cached(model, prefix_cache(model, ids), ids[:, -1:], cand, masks, repl, idc, idw, b)
    return m0, (r["base_margin"] - r["margin"]).numpy(), r["flipped"].numpy(), ids


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    dev, b = cfg["device"], cfg["batch"]
    torch.manual_seed(cfg["seed"])
    preflight(cfg)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    model, tok = load_model(cfg["model"], cfg["dtype"], dev, cfg.get("cache_dir"), cfg.get("lm_head_fp32", False))
    ans = answer_ids(tok)
    src_dir = ROOT / cfg["pilot1_dir"]
    P1 = json.loads((src_dir / "per_example.json").read_text(encoding="utf-8"))
    data = {r["id"]: r for r in load_jsonl(ROOT / cfg["data"])}
    variants = {}
    for r in data.values():
        variants.setdefault(r["base_id"], []).append(r)
    ref = load_jsonl(ROOT / cfg["reference"])
    layers = sorted({c["layer"] for e in P1 for c in e["candidates"]})
    ref_acts = [capture(model, enc(tok, r["prompt"], dev), layers, -1) for r in ref]
    mean_act = {l: torch.stack([a[l] for a in ref_acts]).mean(0) for l in layers}
    run_dir = ROOT / "results" / f"pilot1b_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    K = 12
    masks = all_subsets(K)
    out = []
    for ei, e in enumerate(P1):
        r = data[e["id"]]
        idc, idw = ans[r["correct"]], ans[r["wrong"]]
        cand = [(c["layer"], c["neuron"]) for c in e["candidates"]]
        rand = [tuple(x) for x in e["random"]]
        mean_repl = torch.tensor([float(mean_act[l][i]) for l, i in cand])
        m0, d_zero, fl_zero, ids = measure(model, tok, r["prompt"], idc, idw, cand, torch.zeros(K), masks, dev, b)
        _, d_mean, _, _ = measure(model, tok, r["prompt"], idc, idw, cand, mean_repl, masks, dev, b)
        _, d_rand, _, _ = measure(model, tok, r["prompt"], idc, idw, rand, torch.zeros(K), masks, dev, b)
        # Sayısal denetim: 32 küme önbelleksiz tam geçişle
        chk = masks[torch.arange(0, 4096, 128)]
        f = run_masks(model, ids, ids.shape[1] - 1, cand, chk, torch.zeros(K), idc, idw, 33)
        cache_diff = float(np.abs((f["base_margin"] - f["margin"]).numpy() - d_zero[::128]).max())

        res = {"id": e["id"], "m0": m0, "m0_pilot1_bf16": e["m0"], "cache_diff": cache_diff,
               "candidates": e["candidates"], "full": {"zero": float(d_zero[-1]), "mean": float(d_mean[-1]),
                                                       "random": float(d_rand[-1])},
               "full_flipped": bool(fl_zero[-1]),
               "additivity": {"zero": additivity(d_zero, K), "random": additivity(d_rand, K)},
               "uniqueness": {f"q{int(q * 100)}": uniqueness(d_zero, K, q) for q in QS},
               "uniqueness_mean": {f"q{int(q * 100)}": uniqueness(d_mean, K, q) for q in QS},
               "m0_threshold": {f"tau{int(q * 100)}": minimal_sets(d_zero, K, q * m0)["n_effective"] for q in QS}}
        inter, inter_r = interaction_matrix(d_zero, K), interaction_matrix(d_rand, K)
        iu = np.triu_indices(K, 1)
        thr = 0.05 * d_zero[-1]
        res["interaction"] = {"n_pairs": len(iu[0]), "n_synergy": int((inter[iu] > thr).sum()),
                              "n_redundant": int((inter[iu] < -thr).sum()), "threshold": float(thr),
                              "median_abs": float(np.median(np.abs(inter[iu]))),
                              "random_median_abs": float(np.median(np.abs(inter_r[iu])))}
        # Transfer: aynı 4.096 küme, yeniden seçim yok
        u50 = res["uniqueness"]["q50"]["minimal"]
        rel0 = d_zero / m0
        top = res["uniqueness"]["q50"]["top_neuron_index"]
        res["transfer"] = {}
        for v in sorted(variants[r["base_id"]], key=lambda v: (v["swapped"], v["paraphrase"])):
            if v["id"] == r["id"]:
                continue
            vm0, vd, vfl, _ = measure(model, tok, v["prompt"], ans[v["correct"]], ans[v["wrong"]], cand, torch.zeros(K),
                                      masks, dev, b)
            vrel = vd / vm0
            rho = spearman(rel0[1:], vrel[1:])
            keep = [float(vrel[s] / rel0[s]) for s in u50] if u50 else []
            singles_v = [vd[1 << i] for i in range(K)]
            res["transfer"][v["id"]] = {
                "swapped": v["swapped"], "paraphrase": v["paraphrase"], "m0": vm0, "valid": vm0 >= cfg["min_margin"],
                "spearman_all_sets": rho,
                "full_rel": float(vrel[-1]), "full_flipped": bool(vfl[-1]),
                "minimal_q50_retention_median": float(np.median(keep)) if keep else None,
                "top_still_top": int(np.argmax(singles_v)) == top,
                "uniq_q50_in_variant": {k2: v2 for k2, v2 in uniqueness(vd, K, 0.5).items()
                                        if k2 in ("n_effective", "minimal", "min_cardinality", "n_disjoint_pairs",
                                                  "share_with_top", "max_without_top_rel")}}
            np.save(run_dir / f"{v['id']}_d_zero.npy", vd.astype(np.float32))
        np.savez_compressed(run_dir / f"{e['id']}.npz", d_zero=d_zero, d_mean=d_mean, d_rand=d_rand,
                            interaction=inter, interaction_random=inter_r)
        out.append(res)
        u = res["uniqueness"]["q50"]
        log(f"[{ei + 1}/{len(P1)}] {e['id']} m0={m0:.2f} Δtam/m0={d_zero[-1] / m0:.2f} | q50 minimal {len(u['minimal'])} "
            f"(boyut {sorted(u['minimal_sizes'])[:6]}), ortak nöronsuz çift {u['n_disjoint_pairs']}, en güçlü nöronu "
            f"içeren %{(u['share_with_top'] or 0) * 100:.0f}, o olmadan max {u['max_without_top_rel']:.2f} | "
            f"toplamsallık R² {res['additivity']['zero']['r2']:.3f} | önbellek farkı {cache_diff:.2g}")

    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
    env = env_record(model, cfg["model"], cfg["seed"], {"peak_gpu_gib": peak, "elapsed_s": round(time.time() - t0, 1),
                                                        "config": cfg})
    (run_dir / "per_example.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(env, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(run_dir, env, out)
    log(f"bitti: {run_dir} (tepe GPU {peak and round(peak, 2)} GiB)")


def write_md(run_dir, env, P) -> None:
    def ci(v):
        m, lo, hi = bootstrap_ci(v)
        return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
    L = [f"# Pilot 1b — {run_dir.name}", "",
         f"`{env['model']}` rev {env['revision']}, {env['dtype']}, {env['gpu']}, torch {env['torch']}, transformers "
         f"{env['transformers']}, süre {env['elapsed_s']} sn, tepe GPU {env['peak_gpu_gib']} GiB. Tanımlar: docs/pilot1b_tanimlar.md", "",
         "Aynı 10 soru, pilot 1'in aynı 12 adayı, 4.096 alt kümenin hepsi; float32. Medyan [%95 bootstrap GA].", "",
         "## Tam grup etkisi (Δ_tam / m0)", "",
         f"- zero: {ci([e['full']['zero'] / e['m0'] for e in P])}; mean (referans): {ci([e['full']['mean'] / e['m0'] for e in P])}; "
         f"rastgele: {ci([e['full']['random'] / e['m0'] for e in P])}",
         f"- m0'ın %50'sine ulaşan küme sayısı (pilot 1 eşiği): {[e['m0_threshold']['tau50'] for e in P]}", "",
         "## Toplamsallık (Δ(S) ~ tekli etkilerin toplamı)", "",
         f"- R²: {ci([e['additivity']['zero']['r2'] for e in P])}; eğim: {ci([e['additivity']['zero']['slope_through_origin'] for e in P])}; "
         f"Δ_tam / Σ tekli: {ci([e['additivity']['zero']['full_over_sum'] for e in P])}", "",
         "## İkili etkileşim (66 çift; eşik 0,05·Δ_tam)", "",
         "| örnek | sinerji | yedeklilik | |I| medyan | rastgele |I| medyan |", "|---|---|---|---|---|"]
    for e in P:
        i = e["interaction"]
        L.append(f"| {e['id']} | {i['n_synergy']} | {i['n_redundant']} | {i['median_abs']:.3f} | {i['random_median_abs']:.3f} |")
    L += ["", "## Benzersizlik (eşik = tam grup etkisinin q oranı; zero)", "",
          "| örnek | q | minimal küme | boyutlar | ortak nöronsuz çift | en güçlü nöronu içeren | o nöron olmadan max Δ/Δ_tam |",
          "|---|---|---|---|---|---|---|"]
    for e in P:
        top = e["candidates"][e["uniqueness"]["q50"]["top_neuron_index"]]
        for q in ("q30", "q50", "q70"):
            u = e["uniqueness"][q]
            st = "—" if u["share_with_top"] is None else f"%{u['share_with_top'] * 100:.0f}"
            L.append(f"| {e['id']} (en güçlü L{top['layer']}/{top['neuron']}) | {q} | {len(u['minimal'])} | "
                     f"{sorted(u['minimal_sizes'])[:8]} | {u['n_disjoint_pairs']} | {st} | {u['max_without_top_rel']:.2f} |")
    L += ["", "## Transfer (aynı kümeler, yeniden seçim yok)", "",
          "| örnek | varyant | m0 | Spearman (4.095 küme) | tam grup Δ/m0 | q50 minimal koruma (medyan) | en güçlü nöron yine en güçlü |",
          "|---|---|---|---|---|---|---|"]
    for e in P:
        for vid, t in e["transfer"].items():
            kind = ("şık-değişik" if t["swapped"] else f"yazım {t['paraphrase']}") + ("" if t["valid"] else " (model yanlış/zayıf: dışla)")
            ret = "—" if t["minimal_q50_retention_median"] is None else f"{t['minimal_q50_retention_median']:.2f}"
            L.append(f"| {e['id']} | {kind} | {t['m0']:.2f} | {t['spearman_all_sets']:.2f} | {t['full_rel']:.2f} | {ret} | "
                     f"{'evet' if t['top_still_top'] else 'hayır'} |")
    L += ["", "Sayısal denetim (önbellekli/tam geçiş en büyük Δ farkı): " +
          ", ".join(f"{e['cache_diff']:.1g}" for e in P), "",
          "Pilot; keşif soruları. Hipotez doğrulaması ayrı test bölümünde yapılacak."]
    (run_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "pilot1b.yaml"))
