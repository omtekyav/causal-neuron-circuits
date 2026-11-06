"""Makale A pilotu: 10 örnek × 1 model × 2 katman.

Adımlar (örnek başına): müdahalesiz marj → tekli zero ablation taraması (2 katmanın bütün nöronları) →
activation×gradient → 12 aday (6 tekli en etkili + 6 atama ile, tekli taramada ilk 6'ya girmeyen) →
2^12 alt kümenin hepsi, zero ve mean ablation → kontroller: katman sayısı eşleştirilmiş rastgele 12 nöron,
yüksek aktivasyonlu ama tekli etkisi düşük 12 nöron (ikisi de 2^12, zero) → minimal kümeler (τ = %30/%50/%70).

Paylaşılan GPU güvenliği: başlamadan boş GPU belleği kontrol edilir (yetmezse çalışmaz); süreç bellek
tavanıyla sınırlanır (memory_fraction); ölçülmemiş sonuç üretilmez.

Kullanım: python run_pilot.py configs/pilot.yaml
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.analyze import (bootstrap_ci, break_point, degradation_curve, interaction_matrix, mean_pairwise_jaccard,
                         minimal_sets)
from src.capture import attribution, capture
from src.common import answer_ids, env_record, load_jsonl, load_model, n_neurons
from src.intervene import all_subsets, run_masks
from src.score import scores

ROOT = Path(__file__).parent


def preflight(cfg) -> None:
    if cfg["device"] != "cuda":
        return
    free, total = torch.cuda.mem_get_info()
    need = cfg["mem_fraction"] * total + cfg["mem_margin_mb"] * 2**20
    print(f"GPU boş {free / 2**30:.2f} GiB / toplam {total / 2**30:.2f} GiB; gereken (tavan+pay) {need / 2**30:.2f} GiB")
    if free < need:
        sys.exit("YETERSİZ BOŞ GPU BELLEĞİ — çalışma başlatılmadı.")
    torch.cuda.set_per_process_memory_fraction(cfg["mem_fraction"])


def single_scan(model, ids, pos, layer, nn, idc, idw, batch):
    cand = [(layer, i) for i in range(nn)]
    masks = torch.eye(nn, dtype=torch.bool)
    r = run_masks(model, ids, pos, cand, masks, torch.zeros(nn), idc, idw, batch)
    return (r["base_margin"] - r["margin"]).numpy(), r


def subset_scan(model, ids, pos, cand, repl, idc, idw, batch):
    masks = all_subsets(len(cand))
    r = run_masks(model, ids, pos, cand, masks, repl, idc, idw, batch)
    return (r["base_margin"] - r["margin"]).numpy(), r


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    preflight(cfg)
    t0 = time.time()
    model, tok = load_model(cfg["model"], cfg["dtype"], cfg["device"], cfg.get("cache_dir"), cfg.get("lm_head_fp32", True))
    ans = answer_ids(tok)
    nn = n_neurons(model)
    layers = cfg["layers"]
    run_dir = ROOT / "results" / f"pilot_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    # Örnek seçimi: keşif bölümü, 0. yazım; müdahalesiz model doğru ve marj ≥ min_margin olanlar
    rows = [r for r in load_jsonl(ROOT / cfg["data"])
            if r["split"] == "discovery" and r["paraphrase"] == 0 and not r.get("swapped")]
    random.Random(seed).shuffle(rows)
    accepted, rejected = [], []
    for r in rows:
        if len(accepted) == cfg["n_examples"]:
            break
        ids = tok(r["prompt"], return_tensors="pt").input_ids.to(cfg["device"])
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        s = scores(logits, ans[r["correct"]], ans[r["wrong"]])
        m0 = float(s["margin"][0])
        (accepted if m0 >= cfg["min_margin"] else rejected).append({**r, "m0": m0, "p0": float(s["p_correct"][0]), "ids": ids})
    print(f"kabul {len(accepted)}, dışlanan {len(rejected)} (marj < {cfg['min_margin']} ya da yanlış)")

    # Mean ablation değerleri: kabul edilen örneklerin hedef konumdaki ortalama aktivasyonu
    acts = [capture(model, r["ids"], layers, r["ids"].shape[1] - 1) for r in accepted]
    mean_act = {l: torch.stack([a[l] for a in acts]).mean(0) for l in layers}

    per_example = []
    rng = np.random.default_rng(seed)
    for ei, r in enumerate(accepted):
        ids, pos = r["ids"], r["ids"].shape[1] - 1
        idc, idw = ans[r["correct"]], ans[r["wrong"]]
        b = cfg["batch"]
        single = {}
        noise = []
        for l in layers:
            d, raw = single_scan(model, ids, pos, l, nn, idc, idw, b)
            single[l] = d
            noise.append(float((raw["base_margin"] - r["m0"]).abs().max()))
        attr = attribution(model, ids, layers, pos, idc, idw)
        flat = [(l, i, single[l][i], float(attr[l][i])) for l in layers for i in range(nn)]
        top_single = sorted(flat, key=lambda x: -x[2])[:6]
        chosen = {(l, i) for l, i, _, _ in top_single}
        by_attr = [x for x in sorted(flat, key=lambda x: -x[3]) if (x[0], x[1]) not in chosen][:6]
        cand = [(l, i) for l, i, _, _ in top_single + by_attr]
        k = len(cand)
        zero_repl = torch.zeros(k)
        mean_repl = torch.tensor([float(mean_act[l][i]) for l, i in cand])
        d_zero, _ = subset_scan(model, ids, pos, cand, zero_repl, idc, idw, b)
        d_mean, _ = subset_scan(model, ids, pos, cand, mean_repl, idc, idw, b)

        # Kontrol (b): katman sayısı eşleştirilmiş rastgele 12 nöron
        rand = []
        for l in layers:
            n_l = sum(1 for ll, _ in cand if ll == l)
            rand += [(l, int(i)) for i in rng.choice(nn, n_l, replace=False)]
        d_rand, _ = subset_scan(model, ids, pos, rand, torch.zeros(k), idc, idw, b)
        # Kontrol (c): yüksek |aktivasyon| ama tekli etkisi düşük (|Δ| medyan altı) 12 nöron, katman eşleştirilmiş
        hiact = []
        for l in layers:
            n_l = sum(1 for ll, _ in cand if ll == l)
            a = acts[ei][l].abs().numpy()
            low = np.abs(single[l]) <= np.median(np.abs(single[l]))   # Δ'ların çoğu 0 olabilir: eşitlik dahil
            order = [i for i in np.argsort(-a) if low[i] and (l, int(i)) not in cand][:n_l]
            assert len(order) == n_l, (l, n_l, len(order))
            hiact += [(l, int(i)) for i in order]
        d_hi, _ = subset_scan(model, ids, pos, hiact, torch.zeros(k), idc, idw, b)

        res = {"id": r["id"], "m0": r["m0"], "p0": r["p0"], "noise_floor": max(noise),
               "candidates": [{"layer": l, "neuron": i, "single_delta": float(single[l][i]), "attr": float(attr[l][i]),
                               "source": "single" if n < 6 else "attribution"} for n, (l, i) in enumerate(cand)],
               "random": rand, "hiact": hiact}
        for frac in cfg["tau_fracs"]:
            tau = frac * r["m0"]
            res[f"tau{int(frac * 100)}"] = {name: minimal_sets(d, k, tau) for name, d in
                                            (("zero", d_zero), ("mean", d_mean), ("random", d_rand), ("hiact", d_hi))}
        main_ms = res["tau50"]["zero"]["minimal"]
        res["jaccard_zero_tau50"] = mean_pairwise_jaccard(main_ms)
        # Kademeli kapatma (12→11→…→0): en zayıftan / en güçlüden geri açma ve 20 rastgele sıra
        sd = [c for c in range(k)]
        weak_first = sorted(sd, key=lambda c: single[cand[c][0]][cand[c][1]])
        orders = {"weak_first": weak_first, "strong_first": weak_first[::-1]}
        orders.update({f"rand{j}": list(rng.permutation(k)) for j in range(20)})
        curves = {name: degradation_curve(d_zero, o) for name, o in orders.items()}
        res["break_point_tau50"] = {name: break_point(c, 0.5 * r["m0"]) for name, c in curves.items()}
        inter = interaction_matrix(d_zero, k)
        res["interaction_abs_median"] = float(np.nanmedian(np.abs(inter)))
        res["full_set_delta"] = {"zero": float(d_zero[-1]), "mean": float(d_mean[-1]),
                                 "random": float(d_rand[-1]), "hiact": float(d_hi[-1])}
        np.savez_compressed(run_dir / f"{r['id']}.npz", d_zero=d_zero, d_mean=d_mean, d_rand=d_rand, d_hi=d_hi,
                            interaction=inter, **{f"single_L{l}": single[l] for l in layers},
                            **{f"curve_{n}": c for n, c in curves.items()})
        per_example.append(res)
        t = res["tau50"]["zero"]
        print(f"[{ei + 1}/{len(accepted)}] {r['id']} m0={r['m0']:.2f} minimal={len(t['minimal'])} "
              f"min|S|={t['min_cardinality']} tam küme Δ: aday {d_zero[-1]:.2f} / rastgele {d_rand[-1]:.2f} "
              f"(gürültü {max(noise):.3g})")

    peak = torch.cuda.max_memory_allocated() / 2**30 if cfg["device"] == "cuda" else None
    env = env_record(model, cfg["model"], seed, {"peak_gpu_gib": peak, "elapsed_s": round(time.time() - t0, 1),
                                                "config": cfg})
    env["dtype"] = cfg["dtype"] + (" (lm_head float32)" if cfg.get("lm_head_fp32", True) else "")
    summary = summarize(per_example, rejected, cfg)
    (run_dir / "per_example.json").write_text(json.dumps(per_example, indent=1, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"env": env, **summary}, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(run_dir, env, summary, per_example)
    print(f"bitti: {run_dir}  (tepe GPU {peak and round(peak, 2)} GiB, {env['elapsed_s']} sn)")


def summarize(per_example, rejected, cfg) -> dict:
    out = {"n_accepted": len(per_example), "n_rejected": len(rejected),
           "rejected": [{"id": r["id"], "m0": r["m0"]} for r in rejected]}
    for frac in cfg["tau_fracs"]:
        key = f"tau{int(frac * 100)}"
        for cond in ("zero", "mean", "random", "hiact"):
            n_min = [len(e[key][cond]["minimal"]) for e in per_example]
            out[f"{key}_{cond}_n_minimal"] = {"values": n_min, "median_ci": bootstrap_ci(n_min),
                                              "share_ge2": float(np.mean([n >= 2 for n in n_min]))}
    for cond in ("zero", "mean", "random", "hiact"):
        rel = [e["full_set_delta"][cond] / e["m0"] for e in per_example]
        out[f"full_set_rel_drop_{cond}"] = {"values": rel, "median_ci": bootstrap_ci(rel)}
    out["noise_floor_max"] = max(e["noise_floor"] for e in per_example) if per_example else None
    return out


def write_md(run_dir, env, s, per_example) -> None:
    L = [f"# Pilot sonucu — {run_dir.name}", "",
         f"Model `{env['model']}` (rev {env['revision']}), {env['dtype']}, {env['gpu']}, torch {env['torch']}, "
         f"transformers {env['transformers']}, seed {env['seed']}, süre {env['elapsed_s']} sn, tepe GPU {env['peak_gpu_gib']} GiB.", "",
         f"Kabul edilen örnek {s['n_accepted']}, dışlanan {s['n_rejected']} (müdahalesiz model yanlış ya da marj eşiği altında).",
         f"Gürültü tabanı (aynı toplu çağrıdaki S=∅ marjı ile tekil müdahalesiz marj farkı, en büyük): {s['noise_floor_max']:.3g} logit.", "",
         "## Minimal etkili küme sayısı (örnek başına, medyan [%95 bootstrap GA]; ≥2 küme bulunan örnek oranı)", "",
         "| τ | aday zero | aday mean | rastgele | yüksek akt. |", "|---|---|---|---|---|"]
    for frac in (30, 50, 70):
        cells = []
        for cond in ("zero", "mean", "random", "hiact"):
            v = s.get(f"tau{frac}_{cond}_n_minimal")
            cells.append(f"{v['median_ci'][0]:.1f} [{v['median_ci'][1]:.1f}, {v['median_ci'][2]:.1f}]; {v['share_ge2']:.0%}" if v else "—")
        L.append(f"| %{frac} | " + " | ".join(cells) + " |")
    L += ["", "## 12 nöronun hepsi kapalıyken göreli marj düşüşü Δ/m0 (medyan [%95 GA])", ""]
    for cond in ("zero", "mean", "random", "hiact"):
        v = s[f"full_set_rel_drop_{cond}"]["median_ci"]
        L.append(f"- {cond}: {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]")
    L += ["", "## Örnek başına (τ = %50, zero)", "", "| örnek | m0 | minimal küme | min |S| | Jaccard | |I| medyan |", "|---|---|---|---|---|---|"]
    for e in per_example:
        t = e["tau50"]["zero"]
        j = e["jaccard_zero_tau50"]
        L.append(f"| {e['id']} | {e['m0']:.2f} | {len(t['minimal'])} | {t['min_cardinality']} | "
                 f"{'—' if j is None else f'{j:.2f}'} | {e['interaction_abs_median']:.3f} |")
    L += ["", "## Kademeli kapatma: kaç nöron kapalı kalınca etki eşiğin (τ=%50) altına düşüyor", "",
          "| örnek | zayıftan geri açma | güçlüden geri açma | 20 rastgele sıra (min–medyan–maks) |", "|---|---|---|---|"]
    for e in per_example:
        bp = e["break_point_tau50"]
        rnd = [v for n, v in bp.items() if n.startswith("rand") and v is not None]
        rtxt = f"{min(rnd)}–{int(np.median(rnd))}–{max(rnd)}" if rnd else "—"
        L.append(f"| {e['id']} | {bp['weak_first']} | {bp['strong_first']} | {rtxt} |")
    L += ["", "Not: pilot; tek görev (başkent), tek yazım, 2 katman. Hipotez doğrulaması ayrı test bölümünde yapılacak."]
    (run_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "pilot.yaml"))
