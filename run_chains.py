"""Adım 4 — ana deney: Qwen3-1.7B, iki adımlı zincirler (tanımlar: docs/adim4_zincir_tanimlar.md).

Zincir başına iki 12'li aday grubu (28 katmandan): Z (zero ablation etkisine göre) ve P (hop1 sorusundan yama
etkisine göre). 4.096 alt kümenin hepsi; kontroller; bf16 gürültü tabanı; müdahaleli üretim; aynı zincirin
başka yazımlarına ve diğer zincirlere transfer. Seçim yalnız bileşik yazım 1'de.

Kullanım: python run_chains.py configs/chains.yaml [--n 1]
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

from run_competence import ask, first_tok
from run_pilot import preflight
from run_pilot1b import spearman
from src.analyze import additivity, bootstrap_ci, interaction_matrix, uniqueness
from src.capture import act_grad, capture
from src.common import env_record, load_jsonl, load_model
from src.intervene import all_subsets, generate_with, prefix_cache, run_masks, run_masks_cached

ROOT = Path(__file__).parent
K, QS = 12, (0.3, 0.5, 0.7)


def jdump(obj) -> str:
    """numpy sayıları (float32 vb.) da yazılır."""
    return json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def enc(tok, p, dev):
    return tok(p, return_tensors="pt").input_ids.to(dev)


def vec(src: dict, cand) -> torch.Tensor:
    return torch.tensor([float(src[l][i]) for l, i in cand])


def pick(est_flat: torch.Tensor, nn: int, real_fn, pool: int, n_single: int):
    """Tahmine göre en iyi `pool` → gerçek tekli etki → n_single en etkili + tahmin sırasıyla kalanlar."""
    top = torch.topk(est_flat, pool).indices.tolist()
    cands = [(i // nn, i % nn) for i in top]
    real = real_fn(cands)
    strong = sorted(cands, key=lambda c: -real[c])[:n_single]
    rest = [c for c in cands if c not in strong][:K - n_single]
    return strong + rest, real


def layer_matched_random(cand, nn, rng):
    out = []
    for l in sorted({c[0] for c in cand}):
        n = sum(1 for c in cand if c[0] == l)
        out += [(l, int(i)) for i in rng.choice(nn, n, replace=False)]
    return out


def main(cfg_path: str, n_override: int | None, model_override: str | None = None) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if model_override:   # ikinci model ailesi: protokol aynı, yalnız model değişir
        cfg["model"] = model_override
    dev, b, seed = cfg["device"], cfg["batch"], cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    preflight(cfg)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    model, tok = load_model(cfg["model"], cfg["dtype"], dev, cfg.get("cache_dir"), cfg.get("lm_head_fp32", False))
    nn, layers = model.config.intermediate_size, list(range(model.config.num_hidden_layers))
    rows = {r["id"]: r for r in load_jsonl(ROOT / cfg["data"])}
    chains = sorted({r["chain"] for r in rows.values()})
    tag = cfg["model"].split("/")[-1]
    run_dir = ROOT / "results" / f"chains_{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    # 1) Dahil etme (çalışma içinde yeniden): hop1, hop2, bileşik-1 doğru ve m0 ≥ eşik
    incl, excluded = [], {}
    for c in chains:
        chk = {k: ask(model, tok, rows[f"{c}_{k}"]["prompt"], rows[f"{c}_{k}"]["gold"], rows[f"{c}_{k}"]["alt"],
                      rows[f"{c}_{k}"]["accept"], dev, 10) for k in ("hop1", "hop2", "comp1")}
        ok = all(v["correct"] for v in chk.values()) and chk["comp1"]["margin"] >= cfg["min_margin"]
        (incl.append(c) if ok else excluded.__setitem__(c, {k: [v["text"], round(v["margin"], 2)] for k, v in chk.items()}))
    if n_override:
        incl = incl[:n_override]
    log(f"dahil {len(incl)}: {incl}; dışlanan {len(excluded)}")

    masks = all_subsets(K)
    acts_comp1 = {c: capture(model, enc(tok, rows[f"{c}_comp1"]["prompt"], dev), layers, -1) for c in incl}
    acts_hop1 = {c: capture(model, enc(tok, rows[f"{c}_hop1"]["prompt"], dev), layers, -1) for c in incl}
    out = []
    groups_by_chain = {}
    for ci, c in enumerate(incl):
        r = rows[f"{c}_comp1"]
        other = incl[(ci + 1) % len(incl)]
        ids = enc(tok, r["prompt"], dev)
        pos = ids.shape[1] - 1
        ida, idb = first_tok(tok, r["answer"]), first_tok(tok, r["bridge"])
        cache, last = prefix_cache(model, ids), ids[:, -1:]
        run = lambda cand, rep, m=masks: run_masks_cached(model, cache, last, cand, m, rep, ida, idb, b)
        a_t, g = act_grad(model, ids, layers, pos, ida, idb)
        m0 = float(run([(0, 0)], torch.zeros(1), torch.zeros(1, 1, dtype=torch.bool))["base_margin"][0])

        def real_single(src):
            def fn(cands):
                rep = torch.zeros(len(cands)) if src is None else vec(src, cands)
                rr = run(cands, rep, torch.eye(len(cands), dtype=torch.bool))
                return dict(zip(cands, (rr["base_margin"] - rr["margin"]).tolist()))
            return fn

        est_z = torch.cat([a_t[l] * g[l] for l in layers])
        est_p = torch.cat([(a_t[l] - acts_hop1[c][l]) * g[l] for l in layers])   # Codex: (a_hedef − a_kaynak)·∂m/∂a
        Z, z_real = pick(est_z, nn, real_single(None), cfg["pool"], cfg["n_single"])
        P, p_real = pick(est_p, nn, real_single(acts_hop1[c]), cfg["pool"], cfg["n_single"])
        groups_by_chain[c] = {"Z": Z, "P": P}
        rz, rp = layer_matched_random(Z, nn, rng), layer_matched_random(P, nn, rng)
        conds = {"Z_zero": (Z, torch.zeros(K)), "Z_resample": (Z, vec(acts_comp1[other], Z)), "Z_random": (rz, torch.zeros(K)),
                 "P_hop1": (P, vec(acts_hop1[c], P)), "P_other": (P, vec(acts_comp1[other], P)),
                 "P_random": (rp, vec(acts_hop1[c], rp))}
        res = {"chain": c, "other": other, "m0": m0, "answer": r["answer"], "bridge": r["bridge"],
               "Z": [{"layer": l, "neuron": i, "single": z_real[(l, i)]} for l, i in Z],
               "P": [{"layer": l, "neuron": i, "single": p_real[(l, i)]} for l, i in P],
               "overlap_ZP": len(set(Z) & set(P)), "cond": {}}
        deltas = {}
        for name, (cand, rep) in conds.items():
            rr = run(cand, rep)
            d = (rr["base_margin"] - rr["margin"]).numpy()
            deltas[name] = d
            res["cond"][name] = {"full_rel": float(d[-1] / m0), "full_flipped": bool(rr["flipped"][-1]),
                                 "full_top1_is_bridge": bool(rr["top1"][-1] == idb),
                                 "share_flipped_all_subsets": float(rr["flipped"].float().mean())}
        # logit ayrışması: tam grup için ayrı ölçüm (S=∅ ile aynı toplu çağrıda)
        for name in ("Z_zero", "P_hop1"):
            cand, rep = conds[name]
            rr = run(cand, rep, torch.ones(1, K, dtype=torch.bool))
            base = run(cand, rep, torch.zeros(1, K, dtype=torch.bool))
            res["cond"][name]["d_logit_answer"] = float(rr["logit_correct"][0] - base["logit_correct"][0])
            res["cond"][name]["d_logit_bridge"] = float(rr["logit_wrong"][0] - base["logit_wrong"][0])

        # 2) bf16 gürültü tabanı: aynı kümeler toplu boyut 2 ile tam ileri geçişte
        sel = torch.cat([torch.zeros(1, K, dtype=torch.bool), torch.ones(1, K, dtype=torch.bool), torch.eye(K, dtype=torch.bool),
                         masks[torch.from_numpy(rng.choice(4096, cfg["noise_sets"] - 2 - K, replace=False))]])
        sel_idx = [int((s.long() << torch.arange(K)).sum()) for s in sel]
        noise = {}
        for name in ("Z_zero", "P_hop1"):
            cand, rep = conds[name]
            f = run_masks(model, ids, pos, cand, sel, rep, ida, idb, 2)
            noise[name] = float(np.abs((f["base_margin"] - f["margin"]).numpy() - deltas[name][sel_idx]).max())
        res["noise"] = noise

        # 3) Analizler
        res["analysis"] = {}
        for name in ("Z_zero", "Z_resample", "P_hop1", "P_other"):
            d = deltas[name]
            nz = noise["Z_zero" if name.startswith("Z") else "P_hop1"]
            interpretable = abs(d[-1]) >= 3 * nz and d[-1] > 0
            inter = interaction_matrix(d, K)
            iu = np.triu_indices(K, 1)
            thr = max(0.05 * abs(d[-1]), 3 * nz)
            res["analysis"][name] = {
                "interpretable": bool(interpretable), "additivity": additivity(d, K),
                "interaction": {"threshold": thr, "n_synergy": int((inter[iu] > thr).sum()),
                                "n_redundant": int((inter[iu] < -thr).sum()), "median_abs": float(np.median(np.abs(inter[iu])))},
                "uniqueness": {f"q{int(q * 100)}": {k2: v2 for k2, v2 in uniqueness(d, K, q).items() if k2 != "minimal"}
                               for q in QS} if interpretable else None}

        # 4) Müdahaleli üretim: tam grup + en küçük q50 kümeleri
        res["generation"] = {}
        for name in ("Z_zero", "P_hop1"):
            cand, rep = conds[name]
            gens = {"none": generate_with(model, tok, ids, cand, torch.zeros(K, dtype=torch.bool), rep),
                    "full": generate_with(model, tok, ids, cand, torch.ones(K, dtype=torch.bool), rep)}
            u = uniqueness(deltas[name], K, 0.5)
            for j, s in enumerate(sorted(u["minimal"], key=lambda s: bin(s).count("1"))[:cfg["gen_sets"]]):
                gens[f"min{j}_{bin(s).count('1')}n"] = generate_with(model, tok, ids, cand, masks[s], rep)
            res["generation"][name] = {k2: {"text": v2, "starts_with_bridge": v2.lower().startswith(r["bridge"].lower()),
                                            "has_answer": any(a in v2.lower() for a in r["accept"])} for k2, v2 in gens.items()}

        # 5) Transfer: aynı zincir yazım 2–3 (4.096 küme), diğer zincirler (tam grup)
        res["transfer"] = {}
        for k in ("comp2", "comp3"):
            v = rows[f"{c}_{k}"]
            vids = enc(tok, v["prompt"], dev)
            vc = prefix_cache(model, vids)
            vrun = lambda cand, rep: run_masks_cached(model, vc, vids[:, -1:], cand, masks, rep, ida, idb, b)
            entry = {}
            for name in ("Z_zero", "P_hop1"):
                cand, rep = conds[name]
                rr = vrun(cand, rep)
                vd = (rr["base_margin"] - rr["margin"]).numpy()
                vm0 = float(rr["base_margin"][0])
                entry[name] = {"m0": vm0, "valid": vm0 >= cfg["min_margin"], "full_rel": float(vd[-1] / vm0) if vm0 else None,
                               "spearman": spearman(deltas[name][1:], vd[1:]), "full_flipped": bool(rr["flipped"][-1])}
            res["transfer"][k] = entry
        out.append(res)
        z, p = res["cond"], res["analysis"]
        log(f"[{ci + 1}/{len(incl)}] {c} m0={m0:.2f} | Z zero {z['Z_zero']['full_rel']:.2f} resample {z['Z_resample']['full_rel']:.2f} "
            f"rastgele {z['Z_random']['full_rel']:.2f} | P hop1 {z['P_hop1']['full_rel']:.2f} başka-zincir {z['P_other']['full_rel']:.2f} "
            f"rastgele {z['P_random']['full_rel']:.2f} | gürültü {noise['Z_zero']:.3f}/{noise['P_hop1']:.3f} | "
            f"üretim P-tam: {res['generation']['P_hop1']['full']['text'][:25]!r}")
        np.savez_compressed(run_dir / f"{c}.npz", **{k2: v2.astype(np.float32) for k2, v2 in deltas.items()})
        (run_dir / f"{c}.json").write_text(jdump(res), encoding="utf-8")   # zincir biter bitmez

    # 6) Zincirler arası transfer (tam grup): A zincirinin grubu, B zincirinin bileşik sorusunda
    cross = []
    for c in incl:
        for c2 in incl:
            if c2 == c:
                continue
            r2 = rows[f"{c2}_comp1"]
            ids2 = enc(tok, r2["prompt"], dev)
            ia2, ib2 = first_tok(tok, r2["answer"]), first_tok(tok, r2["bridge"])
            cc = prefix_cache(model, ids2)
            full = torch.ones(1, K, dtype=torch.bool)
            Zc, Pc = groups_by_chain[c]["Z"], groups_by_chain[c]["P"]
            rz = run_masks_cached(model, cc, ids2[:, -1:], Zc, full, torch.zeros(K), ia2, ib2, b)
            rp = run_masks_cached(model, cc, ids2[:, -1:], Pc, full, vec(acts_hop1[c2], Pc), ia2, ib2, b)
            m02 = float(rz["base_margin"][0])
            cross.append({"group_of": c, "applied_to": c2, "m0": m02,
                          "Z_zero_rel": float((rz["base_margin"][0] - rz["margin"][0]) / m02),
                          "P_hop1_rel": float((rp["base_margin"][0] - rp["margin"][0]) / m02)})
    log(f"zincirler arası: {len(cross)} çift")

    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
    env = env_record(model, cfg["model"], seed, {"peak_gpu_gib": peak, "elapsed_s": round(time.time() - t0, 1), "config": cfg})
    (run_dir / "results.json").write_text(jdump({"env": env, "included": incl, "excluded": excluded, "chains": out,
                                                 "cross": cross}), encoding="utf-8")
    write_md(run_dir, env, incl, excluded, out, cross)
    log(f"bitti: {run_dir} (tepe GPU {peak and round(peak, 2)} GiB)")


def write_md(run_dir, env, incl, excluded, out, cross) -> None:
    def ci(v):
        v = [x for x in v if x is not None]
        if not v:
            return "—"
        m, lo, hi = bootstrap_ci(v)
        return f"{m:.2f} [{lo:.2f}, {hi:.2f}] (n={len(v)})"
    L = [f"# Adım 4 — zincirler — {run_dir.name}", "",
         f"`{env['model']}` rev {env['revision']}, {env['dtype']}, {env['gpu']}, süre {env['elapsed_s']} sn, tepe GPU "
         f"{env['peak_gpu_gib']} GiB. Tanımlar: docs/adim4_zincir_tanimlar.md. Medyan [%95 bootstrap GA], birim = zincir.", "",
         f"Dahil {len(incl)}: {incl}", f"Dışlanan: {excluded}", "",
         "## Tam grup etkisi Δ/m0 (m = logit cevap − logit köprü)", ""]
    for name, lab in (("Z_zero", "Z grubu, zero"), ("Z_resample", "Z grubu, başka zincirden resample"), ("Z_random", "rastgele 12, zero"),
                      ("P_hop1", "P grubu, hop1 yaması"), ("P_other", "P grubu, başka zincirden yama"), ("P_random", "rastgele 12, hop1 yaması")):
        L.append(f"- {lab}: {ci([e['cond'][name]['full_rel'] for e in out])}; cevap değişti (marj<0) "
                 f"{sum(e['cond'][name]['full_flipped'] for e in out)}/{len(out)}; en olası token köprü "
                 f"{sum(e['cond'][name]['full_top1_is_bridge'] for e in out)}/{len(out)}")
    diff = [e["cond"]["P_hop1"]["full_rel"] - e["cond"]["P_random"]["full_rel"] for e in out]
    L.append(f"- P hop1 − rastgele (zincir içi fark): {ci(diff)}")
    L += ["", "## Logit ayrışması (tam grup; cevap ve köprü logitlerindeki değişim)", "",
          "| zincir | cevap→köprü | Z: Δlogit cevap | Z: Δlogit köprü | P: Δlogit cevap | P: Δlogit köprü | Z∩P |", "|---|---|---|---|---|---|---|"]
    for e in out:
        z, p = e["cond"]["Z_zero"], e["cond"]["P_hop1"]
        L.append(f"| {e['chain']} | {e['answer']}→{e['bridge']} | {z['d_logit_answer']:+.2f} | {z['d_logit_bridge']:+.2f} | "
                 f"{p['d_logit_answer']:+.2f} | {p['d_logit_bridge']:+.2f} | {e['overlap_ZP']} |")
    L += ["", "## Müdahaleli üretim (açgözlü, ilk satır)", "", "| zincir | normal | Z tam | P tam | P en küçük q50 kümeleri |", "|---|---|---|---|---|"]
    for e in out:
        gz, gp = e["generation"]["Z_zero"], e["generation"]["P_hop1"]
        mins = "; ".join(f"{k}: {v['text'][:18]}" for k, v in gp.items() if k.startswith("min"))
        L.append(f"| {e['chain']} | {gp['none']['text'][:22]} | {gz['full']['text'][:22]} | {gp['full']['text'][:22]} | {mins} |")
    L += ["", "## Toplamsallık, etkileşim, benzersizlik (yorumlanabilir olanlar: |Δ_tam| ≥ 3·gürültü)", "",
          "| zincir | koşul | gürültü | R² | eğim | sinerji/yedek | q50 minimal | ayrık çift | en güçlü olmadan max |", "|---|---|---|---|---|---|---|---|---|"]
    for e in out:
        for name in ("Z_zero", "P_hop1"):
            a = e["analysis"][name]
            nz = e["noise"][name]
            if not a["interpretable"]:
                L.append(f"| {e['chain']} | {name} | {nz:.3f} | yorumlanmaz (etki gürültü düzeyinde) | | | | | |")
                continue
            u = a["uniqueness"]["q50"]
            L.append(f"| {e['chain']} | {name} | {nz:.3f} | {a['additivity']['r2']:.3f} | {a['additivity']['slope_through_origin']:.2f} | "
                     f"{a['interaction']['n_synergy']}/{a['interaction']['n_redundant']} | {len(u['minimal_sizes'])} | "
                     f"{u['n_disjoint_pairs']} | {u['max_without_top_rel']:.2f} |")
    L += ["", "## Transfer: aynı zincirin başka yazımları (aynı grup, seçim yok)", "",
          "| zincir | yazım | Z: Δ/m0 | Z: Spearman | P: Δ/m0 | P: Spearman |", "|---|---|---|---|---|---|"]
    for e in out:
        for k, t in e["transfer"].items():
            z, p = t["Z_zero"], t["P_hop1"]
            fmt = lambda x: "geçersiz" if not x["valid"] else f"{x['full_rel']:.2f}"
            L.append(f"| {e['chain']} | {k} | {fmt(z)} | {z['spearman']:.2f} | {fmt(p)} | {p['spearman']:.2f} |")
    own = {e["chain"]: (e["cond"]["Z_zero"]["full_rel"], e["cond"]["P_hop1"]["full_rel"]) for e in out}
    L += ["", "## Zincirler arası transfer (A'nın grubu B'nin sorusunda; içeriğe özgü grup başka zincirde zayıf olmalı)", "",
          f"- Z (zero): kendi zinciri {ci([v[0] for v in own.values()])}; başka zincir {ci([x['Z_zero_rel'] for x in cross if x['m0'] > 0])}",
          f"- P (hop1 yaması, hedef zincirin hop1'inden): kendi zinciri {ci([v[1] for v in own.values()])}; başka zincir "
          f"{ci([x['P_hop1_rel'] for x in cross if x['m0'] > 0])}", "",
          "Pilot ölçekli (~10 zincir); tek model; bf16. Sonuçlar tahmin, kanıt dili kullanılmaz."]
    (run_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else None
    m = args[args.index("--model") + 1] if "--model" in args else None
    main(args[0] if args and not args[0].startswith("--") else str(ROOT / "configs" / "chains.yaml"), n, m)
