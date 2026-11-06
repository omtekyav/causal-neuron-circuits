"""Yetkinlik kontrolü: modeller iki adımlı zincirleri çözebiliyor mu? (nöron analizi YOK)

Her soru için: açgözlü üretim (en fazla 10 token) → kabul edilen cevaplardan biri geçiyor mu; ve cevap konumunda
marj = logit(" "+gold ilk token) − logit(" "+alt ilk token) (bileşik soruda alt = köprü; Scar/Simba karşılığı).
Zincir "çözülmüş" sayılır: hop1, hop2 ve iki bileşik yazımın ikisi de doğru.
Kullanım: python run_competence.py configs/competence.yaml
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from run_pilot import preflight
from src.common import load_jsonl, load_model

ROOT = Path(__file__).parent


def first_tok(tok, word: str) -> int:
    return tok.encode(" " + word, add_special_tokens=False)[0]


@torch.inference_mode()
def ask(model, tok, prompt, gold, alt, accept, dev, max_new):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    g, a = first_tok(tok, gold), first_tok(tok, alt)
    probs = torch.softmax(logits, -1)
    ok = any(x in text.lower() for x in accept)
    return {"text": text, "correct": ok, "margin": float(logits[g] - logits[a]),
            "p_gold_first": float(probs[g]), "p_alt_first": float(probs[a]),
            "gold_tok": tok.decode([g]), "alt_tok": tok.decode([a])}


def main(cfg_path: str, only_model: str | None = None) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if only_model:   # her model ayrı süreçte: iki model aynı anda GPU'da olmaz
        cfg["models"] = [only_model]
    rows = load_jsonl(ROOT / cfg["data"])
    tag = only_model.split("/")[-1] if only_model else "hepsi"
    run_dir = ROOT / "results" / f"competence_{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    report = {}
    for mname in cfg["models"]:
        preflight(cfg)
        t0 = time.time()
        model, tok = load_model(mname, cfg["dtype"], cfg["device"], cfg.get("cache_dir"), False)
        res = {r["id"]: {**ask(model, tok, r["prompt"], r["gold"], r["alt"], r["accept"], cfg["device"], cfg["max_new"]),
                         "chain": r["chain"], "kind": r["kind"]} for r in rows}
        chains = sorted({r["chain"] for r in rows})
        solved = [c for c in chains if all(res[f"{c}_{k}"]["correct"] for k in ("hop1", "hop2", "comp1", "comp2"))]
        acc = {k: sum(res[f"{c}_{k}"]["correct"] for c in chains if f"{c}_{k}" in res)
               for k in ("hop1", "hop2", "comp1", "comp2", "comp3")}
        report[mname] = {"per_question": res, "solved_chains": solved, "accuracy": acc, "n_chains": len(chains),
                         "elapsed_s": round(time.time() - t0, 1),
                         "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if cfg["device"] == "cuda" else None}
        print(f"{mname}: doğru hop1 {acc['hop1']}/{len(chains)}, hop2 {acc['hop2']}, bileşik-1 {acc['comp1']}, "
              f"bileşik-2 {acc['comp2']}; tam çözülen zincir {len(solved)}: {solved}", flush=True)
        del model
        if cfg["device"] == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    (run_dir / "competence.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    L = [f"# Yetkinlik kontrolü — {run_dir.name}", "", "Nöron analizi yok. Açgözlü üretim, ilk satır; marj = logit(cevap) − logit(köprü) (bileşik).", ""]
    for mname, rep in report.items():
        a = rep["accuracy"]
        L += [f"## {mname}", "", f"hop1 {a['hop1']}/{rep['n_chains']}, hop2 {a['hop2']}, bileşik-1 {a['comp1']}, bileşik-2 {a['comp2']}; "
              f"tam çözülen {len(rep['solved_chains'])}: {rep['solved_chains']}", "",
              "| zincir | hop1 | hop2 | bileşik-1 (marj) | bileşik-2 (marj) |", "|---|---|---|---|---|"]
        for c in sorted({v["chain"] for v in rep["per_question"].values()}):
            q = lambda k: rep["per_question"][f"{c}_{k}"]
            cell = lambda k: f"{'✓' if q(k)['correct'] else '✗'} {q(k)['text'][:28]}"
            L.append(f"| {c} | {cell('hop1')} | {cell('hop2')} | {cell('comp1')} ({q('comp1')['margin']:.1f}) | "
                     f"{cell('comp2')} ({q('comp2')['margin']:.1f}) |")
        L.append("")
    (run_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"bitti: {run_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    model_arg = args[args.index("--model") + 1] if "--model" in args else None
    main(args[0] if args and not args[0].startswith("--") else str(ROOT / "configs" / "competence.yaml"), model_arg)
