"""Tekli ve grup ablation: seçilen MLP ara kanallarını yalnız hedef token konumunda değiştirir.

Toplu çalışma: aynı prompt B kez tekrarlanır; satır b'de `mask[b, k] = True` olan adaylar kapatılır.
Her toplu çağrının 0. satırı boş kümedir (S=∅); Δ bu satıra göre hesaplanır, böylece toplu işleme
kaynaklı sayısal fark Δ'ya girmez, ayrıca gürültü tabanı olarak raporlanır.
Hook'lar `finally` içinde kaldırılır.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .common import mlp_down
from .score import scores


@contextmanager
def ablate(model, specs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pos: int):
    """specs: {katman: (nöron_idx [k], mask [B, k] bool, yerine_değer [k])}."""
    handles = []

    def make(idx, mask, repl):
        def hook(_module, args):
            x = args[0]
            if x.shape[1] <= pos:   # üretimde sonraki adımlar (tek yeni token): müdahale yalnız prompt geçişinde
                return None
            sub = x[:, pos, idx]
            new = torch.where(mask, repl.to(x.dtype).unsqueeze(0).expand_as(sub), sub)
            x = x.clone()
            x[:, pos, idx] = new
            return (x,) + tuple(args[1:])
        return hook

    try:
        for layer, (idx, mask, repl) in specs.items():
            handles.append(mlp_down(model, layer).register_forward_pre_hook(make(idx, mask, repl)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def run_masks(model, input_ids: torch.Tensor, pos: int, cand: list[tuple[int, int]], masks: torch.Tensor,
              repl: torch.Tensor, id_correct: int, id_wrong: int, batch: int = 256) -> dict[str, torch.Tensor]:
    """cand: [(katman, nöron)] K aday; masks: [N, K] bool; repl: [K] yerine konacak değerler.

    Dönüş: her maske için margin, p_correct, flipped ve aynı toplu çağrıdaki S=∅ marjı (base_margin).
    """
    device = input_ids.device
    layers = sorted({l for l, _ in cand})
    cols = {l: [k for k, (ll, _) in enumerate(cand) if ll == l] for l in layers}
    idx = {l: torch.tensor([cand[k][1] for k in cols[l]], device=device) for l in layers}
    out = {"base_margin": []}
    for start in range(0, masks.shape[0], batch - 1):
        chunk = masks[start:start + batch - 1]
        full = torch.cat([torch.zeros(1, masks.shape[1], dtype=torch.bool), chunk]).to(device)
        specs = {l: (idx[l], full[:, cols[l]], repl[cols[l]].to(device)) for l in layers}
        ids = input_ids.expand(full.shape[0], -1)
        with ablate(model, specs, pos):
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        _collect(out, scores(logits, id_correct, id_wrong), chunk.shape[0])
    return {k: torch.cat(v) for k, v in out.items()}


def _collect(out: dict, s: dict, n: int) -> None:
    """S=∅ satırını (0.) ayırır; geri kalan satırların bütün skorlarını biriktirir."""
    for k, v in s.items():
        out.setdefault(k, []).append(v[1:].cpu())
    out["base_margin"].append(s["margin"][:1].expand(n).cpu())


@torch.inference_mode()
def prefix_cache(model, input_ids: torch.Tensor):
    """Son token hariç promptun KV önbelleği (bir kez hesaplanır)."""
    return model(input_ids=input_ids[:, :-1], use_cache=True).past_key_values


@torch.inference_mode()
def run_masks_cached(model, cache, last_id: torch.Tensor, cand: list[tuple[int, int]], masks: torch.Tensor,
                     repl: torch.Tensor, id_correct: int, id_wrong: int, batch: int = 128) -> dict[str, torch.Tensor]:
    """run_masks ile aynı sonuç, ama yalnız son token çalıştırılır (müdahale son konumda olduğundan önceki
    konumların KV değerleri değişmez). Her toplu çağrının 0. satırı yine S=∅.
    """
    import copy

    device = last_id.device
    layers = sorted({l for l, _ in cand})
    cols = {l: [k for k, (ll, _) in enumerate(cand) if ll == l] for l in layers}
    idx = {l: torch.tensor([cand[k][1] for k in cols[l]], device=device) for l in layers}
    out = {"base_margin": []}
    for start in range(0, masks.shape[0], batch - 1):
        chunk = masks[start:start + batch - 1]
        full = torch.cat([torch.zeros(1, masks.shape[1], dtype=torch.bool), chunk]).to(device)
        specs = {l: (idx[l], full[:, cols[l]], repl[cols[l]].to(device)) for l in layers}
        c = copy.deepcopy(cache)
        c.batch_repeat_interleave(full.shape[0])
        with ablate(model, specs, pos=0):
            logits = model(input_ids=last_id.expand(full.shape[0], 1), past_key_values=c, use_cache=True,
                           logits_to_keep=1).logits[:, -1, :]
        _collect(out, scores(logits, id_correct, id_wrong), chunk.shape[0])
    return {k: torch.cat(v) for k, v in out.items()}


@torch.inference_mode()
def generate_with(model, tok, input_ids: torch.Tensor, cand: list[tuple[int, int]], mask: torch.Tensor,
                  repl: torch.Tensor, max_new: int = 8) -> str:
    """Seçili kümeye (mask [K]) müdahale ederek açgözlü üretim; müdahale yalnız prompt geçişinin son konumunda."""
    device = input_ids.device
    pos = input_ids.shape[1] - 1
    layers = sorted({l for l, _ in cand})
    specs = {}
    for l in layers:
        cols = [k for k, (ll, _) in enumerate(cand) if ll == l]
        specs[l] = (torch.tensor([cand[k][1] for k in cols], device=device), mask[cols].unsqueeze(0).to(device),
                    repl[cols].to(device))
    with ablate(model, specs, pos):
        out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id,
                             attention_mask=torch.ones_like(input_ids))
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()


def all_subsets(k: int) -> torch.Tensor:
    """[2^k, k] bool; satır s, s'nin ikili açılımı (bit i → aday i)."""
    s = torch.arange(2 ** k).unsqueeze(1)
    return ((s >> torch.arange(k)) & 1).bool()
