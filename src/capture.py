"""MLP ara aktivasyonlarını (down_proj girdisi) yakalama ve activation × gradient ataması."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .common import mlp_down
from .score import margin


@contextmanager
def _capture_hooks(model, layers, store: dict, keep_grad: bool):
    handles = []

    def make(layer):
        def hook(_module, args):
            x = args[0]
            if keep_grad:
                x.retain_grad()
            store[layer] = x
        return hook

    try:
        for layer in layers:
            handles.append(mlp_down(model, layer).register_forward_pre_hook(make(layer)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def capture(model, input_ids: torch.Tensor, layers, pos: int) -> dict[int, torch.Tensor]:
    """{katman: [n_neurons]} — tek girdi, tek konum (float32, CPU)."""
    store: dict = {}
    with _capture_hooks(model, layers, store, keep_grad=False):
        model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    return {l: store[l][0, pos].float().cpu() for l in layers}


def act_grad(model, input_ids: torch.Tensor, layers, pos: int, id_correct: int, id_wrong: int):
    """(a, ∂m/∂a) hedef konumda, katman başına float32 CPU. Atamalı yama tahmini için: (a_hedef − a_kaynak)·∂m/∂a."""
    store: dict = {}
    with torch.enable_grad():
        emb = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        with _capture_hooks(model, layers, store, keep_grad=True):
            out = model(inputs_embeds=emb, use_cache=False, logits_to_keep=1)
        margin(out.logits[:, -1, :], id_correct, id_wrong)[0].backward()
    return ({l: store[l][0, pos].detach().float().cpu() for l in layers},
            {l: store[l].grad[0, pos].float().cpu() for l in layers})


def attribution(model, input_ids: torch.Tensor, layers, pos: int, id_correct: int, id_wrong: int):
    """a_i · ∂m/∂a_i: sıfırlama etkisinin birinci derece tahmini (Δ ≈ a·∂m/∂a).

    Gradyan yalnız aktivasyonlarda; parametreler dondurulmuş. Akışın bütün katmanlardan geçmesi için
    girdi gömmeleri grad isteyen yaprak yapılır (bir katmanda detach edilmez).
    """
    store: dict = {}
    with torch.enable_grad():
        emb = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        with _capture_hooks(model, layers, store, keep_grad=True):
            out = model(inputs_embeds=emb, use_cache=False, logits_to_keep=1)
        m = margin(out.logits[:, -1, :], id_correct, id_wrong)[0]
        m.backward()
    res = {}
    for l in layers:
        a = store[l][0, pos].detach().float()
        g = store[l].grad[0, pos].float()
        res[l] = (a * g).cpu()
    return res
