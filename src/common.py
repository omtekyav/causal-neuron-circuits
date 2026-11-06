"""Model yükleme, cevap tokenları ve ortam kaydı."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import torch


class Float32Head(torch.nn.Module):
    """lm_head'in float32 kopyası: gövde bf16 kalır, logitler (ve marj) float32 çözünürlükte hesaplanır.

    bf16 logitlerde adım ~0.06 (|logit|≈10); küçük Δ'lar sıfıra yuvarlanıyordu. Ağırlık bağlı (tied) olsa da
    kopya ayrıdır; gömme katmanına dokunulmaz.
    """

    def __init__(self, head: torch.nn.Linear):
        super().__init__()
        self.weight = torch.nn.Parameter(head.weight.detach().float().clone(), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.weight)


def load_model(name: str, dtype: str, device: str, cache_dir: str | None = None, lm_head_fp32: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), cache_dir=cache_dir).to(device)
    if lm_head_fp32:
        model.lm_head = Float32Head(model.lm_head)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def mlp_down(model, layer: int):
    """Nöron i = down_proj girdisinin i. bileşeni (MLP ara aktivasyonu, act(gate)·up)."""
    return model.model.layers[layer].mlp.down_proj


def n_neurons(model) -> int:
    return model.config.intermediate_size


def answer_ids(tok, labels=(" A", " B")) -> dict[str, int]:
    """Cevap etiketleri tek token olmalı; değilse çalışma durur."""
    out = {}
    for lab in labels:
        ids = tok.encode(lab, add_special_tokens=False)
        assert len(ids) == 1, f"{lab!r} tek token değil: {ids}"
        out[lab.strip()] = ids[0]
    return out


def env_record(model, model_name: str, seed: int, extra: dict | None = None) -> dict:
    import transformers

    rec = {
        "model": model_name,
        "revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype),
        "n_layers": model.config.num_hidden_layers,
        "intermediate_size": model.config.intermediate_size,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": seed,
    }
    rec.update(extra or {})
    return rec


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
