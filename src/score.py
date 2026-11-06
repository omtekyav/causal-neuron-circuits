"""Skorlar: logit marjı (birincil), doğru cevap olasılığı, cevap değişimi."""
from __future__ import annotations

import torch


def margin(last_logits: torch.Tensor, id_correct: int, id_wrong: int) -> torch.Tensor:
    """m = logit(doğru) − logit(en güçlü yanlış). İki seçenekli görevde yanlış seçenek tektir."""
    return last_logits[:, id_correct].float() - last_logits[:, id_wrong].float()


def prob_correct(last_logits: torch.Tensor, id_correct: int) -> torch.Tensor:
    return torch.softmax(last_logits.float(), dim=-1)[:, id_correct]


def scores(last_logits: torch.Tensor, id_correct: int, id_wrong: int) -> dict[str, torch.Tensor]:
    m = margin(last_logits, id_correct, id_wrong)
    return {"margin": m, "p_correct": prob_correct(last_logits, id_correct), "flipped": m < 0,
            "logit_correct": last_logits[:, id_correct].float(), "logit_wrong": last_logits[:, id_wrong].float(),
            "top1": last_logits.argmax(-1)}
