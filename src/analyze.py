"""Minimal kümeler, Jaccard örtüşmesi, ikili etkileşim, bootstrap GA."""
from __future__ import annotations

import numpy as np


def minimal_sets(delta: np.ndarray, k: int, tau: float) -> dict:
    """delta: [2^k], indeks = alt küme bit maskesi. Etkili: Δ(S) ≥ tau.

    Dahil etme bakımından minimal: S etkili ve S'nin HİÇBİR öz alt kümesi etkili değil. Etki monoton olmak
    zorunda değil ({a} etkili, {a,b} etkisiz, {a,b,c} etkili olabilir); bu yüzden yalnız bir nöron çıkarmak yetmez.
    En küçük kardinalite: etkili kümeler arasında en az nöronlu olanlar.
    """
    eff = delta >= tau
    eff[0] = False
    masks = np.arange(2 ** k)
    sizes = np.array([bin(int(s)).count("1") for s in masks])
    # sub[S] = S'nin kendisi ya da bir alt kümesi etkili mi (alt kümeler üzerinden OR; O(k·2^k))
    sub = eff.copy()
    for i in range(k):
        bit = 1 << i
        has = (masks & bit) != 0
        sub[has] |= sub[masks[has] ^ bit]
    minimal = []
    for s in masks[eff]:
        if not any(sub[s ^ (1 << i)] for i in range(k) if s >> i & 1):
            minimal.append(int(s))
    eff_sizes = sizes[eff & (masks != 0)]
    min_card = int(eff_sizes.min()) if eff_sizes.size else None
    return {
        "n_effective": int((eff & (masks != 0)).sum()),
        "minimal": minimal,
        "minimal_sizes": [int(sizes[s]) for s in minimal],
        "min_cardinality": min_card,
        "n_min_cardinality": int((eff_sizes == min_card).sum()) if min_card else 0,
        "full_set_effective": bool(eff[2 ** k - 1]),
    }


def additivity(delta: np.ndarray, k: int) -> dict:
    """Δ(S) ile tekli etkilerin toplamı A(S) arasındaki ilişki (S boş olmayan bütün kümeler)."""
    singles = np.array([delta[1 << i] for i in range(k)])
    masks = np.arange(1, 2 ** k)
    bits = ((masks[:, None] >> np.arange(k)) & 1).astype(float)
    A = bits @ singles
    D = delta[masks]
    slope = float(np.dot(A, D) / np.dot(A, A)) if np.dot(A, A) else float("nan")
    ss_res = float(((D - slope * A) ** 2).sum())
    ss_tot = float(((D - D.mean()) ** 2).sum())
    return {"slope_through_origin": slope, "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
            "full_over_sum": float(delta[-1] / singles.sum()) if singles.sum() else float("nan"),
            "max_abs_residual": float(np.abs(D - A).max())}


def uniqueness(delta: np.ndarray, k: int, q: float) -> dict:
    """Tam grubun etkisinin q oranına göre minimal kümeler, ortak nöronsuz çiftler, en güçlü nöronsuz ulaşılabilirlik."""
    full = delta[-1]
    ms = minimal_sets(delta, k, q * full)
    top = int(np.argmax([delta[1 << i] for i in range(k)]))
    mins = ms["minimal"]
    disjoint = [(a, b) for i, a in enumerate(mins) for b in mins[i + 1:] if a & b == 0]
    without_top = [s for s in range(2 ** k) if not s >> top & 1]
    return {**ms, "q": q, "top_neuron_index": top,
            "share_with_top": float(np.mean([bool(s >> top & 1) for s in mins])) if mins else None,
            "n_disjoint_pairs": len(disjoint), "disjoint_example": disjoint[0] if disjoint else None,
            "max_without_top_rel": float(delta[without_top].max() / full) if full else float("nan")}


def jaccard(a: int, b: int) -> float:
    inter, union = bin(a & b).count("1"), bin(a | b).count("1")
    return inter / union if union else 1.0


def mean_pairwise_jaccard(sets: list[int]) -> float | None:
    if len(sets) < 2:
        return None
    vals = [jaccard(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(vals))


def interaction_matrix(delta: np.ndarray, k: int) -> np.ndarray:
    """I(i,j) = Δ({i,j}) − Δ({i}) − Δ({j})."""
    m = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i + 1, k):
            m[i, j] = m[j, i] = delta[(1 << i) | (1 << j)] - delta[1 << i] - delta[1 << j]
    return m


def degradation_curve(delta: np.ndarray, order: list[int]) -> np.ndarray:
    """Kademeli kapatma: hepsi kapalıdan başlayıp `order` sırasıyla nöronları geri açar (12→11→…→0 kapalı).

    Dönüş [k+1]: i. eleman, i nöron geri açıldıktan sonraki Δ. Ek koşu gerekmez; 2^k tablosundan okunur.
    """
    k = len(order)
    mask = 2 ** k - 1
    out = [delta[mask]]
    for i in order:
        mask &= ~(1 << i)
        out.append(delta[mask])
    return np.array(out)


def break_point(curve: np.ndarray, tau: float) -> int | None:
    """Δ'nın ilk kez eşiğin altına düştüğü andaki kapalı nöron sayısı. Tam küme bile etkisizse None."""
    k = len(curve) - 1
    if curve[0] < tau:
        return None
    for n_on, d in enumerate(curve):
        if d < tau:
            return k - n_on
    return None


def bootstrap_ci(x, stat=np.median, n: int = 10000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(x, x.size)) for _ in range(n)])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
