"""Küçük, rastgele ağırlıklı Qwen3 ile çekirdek denetimler (model indirmeden, CPU, float32)."""
import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.analyze import interaction_matrix, minimal_sets
from src.capture import attribution, capture
from src.common import mlp_down
from src.intervene import ablate, all_subsets, run_masks
from src.score import margin

IDC, IDW = 5, 9


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    m = Qwen3ForCausalLM(cfg).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def ids():
    return torch.tensor([[1, 7, 3, 12, 40, 22, 8]])


def _m(model, ids):
    with torch.inference_mode():
        return margin(model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :], IDC, IDW)


def test_hooks_removed(model, ids):
    before = _m(model, ids)
    mask = torch.ones(1, 2, dtype=torch.bool)
    with ablate(model, {1: (torch.tensor([0, 1]), mask, torch.zeros(2))}, pos=6):
        changed = _m(model, ids)
    after = _m(model, ids)
    assert not torch.equal(before, changed)
    assert torch.equal(before, after)
    assert all(len(mlp_down(model, l)._forward_pre_hooks) == 0 for l in range(3))


def test_empty_set_equals_base(model, ids):
    cand = [(1, 3), (2, 10)]
    r = run_masks(model, ids, 6, cand, all_subsets(2), torch.zeros(2), IDC, IDW, batch=8)
    base = _m(model, ids)
    assert torch.allclose(r["margin"][0], base, atol=1e-5)          # S=∅ satırı
    assert torch.allclose(r["base_margin"], base.expand(4), atol=1e-5)


def test_batched_equals_single_run(model, ids):
    cand = [(0, 2), (1, 5), (2, 7)]
    masks = all_subsets(3)
    r = run_masks(model, ids, 6, cand, masks, torch.zeros(3), IDC, IDW, batch=3)   # küçük toplu: parçalara bölünür
    for s in range(8):
        specs = {}
        for l in {c[0] for c in cand}:
            cols = [k for k, c in enumerate(cand) if c[0] == l]
            specs[l] = (torch.tensor([cand[k][1] for k in cols]), masks[s:s + 1, cols], torch.zeros(len(cols)))
        with ablate(model, specs, pos=6):
            assert torch.allclose(_m(model, ids), r["margin"][s:s + 1], atol=1e-5)


def test_cached_equals_full(model, ids):
    """Önbellekli son-token yolu, tam ileri geçişle aynı Δ'ları verir."""
    from src.intervene import prefix_cache, run_masks_cached
    cand = [(0, 2), (1, 5), (2, 7), (2, 11)]
    masks = all_subsets(4)
    full = run_masks(model, ids, 6, cand, masks, torch.zeros(4), IDC, IDW, batch=5)
    cached = run_masks_cached(model, prefix_cache(model, ids), ids[:, -1:], cand, masks, torch.zeros(4), IDC, IDW, batch=5)
    assert torch.allclose(full["margin"], cached["margin"], atol=1e-5)
    assert torch.allclose(full["base_margin"], cached["base_margin"], atol=1e-5)


def test_neuron_is_intermediate_activation(model):
    """Tek konumlu girdide a_i'yi sıfırlamak, down_proj'un i. sütununu sıfırlamakla aynıdır."""
    one = torch.tensor([[17]])
    with ablate(model, {1: (torch.tensor([4]), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1))}, pos=0):
        hooked = _m(model, one)
    w = mlp_down(model, 1).weight
    saved = w[:, 4].clone()
    w[:, 4] = 0
    try:
        weight_zeroed = _m(model, one)
    finally:
        w[:, 4] = saved
    assert torch.allclose(hooked, weight_zeroed, atol=1e-6)


def test_only_target_position(model, ids):
    """Hedef konumdan önceki konumlara müdahale edilmez: son konum dışındaki aktivasyonlar aynı kalır."""
    store = {}
    h = mlp_down(model, 2).register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].clone()))
    try:
        with ablate(model, {1: (torch.tensor([0, 1, 2]), torch.ones(1, 3, dtype=torch.bool), torch.zeros(3))}, pos=6):
            _m(model, ids)
        abl = store["x"]
        _m(model, ids)
        base = store["x"]
    finally:
        h.remove()
    assert torch.allclose(abl[0, :6], base[0, :6], atol=1e-6)
    assert not torch.allclose(abl[0, 6], base[0, 6])


def test_capture_and_attribution_shapes(model, ids):
    a = capture(model, ids, [0, 2], 6)
    g = attribution(model, ids, [0, 2], 6, IDC, IDW)
    assert a[0].shape == (48,) and g[2].shape == (48,)
    assert torch.isfinite(g[0]).all() and g[0].abs().sum() > 0
    assert all(len(mlp_down(model, l)._forward_pre_hooks) == 0 for l in range(3))


def test_float32_head_matches(model, ids):
    """float32 kopya baş, aynı logitleri verir; özgün (bağlı) ağırlığa dokunmaz."""
    from src.common import Float32Head
    with torch.inference_mode():
        h = model.model(input_ids=ids).last_hidden_state
        ref = model.lm_head(h)
        new = Float32Head(model.lm_head)(h)
    assert new.dtype == torch.float32 and torch.allclose(ref, new, atol=1e-6)


def test_minimal_sets_synthetic():
    # k=3: {0} tek başına etkili; {1,2} birlikte etkili, ayrı ayrı değil → iki minimal küme
    k, delta = 3, np.zeros(8)
    for s in range(8):
        if s & 1 or (s & 6) == 6:
            delta[s] = 1.0
    r = minimal_sets(delta, k, tau=0.5)
    assert sorted(r["minimal"]) == [0b001, 0b110]
    assert r["min_cardinality"] == 1 and r["n_min_cardinality"] == 1
    inter = interaction_matrix(delta, k)
    assert inter[1, 2] == pytest.approx(1.0) and inter[0, 1] == pytest.approx(0.0)


def test_minimal_sets_non_monotone():
    """{a} etkili, ikililer etkisiz, {a,b,c} etkili: {a,b,c} minimal DEĞİL (öz alt kümesi {a} etkili)."""
    k, delta = 3, np.zeros(8)
    delta[0b001] = 1.0
    delta[0b111] = 1.0
    r = minimal_sets(delta, k, tau=0.5)
    assert r["minimal"] == [0b001]


def test_additivity_and_uniqueness():
    from src.analyze import additivity, uniqueness
    k = 4
    singles = np.array([1.0, 0.5, 0.5, 0.0])
    delta = np.array([sum(singles[i] for i in range(k) if s >> i & 1) for s in range(2 ** k)])
    a = additivity(delta, k)
    assert a["slope_through_origin"] == pytest.approx(1.0) and a["r2"] == pytest.approx(1.0)
    u = uniqueness(delta, k, q=0.5)          # tam 2.0 → eşik 1.0: {0} ve {1,2} minimal, ortak nöronsuz
    assert sorted(u["minimal"]) == [0b0001, 0b0110] and u["n_disjoint_pairs"] == 1
    assert u["top_neuron_index"] == 0 and u["max_without_top_rel"] == pytest.approx(0.5)


def test_degradation_curve():
    from src.analyze import break_point, degradation_curve
    k, delta = 3, np.zeros(8)
    for s in range(8):
        if s & 1 or (s & 6) == 6:
            delta[s] = 1.0
    # 0'ı önce geri aç: {1,2} kalır (etkili) → 2 geri açılınca {1} (etkisiz): kırılma 1 kapalıyken
    c = degradation_curve(delta, [0, 2, 1])
    assert list(c) == [1.0, 1.0, 0.0, 0.0]
    assert break_point(c, 0.5) == 1
    assert break_point(degradation_curve(np.zeros(8), [0, 1, 2]), 0.5) is None
