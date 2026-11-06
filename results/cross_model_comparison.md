# Üç model ailesi — ana deney karşılaştırması

Yeni ölçüm yok; kayıtlı sonuçlardan. Müdahale: P grubu = hop1 yaması (12 nöron, son konum). "Çeviren" küme: Δ(S) ≥ m0 (marj ≤ 0, cevap köprüye döner). Medyan [%95 bootstrap GA], birim = zincir.

Ortak zincirler (7): ['apple', 'eiffel', 'lisbon', 'lk', 'madrid', 'nairobi', 'ninth']

## Her modelin kendi zincirleri

| | Qwen3-1.7B-Base | Llama-3.2-1B | gemma-3-1b-pt |
|---|---|---|---|
| zincir sayısı | 11 | 11 | 10 |
| **1. Yama cevabı çeviriyor (marj<0)** | 10/11 | 10/11 | 8/10 |
| en olası token köprü | 9/11 | 9/11 | 8/10 |
| üretim köprüyle başlıyor (Scar→Simba) | 9/11 | 9/11 | 8/10 |
| rastgele 12 nöron çeviriyor | 0/11 | 0/11 | 0/10 |
| yama Δ/m0 | 1.52 [1.23, 2.20] | 1.89 [1.35, 3.71] | 1.82 [1.16, 3.17] |
| rastgele Δ/m0 | 0.00 [-0.01, 0.01] | 0.00 [-0.01, 0.04] | 0.00 [-0.02, 0.01] |
| **2. Cevabı çeviren en küçük alt küme (nöron)** | [1, 2, 4, 4, 4, 4, 4, 5, 6, 6] | [1, 1, 1, 1, 2, 2, 2, 4, 5, 8] | [1, 1, 1, 2, 2, 2, 4, 6] |
| ortak nöronsuz iki çeviren alt küme olan zincir | 4/11 | 7/11 | 7/10 |
| **3. Aynı zincir, başka yazım: Δ/m0** | 1.48 [1.26, 1.60] | 1.77 [1.31, 2.01] | 1.77 [1.18, 2.33] |
| aynı zincir, başka yazım: Spearman | 0.98 [0.97, 0.99] | 0.98 [0.91, 0.98] | 0.97 [0.90, 0.98] |
| başka zincir: Δ/m0 | 0.18 [0.07, 0.34] | 0.09 [0.03, 0.26] | 0.11 [0.04, 0.26] |
| zero Δ/m0 / resample Δ/m0 | 1.31 [0.88, 2.89] / 1.37 [0.72, 2.70] | 2.04 [1.20, 3.26] / 1.41 [0.78, 2.25] | 2.12 [1.31, 2.96] / 1.90 [0.73, 2.57] |

### Zincir bazında: m0 ve ortak nöronsuz çeviren alt küme var mı

| model | zincir | m0 | çeviren alt küme | en küçük | ortak nöronsuz çift |
|---|---|---|---|---|---|
| Qwen3-1.7B-Base | apple | 4.62 | 0 | None | 0 |
| Qwen3-1.7B-Base | beatles | 1.25 | 4069 | 1 | 231 |
| Qwen3-1.7B-Base | eiffel | 8.12 | 1539 | 4 | 0 |
| Qwen3-1.7B-Base | lisbon | 9.50 | 1827 | 4 | 0 |
| Qwen3-1.7B-Base | lk | 1.88 | 3120 | 4 | 3119 |
| Qwen3-1.7B-Base | madrid | 8.12 | 1641 | 4 | 0 |
| Qwen3-1.7B-Base | mona | 6.50 | 1081 | 4 | 0 |
| Qwen3-1.7B-Base | nairobi | 6.75 | 1805 | 5 | 1 |
| Qwen3-1.7B-Base | ninth | 10.25 | 634 | 6 | 0 |
| Qwen3-1.7B-Base | pyramids | 3.38 | 3903 | 2 | 2021 |
| Qwen3-1.7B-Base | relativity | 9.62 | 395 | 6 | 0 |
| Llama-3.2-1B | apple | 4.94 | 0 | None | 0 |
| Llama-3.2-1B | eiffel | 2.38 | 3712 | 1 | 378 |
| Llama-3.2-1B | everest | 3.50 | 3300 | 1 | 155 |
| Llama-3.2-1B | lisbon | 7.12 | 1793 | 4 | 0 |
| Llama-3.2-1B | lk | 2.88 | 145 | 8 | 0 |
| Llama-3.2-1B | madrid | 5.75 | 2811 | 2 | 512 |
| Llama-3.2-1B | nairobi | 5.12 | 2891 | 2 | 988 |
| Llama-3.2-1B | ninth | 3.12 | 4074 | 2 | 1131 |
| Llama-3.2-1B | relativity | 7.50 | 1007 | 5 | 0 |
| Llama-3.2-1B | titanic | 1.94 | 4045 | 1 | 679 |
| Llama-3.2-1B | tokyo | 1.62 | 3907 | 1 | 1576 |
| gemma-3-1b-pt | apple | 7.00 | 0 | None | 0 |
| gemma-3-1b-pt | eiffel | 3.75 | 3429 | 2 | 1955 |
| gemma-3-1b-pt | everest | 3.88 | 2935 | 2 | 926 |
| gemma-3-1b-pt | lisbon | 6.88 | 2379 | 4 | 398 |
| gemma-3-1b-pt | lk | 2.75 | 560 | 6 | 0 |
| gemma-3-1b-pt | madrid | 2.88 | 3969 | 1 | 1324 |
| gemma-3-1b-pt | nairobi | 5.50 | 1984 | 2 | 79 |
| gemma-3-1b-pt | ninth | 2.25 | 4052 | 1 | 359 |
| gemma-3-1b-pt | pyramids | 1.25 | 4079 | 1 | 227 |
| gemma-3-1b-pt | sherlock | 7.56 | 0 | None | 0 |

Marj eğilimi (üç model birlikte): m0 < 4 olan 15 zincirin 13'inde, m0 ≥ 4 olan 17 zincirin 5'inde ortak nöronsuz çeviren alt küme var.

## Üç modelin ortak zincirleri

| | Qwen3-1.7B-Base | Llama-3.2-1B | gemma-3-1b-pt |
|---|---|---|---|
| zincir sayısı | 7 | 7 | 7 |
| **1. Yama cevabı çeviriyor (marj<0)** | 6/7 | 6/7 | 6/7 |
| en olası token köprü | 6/7 | 6/7 | 6/7 |
| üretim köprüyle başlıyor (Scar→Simba) | 6/7 | 6/7 | 6/7 |
| rastgele 12 nöron çeviriyor | 0/7 | 0/7 | 0/7 |
| yama Δ/m0 | 1.52 [1.23, 1.62] | 1.83 [1.17, 2.21] | 1.80 [1.23, 3.17] |
| rastgele Δ/m0 | 0.00 [0.00, 0.01] | 0.00 [-0.01, 0.05] | 0.00 [-0.02, 0.00] |
| **2. Cevabı çeviren en küçük alt küme (nöron)** | [4, 4, 4, 4, 5, 6] | [1, 2, 2, 2, 4, 8] | [1, 1, 2, 2, 4, 6] |
| ortak nöronsuz iki çeviren alt küme olan zincir | 2/7 | 4/7 | 5/7 |
| **3. Aynı zincir, başka yazım: Δ/m0** | 1.46 [1.21, 1.58] | 1.67 [1.07, 1.95] | 1.83 [1.27, 2.36] |
| aynı zincir, başka yazım: Spearman | 0.98 [0.94, 0.99] | 0.98 [0.90, 0.98] | 0.98 [0.93, 0.98] |
| başka zincir: Δ/m0 | 0.01 [0.00, 0.04] | 0.02 [0.01, 0.25] | 0.03 [0.00, 0.10] |
| zero Δ/m0 / resample Δ/m0 | 1.31 [0.88, 1.54] / 1.37 [0.72, 1.57] | 1.74 [1.20, 2.38] / 1.32 [0.65, 1.54] | 2.20 [1.60, 2.96] / 1.70 [0.73, 2.57] |

