# Pilot sonucu — pilot_20261106_102928

Model `Qwen/Qwen3-0.6B-Base` (rev da87bfb608c14b7cf20ba1ce41287e8de496c0cd), bfloat16 (lm_head float32), NVIDIA L40S, torch 2.11.0+cu130, transformers 5.9.0, seed 20261106, süre 365.8 sn, tepe GPU 2.6051058769226074 GiB.

Kabul edilen örnek 10, dışlanan 0 (müdahalesiz model yanlış ya da marj eşiği altında).
Gürültü tabanı (aynı toplu çağrıdaki S=∅ marjı ile tekil müdahalesiz marj farkı, en büyük): 0.119 logit.

## Minimal etkili küme sayısı (örnek başına, medyan [%95 bootstrap GA]; ≥2 küme bulunan örnek oranı)

| τ | aday zero | aday mean | rastgele | yüksek akt. |
|---|---|---|---|---|
| %30 | 0.0 [0.0, 2.0]; 30% | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% |
| %50 | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% |
| %70 | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% | 0.0 [0.0, 0.0]; 0% |

## 12 nöronun hepsi kapalıyken göreli marj düşüşü Δ/m0 (medyan [%95 GA])

- zero: 0.27 [0.25, 0.31]
- mean: -0.00 [-0.01, 0.01]
- random: -0.00 [-0.01, 0.00]
- hiact: 0.01 [-0.01, 0.04]

## Örnek başına (τ = %50, zero)

| örnek | m0 | minimal küme | min |S| | Jaccard | |I| medyan |
|---|---|---|---|---|---|
| cap30_p0 | 4.59 | 0 | None | — | 0.037 |
| cap38_p0 | 4.85 | 0 | None | — | 0.030 |
| cap33_p0 | 6.59 | 0 | None | — | 0.037 |
| cap22_p0 | 4.02 | 0 | None | — | 0.032 |
| cap29_p0 | 6.84 | 0 | None | — | 0.026 |
| cap23_p0 | 4.55 | 0 | None | — | 0.029 |
| cap16_p0 | 5.18 | 0 | None | — | 0.049 |
| cap24_p0 | 4.70 | 0 | None | — | 0.027 |
| cap26_p0 | 4.58 | 0 | None | — | 0.025 |
| cap09_p0 | 5.78 | 0 | None | — | 0.039 |

## Kademeli kapatma: kaç nöron kapalı kalınca etki eşiğin (τ=%50) altına düşüyor

| örnek | zayıftan geri açma | güçlüden geri açma | 20 rastgele sıra (min–medyan–maks) |
|---|---|---|---|
| cap30_p0 | None | None | — |
| cap38_p0 | None | None | — |
| cap33_p0 | None | None | — |
| cap22_p0 | None | None | — |
| cap29_p0 | None | None | — |
| cap23_p0 | None | None | — |
| cap16_p0 | None | None | — |
| cap24_p0 | None | None | — |
| cap26_p0 | None | None | — |
| cap09_p0 | None | None | — |

Not: pilot; tek görev (başkent), tek yazım, 2 katman. Hipotez doğrulaması ayrı test bölümünde yapılacak.
