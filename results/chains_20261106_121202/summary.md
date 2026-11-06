# Adım 4 — zincirler — chains_20261106_121202

`Qwen/Qwen3-1.7B-Base` rev ea980cb0a6c2ae4b936e82123acc929f1cec04c1, torch.bfloat16, NVIDIA L40S, süre 344.4 sn, tepe GPU 3.8282151222229004 GiB. Tanımlar: docs/adim4_zincir_tanimlar.md. Medyan [%95 bootstrap GA], birim = zincir.

Dahil 11: ['apple', 'beatles', 'eiffel', 'lisbon', 'lk', 'madrid', 'mona', 'nairobi', 'ninth', 'pyramids', 'relativity']
Dışlanan: {'alex': {'hop1': ['Aristotle', 6.5], 'hop2': ['Plato', 5.25], 'comp1': ['Aristotle', -4.88]}, 'everest': {'hop1': ['Nepal', 12.12], 'hop2': ['Kathmandu', 4.62], 'comp1': ['Lukla', 0.75]}, 'hamlet': {'hop1': ['Denmark', 10.0], 'hop2': ['Copenhagen', 9.62], 'comp1': ['Denmark', -1.88]}, 'hp': {'hop1': ['Voldemort', 4.0], 'hop2': ['Tom Riddle', 4.5], 'comp1': ['Severus Snape', -2.25]}, 'kangaroo': {'hop1': ['Australia', 17.62], 'hop2': ['Canberra', 9.12], 'comp1': ['Sydney', 2.12]}, 'romeo': {'hop1': ['William Shakespeare', 18.44], 'hop2': ['Stratford-upon-Avon', 9.0], 'comp1': ['Verona', -0.88]}, 'sherlock': {'hop1': ['Sir Arthur Conan Doyle', 3.75], 'hop2': ['Scotland', 17.5], 'comp1': ['England', 11.5]}, 'titanic': {'hop1': ['Titanic', 6.12], 'hop2': ['James Cameron', 11.94], 'comp1': ['Alfred Hitchcock', 6.88]}, 'tokyo': {'hop1': ['Japan', 19.88], 'hop2': ['The Japanese Yen', -0.12], 'comp1': ['Japanese Yen', -2.0]}}

## Tam grup etkisi Δ/m0 (m = logit cevap − logit köprü)

- Z grubu, zero: 1.31 [0.88, 2.89] (n=11); cevap değişti (marj<0) 8/11; en olası token köprü 6/11
- Z grubu, başka zincirden resample: 1.37 [0.72, 2.70] (n=11); cevap değişti (marj<0) 8/11; en olası token köprü 6/11
- rastgele 12, zero: 0.00 [0.00, 0.00] (n=11); cevap değişti (marj<0) 0/11; en olası token köprü 0/11
- P grubu, hop1 yaması: 1.52 [1.23, 2.20] (n=11); cevap değişti (marj<0) 10/11; en olası token köprü 9/11
- P grubu, başka zincirden yama: 0.85 [0.02, 1.40] (n=11); cevap değişti (marj<0) 4/11; en olası token köprü 4/11
- rastgele 12, hop1 yaması: 0.00 [-0.01, 0.01] (n=11); cevap değişti (marj<0) 0/11; en olası token köprü 0/11
- P hop1 − rastgele (zincir içi fark): 1.52 [1.23, 2.20] (n=11)

## Logit ayrışması (tam grup; cevap ve köprü logitlerindeki değişim)

| zincir | cevap→köprü | Z: Δlogit cevap | Z: Δlogit köprü | P: Δlogit cevap | P: Δlogit köprü | Z∩P |
|---|---|---|---|---|---|---|
| apple | Steve→Jobs | -6.50 | +0.62 | -1.50 | +1.00 | 0 |
| beatles | England→Liverpool | -6.12 | +1.25 | -5.88 | +0.75 | 3 |
| eiffel | Paris→France | -10.62 | +0.12 | -2.75 | +8.38 | 5 |
| lisbon | Atlantic→Portugal | -7.88 | +4.00 | -2.62 | +12.38 | 4 |
| lk | Scar→Simba | -9.00 | -1.62 | -2.12 | +2.00 | 2 |
| madrid | Spanish→Spain | -8.25 | -1.12 | -4.50 | +8.62 | 6 |
| mona | Italy→Leonardo | -7.12 | -1.38 | -5.00 | +3.62 | 4 |
| nairobi | Africa→Kenya | -9.88 | -1.00 | -4.25 | +6.00 | 8 |
| ninth | German→Beethoven | -8.00 | +1.00 | -4.88 | +7.75 | 4 |
| pyramids | Cairo→Egypt | -5.12 | +4.62 | -3.25 | +7.50 | 5 |
| relativity | Germany→Einstein | -10.75 | +0.50 | -4.25 | +7.50 | 5 |

## Müdahaleli üretim (açgözlü, ilk satır)

| zincir | normal | Z tam | P tam | P en küçük q50 kümeleri |
|---|---|---|---|---|
| apple | Steve Jobs | Jobs | Steve Jobs | min0_3n: Steve Jobs; min1_3n: Steve Jobs; min2_3n: Steve Jobs; min3_4n: Steve Jobs |
| beatles | England | Liverpool, England | Liverpool, England | min0_3n: Liverpool, England; min1_3n: Liverpool, England; min2_3n: Liverpool, England; min3_3n: Liverpool, England |
| eiffel | Paris | France | France | min0_2n: Paris; min1_2n: Paris; min2_2n: Paris; min3_3n: Paris |
| lisbon | Atlantic Ocean | The Atlantic Ocean | Portugal | min0_3n: The Atlantic Ocean; min1_3n: Atlantic Ocean; min2_3n: Atlantic Ocean; min3_3n: The Atlantic Ocean |
| lk | Scar | Simba | Simba | min0_4n: Simba; min1_4n: Simba; min2_4n: Simba; min3_4n: Simba |
| madrid | Spanish | Spanish | Spain | min0_2n: Spanish; min1_3n: Spanish; min2_3n: Spanish; min3_3n: Spanish |
| mona | Italy | France | Leonardo da Vinci was  | min0_3n: Italy; min1_3n: Italy; min2_3n: Italy; min3_3n: Italy |
| nairobi | Africa | Kenya | Kenya | min0_3n: Africa; min1_3n: Africa; min2_3n: Africa; min3_3n: Africa |
| ninth | German | German | Beethoven | min0_3n: German; min1_3n: German; min2_3n: German; min3_3n: German |
| pyramids | Cairo | Egypt | Egypt | min0_3n: Egypt; min1_3n: Egypt; min2_3n: Egypt; min3_3n: Egypt |
| relativity | Germany | Albert Einstein was bo | Albert Einstein was bo | min0_3n: Germany; min1_3n: Germany; min2_3n: Germany; min3_3n: Germany |

## Toplamsallık, etkileşim, benzersizlik (yorumlanabilir olanlar: |Δ_tam| ≥ 3·gürültü)

| zincir | koşul | gürültü | R² | eğim | sinerji/yedek | q50 minimal | ayrık çift | en güçlü olmadan max |
|---|---|---|---|---|---|---|---|---|
| apple | Z_zero | 0.375 | 0.776 | 0.74 | 0/0 | 222 | 265 | 0.95 |
| apple | P_hop1 | 0.250 | 0.834 | 0.72 | 0/0 | 221 | 91 | 0.75 |
| beatles | Z_zero | 0.375 | 0.942 | 1.00 | 0/0 | 435 | 539 | 0.86 |
| beatles | P_hop1 | 0.375 | 0.875 | 0.75 | 0/0 | 225 | 1712 | 0.87 |
| eiffel | Z_zero | 0.250 | 0.844 | 0.91 | 0/0 | 208 | 667 | 0.85 |
| eiffel | P_hop1 | 0.250 | 0.830 | 0.76 | 0/0 | 166 | 1065 | 0.85 |
| lisbon | Z_zero | 0.375 | 0.945 | 0.99 | 0/0 | 168 | 390 | 0.75 |
| lisbon | P_hop1 | 0.250 | 0.941 | 1.02 | 0/0 | 298 | 1515 | 0.82 |
| lk | Z_zero | 0.375 | 0.919 | 0.84 | 0/0 | 230 | 1370 | 0.92 |
| lk | P_hop1 | 0.250 | 0.915 | 0.90 | 0/0 | 478 | 802 | 0.94 |
| madrid | Z_zero | 0.375 | 0.776 | 1.03 | 0/0 | 319 | 106 | 0.91 |
| madrid | P_hop1 | 0.375 | 0.919 | 1.11 | 0/0 | 147 | 382 | 0.76 |
| mona | Z_zero | 0.375 | 0.912 | 0.87 | 0/0 | 554 | 2428 | 0.89 |
| mona | P_hop1 | 0.250 | 0.928 | 0.78 | 0/0 | 294 | 2335 | 0.86 |
| nairobi | Z_zero | 0.375 | 0.806 | 0.80 | 0/0 | 296 | 3278 | 0.96 |
| nairobi | P_hop1 | 0.250 | 0.810 | 0.85 | 0/0 | 377 | 3562 | 0.96 |
| ninth | Z_zero | 0.375 | 0.822 | 0.94 | 0/0 | 379 | 1517 | 0.93 |
| ninth | P_hop1 | 0.375 | 0.804 | 0.74 | 0/0 | 201 | 2072 | 0.92 |
| pyramids | Z_zero | 0.250 | 0.885 | 0.84 | 0/0 | 212 | 870 | 0.91 |
| pyramids | P_hop1 | 0.250 | 0.911 | 0.76 | 0/1 | 211 | 1498 | 0.88 |
| relativity | Z_zero | 0.250 | 0.892 | 1.01 | 0/0 | 402 | 732 | 0.87 |
| relativity | P_hop1 | 0.125 | 0.897 | 0.88 | 2/1 | 314 | 1685 | 0.82 |

## Transfer: aynı zincirin başka yazımları (aynı grup, seçim yok)

| zincir | yazım | Z: Δ/m0 | Z: Spearman | P: Δ/m0 | P: Spearman |
|---|---|---|---|---|---|
| apple | comp2 | 0.64 | 0.55 | 0.90 | 0.80 |
| apple | comp3 | 1.02 | 0.95 | 0.63 | 0.93 |
| beatles | comp2 | geçersiz | 0.96 | geçersiz | 0.89 |
| beatles | comp3 | 3.73 | 0.98 | 3.60 | 0.97 |
| eiffel | comp2 | 1.39 | 0.95 | 1.48 | 0.95 |
| eiffel | comp3 | 1.20 | 0.97 | 1.27 | 0.99 |
| lisbon | comp2 | 1.24 | 0.99 | 1.44 | 0.99 |
| lisbon | comp3 | 1.30 | 1.00 | 1.74 | 1.00 |
| lk | comp2 | 2.50 | 0.98 | 1.50 | 0.92 |
| lk | comp3 | 2.45 | 0.96 | 1.64 | 0.96 |
| madrid | comp2 | 0.84 | 0.99 | 1.58 | 1.00 |
| madrid | comp3 | 0.82 | 0.99 | 1.57 | 1.00 |
| mona | comp2 | 0.69 | 0.95 | 1.42 | 0.98 |
| mona | comp3 | 0.66 | 0.91 | 1.23 | 0.99 |
| nairobi | comp2 | 1.07 | 0.97 | 1.19 | 0.99 |
| nairobi | comp3 | 1.81 | 0.94 | 2.12 | 0.97 |
| ninth | comp2 | 0.68 | 0.94 | 1.17 | 0.94 |
| ninth | comp3 | 0.88 | 0.98 | 1.26 | 0.99 |
| pyramids | comp2 | 2.07 | 0.93 | 2.26 | 0.97 |
| pyramids | comp3 | 2.27 | 0.98 | 2.67 | 0.99 |
| relativity | comp2 | 0.99 | 0.98 | 1.16 | 0.99 |
| relativity | comp3 | 1.67 | 0.93 | 1.60 | 0.98 |

## Zincirler arası transfer (A'nın grubu B'nin sorusunda; içeriğe özgü grup başka zincirde zayıf olmalı)

- Z (zero): kendi zinciri 1.31 [0.88, 2.89] (n=11); başka zincir 0.12 [0.11, 0.17] (n=110)
- P (hop1 yaması, hedef zincirin hop1'inden): kendi zinciri 1.52 [1.23, 2.20] (n=11); başka zincir 0.18 [0.07, 0.34] (n=110)

Pilot ölçekli (~10 zincir); tek model; bf16. Sonuçlar tahmin, kanıt dili kullanılmaz.
