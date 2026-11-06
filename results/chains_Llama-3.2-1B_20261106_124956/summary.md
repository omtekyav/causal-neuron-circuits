# Adım 4 — zincirler — chains_Llama-3.2-1B_20261106_124956

`unsloth/Llama-3.2-1B` rev 9535bd9b1d1dea6acafbdc4813b728796aeb28da, torch.bfloat16, NVIDIA L40S, süre 183.9 sn, tepe GPU 2.541684627532959 GiB. Tanımlar: docs/adim4_zincir_tanimlar.md. Medyan [%95 bootstrap GA], birim = zincir.

Dahil 11: ['apple', 'eiffel', 'everest', 'lisbon', 'lk', 'madrid', 'nairobi', 'ninth', 'relativity', 'titanic', 'tokyo']
Dışlanan: {'alex': {'hop1': ['Aristotle', 4.12], 'hop2': ['Plato', 4.38], 'comp1': ['Aristotle', -3.62]}, 'beatles': {'hop1': ['Liverpool', 8.25], 'hop2': ['England', 3.62], 'comp1': ['Liverpool', -2.0]}, 'hamlet': {'hop1': ['England', 7.88], 'hop2': ['Copenhagen', 6.25], 'comp1': ['London', -0.38]}, 'hp': {'hop1': ['Voldemort', 3.94], 'hop2': ['Tom Riddle', 4.25], 'comp1': ['Voldemort', -2.81]}, 'kangaroo': {'hop1': ['Australia', 13.75], 'hop2': ['Canberra', 5.38], 'comp1': ['Australia', -2.25]}, 'mona': {'hop1': ['Leonardo da Vinci', 13.19], 'hop2': ['Italy', 9.12], 'comp1': ['France', 3.0]}, 'pyramids': {'hop1': ['Egypt', 8.0], 'hop2': ['Cairo', 6.88], 'comp1': ['Cairo', 0.38]}, 'romeo': {'hop1': ['William Shakespeare', 10.5], 'hop2': ['Stratford-upon-Avon', 7.81], 'comp1': ['Verona', 2.06]}, 'sherlock': {'hop1': ['Sir Arthur Conan Doyle', 3.0], 'hop2': ['Scotland', 9.62], 'comp1': ['England', 6.19]}}

## Tam grup etkisi Δ/m0 (m = logit cevap − logit köprü)

- Z grubu, zero: 2.04 [1.20, 3.26] (n=11); cevap değişti (marj<0) 11/11; en olası token köprü 7/11
- Z grubu, başka zincirden resample: 1.41 [0.78, 2.25] (n=11); cevap değişti (marj<0) 8/11; en olası token köprü 6/11
- rastgele 12, zero: 0.00 [-0.04, 0.02] (n=11); cevap değişti (marj<0) 0/11; en olası token köprü 0/11
- P grubu, hop1 yaması: 1.89 [1.35, 3.71] (n=11); cevap değişti (marj<0) 10/11; en olası token köprü 9/11
- P grubu, başka zincirden yama: 0.92 [0.58, 1.25] (n=11); cevap değişti (marj<0) 4/11; en olası token köprü 3/11
- rastgele 12, hop1 yaması: 0.00 [-0.01, 0.04] (n=11); cevap değişti (marj<0) 0/11; en olası token köprü 0/11
- P hop1 − rastgele (zincir içi fark): 1.87 [1.35, 3.68] (n=11)

## Logit ayrışması (tam grup; cevap ve köprü logitlerindeki değişim)

| zincir | cevap→köprü | Z: Δlogit cevap | Z: Δlogit köprü | P: Δlogit cevap | P: Δlogit köprü | Z∩P |
|---|---|---|---|---|---|---|
| apple | Steve→Jobs | -5.00 | +0.94 | -0.50 | +1.06 | 5 |
| eiffel | Paris→France | -9.38 | -0.75 | -2.00 | +3.25 | 4 |
| everest | Kathmandu→Nepal | -2.25 | +5.75 | -3.75 | +5.12 | 3 |
| lisbon | Atlantic→Portugal | -5.50 | +2.75 | -4.12 | +6.75 | 3 |
| lk | Scar→Simba | -4.12 | +1.75 | -0.75 | +2.62 | 2 |
| madrid | Spanish→Spain | -5.50 | +4.50 | -4.62 | +6.25 | 6 |
| nairobi | Africa→Kenya | -5.38 | +2.75 | -5.38 | +4.00 | 5 |
| ninth | German→Beethoven | -10.25 | -2.81 | -8.31 | +3.62 | 3 |
| relativity | Germany→Einstein | -4.00 | +4.38 | -6.88 | +3.25 | 2 |
| titanic | James→Titanic | -4.81 | +1.50 | -3.69 | +3.50 | 3 |
| tokyo | yen→Japan | -9.25 | +0.62 | -1.62 | +5.88 | 1 |

## Müdahaleli üretim (açgözlü, ilk satır)

| zincir | normal | Z tam | P tam | P en küçük q50 kümeleri |
|---|---|---|---|---|
| apple | Steve Jobs | Jobs | Steve Jobs | min0_4n: Steve Jobs; min1_4n: Steve Jobs; min2_4n: Steve Jobs; min3_4n: Steve Jobs |
| eiffel | Paris | France | France | min0_1n: France; min1_1n: France; min2_3n: France; min3_4n: France |
| everest | Kathmandu | Nepal | Nepal | min0_2n: Nepal; min1_2n: Nepal; min2_2n: Nepal; min3_2n: Nepal |
| lisbon | Atlantic Ocean | The Atlantic Ocean | Portugal | min0_3n: Atlantic Ocean; min1_3n: Atlantic Ocean; min2_3n: Atlantic Ocean; min3_3n: Atlantic Ocean |
| lk | Scar | Simba's uncle Scar | Simba's uncle Scar | min0_4n: Scar; min1_4n: Scar; min2_4n: Scar; min3_4n: Scar |
| madrid | Spanish | Spain | Spain | min0_2n: Spain; min1_2n: Spanish; min2_3n: Spanish; min3_3n: Spain |
| nairobi | Africa | Kenya | Kenya | min0_2n: Kenya; min1_3n: Africa; min2_3n: Africa; min3_3n: Kenya |
| ninth | German | Ludwig van Beethoven | Beethoven | min0_3n: Beethoven; min1_3n: Johann Sebastian B; min2_3n: Beethoven; min3_3n: Beethoven |
| relativity | Germany | Albert Einstein | Albert Einstein | min0_2n: Albert Einstein; min1_2n: Albert Einstein; min2_2n: Albert Einstein; min3_2n: Germany |
| titanic | James Cameron | Titanic | Titanic | min0_3n: Titanic; min1_3n: Titanic; min2_3n: Titanic; min3_3n: Titanic |
| tokyo | Yen | The Japanese yen | Japan yen | min0_2n: Yen; min1_3n: Japanese Yen; min2_3n: Japanese Yen; min3_3n: Japanese Yen |

## Toplamsallık, etkileşim, benzersizlik (yorumlanabilir olanlar: |Δ_tam| ≥ 3·gürültü)

| zincir | koşul | gürültü | R² | eğim | sinerji/yedek | q50 minimal | ayrık çift | en güçlü olmadan max |
|---|---|---|---|---|---|---|---|---|
| apple | Z_zero | 0.188 | 0.857 | 0.77 | 0/0 | 188 | 1353 | 0.89 |
| apple | P_hop1 | 0.250 | 0.803 | 0.68 | 0/0 | 349 | 796 | 0.92 |
| eiffel | Z_zero | 0.250 | 0.657 | 0.72 | 0/0 | 151 | 956 | 0.91 |
| eiffel | P_hop1 | 0.250 | 0.700 | 0.59 | 0/1 | 95 | 194 | 0.98 |
| everest | Z_zero | 0.250 | 0.938 | 0.90 | 0/0 | 24 | 23 | 0.61 |
| everest | P_hop1 | 0.250 | 0.900 | 0.81 | 0/1 | 67 | 367 | 0.82 |
| lisbon | Z_zero | 0.188 | 0.898 | 0.84 | 0/0 | 198 | 611 | 0.91 |
| lisbon | P_hop1 | 0.188 | 0.930 | 0.86 | 0/0 | 178 | 880 | 0.83 |
| lk | Z_zero | 0.188 | 0.933 | 0.88 | 0/0 | 310 | 1533 | 0.86 |
| lk | P_hop1 | 0.250 | 0.908 | 0.78 | 0/0 | 443 | 1320 | 0.94 |
| madrid | Z_zero | 0.250 | 0.823 | 1.05 | 4/0 | 73 | 143 | 0.94 |
| madrid | P_hop1 | 0.250 | 0.815 | 0.93 | 1/0 | 126 | 545 | 0.89 |
| nairobi | Z_zero | 0.375 | 0.760 | 0.86 | 0/0 | 123 | 510 | 1.00 |
| nairobi | P_hop1 | 0.250 | 0.892 | 0.83 | 0/0 | 170 | 1251 | 0.88 |
| ninth | Z_zero | 0.500 | -0.289 | 0.61 | 0/0 | 46 | 477 | 1.46 |
| ninth | P_hop1 | 0.375 | 0.488 | 0.70 | 0/0 | 252 | 5919 | 1.00 |
| relativity | Z_zero | 0.500 | 0.931 | 0.86 | 0/0 | 325 | 166 | 0.82 |
| relativity | P_hop1 | 0.250 | 0.940 | 0.80 | 0/0 | 103 | 289 | 0.69 |
| titanic | Z_zero | 0.250 | 0.821 | 0.83 | 0/0 | 210 | 1720 | 0.95 |
| titanic | P_hop1 | 0.188 | 0.671 | 0.68 | 0/3 | 283 | 4137 | 0.88 |
| tokyo | Z_zero | 0.375 | 0.802 | 1.03 | 0/0 | 115 | 100 | 0.93 |
| tokyo | P_hop1 | 0.250 | 0.962 | 1.19 | 1/0 | 178 | 455 | 0.72 |

## Transfer: aynı zincirin başka yazımları (aynı grup, seçim yok)

| zincir | yazım | Z: Δ/m0 | Z: Spearman | P: Δ/m0 | P: Spearman |
|---|---|---|---|---|---|
| apple | comp2 | 0.81 | 0.95 | -0.05 | -0.23 |
| apple | comp3 | 1.00 | 0.97 | 0.19 | 0.63 |
| eiffel | comp2 | geçersiz | 0.79 | geçersiz | 0.91 |
| eiffel | comp3 | 1.86 | 0.81 | 1.30 | 0.98 |
| everest | comp2 | 1.93 | 0.84 | 3.43 | 0.97 |
| everest | comp3 | 1.58 | 0.97 | 2.00 | 1.00 |
| lisbon | comp2 | 0.81 | 0.72 | 1.67 | 0.98 |
| lisbon | comp3 | 1.22 | 0.94 | 1.70 | 0.98 |
| lk | comp2 | 1.49 | 0.95 | 0.96 | 0.80 |
| lk | comp3 | 1.78 | 0.95 | 1.07 | 0.94 |
| madrid | comp2 | 1.51 | 0.91 | 1.95 | 0.99 |
| madrid | comp3 | 1.80 | 0.98 | 1.84 | 0.99 |
| nairobi | comp2 | 1.04 | 0.89 | 1.32 | 0.98 |
| nairobi | comp3 | 1.93 | 0.92 | 2.48 | 0.98 |
| ninth | comp2 | 0.88 | 0.58 | 2.02 | 0.90 |
| ninth | comp3 | 1.29 | 0.80 | 2.47 | 0.90 |
| relativity | comp2 | 0.88 | 0.90 | 1.29 | 0.99 |
| relativity | comp3 | geçersiz | 0.87 | geçersiz | 0.94 |
| titanic | comp2 | 1.03 | 0.76 | 1.67 | 0.90 |
| titanic | comp3 | 1.68 | 0.93 | 1.86 | 0.91 |
| tokyo | comp2 | 6.00 | 0.91 | 4.38 | 0.99 |
| tokyo | comp3 | 5.46 | 0.98 | 4.46 | 1.00 |

## Zincirler arası transfer (A'nın grubu B'nin sorusunda; içeriğe özgü grup başka zincirde zayıf olmalı)

- Z (zero): kendi zinciri 2.04 [1.20, 3.26] (n=11); başka zincir 0.45 [0.32, 0.56] (n=110)
- P (hop1 yaması, hedef zincirin hop1'inden): kendi zinciri 1.89 [1.35, 3.71] (n=11); başka zincir 0.09 [0.03, 0.26] (n=110)

Pilot ölçekli (~10 zincir); tek model; bf16. Sonuçlar tahmin, kanıt dili kullanılmaz.
