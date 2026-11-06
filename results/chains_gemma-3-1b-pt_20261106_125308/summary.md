# Adım 4 — zincirler — chains_gemma-3-1b-pt_20261106_125308

`unsloth/gemma-3-1b-pt` rev 34a98bf3eaecc08c699516f625eb5e56e4c5d99d, torch.bfloat16, NVIDIA L40S, süre 373.6 sn, tepe GPU 2.21444034576416 GiB. Tanımlar: docs/adim4_zincir_tanimlar.md. Medyan [%95 bootstrap GA], birim = zincir.

Dahil 10: ['apple', 'eiffel', 'everest', 'lisbon', 'lk', 'madrid', 'nairobi', 'ninth', 'pyramids', 'sherlock']
Dışlanan: {'alex': {'hop1': ['Aristotle', 5.5], 'hop2': ['Plato', 3.88], 'comp1': ['Aristotle', -3.62]}, 'beatles': {'hop1': ['Liverpool', 8.38], 'hop2': ['England', 5.62], 'comp1': ['Liverpool', -1.38]}, 'hamlet': {'hop1': ['Denmark', 7.56], 'hop2': ['Copenhagen', 8.31], 'comp1': ['London', -1.38]}, 'hp': {'hop1': ['Voldemort', 6.0], 'hop2': ['Tom Riddle', 1.62], 'comp1': ['Voldemort', -3.62]}, 'kangaroo': {'hop1': ['Australia', 13.88], 'hop2': ['Canberra', 4.0], 'comp1': ['Australia', -5.88]}, 'mona': {'hop1': ['Leonardo da Vinci', 11.44], 'hop2': ['Italy', 6.12], 'comp1': ['France', -0.12]}, 'relativity': {'hop1': ['Albert Einstein', 12.56], 'hop2': ['Germany', 11.56], 'comp1': ['Albert Einstein', 2.5]}, 'romeo': {'hop1': ['William Shakespeare', 11.0], 'hop2': ['Stratford-upon-Avon', 8.62], 'comp1': ['Verona', 2.69]}, 'titanic': {'hop1': ['Titanic', 10.62], 'hop2': ['James Cameron', 7.88], 'comp1': ['Baz Luhrmann', 0.62]}, 'tokyo': {'hop1': ['Japan', 19.47], 'hop2': ['Yen', 3.44], 'comp1': ['Japanese Yen', -1.25]}}

## Tam grup etkisi Δ/m0 (m = logit cevap − logit köprü)

- Z grubu, zero: 2.12 [1.31, 2.96] (n=10); cevap değişti (marj<0) 8/10; en olası token köprü 8/10
- Z grubu, başka zincirden resample: 1.90 [0.73, 2.57] (n=10); cevap değişti (marj<0) 7/10; en olası token köprü 7/10
- rastgele 12, zero: 0.00 [-0.01, 0.02] (n=10); cevap değişti (marj<0) 0/10; en olası token köprü 0/10
- P grubu, hop1 yaması: 1.82 [1.16, 3.17] (n=10); cevap değişti (marj<0) 8/10; en olası token köprü 8/10
- P grubu, başka zincirden yama: 0.85 [0.28, 1.67] (n=10); cevap değişti (marj<0) 4/10; en olası token köprü 4/10
- rastgele 12, hop1 yaması: 0.00 [-0.02, 0.01] (n=10); cevap değişti (marj<0) 0/10; en olası token köprü 0/10
- P hop1 − rastgele (zincir içi fark): 1.83 [1.17, 3.17] (n=10)

## Logit ayrışması (tam grup; cevap ve köprü logitlerindeki değişim)

| zincir | cevap→köprü | Z: Δlogit cevap | Z: Δlogit köprü | P: Δlogit cevap | P: Δlogit köprü | Z∩P |
|---|---|---|---|---|---|---|
| apple | Steve→Jobs | -2.25 | +3.75 | -0.62 | +0.88 | 1 |
| eiffel | Paris→France | -0.25 | +8.00 | -3.12 | +3.62 | 5 |
| everest | Kathmandu→Nepal | -3.38 | +4.50 | -3.75 | +4.12 | 6 |
| lisbon | Atlantic→Portugal | -2.25 | +8.75 | -5.12 | +7.50 | 4 |
| lk | Scar→Simba | -3.75 | +2.62 | -2.00 | +1.38 | 3 |
| madrid | Spanish→Spain | -5.62 | +2.88 | -6.25 | +2.88 | 5 |
| nairobi | Africa→Kenya | -2.25 | +7.50 | -3.50 | +5.25 | 6 |
| ninth | German→Beethoven | -3.75 | +3.50 | -3.25 | +4.62 | 3 |
| pyramids | Cairo→Egypt | -2.62 | +9.12 | -5.88 | +1.62 | 6 |
| sherlock | Scotland→Doyle | -0.38 | +5.19 | -0.62 | +4.94 | 8 |

## Müdahaleli üretim (açgözlü, ilk satır)

| zincir | normal | Z tam | P tam | P en küçük q50 kümeleri |
|---|---|---|---|---|
| apple | Steve Jobs | Steve Jobs | Steve Jobs | min0_5n: Steve Jobs; min1_5n: Steve Jobs; min2_5n: Steve Jobs; min3_6n: Steve Jobs |
| eiffel | Paris | France | France | min0_2n: France; min1_2n: Paris; min2_2n: Paris; min3_3n: France |
| everest | Kathmandu | Nepal | Nepal | min0_2n: Nepal; min1_3n: Nepal; min2_3n: Nepal; min3_3n: Nepal |
| lisbon | Atlantic | Portugal | Portugal | min0_4n: The Atlantic Ocean; min1_4n: The Atlantic Ocean; min2_4n: Portugal; min3_4n: The Atlantic Ocean |
| lk | Scar | Simba | Simba | min0_3n: Scar; min1_3n: Scar; min2_3n: Scar; min3_3n: Scar |
| madrid | Spanish | Spain | Spain | min0_2n: Spain; min1_2n: Spain; min2_2n: Spain; min3_2n: Spain |
| nairobi | Africa | Kenya | Kenya | min0_2n: Kenya; min1_2n: Africa; min2_3n: Africa; min3_3n: Kenya |
| ninth | German | Beethoven | Beethoven | min0_2n: Beethoven; min1_2n: Beethoven; min2_2n: Beethoven; min3_2n: Beethoven |
| pyramids | Cairo | Egypt | Egypt | min0_2n: Egypt; min1_2n: Egypt; min2_2n: Egypt; min3_2n: Egypt |
| sherlock | Scotland | Arthur Conan Doyle | Arthur Conan Doyle | min0_4n: Scotland; min1_4n: Scotland; min2_4n: Scotland; min3_4n: Scotland |

## Toplamsallık, etkileşim, benzersizlik (yorumlanabilir olanlar: |Δ_tam| ≥ 3·gürültü)

| zincir | koşul | gürültü | R² | eğim | sinerji/yedek | q50 minimal | ayrık çift | en güçlü olmadan max |
|---|---|---|---|---|---|---|---|---|
| apple | Z_zero | 0.250 | 0.888 | 1.19 | 0/0 | 367 | 167 | 0.81 |
| apple | P_hop1 | 0.250 | -0.717 | 2.03 | 0/0 | 331 | 8 | 0.75 |
| eiffel | Z_zero | 0.250 | 0.660 | 0.84 | 0/0 | 238 | 2100 | 1.05 |
| eiffel | P_hop1 | 0.250 | 0.577 | 0.69 | 0/0 | 163 | 2377 | 1.00 |
| everest | Z_zero | 0.125 | 0.885 | 1.04 | 3/0 | 236 | 200 | 0.78 |
| everest | P_hop1 | 0.125 | 0.880 | 0.86 | 1/0 | 180 | 582 | 0.86 |
| lisbon | Z_zero | 0.250 | 0.915 | 0.96 | 0/0 | 316 | 370 | 0.90 |
| lisbon | P_hop1 | 0.125 | 0.964 | 0.97 | 0/0 | 412 | 1368 | 0.85 |
| lk | Z_zero | 0.250 | 0.951 | 0.97 | 0/0 | 431 | 349 | 0.90 |
| lk | P_hop1 | 0.125 | 0.918 | 0.82 | 0/0 | 298 | 1939 | 0.85 |
| madrid | Z_zero | 0.250 | 0.794 | 0.78 | 0/2 | 184 | 1122 | 0.88 |
| madrid | P_hop1 | 0.250 | 0.738 | 0.67 | 0/3 | 143 | 1076 | 0.88 |
| nairobi | Z_zero | 0.125 | 0.843 | 0.93 | 1/3 | 121 | 347 | 0.91 |
| nairobi | P_hop1 | 0.125 | 0.922 | 0.82 | 0/3 | 139 | 697 | 0.84 |
| ninth | Z_zero | 0.375 | 0.671 | 0.69 | 0/0 | 240 | 3522 | 0.97 |
| ninth | P_hop1 | 0.250 | 0.581 | 0.63 | 0/4 | 151 | 2031 | 0.92 |
| pyramids | Z_zero | 0.250 | 0.787 | 1.04 | 1/0 | 200 | 242 | 0.94 |
| pyramids | P_hop1 | 0.250 | 0.806 | 0.84 | 0/0 | 111 | 712 | 0.92 |
| sherlock | Z_zero | 0.188 | 0.774 | 0.88 | 0/0 | 407 | 628 | 0.94 |
| sherlock | P_hop1 | 0.188 | 0.868 | 0.83 | 0/0 | 342 | 807 | 0.83 |

## Transfer: aynı zincirin başka yazımları (aynı grup, seçim yok)

| zincir | yazım | Z: Δ/m0 | Z: Spearman | P: Δ/m0 | P: Spearman |
|---|---|---|---|---|---|
| apple | comp2 | 0.81 | 0.93 | 0.16 | 0.55 |
| apple | comp3 | 0.87 | 0.96 | 0.19 | 0.80 |
| eiffel | comp2 | 1.25 | 0.72 | 1.28 | 0.75 |
| eiffel | comp3 | 2.89 | 0.96 | 2.53 | 0.98 |
| everest | comp2 | 1.44 | 0.94 | 1.71 | 0.96 |
| everest | comp3 | 2.12 | 0.98 | 2.19 | 0.98 |
| lisbon | comp2 | 3.83 | 0.93 | 3.14 | 0.98 |
| lisbon | comp3 | 1.84 | 0.98 | 1.94 | 0.99 |
| lk | comp2 | 1.52 | 0.95 | 1.08 | 0.94 |
| lk | comp3 | 2.23 | 0.98 | 1.27 | 0.98 |
| madrid | comp2 | 1.59 | 0.89 | 2.29 | 0.93 |
| madrid | comp3 | 3.37 | 0.99 | 3.68 | 0.99 |
| nairobi | comp2 | 1.98 | 0.99 | 1.83 | 0.99 |
| nairobi | comp3 | 1.84 | 0.98 | 1.68 | 0.98 |
| ninth | comp2 | geçersiz | 0.90 | geçersiz | 0.94 |
| ninth | comp3 | 2.18 | 0.98 | 2.36 | 0.98 |
| pyramids | comp2 | 5.91 | 0.85 | 3.82 | 0.88 |
| pyramids | comp3 | geçersiz | 0.89 | geçersiz | 0.95 |
| sherlock | comp2 | 0.60 | 0.83 | 0.60 | 0.86 |
| sherlock | comp3 | 0.74 | 0.95 | 0.70 | 0.92 |

## Zincirler arası transfer (A'nın grubu B'nin sorusunda; içeriğe özgü grup başka zincirde zayıf olmalı)

- Z (zero): kendi zinciri 2.12 [1.31, 2.96] (n=10); başka zincir 0.01 [0.00, 0.05] (n=90)
- P (hop1 yaması, hedef zincirin hop1'inden): kendi zinciri 1.82 [1.16, 3.17] (n=10); başka zincir 0.11 [0.04, 0.26] (n=90)

Pilot ölçekli (~10 zincir); tek model; bf16. Sonuçlar tahmin, kanıt dili kullanılmaz.
