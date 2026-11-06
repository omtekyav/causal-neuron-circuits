# Adım 2 — step2_20261106_112514

`Qwen/Qwen3-0.6B-Base` rev da87bfb608c14b7cf20ba1ce41287e8de496c0cd, torch.float32, NVIDIA L40S, süre 325.5 sn, tepe GPU 3.021510601043701 GiB. Tanımlar: docs/adim2_tanimlar.md. Medyan [%95 bootstrap GA].

## A. Resample (keşif soruları, pilot 1 adayları): tam grup Δ/m0

- zero: 0.27 [0.26, 0.30] (n=10)
- referans ortalaması: 0.02 [0.01, 0.03] (n=10)
- görev içi ortalama: -0.00 [-0.01, 0.00] (n=10)
- resample, aynı doğru harf (3 donör ort.): 0.00 [-0.01, 0.01] (n=10)
- resample, farklı doğru harf (3 donör ort.): -0.01 [-0.02, -0.00] (n=10)

| örnek | doğru | zero | ref ort. | görev içi ort. | aynı harf | farklı harf |
|---|---|---|---|---|---|---|
| cap30_p0 | A | 0.30 | 0.01 | -0.01 | -0.01 | -0.01 |
| cap38_p0 | A | 0.32 | 0.03 | 0.00 | 0.01 | -0.00 |
| cap33_p0 | B | 0.26 | 0.03 | -0.00 | 0.00 | -0.01 |
| cap22_p0 | A | 0.32 | 0.03 | -0.01 | -0.00 | -0.02 |
| cap29_p0 | B | 0.25 | 0.03 | 0.00 | 0.01 | 0.01 |
| cap23_p0 | B | 0.29 | 0.01 | -0.02 | -0.02 | -0.03 |
| cap16_p0 | A | 0.24 | 0.01 | -0.01 | 0.00 | -0.02 |
| cap24_p0 | A | 0.27 | 0.01 | -0.00 | -0.01 | -0.00 |
| cap26_p0 | A | 0.26 | 0.01 | -0.00 | 0.00 | -0.01 |
| cap09_p0 | B | 0.27 | 0.03 | 0.00 | 0.00 | -0.00 |

## B1. Dondurulmuş küresel grup (keşifte en sık 12 nöron) → test soruları, yeniden seçim yok

Grup: [[14, 692], [14, 1810], [14, 40], [14, 145], [14, 209], [21, 362], [21, 267], [21, 1489], [14, 1802], [14, 207], [21, 1046], [14, 954]]

- özgün yazım: grup 0.24 [0.22, 0.26] (n=10); rastgele 12 -0.01 [-0.01, 0.00] (n=10)
- diğer yazımlar: grup 0.20 [0.19, 0.20] (n=30); rastgele 12 -0.01 [-0.01, -0.00] (n=30)
- şık-değişik: grup 0.22 [0.20, 0.24] (n=10); rastgele 12 0.00 [-0.01, 0.00] (n=10)
- geçersiz (müdahalesiz marj < eşik) varyant: 0/50

## B2. Test sorularında yeniden seçim + 4.096 alt küme (bulguların tekrarı)

| örnek | m0 | Δtam/m0 | R² | eğim | sinerji/yedek | q50 minimal küme | ayrık çift | en güçlü nöronsuz max | yazım Spearman (min) | şık-değişik Spearman | en güçlü yine en güçlü |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cap00_p0 | 5.75 | 0.30 | 0.993 | 1.01 | 0/0 | 157 | 33 | 0.66 | 0.95 | 0.95 | 4/4 |
| cap03_p0 | 5.96 | 0.31 | 0.996 | 1.03 | 0/0 | 12 | 0 | 0.50 | 1.00 | 0.63 | 4/4 |
| cap05_p0 | 6.90 | 0.30 | 0.995 | 1.01 | 0/0 | 12 | 1 | 0.52 | 0.99 | 0.84 | 4/4 |
| cap08_p0 | 5.12 | 0.30 | 0.996 | 0.98 | 0/0 | 78 | 51 | 0.64 | 0.98 | 0.94 | 4/4 |
| cap15_p0 | 6.60 | 0.27 | 0.994 | 0.98 | 0/0 | 31 | 22 | 0.60 | 0.99 | 0.74 | 4/4 |
| cap17_p0 | 7.00 | 0.27 | 0.989 | 0.97 | 0/0 | 43 | 41 | 0.64 | 0.99 | 0.76 | 4/4 |
| cap20_p0 | 5.42 | 0.29 | 0.993 | 1.05 | 0/0 | 177 | 7 | 0.65 | 0.99 | 0.98 | 4/4 |
| cap25_p0 | 6.63 | 0.26 | 0.993 | 0.98 | 0/0 | 27 | 2 | 0.57 | 0.99 | 0.75 | 4/4 |
| cap34_p0 | 4.47 | 0.28 | 0.994 | 0.99 | 0/0 | 136 | 60 | 0.65 | 0.99 | 0.97 | 4/4 |
| cap35_p0 | 7.21 | 0.24 | 0.990 | 0.97 | 0/0 | 101 | 87 | 0.66 | 0.93 | 0.76 | 4/4 |

Test sorularında en güçlü tekli nöron: {'L14/692': 10}

Ön kayıtlı tekrar ölçütü: toplamsallık R² ≥ 0,95 ve sinerji/yedeklilik çifti yok; yazım Spearman ≥ 0,9.
