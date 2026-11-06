# Yetkinlik kontrolü — competence_gemma-3-1b-pt_20261106_124334

Nöron analizi yok. Açgözlü üretim, ilk satır; marj = logit(cevap) − logit(köprü) (bileşik).

## unsloth/gemma-3-1b-pt

hop1 20/20, hop2 20, bileşik-1 11, bileşik-2 11; tam çözülen 9: ['apple', 'eiffel', 'everest', 'lisbon', 'lk', 'madrid', 'nairobi', 'ninth', 'pyramids']

| zincir | hop1 | hop2 | bileşik-1 (marj) | bileşik-2 (marj) |
|---|---|---|---|---|
| alex | ✓ Aristotle | ✓ Plato | ✗ Aristotle (-3.6) | ✗ Socrates (0.0) |
| apple | ✓ Steve Jobs | ✓ Steve | ✓ Steve Jobs (7.0) | ✓ Steve (7.1) |
| beatles | ✓ Liverpool | ✓ England | ✗ Liverpool (-1.4) | ✓ England (0.1) |
| eiffel | ✓ France | ✓ Paris | ✓ Paris (3.8) | ✓ Paris (4.5) |
| everest | ✓ Nepal | ✓ Kathmandu | ✓ Kathmandu (3.9) | ✓ Kathmandu (4.4) |
| hamlet | ✓ Denmark | ✓ Copenhagen | ✗ London (-1.4) | ✗ Denmark (-2.0) |
| hp | ✓ Voldemort | ✓ Tom Riddle | ✗ Voldemort (-3.6) | ✗ Voldemort (-4.6) |
| kangaroo | ✓ Australia | ✓ Canberra | ✗ Australia (-5.9) | ✗ Australia (-2.2) |
| lisbon | ✓ Portugal | ✓ Atlantic | ✓ Atlantic (6.9) | ✓ Atlantic (3.9) |
| lk | ✓ Simba | ✓ Scar | ✓ Scar (2.8) | ✓ Scar (3.0) |
| madrid | ✓ Spain | ✓ Spanish | ✓ Spanish (2.9) | ✓ Spanish (4.2) |
| mona | ✓ Leonardo da Vinci | ✓ Italy | ✗ France (-0.1) | ✗ France (2.5) |
| nairobi | ✓ Kenya | ✓ Africa | ✓ Africa (5.5) | ✓ Africa (5.0) |
| ninth | ✓ Beethoven | ✓ German | ✓ German (2.1) | ✓ German (1.1) |
| pyramids | ✓ Egypt | ✓ Cairo | ✓ Cairo (1.2) | ✓ Cairo (1.4) |
| relativity | ✓ Albert Einstein | ✓ Germany | ✗ Albert Einstein (2.5) | ✓ Germany (4.2) |
| romeo | ✓ William Shakespeare | ✓ Stratford-upon-Avon | ✗ Verona (2.7) | ✗ Verona (3.8) |
| sherlock | ✓ Arthur Conan Doyle | ✓ Scotland | ✓ Scotland (7.5) | ✗ England (8.7) |
| titanic | ✓ Titanic | ✓ James Cameron | ✗ Baz Luhrmann (0.6) | ✗ Baz Luhrmann (0.0) |
| tokyo | ✓ Japan | ✓ Yen | ✓ Japanese Yen (-1.2) | ✗ Japan (-4.9) |

