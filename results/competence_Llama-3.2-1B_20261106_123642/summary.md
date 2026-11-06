# Yetkinlik kontrolü — competence_Llama-3.2-1B_20261106_123642

Nöron analizi yok. Açgözlü üretim, ilk satır; marj = logit(cevap) − logit(köprü) (bileşik).

## unsloth/Llama-3.2-1B

hop1 19/20, hop2 20, bileşik-1 12, bileşik-2 11; tam çözülen 10: ['apple', 'eiffel', 'everest', 'lisbon', 'lk', 'madrid', 'nairobi', 'ninth', 'relativity', 'tokyo']

| zincir | hop1 | hop2 | bileşik-1 (marj) | bileşik-2 (marj) |
|---|---|---|---|---|
| alex | ✓ Aristotle | ✓ Plato | ✗ Aristotle (-3.6) | ✗ Aristotle (-0.5) |
| apple | ✓ Steve Jobs and Steve Wozniak | ✓ Steve | ✓ Steve Jobs (5.0) | ✓ Steve (4.6) |
| beatles | ✓ Liverpool | ✓ England | ✗ Liverpool (-2.0) | ✓ England (0.1) |
| eiffel | ✓ France | ✓ Paris | ✓ Paris (2.4) | ✓ Paris (0.5) |
| everest | ✓ Nepal | ✓ Kathmandu | ✓ Kathmandu (3.4) | ✓ Kathmandu (1.6) |
| hamlet | ✗ England | ✓ Copenhagen | ✗ London (-0.4) | ✗ Denmark (-0.6) |
| hp | ✓ Voldemort | ✓ Tom Riddle | ✗ Voldemort (-2.8) | ✗ Voldemort (-2.2) |
| kangaroo | ✓ Australia | ✓ Canberra | ✗ Australia (-2.2) | ✗ Australia (-0.8) |
| lisbon | ✓ Portugal | ✓ Atlantic Ocean | ✓ Atlantic Ocean (7.2) | ✓ Atlantic (6.0) |
| lk | ✓ Simba | ✓ Scar | ✓ Scar (3.1) | ✓ Scar (3.4) |
| madrid | ✓ Spain | ✓ Spanish | ✓ Spanish (5.8) | ✓ Spanish (4.9) |
| mona | ✓ Leonardo da Vinci | ✓ Italy | ✗ France (3.0) | ✗ France (1.5) |
| nairobi | ✓ Kenya | ✓ Africa | ✓ Africa (5.1) | ✓ Africa (7.1) |
| ninth | ✓ Ludwig van Beethoven | ✓ German | ✓ German (3.2) | ✓ German (5.2) |
| pyramids | ✓ Egypt | ✓ Cairo | ✓ Cairo (0.4) | ✗ Egypt (-0.5) |
| relativity | ✓ Albert Einstein | ✓ Germany | ✓ Germany (7.4) | ✓ Germany (7.4) |
| romeo | ✓ William Shakespeare | ✓ Stratford-upon-Avon | ✗ Verona (2.1) | ✗ Verona (1.3) |
| sherlock | ✓ Sir Arthur Conan Doyle | ✓ Scotland | ✗ England (6.2) | ✗ England (3.4) |
| titanic | ✓ Titanic | ✓ James Cameron | ✓ James Cameron (1.9) | ✗ Steven Spielberg (4.4) |
| tokyo | ✓ Japan | ✓ Yen | ✓ Yen (1.8) | ✓ Yen (1.6) |

