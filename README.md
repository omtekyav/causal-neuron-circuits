# Are Causal Neuron Circuits Unique and Transferable?

Code, data and results for the paper of the same name (Ömer Tekin Yavuz, 2026; arXiv link to be added).

**Question.** When intervening on a group of neurons changes a model's answer, how much of that group is actually
necessary? Could a different group do the same? Does the group do the same thing on other inputs?

The paper starts from Goodfire's Lion King example ([Neural Networks Are Already Clean](https://already-clean.goodfire.com/)):
removing an 18-neuron group in Qwen3-32B makes the model answer "Simba" instead of "Scar". We measure **every subset**
of 12-neuron groups (2¹² = 4,096 subsets per group) in four small open models. We study neuron-level causal
interventions and alternative subsets; we do not recover the full connectivity of circuits, and we did not test
Goodfire's own neurons.

## Main results

| | Qwen3-1.7B | Llama-3.2-1B | Gemma-3-1B |
|---|---|---|---|
| Patching 12 neurons from the first-hop question flips the answer to the bridge (Scar → Simba), 7 shared chains | 6/7 | 6/7 | 6/7 |
| 12 random neurons flip the answer | 0/7 | 0/7 | 0/7 |
| Chains in which two **disjoint** subsets each flip the answer | 2/7 | 4/7 | 5/7 |
| Same group on other phrasings of the same chain (Spearman over 4,095 subsets) | 0.98 | 0.98 | 0.98 |
| Same group on other chains (relative effect; ~1.5–1.8 on its own chain) | 0.01 | 0.02 | 0.03 |

- **Necessity.** The full group is rarely necessary; a few neurons usually suffice (Lion King: 4, 8 and 6 neurons).
- **Uniqueness (exploratory).** Disjoint alternatives are common when the model is less confident (13/15 chains with
  clean margin < 4) and rare when it is confident (5/17); the cut-off was chosen after seeing data.
- **A cautionary case (Study 1, Qwen3-0.6B, capital questions).** Twelve neurons selected by zero ablation lower the
  margin by 27% (random neurons: 0%), add up almost perfectly and transfer to new questions — yet replacing them with
  their values from other questions has no effect (0%). They carry a near-constant signal, not question-specific content.

All numbers in the paper are computed from the files in `results/` by `paper/stats.py`.

## Layout

```
src/            neuron capture, interventions (zero / mean / resample / patch), scores, subset analyses
data/           question generators and the generated JSONL files (capitals, reference A/B set, 20 two-hop chains)
configs/        one YAML file per experiment
run_pilot.py    Study 1 pilot: 12 candidates, all 4,096 subsets (bf16)
run_pilot1b.py  Study 1: same candidates in float32; additivity, interaction, uniqueness, transfer
run_step2.py    Study 1: resample/mean controls and held-out test questions
run_competence.py  which models solve the two-hop chains (no neuron analysis)
run_chains.py   Study 2: Z (zero) and P (patching) groups, all subsets, generation, transfer, cross-chain
analyze_models.py  cross-model comparison
paper/          stats.py (all numbers), figures.py, build_paper.py (Word/PDF), figures/, the paper PDF
results/        saved outputs of the runs reported in the paper
tests/          unit tests on a tiny random Qwen3 (no download needed)
```

Code comments and console messages are in Turkish.

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest -q tests                         # 12 tests, CPU, no model download

# Study 1 (Qwen3-0.6B-Base)
python run_pilot.py configs/pilot.yaml            # writes results/pilot_<time>/
python run_pilot1b.py configs/pilot1b.yaml        # set pilot1_dir in the config to the pilot output
python run_step2.py configs/step2.yaml

# Study 2
python run_competence.py configs/competence.yaml --model Qwen/Qwen3-1.7B-Base
python run_chains.py configs/chains.yaml                                # Qwen3-1.7B-Base
python run_chains.py configs/chains.yaml --model unsloth/Llama-3.2-1B
python run_chains.py configs/chains.yaml --model unsloth/gemma-3-1b-pt
python analyze_models.py results/chains_<qwen> results/chains_<llama> results/chains_<gemma>

# Paper numbers and figures from the saved results
cd paper && python stats.py && python figures.py
```

- Each run took 5–7 minutes on one NVIDIA L40S. Peak GPU memory was 2–4 GB.
- `mem_fraction` in the configs caps GPU memory per process because the experiments were run on a shared GPU.
  Raise it on a dedicated GPU.
- Run one model per process.
- Llama-3.2-1B and Gemma-3-1B were loaded from the public `unsloth/` mirrors of the official base checkpoints. Their use
  is governed by the Llama 3.2 Community License and the Gemma Terms of Use. Qwen3 models are Apache-2.0.
- Study 2 runs in bfloat16. Each chain's noise floor is measured and reported. Results can differ slightly across GPUs
  and library versions. The original runs used torch 2.11 and transformers 5.9.
- `paper/build_paper.py` writes the Word document and exports a PDF through Microsoft Word on Windows.

## Pre-specification

Definitions, inclusion rules and decision criteria were written down before each step. Every change made after seeing
data is listed in Appendix B of the paper. In particular, the Study 1 effect threshold and the minimal-set definition
were revised between pilot runs. The margin cut-off in the uniqueness analysis is post hoc.

## License

Code and data: MIT (see `LICENSE`). Model weights are not included and keep their own licenses.

## Citation

```
@misc{yavuz2026neuroncircuits,
  title  = {Are Causal Neuron Circuits Unique and Transferable?},
  author = {Yavuz, {\"O}mer Tekin},
  year   = {2026},
  note   = {arXiv preprint (link to be added)}
}
```
