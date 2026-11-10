"""arXiv makalesi: Word (.docx) + PDF (Word ile). Bütün sayılar stats.py'den; grafikler figures.py'den.

Kullanım: python -X utf8 build_paper.py  →  paper/out/neuron_circuits.docx (+ .pdf, Word kuruluysa)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import docx
import numpy as np
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Inches, Pt, RGBColor

from stats import competence, study1, study2

HERE = Path(__file__).parent
FIG = HERE / "figures"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
# Kurum satırı isteğe bağlı: AFFIL="<kurum adı>" verilirse ayrı dosyaya (neuron_circuits_affil) üretilir.
AFFIL = os.environ.get("AFFIL", "").strip()

s1, comp, s2 = study1(), competence(), study2()
M = s2["models"]
MODELS = list(M)
own = {m: M[m]["own"] for m in MODELS}
com = {m: M[m]["common"] for m in MODELS}
mt = s2["margin_trend"]


def f2(x):
    return f"{x:.2f}"


def ci(c):
    return f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]"


def frac(a, b):
    return f"{a}/{b}"


# Lion King: cevabı çeviren en küçük alt küme, model başına
lk_min = {m: own[m]["flip"]["lk"]["min_size"] for m in MODELS}
lk_dis = {m: own[m]["flip"]["lk"]["disjoint"] for m in MODELS}
dl = {m: np.array(own[m]["dlog"]) for m in MODELS}
dl_all = np.vstack(list(dl.values()))
med = lambda col: float(np.median(dl_all[:, col]))

# ---------------------------------------------------------------- belge
# arXiv ön baskı düzeni (NeurIPS stili): Letter, 5,5 inç metin sütunu, 1 inç üst/alt
d = docx.Document()
sec = d.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Inches(1.5))
sec.top_margin, sec.bottom_margin = Inches(1), Inches(1)
TEXT_W = 13.97  # cm
st = d.styles["Normal"]
st.font.name = "Times New Roman"
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.font.size = Pt(10)
st.paragraph_format.space_after = Pt(5.5)
st.paragraph_format.line_spacing = 1.0
for lvl, size in ((1, 12), (2, 10)):
    h = d.styles[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.bold = "Times New Roman", Pt(size), True
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(11 if lvl == 1 else 8), Pt(5)


def rule(p, side, size_eighths):
    """Paragrafın üstüne/altına yatay çizgi (başlık bloğu)."""
    pPr = p._p.get_or_add_pPr()
    bdr = pPr.find(qn("w:pBdr"))
    if bdr is None:
        bdr = OxmlElement("w:pBdr"); pPr.append(bdr)
    e = OxmlElement(f"w:{side}")
    for k, v in (("val", "single"), ("sz", str(size_eighths)), ("space", "8"), ("color", "000000")):
        e.set(qn(f"w:{k}"), v)
    bdr.append(e)


def page_number_footer():
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), tag); r._r.append(f)
        else:
            t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = text; r._r.append(t)
    r.font.size = Pt(10)


page_number_footer()


def P(text: str, italic_lead: str | None = None, align=None, size=None):
    p = d.add_paragraph()
    if italic_lead:
        r = p.add_run(italic_lead + " ")
        r.italic = True
        r.bold = True
    _runs(p, text, size)
    if align:
        p.alignment = align
    p.paragraph_format.alignment = align or WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def _runs(p, text, size=None):
    """**kalın** ve _italik_ işaretleri."""
    import re
    for part in re.split(r"(\*\*[^*]+\*\*|_[^_]+_)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("_") and part.endswith("_") and len(part) > 2:
            r = p.add_run(part[1:-1]); r.italic = True
        else:
            r = p.add_run(part)
        if size:
            r.font.size = Pt(size)


def H(text, lvl=1):
    d.add_heading(text, level=lvl)


def BUL(items):
    for it in items:
        p = d.add_paragraph(style="List Bullet")
        _runs(p, it)
        p.paragraph_format.space_after = Pt(2)


def FIGURE(name, caption, width_cm=15.5):
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / name), width=Cm(min(width_cm, TEXT_W)))
    c = d.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _runs(c, caption, size=9)


def TABLE(header, rows, caption, widths=None, size=8.5):
    c = d.add_paragraph()
    _runs(c, caption, size=9)
    c.paragraph_format.keep_with_next = True
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            _runs(cells[i].paragraphs[0], str(v), size)
    if widths:
        k = min(1.0, TEXT_W / sum(widths))   # dar sütuna sığdır
        widths = [w * k for w in widths]
        for row in t.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Cm(w)
    if len(t.rows) <= 14:   # kısa tablolar sayfa arasında bölünmesin
        for row in t.rows[:-1]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
    d.add_paragraph().paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------- başlık
t = d.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("Are Causal Neuron Circuits Unique and Transferable?"); r.bold = True; r.font.size = Pt(17)
rule(t, "top", 32); rule(t, "bottom", 8)
t.paragraph_format.space_before, t.paragraph_format.space_after = Pt(6), Pt(22)
author_lines = [("Ömer Tekin Yavuz", 10, True, False), ("R&D and AI Engineer", 10, False, False),
                ("Bursa, Turkey", 10, False, False), ("omtekyav@gmail.com", 9.5, False, True)]
if AFFIL:
    author_lines.insert(2, (AFFIL, 10, False, False))
for text, size, bold, mono in author_lines:
    a = d.add_paragraph(); a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    a.paragraph_format.space_after = Pt(0)
    r = a.add_run(text); r.font.size = Pt(size); r.bold = bold
    if mono:
        r.font.name = "Courier New"

ab = d.add_paragraph(); ab.alignment = WD_ALIGN_PARAGRAPH.CENTER
ab.paragraph_format.space_before, ab.paragraph_format.space_after = Pt(22), Pt(6)
r = ab.add_run("Abstract"); r.bold = True; r.font.size = Pt(12)
abstract = P(
    "Interpretability demonstrations often show that intervening on a group of neurons changes a model's answer. A recent "
    "example removes an 18-neuron group in Qwen3-32B and the model answers “Simba” instead of “Scar” to a two-hop Lion King "
    "riddle. Such a demonstration establishes that the group matters, but not how much of it is necessary, whether other groups "
    "could do the same, or whether the effect carries over to other inputs. We ask: when intervening on a group of neurons changes "
    "a model's answer, how much of that group is actually necessary? For every selected group we measure all 4,096 subsets of "
    "12 MLP neurons. On a two-alternative capital task (Qwen3-0.6B), zero-ablating 12 selected neurons lowers the answer margin "
    f"by {s1['zero'][0]*100:.0f}% while random neurons barely change it, yet replacing the same neurons with their values from other "
    "questions also barely changes it: their activations are nearly constant across questions, and we find no evidence that they "
    "carry question-specific content. On two-hop relation "
    f"chains, patching 12 neurons with activations from the first-hop question shifts the answer to the bridge entity "
    f"(from Scar to Simba): the greedy answer starts with the bridge in {own['Qwen3-1.7B']['P_gen_bridge']}/{own['Qwen3-1.7B']['n']} chains for Qwen3-1.7B, "
    f"{own['Llama-3.2-1B']['P_gen_bridge']}/{own['Llama-3.2-1B']['n']} for Llama-3.2-1B and "
    f"{own['Gemma-3-1B']['P_gen_bridge']}/{own['Gemma-3-1B']['n']} for Gemma-3-1B; random neurons never do. The full group is "
    "rarely necessary: a few neurons often suffice, and in many chains disjoint subsets each flip the answer on their own, "
    "in an exploratory analysis most often when the model is less confident. Group effects transfer to rephrasings of the same chain but are much weaker "
    "on other tested chains. We study neuron-level causal interventions and alternative subsets; we do not recover the full connectivity "
    "of circuits.", size=10)
abstract.paragraph_format.left_indent = abstract.paragraph_format.right_indent = Inches(0.5)
abstract.paragraph_format.space_after = Pt(14)

# ---------------------------------------------------------------- 1
H("1  Introduction")
P("A common form of evidence in mechanistic interpretability is a group intervention: a set of internal components is selected, "
  "removed or overwritten, and the model's output changes. Goodfire's recent essay on neuron interpretability gives a vivid "
  "example [1]. Qwen3-32B answers a riddle about the Pride Lands with “Scar”; removing an 18-neuron “Lion King” group found in "
  "an attribution graph makes it answer “Simba” (P(Scar) = 0.24), whereas removing 18 random neurons from the same layers "
  "leaves P(Scar) at 0.91. The demonstration is a genuine causal experiment, and the random control is appropriate.")
P("What such a demonstration does not show is the internal structure of the group. Would 17 neurons have been enough? 10? 4? "
  "Could a completely different set of four neurons produce the same change? Do the neurons act independently, so that their "
  "effects add up, or only in combination? And does the group do the same thing for a rephrased question, or for a different "
  "question? Without these answers it is hard to say how much explanatory weight “these 18 neurons” can carry. The public "
  "write-up does not report how the group size was chosen, what intervention was used to remove the neurons, or how subsets "
  "of the group behave; we treat these as open questions rather than as shortcomings of that work, whose broader claims do "
  "not rest on this one example.")
P("Our central question is therefore: **when intervening on a group of neurons changes a model's answer, how much of that "
  "group is actually necessary?** We break it into three parts. _Uniqueness:_ can other, disjoint subsets produce the same "
  "change? _Interaction:_ are the neurons' effects additive, or do they depend on each other? _Transfer:_ does the same frozen "
  "group have the same effect on rephrasings of the question, and on other questions?")
P("Méloux et al. have shown in small networks trained on toy tasks that mechanistic explanations need not be unique [16]. "
  "We ask the corresponding question for pre-trained language models, at the level at which group demonstrations are "
  "actually made: a specific, selected group of neurons. Because a group of 12 neurons has only 4,096 subsets, these "
  "questions can be answered exhaustively rather than by sampling. We do this in two studies and three model families:")
BUL([
    f"**Study 1 (capital questions, Qwen3-0.6B).** Twelve neurons selected by zero-ablation effect pass the usual random-neuron "
    f"control (−{s1['zero'][0]*100:.0f}% margin vs. ~0%), their effects are almost perfectly additive and they transfer to "
    "rephrasings and new questions. Yet replacing them with values from _other_ questions has almost no effect. A group can look "
    "causal, additive and transferable without any evidence of question-specific content on the tested questions.",
    "**Study 2 (two-hop relation chains, Qwen3-1.7B, Llama-3.2-1B, Gemma-3-1B).** Patching 12 neurons with activations from "
    "the first-hop question reproduces the Scar → Simba shift in most chains of all three families, random neurons never do, "
    "and the frozen group has much smaller effects on other chains. Within the group, a few neurons usually suffice, and in many chains disjoint "
    "subsets each flip the answer.",
    "**A practical point for group demonstrations.** Exhaustive subset analysis is cheap at this scale, and a "
    "content-matched control (resampling or patching from other inputs) distinguishes generic from content-specific groups "
    "where a random-neuron control does not.",
])
P("We study neuron-level causal interventions and alternative subsets; we do not recover the full connectivity of circuits. "
  "Our models and our neuron groups are our own; we did not test Goodfire's 18 neurons and do not claim to refute their result.")

# ---------------------------------------------------------------- 2
H("2  Related work")
P("**Circuits and group interventions.** Circuit analyses localize behaviours to sets of components and validate them by "
  "intervention, from factual recall [4] and indirect-object identification [5] to automated circuit discovery [6] and "
  "attribution graphs built with transcoders [2, 3]. Goodfire's essay [1] argues that many individual MLP neurons are already "
  "interpretable and illustrates it with neuron groups selected from such graphs.")
P("**Redundancy and self-repair.** Wang et al. found backup heads that take over when primary heads are ablated [5], and "
  "McGrath et al. described the Hydra effect, in which downstream layers compensate for ablated ones [7]. These results suggest "
  "that a single ablation can under- or over-state a component's role; our exhaustive subset analysis quantifies this for "
  "neuron groups.")
P("**Non-uniqueness of explanations.** The work closest to ours in question is that of Méloux et al. [16], who ask whether "
  "mechanistic interpretability is identifiable. On small networks trained on toy tasks, where every candidate explanation "
  "can be enumerated, they show that explanations are generally not unique: several circuits reproduce the same behaviour "
  "and one circuit admits several interpretations. Chen et al. found structurally different, faithful and sparse "
  "circuits for the same task in large language models, including an ultra-sparse one in which no single edge is "
  "indispensable [17]. Our results agree that sufficient neuron groups need not be unique. They differ in setting and "
  "scope: instead of enumerating explanations of a small network built for the purpose, we take pre-trained language "
  "models of three families on natural-language questions and exhaustively test every subset of the specific neuron group "
  "that a group demonstration would report. This turns the identifiability concern into a measurement on real models, and "
  "adds the observation that alternatives are common when the model is unsure of its answer and less common when it is confident.")
P("**Choosing the intervention.** Zhang and Nanda showed that conclusions from activation patching depend on the corruption "
  "method and the metric [8], and attribution patching approximates patching effects with gradients [9]. Our Study 1 is a "
  "concrete instance of the first point: zero ablation and resampling disagree completely about the same neurons.")
P("**Two-hop reasoning.** Language models often know both facts of a two-hop question yet fail to compose them (the "
  "compositionality gap [10]); whether and where they resolve the bridge entity internally has been studied with probing and "
  "patching [11, 12]. Closest to our Study 2, Arora et al. traced circuits in the MLP-neuron basis and found that small "
  "neuron sets encode intermediate steps of a multi-hop city–state–capital task and can be steered to change the output "
  "[18]; we ask how much of such a set is necessary and whether disjoint parts of it can each do the job. We use two-hop "
  "chains because they give a natural, content-specific alternative answer (the bridge).")

# ---------------------------------------------------------------- 3
H("3  Method")
H("3.1  Models, neurons and interventions", 2)
P("We use the base (pre-trained) checkpoints of Qwen3-0.6B and Qwen3-1.7B [13], Llama-3.2-1B [14] and Gemma-3-1B [15]. The "
  "Llama and Gemma weights were obtained from the public unsloth mirrors of the official checkpoints. A **neuron** is one "
  "coordinate of the input to an MLP's down-projection, i.e. the gated MLP activation, at a given layer. All interventions act "
  "only at the last prompt token (the position from which the answer is generated) and leave all other positions untouched. "
  "We use four interventions on a neuron set S: _zero_ ablation; _mean_ ablation with the average activation over a separate "
  "set of prompts; _resampling_, which writes the activation the neuron takes on a different prompt; and _patching_, which "
  "writes the activation from a designated source prompt.")
H("3.2  Metric", 2)
P("For a prompt with correct answer token c and alternative token w, the margin is m = logit(c) − logit(w). In Study 1, w is "
  "the other option letter; in Study 2, c is the first token of the final answer (“Scar”) and w the first token of the bridge "
  "entity (“Simba”). The effect of intervening on S is Δ(S) = m(clean) − m(S), reported relative to the clean margin m₀. "
  "Δ/m₀ = 1 means the margin reaches zero; Δ/m₀ > 1 means the bridge is now preferred (the answer **flips**). Exact ties, "
  "which are frequent in bfloat16, are not counted as flips. We also record "
  "whether the most likely next token becomes the bridge and generate greedily under the intervention.")
H("3.3  Selecting 12 candidate neurons", 2)
P("All neurons of the relevant layers are ranked by a first-order estimate of their effect, a·∂m/∂a for zero ablation "
  "and (a_target − a_source)·∂m/∂a for patching [9]. Real single-neuron interventions are then measured for every neuron of "
  "layers 14 and 21 in Study 1, and for the top 300 by estimated effect in Study 2. In both studies the "
  "12 candidates are the 6 strongest by measured effect plus the next 6 by estimated effect, so that neurons that are weak "
  "alone are not excluded a priori. Candidates are selected on one prompt per question and frozen; rephrasings and other "
  "questions are never used for selection.")
H("3.4  Exhaustive subset analysis", 2)
P("For each frozen group we measure Δ(S) for all 2¹² = 4,096 subsets. From this table we derive: (i) **additivity**, the fit "
  "of Δ(S) by the sum of single-neuron effects (R² and slope through the origin); (ii) **pairwise interactions** "
  "I(i, j) = Δ({i, j}) − Δ({i}) − Δ({j}); (iii) **minimal sets**: an effective subset (Δ(S) above a threshold) none of whose "
  "proper subsets is effective; we use thresholds relative to the full group's effect (30/50/70%) and, in Study 2, the flip "
  "threshold Δ(S) > m₀; (iv) **alternatives**: pairs of minimal sets that share no neuron; and (v) **transfer**, by applying the "
  "same frozen group to other prompts.")
H("3.5  Controls, noise and procedure", 2)
P("Every result is compared with 12 random neurons matched in layer counts and receiving the same intervention (one random "
  "draw per question with a fixed seed; Study 1 also uses high-activation neurons with low estimated effect). Every batch "
  "of interventions contains an unmodified row, and Δ is computed against it. Study 1 runs in float32. Study 2 runs in bfloat16 "
  "because of GPU memory; there, we re-measured 30 subsets per chain with a different batch size and full (uncached) forward "
  "passes and report the maximum discrepancy as a noise floor "
  f"({own['Qwen3-1.7B']['noise_range'][0]:.3f}–{max(own[m]['noise_range'][1] for m in MODELS):.3f} logits across models). "
  "Interaction thresholds are set to max(5% of the full effect, three times the noise floor). Before each step, definitions and "
  "decision rules were written down; Appendix B lists every change made after seeing data. Confidence intervals are 95% "
  "bootstrap intervals over questions (chains). For transfer, the resampled units are valid rephrasings or ordered "
  "source–target chain pairs, without adjustment for observations that share a chain.")

# ---------------------------------------------------------------- 4
H("4  Study 1: a causal-looking group without evidence of question-specific content")
P(f"**Setup.** Qwen3-0.6B answers {s1['n']} capital questions in a two-option format (“A. Paris / B. Stockholm”), with a "
  "three-example prompt. Candidates come from layers 14 and 21 and are selected by zero-ablation effect. A separate set of 10 "
  "questions is held out for testing.")
P(f"**The group passes the usual checks.** Zero-ablating all 12 neurons lowers the margin by {ci(s1['zero'])} (Δ/m₀), while 12 "
  f"random neurons give {ci(s1['random'])}. The effects are almost perfectly additive (R² = {ci(s1['r2'])}, slope "
  f"{ci(s1['slope'])}), no pair of neurons interacts beyond the threshold ({s1['n_interacting_pairs']} pairs in total), and many "
  f"alternative subsets exist ({s1['q50_minimal_range'][0]}–{s1['q50_minimal_range'][1]} minimal sets at 50% of the full effect, "
  f"disjoint pairs in {s1['q50_disjoint_examples']}/{s1['n']} questions). The group transfers: across all 4,095 subsets, effects on "
  f"rephrasings correlate with the original at Spearman {s1['para_spearman_range'][0]:.2f}–{s1['para_spearman_range'][1]:.2f}. "
  f"The same neuron, (14, 692), is the strongest single neuron for all {s1['n']} questions, and remains so when the options are "
  f"swapped ({s1['top_still_top_swapped']}/{s1['n']}).")
P("**But we find no evidence of question-specific content.** Replacing the 12 neurons with their values from other capital questions "
  f"leaves the answer unchanged (same correct letter: {ci(s1['resample_same'])}; other correct letter: "
  f"{ci(s1['resample_diff'])}; mean of other capital questions: {ci(s1['task_mean'])}; Figure 1). The activations are nearly "
  f"constant across questions: the median absolute deviation of a neuron from its mean activation on the other discovery "
  f"questions is "
  f"{s1['act_dev_median']:.2f}, against a median magnitude of {s1['act_mag_median']:.2f}. Zero ablation removes a nearly constant signal "
  "that the answer depends on; within the tested questions we see no sign that it removes information about "
  "Paris specifically.")
P(f"**Replication on held-out questions.** Re-running the same selection procedure on the 10 test questions reproduces every "
  f"pattern: additivity R² {s1['test_r2_range'][0]:.3f}–{s1['test_r2_range'][1]:.3f}, {s1['test_interacting']} interacting pairs, "
  f"and (14, 692) as the strongest neuron in {s1['test_top692']}/{s1['test_n']} questions. The group frozen from the discovery "
  f"questions lowers the margin on the 10 new questions by {ci(s1['frozen_test_orig'])} (random group on the same prompts: "
  f"{ci(s1['frozen_test_random'])}).")
FIGURE("fig1_study1_interventions.png",
       "**Figure 1.** Study 1 (Qwen3-0.6B, capital questions). Relative margin drop when the 12 selected neurons are replaced, "
       "median and 95% bootstrap CI over 10 questions. Zero ablation produces by far the largest margin reduction; mean "
       "ablation and resampling the same neurons from other questions have effects close to zero.", 14.5)
P("Study 1 shows that the standard evidence (a large effect, a null random control, additivity, transfer) is compatible with a "
  "group that is irrelevant to the question's content. This motivated Study 2, which uses a task where the alternative answer "
  "is itself content: the bridge entity.")

# ---------------------------------------------------------------- 5
H("5  Study 2: two-hop chains and the Scar → Simba shift")
H("5.1  Task and competence", 2)
P("We wrote 20 two-hop chains in the spirit of the Lion King riddle: subject → bridge → answer (e.g. The Lion King → Simba → "
  "Scar; Eiffel Tower → France → Paris; Mona Lisa → Leonardo → Italy). Each chain has a first-hop question (answer: the "
  "bridge), a second-hop question (answer: the final answer), and three phrasings of the composed question (Appendix A). Before "
  "any neuron analysis we checked which models solve the chains (Table 1). All three know nearly every single hop but compose "
  "only about half of the chains, the compositionality gap [10]; their typical errors are to answer with the bridge itself "
  "(e.g. “Denmark” for the capital of Hamlet's country), the very shift that the Goodfire intervention induces. A chain is "
  "included when both hops and the first composed phrasing are answered correctly with m₀ ≥ 1.")
TABLE(["Model", "Hop 1", "Hop 2", "Composed (phr. 1)", "Composed (phr. 2)", "All four correct", "Included"],
      [[m, frac(comp[m]["hop1"], 20), frac(comp[m]["hop2"], 20), frac(comp[m]["comp1"], 20), frac(comp[m]["comp2"], 20),
        frac(comp[m]["solved"], 20), own[m]["n"]] for m in MODELS],
      "**Table 1.** Competence on the 20 chains (greedy generation). Included = chains entering the neuron analysis.",
      widths=[2.7, 1.5, 1.5, 2.1, 2.1, 2.0, 1.6])
H("5.2  Patching 12 neurons shifts the answer to the bridge", 2)
P("For each chain we select a patching group (P): neurons whose activations, copied from the first-hop question (“Who is the "
  "main character of The Lion King?”, answer Simba), most reduce the margin of the composed question. For comparison we also "
  "select a zero-ablation group (Z). Patching the P group flips the answer in "
  + ", ".join(f"{own[m]['P_flip']}/{own[m]['n']} chains ({m})" for m in MODELS)
  + "; under greedy generation the model then starts its answer with the bridge in "
  + ", ".join(f"{own[m]['P_gen_bridge']}/{own[m]['n']}" for m in MODELS)
  + " chains. A generation counts as starting with the bridge only if its text begins with the bridge string "
  "(case-insensitive); “Albert Einstein was born…” for the bridge Einstein is therefore not counted, which is why this "
  "count can be lower than the number of flips. The prefix count measures how the continuation begins, not whether it "
  "later mentions the correct answer as well. Twelve random neurons patched the same way never flip an answer (" + ", ".join(f"{own[m]['rand_flip']}/{own[m]['n']}" for m in MODELS)
  + "). Table 2 shows the generations for Qwen3-1.7B.")
g = own["Qwen3-1.7B"]["gen"]
ab_ = own["Qwen3-1.7B"]["answer_bridge"]
cut = lambda s: s if len(s) <= 24 else s[:23].rstrip() + "…"
TABLE(["Chain", "Answer → bridge", "No intervention", "P group patched", "Z group zeroed"],
      [[c, f"{ab_[c][0]} → {ab_[c][1]}", *[cut(g[c][k]) for k in range(3)]] for c in own["Qwen3-1.7B"]["chains"]],
      "**Table 2.** Greedy generations of Qwen3-1.7B under intervention on the full 12-neuron group (first line; “…” marks "
      "truncation for display).",
      widths=[2.2, 3.6, 3.4, 3.8, 3.4])
P(f"The two groups act differently (Figure 2). Patching both raises the bridge's logit (median change {med(3):+.1f}) and "
  f"lowers the answer's logit ({med(2):+.1f}): it pushes the model _towards_ Simba as well as away from Scar. Zero ablation "
  f"lowers the answer's logit more strongly (median {med(0):+.1f}) and raises the bridge less ({med(1):+.1f}; in "
  f"Qwen3-1.7B the median bridge change is only {float(np.median(dl['Qwen3-1.7B'][:, 1])):+.1f}). In the Z group, values taken from another "
  f"chain are about as damaging as zeros (Qwen3-1.7B: zero {ci(own['Qwen3-1.7B']['Z_rel'])}, resample "
  f"{ci(own['Qwen3-1.7B']['Z_resample'])}): unlike in Study 1, these neurons are sensitive to replacement with values from "
  "another chain. The same holds for the P group: patching it with activations from the composed question of a _different_ chain also "
  "lowers the margin substantially (Δ/m₀ " + ", ".join(f"{ci(own[m]['P_other'])}" for m in MODELS) + "). Part of the "
  "patching effect is therefore a disruption of the answer that does not require the matching bridge, and we do not claim "
  "that the group encodes the bridge exclusively.")
FIGURE("fig2_logit_decomposition.png",
       "**Figure 2.** Change in the answer logit (x) and the bridge logit (y) when the full 12-neuron group is intervened on, "
       "one point per chain and model. Patching from the first-hop question (filled) raises the bridge; zero ablation (open) "
       "mainly lowers the answer.", 10.5)

H("5.3  How much of the group is necessary?", 2)
P("Figure 3 addresses a question left open by the Goodfire example, how far the group can be reduced, for the Lion King "
  "chain: for every number of patched neurons it "
  "shows the distribution of effects over all subsets of that size. The full group is not necessary. The smallest "
  f"answer-flipping subset has {lk_min['Qwen3-1.7B']} neurons in Qwen3-1.7B, {lk_min['Llama-3.2-1B']} in Llama-3.2-1B and "
  f"{lk_min['Gemma-3-1B']} in Gemma-3-1B. The curves are smooth rather than step-like: effects accumulate neuron by neuron, "
  "and which neurons are chosen matters (the width of the 5–95% band and the gap between the median and the strongest subset).")
FIGURE("fig3_necessity_lion_king.png",
       "**Figure 3.** The Lion King chain. Δ/m₀ against the number of patched neurons, over all subsets of each size (band: "
       "5–95%, solid: median, dashed: strongest subset). Above the dotted line the model prefers “Simba”.", 16)
P("Across chains, the smallest flipping subset ranges over "
  + "; ".join(f"{m}: {', '.join(map(str, own[m]['min_sizes']))}" for m in MODELS)
  + " neurons (chains in which even the full group does not flip are omitted).")

H("5.4  Uniqueness: disjoint subsets that flip the answer", 2)
P("A key question is whether the same shift can be produced by _different_ neurons. We count pairs of minimal flipping subsets "
  "that share no neuron. Among the chains whose answer the full group flips, such pairs exist in "
  + ", ".join(f"{own[m]['chains_disjoint']}/{own[m]['P_flip']} ({m})" for m in MODELS)
  + ". In these chains the 12-neuron group is not a single indispensable unit: two non-overlapping parts of it each suffice. "
  "We note a caveat about our own examples: the four four-neuron subsets of Qwen3-1.7B that we checked by generation on the "
  "Lion King chain (each produces “Simba”) all contain the same two neurons; disjoint alternatives are identified by the "
  "margin criterion and were not checked by generation.")
P(f"Whether disjoint alternatives exist appears to depend on how confident the model is (Figure 4). Pooling the three models "
  f"and counting only the {mt['lo_n'] + mt['hi_n']} chains whose answer the full group flips, {mt['lo_disj']} of {mt['lo_n']} "
  f"chains with m₀ < 4 have disjoint flipping subsets, against {mt['hi_disj']} of {mt['hi_n']} chains with m₀ ≥ 4. In the other "
  f"{mt['hi_n'] - mt['hi_disj']} confident chains every pair of flipping subsets shares at least one neuron, so no disjoint "
  f"alternative exists among the 12 selected neurons. The remaining {mt['noflip_n']} included chains are not flipped by the "
  f"full group, and exhaustive enumeration finds no flipping subset in them either; all of them have "
  f"m₀ ≥ {mt['noflip_m0_min']:.1f}. This is an exploratory "
  "observation: the cut-off of 4 was chosen after inspecting the Qwen3-1.7B results, and margins and chains are confounded.")
FIGURE("fig4_margin_vs_alternatives.png",
       "**Figure 4.** Number of disjoint pairs of minimal flipping subsets against the clean margin, one point per model "
       "and chain. Disjoint alternatives are more common below the margin cut-off than above it. Chains whose answer the full "
       "group does not flip are omitted. Exploratory.", 11)

H("5.5  Interaction", 2)
P("Within the P groups, subset effects are well fit by a scaled additive model: R² of the additive fit is "
  + ", ".join(f"{ci(own[m]['r2'])} ({m})" for m in MODELS)
  + ", with slopes below one ("
  + ", ".join(f"{ci(own[m]['slope'])}" for m in MODELS)
  + "), i.e. on average subsets do less than the sum of their neurons' single effects. Pairs of neurons whose joint effect departs from additivity "
  "by more than the threshold are rare (" + ", ".join(f"{own[m]['n_pairs_interacting']} of {66*own[m]['n']} pairs ({m})" for m in MODELS)
  + "). We cannot exclude smaller interactions below the bfloat16 noise floor.")

H("5.6  Transfer", 2)
P("The ranking of subset effects carries over to other phrasings of the same chain (those with m₀ ≥ 1): across all 4,095 "
  "subsets, Spearman correlations with the original effects are " + ", ".join(f"{ci(own[m]['para_spearman'])} ({m})" for m in MODELS)
  + ". Applied to the composed question of a _different_ chain (patching from that chain's own first-hop question), the same "
  "neurons have little effect: Δ/m₀ " + ", ".join(f"{ci(own[m]['cross_rel'])} ({m})" for m in MODELS)
  + ", against " + ", ".join(f"{ci(own[m]['P_rel'])}" for m in MODELS)
  + " on their own chain. Unlike in Study 1, the selected groups act much more strongly on their own chain than on the "
  "other tested chains.")

H("6  The same pattern in three model families")
P(f"To compare models on identical questions we restrict to the {len(s2['common'])} chains included for all three "
  f"({', '.join(s2['common'])}). Table 3 and Figure 5 summarize the three answers. Every family shows the Scar → Simba shift "
  "under patching and none under random patching, strong transfer to rephrasings and weak transfer to other chains, and "
  "small sufficient subsets. The families differ in degree: disjoint alternatives are least common in Qwen3-1.7B, whose "
  "included chains also have the largest margins.")
TABLE(["", *MODELS],
      [["Patching flips the answer", *[frac(com[m]["P_flip"], com[m]["n"]) for m in MODELS]],
       ["Generation starts with the bridge", *[frac(com[m]["P_gen_bridge"], com[m]["n"]) for m in MODELS]],
       ["Random neurons flip the answer", *[frac(com[m]["rand_flip"], com[m]["n"]) for m in MODELS]],
       ["Δ/m₀, full group", *[ci(com[m]["P_rel"]) for m in MODELS]],
       ["Smallest flipping subset (neurons)", *[", ".join(map(str, com[m]["min_sizes"])) for m in MODELS]],
       ["Chains with disjoint flipping subsets", *[frac(com[m]["chains_disjoint"], com[m]["n"]) for m in MODELS]],
       ["Transfer: same chain, Spearman", *[ci(com[m]["para_spearman"]) for m in MODELS]],
       ["Transfer: other chains, Δ/m₀", *[ci(com[m]["cross_rel"]) for m in MODELS]]],
      f"**Table 3.** The {len(s2['common'])} chains shared by all three models. Medians with 95% bootstrap CIs. Fractions "
      f"use all {len(s2['common'])} shared chains as denominator; subset sizes are listed only for chains the full group flips.",
      widths=[5.4, 3.7, 3.7, 3.7])
FIGURE("fig5_cross_model.png",
       "**Figure 5.** Left: share of the shared chains whose answer flips when 12 selected (blue) or 12 random (gray) neurons "
       "are patched. Right: effect of the frozen group on other phrasings of the same chain (blue) and on other chains (orange).", 16)

# ---------------------------------------------------------------- 7
H("7  Discussion")
P("**How much of the group is necessary?** In our experiments, rarely all of it. A handful of neurons usually suffices to "
  "produce the shift, effects accumulate smoothly, and in many chains two disjoint parts of the group can each produce it. A "
  "statement such as “removing these 18 neurons changes the answer” is therefore an existence proof for _a_ sufficient "
  "intervention, not a description of the necessary mechanism. The flip of an answer also depends on how far the model was "
  "from flipping: when the margin is small, many small subsets cross the line and disjoint ones are common; when it is "
  "large, the flipping subsets usually overlap.")
P("**Uniqueness is not a property of the group alone.** Whether a group looks unique depends on the threshold (half the "
  "effect, or a flip), on the margin, and on the 12 neurons that were selected. Minimal sets are minimal only relative to "
  "the candidate group; they are not the model's circuit.")
P("**A random control is necessary but not sufficient.** Study 1 produced a group that beats random neurons, is additive, "
  "and transfers, yet showed no evidence of carrying anything specific to the tested questions. The control that exposed this was a content-matched one: "
  "resampling activations from other questions. In Study 2 the same checks behave differently: values from other chains "
  "disrupt the answer and the groups act much more weakly on other chains, so the groups are tied to their chain, although "
  "patching from another chain shows that part of the effect is not specific to the bridge. We suggest that group demonstrations report (i) a "
  "content-matched control in addition to random neurons, (ii) the size distribution of sufficient subsets, and (iii) the "
  "clean margin, since it shapes every necessity and uniqueness conclusion.")
P("**Relation to the Lion King example.** We reproduced the _kind_ of effect reported by Goodfire in three smaller models from "
  "three families, using activation patching rather than removal, 12 rather than 18 neurons, and our own selection procedure. "
  "Our results do not say anything about their specific neurons; they show that the structural questions such an example "
  "leaves open have non-trivial and model-dependent answers.")

H("8  Limitations")
BUL([
    "**Scale.** Up to 11 chains per model and models of 0.6–1.7B parameters; results are pilot-scale estimates. Goodfire's "
    "example uses a 32B model, which did not fit in the GPU memory available to us.",
    "**Candidate groups.** All uniqueness and necessity statements are relative to 12 neurons chosen by single-neuron and "
    "first-order criteria, which favour neurons that matter individually and may miss purely synergistic ones.",
    "**Controls.** One layer-matched random group is drawn per question (fixed seed), not a distribution of random groups.",
    "**Single position.** Interventions act only on the last prompt token; computation at earlier positions is not tested.",
    "**Precision.** Study 2 runs in bfloat16 (noise floor up to "
    f"{max(own[m]['noise_range'][1] for m in MODELS):.2f} logits); small interactions may be invisible.",
    "**Exploratory analysis.** The margin cut-off in Section 5.4 was chosen after seeing data. Several definitions were revised "
    "between pilot runs (Appendix B).",
    "**Task format.** First-token margins between answer and bridge; multi-token answers are judged by the first token, "
    "generation checks the first line.",
])

H("9  Conclusion")
P("A group intervention that changes a model's answer tells us that the group matters; it does not tell us how much of it is "
  "needed or whether other neurons could do the same. Measuring every subset of 12-neuron groups in four models, we find that "
  "the answer-changing effect is usually carried by a few neurons, that disjoint parts of a group can each produce it when the "
  "model is not confident, that such groups transfer to rephrasings but act much more weakly on other chains, and that a group can pass a "
  "random-neuron control without any evidence of carrying question-specific content. Neuron groups are useful handles on model behaviour, but their "
  "internal structure has to be measured, not assumed.")

H("Code and data")
P("All code, the question sets, the saved results of every run reported here and the scripts that compute every number "
  "and figure in this paper are available at https://github.com/omtekyav/causal-neuron-circuits (MIT license).")

H("Acknowledgment")
P("Generative AI tools were used for writing analysis code and for language editing. All experiments were run and all numbers "
  "were computed from saved result files by the author, who is responsible for the content.")

H("References")
REFS = [
    "Goodfire. “Neural networks are already clean.” 2026. https://already-clean.goodfire.com/",
    "E. Ameisen et al. “Circuit tracing: Revealing computational graphs in language models.” Transformer Circuits Thread, 2025.",
    "J. Lindsey et al. “On the biology of a large language model.” Transformer Circuits Thread, 2025.",
    "K. Meng, D. Bau, A. Andonian, Y. Belinkov. “Locating and editing factual associations in GPT.” NeurIPS, 2022.",
    "K. Wang, A. Variengien, A. Conmy, B. Shlegeris, J. Steinhardt. “Interpretability in the wild: A circuit for indirect "
    "object identification in GPT-2 small.” ICLR, 2023.",
    "A. Conmy, A. Mavor-Parker, A. Lynch, S. Heimersheim, A. Garriga-Alonso. “Towards automated circuit discovery for "
    "mechanistic interpretability.” NeurIPS, 2023.",
    "T. McGrath, M. Rahtz, J. Kramár, V. Mikulik, S. Legg. “The Hydra effect: Emergent self-repair in language model "
    "computations.” arXiv:2307.15771, 2023.",
    "F. Zhang, N. Nanda. “Towards best practices of activation patching in language models: Metrics and methods.” ICLR, 2024.",
    "J. Kramár, T. Lieberum, R. Shah, N. Nanda. “AtP*: An efficient and scalable method for localizing LLM behaviour to "
    "components.” arXiv:2403.00745, 2024.",
    "O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, M. Lewis. “Measuring and narrowing the compositionality gap in "
    "language models.” Findings of EMNLP, 2023.",
    "S. Yang, E. Gribovskaya, N. Kassner, M. Geva, S. Riedel. “Do large language models latently perform multi-hop "
    "reasoning?” ACL, 2024.",
    "E. Biran, O. Yoran, A. Globerson, M. Geva. “Hopping too late: Exploring the limitations of large language models on "
    "multi-hop queries.” EMNLP, 2024.",
    "Qwen Team. “Qwen3 technical report.” arXiv:2505.09388, 2025.",
    "A. Grattafiori et al. “The Llama 3 herd of models.” arXiv:2407.21783, 2024.",
    "Gemma Team. “Gemma 3 technical report.” arXiv:2503.19786, 2025.",
    "M. Méloux, S. Maniu, F. Portet, M. Peyrard. “Everything, everywhere, all at once: Is mechanistic interpretability "
    "identifiable?” ICLR, 2025.",
    "X. Chen, M. Jin, J. Niu, Y. Yin, J. Zhao, B. Guo, D. N. Metaxas, Z. Wang, Y. Yue, G. Penn. “All circuits lead to "
    "Rome: Rethinking functional anisotropy in circuit and sheaf discovery for LLMs.” ICML, PMLR 306, 2026.",
    "A. Arora, Z. Wu, J. Steinhardt, S. Schwettmann. “Language model circuits are sparse in the neuron basis.” ICML, "
    "PMLR 306, 2026.",
]
for i, ref in enumerate(REFS, 1):
    p = d.add_paragraph()
    _runs(p, f"[{i}] {ref}", size=9)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)

# ---------------------------------------------------------------- ekler
H("Appendix A  The 20 two-hop chains")
import json
chains = [json.loads(l) for l in (HERE.parent / "data" / "chains.jsonl").read_text(encoding="utf-8").splitlines()]
rows = {}
for r in chains:
    rows.setdefault(r["chain"], {})[r["kind"]] = r
inc = {m: set(own[m]["chains"]) for m in MODELS}
TABLE(["Chain", "Composed question (phrasing 1)", "Bridge", "Answer", "Included in"],
      [[c, v["comp1"]["prompt"].split("Q: ")[-1].replace("\nA:", ""), v["comp1"]["bridge"], v["comp1"]["answer"],
        ", ".join(m.split("-")[0] for m in MODELS if c in inc[m]) or "—"] for c, v in rows.items()],
      "**Table A1.** Chains, bridges and answers. Prompts use a three-example question–answer prefix.",
      widths=[1.6, 8.6, 2.0, 1.9, 2.6], size=7.5)

H("Appendix B  Pre-specified decisions and changes")
BUL([
    "Pilot (Study 1): the effect threshold was first fixed at 50% of the clean margin. No subset reached it (all 4,096 measured), "
    "so for the uniqueness analysis it was replaced, before further data were collected, by thresholds relative to the full "
    "group's effect; the 50%-of-margin result is reported unchanged.",
    "The original minimality check only removed one neuron at a time; it was corrected to require that no proper subset be "
    "effective (relevant when effects are not monotone). Results reported here use the corrected definition.",
    "The first pilot computed mean ablation from the same questions (including the target); it was replaced by an independent "
    "reference set, and resampling from other questions was added.",
    "Study 1 was re-run in float32 after bfloat16 logits proved too coarse for single-neuron effects; Study 2 uses bfloat16 with "
    "a measured noise floor.",
    "Study 2 definitions, inclusion rules and success criterion were written before the first run; the same frozen protocol was "
    "applied unchanged to Llama-3.2-1B and Gemma-3-1B. The margin cut-off (m₀ = 4) in Section 5.4 is post hoc.",
    "After review, the counts of disjoint alternatives in Section 5.4 and Figure 4 were restricted to chains whose answer the "
    "full group flips (exhaustive enumeration finds no flipping subset in the four excluded chains), and the subset flip "
    "threshold was changed from Δ(S) ≥ m₀ to Δ(S) > m₀ so that exact ties are not counted, matching the full-group "
    "definition. This changes the Qwen3-1.7B disjoint-alternative count from 4/10 to 3/10 and three minimal subset sizes.",
])

path = OUT / ("neuron_circuits_affil.docx" if AFFIL else "neuron_circuits.docx")
d.core_properties.title = "Are Causal Neuron Circuits Unique and Transferable?"
d.core_properties.author = "Ömer Tekin Yavuz"
d.save(path)
print("kaydedildi", path)
pdf = path.with_suffix(".pdf")
ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; try {{ $doc = $w.Documents.Open('{path}'); "
      f"$doc.ExportAsFixedFormat('{pdf}', 17, $false, 0, 0, 1, 1, 0, $true, $true, 0, $true, $true, $false); "
      f"$doc.Close([ref]$false) }} finally {{ $w.Quit() }}")
subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=False)
print("pdf", pdf.exists(), pdf)
