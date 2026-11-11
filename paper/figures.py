"""Makale grafikleri (kayıtlı sonuçlardan). Palet: dataviz referans paleti, açık yüzey; tek eksen; ince işaretler."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stats import RUNS, study1, study2  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # mavi, turuncu, su yeşili (ilk üç slot, tüm çiftler geçerli)
MODEL_COLOR = dict(zip(RUNS, SERIES))
MODEL_MARK = dict(zip(RUNS, ["o", "s", "^"]))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.titlesize": 9.5, "axes.titlecolor": INK,
})


def fig_study1(s1):
    rows = [("Zero ablation", s1["zero"], SERIES[0]),
            ("Mean of independent A/B questions", s1["ref_mean"], SERIES[0]),
            ("Mean of other capital questions", s1["task_mean"], SERIES[0]),
            ("Resample, same correct letter", s1["resample_same"], SERIES[0]),
            ("Resample, other correct letter", s1["resample_diff"], SERIES[0]),
            ("Zero ablation, 12 random neurons", s1["random"], MUTED)]
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, (m, lo, hi), col) in zip(y, rows):
        ax.barh(yi, m, height=0.55, color=col, zorder=2)
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        ax.text(max(m, hi) + 0.008, yi, f"{m:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0, color=BASE, lw=1)
    ax.set_xlabel("Relative margin drop Δ/m₀ when all 12 selected neurons are replaced")
    ax.set_xlim(-0.05, 0.36)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_study1_interventions.png")
    plt.close(fig)


def fig_logits(s2):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for m, v in s2["models"].items():
        d = np.array(v["own"]["dlog"])
        ax.scatter(d[:, 0], d[:, 1], s=26, marker=MODEL_MARK[m], facecolor="white", edgecolor=SERIES[0], linewidth=1.2, zorder=3)
        ax.scatter(d[:, 2], d[:, 3], s=26, marker=MODEL_MARK[m], color=SERIES[1], edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color=BASE, lw=1); ax.axvline(0, color=BASE, lw=1)
    ax.set_xlabel("Change in answer logit (e.g. “Scar”)")
    ax.set_ylabel("Change in bridge logit (e.g. “Simba”)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc="white", mec=SERIES[0], mew=1.2, label="Zero ablation (Z group)"),
         Line2D([], [], ls="", marker="o", color=SERIES[1], label="Hop-1 patching (P group)")]
    h += [Line2D([], [], ls="", marker=MODEL_MARK[m], color=INK2, mfc="none", label=m) for m in s2["models"]]
    ax.legend(handles=h, loc="upper left", fontsize=7.5, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_logit_decomposition.png")
    plt.close(fig)


def fig_necessity(s2, chain="lk"):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
    sizes = np.array([bin(s).count("1") for s in range(4096)])
    for ax, (m, deltas) in zip(axes, s2["deltas"].items()):
        R = s2["R"][m]
        e = next(e for e in R["chains"] if e["chain"] == chain)
        rel = deltas[chain]["P_hop1"].astype(float) / e["m0"]
        q = np.array([np.quantile(rel[sizes == n], [0.05, 0.5, 0.95]) if n else [0, 0, 0] for n in range(13)])
        mx = np.array([rel[sizes == n].max() for n in range(13)])
        n = np.arange(13)
        col = MODEL_COLOR[m]
        ax.fill_between(n, q[:, 0], q[:, 2], color=col, alpha=0.18, lw=0, label="5–95% of subsets")
        ax.plot(n, q[:, 1], color=col, lw=2, label="median subset")
        ax.plot(n, mx, color=col, lw=1, ls="--", label="strongest subset")
        ax.axhline(1.0, color=INK2, lw=1, ls=":", label="zero margin (Δ/m₀ = 1); flips above")
        ax.set_title(f"{m} (m₀ = {e['m0']:.2f})")
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
        ax.set_xlabel("Neurons patched (of 12)")
    axes[0].set_ylabel("Δ/m₀")
    # Açıklama eğrilerin üstüne binmesin: şeklin altında, renksiz (gri) örneklerle
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=INK2, alpha=0.25, label="5–95% of subsets"),
               Line2D([], [], color=INK2, lw=2, label="median subset"),
               Line2D([], [], color=INK2, lw=1, ls="--", label="strongest subset"),
               Line2D([], [], color=INK2, lw=1, ls=":", label="zero margin (Δ/m₀ = 1); flips above")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig3_necessity_lion_king.png")
    plt.close(fig)


def fig_margin(s2):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for m, v in s2["models"].items():
        o = v["own"]
        cs = [c for c in o["flip"] if o["full_flip"][c]]
        x = [o["m0"][c] for c in cs]
        y = [np.log10(1 + o["flip"][c]["disjoint"]) for c in cs]
        ax.scatter(x, y, s=30, marker=MODEL_MARK[m], color=MODEL_COLOR[m], edgecolor="white", linewidth=0.6, label=m, zorder=3)
    ax.axvline(4, color=INK2, lw=1, ls=":")
    ax.text(4.15, 3.35, "m₀ = 4 (chosen post hoc)", fontsize=7, color=INK2)
    ax.set_yticks([np.log10(1 + n) for n in (0, 10, 100, 1000)], ["0", "10", "100", "1000"])
    ax.set_xlabel("Clean margin m₀ (answer vs. bridge logit)")
    ax.set_ylabel("Disjoint pairs of\nanswer-flipping subsets")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_margin_vs_alternatives.png")
    plt.close(fig)


def fig_models(s2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    models = list(s2["models"])
    x = np.arange(len(models))
    w = 0.36
    for i, (key, lab, col) in enumerate((("P_flip", "12 selected neurons", SERIES[0]),
                                         ("rand_flip", "12 random neurons", MUTED))):
        vals = [100 * s2["models"][m]["common"][key] / s2["models"][m]["common"]["n"] for m in models]
        a1.bar(x + (i - 0.5) * w, vals, w * 0.92, color=col, label=lab, zorder=2)
        for xi, v in zip(x, vals):
            a1.text(xi + (i - 0.5) * w, v + 2, f"{v:.0f}%", ha="center", fontsize=7, color=INK)
    a1.set_xticks(x, models); a1.set_ylim(0, 128); a1.set_yticks([0, 20, 40, 60, 80, 100])
    a1.set_ylabel("Chains whose answer flips (%)")
    a1.set_title(f"Hop-1 patching flips the answer ({len(s2['common'])} shared chains)")
    a1.legend(fontsize=7, loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.2)
    for i, (key, lab, col) in enumerate((("para_rel", "Same chain, other phrasings", SERIES[0]),
                                         ("cross_rel", "Other chains", SERIES[1]))):
        mids = [s2["models"][m]["common"][key] for m in models]
        a2.bar(x + (i - 0.5) * w, [c[0] for c in mids], w * 0.92, color=col, label=lab, zorder=2)
        a2.errorbar(x + (i - 0.5) * w, [c[0] for c in mids], yerr=[[c[0] - c[1] for c in mids], [c[2] - c[0] for c in mids]],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    a2.set_xticks(x, models); a2.set_ylabel("Δ/m₀ of the frozen group")
    a2.set_title("Transfer of the frozen 12-neuron group")
    a2.legend(fontsize=7, loc="upper left")
    for a in (a1, a2):
        a.grid(axis="x", visible=False)
        a.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_cross_model.png")
    plt.close(fig)


if __name__ == "__main__":
    s1, s2 = study1(), study2()
    fig_study1(s1); fig_logits(s2); fig_necessity(s2); fig_margin(s2); fig_models(s2)
    print(sorted(p.name for p in OUT.glob("*.png")))
