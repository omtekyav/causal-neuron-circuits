"""arXiv için LaTeX kaynağı: metin, sayılar ve şekiller build_paper.py ile aynı kaynaktan.

build_paper.py'nin içerik bölümü, Word yerine LaTeX yazan P/H/BUL/FIGURE/TABLE ile çalıştırılır.
Kullanım: python -X utf8 build_tex.py  →  paper/out/tex/ (main.tex + figures/) ve paper/out/arxiv_source.zip
AFFIL="<kurum adı>" verilirse yazar bloğuna kurum satırı eklenir (build_paper.py ile aynı).
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
SRC = (HERE / "build_paper.py").read_text(encoding="utf-8")
OUT = HERE / "out" / "tex"
TITLE = "Are Causal Neuron Circuits Unique and Transferable?"

# build_paper.py'yi üç parçaya böl: başlangıç (veri + yardımcılar), içerik, kayıt.
pre = SRC[:SRC.index("# ---------------------------------------------------------------- belge")]
body = SRC[SRC.index("# ---------------------------------------------------------------- başlık"):SRC.index("path = OUT")]


# ---------------------------------------------------------------- metin → LaTeX
UNI = {
    "Δ": r"$\Delta$", "₀": r"$_0$", "→": r"$\rightarrow$", "−": r"$-$", "≥": r"$\geq$", "≤": r"$\leq$",
    "×": r"$\times$", "·": r"$\cdot$", "∂": r"$\partial$", "²": r"$^2$", "≈": r"$\approx$", "…": r"\ldots{}",
    "–": "--", "—": "---", "“": "``", "”": "''", "’": "'", "‘": "`", "ı": r"{\i}", "İ": r"\.I", "ş": r"\c{s}",
    "Ş": r"\c{S}", "ğ": r"\u{g}", "Ğ": r"\u{G}", "ç": r"\c{c}", "Ç": r"\c{C}", "ö": r"\"o", "Ö": r"\"O",
    "ü": r"\"u", "Ü": r"\"U", "é": r"\'e", "á": r"\'a", "à": r"\`a",
}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"$\sim$", "^": r"\^{}"}


def esc(s: str) -> str:
    s = s.replace("2¹²", "\x00")
    out = []
    for part in re.split(r"(https?://[^\s)]+[^\s).,;])", s):
        if part.startswith("http"):
            out.append(r"\url{" + part + "}")
            continue
        buf = []
        for ch in part:
            buf.append(SPECIAL.get(ch) or UNI.get(ch) or ch)
        out.append("".join(buf))
    s = "".join(out).replace("\x00", "$2^{12}$")
    bad = sorted({c for c in s if ord(c) > 127})
    if bad:
        raise ValueError(f"LaTeX karşılığı tanımsız karakter: {bad}")
    return s


def md(text: str) -> str:
    """**kalın** ve _italik_ işaretleri (build_paper._runs ile aynı kural)."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|_[^_]+_)", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(r"\textbf{" + esc(part[2:-2]) + "}")
        elif part.startswith("_") and part.endswith("_") and len(part) > 2:
            out.append(r"\emph{" + esc(part[1:-1]) + "}")
        else:
            out.append(esc(part))
    return "".join(out)


def caption_parts(caption: str):
    """'**Table 2.** metin' → ('Table 2.', 'metin'); numaralar makaledeki gibi elle."""
    m = re.match(r"\*\*((?:Table|Figure) [A-Z]?\d+\.)\*\*\s*(.*)", caption, re.S)
    return (m.group(1), m.group(2)) if m else ("", caption)


# ---------------------------------------------------------------- Word yerine LaTeX
class Sink:
    """Word'e özgü çağrıları (d.add_paragraph, paragraph_format …) sessizce yutar."""
    def __getattr__(self, _):
        return self

    def __call__(self, *a, **k):
        return self

    def __setattr__(self, *_):
        pass


L: list[str] = []
REF_ITEMS: list[str] = []
state = {"abstract": None, "refs": False}


def P(text, italic_lead=None, align=None, size=None):
    if state["abstract"] is None and size == 10:      # build_paper'da yalnız özet size=10 ile yazılır
        state["abstract"] = md(text)
        return Sink()
    L.append(md(text) + "\n")
    return Sink()


def _runs(p, text, size=None):
    if state["refs"]:
        REF_ITEMS.append(md(re.sub(r"^\[\d+\]\s*", "", text)))


def H(text, lvl=1):
    if text == "References":
        state["refs"] = True
        return
    state["refs"] = False
    title = r"\quad ".join(esc(x) for x in re.split(r"\s{2,}", text))
    if text.startswith("Appendix A"):
        L.append(r"\clearpage" + "\n")
    L.append((r"\section*{" if lvl == 1 else r"\subsection*{") + title + "}\n")


def BUL(items):
    L.append(r"\begin{itemize}" + "\n" + "".join(r"  \item " + md(it) + "\n" for it in items) + r"\end{itemize}" + "\n")


FIGS: list[str] = []


def FIGURE(name, caption, width_cm=15.5):
    FIGS.append(name)
    lab, cap = caption_parts(caption)
    w = min(width_cm, 13.97) / 13.97
    L.append(r"\begin{figure}[htbp]\centering" + "\n"
             + rf"\includegraphics[width={w:.3f}\linewidth]{{figures/{name}}}" + "\n"
             + r"\caption*{\textbf{" + esc(lab) + "} " + md(cap) + "}\n" + r"\end{figure}" + "\n")


def TABLE(header, rows, caption, widths=None, size=8.5):
    lab, cap = caption_parts(caption)
    n = len(header)
    widths = widths or [1] * n
    tot = sum(widths)
    cols = "".join(rf">{{\raggedright\arraybackslash}}p{{{w / tot:.3f}\dimexpr\linewidth-{2 * n}\tabcolsep\relax}}"
                   for w in widths)
    fs = r"\footnotesize" if size < 8.5 else r"\small"
    lines = [r"\begin{table}[htbp]\centering" + fs,
             r"\caption*{\textbf{" + esc(lab) + "} " + md(cap) + "}",
             r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(r"\textbf{" + md(h) + "}" for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(md(str(v)) for v in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    L.append("\n".join(lines))


# ---------------------------------------------------------------- çalıştır
ns: dict = {"__file__": str(HERE / "build_paper.py"), "__name__": "build_tex"}
exec(compile(pre, "build_paper.py:pre", "exec"), ns)
ns.update(P=P, _runs=_runs, H=H, BUL=BUL, FIGURE=FIGURE, TABLE=TABLE, d=Sink(), rule=lambda *a, **k: None)
exec(compile(body, "build_paper.py:body", "exec"), ns)
AFFIL = ns["AFFIL"]

author_lines = [r"\textbf{\"Omer Tekin Yavuz}", "R\\&D and AI Engineer"] + ([esc(AFFIL)] if AFFIL else []) + \
               ["Bursa, Turkey", r"\texttt{omtekyav@gmail.com}"]
refs = "\n".join(r"\bibitem{r" + str(i) + "} " + r for i, r in enumerate(REF_ITEMS, 1))

TEX = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage[letterpaper,textwidth=5.5in,textheight=9in,top=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}
\captionsetup{font=small,justification=justified,singlelinecheck=false}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5.5pt}
\renewcommand{\refname}{References}
% Şekiller metinle aynı sayfada kalsın; yarı boş "yalnız şekil" sayfası oluşmasın
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}

\title{""" + esc(TITLE) + r"""}
\author{""" + r" \\ ".join(author_lines) + r"""}
\date{}

\makeatletter
\renewcommand{\@maketitle}{%
  \begin{center}
  \rule{\linewidth}{4pt}\par\vspace{10pt}
  {\LARGE\bfseries \@title\par}\vspace{8pt}
  \rule{\linewidth}{1pt}\par\vspace{18pt}
  \begin{tabular}[t]{c}\@author\end{tabular}\par
  \end{center}\vspace{6pt}}
\makeatother

\begin{document}
\maketitle

\begin{abstract}
""" + state["abstract"] + r"""
\end{abstract}

""" + "\n".join(L).split(r"\clearpage")[0] + r"""
\begin{thebibliography}{99}\small
""" + refs + r"""
\end{thebibliography}
""" + "\n".join(L).split(r"\clearpage", 1)[1] + r"""
\end{document}
"""

OUT.mkdir(parents=True, exist_ok=True)
for p in OUT.iterdir():   # klasörü değil içini temizle (açık bir kabuk içinde olabilir)
    shutil.rmtree(p) if p.is_dir() else p.unlink()
(OUT / "figures").mkdir()
(OUT / "main.tex").write_text(TEX, encoding="utf-8")
for f in FIGS:
    shutil.copy(HERE / "figures" / f, OUT / "figures" / f)
z = HERE / "out" / ("arxiv_source_affil.zip" if AFFIL else "arxiv_source.zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(OUT).as_posix())
print("yazıldı", OUT / "main.tex", "|", len(FIGS), "şekil |", len(REF_ITEMS), "kaynak |", z.name)
