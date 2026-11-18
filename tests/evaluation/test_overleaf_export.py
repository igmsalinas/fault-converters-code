from __future__ import annotations

from pathlib import Path

from src.evaluation.overleaf_export import (
    export_overleaf_bundle,
    markdown_to_ieeetran_tex,
    validate_overleaf_project,
)


def _sample_markdown() -> str:
    return """---
title: "Sample Variable-Load Converter Paper"
authors:
  - name: "Ignacio Martin-Salinas"
    affiliation: "UC3M, Leganés, Spain"
    corresponding: true
  - name: "Adrián Amor-Martín"
    affiliation: "UC3M, Leganés, Spain"
keywords:
  - "Frequency-response analysis (FRA)"
  - "Contrastive learning"
bibliography: "references.bib"
---

# Sample Variable-Load Converter Paper

**Ignacio Martin-Salinas\\*, Adrián Amor-Martín**
*UC3M, Leganés, Spain*

---

## Abstract

We achieve $\\mathrm{F1} = 0.9417 \\pm 0.0128$ and recover $89.6\\%$ ESR faults with `CARLA-Transformer` across $4\\times$ load swings.

---

## Research Highlights

- Bullet ignored in main IEEE body cleanly.

---

## 1. Introduction

Previous work [@darban2025carla; @vaswani2017attention] fails under 100% load swings—see Table 1 and Figure 1.

![Figure 1: System overview](figures/fig01_system.png)
*Figure 1. Detailed vector overview of `CARLA-Transformer` with $R_{\\mathrm{out}} \\in [2, 8]\\,\\Omega$.*

## 3. Physics Model

### 3.2 Exact SSA Model on `dataset_01`

We compute the exact transfer ratio using Eq. (2) and (Eq. 2):

$$
G_{vd}(s) = V_{\\mathrm{in}} \\frac{1}{1 + sRC}
$$

**Table 1. Converter parameters for `dataset_02`.**

| Param | Value |
| :--- | :---: |
| Load | $4\\,\\Omega$ |

See Eq. (1) on `dataset_02` (`mean ± SD`) and **highlighted note** in *italic text* (Section 3.2).

## Data and Code Availability

Code is hosted via `uv` and `pytest`.

## Acknowledgments

Supported by UC3M, Leganés, Spain.
"""


def test_markdown_to_ieeetran_converts_structure_citations_equations_tables_and_figures() -> None:
    tex = markdown_to_ieeetran_tex(
        _sample_markdown(),
        table_files={1: "tables/table01_converter_parameters.tex"},
        figure_labels={1: "fig:fig01"},
        table_labels={1: "tab:converter_parameters"},
    )
    assert r"\documentclass[journal]{IEEEtran}" in tex
    assert r"\title{Sample Variable-Load Converter Paper}" in tex
    assert r"\begin{IEEEkeywords}" in tex
    assert r"\begin{abstract}" in tex
    assert r"\cite{darban2025carla,vaswani2017attention}" in tex
    assert r"\includegraphics[width=\textwidth]{fig01_system.pdf}" in tex
    assert r"\label{fig:fig01}" in tex
    assert r"\input{tables/table01_converter_parameters.tex}" in tex
    assert r"\begin{equation}" in tex
    assert r"G_{vd}(s) = V_{\mathrm{in}} \frac{1}{1 + sRC}" in tex
    assert r"\texttt{CARLA-Transformer}" in tex
    assert r"\texttt{dataset\_02}" in tex
    assert r"\texttt{mean \ensuremath{\pm} SD}" in tex
    assert r"\$\pm\$" not in tex
    assert r"\textbf{highlighted note}" in tex
    assert r"\textit{italic text}" in tex
    assert r"100\%" in tex
    assert r"---" in tex
    assert r"\section{Introduction}" in tex
    assert r"\subsection{Exact SSA Model on \texttt{dataset\_01}}" in tex
    assert r"All authors are with the UC3M, Legan\'es, Spain.}" in tex
    assert tex.count(r"UC3M, Legan\'es, Spain") == 2  # once in thanks, once in Acknowledgments
    assert r"\eqref{eq:gvd_exact}" in tex
    assert r"(Eq.~\eqref{eq:gvd_exact})" in tex
    assert r"eq:gvd\_exact" not in tex
    assert r"tab:converter\_parameters" not in tex
    assert r"\section*{Data and Code Availability}" in tex
    assert r"\section*{Acknowledgments}" in tex
    assert "Bullet ignored in main IEEE body cleanly." not in tex
    assert "[@" not in tex
    assert "**" not in tex


def test_validate_overleaf_project_catches_unresolved_assets_and_bad_braces(tmp_path: Path) -> None:
    (tmp_path / "references.bib").write_text("@article{known2024,\n title={Known}\n}\n", encoding="utf-8")
    (tmp_path / "main.tex").write_text(
        r"""\documentclass[journal]{IEEEtran}
\begin{document}
Text \cite{missing_key} and \ref{missing_ref} and [@markdown_cite] with \includegraphics{missing_fig.pdf} and \input{tables/missing.tex} and {unclosed
\begin{abstract}
x
\end{document}
""",
        encoding="utf-8",
    )
    errors = validate_overleaf_project(tmp_path)
    joined = " | ".join(errors)
    assert "missing_key" in joined
    assert "missing_ref" in joined
    assert "markdown_cite" in joined or "Markdown" in joined
    assert "missing_fig.pdf" in joined
    assert "missing.tex" in joined
    assert "abstract" in joined
    assert "braces" in joined.lower() or "brace" in joined.lower()


def test_export_overleaf_bundle_from_real_variable_load_manuscript(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_paper_dir = repo_root / "papers" / "variable_load_fra_ad"
    out_dir = tmp_path / "overleaf_repo"

    summary = export_overleaf_bundle(source_dir=src_paper_dir, output_dir=out_dir)
    assert summary["errors"] == []
    assert validate_overleaf_project(out_dir) == []

    main_tex = (out_dir / "main.tex").read_text(encoding="utf-8")
    assert r"\documentclass[journal]{IEEEtran}" in main_tex
    assert r"\bibliography{references}" in main_tex
    assert "0.9417" in main_tex
    assert "0.0952" in main_tex

    for fig_idx in range(1, 10):
        pdf_matches = list((out_dir / "figures").glob(f"fig0{fig_idx}_*.pdf"))
        assert len(pdf_matches) == 1
        assert pdf_matches[0].name in main_tex
    assert r"s(\mathbf{z}_i, \mathbf{z}_i^{+}) - \log \Big(" in main_tex

    for tbl_idx in range(1, 8):
        tex_matches = list((out_dir / "tables").glob(f"table0{tbl_idx}_*.tex"))
        assert len(tex_matches) == 1
        assert f"tables/{tex_matches[0].name}" in main_tex
        tbl_text = tex_matches[0].read_text(encoding="utf-8")
        assert rf"\caption{{Table {tbl_idx}." not in tbl_text
        assert r"\caption{" in tbl_text

    assert (out_dir / "references.bib").is_file()
    assert (out_dir / "manuscript.md").is_file()
    assert (out_dir / ".gitignore").is_file()
