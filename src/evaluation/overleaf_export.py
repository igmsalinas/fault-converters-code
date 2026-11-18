"""Utilities to export `papers/variable_load_fra_ad/` as an Overleaf-ready IEEEtran project."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

DEFAULT_TABLE_LABELS = {
    1: "tab:converter_parameters",
    2: "tab:variable_load_benchmark",
    3: "tab:10seed_ablation",
    4: "tab:encoder_backbones",
    5: "tab:representation_vs_scorer",
    6: "tab:fra_noise_sweep",
    7: "tab:lofo_generalization",
}

DEFAULT_FIGURE_LABELS = {
    1: "fig:fig01",
    2: "fig:fig02",
    3: "fig:fig03",
    4: "fig:fig04",
    5: "fig:fig05",
    6: "fig:fig06",
    7: "fig:fig07",
    8: "fig:fig08",
    9: "fig:fig09",
}

DEFAULT_EQUATION_LABELS = [
    "eq:resonance",
    "eq:gvd_exact",
    "eq:cartesian",
    "eq:gamma_ratio",
    "eq:carla_loss",
    "eq:knn_score",
    "eq:prec_pi",
]

UNICODE_PROSE_REPLACEMENTS = {
    "—": "---",
    "–": "--",
    "−": "-",
    "’": "'",
    "‘": "`",
    "“": "``",
    "”": "''",
    "á": r"\'a",
    "é": r"\'e",
    "í": r"\'i",
    "ó": r"\'o",
    "ú": r"\'u",
    "Á": r"\'A",
    "É": r"\'E",
    "Í": r"\'I",
    "Ó": r"\'O",
    "Ú": r"\'U",
    "ñ": r"\~n",
    "Ñ": r"\~N",
    "ü": r'\"u',
    "Ü": r'\"U',
    "µ": r"$\mu$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "≈": r"$\approx$",
    "≤": r"$\le$",
    "≥": r"$\ge$",
}


def _escape_code_text(text: str) -> str:
    """Escape characters inside \\texttt{...} safely for pdflatex."""
    escaped = text.replace("\\", r"\textbackslash{}")
    for ch, rep in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        escaped = escaped.replace(ch, rep)
    if len(text) > 20:
        escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
    escaped = escaped.replace("PhysicsAnomalyInjector", r"Physics\allowbreak{}Anomaly\allowbreak{}Injector")
    escaped = escaped.replace("BuckConverter.", r"Buck\allowbreak{}Converter.\allowbreak{}")
    code_unicode = {
        "—": "---",
        "–": "--",
        "−": "-",
        "’": "'",
        "‘": "`",
        "“": "``",
        "”": "''",
        "á": r"\'a",
        "é": r"\'e",
        "í": r"\'i",
        "ó": r"\'o",
        "ú": r"\'u",
        "ñ": r"\~n",
        "µ": r"\ensuremath{\mu}",
        "±": r"\ensuremath{\pm}",
        "×": r"\ensuremath{\times}",
        "≈": r"\ensuremath{\approx}",
        "≤": r"\ensuremath{\le}",
        "≥": r"\ensuremath{\ge}",
    }
    for src, dst in code_unicode.items():
        escaped = escaped.replace(src, dst)
    return escaped


def _escape_plain_prose(text: str) -> str:
    """Escape non-math, non-code prose and normalize punctuation to pure ASCII LaTeX."""
    out = text
    for src, dst in UNICODE_PROSE_REPLACEMENTS.items():
        out = out.replace(src, dst)
    out = out.replace(r"\*", "*")
    for ch, rep in (("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("_", r"\_")):
        out = out.replace(ch, rep)
    return out


def _convert_inline(
    text: str,
    *,
    figure_labels: dict[int, str] | None = None,
    table_labels: dict[int, str] | None = None,
    equation_labels: list[str] | None = None,
    section_labels: dict[str, str] | None = None,
) -> str:
    """Convert inline Markdown (math, code, citations, bold/italic, cross-refs) to LaTeX."""
    math_tokens: list[str] = []
    code_tokens: list[str] = []
    cite_tokens: list[str] = []
    ref_tokens: list[str] = []

    def _stash_math(match: re.Match[str]) -> str:
        idx = len(math_tokens)
        math_tokens.append(match.group(0))
        return f"@@MATH{idx}@@"

    def _stash_code(match: re.Match[str]) -> str:
        idx = len(code_tokens)
        code_tokens.append(rf"\texttt{{{_escape_code_text(match.group(1))}}}")
        return f"@@CODE{idx}@@"

    def _stash_cite(match: re.Match[str]) -> str:
        raw_keys = [part.strip().lstrip("@") for part in match.group(1).split(";")]
        keys = [key for key in raw_keys if key]
        idx = len(cite_tokens)
        cite_tokens.append(rf"\cite{{{','.join(keys)}}}")
        return f"@@CITE{idx}@@"

    def _stash_ref(token: str) -> str:
        idx = len(ref_tokens)
        ref_tokens.append(token)
        return f"@@REF{idx}@@"

    working = re.sub(r"(?<!\\)\$[^$\n]+?(?<!\\)\$", _stash_math, text)
    working = re.sub(r"`([^`\n]+?)`", _stash_code, working)
    working = re.sub(r"\[(@[^\]]+)\]", _stash_cite, working)

    # Cross-references outside math/code spans (stashed before _ escaping)
    if figure_labels:
        working = re.sub(
            r"\b(Figure|Fig\.)\s+(\d+)(\([a-z]\)|[a-z](?=\b))?",
            lambda m: _stash_ref(_repl_fig_with_parens(m, figure_labels)),
            working,
        )

    if table_labels:
        def _repl_tab(match: re.Match[str]) -> str:
            tbl_num = int(match.group(1))
            label = table_labels.get(tbl_num)
            return _stash_ref(rf"Table~\ref{{{label}}}") if label else match.group(0)

        working = re.sub(r"\bTable\s+(\d+)\b", _repl_tab, working)

    if equation_labels:
        def _repl_eq(match: re.Match[str]) -> str:
            word = match.group(1)
            eq_idx = int(match.group(2) or match.group(3)) - 1
            if not (0 <= eq_idx < len(equation_labels)):
                return match.group(0)
            label = equation_labels[eq_idx]
            prefix = "Equation~" if word == "Equation" else "Eq.~"
            return _stash_ref(rf"{prefix}\eqref{{{label}}}")

        working = re.sub(r"\b(Equation|Eq\.)\s+(?:\((\d+)\)|(\d+))", _repl_eq, working)

    if section_labels:
        def _repl_sec(match: re.Match[str]) -> str:
            word = match.group(1)
            sec_num = match.group(2)
            label = section_labels.get(sec_num)
            if not label:
                return match.group(0)
            prefix = "Sec.~" if word.startswith("Sec.") else "Section~"
            return _stash_ref(rf"{prefix}\ref{{{label}}}")

        working = re.sub(r"\b(Section|Sec\.)\s+(\d+(?:\.\d+)?)", _repl_sec, working)

    # Escape plain text first, then convert bold / italic spans
    working = _escape_plain_prose(working)
    working = re.sub(r"\*\*([^\n]+?)\*\*", lambda m: rf"\textbf{{{m.group(1)}}}", working)
    working = re.sub(
        r"(?<![\\A-Za-z0-9])\*([^\s*][^\n]*?[^\s*]|[^\s*])\*(?![A-Za-z0-9])",
        lambda m: rf"\textit{{{m.group(1)}}}",
        working,
    )

    for idx, val in enumerate(ref_tokens):
        working = working.replace(f"@@REF{idx}@@", val)
    for idx, val in enumerate(cite_tokens):
        working = working.replace(f"@@CITE{idx}@@", val)
    for idx, val in enumerate(code_tokens):
        working = working.replace(f"@@CODE{idx}@@", val)
    for idx, val in enumerate(math_tokens):
        working = working.replace(f"@@MATH{idx}@@", val)

    return working


def _repl_fig_with_parens(match: re.Match[str], figure_labels: dict[int, str]) -> str:
    lead = match.group(1)
    fig_num = int(match.group(2))
    panel = match.group(3) or ""
    label = figure_labels.get(fig_num)
    if not label:
        return match.group(0)
    prefix = "Figure~" if lead == "Figure" else "Fig.~"
    if panel and not panel.startswith("("):
        panel = f"({panel})"
    return rf"{prefix}\ref{{{label}}}{panel}"


def _parse_frontmatter(md_text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter fields used for IEEEtran title, authors, and keywords."""
    if not md_text.startswith("---"):
        return {}, md_text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", md_text, flags=re.DOTALL)
    if not match:
        return {}, md_text
    fm_block, body = match.group(1), match.group(2)

    title_match = re.search(r'^title:\s*"(.+?)"\s*$', fm_block, flags=re.MULTILINE)
    title = title_match.group(1) if title_match else ""

    authors: list[dict[str, Any]] = []
    current_author: dict[str, Any] | None = None
    in_authors = False
    in_keywords = False
    keywords: list[str] = []

    for line in fm_block.splitlines():
        if line.startswith("authors:"):
            in_authors = True
            in_keywords = False
            continue
        if line.startswith("keywords:"):
            in_keywords = True
            in_authors = False
            continue
        if re.match(r"^[a-zA-Z_]+:", line):
            in_authors = False
            in_keywords = False
            continue
        if in_authors:
            name_m = re.match(r'^\s*-\s+name:\s*"(.+?)"\s*$', line)
            if name_m:
                current_author = {"name": name_m.group(1), "affiliation": "", "corresponding": False}
                authors.append(current_author)
            elif current_author is not None:
                aff_m = re.match(r'^\s+affiliation:\s*"(.+?)"\s*$', line)
                corr_m = re.match(r"^\s+corresponding:\s*(true|false)\s*$", line, flags=re.IGNORECASE)
                if aff_m:
                    current_author["affiliation"] = aff_m.group(1)
                elif corr_m:
                    current_author["corresponding"] = corr_m.group(1).lower() == "true"
        elif in_keywords:
            kw_m = re.match(r'^\s*-\s+"(.+?)"\s*$', line)
            if kw_m:
                keywords.append(kw_m.group(1))

    return {"title": title, "authors": authors, "keywords": keywords}, body


def _format_equation(expr: str, eq_label: str) -> str:
    """Format a display math block cleanly for 2-column IEEEtran."""
    body = expr.strip()
    if body.endswith(","):
        # Trailing punctuation belongs inside the math environment
        pass
    if eq_label == "eq:resonance" and "f_0" in body and "f_z" in body:
        body = (
            r"\begin{split}" + "\n"
            r"f_0 &= \frac{1}{2\pi\sqrt{L_{\mathrm{out}}^{(0)} C_{\mathrm{out}}^{(0)}}} \approx 1.125\,\text{kHz}, \\" + "\n"
            r"f_z &= \frac{1}{2\pi C_{\mathrm{out}}^{(0)} \mathrm{ESR}_C^{(0)}} \approx 79.58\,\text{kHz}."
            "\n" + r"\end{split}"
        )
    elif eq_label == "eq:gvd_exact" and r"V_{\mathrm{eff}}" in body:
        body = (
            r"\begin{split}" + "\n"
            r"&G_{vd}(s; \boldsymbol{\theta}, R_{\mathrm{out}}) = V_{\mathrm{eff}} \frac{R_{\mathrm{out}}}{R_{\mathrm{out}} + R_s} \\" + "\n"
            r"&\times \frac{1 + s C_{\mathrm{out}} \mathrm{ESR}_C}{\splitfrac{s^2 L_{\mathrm{out}} C_{\mathrm{out}} \alpha_C + 1}{+ s\!\left[ C_{\mathrm{out}}\!\left( R_s \alpha_C + \frac{R_{\mathrm{out}} \mathrm{ESR}_C}{R_{\mathrm{out}} + R_s} \right) + \frac{L_{\mathrm{out}}}{R_{\mathrm{out}} + R_s} \right]}},"
            "\n" + r"\end{split}"
        )
    elif eq_label == "eq:carla_loss" and r"\mathcal{L}" in body:
        body = (
            r"\begin{split}" + "\n"
            r"&\mathcal{L}(\theta, \psi, \omega) = \frac{w_{\mathrm{rec}}}{B} \sum_{i=1}^{B} \left\| \mathbf{X}_i - g_{\psi}(f_{\theta}(\mathbf{X}_i)) \right\|_F^2 \\" + "\n"
            r"&\quad - \frac{w_c}{B} \sum_{i=1}^{B} \Big[ s(\mathbf{z}_i, \mathbf{z}_i^{+}) - \log \Big( \exp\!\big(s(\mathbf{z}_i, \mathbf{z}_i^{+})\big) \\" + "\n"
            r"&\qquad + \sum_{k \neq i} \exp\!\big(s(\mathbf{z}_i, \mathbf{z}_k)\big) + \sum_{j=1}^{B} \exp\!\big(s(\mathbf{z}_i, \mathbf{z}_j^{-})\big) \Big) \Big],"
            "\n" + r"\end{split}"
        )
    elif eq_label == "eq:knn_score" and r"\mathcal{N}_k" in body:
        body = (
            r"\begin{split}" + "\n"
            r"a(\mathbf{X}) &= \frac{1}{k} \sum_{\mathbf{z}_{(r)} \in \mathcal{N}_k(\pi_{\omega}(f_{\theta}(\mathbf{X}));\, \mathcal{Z}_{\mathrm{train}})} \\" + "\n"
            r"&\qquad \big\| \pi_{\omega}(f_{\theta}(\mathbf{X})) - \mathbf{z}_{(r)} \big\|_2."
            "\n" + r"\end{split}"
        )
    elif eq_label == "eq:cartesian" and r"\tilde{\mathbf{X}}_i" in body:
        body = (
            r"\begin{split}" + "\n"
            r"\tilde{\mathbf{X}}_i &= \big[ \operatorname{Re} H_i(j\omega_k),\; \operatorname{Im} H_i(j\omega_k) \big]_{k=1}^{N_f} \\" + "\n"
            r"&\quad \in \mathbb{R}^{N_f \times 2}, \quad H_i(j\omega_k) = 10^{|H_i(j\omega_k)|_{\mathrm{dB}}/20} e^{j \angle H_i(j\omega_k)}."
            "\n" + r"\end{split}"
        )
    elif eq_label == "eq:gamma_ratio" and r"\Gamma_{\boldsymbol{\alpha}}" in body:
        body = (
            r"\begin{split}" + "\n"
            r"\Gamma_{\boldsymbol{\alpha}}(j\omega_k) &= \frac{G_{vd}\!\left(j\omega_k,\, \boldsymbol{\alpha} \odot \boldsymbol{\theta}^{(0)},\, R_{\mathrm{out}}^{(0)}\right)}{G_{vd}\!\left(j\omega_k,\, \boldsymbol{\theta}^{(0)},\, R_{\mathrm{out}}^{(0)}\right)}, \\" + "\n"
            r"H_i^{-}(j\omega_k) &= H_i(j\omega_k) \cdot \Gamma_{\boldsymbol{\alpha}}(j\omega_k)."
            "\n" + r"\end{split}"
        )
    elif eq_label == "eq:prec_pi" and r"\operatorname{Prec}" in body:
        body = (
            r"\begin{split}" + "\n"
            r"\operatorname{Prec}_{\pi}(\tau) &= \frac{\pi \operatorname{TPR}(\tau)}{\pi \operatorname{TPR}(\tau) + (1 - \pi)\operatorname{FPR}(\tau)}, \\" + "\n"
            r"\text{PR-AUC}_{\pi=0.05} &= \int_0^1 \operatorname{Prec}_{0.05}(\operatorname{TPR})\, d\operatorname{TPR},"
            "\n" + r"\end{split}"
        )

    return "\n".join(
        [
            r"\begin{equation}",
            rf"\label{{{eq_label}}}",
            body,
            r"\end{equation}",
        ]
    )


def _format_authors(authors: list[dict[str, Any]]) -> str:
    """Format author block with ASCII-safe accents and UC3M thanks footnote."""
    if not authors:
        return r"\author{Anonymous}"
    formatted_names: list[str] = []
    affiliations: list[str] = []
    for entry in authors:
        name_ascii = _escape_plain_prose(entry["name"]).replace(" ", "~")
        if entry.get("corresponding"):
            name_ascii = f"{name_ascii}$^*$"
        formatted_names.append(name_ascii)
        aff = _escape_plain_prose(entry.get("affiliation", "").strip())
        if aff and aff not in affiliations:
            affiliations.append(aff)

    if len(formatted_names) > 1:
        names_joined = ", ".join(formatted_names[:-1]) + ", and~" + formatted_names[-1]
    else:
        names_joined = formatted_names[0]

    thanks_parts: list[str] = []
    if affiliations:
        thanks_parts.append(
            rf"\thanks{{All authors are with the {'; '.join(affiliations)}.}}"
        )
    if any(a.get("corresponding") for a in authors):
        thanks_parts.append(r"\thanks{$^*$Corresponding author.}")
    return "\\author{" + names_joined + "".join(thanks_parts) + "}"


def _scan_section_labels(body: str) -> dict[str, str]:
    """Collect explicit section/subsection numbers -> LaTeX labels."""
    labels: dict[str, str] = {}
    for line in body.splitlines():
        sec_m = re.match(r"^##\s+(\d+)\.\s+.+$", line)
        if sec_m:
            num = sec_m.group(1)
            labels[num] = f"sec:{num}"
        sub_m = re.match(r"^###\s+(\d+\.\d+)\s+.+$", line)
        if sub_m:
            num = sub_m.group(1)
            labels[num] = "sec:" + num.replace(".", "_")
    return labels


def markdown_to_ieeetran_tex(
    md_text: str,
    *,
    table_files: dict[int, str],
    figure_labels: dict[int, str] | None = None,
    table_labels: dict[int, str] | None = None,
    equation_labels: list[str] | None = None,
) -> str:
    """Convert the variable-load Markdown manuscript into a compilable IEEEtran main.tex."""
    fig_labels = figure_labels if figure_labels is not None else DEFAULT_FIGURE_LABELS
    tab_labels = table_labels if table_labels is not None else DEFAULT_TABLE_LABELS
    eq_labels = equation_labels if equation_labels is not None else DEFAULT_EQUATION_LABELS

    meta, raw_body = _parse_frontmatter(md_text)
    section_labels = _scan_section_labels(raw_body)
    lines = raw_body.splitlines()

    abstract_lines: list[str] = []
    body_chunks: list[str] = []
    in_preamble_header = True
    in_abstract = False
    in_highlights = False
    eq_counter = 0

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Skip repeated Markdown title/author banner before ## Abstract
        if in_preamble_header:
            if stripped.startswith("## Abstract"):
                in_preamble_header = False
                in_abstract = True
                i += 1
                continue
            i += 1
            continue

        # Abstract block until horizontal rule or next ## heading
        if in_abstract:
            if stripped == "---" or stripped.startswith("## "):
                in_abstract = False
                if stripped == "---":
                    i += 1
                continue
            if stripped:
                abstract_lines.append(
                    _convert_inline(
                        stripped,
                        figure_labels=fig_labels,
                        table_labels=tab_labels,
                        equation_labels=eq_labels,
                        section_labels=section_labels,
                    )
                )
            i += 1
            continue

        # Skip Research Highlights section cleanly (IEEEtran doesn't carry Elsevier highlight box)
        if stripped.startswith("## Research Highlights"):
            in_highlights = True
            i += 1
            continue
        if in_highlights:
            if stripped.startswith("## "):
                in_highlights = False
                continue
            i += 1
            continue

        if not stripped or stripped == "---":
            i += 1
            continue

        # Display equations $$ ... $$
        if stripped == "$$":
            i += 1
            eq_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "$$":
                eq_lines.append(lines[i])
                i += 1
            i += 1  # consume closing $$
            eq_label = (
                eq_labels[eq_counter]
                if eq_counter < len(eq_labels)
                else f"eq:eq{eq_counter + 1}"
            )
            eq_counter += 1
            body_chunks.append(_format_equation("\n".join(eq_lines), eq_label))
            continue

        # Markdown figure + italic caption pair
        fig_m = re.match(r"^!\[(.*?)\]\((?:figures/)?([^)/]+\.(?:png|pdf))\)\s*$", stripped)
        if fig_m:
            img_stem = Path(fig_m.group(2)).stem
            fig_num_m = re.search(r"Figure\s+(\d+)", fig_m.group(1))
            fig_num = int(fig_num_m.group(1)) if fig_num_m else None
            caption_raw = fig_m.group(1)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("*Figure "):
                caption_raw = lines[i + 1].strip().strip("*")
                caption_raw = re.sub(r"^Figure\s+\d+\.\s*", "", caption_raw)
                i += 1
            caption_tex = _convert_inline(
                caption_raw,
                figure_labels=fig_labels,
                table_labels=tab_labels,
                equation_labels=eq_labels,
                section_labels=section_labels,
            )
            lbl = (
                fig_labels.get(fig_num, f"fig:fig{fig_num:02d}")
                if fig_num is not None
                else f"fig:{img_stem}"
            )
            body_chunks.append(
                "\n".join(
                    [
                        r"\begin{figure*}[!t]",
                        r"\centering",
                        rf"\includegraphics[width=\textwidth]{{{img_stem}.pdf}}",
                        rf"\caption{{{caption_tex}}}",
                        rf"\label{{{lbl}}}",
                        r"\end{figure*}",
                    ]
                )
            )
            i += 1
            continue

        # Standalone Markdown Table -> \input{tables/...}
        tbl_m = re.match(r"^\*\*Table\s+(\d+)\.\s+.+\*\*$", stripped)
        if tbl_m:
            tbl_num = int(tbl_m.group(1))
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("|")):
                i += 1
            # Consume optional italic table footnote duplicated inside tables/*.tex
            if tbl_num == 6 and i < len(lines) and lines[i].strip().startswith("*On the primary matched"):
                i += 1
            tbl_rel = table_files.get(tbl_num)
            if tbl_rel:
                body_chunks.append(rf"\input{{{tbl_rel}}}")
            continue

        # Headings
        h2_numbered = re.match(r"^##\s+(\d+)\.\s+(.+)$", stripped)
        if h2_numbered:
            sec_num, title = h2_numbered.group(1), h2_numbered.group(2)
            body_chunks.append(
                rf"\section{{{_convert_inline(title)}}}"
                + rf"\label{{{section_labels[sec_num]}}}"
            )
            i += 1
            continue

        h2_unnumbered = re.match(r"^##\s+(.+)$", stripped)
        if h2_unnumbered:
            title = h2_unnumbered.group(1)
            body_chunks.append(rf"\section*{{{_convert_inline(title)}}}")
            i += 1
            continue

        h3_numbered = re.match(r"^###\s+(\d+\.\d+)\s+(.+)$", stripped)
        if h3_numbered:
            sub_num, title = h3_numbered.group(1), h3_numbered.group(2)
            body_chunks.append(
                rf"\subsection{{{_convert_inline(title)}}}"
                + rf"\label{{{section_labels[sub_num]}}}"
            )
            i += 1
            continue

        # Bullet lists
        if stripped.startswith("- "):
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                item_text = lines[i].strip()[2:].strip()
                items.append(
                    r"\item "
                    + _convert_inline(
                        item_text,
                        figure_labels=fig_labels,
                        table_labels=tab_labels,
                        equation_labels=eq_labels,
                        section_labels=section_labels,
                    )
                )
                i += 1
            body_chunks.append(
                "\n".join([r"\begin{itemize}", *items, r"\end{itemize}"])
            )
            continue

        # Numbered lists
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item_text = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(
                    r"\item "
                    + _convert_inline(
                        item_text,
                        figure_labels=fig_labels,
                        table_labels=tab_labels,
                        equation_labels=eq_labels,
                        section_labels=section_labels,
                    )
                )
                i += 1
            body_chunks.append(
                "\n".join([r"\begin{enumerate}", *items, r"\end{enumerate}"])
            )
            continue

        # Regular paragraph (join multi-line prose paragraph until blank or structural marker)
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt in ("---", "$$")
                or nxt.startswith(("## ", "### ", "![", "**Table ", "- ", "|"))
                or re.match(r"^\d+\.\s+", nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        para_tex = _convert_inline(
            " ".join(para_lines),
            figure_labels=fig_labels,
            table_labels=tab_labels,
            equation_labels=eq_labels,
            section_labels=section_labels,
        )
        body_chunks.append(para_tex)

    title_tex = _escape_plain_prose(meta.get("title") or "Untitled Manuscript")
    author_tex = _format_authors(meta.get("authors") or [])
    kw_list = [_escape_plain_prose(k) for k in (meta.get("keywords") or [])]
    keywords_tex = ", ".join(kw_list)

    preamble = [
        r"\documentclass[journal]{IEEEtran}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{mathtools}",
        r"\usepackage{microtype}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{array}",
        r"\usepackage{cite}",
        r"\usepackage{url}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\graphicspath{{figures/}}",
        r"\emergencystretch=1.5em",
        "",
        rf"\title{{{title_tex}}}",
        author_tex,
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        "\n\n".join(abstract_lines),
        r"\end{abstract}",
        "",
        r"\begin{IEEEkeywords}",
        keywords_tex,
        r"\end{IEEEkeywords}",
        "",
    ]

    closing = [
        "",
        r"\bibliographystyle{IEEEtran}",
        r"\bibliography{references}",
        r"\end{document}",
        "",
    ]

    return "\n\n".join(["\n".join(preamble), *body_chunks, "\n".join(closing)])


def validate_overleaf_project(project_dir: Path) -> list[str]:
    """Validate an Overleaf-ready LaTeX folder for unresolved citations, assets, and syntax."""
    errors: list[str] = []
    main_tex_path = project_dir / "main.tex"
    bib_path = project_dir / "references.bib"
    if not main_tex_path.is_file():
        errors.append("Missing main.tex")
        return errors
    if not bib_path.is_file():
        errors.append("Missing references.bib")
        return errors

    tex = main_tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")

    # 1. Citations vs BibTeX
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib, flags=re.IGNORECASE))
    cite_keys: set[str] = set()
    for grp in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
        cite_keys.update(k.strip() for k in grp.split(",") if k.strip())
    missing_cites = sorted(cite_keys - bib_keys)
    if missing_cites:
        errors.append(f"Unresolved citation keys: {missing_cites}")

    # 2. Figures exist
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (project_dir / fig).is_file() and not (project_dir / "figures" / fig).is_file():
            errors.append(f"Missing referenced figure: {fig}")

    # 3. Tables / \input exist
    for inc in re.findall(r"\\input\{([^}]+)\}", tex):
        inc_path = project_dir / inc
        if not inc_path.is_file() and not inc_path.with_suffix(".tex").is_file():
            errors.append(f"Missing \\input file: {inc}")

    # 4. Unconverted Markdown remnants
    if re.search(r"\[@[A-Za-z0-9_-]+", tex):
        errors.append("Detected unconverted Markdown [@citation] syntax in main.tex")
    if "**" in tex:
        errors.append("Detected unconverted Markdown '**' bold markup in main.tex")
    if re.search(r"^(?:#{1,4}\s|!\[Figure)", tex, flags=re.MULTILINE):
        errors.append("Detected unconverted Markdown heading or image link in main.tex")

    # 5. Balanced LaTeX environments & cross-reference labels across main.tex + tables/*.tex
    tex_files_to_check = [main_tex_path]
    tables_dir = project_dir / "tables"
    if tables_dir.is_dir():
        tex_files_to_check.extend(sorted(tables_dir.glob("*.tex")))

    defined_labels: set[str] = set()
    referenced_labels: set[str] = set()
    for tex_file in tex_files_to_check:
        content = tex_file.read_text(encoding="utf-8")
        defined_labels.update(re.findall(r"\\label\{([^}]+)\}", content))
        referenced_labels.update(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", content))
        begins: dict[str, int] = {}
        ends: dict[str, int] = {}
        for env in re.findall(r"\\begin\{([^}]+)\}", content):
            begins[env] = begins.get(env, 0) + 1
        for env in re.findall(r"\\end\{([^}]+)\}", content):
            ends[env] = ends.get(env, 0) + 1
        for env in sorted(set(begins) | set(ends)):
            if begins.get(env, 0) != ends.get(env, 0):
                errors.append(
                    f"{tex_file.name}: unbalanced environment '{env}' "
                    f"({begins.get(env, 0)} vs {ends.get(env, 0)})"
                )
        # Check non-escaped curly brace balance
        depth = 0
        escaped = False
        for ch in content:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"{tex_file.name}: extra closing brace '}}'")
                    break
        if depth > 0:
            errors.append(f"{tex_file.name}: {depth} unclosed brace(s) '{{'")

    unresolved_refs = sorted(referenced_labels - defined_labels)
    if unresolved_refs:
        errors.append(f"Unresolved \\ref/\\eqref labels: {unresolved_refs}")

    return errors


def _strip_redundant_table_prefix(tex_content: str) -> str:
    """Clean table captions for IEEEtran (strip 'Table N. ', prevent small-caps ttfamily)."""
    cleaned = re.sub(r"(\\caption\{)\s*Table\s+\d+\.\s*", r"\1", tex_content)
    cleaned = cleaned.replace("(Sec. 3.3)", "(Section~III-C)")

    def _fix_caption_tt(match: re.Match[str]) -> str:
        inner = re.sub(r"\\texttt\{([^}]*)\}", r"{\\upshape\\ttfamily \1}", match.group(1))
        return rf"\caption{{{inner}}}"

    cleaned = re.sub(r"\\caption\{([^\n]+)\}", _fix_caption_tt, cleaned)
    return cleaned


def _strip_unicode_for_pdflatex(tex_content: str) -> str:
    """Replace any stray Unicode punctuation in table .tex exports with pure ASCII LaTeX."""
    out = tex_content
    for src, dst in UNICODE_PROSE_REPLACEMENTS.items():
        out = out.replace(src, dst)
    return out


def export_overleaf_bundle(
    source_dir: Path,
    output_dir: Path,
    *,
    clean_output: bool = True,
) -> dict[str, Any]:
    """Build a self-contained Overleaf IEEEtran bundle from `papers/variable_load_fra_ad/`."""
    if clean_output and output_dir.exists():
        for child in output_dir.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    # Map table index -> Overleaf relative .tex path
    table_files: dict[int, str] = {}
    src_tables = source_dir / "tables"
    for item in sorted(src_tables.iterdir()):
        if not item.is_file():
            continue
        dst = output_dir / "tables" / item.name
        if item.suffix == ".tex":
            cleaned = _strip_unicode_for_pdflatex(
                _strip_redundant_table_prefix(item.read_text(encoding="utf-8"))
            )
            dst.write_text(cleaned, encoding="utf-8")
            m = re.match(r"table0?(\d+)_", item.name)
            if m:
                table_files[int(m.group(1))] = f"tables/{item.name}"
        else:
            shutil.copy2(item, dst)

    # Copy all publication figures (.pdf vector + .png raster)
    src_figures = source_dir / "figures"
    copied_figures: list[str] = []
    for item in sorted(src_figures.iterdir()):
        if item.is_file():
            shutil.copy2(item, output_dir / "figures" / item.name)
            copied_figures.append(item.name)

    # Copy bibliography & markdown source
    shutil.copy2(source_dir / "references.bib", output_dir / "references.bib")
    shutil.copy2(source_dir / "manuscript.md", output_dir / "manuscript.md")

    # Convert manuscript.md -> IEEEtran main.tex
    md_text = (source_dir / "manuscript.md").read_text(encoding="utf-8")
    main_tex = markdown_to_ieeetran_tex(
        md_text,
        table_files=table_files,
        figure_labels=DEFAULT_FIGURE_LABELS,
        table_labels=DEFAULT_TABLE_LABELS,
        equation_labels=DEFAULT_EQUATION_LABELS,
    )
    (output_dir / "main.tex").write_text(main_tex, encoding="utf-8")

    # Add Overleaf/LaTeX .gitignore and README.md
    (output_dir / ".gitignore").write_text(
        "\n".join(
            [
                "*.aux",
                "*.bbl",
                "*.bcf",
                "*.blg",
                "*.fdb_latexmk",
                "*.fls",
                "*.log",
                "*.out",
                "*.run.xml",
                "*.synctex.gz",
                "*.toc",
                "main.pdf",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Disentangling Component Degradation from Operating-Load Variations in Power-Converter Frequency Responses",
                "",
                "Overleaf-ready IEEE `IEEEtran` LaTeX repository synchronized from `fault_converters/papers/variable_load_fra_ad/`.",
                "",
                "## Contents",
                "- `main.tex`: Main IEEE `IEEEtran` manuscript referencing vector `.pdf` figures and standalone `.tex` tables.",
                "- `references.bib`: BibTeX bibliography.",
                "- `figures/`: 8 publication figures (`fig01`--`fig08`, vector `.pdf` + 300-DPI `.png`).",
                "- `tables/`: 7 standalone tables (`table01`--`table07`, `.tex`, `.csv`, `.md`).",
                "- `manuscript.md`: Format-agnostic source manuscript.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    errors = validate_overleaf_project(output_dir)
    return {
        "output_dir": str(output_dir),
        "n_figures": len(copied_figures),
        "n_tables": len(table_files),
        "errors": errors,
    }
