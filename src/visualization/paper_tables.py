"""
Publication Table Builder & Multi-Format Exporter
=================================================

Generates synchronized CSV, GitHub Markdown, and Elsevier/IEEE `booktabs` LaTeX
tables directly from persisted experimental JSON/CSV artifacts to eliminate
manual transcription drift across manuscripts.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    """Canonical format-agnostic representation of a publication table."""

    table_id: str
    caption: str
    label: str
    columns: Sequence[str]
    alignments: Sequence[str]
    rows: Sequence[Sequence[str]]
    notes: Optional[str] = None
    wide: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def _strip_markdown_emphasis(cell: str) -> str:
    """Remove Markdown **bold**, *italic*, and `code` wrappers for plain CSV export."""
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", str(cell))
    out = re.sub(r"\*(.+?)\*", r"\1", out)
    out = re.sub(r"`(.+?)`", r"\1", out)
    return out.strip()


def _escape_outside_math(segment: str) -> str:
    """Escape LaTeX special characters in non-math text segments."""
    s = segment.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("±", r"$\pm$")
    s = s.replace("×", r"$\times$")
    s = s.replace("—", "---")
    s = s.replace("–", "--")
    s = s.replace("°", r"$^\circ$")
    return s


def _markdown_cell_to_latex(cell: str) -> str:
    """Convert a Markdown-formatted table cell (with `$...$`, `**...**`, `` `...` ``) to LaTeX."""
    text = str(cell).strip()
    is_bold = False
    is_italic = False
    if text.startswith("**") and text.endswith("**") and len(text) > 4:
        is_bold = True
        text = text[2:-2].strip()
    elif text.startswith("*") and text.endswith("*") and len(text) > 2:
        is_italic = True
        text = text[1:-1].strip()

    # Tokenize inline math `$...$` and inline code `` `...` `` so math stays untouched
    pattern = re.compile(r"(\$[^$]+\$|`[^`]+`|\*\*[^*]+\*\*)")
    parts: List[str] = []
    last_idx = 0
    for match in pattern.finditer(text):
        if match.start() > last_idx:
            parts.append(_escape_outside_math(text[last_idx : match.start()]))
        tok = match.group(0)
        if tok.startswith("$") and tok.endswith("$"):
            parts.append(tok)
        elif tok.startswith("`") and tok.endswith("`"):
            inner = _escape_outside_math(tok[1:-1])
            parts.append(rf"\texttt{{{inner}}}")
        elif tok.startswith("**") and tok.endswith("**"):
            inner = _markdown_cell_to_latex(tok[2:-2])
            parts.append(rf"\textbf{{{inner}}}")
        last_idx = match.end()
    if last_idx < len(text):
        parts.append(_escape_outside_math(text[last_idx:]))

    rendered = "".join(parts)
    if is_bold:
        return rf"\textbf{{{rendered}}}"
    if is_italic:
        return rf"\textit{{{rendered}}}"
    return rendered


class PublicationTableExporter:
    """Exports `TableSpec` instances to synchronized `.csv`, `.md`, and `.tex` files."""

    def to_csv(self, spec: TableSpec, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([_strip_markdown_emphasis(c) for c in spec.columns])
            for row in spec.rows:
                writer.writerow([_strip_markdown_emphasis(c) for c in row])
        return path

    def to_markdown(self, spec: TableSpec, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        align_map = {"l": ":---", "c": ":---:", "r": "---:"}
        md_align = [align_map.get(a, ":---") for a in spec.alignments]

        lines = [
            f"**{spec.caption}**",
            "",
            "| " + " | ".join(spec.columns) + " |",
            "| " + " | ".join(md_align) + " |",
        ]
        for row in spec.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        if spec.notes:
            lines.extend(["", f"*{spec.notes}*"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def to_latex_booktabs(self, spec: TableSpec, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        env = "table*" if spec.wide else "table"
        col_spec = "".join(spec.alignments)
        header_tex = " & ".join(_markdown_cell_to_latex(c) for c in spec.columns) + r" \\"
        caption_tex = _markdown_cell_to_latex(spec.caption)

        lines = [
            rf"\begin{{{env}}}[t]",
            r"\centering",
            rf"\caption{{{caption_tex}}}",
            rf"\label{{{spec.label}}}",
            r"\resizebox{\textwidth}{!}{%" if spec.wide else r"\small",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header_tex,
            r"\midrule",
        ]
        for row in spec.rows:
            row_tex = " & ".join(_markdown_cell_to_latex(c) for c in row) + r" \\"
            lines.append(row_tex)
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}" + ("%" if spec.wide else ""),
            ]
        )
        if spec.wide:
            lines.append("}")
        if spec.notes:
            notes_tex = _markdown_cell_to_latex(spec.notes)
            lines.append(rf"\vspace{{2pt}}\par\footnotesize\textit{{Note:}} {notes_tex}")
        lines.append(rf"\end{{{env}}}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def export_all_formats(self, spec: TableSpec, out_dir: Path) -> Dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "csv": self.to_csv(spec, out_dir / f"{spec.table_id}.csv"),
            "md": self.to_markdown(spec, out_dir / f"{spec.table_id}.md"),
            "tex": self.to_latex_booktabs(spec, out_dir / f"{spec.table_id}.tex"),
        }


def _fmt_ms(mean_v: float, std_v: float, digits: int = 4, bold: bool = False) -> str:
    txt = f"{mean_v:.{digits}f} ± {std_v:.{digits}f}"
    return f"**{txt}**" if bold else txt


def _fmt_pct_ms(mean_v: float, std_v: float, digits: int = 1, bold: bool = False) -> str:
    txt = f"{100.0 * mean_v:.{digits}f}% ± {100.0 * std_v:.{digits}f}%"
    return f"**{txt}**" if bold else txt


def build_variable_load_paper_tables(
    ds02_dir: Path,
    ds01_dir: Optional[Path] = None,
) -> List[TableSpec]:
    """Construct all 7 canonical tables for the Variable-Load FRA paper from artifacts."""
    ablation_10 = json.loads((ds02_dir / "ablation_transformer_10seed_zero_label.json").read_text())
    ablation_full = json.loads((ds02_dir / "ablation_transformer_10seed_summary.json").read_text())
    enc_5 = json.loads((ds02_dir / "encoder_comparison_ssa_fixed_summary.json").read_text())
    latent_ctrl = json.loads((ds02_dir / "latent_controls_summary.json").read_text())
    fra_sweep = json.loads((ds02_dir / "fra_noise_sweep_dataset02_summary.json").read_text())
    lofo_5 = json.loads((ds02_dir / "lofo_transformer_ssa_fixed_5seed_summary.json").read_text())

    zero_base_df = pd.read_csv(ds02_dir / "zero_label_baselines.csv")
    pca_sweep_df = pd.read_csv(ds02_dir / "raw_pca_dimension_sweep.csv")
    sup_base_df = pd.read_csv(ds02_dir / "baselines_comparison.csv")

    # --- Table 1: Converter Parameters & Operating Regimes ---
    t1 = TableSpec(
        table_id="table01_converter_parameters",
        caption=(
            "Table 1. Nominal electrical parameters, healthy manufacturing tolerance bands, "
            "degraded fault intervals, and variable operating-load range of the switched "
            "synchronous Buck converter benchmark (`dataset_02`)."
        ),
        label="tab:converter_parameters",
        columns=[
            "Parameter",
            "Symbol",
            "Nominal Value",
            "Healthy Tolerance Band",
            "Degraded Fault Band(s)",
            "Physical Role / Aging Mechanism",
        ],
        alignments=["l", "c", "c", "c", "c", "l"],
        rows=[
            ["Input voltage", "$V_{\\mathrm{in}}$", "$50.0\\,\\mathrm{V}$", "Fixed", "—", "Regulated DC bus"],
            [
                "Duty cycle / Switching freq.",
                "$D,\\, f_s$",
                "$0.40,\\, 100\\,\\mathrm{kHz}$",
                "Fixed ($t_{\\mathrm{dead}} = 100\\,\\mathrm{ns}$)",
                "—",
                "Complementary PWM excitation",
            ],
            [
                "Output capacitance",
                "$C_{\\mathrm{out}}$",
                "$100\\,\\mu\\mathrm{F}$",
                "$\\pm 7.5\\%$ ($[92.5, 107.5]\\,\\mu\\mathrm{F}$)",
                "$[-45\\%, -15\\%] \\cup [+15\\%, +45\\%]$",
                "Dielectric loss / film capacitance drift",
            ],
            [
                "Capacitor ESR",
                "$\\mathrm{ESR}_C$",
                "$20\\,\\mathrm{m}\\Omega$",
                "$\\pm 20\\%$ ($[16.0, 24.0]\\,\\mathrm{m}\\Omega$)",
                "$[+50\\%, +200\\%]$ ($[30, 60]\\,\\mathrm{m}\\Omega$)",
                "Electrolyte dry-out / contact aging",
            ],
            [
                "Filter inductance",
                "$L_{\\mathrm{out}}$",
                "$200\\,\\mu\\mathrm{H}$",
                "$\\pm 7.5\\%$ ($[185.0, 215.0]\\,\\mu\\mathrm{H}$)",
                "$[-45\\%, -15\\%] \\cup [+15\\%, +45\\%]$",
                "Core permeability drift / partial saturation",
            ],
            [
                "Inductor ESR",
                "$\\mathrm{ESR}_L$",
                "$100\\,\\mathrm{m}\\Omega$",
                "$\\pm 27.5\\%$ ($[72.5, 127.5]\\,\\mathrm{m}\\Omega$)",
                "$[+50\\%, +200\\%]$ ($[150, 300]\\,\\mathrm{m}\\Omega$)",
                "Winding/solder contact degradation",
            ],
            [
                "High-side MOSFET $R_{\\mathrm{ds(on)}}$",
                "$R_{\\mathrm{ds},1}$",
                "$10\\,\\mathrm{m}\\Omega$",
                "$\\pm 27.5\\%$ ($[7.25, 12.75]\\,\\mathrm{m}\\Omega$)",
                "$[+80\\%, +560\\%]$ ($[18, 66]\\,\\mathrm{m}\\Omega$)",
                "Bond-wire lift / thermal die fatigue",
            ],
            [
                "Sync-rectifier MOSFET $R_{\\mathrm{ds(on)}}$",
                "$R_{\\mathrm{ds},2}$",
                "$10\\,\\mathrm{m}\\Omega$",
                "Clamped in PSIM",
                "Unobservable via $V_{\\mathrm{out}}$ FRA",
                "Clamped by antiparallel body diode (Sec. 3.3)",
            ],
            [
                "**Operating load resistance**",
                "$R_{\\mathrm{out}}$",
                "$4.0\\,\\Omega$",
                "**$[-50\\%, +100\\%]$ ($[2.0, 8.0]\\,\\Omega$)**",
                "**$[-50\\%, +100\\%]$ ($[2.0, 8.0]\\,\\Omega$)**",
                "**Legitimate $4\\times$ load variation ($2.5\\text{--}10\\,\\mathrm{A}$)**",
            ],
        ],
    )

    # --- Table 2: Main Variable-Load Zero-Label Benchmark ---
    def _base_row(det_name: str, label_disp: str, uses_lbl: str) -> List[str]:
        r975 = zero_base_df[
            (zero_base_df["detector"] == det_name) & (zero_base_df["healthy_quantile"] == "p97.5")
        ].iloc[0]
        r95 = zero_base_df[
            (zero_base_df["detector"] == det_name) & (zero_base_df["healthy_quantile"] == "p95")
        ].iloc[0]
        return [
            label_disp,
            uses_lbl,
            f"{float(r975['f1']):.4f}",
            f"{float(r95['f1']):.4f}",
            f"{float(r975['auc_pr_pi05']):.4f}",
            f"{float(r975['auc_roc']):.4f}",
            f"{100.0 * float(r975['recall_cout_single']):.1f}%",
            f"{100.0 * float(r975['recall_esrc_single']):.1f}%",
            f"{100.0 * float(r975['recall_lout_single']):.1f}%",
            f"{100.0 * float(r975['recall_esrl_single']):.1f}%",
            f"{100.0 * float(r975['recall_rds1_single']):.1f}%",
            f"{100.0 * float(r975['recall_multi_fault']):.1f}%",
        ]

    pca4 = pca_sweep_df[pca_sweep_df["n_components"] == 4].iloc[0]
    pca4_row = [
        "`PCA (99% Var., d=4)`",
        "No",
        f"{float(pca4['zero_p975_f1']):.4f}",
        f"{float(pca4['zero_p95_f1']):.4f}",
        f"{float(pca4['auc_pr_pi05']):.4f}",
        f"{float(pca4['auc_roc']):.4f}",
        f"{100.0 * float(pca4['recall_cout_single']):.1f}%",
        f"{100.0 * float(pca4['recall_esrc_single']):.1f}%",
        f"{100.0 * float(pca4['recall_lout_single']):.1f}%",
        f"{100.0 * float(pca4['recall_esrl_single']):.1f}%",
        f"{100.0 * float(pca4['recall_rds1_single']):.1f}%",
        f"{100.0 * float(pca4['recall_multi_fault']):.1f}%",
    ]

    ssa = ablation_10["regimes"]["tf_physics_ssa_fixed_load"]
    carla_row = [
        "**`CARLA-Transformer (SSA)`**",
        "**No**",
        _fmt_ms(ssa["zero_p975_f1"]["mean"], ssa["zero_p975_f1"]["sample_std"], bold=True),
        _fmt_ms(ssa["zero_p95_f1"]["mean"], ssa["zero_p95_f1"]["sample_std"], bold=True),
        _fmt_ms(ssa["auc_pr_pi05"]["mean"], ssa["auc_pr_pi05"]["sample_std"], bold=True),
        _fmt_ms(ssa["auc_roc"]["mean"], ssa["auc_roc"]["sample_std"], bold=True),
        f"**{100.0 * ssa['recall_cout_single']['mean']:.1f}%**",
        f"**{100.0 * ssa['recall_esrc_single']['mean']:.1f}%**",
        f"**{100.0 * ssa['recall_lout_single']['mean']:.1f}%**",
        f"**{100.0 * ssa['recall_esrl_single']['mean']:.1f}%**",
        f"**{100.0 * ssa['recall_rds1_single']['mean']:.1f}%**",
        f"**{100.0 * ssa['recall_multi_fault']['mean']:.1f}%**",
    ]

    gbm = sup_base_df[sup_base_df["detector"] == "Supervised GBM Oracle"].iloc[0]
    gbm_row = [
        "*`Supervised GBM Oracle`*",
        "*Yes (90 faults)*",
        f"*{float(gbm['f1']):.4f}*",
        "*—*",
        f"*{float(gbm['auc_pr_pi05']):.4f}*",
        f"*{float(gbm['auc_roc']):.4f}*",
        f"*{100.0 * float(gbm['recall_cout_single']):.1f}%*",
        f"*{100.0 * float(gbm['recall_esrc_single']):.1f}%*",
        f"*{100.0 * float(gbm['recall_lout_single']):.1f}%*",
        f"*{100.0 * float(gbm['recall_esrl_single']):.1f}%*",
        f"*{100.0 * float(gbm['recall_rds1_single']):.1f}%*",
        f"*{100.0 * float(gbm['recall_multi_fault']):.1f}%*",
    ]

    t2 = TableSpec(
        table_id="table02_variable_load_benchmark",
        caption=(
            "Table 2. Zero-anomaly-label (`p97.5`) anomaly detection performance on the "
            "variable-load benchmark (`dataset_02`, $R_{\\mathrm{out}} \\in [2, 8]\\,\\Omega$). "
            "Deep models report mean ± SD over 10 random seeds; `Supervised GBM Oracle` uses "
            "90 labeled validation anomalies solely as a supervised reference."
        ),
        label="tab:variable_load_benchmark",
        columns=[
            "Detector / Paradigm",
            "Uses Fault Labels?",
            "$\\mathrm{F1}_{\\mathrm{p97.5}}$",
            "$\\mathrm{F1}_{\\mathrm{p95}}$",
            "$\\mathrm{PR\\text{-}AUC}_{\\pi=0.05}$",
            "$\\mathrm{ROC\\text{-}AUC}$",
            "$C_{\\mathrm{out}}$ Rec.",
            "$\\mathrm{ESR}_C$ Rec.",
            "$L_{\\mathrm{out}}$ Rec.",
            "$\\mathrm{ESR}_L$ Rec.",
            "$R_{\\mathrm{ds},1}$ Rec.",
            "Multi Rec.",
        ],
        alignments=["l", "c", "c", "c", "c", "c", "r", "r", "r", "r", "r", "r"],
        rows=[
            _base_row("One-Class SVM (RBF)", "`One-Class SVM (RBF)`", "No"),
            _base_row("Raw k-NN (k=5)", "`Raw k-NN (k=5)`", "No"),
            _base_row("Mahalanobis (Ridge 1e-6)", "`Mahalanobis (Ridge 1e-6)`", "No"),
            _base_row("Isolation Forest (200)", "`Isolation Forest (200)`", "No"),
            _base_row("Mahalanobis (Ledoit-Wolf)", "`Mahalanobis (Ledoit-Wolf)`", "No"),
            pca4_row,
            _base_row("PCA Reconstruction (d=10)", "`PCA (Oracle d=10, Clean)`", "Oracle $d$"),
            carla_row,
            gbm_row,
        ],
    )

    # --- Table 3: 10-Seed Paired Contrastive Ablation ---
    ab_rows: List[List[str]] = []
    ab_specs = [
        ("tf_recon_only_mse", "`AE-Transformer (MSE, w_c=0)`", False),
        ("tf_recon_only_latent", "`AE-Transformer (Latent kNN5)`", False),
        ("tf_heuristic", "`CARLA-Transformer (Heuristic)`", False),
        ("tf_physics_ssa_fixed_load", "**`CARLA-Transformer (SSA)`**", True),
    ]
    for rkey, disp, is_best in ab_specs:
        rg = ablation_10["regimes"][rkey]
        if is_best:
            p_str = "—"
        else:
            test_obj = ablation_10["paired_tests"][
                f"tf_physics_ssa_fixed_load_vs_{rkey}"
            ]["zero_p975_f1"]
            delta = float(test_obj["mean_diff"])
            pval = float(test_obj["p_value_two_sided"])
            p_str = f"$+{delta:.4f}$ ($p = {pval:.5f}$)"
        ab_rows.append(
            [
                disp,
                _fmt_ms(rg["zero_p975_f1"]["mean"], rg["zero_p975_f1"]["sample_std"], bold=is_best),
                _fmt_ms(rg["auc_pr_pi05"]["mean"], rg["auc_pr_pi05"]["sample_std"], bold=is_best),
                _fmt_ms(rg["auc_roc"]["mean"], rg["auc_roc"]["sample_std"], bold=is_best),
                _fmt_pct_ms(
                    rg["recall_passive_single"]["mean"],
                    rg["recall_passive_single"]["sample_std"],
                    bold=is_best,
                ),
                _fmt_pct_ms(
                    rg["recall_esrc_single"]["mean"],
                    rg["recall_esrc_single"]["sample_std"],
                    bold=is_best,
                ),
                _fmt_pct_ms(
                    rg["recall_esrl_single"]["mean"],
                    rg["recall_esrl_single"]["sample_std"],
                    bold=is_best,
                ),
                _fmt_pct_ms(
                    rg["recall_rds1_single"]["mean"],
                    rg["recall_rds1_single"]["sample_std"],
                    bold=is_best,
                ),
                p_str,
            ]
        )

    t3 = TableSpec(
        table_id="table03_10seed_contrastive_ablation",
        caption=(
            "Table 3. 10-seed paired ablation of objective formulation and negative-sampling "
            "physics on `dataset_02` under identical `Transformer` backbones and strict "
            "zero-label `p97.5` calibration ($p_{\\min} = 0.00195$ for $n=10$)."
        ),
        label="tab:10seed_ablation",
        columns=[
            "Training & Scoring Regime",
            "$\\mathrm{F1}_{\\mathrm{p97.5}}$",
            "$\\mathrm{PR\\text{-}AUC}_{\\pi=0.05}$",
            "$\\mathrm{ROC\\text{-}AUC}$",
            "Passive Single Rec.",
            "$\\mathrm{ESR}_C$ Rec.",
            "$\\mathrm{ESR}_L$ Rec.",
            "$R_{\\mathrm{ds},1}$ Rec.",
            "Paired $\\Delta\\mathrm{F1}$ ($p$-value)",
        ],
        alignments=["l", "c", "c", "c", "c", "c", "c", "c", "c"],
        rows=ab_rows,
    )

    # --- Table 4: 5-Architecture Backbone Comparison ---
    enc_rows: List[List[str]] = []
    for arch_key, disp, is_best in [
        ("Transformer", "**`Transformer` (Self-Attention)**", True),
        ("LSTM", "`LSTM` (Recurrent)", False),
        ("Conv1D", "`Conv1D` (Local Kernel)", False),
        ("MLP", "`MLP` (Dense Flattened)", False),
        ("GRU", "`GRU` (Gated Recurrent)", False),
    ]:
        ac = enc_5["architectures"][arch_key]
        if arch_key == "GRU":
            f1_str = f"{ac['zero_p975_f1']['mean']:.4f} ± {ac['zero_p975_f1']['sample_std']:.4f} [{ac['zero_p975_f1']['median']:.4f}]"
            pr_str = f"{ac['auc_pr_pi05']['mean']:.4f} ± {ac['auc_pr_pi05']['sample_std']:.4f} [{ac['auc_pr_pi05']['median']:.4f}]"
            roc_str = f"{ac['auc_roc']['mean']:.4f} ± {ac['auc_roc']['sample_std']:.4f} [{ac['auc_roc']['median']:.4f}]"
            fra_str = f"{ac['fra015_matched_p975_f1']['mean']:.4f} ± {ac['fra015_matched_p975_f1']['sample_std']:.4f} [{ac['fra015_matched_p975_f1']['median']:.4f}]"
        else:
            f1_str = _fmt_ms(ac["zero_p975_f1"]["mean"], ac["zero_p975_f1"]["sample_std"], bold=is_best)
            pr_str = _fmt_ms(ac["auc_pr_pi05"]["mean"], ac["auc_pr_pi05"]["sample_std"], bold=is_best)
            roc_str = _fmt_ms(ac["auc_roc"]["mean"], ac["auc_roc"]["sample_std"], bold=is_best)
            fra_str = _fmt_ms(
                ac["fra015_matched_p975_f1"]["mean"],
                ac["fra015_matched_p975_f1"]["sample_std"],
                bold=is_best,
            )
        enc_rows.append(
            [
                disp,
                f1_str,
                pr_str,
                roc_str,
                fra_str,
                _fmt_pct_ms(ac["recall_esrc_single"]["mean"], ac["recall_esrc_single"]["sample_std"], bold=is_best),
                _fmt_pct_ms(ac["recall_esrl_single"]["mean"], ac["recall_esrl_single"]["sample_std"], bold=is_best),
                _fmt_pct_ms(ac["recall_rds1_single"]["mean"], ac["recall_rds1_single"]["sample_std"], bold=is_best),
            ]
        )

    t4 = TableSpec(
        table_id="table04_encoder_backbone_comparison",
        caption=(
            "Table 4. 5-architecture encoder backbone comparison under identical `CARLA (SSA)` "
            "nominal-load physics training and strict zero-label `p97.5` calibration on "
            "`dataset_02` (5 paired seeds, mean ± SD; median in brackets for bimodal `GRU`)."
        ),
        label="tab:encoder_backbones",
        columns=[
            "Encoder Backbone",
            "$\\mathrm{F1}_{\\mathrm{p97.5}}$ (Clean)",
            "$\\mathrm{PR\\text{-}AUC}_{\\pi=0.05}$",
            "$\\mathrm{ROC\\text{-}AUC}$",
            "$\\mathrm{F1}_{\\mathrm{p97.5}}$ ($0.15\\,\\mathrm{dB}$ FRA)",
            "$\\mathrm{ESR}_C$ Rec.",
            "$\\mathrm{ESR}_L$ Rec.",
            "$R_{\\mathrm{ds},1}$ Rec.",
        ],
        alignments=["l", "c", "c", "c", "c", "c", "c", "c"],
        rows=enc_rows,
    )

    # --- Table 5: Representation vs Scorer Matrix ---
    ctrl_rows: List[List[str]] = []
    for prefix, disp, is_ssa in [
        ("tf_recon_only__latent", "`AE-Transformer` ($\\mathbf{h}$, $w_c=0$)", False),
        ("tf_heuristic__projection", "`CARLA (Heuristic)` ($\\mathbf{z}$)", False),
        ("tf_physics_ssa_fixed_load__latent", "**`CARLA (SSA)` Latent ($\\mathbf{h}$)**", True),
        ("tf_physics_ssa_fixed_load__projection", "**`CARLA (SSA)` Projection ($\\mathbf{z}$)**", True),
    ]:
        c_knn = latent_ctrl[f"{prefix}__knn_k5"]
        c_lw = latent_ctrl[f"{prefix}__sklearn_ledoit_wolf"]
        c_p5 = latent_ctrl[f"{prefix}__pca_d5"]
        c_p10 = latent_ctrl[f"{prefix}__pca_d10"]
        ctrl_rows.append(
            [
                disp,
                _fmt_ms(c_knn["zero_p975_f1"]["mean"], c_knn["zero_p975_f1"]["sample_std"], bold=is_ssa),
                _fmt_ms(c_lw["zero_p975_f1"]["mean"], c_lw["zero_p975_f1"]["sample_std"], bold=is_ssa),
                _fmt_ms(c_p5["zero_p975_f1"]["mean"], c_p5["zero_p975_f1"]["sample_std"], bold=is_ssa),
                _fmt_ms(c_p10["zero_p975_f1"]["mean"], c_p10["zero_p975_f1"]["sample_std"], bold=is_ssa),
                _fmt_ms(c_knn["auc_pr_pi05"]["mean"], c_knn["auc_pr_pi05"]["sample_std"], bold=is_ssa),
                _fmt_ms(c_lw["auc_pr_pi05"]["mean"], c_lw["auc_pr_pi05"]["sample_std"], bold=is_ssa),
            ]
        )

    t5 = TableSpec(
        table_id="table05_representation_vs_scorer_matrix",
        caption=(
            "Table 5. Representation-versus-scorer control matrix on `dataset_02` "
            "(10 seeds, zero-label `p97.5` F1 and $\\mathrm{PR\\text{-}AUC}_{\\pi=0.05}$)."
        ),
        label="tab:representation_vs_scorer",
        columns=[
            "Learned Representation (10 Seeds)",
            "`k-NN (k=5)` $\\mathrm{F1}_{\\mathrm{p97.5}}$",
            "`Ledoit-Wolf` $\\mathrm{F1}_{\\mathrm{p97.5}}$",
            "`PCA (d=5)` $\\mathrm{F1}_{\\mathrm{p97.5}}$",
            "`PCA (d=10)` $\\mathrm{F1}_{\\mathrm{p97.5}}$",
            "`k-NN (k=5)` $\\mathrm{PR\\text{-}AUC}_{\\pi=0.05}$",
            "`Ledoit-Wolf` $\\mathrm{PR\\text{-}AUC}_{\\pi=0.05}$",
        ],
        alignments=["l", "c", "c", "c", "c", "c", "c"],
        rows=ctrl_rows,
    )

    # --- Table 6: 7-Level Parametric FRA Instrument Noise Sweep ---
    noise_keys = [nl["key"] for nl in fra_sweep["noise_levels"]]
    noise_cols = ["Detector (Zero-Label `p97.5`)"] + [
        f"${nl['sigma_db']:.2f}\\,\\mathrm{{dB}}$ (${nl['sigma_deg']:.2f}^\\circ$)"
        for nl in fra_sweep["noise_levels"]
    ]
    noise_rows: List[List[str]] = []

    carla_n_row = ["**`CARLA-Transformer (SSA)` (10 seeds)**"]
    for nk in noise_keys:
        obj = fra_sweep["detectors"]["carla_transformer_ssa"][nk]["zero_p975_f1"]
        carla_n_row.append(_fmt_ms(obj["mean"], obj["sample_std"], bold=(nk != "0.00dB_0.00deg" and nk != "0.05dB_0.25deg") or True))
    noise_rows.append(carla_n_row)

    for det_key, disp in [
        ("pca_d8_cleanfit", "`PCA Reconstruction (Oracle d=8)`"),
        ("pca_d10_cleanfit", "`PCA Reconstruction (Oracle d=10)`"),
        ("pca_d8_noisyrefit", "`PCA (d=8, Re-fit on Noisy Train)`"),
        ("mahalanobis_lw_cleanfit", "`Mahalanobis (Ledoit-Wolf)`"),
        ("raw_knn5_cleanfit", "`Raw k-NN (k=5)`"),
    ]:
        row_n = [disp]
        for nk in noise_keys:
            val = float(fra_sweep["detectors"][det_key][nk]["zero_p975_f1"])
            cell_txt = f"**{val:.4f}**" if (det_key == "pca_d8_cleanfit" and nk == "0.00dB_0.00deg") else f"{val:.4f}"
            row_n.append(cell_txt)
        noise_rows.append(row_n)

    t6 = TableSpec(
        table_id="table06_fra_noise_robustness_sweep",
        caption=(
            "Table 6. Parametric FRA instrument-noise robustness sweep on `dataset_02` across "
            "seven polar jitter levels $(\\sigma_{\\mathrm{dB}},\\, \\sigma_{\\mathrm{deg}} = "
            "5\\sigma_{\\mathrm{dB}})$ under zero-label `p97.5` calibration."
        ),
        label="tab:fra_noise_sweep",
        columns=noise_cols,
        alignments=["l"] + ["c"] * len(noise_keys),
        rows=noise_rows,
        notes=(
            f"On the primary matched 0.15 dB calibration realization (Fig. 7a), "
            f"CARLA-Transformer (SSA) achieves F1 = "
            f"{ablation_full['regimes']['tf_physics_ssa_fixed_load']['fra015_matched_p975_f1']['mean']:.4f} "
            f"± {ablation_full['regimes']['tf_physics_ssa_fixed_load']['fra015_matched_p975_f1']['sample_std']:.4f}."
        ),
    )

    # --- Table 7: Strict 5-Mode Leave-One-Fault-Out (LOFO) ---
    lofo_rows_out: List[List[str]] = []
    for comp_key, disp in [
        ("Cout", "$C_{\\mathrm{out}}$ (`capacitance_drop`)"),
        ("Lout", "$L_{\\mathrm{out}}$ (`inductor_saturation`)"),
        ("Esr_C", "$\\mathrm{ESR}_C$ (`capacitor_esr`)"),
        ("Esr_L", "$\\mathrm{ESR}_L$ (`inductor_esr`)"),
        ("Rds_1", "$R_{\\mathrm{ds},1}$ (`switch_degradation`)"),
    ]:
        md = lofo_5["modes"][comp_key]
        roc = md["lofo_single_auc_roc"]
        rec = md["lofo_zero_p975_recall_single"]
        ind_rec = md["indist_zero_p975_recall_single"]
        f1_all = md["lofo_zero_p975_overall_f1"]
        lofo_rows_out.append(
            [
                disp,
                f"$n = {int(md['n_test_single_fault'])}$",
                f"**{roc['mean']:.4f} ± {roc['sample_std']:.4f}** [{roc['median']:.4f}, {roc['iqr']:.4f}]",
                f"**{100.0 * rec['mean']:.1f}% ± {100.0 * rec['sample_std']:.1f}%** [{100.0 * rec['median']:.1f}%, {100.0 * rec['iqr']:.1f}%]",
                f"{100.0 * ind_rec['mean']:.1f}% ± {100.0 * ind_rec['sample_std']:.1f}%",
                f"{f1_all['mean']:.4f} ± {f1_all['sample_std']:.4f}",
            ]
        )

    t7 = TableSpec(
        table_id="table07_lofo_generalization",
        caption=(
            "Table 7. Strict 5-mode Leave-One-Fault-Out (LOFO) zero-shot generalization of "
            "`CARLA-Transformer (SSA)` on `dataset_02` (5 seeds, zero-label `p97.5`; "
            "mean ± SD and [median, IQR])."
        ),
        label="tab:lofo_generalization",
        columns=[
            "Omitted Fault Mode in Training",
            "Held-Out Count",
            "Unseen Mode LOFO $\\mathrm{ROC\\text{-}AUC}$",
            "Unseen Mode LOFO $\\mathrm{Recall}_{\\mathrm{p97.5}}$",
            "In-Distribution $\\mathrm{Recall}_{\\mathrm{p97.5}}$",
            "Overall Test $\\mathrm{F1}_{\\mathrm{p97.5}}$",
        ],
        alignments=["l", "c", "c", "c", "c", "c"],
        rows=lofo_rows_out,
    )

    return [t1, t2, t3, t4, t5, t6, t7]


def build_jsc_extension_tables(
    jsc_study_dir: Path,
    jetson_csv_path: Path,
) -> List[TableSpec]:
    """Construct 3 publication tables for the JSC Edge Inference & INT8 Rescue study."""
    int8_df = pd.read_csv(jsc_study_dir / "int8_recalibration_rescue.csv")
    conv_p = pd.read_csv(jsc_study_dir / "scorer_pareto_conv1d.csv")
    mlp_p = pd.read_csv(jsc_study_dir / "scorer_pareto_mlp.csv")
    jetson_df = pd.read_csv(jetson_csv_path)

    # --- JSC Table 1: INT8 Input-Saturation + Bank Recalibration Rescue ---
    j1_rows: List[List[str]] = []
    arch_labels = {
        "conv1d": "`Conv1D`",
        "mlp": "`MLP`",
        "transformer": "`Transformer`",
        "lstm": "`LSTM`",
        "gru": "`GRU`",
    }
    backend_labels = {"tflite_int8": "`TFLite INT8`", "onnx_int8": "`ONNX INT8`"}
    for arch in ["conv1d", "mlp", "transformer", "lstm", "gru"]:
        for bk in ["tflite_int8", "onnx_int8"]:
            sub = int8_df[(int8_df["architecture"] == arch) & (int8_df["backend"] == bk)]
            r_naive = sub[sub["strategy"] == "1_naive_fp32_bank_orig_scorer"].iloc[0]
            r_knn_z = sub[sub["strategy"] == "3_int8_recal_knn5_zero_label_p95"].iloc[0]
            r_p32_v = sub[sub["strategy"] == "4_int8_recal_proto32_val_opt"].iloc[0]
            r_p32_z = sub[sub["strategy"] == "5_int8_recal_proto32_zero_label_p95"].iloc[0]
            j1_rows.append(
                [
                    arch_labels[arch],
                    backend_labels[bk],
                    f"{float(r_naive['f1']):.4f} ({float(r_naive['discordance_vs_fp32_pct']):.1f}%)",
                    f"**{float(r_knn_z['f1']):.4f}** ({float(r_knn_z['discordance_vs_fp32_pct']):.1f}%)",
                    f"**{float(r_p32_v['f1']):.4f}**",
                    f"{float(r_p32_z['f1']):.4f}",
                    f"{float(r_naive['total_detector_mb']):.2f} $\\to$ **{float(r_p32_z['total_detector_mb']):.2f}**",
                ]
            )

    jt1 = TableSpec(
        table_id="jsc_table01_int8_recalibration_rescue",
        caption=(
            "Table J1. Post-audit INT8 quantization rescue across 5 CARLA architectures via "
            "input-clipping saturation, INT8 embedding-bank recalibration, and $M=32$ "
            "prototype compression (F1 and FP32 decision discordance %)."
        ),
        label="tab:jsc_int8_rescue",
        columns=[
            "Architecture",
            "INT8 Runtime",
            "Naive FP32 Bank F1 (Disc.)",
            "Recal. `k-NN (k=5)` Zero-`p95` F1 (Disc.)",
            "Recal. `Proto-32` Val-Opt F1",
            "Recal. `Proto-32` Zero-`p95` F1",
            "Detector Size [MB]",
        ],
        alignments=["l", "c", "c", "c", "c", "c", "c"],
        rows=j1_rows,
    )

    # --- JSC Table 2: Prototype Scorer Latency-Memory-Accuracy Pareto ---
    j2_rows: List[List[str]] = []
    variant_names = {
        "legacy_sklearn_knn_N2419": "`Legacy Sklearn k-NN (N=2419)`",
        "fast_blas_knn_N2419": "`Vectorized BLAS k-NN (N=2419)`",
        "ridge_mahalanobis_1e6": "`Ridge Mahalanobis (d=513)`",
        "ledoit_wolf_precision": "`Ledoit-Wolf Shrinkage (d=513)`",
        "proto_knn_M128": "**`Prototype k-NN (M=128)`**",
        "proto_knn_M64": "**`Prototype k-NN (M=64)`**",
        "proto_knn_M32": "**`Prototype k-NN (M=32)`**",
        "proto_knn_M16": "`Prototype k-NN (M=16)`",
        "centroid_M1": "`Single Centroid (M=1)`",
    }
    for arch_tag, df_p in [("`Conv1D`", conv_p), ("`MLP`", mlp_p)]:
        for v_key, v_disp in variant_names.items():
            matches = df_p[df_p["scorer_variant"] == v_key]
            if matches.empty:
                continue
            r = matches.iloc[0]
            j2_rows.append(
                [
                    arch_tag,
                    v_disp,
                    f"{int(r['n_bank_vectors'])}",
                    f"{float(r['bank_memory_mb']):.4f}",
                    f"{float(r['total_detector_mb']):.3f}",
                    f"{float(r['scorer_latency_p50_ms']):.3f}",
                    f"{float(r['auc_roc']):.4f}",
                    f"{float(r['f1_unsup_p95']):.4f}",
                ]
            )

    jt2 = TableSpec(
        table_id="jsc_table02_scorer_pareto_frontier",
        caption=(
            "Table J2. Embedding anomaly-scorer Pareto trade-off between reference-bank "
            "memory, median CPU scoring latency ($p_{50}$), and zero-label `p95` F1."
        ),
        label="tab:jsc_scorer_pareto",
        columns=[
            "Backbone",
            "Scorer Variant",
            "Vectors / Rank",
            "Bank [MB]",
            "Total [MB]",
            "Scorer $p_{50}$ [ms]",
            "$\\mathrm{ROC\\text{-}AUC}$",
            "Zero-`p95` F1",
        ],
        alignments=["l", "l", "r", "r", "r", "r", "c", "c"],
        rows=j2_rows,
    )

    # --- JSC Table 3: Validated Jetson Nano Runtime Summary ---
    val_df = jetson_df[jetson_df["status"] == "validated"].copy()
    j3_rows: List[List[str]] = []
    for model_name in ["Conv1D", "MLP", "Transformer", "LSTM", "GRU"]:
        sub_m = val_df[val_df["model"] == model_name].sort_values("latency_ms")
        if sub_m.empty:
            continue
        best_r = sub_m.iloc[0]
        fp32_cpu = sub_m[sub_m["backend"] == "Keras FP32 (CPU)"]
        base_lat = float(fp32_cpu.iloc[0]["latency_ms"]) if not fp32_cpu.empty else float(best_r["latency_ms"])
        speedup = base_lat / max(float(best_r["latency_ms"]), 1e-9)
        det_mb = (
            f"{float(best_r['detector_size_mb']):.2f}"
            if pd.notna(best_r["detector_size_mb"])
            else "—"
        )
        j3_rows.append(
            [
                f"`{model_name}`",
                f"`{best_r['backend']}`",
                f"**{float(best_r['latency_ms']):.3f}**",
                f"{speedup:.1f}$\\times$",
                f"{float(best_r['f1']):.4f}",
                f"{float(best_r['roc_auc']):.4f}",
                det_mb,
            ]
        )

    jt3 = TableSpec(
        table_id="jsc_table03_jetson_deployment_summary",
        caption=(
            "Table J3. Fastest validated edge runtime per architecture on NVIDIA Jetson Nano "
            "(Maxwell 128-core GPU) vs. baseline `Keras FP32 (CPU)` latency."
        ),
        label="tab:jsc_jetson_summary",
        columns=[
            "Architecture",
            "Fastest Validated Backend",
            "Graph Latency [ms]",
            "Speedup vs Keras CPU",
            "Test F1",
            "$\\mathrm{ROC\\text{-}AUC}$",
            "Detector Size [MB]",
        ],
        alignments=["l", "l", "r", "r", "c", "c", "r"],
        rows=j3_rows,
    )

    return [jt1, jt2, jt3]
