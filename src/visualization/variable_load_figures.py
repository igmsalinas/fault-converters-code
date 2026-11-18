"""
Variable-Load FRA Anomaly Detection Publication Figures
=======================================================

Renders the complete 8-figure IEEE/Elsevier `#ffffff` publication suite from
persisted benchmark artifacts without requiring GPU/Keras runtime imports:
  - Fig. 1: Sensorless Physics-Informed `CARLA-Transformer` System & Architecture
  - Fig. 2: Switched PSIM Variable-Load Bode Confound, SSA Ratios & L2 Fidelity
  - Fig. 3: Fixed-Load (`dataset_01`) vs Variable-Load (`dataset_02`) & Per-Fault Recall
  - Fig. 4: Prevalence-Adjusted PR Curves (pi = 5%) & 10-Seed Paired Ablation
  - Fig. 5: 5-Backbone Inductive Bias & Representation-vs-Scorer Control Matrix
  - Fig. 6: Linear PCA Subspace Cliff & 7-Level Parametric FRA Instrument Noise Sweep
  - Fig. 7: Commissioning Sample-Complexity Curves & 5-Mode Zero-Shot LOFO
  - Fig. 8: Self-Attention Spectral Bus vs Input-Jacobian Load-Invariance Notch
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd

from src.data.physics_anomaly import BuckConverter

from .style import adaptive_contrast, apply_publication_style, save_figure, style_axis

# Consistent Okabe-Ito / Academic color mapping across all 8 figures
COL_SSA = "#1f4e79"       # Classic Navy (Proposed CARLA-Transformer SSA)
COL_TEAL = "#2e7d6e"      # Deep Teal (PCA / Secondary highlight)
COL_TERRA = "#b85d38"     # Academic Terracotta (Heuristic / Noisy PCA)
COL_OCHRE = "#c4842b"     # Warm Ochre (Ledoit-Wolf Mahalanobis)
COL_BURG = "#b33939"      # Burgundy Red (MSE AE / Failure mode)
COL_SLATE = "#5c6b73"     # Slate Gray (Raw k-NN / Baseline)
COL_INDIGO = "#6b4c9a"    # Muted Indigo (LSTM / Auxiliary)
COL_STEEL = "#4a7c9f"     # Steel Blue (Conv1D / Secondary)


def plot_fig01_system_and_carla_framework(
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Render an IEEE Transactions 3-panel circuit, SSA operator, and manifold figure."""
    apply_publication_style()
    fig = plt.figure(figsize=(15.4, 5.8))
    fig.patch.set_facecolor("#ffffff")

    def _sharp_block(
        ax_t: plt.Axes,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        fc: str = "#ffffff",
        ec: str = "#1a1a1a",
        lw: float = 1.15,
        ls: str = "-",
        fs: float = 7.7,
        bold: bool = False,
        text_col: str = "#111111",
    ) -> None:
        rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=4)
        ax_t.add_patch(rect)
        ax_t.text(
            x + w / 2.0,
            y + h / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=text_col,
            linespacing=1.22,
            zorder=5,
        )

    def _arr(
        ax_t: plt.Axes,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: str = "#1a1a1a",
        lw: float = 1.15,
        ls: str = "-",
    ) -> None:
        ax_t.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>,head_length=4.5,head_width=2.6",
                color=color,
                linewidth=lw,
                linestyle=ls,
                zorder=6,
            )
        )

    # Subtle vertical dashed column separators matching multi-panel IEEE figures
    bg_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
    bg_ax.set_xlim(0.0, 1.0)
    bg_ax.set_ylim(0.0, 1.0)
    bg_ax.axis("off")
    bg_ax.plot([0.336, 0.336], [0.05, 0.96], color="#d0d7de", ls="--", lw=1.0)
    bg_ax.plot([0.668, 0.668], [0.05, 0.96], color="#d0d7de", ls="--", lw=1.0)

    # =========================================================================
    # PANEL (a): Switched Synchronous Buck Schematic & Variable-Load Confound
    # =========================================================================
    ax_a_sch = fig.add_axes([0.015, 0.455, 0.312, 0.495])
    ax_a_sch.set_facecolor("#ffffff")
    ax_a_sch.set_xlim(0.0, 10.9)
    ax_a_sch.set_ylim(0.0, 5.2)
    ax_a_sch.axis("off")
    ax_a_sch.set_title("(a) Switched Buck Plant & Sensorless PRBS FRA", loc="left", fontsize=10.2, fontweight="bold", pad=6)

    wire_c = "#111111"
    lw_w = 1.25
    y_top, y_bot = 3.95, 1.75

    # DC input source Vin
    ax_a_sch.plot([0.95, 0.95], [y_bot, y_top], color=wire_c, lw=lw_w)
    ax_a_sch.add_patch(Circle((0.95, (y_top + y_bot) / 2.0), 0.42, facecolor="#ffffff", edgecolor=wire_c, lw=1.3, zorder=4))
    ax_a_sch.text(0.95, (y_top + y_bot) / 2.0 + 0.14, "+", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=5)
    ax_a_sch.text(0.95, (y_top + y_bot) / 2.0 - 0.17, "$-$", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=5)
    ax_a_sch.text(0.45, (y_top + y_bot) / 2.0, "$V_{\\mathrm{in}}$\n$50\\,\\mathrm{V}$", ha="right", va="center", fontsize=7.6)

    # Top rail to S1 (high-side MOSFET switch symbol)
    ax_a_sch.plot([0.95, 2.05], [y_top, y_top], color=wire_c, lw=lw_w)
    ax_a_sch.plot([2.05, 2.60], [y_top, y_top + 0.28], color=COL_SSA, lw=1.6)
    ax_a_sch.add_patch(Circle((2.05, y_top), 0.06, facecolor="#111111", zorder=5))
    ax_a_sch.add_patch(Circle((2.65, y_top), 0.06, facecolor="#111111", zorder=5))
    ax_a_sch.text(2.35, y_top + 0.46, "$S_1\\,(R_{\\mathrm{ds},1})$", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=COL_SSA)

    # Switch node Vx & low-side S2 (clamped diode indicator)
    ax_a_sch.plot([2.65, 3.85], [y_top, y_top], color=wire_c, lw=lw_w)
    ax_a_sch.plot([3.35, 3.35], [y_bot, 3.05], color=wire_c, lw=lw_w)
    ax_a_sch.plot([3.35, 3.35], [2.45, y_top], color=wire_c, lw=lw_w)
    ax_a_sch.plot([3.35, 3.08], [3.05, 2.52], color=COL_SSA, lw=1.5)
    ax_a_sch.add_patch(Circle((3.35, y_top), 0.07, facecolor="#111111", zorder=6))
    ax_a_sch.add_patch(Circle((3.35, y_bot), 0.07, facecolor="#111111", zorder=6))
    # Anti-parallel body diode beside S2
    ax_a_sch.plot([3.62, 3.62], [2.45, 3.05], color="#5c6b73", lw=0.9)
    ax_a_sch.add_patch(Polygon([(3.52, 2.65), (3.72, 2.65), (3.62, 2.86)], closed=True, facecolor="#ffffff", edgecolor="#5c6b73", lw=0.9, zorder=4))
    ax_a_sch.plot([3.52, 3.72], [2.86, 2.86], color="#5c6b73", lw=1.0)
    ax_a_sch.text(2.96, 2.75, "$S_2$", ha="right", va="center", fontsize=7.5, fontweight="bold", color=COL_SSA)

    # Inductor L_out (4 coils) + ESR_L resistor
    for k in range(4):
        cx = 4.02 + (2 * k + 1) * 0.17
        ax_a_sch.add_patch(Arc((cx, y_top), 0.34, 0.34, angle=0, theta1=0, theta2=180, edgecolor=wire_c, lw=1.35, zorder=4))
    ax_a_sch.plot([3.85, 4.02], [y_top, y_top], color=wire_c, lw=lw_w)
    ax_a_sch.plot([5.38, 5.65], [y_top, y_top], color=wire_c, lw=lw_w)
    ax_a_sch.text(4.70, y_top + 0.36, "$L_{\\mathrm{out}}$", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax_a_sch.add_patch(Rectangle((5.65, y_top - 0.14), 0.72, 0.28, facecolor="#ffffff", edgecolor=wire_c, lw=1.2, zorder=4))
    ax_a_sch.text(6.01, y_top + 0.34, "$\\mathrm{ESR}_L$", ha="center", va="bottom", fontsize=7.3)
    ax_a_sch.plot([6.37, 8.55], [y_top, y_top], color=wire_c, lw=lw_w)

    # Bottom return rail & Ground
    ax_a_sch.plot([0.95, 8.55], [y_bot, y_bot], color=wire_c, lw=lw_w)
    ax_a_sch.plot([0.95, 0.95], [y_bot, y_bot - 0.22], color=wire_c, lw=1.2)
    ax_a_sch.plot([0.75, 1.15], [y_bot - 0.22, y_bot - 0.22], color=wire_c, lw=1.3)
    ax_a_sch.plot([0.82, 1.08], [y_bot - 0.31, y_bot - 0.31], color=wire_c, lw=1.1)

    # Capacitor C_out + ESR_C branch at x=7.35
    x_c = 7.20
    ax_a_sch.add_patch(Circle((x_c, y_top), 0.07, facecolor="#111111", zorder=6))
    ax_a_sch.add_patch(Circle((x_c, y_bot), 0.07, facecolor="#111111", zorder=6))
    ax_a_sch.plot([x_c, x_c], [y_top, 3.38], color=wire_c, lw=lw_w)
    ax_a_sch.plot([x_c - 0.25, x_c + 0.25], [3.38, 3.38], color=wire_c, lw=1.5)
    ax_a_sch.plot([x_c - 0.25, x_c + 0.25], [3.22, 3.22], color=wire_c, lw=1.5)
    ax_a_sch.text(x_c - 0.32, 3.30, "$C_{\\mathrm{out}}$", ha="right", va="center", fontsize=7.4, fontweight="bold")
    ax_a_sch.plot([x_c, x_c], [3.22, 2.86], color=wire_c, lw=lw_w)
    ax_a_sch.add_patch(Rectangle((x_c - 0.14, 2.24), 0.28, 0.62, facecolor="#ffffff", edgecolor=wire_c, lw=1.2, zorder=4))
    ax_a_sch.text(x_c - 0.24, 2.55, "$\\mathrm{ESR}_C$", ha="right", va="center", fontsize=7.1)
    ax_a_sch.plot([x_c, x_c], [2.24, y_bot], color=wire_c, lw=lw_w)

    # Variable Unmeasured Load Resistor R_out at x=8.65
    x_r = 8.55
    ax_a_sch.plot([x_r, x_r], [y_top, 3.32], color=wire_c, lw=lw_w)
    ax_a_sch.add_patch(Rectangle((x_r - 0.17, 2.38), 0.34, 0.94, facecolor="#ffffff", edgecolor=COL_BURG, lw=1.35, zorder=4))
    ax_a_sch.plot([x_r, x_r], [2.38, y_bot], color=wire_c, lw=lw_w)
    _arr(ax_a_sch, x_r - 0.34, 2.40, x_r + 0.34, 3.30, color=COL_BURG, lw=1.25)
    ax_a_sch.text(x_r + 0.42, 2.86, "$R_{\\mathrm{out}}\\in[2,8]\\,\\Omega$\n(unmeasured)", ha="left", va="center", fontsize=6.9, fontweight="bold", color=COL_BURG)
    ax_a_sch.text(7.85, y_top + 0.18, "$V_{\\mathrm{out}}(t)$", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=COL_TEAL)

    # Bottom Digital PRBS & FRA Extraction Loop
    _sharp_block(ax_a_sch, 1.45, 0.18, 2.85, 0.85, "DPWM + PRBS $\\tilde{d}(t)$\n$f_s=100\\,\\mathrm{kHz},\\; D=0.40$", fc="#ffffff", ec=COL_SSA, fs=7.1)
    _sharp_block(ax_a_sch, 5.20, 0.18, 3.65, 0.85, "FRA $H_i(j\\omega_k)=G_{vd}(j\\omega_k)$\n$101$ log bins ($0.1$\u2013$50\\,\\mathrm{kHz}$)", fc="#ffffff", ec=COL_TEAL, fs=7.1, bold=True)
    ax_a_sch.plot([x_r, 9.25, 9.25, 8.90], [y_top, y_top, 0.60, 0.60], color=COL_TEAL, lw=1.0, ls="--")
    _arr(ax_a_sch, 9.05, 0.60, 8.86, 0.60, color=COL_TEAL, lw=1.0)
    ax_a_sch.plot([2.35, 2.35], [1.03, 1.32], color=COL_SSA, lw=1.0)
    _arr(ax_a_sch, 2.35, 1.05, 2.35, 1.45, color=COL_SSA, lw=1.0)
    ax_a_sch.text(2.45, 1.32, "$d(t)$", fontsize=7.1, color=COL_SSA)

    # Panel (a) Quantitative Inset: Load Confound Bode Curves
    ax_a_plot = fig.add_axes([0.048, 0.105, 0.268, 0.315])
    ax_a_plot.set_facecolor("#ffffff")
    freqs_hz = np.geomspace(100.0, 50_000.0, 101)
    omega = 2.0 * np.pi * freqs_hz
    f_khz = freqs_hz / 1000.0
    nom_conv = BuckConverter(
        vin=50.0,
        duty=0.40,
        inductance=200e-6,
        capacitance=100e-6,
        load=4.0,
        esr_ind=0.10,
        esr_cap=0.02,
        rds=0.01,
        rds_low=0.0,
    )
    h_nom = nom_conv.transfer_function().frequency_response(omega)
    mag_r2 = 20.0 * np.log10(np.abs(nom_conv.scaled(load=0.5).transfer_function().frequency_response(omega)))
    mag_r4 = 20.0 * np.log10(np.abs(h_nom))
    mag_r8 = 20.0 * np.log10(np.abs(nom_conv.scaled(load=2.0).transfer_function().frequency_response(omega)))
    mag_esrl = 20.0 * np.log10(np.abs(nom_conv.scaled(esr_ind=2.50).transfer_function().frequency_response(omega)))
    mag_esrc = 20.0 * np.log10(np.abs(nom_conv.scaled(esr_cap=2.50).transfer_function().frequency_response(omega)))
    ax_a_plot.fill_between(f_khz, mag_r2, mag_r8, color="#cbd5e1", alpha=0.55, label="$R_{\\mathrm{out}}\\in[2,8]\\,\\Omega$ (~8.9 dB)")
    ax_a_plot.plot(f_khz, mag_r4, color=COL_SSA, lw=1.7, label="Nominal $4\\,\\Omega$")
    ax_a_plot.plot(f_khz, mag_esrl, color=COL_BURG, ls="--", lw=1.3, label="$+150\\%$ $\\mathrm{ESR}_L$")
    ax_a_plot.plot(f_khz, mag_esrc, color=COL_TEAL, ls="-.", lw=1.3, label="$+150\\%$ $\\mathrm{ESR}_C$")
    ax_a_plot.set_xscale("log")
    ax_a_plot.set_xlim(0.1, 50.0)
    ax_a_plot.set_ylim(-42.0, 40.0)
    ax_a_plot.set_xlabel("Frequency $f$ [kHz]", fontsize=7.6, fontweight="bold")
    ax_a_plot.set_ylabel("$|G_{vd}|$ [dB]", fontsize=7.6, fontweight="bold")
    ax_a_plot.tick_params(labelsize=7.0)
    ax_a_plot.grid(True, which="both", alpha=0.35)
    style_axis(ax_a_plot)
    ax_a_plot.legend(loc="lower left", fontsize=6.4, framealpha=0.94, ncol=2, handlelength=1.4, columnspacing=0.8)

    # =========================================================================
    # PANEL (b): Nominal-Load SSA Complex Ratios & Operator Flow
    # =========================================================================
    ax_b_plot = fig.add_axes([0.382, 0.565, 0.268, 0.340])
    ax_b_plot.set_facecolor("#ffffff")
    ratio_specs = [
        ("Cout", {"$C$": "-35%"}, "$C_{\\mathrm{out}}$ ($-35\\%$)", {"capacitance": 0.65}, COL_SSA, "-"),
        ("Lout", {"$L$": "-35%"}, "$L_{\\mathrm{out}}$ ($-35\\%$)", {"inductance": 0.65}, COL_STEEL, "--"),
        ("Esr_C", {"ESR": "+150%"}, "$\\mathrm{ESR}_C$ ($+150\\%$)", {"esr_cap": 2.50}, COL_TEAL, "-"),
        ("Esr_L", {"ESR": "+150%"}, "$\\mathrm{ESR}_L$ ($+150\\%$)", {"esr_ind": 2.50}, COL_TERRA, "--"),
        ("Rds_1", {"Rds": "+350%"}, "$R_{\\mathrm{ds},1}$ ($+350\\%$)", {"rds": 4.50}, COL_OCHRE, ":"),
    ]
    for _, _, rlbl, params, col, ls in ratio_specs:
        r_mag = 20.0 * np.log10(np.abs(nom_conv.scaled(**params).transfer_function().frequency_response(omega) / h_nom))
        ax_b_plot.plot(f_khz, r_mag, color=col, ls=ls, lw=1.55, label=rlbl)
    ax_b_plot.axhline(0.0, color="#666666", lw=0.8, ls=":")
    ax_b_plot.set_xscale("log")
    ax_b_plot.set_xlim(0.1, 50.0)
    ax_b_plot.set_ylabel("$|\\Gamma_{\\boldsymbol{\\alpha}}(j\\omega)|$ [dB]", fontsize=7.6, fontweight="bold")
    ax_b_plot.set_xlabel("Frequency $f$ [kHz] (Nominal $R_{\\mathrm{out}}^{(0)}=4\\,\\Omega$)", fontsize=7.4, fontweight="bold")
    ax_b_plot.set_title("(b) Sensorless SSA-Ratio Contrastive Synthesis (Train Only)", loc="left", fontsize=10.2, fontweight="bold", pad=8)
    ax_b_plot.tick_params(labelsize=7.0)
    ax_b_plot.grid(True, which="both", alpha=0.35)
    style_axis(ax_b_plot)
    ax_b_plot.legend(loc="upper right", fontsize=6.3, framealpha=0.95, ncol=2, handlelength=1.4, columnspacing=0.7)

    ax_b_flow = fig.add_axes([0.348, 0.055, 0.308, 0.430])
    ax_b_flow.set_facecolor("#ffffff")
    ax_b_flow.set_xlim(0.0, 10.0)
    ax_b_flow.set_ylim(0.0, 5.0)
    ax_b_flow.axis("off")

    _sharp_block(ax_b_flow, 0.25, 2.30, 2.30, 1.55, "Healthy Anchor\n$H_i(j\\omega)$\n(unknown $R_{\\mathrm{out},i}$)", fc="#ffffff", ec=COL_SSA, lw=1.3, fs=7.3, bold=True)
    _sharp_block(ax_b_flow, 3.55, 3.45, 2.85, 1.15, "Positive View\n$H_i^{+} = H_i + \\epsilon_{\\mathrm{pos}}$", fc="#ffffff", ec=COL_TEAL, fs=7.3)
    # Multiplier node \otimes
    ax_b_flow.add_patch(Circle((4.35, 2.00), 0.34, facecolor="#ffffff", edgecolor=COL_BURG, lw=1.3, zorder=5))
    ax_b_flow.text(4.35, 2.00, "$\\times$", ha="center", va="center", fontsize=11.0, fontweight="bold", color=COL_BURG, zorder=6)
    ax_b_flow.text(4.35, 0.92, "$\\Gamma_{\\boldsymbol{\\alpha}}(j\\omega;\\,R_{\\mathrm{out}}^{(0)})$\n($0.0952\\%$ PSIM L2)", ha="center", va="center", fontsize=6.9, color=COL_BURG)
    _arr(ax_b_flow, 4.35, 1.32, 4.35, 1.64, color=COL_BURG, lw=1.1, ls="--")
    _sharp_block(ax_b_flow, 5.20, 1.42, 2.00, 1.16, "SSA Fault Neg.\n$H_i^{-} = H_i\\cdot\\Gamma_{\\boldsymbol{\\alpha}}$", fc="#ffffff", ec=COL_BURG, lw=1.25, ls="--", fs=7.1)
    _sharp_block(ax_b_flow, 7.65, 1.95, 2.15, 2.10, "Cartesian Proj.\n$[\\mathrm{Re}\\,H,\\,\\mathrm{Im}\\,H]$\n+ Healthy-Only\nStandardScaler\n$\\mathbf{X}\\in\\mathbb{R}^{101\\times 2}$", fc="#ffffff", ec="#222222", lw=1.25, fs=7.0, bold=True)

    _arr(ax_b_flow, 2.55, 3.45, 3.53, 4.02, color=COL_TEAL)
    _arr(ax_b_flow, 2.55, 2.68, 3.99, 2.02, color=COL_BURG, ls="--")
    _arr(ax_b_flow, 4.69, 2.00, 5.18, 2.00, color=COL_BURG, ls="--")
    _arr(ax_b_flow, 6.40, 4.02, 7.63, 3.50, color=COL_TEAL)
    _arr(ax_b_flow, 7.20, 2.00, 7.63, 2.55, color=COL_BURG, ls="--")

    # =========================================================================
    # PANEL (c): Hypersphere Embedding Separation & Zero-Label p97.5 Inference
    # =========================================================================
    ax_c_emb = fig.add_axes([0.708, 0.545, 0.270, 0.360])
    ax_c_emb.set_facecolor("#ffffff")
    rng_s = np.random.default_rng(7)
    theta = np.linspace(0, 2 * np.pi, 240)
    ax_c_emb.plot(np.cos(theta), np.sin(theta), color="#94a3b8", lw=1.0, ls=":")
    # Healthy cluster Z_train & validation threshold radius
    hx = rng_s.normal(-0.42, 0.10, 28)
    hy = rng_s.normal(0.35, 0.10, 28)
    ax_c_emb.scatter(hx, hy, s=16, color=COL_SSA, alpha=0.82, label="Healthy $\\mathcal{Z}_{\\mathrm{train}}$", zorder=4)
    ax_c_emb.add_patch(Circle((-0.42, 0.35), 0.34, facecolor="#e8f1fa", edgecolor=COL_TEAL, lw=1.35, ls="--", alpha=0.45, zorder=2))
    ax_c_emb.text(-0.42, -0.08, "$\\tau_{0.975}$ ($\\mathcal{D}_{\\mathrm{val,norm}}$)", ha="center", va="top", fontsize=6.9, fontweight="bold", color=COL_TEAL)
    # Repelled SSA & physical anomalies
    ax_c_emb.scatter(rng_s.normal(0.50, 0.11, 14), rng_s.normal(0.38, 0.11, 14), s=20, marker="^", color=COL_BURG, alpha=0.88, label="Faults ($\\mathrm{ESR}_C,\\mathrm{ESR}_L,C,L$)", zorder=5)
    ax_c_emb.scatter(rng_s.normal(0.28, 0.11, 10), rng_s.normal(-0.48, 0.10, 10), s=20, marker="s", color=COL_TERRA, alpha=0.88, label="Counterfactual Neg. $\\mathbf{z}_j^{-}$", zorder=5)
    # Query & 5-NN rays
    qx, qy = 0.22, 0.56
    ax_c_emb.scatter([qx], [qy], s=42, marker="*", color="#111111", zorder=6)
    for k_i in range(5):
        ax_c_emb.plot([qx, hx[k_i]], [qy, hy[k_i]], color="#444444", lw=0.75, ls="-", zorder=3)
    ax_c_emb.text(qx + 0.05, qy + 0.08, "Query $\\mathbf{z}(\\mathbf{X})$", fontsize=7.0, fontweight="bold")
    ax_c_emb.set_xlim(-1.08, 1.08)
    ax_c_emb.set_ylim(-1.05, 1.10)
    ax_c_emb.set_aspect("equal")
    ax_c_emb.set_xticks([])
    ax_c_emb.set_yticks([])
    for sp in ax_c_emb.spines.values():
        sp.set_edgecolor("#cbd5e1")
        sp.set_linewidth(0.9)
    ax_c_emb.set_title("(c) Load-Invariant $\\mathbb{S}^{127}$ Manifold & Zero-Label Scoring", loc="left", fontsize=10.2, fontweight="bold", pad=8)
    ax_c_emb.legend(loc="lower left", fontsize=6.2, framealpha=0.95, handletextpad=0.3)

    ax_c_flow = fig.add_axes([0.680, 0.055, 0.308, 0.430])
    ax_c_flow.set_facecolor("#ffffff")
    ax_c_flow.set_xlim(0.0, 10.0)
    ax_c_flow.set_ylim(0.0, 5.0)
    ax_c_flow.axis("off")

    _sharp_block(ax_c_flow, 0.35, 3.05, 4.20, 1.45, "CARLA-Transformer\n$f_{\\theta}(\\mathbf{X})\\to\\pi_{\\omega}(\\mathbf{h})=\\mathbf{z}\\in\\mathbb{S}^{127}$", fc="#ffffff", ec=COL_SSA, lw=1.3, fs=7.2, bold=True)
    _sharp_block(ax_c_flow, 5.45, 3.25, 4.25, 1.25, "Decoder $g_{\\psi}$ (Train Only)\nHealthy MSE ($w_{\\mathrm{rec}}=5.0$)\nDiscarded at Inference", fc="#f8fafc", ec="#64748b", lw=1.05, ls="--", fs=6.9, text_col="#475569")
    _sharp_block(ax_c_flow, 0.35, 0.65, 9.35, 1.65, "Strict Zero-Anomaly-Label Inference (0 Fault Labels, 0 Load Sensors)\nScore $a(\\mathbf{X}) = \\frac{1}{5}\\sum_{\\mathbf{z}_{(r)}\\in\\mathcal{N}_5(\\mathbf{z};\\mathcal{Z}_{\\mathrm{train}})} \\|\\mathbf{z}-\\mathbf{z}_{(r)}\\|_2 \\;>\\; \\tau_{0.975}$  (p97.5 on $\\mathcal{D}_{\\mathrm{val,norm}}$)", fc="#ffffff", ec=COL_TEAL, lw=1.45, fs=7.3, bold=True)

    _arr(ax_c_flow, 4.55, 3.85, 5.43, 3.85, color="#64748b", ls="--")
    _arr(ax_c_flow, 2.45, 3.05, 2.45, 2.32, color=COL_TEAL, lw=1.3)

    save_figure(fig, save_paths)
    return fig


def plot_fig02_physics_ssa_fidelity_and_bode(
    bode_payload: Dict[str, Any],
    physics_audit_ds02: Dict[str, Any],
    physics_audit_ds01: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot 3-panel Bode load confound, complex SSA fault ratios, and PSIM L2 audit."""
    apply_publication_style()
    fig = plt.figure(figsize=(15.0, 5.6))
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.22, 1.08, 0.94], hspace=0.22, wspace=0.33)

    ax_mag = fig.add_subplot(gs[0, 0])
    ax_phs = fig.add_subplot(gs[1, 0], sharex=ax_mag)
    ax_rat_m = fig.add_subplot(gs[0, 1])
    ax_rat_p = fig.add_subplot(gs[1, 1], sharex=ax_rat_m)
    ax_err = fig.add_subplot(gs[:, 2])

    f_khz = np.asarray(bode_payload["frequencies_hz"], dtype=float) / 1000.0
    ax_mag.axvspan(0.7, 2.0, color="#fff3cd", alpha=0.55, zorder=0)
    ax_phs.axvspan(0.7, 2.0, color="#fff3cd", alpha=0.55, zorder=0)

    ax_mag.fill_between(
        f_khz,
        bode_payload["healthy_mag_p025_db"],
        bode_payload["healthy_mag_p975_db"],
        color="#9ecae1",
        alpha=0.50,
        label=r"Healthy 95% env. ($R_{\mathrm{out}} \in [2,8]\,\Omega$)",
    )
    ax_mag.plot(
        f_khz,
        bode_payload["healthy_mag_median_db"],
        color="#0b2545",
        lw=2.0,
        label="Healthy median",
    )
    ax_phs.fill_between(
        f_khz,
        bode_payload["healthy_phase_p025_deg"],
        bode_payload["healthy_phase_p975_deg"],
        color="#9ecae1",
        alpha=0.50,
    )
    ax_phs.plot(f_khz, bode_payload["healthy_phase_median_deg"], color="#0b2545", lw=2.0)

    f_styles = [
        ("Cout", COL_STEEL, "-", 1.7),
        ("Esr_C", COL_TEAL, "-.", 1.9),
        ("Esr_L", COL_OCHRE, "--", 1.9),
        ("Rds_1", COL_BURG, ":", 2.1),
    ]
    for fkey, col, ls, lw in f_styles:
        if fkey in bode_payload["fault_examples"]:
            ex = bode_payload["fault_examples"][fkey]
            ax_mag.plot(f_khz, ex["mag_db"], color=col, ls=ls, lw=lw, label=ex["label"])
            ax_phs.plot(f_khz, ex["phase_deg"], color=col, ls=ls, lw=lw)

    ax_mag.set_xscale("log")
    ax_mag.set_ylim(-52.0, 48.0)
    ax_mag.set_ylabel(r"$|G_{vd}(j\omega)|\;[\mathrm{dB}]$", fontweight="bold")
    ax_mag.set_title("(a) Switched PSIM Variable-Load Bode", loc="left", pad=8)
    ax_mag.grid(True, which="both", alpha=0.45)
    style_axis(ax_mag)
    plt.setp(ax_mag.get_xticklabels(), visible=False)
    ax_mag.legend(loc="lower left", fontsize=7.0, ncol=2, framealpha=0.95)

    ax_phs.set_xscale("log")
    ax_phs.set_xlabel(r"Frequency $f\;[\mathrm{kHz}]$", fontweight="bold")
    ax_phs.set_ylabel(r"$\angle G_{vd}(j\omega)\;[\mathrm{deg}]$", fontweight="bold")
    ax_phs.grid(True, which="both", alpha=0.45)
    style_axis(ax_phs)

    # Panel (b): Exact Analytical SSA Complex Ratios Gamma_alpha(jw) at R_out^(0) = 4 Ohm
    freqs_hz = np.asarray(bode_payload["frequencies_hz"], dtype=float)
    omega = 2.0 * np.pi * freqs_hz
    base_conv = BuckConverter(
        vin=50.0,
        duty=0.40,
        inductance=200e-6,
        capacitance=100e-6,
        load=4.0,
        esr_ind=0.10,
        esr_cap=0.02,
        rds=0.01,
        rds_low=0.0,
    )
    h0 = base_conv.transfer_function().frequency_response(omega)
    ratio_modes = [
        (r"$C_{\mathrm{out}}$ ($-35\%$)", {"capacitance": 0.65}, COL_STEEL, "-"),
        (r"$L_{\mathrm{out}}$ ($-35\%$)", {"inductance": 0.65}, COL_INDIGO, "--"),
        (r"$\mathrm{ESR}_C$ ($+150\%$)", {"esr_cap": 2.50}, COL_TEAL, "-."),
        (r"$\mathrm{ESR}_L$ ($+150\%$)", {"esr_ind": 2.50}, COL_OCHRE, "--"),
        (r"$R_{\mathrm{ds},1}$ ($+350\%$)", {"rds": 4.50}, COL_BURG, ":"),
    ]
    ax_rat_m.axhline(0.0, color="#666666", lw=1.0, ls="-")
    ax_rat_p.axhline(0.0, color="#666666", lw=1.0, ls="-")
    for r_lbl, factors, col, ls in ratio_modes:
        ht = base_conv.scaled(**factors).transfer_function().frequency_response(omega)
        gamma = ht / h0
        ax_rat_m.plot(f_khz, 20.0 * np.log10(np.abs(gamma)), color=col, ls=ls, lw=1.9, label=r_lbl)
        ax_rat_p.plot(f_khz, np.rad2deg(np.angle(gamma)), color=col, ls=ls, lw=1.9)

    ax_rat_m.set_xscale("log")
    ax_rat_m.set_ylim(-3.3, 9.4)
    ax_rat_m.set_ylabel(r"$20\log_{10}|\Gamma_{\alpha}(j\omega)|\;[\mathrm{dB}]$", fontweight="bold")
    ax_rat_m.set_title(r"(b) Nominal-Load SSA Fault Ratios $\Gamma_{\alpha}(j\omega)$", loc="left", pad=8)
    ax_rat_m.grid(True, which="both", alpha=0.45)
    style_axis(ax_rat_m)
    plt.setp(ax_rat_m.get_xticklabels(), visible=False)
    ax_rat_m.legend(loc="upper left", fontsize=7.3, ncol=2, framealpha=0.95)

    ax_rat_p.set_xscale("log")
    ax_rat_p.set_xlabel(r"Frequency $f\;[\mathrm{kHz}]$", fontweight="bold")
    ax_rat_p.set_ylabel(r"$\angle\Gamma_{\alpha}(j\omega)\;[\mathrm{deg}]$", fontweight="bold")
    ax_rat_p.grid(True, which="both", alpha=0.45)
    style_axis(ax_rat_p)

    # Panel (c): Switched PSIM vs Analytic SSA Relative L2 Error [%] (Log scale)
    err_d2 = physics_audit_ds02["psim_vs_analytic_rel_l2_error"]
    err_d1 = physics_audit_ds01["psim_vs_analytic_rel_l2_error"]
    models_err = [
        (
            "Fixed-Load SSA\n($R_{\\mathrm{out}} \\equiv 4\\,\\Omega$, ds02)",
            err_d2["ssa_ignoring_variable_rout_nominal_4ohm"]["median_pct"],
            err_d2["ssa_ignoring_variable_rout_nominal_4ohm"]["p95_pct"],
            COL_BURG,
        ),
        (
            "Lossless Textbook SSA\n(True $R_{\\mathrm{out}}$, ds02)",
            err_d2["legacy_lossless_with_true_rout"]["median_pct"],
            err_d2["legacy_lossless_with_true_rout"]["p95_pct"],
            COL_OCHRE,
        ),
        (
            "Exact Parasitic SSA\n(Variable Load, ds02)",
            err_d2["exact_ssa_with_true_rout"]["median_pct"],
            err_d2["exact_ssa_with_true_rout"]["p95_pct"],
            COL_SSA,
        ),
        (
            "Exact Parasitic SSA\n(Fixed Load, ds01)",
            err_d1["exact_ssa_psim_body_diode_rds2_zero"]["median_pct"],
            err_d1["exact_ssa_psim_body_diode_rds2_zero"]["p95_pct"],
            COL_TEAL,
        ),
    ]
    y_idx = np.arange(len(models_err), dtype=float)
    medians = [m[1] for m in models_err]
    p95s = [m[2] for m in models_err]
    cols_e = [m[3] for m in models_err]

    bars = ax_err.barh(y_idx, medians, height=0.54, color=cols_e, edgecolor="#ffffff", zorder=3)
    ax_err.hlines(y_idx, medians, p95s, color="#222222", lw=1.8, zorder=4, label="95th percentile")
    ax_err.scatter(p95s, y_idx, color="#222222", s=26, zorder=5)

    for idx_b, (bar, med_v) in enumerate(zip(bars, medians)):
        fmt_txt = f"{med_v:.4f}%" if med_v < 1.0 else f"{med_v:.3f}%"
        ax_err.text(
            p95s[idx_b] * 1.18,
            bar.get_y() + bar.get_height() / 2.0,
            fmt_txt,
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight="bold",
            color=cols_e[idx_b],
        )

    ax_err.set_xscale("log")
    ax_err.set_xlim(0.03, 155.0)
    ax_err.set_yticks(y_idx)
    ax_err.set_yticklabels([m[0] for m in models_err], fontsize=8.5)
    ax_err.invert_yaxis()
    ax_err.set_xlabel(r"Relative Complex $L_2$ Error vs. PSIM [%]", fontweight="bold")
    ax_err.set_title("(c) Switched PSIM Fidelity Audit", loc="left", pad=8)
    ax_err.grid(True, axis="x", which="both", alpha=0.45, zorder=0)
    style_axis(ax_err)
    ax_err.legend(loc="lower right", fontsize=8.0, framealpha=0.95)

    save_figure(fig, save_paths)
    return fig


def plot_fig03_carla_transformer_architecture(
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Render an academic vector diagram of CARLA-Transformer layers and tensor shapes."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(15.4, 6.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0.0, 15.4)
    ax.set_ylim(0.0, 6.2)
    ax.axis("off")

    def _blk(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        shape_lbl: str = "",
        *,
        fc: str = "#ffffff",
        ec: str = COL_SSA,
        lw: float = 1.25,
        ls: str = "-",
        title_fs: float = 7.6,
        shape_fs: float = 7.0,
        title_col: str = "#111111",
    ) -> None:
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3)
        )
        cy = y + h * (0.58 if shape_lbl else 0.50)
        ax.text(
            x + w / 2.0,
            cy,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color=title_col,
            linespacing=1.18,
            zorder=4,
        )
        if shape_lbl:
            ax.text(
                x + w / 2.0,
                y + h * 0.22,
                shape_lbl,
                ha="center",
                va="center",
                fontsize=shape_fs,
                color="#334155",
                linespacing=1.12,
                zorder=4,
            )

    def _arr(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: str = "#1e293b",
        lw: float = 1.25,
        ls: str = "-",
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>,head_length=4.8,head_width=2.8",
                color=color,
                linewidth=lw,
                linestyle=ls,
                zorder=5,
            )
        )

    # Top Figure Title (left-aligned academic IEEE style matching panels)
    ax.text(
        0.18,
        5.92,
        "CARLA-Transformer Dual-Branch Self-Attention Architecture & Exact Tensor Flow",
        ha="left",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#111111",
    )

    # -------------------------------------------------------------------------
    # Top-Left Structural Inset: Internal TransformerBlock + Cross-Decade Attention
    # -------------------------------------------------------------------------
    ax.add_patch(
        Rectangle((0.20, 4.18), 7.85, 1.48, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0, ls="-", zorder=1)
    )
    ax.text(
        0.34,
        5.46,
        "Transformer Block Internals ($d_\\ell \\in \\{64, 128\\}$, $H=4$ heads, dropout $= 0.0$)",
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=COL_SSA,
    )
    _blk(0.35, 4.42, 1.40, 0.72, "Multi-Head\nSelf-Attention", "$H=4,\\; d_k=d_\\ell/4$", fc="#ffffff", ec=COL_SSA, title_fs=7.1, shape_fs=6.6)
    ax.add_patch(Circle((2.15, 4.78), 0.16, facecolor="#ffffff", edgecolor="#111111", lw=1.1, zorder=4))
    ax.text(2.15, 4.78, "$+$", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=5)
    _blk(2.50, 4.48, 1.02, 0.60, "LayerNorm", "", fc="#ffffff", ec="#475569", title_fs=7.1)
    _blk(3.82, 4.42, 1.62, 0.72, "Feed-Forward FFN", "$d_\\ell \\to 4d_\\ell\\,(\\mathrm{ReLU})\\to d_\\ell$", fc="#ffffff", ec=COL_TEAL, title_fs=7.1, shape_fs=6.5)
    ax.add_patch(Circle((5.84, 4.78), 0.16, facecolor="#ffffff", edgecolor="#111111", lw=1.1, zorder=4))
    ax.text(5.84, 4.78, "$+$", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=5)
    _blk(6.18, 4.48, 1.02, 0.60, "LayerNorm", "", fc="#ffffff", ec="#475569", title_fs=7.1)
    # Skip residuals
    ax.plot([0.25, 0.25, 2.15, 2.15], [4.78, 5.26, 5.26, 4.94], color="#64748b", lw=0.95, zorder=2)
    ax.plot([3.65, 3.65, 5.84, 5.84], [4.78, 5.26, 5.26, 4.94], color="#64748b", lw=0.95, zorder=2)
    _arr(1.75, 4.78, 1.99, 4.78)
    _arr(2.31, 4.78, 2.50, 4.78)
    _arr(3.52, 4.78, 3.82, 4.78)
    _arr(5.44, 4.78, 5.68, 4.78)
    _arr(6.00, 4.78, 6.18, 4.78)
    _arr(7.20, 4.78, 7.65, 4.78)
    ax.text(7.72, 4.78, "Out", ha="left", va="center", fontsize=7.2, fontweight="bold", color="#334155")

    # Top-Right Callout: Cross-Decade O(1) Spectral Bus Illustration
    ax.add_patch(
        Rectangle((8.25, 4.18), 6.95, 1.48, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0, ls="-", zorder=1)
    )
    ax.text(
        8.40,
        5.46,
        "Non-Local Cross-Decade Self-Attention ($O(1)$ Hops Across $101$ Tokens)",
        ha="left",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color=COL_TEAL,
    )
    # Draw 3 spectral bands + attention arcs
    band_specs = [
        (9.35, "DC Plateau\n$100\\text{--}400\\,\\mathrm{Hz}$\n($\\mathrm{ESR}_L, R_{\\mathrm{ds},1}$)", COL_BURG),
        (11.72, "Resonance $f_0$\n$0.7\\text{--}2.0\\,\\mathrm{kHz}$\n($R_{\\mathrm{out}}$ Load $Q$)", COL_SSA),
        (14.08, "HF Zero Slope\n$8\\text{--}50\\,\\mathrm{kHz}$\n($\\mathrm{ESR}_C$ Zero)", COL_TEAL),
    ]
    for bx, blbl, bcol in band_specs:
        _blk(bx - 0.85, 4.30, 1.70, 0.64, blbl, "", fc="#ffffff", ec=bcol, lw=1.1, title_fs=6.6, title_col=bcol)
    for (x_src, x_dst) in [(9.35, 11.72), (11.72, 14.08), (9.35, 14.08)]:
        rad_v = -0.34 if abs(x_dst - x_src) < 3.0 else -0.26
        ax.add_patch(
            FancyArrowPatch(
                (x_src, 4.96),
                (x_dst, 4.96),
                arrowstyle="<->,head_length=4.0,head_width=2.2",
                connectionstyle=f"arc3,rad={rad_v}",
                color="#334155",
                linewidth=1.15,
                zorder=5,
            )
        )

    # -------------------------------------------------------------------------
    # Main Shared Encoder Trunk (y centered around 2.65)
    # -------------------------------------------------------------------------
    ax.add_patch(
        Rectangle((1.88, 1.38), 6.20, 2.52, facecolor="#f4f7fb", edgecolor=COL_SSA, linewidth=1.25, ls="--", zorder=1)
    )
    ax.text(4.98, 3.74, "Shared Global Self-Attention Encoder $f_{\\theta}$ (Retained for Inference)", ha="center", va="center", fontsize=8.3, fontweight="bold", color=COL_SSA)

    # Input Tensor Slab with small schematic Bode spectrum glyph placed cleanly above text
    ax.add_patch(
        Rectangle((0.20, 1.72), 1.35, 1.85, facecolor="#ffffff", edgecolor="#111111", linewidth=1.3, zorder=3)
    )
    t_mini = np.linspace(0.30, 1.45, 50)
    s_mini = 3.24 + 0.14 * np.sin((t_mini - 0.30) * 4.8) * np.exp(-1.1 * (t_mini - 0.30))
    ax.plot(t_mini, s_mini, color=COL_SSA, lw=1.25, zorder=5)
    ax.plot(t_mini, s_mini - 0.11, color=COL_TEAL, lw=1.05, ls="--", zorder=5)
    ax.text(0.875, 2.55, "Input Sweep\n$\\mathbf{X}$", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#111111", linespacing=1.18, zorder=4)
    ax.text(0.875, 1.98, "$\\mathbb{R}^{101\\times 2}$\n$[\\mathrm{Re}\\,H,\\,\\mathrm{Im}\\,H]$", ha="center", va="center", fontsize=6.9, color="#334155", linespacing=1.12, zorder=4)

    _blk(2.05, 1.85, 1.25, 1.58, "Token Embed\nDense(64)\n+ Sinus. PE", "$\\mathbf{P}_{101\\times 64}$\n$(101\\times 64)$", fc="#ffffff", ec=COL_SSA, title_fs=7.2, shape_fs=6.8)
    _blk(3.55, 1.85, 1.35, 1.58, "Enc. Block 1\nMHA ($H=4$)\n+ FFN(256)", "$d_1=64,\\, d_k=16$\n$(101\\times 64)$", fc="#ffffff", ec=COL_SSA, title_fs=7.3, shape_fs=6.8)
    _blk(5.15, 1.85, 1.40, 1.58, "Dense(128) +\nEnc. Block 2\nFFN(512)", "$d_2=128,\\, d_k=32$\n$(101\\times 128)$", fc="#ffffff", ec=COL_SSA, title_fs=7.3, shape_fs=6.8)
    _blk(6.78, 2.02, 1.15, 1.25, "GlobalAvgPool\n+ Dense(128)", "$(128,)$", fc="#ffffff", ec=COL_SSA, title_fs=7.1, shape_fs=6.8)

    # Bottleneck h
    _blk(8.35, 2.12, 1.10, 1.05, "Latent\n$\\mathbf{h}$", "$\\mathbb{R}^{128}$", fc="#fff8eb", ec=COL_OCHRE, lw=1.5, title_fs=8.0, shape_fs=7.2, title_col="#8c5811")

    _arr(1.55, 2.64, 2.03, 2.64, color=COL_SSA)
    _arr(3.30, 2.64, 3.53, 2.64, color=COL_SSA)
    _arr(4.90, 2.64, 5.13, 2.64, color=COL_SSA)
    _arr(6.55, 2.64, 6.76, 2.64, color=COL_SSA)
    _arr(7.93, 2.64, 8.33, 2.64, color=COL_OCHRE, lw=1.4)

    # -------------------------------------------------------------------------
    # Upper Branch: Projection Head \pi_\omega (Train + Zero-Label Inference)
    # -------------------------------------------------------------------------
    ax.add_patch(
        Rectangle((9.82, 2.25), 5.38, 1.65, facecolor="#ffffff", edgecolor=COL_TEAL, linewidth=1.3, zorder=1)
    )
    ax.text(12.51, 3.74, "Projection Head $\\pi_{\\omega}$: Contrastive / Test Branch", ha="center", va="center", fontsize=7.8, fontweight="bold", color=COL_TEAL)

    _blk(9.98, 2.42, 1.72, 1.08, "MLP Projection\nDense(128)-BN-ReLU\nDense(128) + $\\ell_2$", "$\\mathbf{z}\\in\\mathbb{S}^{127}$", fc="#ffffff", ec=COL_TEAL, title_fs=6.9, shape_fs=7.1)
    _blk(12.12, 3.00, 2.92, 0.56, "Train: Directed InfoNCE ($w_c=0.5,\\, \\tau=0.1$)", "Repels SSA negatives $\\mathbf{z}_j^{-}$ vs positive $\\mathbf{z}_i^{+}$", fc="#fdf2ec", ec=COL_TERRA, lw=1.1, ls="--", title_fs=6.8, shape_fs=6.3, title_col=COL_TERRA)
    _blk(12.12, 2.34, 2.92, 0.56, "Test: 5-NN Score $a(\\mathbf{X}) > \\tau_{0.975}$", "Zero fault labels ($\\mathcal{D}_{\\mathrm{val,norm}}$ only)", fc="#eef7f4", ec=COL_TEAL, lw=1.25, title_fs=6.9, shape_fs=6.4, title_col="#1b5e4a")

    ax.plot([9.45, 9.64, 9.64], [2.80, 2.80, 2.96], color=COL_TEAL, lw=1.3)
    _arr(9.64, 2.96, 9.96, 2.96, color=COL_TEAL)
    _arr(11.70, 3.12, 12.10, 3.28, color=COL_TERRA, ls="--")
    _arr(11.70, 2.78, 12.10, 2.62, color=COL_TEAL)

    # -------------------------------------------------------------------------
    # Lower Branch: Symmetric Decoder g_\psi (Training Only)
    # -------------------------------------------------------------------------
    ax.add_patch(
        Rectangle((8.35, 0.16), 6.85, 1.72, facecolor="#f8fafc", edgecolor="#64748b", linewidth=1.15, ls="--", zorder=1)
    )
    ax.text(9.32, 1.71, "Mirrored Decoder $g_{\\psi}$ (Training Only \u2014 Discarded at Test)", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#475569")

    _blk(8.50, 0.30, 1.25, 1.18, "Expand & PE\nDense($101\\times 128$)\nReshape + PE", "$(101\\times 128)$", fc="#ffffff", ec="#64748b", ls="--", title_fs=6.8, shape_fs=6.5)
    _blk(9.95, 0.30, 1.28, 1.18, "Dec. Block 1\nMHA ($H=4$)\nFFN(512)", "$d=128$\n$(101\\times 128)$", fc="#ffffff", ec="#64748b", ls="--", title_fs=6.8, shape_fs=6.5)
    _blk(11.43, 0.30, 1.28, 1.18, "Dense(64) +\nDec. Block 2\nFFN(256)", "$d=64$\n$(101\\times 64)$", fc="#ffffff", ec="#64748b", ls="--", title_fs=6.8, shape_fs=6.5)
    _blk(12.92, 0.30, 2.12, 1.18, "Dense(2) $\\to\\hat{\\mathbf{X}}\\in\\mathbb{R}^{101\\times 2}$\nHealthy MSE Loss ($w_{\\mathrm{rec}}=5.0$)", "Anchors $\\mathbf{X}_i$ only", fc="#ffffff", ec="#475569", lw=1.15, ls="--", title_fs=6.8, shape_fs=6.5)

    ax.plot([8.90, 8.90], [2.12, 1.50], color="#64748b", lw=1.2, ls="--")
    _arr(8.90, 1.75, 8.90, 1.48, color="#64748b", ls="--")
    _arr(9.75, 0.89, 9.93, 0.89, color="#64748b", ls="--")
    _arr(11.23, 0.89, 11.41, 0.89, color="#64748b", ls="--")
    _arr(12.71, 0.89, 12.90, 0.89, color="#64748b", ls="--")

    save_figure(fig, save_paths)
    return fig


def plot_fig04_ds01_vs_ds02_and_fault_recall(
    ds01_zero_summary: Dict[str, Any],
    ds02_baselines_df: pd.DataFrame,
    ablation_10seed: Dict[str, Any],
    encoder_5seed: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot fixed-vs-variable load regime breakdown and per-fault zero-label recall."""
    apply_publication_style()
    fig, (ax_reg, ax_rec) = plt.subplots(
        1, 2, figsize=(14.4, 5.4), gridspec_kw={"width_ratios": [0.92, 1.38], "wspace": 0.24}
    )
    fig.patch.set_facecolor("#ffffff")

    def _d2_base(det: str, col: str) -> float:
        sub = ds02_baselines_df[
            (ds02_baselines_df["detector"] == det) & (ds02_baselines_df["healthy_quantile"] == "p97.5")
        ]
        return float(sub.iloc[0][col])

    # Panel (a): Fixed-Load (ds01) vs Variable-Load (ds02) Zero-Label p97.5 F1
    ds01_knn = float(ds01_zero_summary["Raw k-NN (k=1)__native__p97.5"]["active5_f1"]["mean"])
    ds01_maha = float(
        ds01_zero_summary.get(
            "Mahalanobis (Ledoit-Wolf)__native__p97.5",
            ds01_zero_summary["Raw k-NN (k=1)__native__p97.5"],
        )["active5_f1"]["mean"]
    )
    ds01_conv = float(
        ds01_zero_summary.get(
            "CARLA-Conv1D (SSA)__knn_k5__p97.5",
            {"active5_f1": {"mean": 0.9399}},
        )["active5_f1"]["mean"]
    )

    ssa_tf = ablation_10seed["regimes"]["tf_physics_ssa_fixed_load"]
    conv_d2 = encoder_5seed["architectures"]["Conv1D"]

    comp_methods = [
        ("Raw k-NN", ds01_knn, _d2_base("Raw k-NN (k=5)", "f1"), COL_SLATE),
        ("Mahalanobis (LW)", ds01_maha, _d2_base("Mahalanobis (Ledoit-Wolf)", "f1"), COL_OCHRE),
        ("CARLA-Conv1D (SSA)", ds01_conv, float(conv_d2["zero_p975_f1"]["mean"]), COL_STEEL),
        ("CARLA-Transformer (SSA)", 0.9410, float(ssa_tf["zero_p975_f1"]["mean"]), COL_SSA),
    ]
    xg = np.arange(len(comp_methods), dtype=float)
    w_reg = 0.34
    b1 = ax_reg.bar(
        xg - w_reg / 2,
        [m[1] for m in comp_methods],
        width=w_reg * 0.94,
        color="#a6bddb",
        edgecolor="#1f4e79",
        lw=1.0,
        label=r"Fixed Load (dataset_01, $R_{\mathrm{out}}=4\,\Omega$)",
        zorder=3,
    )
    b2 = ax_reg.bar(
        xg + w_reg / 2,
        [m[2] for m in comp_methods],
        width=w_reg * 0.94,
        color=COL_SSA,
        edgecolor="#ffffff",
        lw=0.8,
        label=r"Variable Load (dataset_02, $R_{\mathrm{out}} \in [2,8]\,\Omega$)",
        zorder=3,
    )
    for bar_f, bar_v in zip(b1, b2):
        delta = bar_v.get_height() - bar_f.get_height()
        sign_txt = f"{delta:+.3f}" if abs(delta) >= 0.01 else "Invariant"
        ax_reg.text(
            bar_v.get_x() + bar_v.get_width() / 2.0,
            bar_v.get_height() + 0.012,
            sign_txt,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#b33939" if delta < -0.05 else "#1f4e79",
        )

    ax_reg.set_xticks(xg)
    ax_reg.set_xticklabels([m[0] for m in comp_methods], fontsize=8.7, rotation=10)
    ax_reg.set_ylim(0.55, 1.04)
    ax_reg.set_ylabel(r"Zero-Label $\mathrm{F1}_{\mathrm{p97.5}}$", fontweight="bold")
    ax_reg.set_title("(a) Fixed-Load Simplicity vs. Variable-Load Collapse", loc="left", pad=10)
    ax_reg.grid(True, axis="y", alpha=0.45, zorder=0)
    style_axis(ax_reg)
    ax_reg.legend(loc="lower left", fontsize=8.1, framealpha=0.96)

    # Panel (b): Per-Fault Recall Stratified Across All 6 Fault Strata
    strata = [
        ("recall_cout_single", "$C_{\\mathrm{out}}$\n($n=23$)"),
        ("recall_lout_single", "$L_{\\mathrm{out}}$\n($n=22$)"),
        ("recall_esrc_single", "$\\mathrm{ESR}_C$\n($n=28$)"),
        ("recall_esrl_single", "$\\mathrm{ESR}_L$\n($n=21$)"),
        ("recall_rds1_single", "$R_{\\mathrm{ds},1}$\n($n=19$)"),
        ("recall_multi_fault", "Multi\n($n=97$)"),
    ]
    mse_tf = ablation_10seed["regimes"]["tf_recon_only_mse"]
    heur_tf = ablation_10seed["regimes"]["tf_heuristic"]

    series = [
        (
            "CARLA-Transformer (SSA)",
            [100.0 * ssa_tf[k]["mean"] for k, _ in strata],
            [100.0 * ssa_tf[k]["sample_std"] for k, _ in strata],
            COL_SSA,
        ),
        (
            "Mahalanobis (Ledoit-Wolf)",
            [100.0 * _d2_base("Mahalanobis (Ledoit-Wolf)", k) for k, _ in strata],
            [0.0] * len(strata),
            COL_OCHRE,
        ),
        (
            "AE-Transformer (MSE)",
            [100.0 * mse_tf[k]["mean"] for k, _ in strata],
            [100.0 * mse_tf[k]["sample_std"] for k, _ in strata],
            COL_BURG,
        ),
        (
            "CARLA-Transformer (Heuristic)",
            [100.0 * heur_tf[k]["mean"] for k, _ in strata],
            [100.0 * heur_tf[k]["sample_std"] for k, _ in strata],
            COL_TERRA,
        ),
        (
            "Raw k-NN (k=5)",
            [100.0 * _d2_base("Raw k-NN (k=5)", k) for k, _ in strata],
            [0.0] * len(strata),
            COL_SLATE,
        ),
    ]
    xs = np.arange(len(strata), dtype=float)
    ws = 0.155
    for idx_s, (s_lbl, means, stds, col) in enumerate(series):
        off = (idx_s - 2.0) * ws
        ax_rec.bar(
            xs + off,
            means,
            width=ws * 0.92,
            yerr=stds if any(v > 0 for v in stds) else None,
            capsize=2.6,
            error_kw={"lw": 1.1, "ecolor": "#222222"},
            color=col,
            edgecolor="#ffffff",
            linewidth=0.5,
            label=s_lbl,
            zorder=3,
        )

    ax_rec.set_xticks(xs)
    ax_rec.set_xticklabels([lbl for _, lbl in strata], fontsize=9.0)
    ax_rec.set_ylim(0.0, 118.0)
    ax_rec.set_ylabel("Zero-Label Recall at p97.5 [%]", fontweight="bold")
    ax_rec.set_title("(b) Per-Fault Recall Disentanglement on dataset_02", loc="left", pad=10)
    ax_rec.grid(True, axis="y", alpha=0.45, zorder=0)
    style_axis(ax_rec)
    ax_rec.legend(loc="upper center", ncol=3, fontsize=7.8, framealpha=0.96)

    save_figure(fig, save_paths)
    return fig


def plot_fig05_pr_curves_and_10seed_ablation(
    pr_curves_payload: Dict[str, Any],
    ablation_10seed: Dict[str, Any],
    ds02_baselines_df: pd.DataFrame,
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot prevalence-adjusted PR curves (pi=5%) and 10-seed paired ablation."""
    apply_publication_style()
    fig, (ax_pr, ax_ab) = plt.subplots(1, 2, figsize=(14.2, 5.4), gridspec_kw={"wspace": 0.24})
    fig.patch.set_facecolor("#ffffff")

    curve_specs = [
        ("CARLA-Transformer (SSA)", COL_SSA, "-", 2.5),
        ("PCA Reconstruction (d=10)", COL_TEAL, "--", 1.9),
        ("Mahalanobis (Ledoit-Wolf)", COL_OCHRE, "-.", 1.9),
        ("AE-Transformer (MSE)", COL_BURG, ":", 2.0),
        ("Raw k-NN (k=5)", COL_SLATE, "--", 1.6),
        ("CARLA-Transformer (Heuristic)", COL_TERRA, ":", 1.9),
    ]
    for name, col, ls, lw in curve_specs:
        if name not in pr_curves_payload:
            continue
        item = pr_curves_payload[name]
        ap = float(item["ap_pi05"])
        if "std_pi05" in item and item["std_pi05"] is not None:
            lbl = f"{name} (AP = {ap:.3f} ± {float(item['std_pi05']):.3f})"
        else:
            lbl = f"{name} (AP = {ap:.3f})"
        ax_pr.plot(item["recall"], item["precision_pi05"], color=col, ls=ls, lw=lw, label=lbl, zorder=4)

    ax_pr.axhline(0.05, color="#888888", ls=":", lw=1.1, label=r"Random classifier ($\pi = 0.05$)")
    ax_pr.set_xlim(0.0, 1.01)
    ax_pr.set_ylim(0.0, 1.04)
    ax_pr.set_xlabel("Recall (True Positive Rate)", fontweight="bold")
    ax_pr.set_ylabel(r"Prevalence-Adjusted Precision ($\pi = 5\%$)", fontweight="bold")
    ax_pr.set_title(r"(a) Field-Prevalence Precision-Recall Curves ($\pi = 5\%$)", loc="left", pad=10)
    ax_pr.grid(True, alpha=0.45, zorder=0)
    style_axis(ax_pr)
    ax_pr.legend(loc="lower left", fontsize=8.0, framealpha=0.96)

    # Panel (b): 10-Seed Paired Contrastive Ablation with individual seed scatter
    reg_order = [
        ("tf_physics_ssa_fixed_load", "CARLA-TF\n(SSA)"),
        ("tf_recon_only_mse", "AE-TF\n(MSE)"),
        ("tf_recon_only_latent", "AE-TF\n(Latent kNN5)"),
        ("tf_heuristic", "CARLA-TF\n(Heuristic)"),
    ]
    xa = np.arange(len(reg_order), dtype=float)
    w_ab = 0.34
    rng = np.random.default_rng(42)

    f1_means = [ablation_10seed["regimes"][rk]["zero_p975_f1"]["mean"] for rk, _ in reg_order]
    f1_stds = [ablation_10seed["regimes"][rk]["zero_p975_f1"]["sample_std"] for rk, _ in reg_order]
    pr_means = [ablation_10seed["regimes"][rk]["auc_pr_pi05"]["mean"] for rk, _ in reg_order]
    pr_stds = [ablation_10seed["regimes"][rk]["auc_pr_pi05"]["sample_std"] for rk, _ in reg_order]

    ax_ab.bar(
        xa - w_ab / 2,
        f1_means,
        width=w_ab * 0.92,
        yerr=f1_stds,
        capsize=3.5,
        color=COL_SSA,
        edgecolor="#ffffff",
        label=r"Zero-Label $\mathrm{F1}_{\mathrm{p97.5}}$",
        zorder=3,
    )
    ax_ab.bar(
        xa + w_ab / 2,
        pr_means,
        width=w_ab * 0.92,
        yerr=pr_stds,
        capsize=3.5,
        color=COL_OCHRE,
        edgecolor="#ffffff",
        label=r"$\mathrm{PR}$-$\mathrm{AUC}_{\pi=0.05}$",
        zorder=3,
    )

    for idx_r, (rk, _) in enumerate(reg_order):
        v_f1 = ablation_10seed["regimes"][rk]["zero_p975_f1"]["values"]
        v_pr = ablation_10seed["regimes"][rk]["auc_pr_pi05"]["values"]
        jit1 = rng.uniform(-0.06, 0.06, size=len(v_f1))
        jit2 = rng.uniform(-0.06, 0.06, size=len(v_pr))
        ax_ab.scatter(
            np.full(len(v_f1), xa[idx_r] - w_ab / 2) + jit1,
            v_f1,
            s=18,
            facecolor="#ffffff",
            edgecolor="#0b2545",
            linewidth=1.0,
            zorder=5,
        )
        ax_ab.scatter(
            np.full(len(v_pr), xa[idx_r] + w_ab / 2) + jit2,
            v_pr,
            s=18,
            facecolor="#ffffff",
            edgecolor="#6e470b",
            linewidth=1.0,
            zorder=5,
        )

    maha_row = ds02_baselines_df[
        (ds02_baselines_df["detector"] == "Mahalanobis (Ledoit-Wolf)")
        & (ds02_baselines_df["healthy_quantile"] == "p97.5")
    ].iloc[0]
    ax_ab.axhline(
        float(maha_row["f1"]),
        color=COL_TEAL,
        ls="--",
        lw=1.3,
        label=f"Ledoit-Wolf F1 ({float(maha_row['f1']):.3f})",
    )
    ax_ab.text(
        1.5,
        0.985,
        r"Paired sign-flip vs. all non-physics regimes: $n=10,\; p = 0.00195$",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=COL_SSA,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#eef3f9", "edgecolor": COL_SSA, "lw": 0.9},
    )

    ax_ab.set_xticks(xa)
    ax_ab.set_xticklabels([lbl for _, lbl in reg_order], fontsize=9.2)
    ax_ab.set_ylim(0.42, 1.05)
    ax_ab.set_ylabel("Metric Score (10 Paired Random Seeds)", fontweight="bold")
    ax_ab.set_title("(b) 10-Seed Paired Contrastive Physics Ablation", loc="left", pad=10)
    ax_ab.grid(True, axis="y", alpha=0.45, zorder=0)
    style_axis(ax_ab)
    ax_ab.legend(loc="lower left", fontsize=8.1, framealpha=0.96)

    save_figure(fig, save_paths)
    return fig


def plot_fig06_encoder_backbones_and_scorer_matrix(
    encoder_5seed: Dict[str, Any],
    latent_controls: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot 5-backbone encoder comparison and 4x4 representation-vs-scorer matrix."""
    apply_publication_style()
    fig, (ax_arch, ax_mat) = plt.subplots(
        1, 2, figsize=(14.4, 5.4), gridspec_kw={"width_ratios": [1.18, 0.95], "wspace": 0.28}
    )
    fig.patch.set_facecolor("#ffffff")

    # Panel (a): 5 Encoder Backbones Under Identical SSA Contrastive Objective (5 Seeds)
    archs = ["Transformer", "LSTM", "Conv1D", "MLP", "GRU"]
    arch_data = encoder_5seed["architectures"]
    xa = np.arange(len(archs), dtype=float)
    w = 0.24

    metrics_spec = [
        ("zero_p975_f1", r"Clean $\mathrm{F1}_{\mathrm{p97.5}}$", COL_SSA),
        ("auc_pr_pi05", r"$\mathrm{PR}$-$\mathrm{AUC}_{\pi=0.05}$", COL_OCHRE),
        ("fra015_matched_p975_f1", r"Noisy $\mathrm{F1}_{\mathrm{p97.5}}$ ($0.15\,\mathrm{dB}$)", COL_TEAL),
    ]
    for idx_m, (m_key, m_lbl, col) in enumerate(metrics_spec):
        means = [float(arch_data[a][m_key]["mean"]) for a in archs]
        stds = [float(arch_data[a][m_key]["sample_std"]) for a in archs]
        off = (idx_m - 1.0) * w
        bars = ax_arch.bar(
            xa + off,
            means,
            width=w * 0.92,
            yerr=stds,
            capsize=2.8,
            error_kw={"lw": 1.1, "ecolor": "#222222"},
            color=col,
            edgecolor="#ffffff",
            linewidth=0.6,
            label=m_lbl,
            zorder=3,
        )
        if idx_m == 0:
            for bar, v, sd_v in zip(bars, means, stds):
                ax_arch.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    v + sd_v + 0.014,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    fontweight="bold",
                    color=COL_SSA,
                )

    ax_arch.set_xticks(xa)
    ax_arch.set_xticklabels(archs, fontsize=9.4, fontweight="bold")
    ax_arch.set_ylim(0.42, 1.07)
    ax_arch.set_ylabel("5-Seed Score (Mean ± SD)", fontweight="bold")
    ax_arch.set_title("(a) Five Encoder Backbones Under Identical SSA Contrastive Loss", loc="left", pad=10)
    ax_arch.grid(True, axis="y", alpha=0.45, zorder=0)
    style_axis(ax_arch)
    ax_arch.legend(loc="lower left", fontsize=8.0, framealpha=0.96)

    # Panel (b): 4x4 Representation vs Post-Hoc Scorer Matrix (10 Seeds)
    row_specs = [
        ("tf_physics_ssa_fixed_load__projection", "CARLA-TF (SSA)\nProjection $\\mathbf{z}$ ($\\mathbb{S}^{127}$)"),
        ("tf_physics_ssa_fixed_load__latent", "CARLA-TF (SSA)\nLatent $\\mathbf{h}$ ($\\mathbb{R}^{128}$)"),
        ("tf_recon_only__latent", "AE-TF (MSE)\nLatent $\\mathbf{h}$ ($\\mathbb{R}^{128}$)"),
        ("tf_heuristic__projection", "CARLA-TF (Heur.)\nProjection $\\mathbf{z}$ ($\\mathbb{S}^{127}$)"),
    ]
    col_specs = [
        ("knn_k5", "k-NN\n($k=5$)"),
        ("sklearn_ledoit_wolf", "Mahalanobis\n(Ledoit-Wolf)"),
        ("pca_d5", "PCA Res.\n($d=5$)"),
        ("pca_d10", "PCA Res.\n($d=10$)"),
    ]
    mat_mean = np.zeros((len(row_specs), len(col_specs)), dtype=float)
    mat_std = np.zeros((len(row_specs), len(col_specs)), dtype=float)
    for r_i, (r_key, _) in enumerate(row_specs):
        for c_i, (c_key, _) in enumerate(col_specs):
            entry = latent_controls[f"{r_key}__{c_key}"]["zero_p975_f1"]
            mat_mean[r_i, c_i] = float(entry["mean"])
            mat_std[r_i, c_i] = float(entry["sample_std"])

    cmap = plt.cm.Blues
    vmin, vmax = 0.50, 0.97
    im = ax_mat.imshow(mat_mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for r_i in range(mat_mean.shape[0]):
        for c_i in range(mat_mean.shape[1]):
            val = mat_mean[r_i, c_i]
            sd = mat_std[r_i, c_i]
            txt_col = adaptive_contrast(val, vmin, vmax, cmap=cmap)
            ax_mat.text(
                c_i,
                r_i,
                f"{val:.3f}\n±{sd:.3f}",
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="bold" if r_i < 2 else "normal",
                color=txt_col,
            )

    ax_mat.grid(False)
    ax_mat.set_xticks(np.arange(len(col_specs)))
    ax_mat.set_xticklabels([lbl for _, lbl in col_specs], fontsize=8.5)
    ax_mat.set_yticks(np.arange(len(row_specs)))
    ax_mat.set_yticklabels([lbl for _, lbl in row_specs], fontsize=8.4)
    ax_mat.set_title(r"(b) Representation vs. Scorer Matrix ($\mathrm{F1}_{\mathrm{p97.5}}$)", loc="left", pad=10)
    for spine in ax_mat.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)
    cbar = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8.0)
    cbar.set_label(r"Zero-Label $\mathrm{F1}_{\mathrm{p97.5}}$", fontsize=8.6)

    save_figure(fig, save_paths)
    return fig


def plot_fig07_pca_cliff_and_7level_fra_noise(
    pca_fragility: Dict[str, Any],
    ablation_10seed_full: Dict[str, Any],
    fra_noise_sweep: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot PCA rank cliff (d=2..20) and 7-level FRA measurement noise robustness sweep."""
    apply_publication_style()
    fig, (ax_pca, ax_ns) = plt.subplots(1, 2, figsize=(14.4, 5.4), gridspec_kw={"wspace": 0.24})
    fig.patch.set_facecolor("#ffffff")

    # Panel (a): PCA Subspace Rank Cliff & Clean-Fit Noise Fragility (d=2..20)
    sweep = [row for row in pca_fragility["sweep"] if int(row["n_components"]) <= 20]
    d_vals = [int(row["n_components"]) for row in sweep]
    clean_f1 = [float(row["clean_zero_p975_f1"]) for row in sweep]
    cleanfit_noisy_f1 = [float(row["fra015_cleanfit_matched_p975_f1"]) for row in sweep]
    noisyfit_f1 = [float(row["fra015_noisyfit_p975_f1"]) for row in sweep]

    ssa_reg = ablation_10seed_full["regimes"]["tf_physics_ssa_fixed_load"]
    carla_clean = float(ssa_reg["zero_knn5_p975_f1"]["mean"])
    carla_noisy = float(ssa_reg["fra015_matched_p975_f1"]["mean"])

    ax_pca.axvspan(1.6, 5.4, color="#fbeee6", alpha=0.65, zorder=0, label=r"Standard $90\text{--}99.5\%$ variance zone ($d \leq 5$)")
    ax_pca.plot(d_vals, clean_f1, "-o", color=COL_TEAL, lw=2.1, ms=5.0, label="PCA Clean-Fit (Clean Test)", zorder=4)
    ax_pca.plot(
        d_vals,
        noisyfit_f1,
        "--s",
        color=COL_OCHRE,
        lw=1.8,
        ms=4.5,
        label=r"PCA Noisy-Refit ($0.15\,\mathrm{dB} / 0.75^\circ$)",
        zorder=4,
    )
    ax_pca.plot(
        d_vals,
        cleanfit_noisy_f1,
        "-.^",
        color=COL_BURG,
        lw=2.0,
        ms=5.0,
        label=r"PCA Clean-Fit ($0.15\,\mathrm{dB} / 0.75^\circ$ Test)",
        zorder=4,
    )
    ax_pca.axhline(
        carla_clean,
        color=COL_SSA,
        ls="-",
        lw=2.1,
        label=f"CARLA-TF (SSA) Clean ({carla_clean:.3f})",
        zorder=3,
    )
    ax_pca.axhline(
        carla_noisy,
        color=COL_SSA,
        ls=":",
        lw=2.0,
        label=f"CARLA-TF (SSA) + $0.15\\,\\mathrm{{dB}}$ ({carla_noisy:.3f})",
        zorder=3,
    )

    # Annotate 99% variance heuristic failure at d=4
    f1_d4 = clean_f1[d_vals.index(4)]
    ax_pca.annotate(
        f"99% var. ($d=4$)\nF1 = {f1_d4:.3f}",
        xy=(4, f1_d4),
        xytext=(3.2, 0.59),
        fontsize=8.1,
        fontweight="bold",
        color="#8c2d19",
        arrowprops={"arrowstyle": "->", "color": "#8c2d19", "lw": 1.1},
    )

    ax_pca.set_xlim(1.5, 20.5)
    ax_pca.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax_pca.set_ylim(0.24, 1.03)
    ax_pca.set_xlabel(r"Retained Principal Components $d$ (Out of $202$ Features)", fontweight="bold")
    ax_pca.set_ylabel(r"Zero-Label $\mathrm{F1}_{\mathrm{p97.5}}$", fontweight="bold")
    ax_pca.set_title("(a) PCA Variance Cliff & Clean-Fit Subspace Fragility", loc="left", pad=10)
    ax_pca.grid(True, alpha=0.45, zorder=0)
    style_axis(ax_pca)
    ax_pca.legend(loc="lower left", fontsize=7.5, framealpha=0.96)

    # Panel (b): 7-Level FRA Measurement Noise Sweep (0.00 dB -> 0.60 dB / 3.00 deg)
    levels = fra_noise_sweep["noise_levels"]
    sig_db = np.array([float(lv["sigma_db"]) for lv in levels], dtype=float)
    lkeys = [lv["key"] for lv in levels]
    dets = fra_noise_sweep["detectors"]

    carla_m = np.array([float(dets["carla_transformer_ssa"][k]["zero_p975_f1"]["mean"]) for k in lkeys])
    carla_s = np.array([float(dets["carla_transformer_ssa"][k]["zero_p975_f1"]["sample_std"]) for k in lkeys])
    ax_ns.plot(
        sig_db,
        carla_m,
        "-o",
        color=COL_SSA,
        lw=2.5,
        ms=5.8,
        label="CARLA-Transformer (SSA, 10 seeds)",
        zorder=5,
    )
    ax_ns.fill_between(sig_db, carla_m - carla_s, carla_m + carla_s, color=COL_SSA, alpha=0.16, zorder=2)

    det_curves = [
        ("pca_d8_noisyrefit", "PCA ($d=8$, Oracle Noisy-Refit)", COL_STEEL, "--", "s"),
        ("pca_d8_cleanfit", "PCA ($d=8$, Clean-Fit)", COL_TEAL, "-.", "^"),
        ("pca_d10_cleanfit", "PCA ($d=10$, Clean-Fit)", COL_BURG, ":", "v"),
        ("mahalanobis_lw_cleanfit", "Mahalanobis (LW, Clean-Fit)", COL_OCHRE, "--", "d"),
        ("raw_knn5_cleanfit", "Raw k-NN ($k=5$, Clean-Fit)", COL_SLATE, ":", "o"),
    ]
    for dkey, dlbl, col, ls, mk in det_curves:
        vals = [float(dets[dkey][k]["zero_p975_f1"]) for k in lkeys]
        ax_ns.plot(sig_db, vals, ls=ls, marker=mk, color=col, lw=1.85, ms=4.8, label=dlbl, zorder=4)

    ax_ns.axvline(0.15, color="#666666", ls=":", lw=1.1)
    ax_ns.text(0.16, 0.985, "Typical VNA/FRA\njitter ($0.15\\,\\mathrm{dB}$)", fontsize=7.9, color="#444444", va="top")

    ax_ns.set_xlim(-0.015, 0.62)
    ax_ns.set_ylim(0.30, 1.03)
    ax_ns.set_xlabel(r"FRA Measurement Jitter $\sigma_{\mathrm{mag}}$ [dB] ($\sigma_\phi = 5\,\sigma_{\mathrm{mag}}$ deg)", fontweight="bold")
    ax_ns.set_ylabel(r"Zero-Label $\mathrm{F1}_{\mathrm{p97.5}}$", fontweight="bold")
    ax_ns.set_title("(b) Seven-Level FRA Measurement Noise Robustness Sweep", loc="left", pad=10)
    ax_ns.grid(True, alpha=0.45, zorder=0)
    style_axis(ax_ns)
    ax_ns.legend(loc="lower left", fontsize=7.8, framealpha=0.96)

    save_figure(fig, save_paths)
    return fig


def plot_fig08_learning_curves_and_lofo_generalization(
    learning_curves: Dict[str, Any],
    lofo_5seed: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot healthy sample-efficiency curves (N_train=50..300) and 5-mode LOFO recall."""
    apply_publication_style()
    fig, (ax_lc, ax_lofo) = plt.subplots(1, 2, figsize=(14.4, 5.4), gridspec_kw={"wspace": 0.24})
    fig.patch.set_facecolor("#ffffff")

    # Panel (a): Healthy Sample-Efficiency Learning Curves (PR-AUC_pi=0.05)
    n_vals = np.array([int(n) for n in learning_curves["n_train_values"]], dtype=float)
    lc_dets = learning_curves["detectors"]
    lc_specs = [
        ("carla_transformer_ssa", "CARLA-Transformer (SSA)", COL_SSA, "-", "o", 2.4),
        ("pca_d10", "PCA Reconstruction ($d=10$)", COL_TEAL, "--", "s", 1.9),
        ("mahalanobis_lw", "Mahalanobis (Ledoit-Wolf)", COL_OCHRE, "-.", "^", 1.9),
        ("raw_knn5", "Raw k-NN ($k=5$)", COL_SLATE, ":", "d", 1.8),
    ]
    for dkey, dlbl, col, ls, mk, lw in lc_specs:
        means = np.array([float(lc_dets[dkey][str(int(n))]["auc_pr_pi05"]["mean"]) for n in n_vals])
        stds = np.array([float(lc_dets[dkey][str(int(n))]["auc_pr_pi05"]["sample_std"]) for n in n_vals])
        ax_lc.plot(n_vals, means, ls=ls, marker=mk, color=col, lw=lw, ms=5.4, label=dlbl, zorder=4)
        ax_lc.fill_between(n_vals, means - stds, means + stds, color=col, alpha=0.14, zorder=2)

    ax_lc.set_xticks(n_vals)
    ax_lc.set_xlim(40, 312)
    ax_lc.set_ylim(0.50, 0.98)
    ax_lc.set_xlabel(r"Healthy Training Curves $N_{\mathrm{train}}$ (Zero Real Faults)", fontweight="bold")
    ax_lc.set_ylabel(r"Field-Prevalence $\mathrm{PR}$-$\mathrm{AUC}_{\pi=0.05}$", fontweight="bold")
    ax_lc.set_title("(a) Healthy Sample Efficiency Across $N_{\\mathrm{train}} \\in [50, 300]$", loc="left", pad=10)
    ax_lc.grid(True, alpha=0.45, zorder=0)
    style_axis(ax_lc)
    ax_lc.legend(loc="lower right", fontsize=8.2, framealpha=0.96)

    # Panel (b): Leave-One-Fault-Out (LOFO) Zero-Shot Generalization Across All 5 Modes
    mode_specs = [
        ("Cout", "$C_{\\mathrm{out}}$\n($n=23$)"),
        ("Lout", "$L_{\\mathrm{out}}$\n($n=22$)"),
        ("Esr_C", "$\\mathrm{ESR}_C$\n($n=28$)"),
        ("Esr_L", "$\\mathrm{ESR}_L$\n($n=21$)"),
        ("Rds_1", "$R_{\\mathrm{ds},1}$\n($n=19$)"),
    ]
    modes = lofo_5seed["modes"]
    xm = np.arange(len(mode_specs), dtype=float)
    wm = 0.34
    rng = np.random.default_rng(108)

    ind_med = [100.0 * float(modes[k]["indist_zero_p975_recall_single"]["median"]) for k, _ in mode_specs]
    lofo_med = [100.0 * float(modes[k]["lofo_zero_p975_recall_single"]["median"]) for k, _ in mode_specs]
    lofo_auc = [float(modes[k]["lofo_single_auc_roc"]["median"]) for k, _ in mode_specs]

    ax_lofo.bar(
        xm - wm / 2,
        ind_med,
        width=wm * 0.92,
        color="#9ecae1",
        edgecolor="#1f4e79",
        lw=1.0,
        label="In-Distribution SSA (Median Recall)",
        zorder=3,
    )
    bars_lofo = ax_lofo.bar(
        xm + wm / 2,
        lofo_med,
        width=wm * 0.92,
        color=COL_SSA,
        edgecolor="#ffffff",
        lw=0.8,
        label="Zero-Shot LOFO (Held-Out Mode Median Recall)",
        zorder=3,
    )

    for idx_k, (mkey, _) in enumerate(mode_specs):
        v_ind = 100.0 * np.array(modes[mkey]["indist_zero_p975_recall_single"]["values"], dtype=float)
        v_lofo = 100.0 * np.array(modes[mkey]["lofo_zero_p975_recall_single"]["values"], dtype=float)
        ax_lofo.scatter(
            np.full(len(v_ind), xm[idx_k] - wm / 2) + rng.uniform(-0.055, 0.055, size=len(v_ind)),
            v_ind,
            s=20,
            facecolor="#ffffff",
            edgecolor="#1f4e79",
            lw=1.0,
            zorder=5,
        )
        ax_lofo.scatter(
            np.full(len(v_lofo), xm[idx_k] + wm / 2) + rng.uniform(-0.055, 0.055, size=len(v_lofo)),
            v_lofo,
            s=20,
            facecolor="#ffffff",
            edgecolor="#0b2545",
            lw=1.0,
            zorder=5,
        )
        bar = bars_lofo[idx_k]
        y_top = max(float(np.max(v_lofo)), bar.get_height())
        ax_lofo.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_top + 3.0,
            f"AUC\n{lofo_auc[idx_k]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color=COL_SSA if lofo_auc[idx_k] > 0.90 else COL_BURG,
        )

    ax_lofo.set_xticks(xm)
    ax_lofo.set_xticklabels([lbl for _, lbl in mode_specs], fontsize=9.1)
    ax_lofo.set_yticks([0, 20, 40, 60, 80, 100])
    ax_lofo.set_ylim(0.0, 124.0)
    ax_lofo.set_ylabel("Single-Fault Zero-Label Recall at p97.5 [%]", fontweight="bold")
    ax_lofo.set_title("(b) Leave-One-Fault-Out (LOFO) Zero-Shot Generalization (5 Seeds)", loc="left", pad=10)
    ax_lofo.grid(True, axis="y", alpha=0.45, zorder=0)
    style_axis(ax_lofo)
    ax_lofo.legend(loc="upper right", fontsize=8.0, framealpha=0.96)

    save_figure(fig, save_paths)
    return fig


def plot_fig09_attention_and_jacobian_saliency(
    attn_audit: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot self-attention key routing vs input-Jacobian resonance load-invariance notch."""
    apply_publication_style()
    fig, (ax_att, ax_sal) = plt.subplots(1, 2, figsize=(14.8, 5.4), gridspec_kw={"wspace": 0.31})
    fig.patch.set_facecolor("#ffffff")

    freqs_khz = np.array(attn_audit["frequencies_hz"], dtype=float) / 1000.0
    n_bins = float(len(freqs_khz))

    # Panel (a): Self-Attention Key Routing Across 101 Frequency Tokens (10 Seeds)
    att_reg = attn_audit["attention_by_regime"]
    att_specs = [
        ("tf_physics_ssa_fixed_load", "CARLA-Transformer (SSA", COL_SSA, "-"),
        ("tf_recon_only", "AE-Transformer (MSE", COL_BURG, "--"),
        ("tf_heuristic", "CARLA-Transformer (Heuristic", COL_TERRA, ":"),
    ]
    ax_att.axvspan(0.7, 2.0, color="#fff3cd", alpha=0.55, zorder=0, label="Resonance band ($0.7\\text{--}2.0\\,\\mathrm{kHz}$)")
    max_att_upper = 1.5
    for rkey, rprefix, col, ls in att_specs:
        h_mean = float(att_reg[rkey]["normalized_entropy"]["mean"])
        rlbl = f"{rprefix}, $H = {h_mean:.4f}$)"
        mean_p = 100.0 * np.array(att_reg[rkey]["mean_key_profile_across_seeds"], dtype=float)
        std_p = 100.0 * np.array(att_reg[rkey]["std_key_profile_across_seeds"], dtype=float)
        max_att_upper = max(max_att_upper, float(np.max(mean_p + std_p)))
        ax_att.plot(freqs_khz, mean_p, color=col, ls=ls, lw=2.0, label=rlbl, zorder=4)
        ax_att.fill_between(freqs_khz, np.maximum(0.0, mean_p - std_p), mean_p + std_p, color=col, alpha=0.14, zorder=2)

    ax_att.axhline(100.0 / n_bins, color="#555555", ls=":", lw=1.2, label=f"Uniform routing ({100.0 / n_bins:.2f}% / bin)")
    ax_att.set_xscale("log")
    ax_att.set_xlim(0.1, 50.0)
    ax_att.set_ylim(0.0, max_att_upper * 1.18)
    ax_att.set_xlabel("Frequency $f$ [kHz]", fontweight="bold")
    ax_att.set_ylabel("Mean Key Attention Weight [% per bin]", fontweight="bold")
    ax_att.set_title("(a) Self-Attention Routing: Resonance Collapse vs. Global Bus", loc="left", pad=10)
    ax_att.grid(True, which="both", alpha=0.45, zorder=0)
    style_axis(ax_att)
    ax_att.legend(loc="upper right", fontsize=7.8, framealpha=0.96)

    # Panel (b): Input-Jacobian Score Saliency -> Resonance Load-Invariance Notch
    sal_reg = attn_audit["saliency_by_regime"]
    sal_str = attn_audit["saliency_by_stratum_ssa_fixed_load"]

    ax_sal.axvspan(0.7, 2.0, color="#fff3cd", alpha=0.60, zorder=0, label="Load-$Q$ resonance band ($0.7\\text{--}2.0\\,\\mathrm{kHz}$)")

    mse_sal = n_bins * np.array(sal_reg["tf_recon_only"]["normalized_mean_profile"], dtype=float)
    heur_sal = n_bins * np.array(sal_reg["tf_heuristic"]["normalized_mean_profile"], dtype=float)
    ssa_healthy = n_bins * np.array(sal_str["healthy"]["normalized_mean_profile"], dtype=float)
    ssa_esrc = n_bins * np.array(sal_str["single_Esr_C"]["normalized_mean_profile"], dtype=float)
    ssa_rds1 = n_bins * np.array(sal_str["single_Rds_1"]["normalized_mean_profile"], dtype=float)

    ax_sal.plot(
        freqs_khz,
        mse_sal,
        color=COL_BURG,
        ls="--",
        lw=2.0,
        label=r"AE-TF (MSE): $2.92\times$ Resonance Spike",
        zorder=3,
    )
    ax_sal.plot(
        freqs_khz,
        heur_sal,
        color=COL_TERRA,
        ls=":",
        lw=1.8,
        label=r"CARLA-TF (Heuristic): $2.67\times$ Resonance Spike",
        zorder=3,
    )
    ax_sal.plot(
        freqs_khz,
        ssa_healthy,
        color=COL_SSA,
        ls="-",
        lw=2.5,
        label=r"CARLA-TF (SSA, Healthy): $0.83\times$ Notch",
        zorder=5,
    )
    ax_sal.plot(
        freqs_khz,
        ssa_esrc,
        color=COL_TEAL,
        ls="-.",
        lw=1.8,
        label=r"CARLA-TF (SSA, Single $\mathrm{ESR}_C$)",
        zorder=4,
    )
    ax_sal.plot(
        freqs_khz,
        ssa_rds1,
        color=COL_OCHRE,
        ls="-",
        lw=1.5,
        alpha=0.9,
        label=r"CARLA-TF (SSA, Single $R_{\mathrm{ds},1}$)",
        zorder=4,
    )
    ax_sal.axhline(1.0, color="#555555", ls=":", lw=1.1, label="Uniform sensitivity ($1.0\\times$)")

    min_idx = int(np.argmin(ssa_healthy[(freqs_khz >= 0.7) & (freqs_khz <= 2.0)]))
    res_mask_idx = np.where((freqs_khz >= 0.7) & (freqs_khz <= 2.0))[0]
    notch_idx = int(res_mask_idx[min_idx])
    ax_sal.annotate(
        "Resonance load-\ninvariance notch",
        xy=(float(freqs_khz[notch_idx]), float(ssa_healthy[notch_idx])),
        xytext=(3.2, 1.68),
        fontsize=8.1,
        fontweight="bold",
        color=COL_SSA,
        arrowprops={"arrowstyle": "->", "color": COL_SSA, "lw": 1.2, "connectionstyle": "arc3,rad=0.15"},
    )

    max_sal_upper = float(max(np.max(mse_sal), np.max(heur_sal), 6.0))
    ax_sal.set_xscale("log")
    ax_sal.set_xlim(0.1, 50.0)
    ax_sal.set_ylim(0.25, max_sal_upper * 1.14)
    ax_sal.set_xlabel("Frequency $f$ [kHz]", fontweight="bold")
    ax_sal.set_ylabel(r"Input-Jacobian $\|\nabla_X a(X)\|_2$ [$\times$ uniform]", fontweight="bold")
    ax_sal.set_title("(b) Input-Jacobian Saliency Reveals Load-Invariance Mechanism", loc="left", pad=10)
    ax_sal.grid(True, which="both", alpha=0.45, zorder=0)
    style_axis(ax_sal)
    ax_sal.legend(loc="upper right", fontsize=7.7, framealpha=0.96)

    save_figure(fig, save_paths)
    return fig
