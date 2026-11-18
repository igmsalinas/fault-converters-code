"""
Frequency Response and Bode Deviation Visualization Module
==========================================================

Plotting routines for power converter transfer functions ($H(s)$ Bode diagrams)
and single/multi-component degradation drift analysis with physical shift callouts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from .style import apply_publication_style, save_figure
from ..data.physics_anomaly import BuckConverter


def compute_buck_bode(
    cout_mult: float = 1.0,
    esr_c_mult: float = 1.0,
    l_mult: float = 1.0,
    esr_l_mult: float = 1.0,
    rds_mult: float = 1.0,
    n_points: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact frequency response (freq, mag_dB, phase_deg) for a parameterized buck converter."""
    conv = BuckConverter(
        vin=50.0,
        inductance=0.2e-3 * l_mult,
        capacitance=100e-6 * cout_mult,
        rds=0.01 * rds_mult,
        esr_ind=0.10 * esr_l_mult,
        esr_cap=0.02 * esr_c_mult,
        load=4.0,
        duty=0.4,
    )
    freqs = np.logspace(np.log10(100.0), np.log10(50000.0), n_points)
    omega = 2.0 * np.pi * freqs
    h = conv.transfer_function().frequency_response(omega)
    mag_db = 20.0 * np.log10(np.abs(h))
    phase_deg = np.unwrap(np.angle(h)) * (180.0 / np.pi)
    return freqs, mag_db, phase_deg


def plot_paper_bode_degradation(
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """
    Plot dual 2x2 Bode degradation curves for Output Capacitance ($C_{out}$) and
    Capacitor ESR ($ESR_C$) across the full dataset ranges with progressive intensity.
    """
    apply_publication_style()

    fig, ((ax_c_mag, ax_e_mag), (ax_c_ph, ax_e_ph)) = plt.subplots(2, 2, figsize=(14.0, 8.0), sharex=True)

    # --- 1. Capacitance Degradation (Downward drift: 1.0 down to 0.30) ---
    cout_multipliers = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
    c_labels = [
        "Nominal (1.00× / 0%)",
        "Normal (0.90× / −10%)",
        "Normal Limit (0.80× / −20%)",
        "Degraded (0.70× / −30%)",
        "Degraded (0.60× / −40%)",
        "Degraded (0.50× / −50%)",
        "Degraded (0.40× / −60%)",
        "Severe Fault (0.30× / −70%)",
    ]
    # Progressive blue/cyan gradient from light to intense dark navy
    c_colors = [
        "#111111",                               # Nominal solid black
        plt.cm.Blues(0.40),
        plt.cm.Blues(0.52),
        plt.cm.Blues(0.64),
        plt.cm.Blues(0.74),
        plt.cm.Blues(0.84),
        plt.cm.Blues(0.92),
        "#082040",                               # Deepest Navy
    ]

    for mult, lbl, col in zip(cout_multipliers, c_labels, c_colors):
        f, mag, ph = compute_buck_bode(cout_mult=mult)
        is_nom = (mult == 1.00)
        lw = 2.6 if is_nom else (1.4 + 1.2 * (1.0 - mult) / 0.70)
        ls = "-" if is_nom else "--" if mult >= 0.80 else "-."
        zorder = 10 if is_nom else 3

        ax_c_mag.semilogx(f, mag, label=lbl, color=col, lw=lw, ls=ls, zorder=zorder)
        ax_c_ph.semilogx(f, ph, label=lbl, color=col, lw=lw, ls=ls, zorder=zorder)

    # Physical resonance shift annotation
    ax_c_mag.annotate(
        r"$\mathbf{\Delta f_0\ Resonance\ Shift:}$" + "\n" + r"$f_0 = \frac{1}{2\pi\sqrt{L C_{out}}} \uparrow$",
        xy=(1.8e3, 39.0), xytext=(3.2e3, 22.0),
        arrowprops=dict(facecolor="#1f4e79", edgecolor="#1f4e79", arrowstyle="->", lw=1.8),
        fontsize=10.0, fontweight="bold", color="#1f4e79",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#edf4fb", edgecolor="#b4d2ee", alpha=0.95)
    )

    ax_c_mag.set_title(r"Output Capacitance Degradation ($C_{out}$)", fontsize=12.5, fontweight="bold", pad=8)
    ax_c_mag.set_ylabel("Magnitude (dB)", fontsize=11.0, fontweight="bold")
    ax_c_mag.set_ylim(-35, 58)
    ax_c_mag.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_c_mag.legend(fontsize=8.0, loc="lower left", framealpha=0.95, edgecolor="#cccccc")

    ax_c_ph.set_xlabel("Frequency (Hz)", fontsize=11.0, fontweight="bold")
    ax_c_ph.set_ylabel("Phase (degrees)", fontsize=11.0, fontweight="bold")
    ax_c_ph.grid(True, which="both", linestyle="--", alpha=0.35)

    # --- 2. Capacitor ESR Degradation (Upward drift: 1.0 up to 8.0) ---
    esr_multipliers = [1.00, 1.50, 2.00, 3.00, 4.00, 5.50, 7.00, 8.00]
    e_labels = [
        "Nominal (1.00× / 0%)",
        "Normal (1.50× / +50%)",
        "Normal Limit (2.00× / +100%)",
        "Degraded (3.00× / +200%)",
        "Degraded (4.00× / +300%)",
        "Degraded (5.50× / +450%)",
        "Degraded (7.00× / +600%)",
        "Severe Fault (8.00× / +700%)",
    ]
    # Progressive amber/terracotta/crimson gradient
    e_colors = [
        "#111111",                               # Nominal solid black
        plt.cm.YlOrRd(0.35),
        plt.cm.YlOrRd(0.48),
        plt.cm.YlOrRd(0.60),
        plt.cm.YlOrRd(0.72),
        plt.cm.YlOrRd(0.82),
        plt.cm.YlOrRd(0.92),
        "#67000d",                               # Deepest Crimson
    ]

    for mult, lbl, col in zip(esr_multipliers, e_labels, e_colors):
        f, mag, ph = compute_buck_bode(esr_c_mult=mult)
        is_nom = (mult == 1.00)
        lw = 2.6 if is_nom else (1.4 + 1.2 * (mult - 1.0) / 7.0)
        ls = "-" if is_nom else "--" if mult <= 2.00 else "-."
        zorder = 10 if is_nom else 3

        ax_e_mag.semilogx(f, mag, label=lbl, color=col, lw=lw, ls=ls, zorder=zorder)
        ax_e_ph.semilogx(f, ph, label=lbl, color=col, lw=lw, ls=ls, zorder=zorder)

    # Physical ESR zero damping annotation
    ax_e_mag.annotate(
        r"$\mathbf{ESR\ Zero\ Damping:}$" + "\n" + r"$f_z = \frac{1}{2\pi R_C C_{out}} \downarrow$",
        xy=(2.2e4, -16.0), xytext=(5.5e3, 24.0),
        arrowprops=dict(facecolor="#b33939", edgecolor="#b33939", arrowstyle="->", lw=1.8),
        fontsize=10.0, fontweight="bold", color="#b33939",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf0ed", edgecolor="#f5c2b7", alpha=0.95)
    )

    ax_e_mag.set_title(r"Capacitor ESR Degradation ($ESR_C$)", fontsize=12.5, fontweight="bold", pad=8)
    ax_e_mag.set_ylabel("Magnitude (dB)", fontsize=11.0, fontweight="bold")
    ax_e_mag.set_ylim(-35, 58)
    ax_e_mag.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_e_mag.legend(fontsize=8.0, loc="lower left", framealpha=0.95, edgecolor="#cccccc")

    ax_e_ph.set_xlabel("Frequency (Hz)", fontsize=11.0, fontweight="bold")
    ax_e_ph.set_ylabel("Phase (degrees)", fontsize=11.0, fontweight="bold")
    ax_e_ph.grid(True, which="both", linestyle="--", alpha=0.35)

    fig.tight_layout()
    save_figure(fig, save_paths)
    return fig


def plot_bode_deviation(
    component_name: str,
    variation_paths: Optional[Dict[float, Union[str, Path]]] = None,
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
    target_variations: Optional[List[float]] = None,
) -> plt.Figure:
    """
    Plot Bode magnitude and phase deviations for a single degraded component across the full range.
    """
    apply_publication_style()

    # Generate full dataset sweep based on component
    comp_map = {
        "Cout": ("cout_mult", [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30], plt.cm.Blues, "Output Capacitance ($C_{out}$)"),
        "Esr_C": ("esr_c_mult", [1.00, 1.50, 2.00, 3.00, 4.00, 5.50, 7.00, 8.00], plt.cm.YlOrRd, "Capacitor ESR ($ESR_C$)"),
        "Lout": ("l_mult", [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40], plt.cm.Blues, "Inductor ($L$)"),
        "Esr_L": ("esr_l_mult", [1.00, 1.20, 1.40, 2.00, 3.00, 4.50, 6.00], plt.cm.YlOrRd, "Inductor ESR ($ESR_L$)"),
        "Rds_1": ("rds_mult", [1.00, 1.50, 2.20, 4.00, 7.00, 10.00, 15.00], plt.cm.YlOrRd, "MOSFET On-Resistance ($R_{ds}$)"),
    }

    arg_name, multipliers, cmap, display_title = comp_map.get(
        component_name, ("cout_mult", [1.0, 0.8, 0.6, 0.4], plt.cm.Blues, f"Component ({component_name})")
    )

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8.0, 7.5), sharex=True)

    n_m = len(multipliers)
    for i, mult in enumerate(multipliers):
        kwargs = {arg_name: mult}
        f, amp, phase = compute_buck_bode(**kwargs)
        is_nom = (mult == 1.00)
        pct = (mult - 1.0) * 100.0
        label = "Nominal (0%)" if is_nom else f"{pct:+.0f}% ({mult:.2f}×)"
        col = "#111111" if is_nom else cmap(0.35 + 0.60 * (i / max(1, n_m - 1)))
        lw = 2.6 if is_nom else 1.8
        ls = "-" if is_nom else "--" if (0.80 <= mult <= 1.20 or (arg_name == 'esr_c_mult' and mult <= 2.0)) else "-."

        ax_mag.semilogx(f, amp, label=label, color=col, linewidth=lw, linestyle=ls)
        ax_phase.semilogx(f, phase, label=label, color=col, linewidth=lw, linestyle=ls)

    if component_name == "Cout":
        ax_mag.annotate(
            r"$\mathbf{\Delta f_0\ Resonance\ Shift:}$" + "\n" + r"$f_0 \propto 1/\sqrt{L C_{out}}$",
            xy=(2.0e3, 49.0), xytext=(3.5e3, 30.0),
            arrowprops=dict(facecolor="#1f4e79", edgecolor="#1f4e79", arrowstyle="->", lw=1.8),
            fontsize=10.0, fontweight="bold", color="#1f4e79",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#edf4fb", edgecolor="#b4d2ee", alpha=0.95)
        )
    elif component_name == "Esr_C":
        ax_mag.annotate(
            r"$\mathbf{ESR\ Zero\ Damping:}$" + "\n" + r"$f_z \approx 1/(2\pi R_C C_{out})$",
            xy=(1.2e4, -12.0), xytext=(5.5e3, 24.0),
            arrowprops=dict(facecolor="#b33939", edgecolor="#b33939", arrowstyle="->", lw=1.8),
            fontsize=10.0, fontweight="bold", color="#b33939",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdf0ed", edgecolor="#f5c2b7", alpha=0.95)
        )

    ax_mag.set_title(f"Bode Response Drift · {display_title}", fontsize=12.5, fontweight="bold", pad=8)
    ax_mag.set_ylabel("Magnitude (dB)", fontsize=11.0, fontweight="bold")
    ax_mag.set_ylim(-35, 58)
    ax_mag.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_mag.legend(ncol=2, fontsize=8.5, loc="lower left", framealpha=0.95, edgecolor="#cccccc")

    ax_phase.set_xlabel("Frequency (Hz)", fontsize=11.0, fontweight="bold")
    ax_phase.set_ylabel("Phase (degrees)", fontsize=11.0, fontweight="bold")
    ax_phase.grid(True, which="both", linestyle="--", alpha=0.35)

    fig.tight_layout()
    save_figure(fig, save_paths)
    return fig


def plot_transfer_function_comparison(
    curves: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]],
    title: str = "Transfer Function Comparison",
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """
    Plot a comparison between arbitrary transfer function curves (freq, amp, phase, label, color).
    """
    apply_publication_style()
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    for freq, amp, phase, label, color in curves:
        ax_mag.semilogx(freq, amp, label=label, color=color, linewidth=1.8)
        ax_phase.semilogx(freq, phase, label=label, color=color, linewidth=1.8)

    ax_mag.set_title(title, fontsize=12, fontweight="bold")
    ax_mag.set_ylabel("Magnitude (dB)", fontsize=11, fontweight="bold")
    ax_mag.grid(True, linestyle="--", alpha=0.35)
    ax_mag.legend(fontsize=9.5)

    ax_phase.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax_phase.set_ylabel("Phase (degrees)", fontsize=11, fontweight="bold")
    ax_phase.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    save_figure(fig, save_paths)
    return fig
