"""
Circuit Schematic Visualization Module
======================================

Renders publication-grade electronic power converter circuit schematics
with authentic IEEE standard symbols (N-channel MOSFETs, Inductors, Capacitors,
Resistors, Sources) and small-signal frequency response probe annotations.
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arc, Polygon

from .style import apply_publication_style, save_figure


def draw_buck_converter_schematic(
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
    vin: float = 50.0,
    vout: float = 20.0,
    l_nom: float = 0.2,
    rl_nom: float = 100.0,
    c_nom: float = 100.0,
    rc_nom: float = 20.0,
    rds1_nom: float = 10.0,
    rds2_nom: float = 10.0,
    rout_nom: float = 4.0,
    fsw_khz: float = 100.0,
) -> plt.Figure:
    """
    Draw an IEEE-standard synchronous Buck converter schematic with authentic
    MOSFET transistor symbols, PWM gate drives, and AC frequency response probe.
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(14.5, 7.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    wire_col = "#111111"
    accent_blue = "#1f4e79"
    accent_teal = "#2e7d6e"
    accent_orange = "#b85d38"
    lw_wire = 2.6
    lw_comp = 2.8

    def draw_wire(pts, color=wire_col, lw=lw_wire, ls="-", zorder=2):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=zorder)

    def draw_node(x, y, r=0.10, color=wire_col):
        circle = Circle((x, y), r, facecolor=color, edgecolor=color, zorder=6)
        ax.add_patch(circle)

    def draw_ground(x, y):
        draw_wire([(x, y), (x, y - 0.32)])
        draw_wire([(x - 0.40, y - 0.32), (x + 0.40, y - 0.32)], lw=3.0)
        draw_wire([(x - 0.25, y - 0.46), (x + 0.25, y - 0.46)], lw=2.4)
        draw_wire([(x - 0.12, y - 0.60), (x + 0.12, y - 0.60)], lw=1.8)

    def draw_dc_source(x, y, r=0.65, label=f"$V_{{in}} = {vin:.0f}\\,\\mathrm{{V}}$"):
        c = Circle((x, y), r, facecolor="#ffffff", edgecolor=wire_col, lw=lw_comp, zorder=4)
        ax.add_patch(c)
        ax.text(x, y + 0.22, "+", ha="center", va="center", fontsize=20, fontweight="bold", color="#111111", zorder=10)
        ax.text(x, y - 0.24, "−", ha="center", va="center", fontsize=22, fontweight="bold", color="#111111", zorder=10)
        ax.text(x - 0.85, y, label, ha="right", va="center", fontsize=14.0, fontweight="bold", color=accent_blue)

    def draw_n_mosfet_high_side(x, y, label="$Q_1$", rds_val="10 mΩ", pwm_label="$v_{g1}(t)$"):
        draw_wire([(x - 1.00, y), (x - 0.35, y)])
        draw_wire([(x - 0.35, y + 0.18), (x - 0.35, y - 0.18)], lw=lw_comp)

        draw_wire([(x + 1.00, y), (x + 0.35, y)])
        draw_wire([(x + 0.35, y + 0.18), (x + 0.35, y - 0.18)], lw=lw_comp)

        draw_wire([(x, y), (x, y - 0.18)], lw=lw_comp)
        ax.add_patch(Polygon([(x - 0.11, y - 0.14), (x + 0.11, y - 0.14), (x, y + 0.04)],
                             closed=True, facecolor=accent_blue, edgecolor=accent_blue, zorder=5))

        draw_wire([(x - 0.45, y - 0.26), (x + 0.45, y - 0.26)], lw=lw_comp + 0.8)
        draw_wire([(x, y - 0.26), (x, y - 0.52)])

        box = FancyBboxPatch((x - 1.10, y - 0.82), 2.10, 1.80,
                             boxstyle="Round,pad=0.08,rounding_size=0.15",
                             facecolor="#edf2f7", edgecolor=accent_blue, lw=1.6, ls="--", zorder=1)
        ax.add_patch(box)
        ax.text(x, y + 1.10, f"{label} (High-Side N-MOS)", ha="center", va="bottom",
                fontsize=13.0, fontweight="bold", color=accent_blue)
        ax.text(x, y + 0.78, f"$R_{{ds,1}} = {rds_val}$", ha="center", va="bottom",
                fontsize=11.5, color="#2d3748")
        ax.text(x, y - 0.62, f"Gate: {pwm_label} ($D$)", ha="center", va="center",
                fontsize=11.0, fontweight="bold", color=accent_orange)

    def draw_n_mosfet_low_side(x, y, label="$Q_2$", rds_val="10 mΩ", pwm_label="$v_{g2}(t)$"):
        draw_wire([(x, y + 1.00), (x, y + 0.35)])
        draw_wire([(x - 0.18, y + 0.35), (x + 0.18, y + 0.35)], lw=lw_comp)

        draw_wire([(x, y - 1.00), (x, y - 0.35)])
        draw_wire([(x - 0.18, y - 0.35), (x + 0.18, y - 0.35)], lw=lw_comp)

        draw_wire([(x, y), (x - 0.18, y)], lw=lw_comp)
        ax.add_patch(Polygon([(x - 0.14, y - 0.11), (x - 0.14, y + 0.11), (x + 0.04, y)],
                             closed=True, facecolor=accent_blue, edgecolor=accent_blue, zorder=5))

        draw_wire([(x - 0.26, y - 0.45), (x - 0.26, y + 0.45)], lw=lw_comp + 0.8)
        draw_wire([(x - 0.26, y), (x - 0.60, y)])

        box = FancyBboxPatch((x - 1.80, y - 1.05), 3.80, 2.10,
                             boxstyle="Round,pad=0.08,rounding_size=0.15",
                             facecolor="#edf2f7", edgecolor=accent_blue, lw=1.6, ls="--", zorder=1)
        ax.add_patch(box)
        ax.text(x + 1.00, y, f"{label} (Low-Side)\n$R_{{ds,2}} = {rds_val}$",
                ha="left", va="center", fontsize=12.0, fontweight="bold", color=accent_blue)
        ax.text(x - 0.75, y, f"Gate: {pwm_label}\n($1-D$)",
                ha="right", va="center", fontsize=11.0, fontweight="bold", color=accent_orange)

    def draw_inductor_with_esr(x_l, y, x_r, l_val="0.2 mH", rl_val="100 mΩ"):
        n_coils = 4
        r_coil = 0.22
        start_x = x_l - (n_coils * 2 * r_coil) / 2
        for i in range(n_coils):
            cx = start_x + (2 * i + 1) * r_coil
            arc = Arc((cx, y), 2*r_coil, 2*r_coil, angle=0, theta1=0, theta2=180,
                      edgecolor=wire_col, lw=lw_comp + 0.8, zorder=4)
            ax.add_patch(arc)
        draw_wire([(start_x - 0.22, y), (start_x, y)])
        
        rx = x_r
        rw, rh = 0.95, 0.40
        draw_wire([(start_x + n_coils * 2 * r_coil, y), (rx - rw/2, y)])
        rect = Rectangle((rx - rw/2, y - rh/2), rw, rh, facecolor="#ffffff",
                         edgecolor=wire_col, lw=lw_comp, zorder=4)
        ax.add_patch(rect)
        draw_wire([(rx + rw/2, y), (rx + rw/2 + 0.30, y)])
        
        ax.text(x_l, y + 0.48, f"Inductor $L = {l_val}$", ha="center", va="bottom",
                fontsize=13.0, fontweight="bold", color=accent_blue)
        ax.text(rx, y + 0.34, f"$R_L = {rl_val}$", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color="#2d3748")
        ax.text(rx, y - 0.34, "(Inductor ESR)", ha="center", va="top",
                fontsize=10.0, color="#718096", style="italic")

    def draw_capacitor_with_esr(x, y_top, y_bot, c_val="100 μF", rc_val="20 mΩ"):
        y_cap = (y_top + y_bot) / 2 + 0.70
        p_w = 0.75
        sep = 0.18
        draw_wire([(x, y_top), (x, y_cap + sep/2)])
        draw_wire([(x - p_w/2, y_cap + sep/2), (x + p_w/2, y_cap + sep/2)], lw=lw_comp + 1.0)
        draw_wire([(x - p_w/2, y_cap - sep/2), (x + p_w/2, y_cap - sep/2)], lw=lw_comp + 1.0)

        ax.text(x + p_w/2 + 0.30, y_cap, f"$C_{{out}} = {c_val}$", ha="left", va="center",
                fontsize=13.0, fontweight="bold", color=accent_blue)

        y_esr = (y_top + y_bot) / 2 - 0.70
        rw, rh = 0.40, 0.95
        draw_wire([(x, y_cap - sep/2), (x, y_esr + rh/2)])
        rect = Rectangle((x - rw/2, y_esr - rh/2), rw, rh, facecolor="#ffffff",
                         edgecolor=wire_col, lw=lw_comp, zorder=4)
        ax.add_patch(rect)
        draw_wire([(x, y_esr - rh/2), (x, y_bot)])
        ax.text(x + rw/2 + 0.30, y_esr + 0.12, f"$R_C = {rc_val}$", ha="left", va="center",
                fontsize=11.5, fontweight="bold", color="#2d3748")
        ax.text(x + rw/2 + 0.30, y_esr - 0.22, "(Capacitor ESR)", ha="left", va="center",
                fontsize=10.0, color="#718096", style="italic")

    def draw_load_resistor(x, y_top, y_bot, r_val="4.0 Ω"):
        y_mid = (y_top + y_bot) / 2
        rw, rh = 0.42, 1.35
        draw_wire([(x, y_top), (x, y_mid + rh/2)])
        rect = Rectangle((x - rw/2, y_mid - rh/2), rw, rh, facecolor="#ffffff",
                         edgecolor=wire_col, lw=lw_comp, zorder=4)
        ax.add_patch(rect)
        draw_wire([(x, y_mid - rh/2), (x, y_bot)])
        ax.text(x + rw/2 + 0.30, y_mid, f"Load $R_{{\\mathrm{{out}}}} = {r_val}$", ha="left", va="center",
                fontsize=13.0, fontweight="bold", color=accent_blue)

    # --- Schematic Layout ----------------------------------------------------
    y_rail_top = 5.20
    y_rail_bot = 1.20

    # 1. DC Source
    draw_dc_source(1.2, (y_rail_top + y_rail_bot) / 2, label=f"$V_{{in}} = {vin:.0f}\\,\\mathrm{{V}}$")
    draw_wire([(1.2, (y_rail_top + y_rail_bot)/2 + 0.65), (1.2, y_rail_top), (2.80, y_rail_top)])
    draw_wire([(1.2, (y_rail_top + y_rail_bot)/2 - 0.65), (1.2, y_rail_bot), (13.3, y_rail_bot)])
    draw_ground(1.2, y_rail_bot)

    # 2. Q1 High-Side MOSFET
    draw_n_mosfet_high_side(3.80, y_rail_top, label="$Q_1$", rds_val=f"{rds1_nom:.0f} mΩ", pwm_label="$v_{g1}(t)$")
    draw_wire([(4.80, y_rail_top), (5.60, y_rail_top)])

    # 3. Switching Node Vx
    draw_node(5.60, y_rail_top)
    ax.text(5.60, y_rail_top + 0.45, "Switch Node\n$V_x(t)$", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color=accent_blue)

    # 4. Q2 Low-Side MOSFET
    draw_wire([(5.60, y_rail_top), (5.60, 4.20)])
    draw_n_mosfet_low_side(5.60, 3.20, label="$Q_2$", rds_val=f"{rds2_nom:.0f} mΩ", pwm_label="$v_{g2}(t)$")
    draw_wire([(5.60, 2.20), (5.60, y_rail_bot)])
    draw_node(5.60, y_rail_bot)

    # 5. Inductor + ESR
    draw_wire([(5.60, y_rail_top), (6.45, y_rail_top)])
    draw_inductor_with_esr(7.45, y_rail_top, 9.05, l_val=f"{l_nom:.1f}\\,\\mathrm{{mH}}", rl_val=f"{rl_nom:.0f}\\,\\mathrm{{m}}\\Omega")
    draw_wire([(9.60, y_rail_top), (10.60, y_rail_top)])

    # 6. Output Node & Voltage Vout
    draw_node(10.60, y_rail_top)
    ax.text(10.60, y_rail_top + 0.40, f"$V_{{out}} = {vout:.0f}\\,\\mathrm{{V}}$", ha="center", va="bottom",
            fontsize=14.0, fontweight="bold", color=accent_teal)

    # 7. Output Capacitor Branch
    draw_capacitor_with_esr(10.60, y_rail_top, y_rail_bot, c_val=f"{c_nom:.0f}\\,\\mu\\mathrm{{F}}", rc_val=f"{rc_nom:.0f}\\,\\mathrm{{m}}\\Omega")
    draw_node(10.60, y_rail_bot)
    draw_ground(10.60, y_rail_bot)

    # 8. Load Resistor Branch
    draw_wire([(10.60, y_rail_top), (13.30, y_rail_top)])
    draw_load_resistor(13.30, y_rail_top, y_rail_bot, r_val=f"{rout_nom:.0f}\\,\\Omega")
    draw_node(13.30, y_rail_bot)

    # 9. Small-Signal AC Frequency Sweep Annotation Box
    ac_box = FancyBboxPatch(
        (0.40, -0.35), 13.5, 0.95,
        boxstyle="Round,pad=0.08,rounding_size=0.15",
        facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.6, zorder=3
    )
    ax.add_patch(ac_box)
    ax.text(7.15, 0.12,
            r"$\mathbf{Small\text{-}Signal\ AC\ Frequency\ Response\ Setup:}\quad H(s) = \frac{\tilde{v}_{out}(s)}{\tilde{v}_{inj}(s)} \quad \text{across } 101 \text{ log frequencies } (100\,\mathrm{Hz} - 50\,\mathrm{kHz}),\quad f_{sw} = 100\,\mathrm{kHz}$",
            ha="center", va="center", fontsize=11.5, color="#111111")

    ax.set_xlim(-0.6, 15.0)
    ax.set_ylim(-0.55, 6.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig
