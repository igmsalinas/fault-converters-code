"""
Neural Architecture Visualization Module
========================================

Generates publication-quality modular block diagrams and isometric tensor stage
diagrams for autoencoder architectures (Conv1D, GRU, LSTM, Transformer, VAE, CARLA).
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
import colorsys
import matplotlib.colors as mc

from .style import apply_publication_style, save_figure

HERO_MODELS = ["conv1d_ae", "lstm_ae", "gru_ae", "transformer_ae", "vae"]

STEP = 2.6
X = "\u00d7"


def _shade(colour: str, amt: float):
    r, g, b = mc.to_rgb(colour)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, l * amt)), s)


def _iso_block(ax, xc, length, chan, colour, yc=0.0, depth=0.30):
    h = 0.55 + (min(length, 101) / 101.0) * 3.45
    w = 0.16 + (min(chan, 128) / 128.0) * 0.80
    x0 = xc - w / 2.0
    y0 = yc - h / 2.0
    front = _shade(colour, 1.0)
    top = _shade(colour, 1.18)
    side = _shade(colour, 0.78)
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=front, edgecolor="#111111", lw=1.1, zorder=4))
    ax.add_patch(Polygon([(x0, y0 + h), (x0 + depth, y0 + h + depth),
                          (x0 + w + depth, y0 + h + depth), (x0 + w, y0 + h)],
                         closed=True, facecolor=top, edgecolor="#111111", lw=0.9, zorder=3))
    ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + depth, y0 + depth),
                          (x0 + w + depth, y0 + h + depth), (x0 + w, y0 + h)],
                         closed=True, facecolor=side, edgecolor="#111111", lw=0.9, zorder=3))
    return h, w


def _labeled_block(ax, x, yc, length, chan, colour, top_lab, shp_lab):
    h, w = _iso_block(ax, x, length, chan, colour, yc=yc)
    if top_lab:
        ax.text(x, yc + h / 2.0 + 0.42, top_lab, ha="center", va="bottom",
                fontsize=10.5, color="#111111", linespacing=1.25, fontweight="bold")
    if shp_lab:
        ax.text(x, yc - h / 2.0 - 0.42, shp_lab, ha="center", va="top",
                fontsize=10, color="#444444", style="italic")
    return h, w


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#1d3557",
                                lw=1.6, shrinkA=0, shrinkB=0), zorder=1)


def _bracket(ax, x0, x1, label, colour, y=-3.5):
    ax.plot([x0, x0, x1, x1], [y + 0.18, y, y, y + 0.18], color=colour, lw=1.6, zorder=1)
    ax.text((x0 + x1) / 2.0, y - 0.32, label, ha="center", va="top",
            fontsize=11.5, color=colour, fontweight="bold")


HERO_STAGES = {
    "conv1d_ae": (
        "Conv1D-AE  \u00b7  Symmetric Convolutional Autoencoder",
        [
            (101, 2, "#577590", "Input\n$H(s)$ response", f"101 {X} 2"),
            (50, 64, "#2a9d8f", f"Conv1D {X}64\nMish \u00b7 MaxPool", f"50 {X} 64"),
            (25, 128, "#2a9d8f", f"Conv1D {X}128\nMish \u00b7 MaxPool", f"25 {X} 128"),
            (32, 1, "#e76f51", "Dense\nlatent z", r"$z \in \mathbb{R}^{32}$"),
            (25, 128, "#2b5c8f", "Dense \u00b7 Reshape", f"25 {X} 128"),
            (50, 64, "#2b5c8f", f"UpSample \u00b7 Conv1D {X}64\nMish", f"50 {X} 64"),
            (101, 2, "#577590", f"Conv1D {X}2\nlinear", f"101 {X} 2"),
        ],
    ),
    "lstm_ae": (
        "LSTM-AE  \u00b7  Recurrent Sequence Autoencoder",
        [
            (101, 2, "#577590", "Input\n$H(s)$ response", f"101 {X} 2"),
            (101, 32, "#2a9d8f", "LSTM 32\nreturn sequences \u00b7 GELU", f"101 {X} 32"),
            (16, 16, "#2a9d8f", "LSTM 16\nlast state", "16"),
            (64, 1, "#e76f51", "Dense\nlatent z", r"$z \in \mathbb{R}^{64}$"),
            (101, 16, "#2b5c8f", f"RepeatVector {X}101", f"101 {X} 16"),
            (101, 32, "#2b5c8f", r"LSTM $16\to32$" + "\nreturn sequences", f"101 {X} 32"),
            (101, 2, "#577590", f"TimeDistributed\nDense {X}2", f"101 {X} 2"),
        ],
    ),
    "gru_ae": (
        "GRU-AE  \u00b7  Gated Recurrent Autoencoder",
        [
            (101, 2, "#577590", "Input\n$H(s)$ response", f"101 {X} 2"),
            (101, 128, "#2a9d8f", "GRU 128\nreturn sequences \u00b7 Swish", f"101 {X} 128"),
            (64, 64, "#2a9d8f", "GRU 64\nlast state", "64"),
            (32, 1, "#e76f51", "Dense\nlatent z", r"$z \in \mathbb{R}^{32}$"),
            (101, 64, "#2b5c8f", f"RepeatVector {X}101", f"101 {X} 64"),
            (101, 128, "#2b5c8f", r"GRU $64\to128$" + "\nreturn sequences", f"101 {X} 128"),
            (101, 2, "#577590", f"TimeDistributed\nDense {X}2", f"101 {X} 2"),
        ],
    ),
    "transformer_ae": (
        "Transformer-AE  \u00b7  Self-Attention Autoencoder",
        [
            (101, 2, "#577590", "Input\n$H(s)$ response", f"101 {X} 2"),
            (101, 64, "#2a9d8f", "Dense $d{=}64$\n+ Positional Enc.", f"101 {X} 64"),
            (101, 64, "#2a9d8f", f"Transformer {X}2\nMHA(4 heads) + FFN", f"101 {X} 64"),
            (64, 1, "#e76f51", "Global Avg-Pool\nDense latent z", r"$z \in \mathbb{R}^{64}$"),
            (101, 64, "#2b5c8f", "Dense \u00b7 Reshape\n+ Positional Enc.", f"101 {X} 64"),
            (101, 64, "#2b5c8f", f"Transformer {X}2\nMHA + FFN", f"101 {X} 64"),
            (101, 2, "#577590", f"Dense {X}2", f"101 {X} 2"),
        ],
    ),
}


def plot_hero_autoencoder(
    name: str = "conv1d_ae",
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot an isometric 3D stage diagram for an autoencoder architecture."""
    apply_publication_style()

    if name == "vae":
        return _plot_hero_vae(save_paths=save_paths)

    if name not in HERO_STAGES:
        raise ValueError(f"Unknown hero model '{name}'. Available: {list(HERO_STAGES.keys()) + ['vae']}")

    title, stages = HERO_STAGES[name]
    n = len(stages)

    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for i, (length, chan, colour, top_lab, shp_lab) in enumerate(stages):
        x = i * STEP
        _labeled_block(ax, x, 0.0, length, chan, colour, top_lab, shp_lab)
        if i < n - 1:
            _arrow(ax, x + 0.95, 0.15, (i + 1) * STEP - 0.95, 0.15)

    _bracket(ax, -0.95, 2 * STEP + 0.95, "Encoder  $f_{enc}$", "#2a9d8f")
    _bracket(ax, 3 * STEP - 0.75, 3 * STEP + 0.75, "Bottleneck", "#e76f51")
    _bracket(ax, 4 * STEP - 0.95, (n - 1) * STEP + 0.95, "Decoder  $f_{dec}$", "#2b5c8f")

    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#111111", pad=18)
    ax.set_xlim(-1.6, (n - 1) * STEP + 1.6)
    ax.set_ylim(-4.6, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig


def _plot_hero_vae(save_paths: Optional[Sequence[Union[str, Path]]] = None) -> plt.Figure:
    apply_publication_style()
    xs = [0.0, 2.6, 5.2, 7.9, 10.5, 13.1, 15.7, 18.3]
    x_in, x_conv, x_dense, x_dist, x_z, x_d1, x_d2, x_out = xs

    fig, ax = plt.subplots(figsize=(17, 6.9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    _labeled_block(ax, x_in, 0.0, 101, 2, "#577590", "Input\n$H(s)$ response", f"101 {X} 2")
    _labeled_block(ax, x_conv, 0.0, 25, 64, "#2a9d8f", f"Conv1D {X}32, {X}64\nSwish \u00b7 MaxPool", f"25 {X} 64")
    _labeled_block(ax, x_dense, 0.0, 64, 1, "#2a9d8f", r"Dense $128\to64$" + "\nSwish", "64")

    y_off = 1.65
    hmu, wmu = _iso_block(ax, x_dist, 32, 1, "#5b6b8a", yc=y_off)
    ax.text(x_dist, y_off + hmu / 2.0 + 0.30, r"$\mu \in \mathbb{R}^{32}$",
            ha="center", va="bottom", fontsize=10.5, color="#111111", fontweight="bold")
    hlv, wlv = _iso_block(ax, x_dist, 32, 1, "#8c5361", yc=-y_off)
    ax.text(x_dist, -y_off - hlv / 2.0 - 0.30, r"$\log\sigma^2 \in \mathbb{R}^{32}$",
            ha="center", va="top", fontsize=10.5, color="#111111", fontweight="bold")

    hz, wz = _iso_block(ax, x_z, 32, 1, "#e76f51")
    ax.text(x_z, hz / 2.0 + 0.42, "Sampling\n(reparam.)", ha="center", va="bottom",
            fontsize=10.5, color="#111111", linespacing=1.25, fontweight="bold")
    ax.text(x_z, -hz / 2.0 - 0.42, r"$z \sim \mathcal{N}(\mu,\,\sigma^2)$",
            ha="center", va="top", fontsize=10, color="#444444", style="italic")

    _labeled_block(ax, x_d1, 0.0, 25, 64, "#2b5c8f", "Dense \u00b7 Reshape", f"25 {X} 64")
    _labeled_block(ax, x_d2, 0.0, 50, 32, "#2b5c8f", f"UpSample \u00b7 Conv1D\nSwish", f"50 {X} 32")
    _labeled_block(ax, x_out, 0.0, 101, 2, "#577590", f"Conv1D {X}2\nlinear", f"101 {X} 2")

    _arrow(ax, x_in + 0.95, 0.15, x_conv - 0.95, 0.15)
    _arrow(ax, x_conv + 0.95, 0.15, x_dense - 0.95, 0.15)
    _arrow(ax, x_dense + 0.7, 0.35, x_dist - 0.7, y_off)
    _arrow(ax, x_dense + 0.7, -0.35, x_dist - 0.7, -y_off)
    _arrow(ax, x_dist + 0.7, y_off, x_z - 0.7, 0.35)
    _arrow(ax, x_dist + 0.7, -y_off, x_z - 0.7, -0.35)
    _arrow(ax, x_z + 0.95, 0.15, x_d1 - 0.95, 0.15)
    _arrow(ax, x_d1 + 0.95, 0.15, x_d2 - 0.95, 0.15)
    _arrow(ax, x_d2 + 0.95, 0.15, x_out - 0.95, 0.15)

    _bracket(ax, x_in - 0.95, x_dense + 0.95, "Encoder  $f_{enc}$", "#2a9d8f")
    _bracket(ax, x_dist - 0.95, x_z + 0.95, "Variational Bottleneck", "#e76f51")
    _bracket(ax, x_d1 - 0.95, x_out + 0.95, "Decoder  $f_{dec}$", "#2b5c8f")

    ax.set_title("VAE  \u00b7  Variational Autoencoder with Probabilistic Bottleneck",
                 loc="left", fontsize=14, fontweight="bold", color="#111111", pad=18)
    ax.set_xlim(-1.6, x_out + 1.6)
    ax.set_ylim(-4.8, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig


def plot_carla_architecture(save_paths: Optional[Sequence[Union[str, Path]]] = None) -> plt.Figure:
    """
    Render a comprehensive, publication-grade modular architecture diagram for the CARLA
    physics-informed contrastive anomaly detection framework with dual-head multi-task learning
    and edge deployment scoring engine.
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    def draw_card(x, y, w, h, title, subtitle="", badge="", fill="#ffffff", border="#1f4e79", badge_bg="#1f4e79", badge_fg="#ffffff", lw=1.6):
        # Card shadow / border
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="Round,pad=0.08,rounding_size=0.15",
            facecolor=fill, edgecolor=border, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        
        # Badge Pill
        if badge:
            bw = len(badge) * 0.12 + 0.50
            bh = 0.36
            bx = x - w / 2 + 0.15
            by = y + h / 2 - 0.18
            bp = FancyBboxPatch(
                (bx, by), bw, bh,
                boxstyle="Round,pad=0.04,rounding_size=0.08",
                facecolor=badge_bg, edgecolor="none", zorder=4
            )
            ax.add_patch(bp)
            ax.text(bx + bw / 2, by + bh / 2, badge,
                    fontsize=8.5, fontweight="bold", color=badge_fg, ha="center", va="center", zorder=5)
        
        if subtitle:
            ax.text(x, y - h / 2 + 0.30, subtitle, ha="center", va="center",
                    fontsize=9.0, color="#333333", zorder=3)
            y_title = y + 0.25
        else:
            y_title = y

        ax.text(x, y_title, title, ha="center", va="center", fontsize=10.5, fontweight="bold",
                color="#111111", zorder=3)

    def draw_arrow(x0, y0, x1, y1, label="", color="#1f4e79", lw=2.0, style="-|>", offset_y=0.22, offset_x=0.0):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3),
            zorder=4
        )
        if label:
            ax.text((x0 + x1) / 2 + offset_x, (y0 + y1) / 2 + offset_y, label,
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color=color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="none", alpha=0.85))

    # --- 1. Input Signatures (Left) ---
    draw_card(1.6, 5.0, 2.7, 1.45,
              "Anchor Sample $X$",
              r"$\mathbb{R}^{101 \times 2}$ (Real, Imag)",
              badge="Nominal Input", fill="#edf4fb", border="#1f4e79", badge_bg="#1f4e79")

    draw_card(1.6, 3.2, 2.7, 1.45,
              "Positive View $X^+$",
              r"Normal Tolerance Jitter ($\pm 20\%$)",
              badge="Positive Pair", fill="#edf4fb", border="#2b4c6f", badge_bg="#2b4c6f")

    draw_card(1.6, 1.3, 2.7, 1.65,
              r"Physics Negative $\tilde{X}$" + "\n" + r"$\tilde{X} = X \odot [\tilde{H}(s) / H^{(0)}(s)]$",
              r"Analytic Fault Drift ($C_{out}, ESR, L$)",
              badge="Hard Negative", fill="#fdf0ed", border="#b33939", badge_bg="#b33939")

    # --- 2. Shared Backbone Encoder ---
    draw_card(5.4, 3.15, 3.0, 5.4,
              "Shared Backbone $f_\\theta$\n\n" +
              "Modular Families:\n" +
              "• Conv1D (Mish + MaxPool)\n" +
              "• LSTM / GRU (2 Layers)\n" +
              "• Transformer (MHA + FFN)\n" +
              "• MLP (Dense Layers)",
              r"Maps: $\mathbb{R}^{101 \times 2} \to \mathbb{R}^{d_z}$",
              badge="Universal Encoder", fill="#edf7f5", border="#1b7a6e", badge_bg="#1b7a6e")

    # --- 3. Latent Bottleneck ---
    draw_card(8.5, 3.15, 1.8, 1.6,
              r"Latent State" + "\n" + r"$z \in \mathbb{R}^{d_z}$",
              r"$d_z \in \{32, 64\}$",
              badge="Bottleneck", fill="#fbf3e6", border="#c4842b", badge_bg="#c4842b")

    # --- 4. Dual Multi-Task Branches ---
    # Branch A: Reconstruction Decoder (Top)
    draw_card(11.8, 4.8, 2.9, 1.85,
              "Decoder $g_\\psi$ (Branch A)\nDense $\\to$ UpSample $\\to$ Conv",
              r"Reconstructed $\hat{X} \in \mathbb{R}^{101 \times 2}$",
              badge="Reconstruction Head", fill="#edf4fb", border="#1f4e79", badge_bg="#1f4e79")

    draw_card(14.7, 4.8, 2.2, 1.6,
              "Recon Loss\n$\\mathcal{L}_{\\mathrm{recon}}$",
              r"$\|X - \hat{X}\|_2^2$",
              badge="Objective A", fill="#edf4fb", border="#1f4e79", badge_bg="#1f4e79")

    # Branch B: Contrastive Projection Head (Bottom)
    draw_card(11.8, 1.5, 2.9, 1.85,
              "Projection $h_\\phi$ (Branch B)\nDense $\\to L_2$ Normalize",
              r"Unit Sphere $p \in \mathbb{S}^{d_p - 1}$",
              badge="Contrastive Head", fill="#fdf0ed", border="#b33939", badge_bg="#b33939")

    draw_card(14.7, 1.5, 2.2, 1.6,
              "InfoNCE Loss\n$\\mathcal{L}_{\\mathrm{NCE}}$",
              r"$-\log \frac{e^{\mathrm{sim}(p,p^+)/\tau}}{\sum e^{\mathrm{sim}/\tau}}$",
              badge="Objective B", fill="#fdf0ed", border="#b33939", badge_bg="#b33939")

    # --- 5. Edge Deployment & Scoring Engine (Bottom Banner) ---
    draw_card(7.8, -1.05, 14.8, 1.25,
              r"Edge Deployment & Anomaly Scoring (Jetson Nano): $\quad s(x) = \frac{1}{k} \sum_{j \in \mathrm{NN}_k(p)} \|p(x) - b_j\|_2 \quad \geq \quad \tau_{\mathrm{threshold}}$",
              "Decoder Discarded at Inference  •  TensorRT / ONNX Engine Extracted from $f_\\theta + h_\\phi$  •  Sub-millisecond Latency (0.34 ms)",
              badge="Edge Engine", fill="#f8f9fa", border="#111111", badge_bg="#111111", lw=1.8)

    # --- Arrows ---
    draw_arrow(2.95, 5.0, 3.95, 4.3, label="$X$", color="#1f4e79")
    draw_arrow(2.95, 3.2, 3.95, 3.2, label="$X^+$", color="#2b4c6f")
    draw_arrow(2.95, 1.3, 3.95, 2.1, label=r"$\tilde{X}$", color="#b33939")

    draw_arrow(6.85, 3.15, 7.60, 3.15, label="$z$", color="#1b7a6e")

    draw_arrow(9.40, 3.65, 10.35, 4.8, label="$z$", color="#1f4e79", offset_y=0.25)
    draw_arrow(9.40, 2.65, 10.35, 1.5, label="$z$", color="#b33939", offset_y=-0.25)

    draw_arrow(13.25, 4.8, 13.60, 4.8, label=r"$\hat{X}$", color="#1f4e79")
    draw_arrow(13.25, 1.5, 13.60, 1.5, label="$p$", color="#b33939")

    # Connect projection to deployment banner
    draw_arrow(11.8, 0.55, 11.8, -0.42, label="Deploy $p(x)$", color="#b33939", offset_x=0.75, offset_y=0.0)

    ax.set_xlim(-0.3, 16.3)
    ax.set_ylim(-2.0, 6.3)
    ax.axis("off")
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig
