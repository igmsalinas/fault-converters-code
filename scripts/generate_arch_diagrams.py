"""
Generate deep-learning architecture diagrams for every model in the suite.

Uses ``keras.utils.model_to_dot`` (Graphviz) to render, for each model, a
node/edge graph annotated with layer types, activations and input/output tensor
shapes. Nodes are recoloured with the presentation palette so the figures match
the CMMSE deck.

Models are rebuilt from each experiment's ``model_config.json`` using the exact
same hyper-parameter resolution as ``src/inference/predictor.py`` (no weights
are loaded - only the architecture is needed).

Run:  uv run python scripts/generate_arch_diagrams.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import keras
from keras import Model

from src.inference.predictor import MODEL_CLASSES
from src.models.contrastive_ae import ContrastiveAutoencoder
from src.models.gru_ae import GRUAutoencoder
from src.training.hyperparameter_search import SearchSpace
from src.visualization.deck_style import CATEGORICAL, INK, PAPER, text_on

# predictor.MODEL_CLASSES omits GRU; complete the map here.
MODEL_CLASSES = {**MODEL_CLASSES, "gru_ae": GRUAutoencoder}

EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

SS = SearchSpace()

# --- palette assignment by layer family -------------------------------------
C = CATEGORICAL
LAYER_COLORS = [
    (("inputlayer", "input_layer"), INK),
    (("conv",), C[1]),          # terracotta   - convolution
    (("dense", "latent", "projection"), C[2]),  # slate teal - linear / bottleneck
    (("lstm", "gru", "rnn", "bidirectional"), C[3]),  # ochre - recurrent
    (("attention", "multihead", "transformer", "positional"), C[5]),  # dusty plum - attention
    (("activation", "mish", "relu", "gelu", "swish", "tanh"), C[6]),  # pine - activation
    (("normalization", "batchnorm", "layernorm"), C[8]),  # muted indigo - norm
    (("dropout",), C[7]),       # warm grey - regularisation
    (("pool", "upsampling", "reshape", "flatten", "cropping",
      "zeropadding", "lambda", "add", "concat", "sampling"), C[4]),  # sage - shape ops
    (("output",), INK),
]
DEFAULT_COLOR = C[7]


def _color_for(label: str) -> str:
    low = label.lower()
    for keys, colour in LAYER_COLORS:
        if any(k in low for k in keys):
            return colour
    return DEFAULT_COLOR


def _style_nodes(graph) -> None:
    """Recursively recolour every node in a pydot graph / cluster."""
    for sub in graph.get_subgraphs():
        sub.set("style", "rounded")
        sub.set("color", "#c9bfb0")
        sub.set("bgcolor", "#efe9e0")
        sub.set("fontname", "Helvetica-Bold")
        sub.set("fontcolor", INK)
        _style_nodes(sub)
    for node in graph.get_nodes():
        name = node.get_name().strip('"')
        if name in ("node", "graph", "edge"):
            continue
        label = node.get_label() or ""
        colour = _color_for(label)
        node.set("style", "filled,rounded")
        node.set("shape", "box")
        node.set("fillcolor", colour)
        node.set("color", INK)
        node.set("penwidth", "1.2")
        node.set("fontname", "Helvetica")
        node.set("fontcolor", text_on(colour))


def build_model(cfg: dict):
    name = cfg.get("name", "conv1d_ae")
    input_shape = tuple(cfg["input_shape"])
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "type", "input_shape")}

    if "filters_idx" in kwargs:
        kwargs["filters"] = SS.filter_options[kwargs.pop("filters_idx")]
    if "units_idx" in kwargs:
        idx = kwargs.pop("units_idx")
        if "lstm" in name:
            kwargs["lstm_units"] = SS.lstm_unit_options[idx]
        elif "gru" in name:
            kwargs["gru_units"] = SS.gru_unit_options[idx]

    is_carla = cfg.get("type") == "carla" or name == "carla"
    cls = ContrastiveAutoencoder if is_carla else MODEL_CLASSES[name]

    valid = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in valid}

    model = cls(input_shape=input_shape, **kwargs)
    model.build()

    if is_carla:
        inp = keras.Input(shape=input_shape, name="input")
        z = model.encoder(inp)
        rec = model.decoder(z)
        proj = model.projection_head(z)
        plot_target = Model(inp, [rec, proj], name="carla")
    else:
        plot_target = model.autoencoder
    return model, plot_target


def render_graph(plot_target, name: str) -> Path:
    dot = keras.utils.model_to_dot(
        plot_target,
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        expand_nested=True,
        rankdir="LR",
        dpi=200,
    )
    dot.set("bgcolor", PAPER)
    dot.set("fontname", "Helvetica")
    _style_nodes(dot)
    path = OUT / f"arch_diagram_{name}.png"
    dot.write_png(str(path))
    print(f"  wrote {path}")
    return path


# --- polished "hero" isometric diagram --------------------------------------
def _shade(colour: str, amt: float):
    import colorsys
    import matplotlib.colors as mc

    r, g, b = mc.to_rgb(colour)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, l * amt)), s)


def _iso_block(ax, xc, length, chan, colour, yc=0.0, depth=0.30):
    """Draw a parallelepiped whose height encodes sequence length and whose
    front-face width encodes channel count. Returns (height, width)."""
    from matplotlib.patches import Polygon, Rectangle

    h = 0.55 + (min(length, 101) / 101.0) * 3.45
    w = 0.16 + (min(chan, 128) / 128.0) * 0.80
    x0 = xc - w / 2.0
    y0 = yc - h / 2.0
    front = _shade(colour, 1.0)
    top = _shade(colour, 1.18)
    side = _shade(colour, 0.78)
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=front, edgecolor=INK, lw=1.1, zorder=4))
    ax.add_patch(Polygon([(x0, y0 + h), (x0 + depth, y0 + h + depth),
                          (x0 + w + depth, y0 + h + depth), (x0 + w, y0 + h)],
                         closed=True, facecolor=top, edgecolor=INK, lw=0.9, zorder=3))
    ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + depth, y0 + depth),
                          (x0 + w + depth, y0 + h + depth), (x0 + w, y0 + h)],
                         closed=True, facecolor=side, edgecolor=INK, lw=0.9, zorder=3))
    return h, w


def _labeled_block(ax, x, yc, length, chan, colour, top_lab, shp_lab):
    h, w = _iso_block(ax, x, length, chan, colour, yc=yc)
    if top_lab:
        ax.text(x, yc + h / 2.0 + 0.42, top_lab, ha="center", va="bottom",
                fontsize=10.5, color=INK, linespacing=1.25)
    if shp_lab:
        ax.text(x, yc - h / 2.0 - 0.42, shp_lab, ha="center", va="top",
                fontsize=10, color=_shade(INK, 1.9), style="italic")
    return h, w


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=_shade(INK, 1.6),
                                lw=1.6, shrinkA=0, shrinkB=0), zorder=1)


def _bracket(ax, x0, x1, label, colour, y=-3.5):
    ax.plot([x0, x0, x1, x1], [y + 0.18, y, y, y + 0.18], color=colour, lw=1.6, zorder=1)
    ax.text((x0 + x1) / 2.0, y - 0.32, label, ha="center", va="top",
            fontsize=11.5, color=colour, fontweight="bold")


STEP = 2.6

C = CATEGORICAL
IO_C, ENC_C, LAT_C, DEC_C = C[7], C[2], C[1], C[3]
MU_C, LV_C = C[8], C[5]  # muted indigo / dusty plum for the VAE heads

X = "\u00d7"

# (length, channels, colour, top-label, shape-label) per stage, left -> right.
HERO_STAGES = {
    "conv1d_ae": (
        "Conv1D-AE  \u00b7  symmetric convolutional autoencoder",
        [
            (101, 2, IO_C, "Input\n$H(s)$ response", f"101 {X} 2"),
            (50, 64, ENC_C, f"Conv1D {X}64\nMish \u00b7 MaxPool", f"50 {X} 64"),
            (25, 128, ENC_C, f"Conv1D {X}128\nMish \u00b7 MaxPool", f"25 {X} 128"),
            (32, 1, LAT_C, "Dense\nlatent z", r"$z \in \mathbb{R}^{32}$"),
            (25, 128, DEC_C, "Dense \u00b7 Reshape", f"25 {X} 128"),
            (50, 64, DEC_C, f"UpSample \u00b7 Conv1D {X}64\nMish", f"50 {X} 64"),
            (101, 2, IO_C, f"Conv1D {X}2\nlinear", f"101 {X} 2"),
        ],
    ),
    "lstm_ae": (
        "LSTM-AE  \u00b7  recurrent sequence autoencoder",
        [
            (101, 2, IO_C, "Input\n$H(s)$ response", f"101 {X} 2"),
            (101, 32, ENC_C, "LSTM 32\nreturn sequences \u00b7 GELU", f"101 {X} 32"),
            (16, 16, ENC_C, "LSTM 16\nlast state", "16"),
            (64, 1, LAT_C, "Dense\nlatent z", r"$z \in \mathbb{R}^{64}$"),
            (101, 16, DEC_C, f"RepeatVector {X}101", f"101 {X} 16"),
            (101, 32, DEC_C, r"LSTM $16\to32$" + "\nreturn sequences", f"101 {X} 32"),
            (101, 2, IO_C, f"TimeDistributed\nDense {X}2", f"101 {X} 2"),
        ],
    ),
    "gru_ae": (
        "GRU-AE  \u00b7  gated recurrent autoencoder",
        [
            (101, 2, IO_C, "Input\n$H(s)$ response", f"101 {X} 2"),
            (101, 128, ENC_C, "GRU 128\nreturn sequences \u00b7 Swish", f"101 {X} 128"),
            (64, 64, ENC_C, "GRU 64\nlast state", "64"),
            (32, 1, LAT_C, "Dense\nlatent z", r"$z \in \mathbb{R}^{32}$"),
            (101, 64, DEC_C, f"RepeatVector {X}101", f"101 {X} 64"),
            (101, 128, DEC_C, r"GRU $64\to128$" + "\nreturn sequences", f"101 {X} 128"),
            (101, 2, IO_C, f"TimeDistributed\nDense {X}2", f"101 {X} 2"),
        ],
    ),
    "transformer_ae": (
        "Transformer-AE  \u00b7  self-attention autoencoder",
        [
            (101, 2, IO_C, "Input\n$H(s)$ response", f"101 {X} 2"),
            (101, 64, ENC_C, "Dense $d{=}64$\n+ positional enc.", f"101 {X} 64"),
            (101, 64, ENC_C, f"Transformer {X}2\nMHA(4 heads) + FFN \u00b7 Mish", f"101 {X} 64"),
            (64, 1, LAT_C, "Global avg-pool\nDense latent z", r"$z \in \mathbb{R}^{64}$"),
            (101, 64, DEC_C, "Dense \u00b7 Reshape\n+ positional enc.", f"101 {X} 64"),
            (101, 64, DEC_C, f"Transformer {X}2\nMHA + FFN", f"101 {X} 64"),
            (101, 2, IO_C, f"Dense {X}2", f"101 {X} 2"),
        ],
    ),
}


def render_hero_linear(name: str) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
    })

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
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        path = OUT / f"arch_hero_{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        # Also copy directly to paper figures
        paper_dest = ROOT / "papers" / "mmas_anomaly_detection" / "figures" / f"arch_hero_{name}.{suffix}"
        paper_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, paper_dest)
    plt.close(fig)
    print(f"  wrote arch_hero_{name}.png and .pdf")


def render_carla_modular_diagram() -> None:
    """Render a clean, publication-quality modular block diagram for the CARLA architecture."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, ArrowStyle

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
    })

    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    def draw_box(x, y, w, h, title, subtitle="", badge="", fill="#f8f9fa", border="#1d3557", title_color="#111111"):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="Round,pad=0.2,rounding_size=0.15",
            facecolor=fill, edgecolor=border, linewidth=1.5, zorder=2
        )
        ax.add_patch(box)
        if badge:
            badge_box = FancyBboxPatch(
                (x - w / 2 + 0.1, y + h / 2 - 0.35), len(badge) * 0.16 + 0.3, 0.4,
                boxstyle="Round,pad=0.08,rounding_size=0.08",
                facecolor=border, edgecolor="none", zorder=3
            )
            ax.add_patch(badge_box)
            ax.text(x - w / 2 + 0.25, y + h / 2 - 0.18, badge,
                    fontsize=8.5, fontweight="bold", color="#ffffff", va="center")
        
        y_text = y + 0.15 if subtitle else y
        ax.text(x, y_text, title, ha="center", va="center", fontsize=11, fontweight="bold",
                color=title_color, zorder=3)
        if subtitle:
            ax.text(x, y - 0.35, subtitle, ha="center", va="center", fontsize=9.5,
                    color="#444444", style="italic", zorder=3)

    def draw_arrow(x0, y0, x1, y1, label="", color="#1d3557", lw=1.8, style="-|>", offset_y=0.25):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3),
            zorder=4
        )
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + offset_y, label,
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=color, zorder=5)

    # 1. Inputs (Left)
    draw_box(1.5, 4.2, 2.4, 1.4, "Real Response $x$\n(Nominal / Real)", r"$\mathbb{R}^{101 \times 2}$ (Mag, Phase)",
             badge="Input", fill="#e9ecef", border="#495057")
    draw_box(1.5, 1.6, 2.4, 1.4, "Physics Negative $\\tilde{x}$\n(Analytic Perturb.)", r"$\mathbb{R}^{101 \times 2}$ ($\pm$ Degradation)",
             badge="Synthesis", fill="#ffe8d6", border="#dd6b20")

    # 2. Shared Conv1D Encoder
    draw_box(5.0, 2.9, 2.6, 3.8, "Shared Conv1D\nEncoder $f_{\\mathrm{enc}}$",
             "Conv1D (64, 128)\nMish Act. + MaxPool\nDense Bottleneck",
             badge="Backbone", fill="#e2f0cb", border="#2a9d8f")

    # 3. Latent Representation z
    draw_box(8.0, 2.9, 1.8, 1.4, "Latent $z$", r"$z \in \mathbb{R}^{32}$",
             badge="Bottleneck", fill="#fed9b7", border="#f4a261")

    # 4. Upper Branch: Reconstruction Decoder
    draw_box(11.2, 4.4, 2.6, 1.8, "Reconstruction\nDecoder $f_{\\mathrm{dec}}$",
             "Dense $\\to$ Reshape\nUpSample + Conv1D\nOutput $\\hat{x} \\in \\mathbb{R}^{101 \\times 2}$",
             badge="Branch A", fill="#d8e2dc", border="#2b5c8f")
    draw_box(14.4, 4.4, 2.0, 1.4, "Recon. Loss\n$\\mathcal{L}_{\\mathrm{recon}}$",
             r"$\|x - \hat{x}\|_2^2$",
             badge="Objective", fill="#edf2f4", border="#1d3557")

    # 5. Lower Branch: Contrastive Projection Head
    draw_box(11.2, 1.5, 2.6, 1.8, "Contrastive\nProjection $g_{\\mathrm{proj}}$",
             "Dense (128 $\\to$ 64/16)\nL2 Normalization\nOutput $p \\in \\mathbb{R}^{16}$",
             badge="Branch B", fill="#fcd5ce", border="#e76f51")
    draw_box(14.4, 1.5, 2.2, 1.4, "Contrastive Loss\n$\\mathcal{L}_{\\mathrm{contrast}}$",
             "InfoNCE / Triplet\nvs. Reference Bank $\\mathcal{B}$",
             badge="Objective", fill="#feeafa", border="#c77dff")

    # 6. Combined Inference Anomaly Score (Bottom Banner)
    draw_box(8.0, -0.6, 14.8, 1.1,
             r"$\mathbf{Inference\ Anomaly\ Score:}\ S(x) = \alpha \cdot \mathrm{MSE}(x, \hat{x}) + (1-\alpha) \cdot \min_{b \in \mathcal{B}} \|p(x) - b\|_2 \quad \gtrless \quad \tau\ \ (\mathrm{Threshold})$",
             subtitle="Self-supervised feature learning + Physics-informed boundary regularization",
             badge="Deployment", fill="#f1f3f5", border="#1d3557")

    # Arrows linking nodes
    draw_arrow(2.7, 4.2, 3.7, 3.8, label="Real $x$", color="#1d3557")
    draw_arrow(2.7, 1.6, 3.7, 2.0, label="Synth $\\tilde{x}$", color="#dd6b20")
    draw_arrow(6.3, 2.9, 7.1, 2.9, label="", color="#2a9d8f")
    
    # Split from z to Decoder & Projection
    draw_arrow(8.9, 3.3, 9.9, 4.4, label="Latent $z$", color="#2b5c8f", offset_y=0.3)
    draw_arrow(8.9, 2.5, 9.9, 1.5, label="Latent $z$", color="#e76f51", offset_y=-0.3)

    draw_arrow(12.5, 4.4, 13.4, 4.4, label="$\\hat{x}$", color="#2b5c8f")
    draw_arrow(12.5, 1.5, 13.3, 1.5, label="$p(x)$", color="#e76f51")

    # Scoring feed lines (dotted arrows to bottom banner)
    draw_arrow(11.2, 3.5, 11.2, 0.0, label="MSE", color="#2b5c8f", style="-|>", lw=1.4)
    draw_arrow(11.2, 0.6, 11.2, 0.0, label="Embedding Distance", color="#e76f51", style="-|>", lw=1.4)

    ax.set_xlim(-0.2, 16.0)
    ax.set_ylim(-1.4, 5.8)
    ax.axis("off")
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        path = OUT / f"arch_diagram_carla.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paper_dest = ROOT / "papers" / "mmas_anomaly_detection" / "figures" / f"arch_diagram_carla.{suffix}"
        paper_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, paper_dest)
    plt.close(fig)
    print("  wrote arch_diagram_carla.png and .pdf")


def render_hero_vae(name: str = "vae") -> None:
    """Custom hero with the probabilistic (mu, log-sigma^2, sampled-z) bottleneck."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
    })

    xs = [0.0, 2.6, 5.2, 7.9, 10.5, 13.1, 15.7, 18.3]
    x_in, x_conv, x_dense, x_dist, x_z, x_d1, x_d2, x_out = xs

    fig, ax = plt.subplots(figsize=(17, 6.9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # encoder chain
    _labeled_block(ax, x_in, 0.0, 101, 2, IO_C, "Input\n$H(s)$ response", f"101 {X} 2")
    _labeled_block(ax, x_conv, 0.0, 25, 64, ENC_C,
                   f"Conv1D {X}32, {X}64\nSwish \u00b7 MaxPool", f"25 {X} 64")
    _labeled_block(ax, x_dense, 0.0, 64, 1, ENC_C, r"Dense $128\to64$" + "\nSwish", "64")

    # probabilistic bottleneck: mu (top) and log-sigma^2 (bottom) -> sampled z
    y_off = 1.65
    hmu, wmu = _iso_block(ax, x_dist, 32, 1, MU_C, yc=y_off)
    ax.text(x_dist, y_off + hmu / 2.0 + 0.30, r"$\mu \in \mathbb{R}^{32}$",
            ha="center", va="bottom", fontsize=10.5, color=INK)
    hlv, wlv = _iso_block(ax, x_dist, 32, 1, LV_C, yc=-y_off)
    ax.text(x_dist, -y_off - hlv / 2.0 - 0.30, r"$\log\sigma^2 \in \mathbb{R}^{32}$",
            ha="center", va="top", fontsize=10.5, color=INK)

    hz, wz = _iso_block(ax, x_z, 32, 1, LAT_C)
    ax.text(x_z, hz / 2.0 + 0.42, "Sampling\n(reparam.)", ha="center", va="bottom",
            fontsize=10.5, color=INK, linespacing=1.25)
    ax.text(x_z, -hz / 2.0 - 0.42, r"$z \sim \mathcal{N}(\mu,\,\sigma^2)$",
            ha="center", va="top", fontsize=10, color=_shade(INK, 1.9), style="italic")

    # decoder chain
    _labeled_block(ax, x_d1, 0.0, 25, 64, DEC_C, "Dense \u00b7 Reshape", f"25 {X} 64")
    _labeled_block(ax, x_d2, 0.0, 50, 32, DEC_C, f"UpSample \u00b7 Conv1D\nSwish", f"50 {X} 32")
    _labeled_block(ax, x_out, 0.0, 101, 2, IO_C, f"Conv1D {X}2\nlinear", f"101 {X} 2")

    # arrows
    _arrow(ax, x_in + 0.95, 0.15, x_conv - 0.95, 0.15)
    _arrow(ax, x_conv + 0.95, 0.15, x_dense - 0.95, 0.15)
    _arrow(ax, x_dense + 0.7, 0.35, x_dist - 0.7, y_off)      # -> mu
    _arrow(ax, x_dense + 0.7, -0.35, x_dist - 0.7, -y_off)    # -> log-sigma^2
    _arrow(ax, x_dist + 0.7, y_off, x_z - 0.7, 0.35)          # mu -> z
    _arrow(ax, x_dist + 0.7, -y_off, x_z - 0.7, -0.35)        # log-sigma^2 -> z
    _arrow(ax, x_z + 0.95, 0.15, x_d1 - 0.95, 0.15)
    _arrow(ax, x_d1 + 0.95, 0.15, x_d2 - 0.95, 0.15)
    _arrow(ax, x_d2 + 0.95, 0.15, x_out - 0.95, 0.15)

    _bracket(ax, x_in - 0.95, x_dense + 0.95, "Encoder  $f_{enc}$", "#2a9d8f")
    _bracket(ax, x_dist - 0.95, x_z + 0.95, "Variational bottleneck", "#e76f51")
    _bracket(ax, x_d1 - 0.95, x_out + 0.95, "Decoder  $f_{dec}$", "#2b5c8f")

    ax.set_title("VAE  \u00b7  variational autoencoder with probabilistic bottleneck",
                 loc="left", fontsize=14, fontweight="bold", color="#111111", pad=18)
    ax.set_xlim(-1.6, x_out + 1.6)
    ax.set_ylim(-4.8, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    path = OUT / f"arch_hero_{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


from src.visualization.architecture import (
    plot_carla_architecture,
    plot_hero_autoencoder,
    HERO_MODELS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        help=(
            "Experiment root containing carla_* model directories. When set, "
            "diagrams are built only from those persisted final configurations."
        ),
    )
    args = parser.parse_args()

    # 1. Render publication CARLA modular block diagram
    plot_carla_architecture(
        save_paths=[
            OUT / "arch_diagram_carla.png",
            OUT / "arch_diagram_carla.pdf",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "arch_diagram_carla.png",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "arch_diagram_carla.pdf",
        ]
    )

    # 2. Render hero isometric stage diagrams
    for hero_name in HERO_MODELS:
        save_paths = [
            OUT / f"arch_hero_{hero_name}.png",
            OUT / f"arch_hero_{hero_name}.pdf",
        ]
        if hero_name in ["conv1d_ae", "gru_ae", "lstm_ae", "transformer_ae"]:
            save_paths.extend([
                ROOT / "papers" / "mmas_anomaly_detection" / "figures" / f"arch_hero_{hero_name}.png",
                ROOT / "papers" / "mmas_anomaly_detection" / "figures" / f"arch_hero_{hero_name}.pdf",
            ])
        plot_hero_autoencoder(hero_name, save_paths=save_paths)

    if args.experiment:
        experiment = Path(args.experiment)
        model_dirs = sorted(
            path for path in experiment.glob("carla_*")
            if (path / "model_config.json").exists()
        )
        if not model_dirs:
            raise FileNotFoundError(
                f"No completed CARLA model configurations under {experiment}"
            )
        for model_dir in model_dirs:
            name = model_dir.name
            print(f"[{name}]")
            cfg = json.loads((model_dir / "model_config.json").read_text())
            try:
                model, plot_target = build_model(cfg)
                render_graph(plot_target, name)
            except Exception as e:
                print(f"  Graphviz dot rendering skipped for {name}: {e}")
        return

    models = ["conv1d_ae", "lstm_ae", "gru_ae", "mlp_ae", "transformer_ae", "vae", "carla"]
    for name in models:
        cfg_path = EXP / name / "model_config.json"
        if not cfg_path.exists():
            print(f"skip {name}: no config")
        else:
            print(f"[{name}]")
            cfg = json.loads(cfg_path.read_text())
            try:
                model, plot_target = build_model(cfg)
                render_graph(plot_target, name)
            except Exception as e:
                print(f"  Graphviz dot rendering skipped for {name}: {e}")


if __name__ == "__main__":
    main()

