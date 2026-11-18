"""
Unified Styling and Color System for Visualizations
===================================================

Provides publication-grade and presentation-deck styling themes, color palettes,
luminance-based adaptive contrast calculations, and figure saving utilities.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# --- Presentation Deck Palette -----------------------------------------------
PAPER = "#f7f4ef"
PAPER_SOFT = "#fbf9f5"
INK = "#121212"
INK_SOFT = "#43413d"
MUTED = "#8a837b"
LINE = "#e4ded4"
LINE_STRONG = "#c9c1b5"

# --- Professional Academic Palette --------------------------------------------
PUBLICATION_PALETTE: List[str] = [
    "#1f4e79",  # Classic Navy
    "#2e7d6e",  # Deep Teal
    "#b85d38",  # Academic Terracotta
    "#c4842b",  # Warm Ochre
    "#4a7c9f",  # Steel Blue
    "#b33939",  # Burgundy Red
    "#5c6b73",  # Slate Gray
    "#6b4c9a",  # Muted Indigo
    "#3f5a52",  # Pine Green
    "#8d6e63",  # Muted Earth
]

RUNTIME_COLORS = {
    "Keras CPU": "#2b4c6f",        # Navy Slate
    "Keras GPU FP32": "#4a7c9f",   # Medium Steel Blue
    "Keras GPU FP16": "#7390a8",   # Muted Slate Blue
    "TFLite FP32": "#1b7a6e",      # Deep Forest Teal
    "TFLite FP16": "#40a597",      # Sage Green
    "ONNX FP32": "#c4842b",        # Warm Ochre
    "ONNX FP16": "#dfa759",        # Muted Sandstone
    "TensorRT FP32": "#b33939",    # Deep Burgundy Red
    "TensorRT FP16": "#d9534f",    # Rich Brick Red
}


def apply_publication_style() -> None:
    """Apply high-readability IEEE/Elsevier publication style to matplotlib rcParams."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": "#ffffff",
        "figure.edgecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#111111",
        "axes.labelsize": 11.0,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.prop_cycle": cycler(color=PUBLICATION_PALETTE),
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "grid.color": "#dcdcdc",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "savefig.facecolor": "#ffffff",
        "savefig.edgecolor": "#ffffff",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


def apply_deck_style() -> None:
    """Apply presentation slide deck style to matplotlib rcParams."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": PAPER,
        "figure.edgecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": LINE_STRONG,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SOFT,
        "axes.prop_cycle": cycler(color=PUBLICATION_PALETTE),
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "grid.color": LINE,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "savefig.dpi": 300,
    })


def adaptive_contrast(
    value: float,
    vmin: float,
    vmax: float,
    cmap: Union[str, mpl.colors.Colormap] = "viridis",
    light_text: str = "#ffffff",
    dark_text: str = "#111111",
) -> str:
    """Calculate adaptive text color for maximum contrast over colormaps based on luminance."""
    if not np.isfinite(value) or vmax <= vmin:
        return dark_text
    norm_val = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    rgba = cm(norm_val)
    # ITU-R BT.709 perceived luminance
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return dark_text if luminance > 0.48 else light_text


def style_axis(ax: plt.Axes) -> None:
    """Apply standard clean borders and gridlines to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")


def save_figure(
    fig: plt.Figure,
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
    dpi: int = 300,
) -> None:
    """Save a matplotlib figure to multiple paths (PNG/PDF/SVG)."""
    if not save_paths:
        return
    for path_str in save_paths:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"dpi": dpi} if p.suffix.lower() == ".png" else {}
        fig.savefig(p, bbox_inches="tight", **kwargs)
