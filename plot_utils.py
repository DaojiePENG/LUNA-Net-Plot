"""
Shared drawing utilities for LUNA-Net architecture figures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# ---------------------------------------------------------------------------
# Color palette (consistent across all figures)
# ---------------------------------------------------------------------------
COLORS = {
    "blue":       "#4C72B0",
    "light_blue": "#AEC6E8",
    "green":      "#55A868",
    "light_green":"#B5D9C2",
    "orange":     "#C44E52",
    "yellow":     "#CCB974",
    "purple":     "#8172B2",
    "gray":       "#8C8C8C",
    "light_gray": "#D9D9D9",
    "white":      "#FFFFFF",
    "black":      "#000000",
}

FONT_SIZE_LABEL  = 9
FONT_SIZE_TITLE  = 10
FONT_FAMILY      = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def draw_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str = "",
    facecolor: str = COLORS["light_blue"],
    edgecolor: str = COLORS["blue"],
    fontsize: int = FONT_SIZE_LABEL,
    linewidth: float = 1.2,
    style: str = "round,pad=0.05",
    zorder: int = 3,
) -> FancyBboxPatch:
    """Draw a rounded rectangle and annotate it with *label*."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=style,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    if label:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily=FONT_FAMILY,
            zorder=zorder + 1,
            wrap=True,
        )
    return box


def draw_arrow(
    ax,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    color: str = COLORS["gray"],
    linewidth: float = 1.2,
    arrowstyle: str = "-|>",
    mutation_scale: float = 12,
    zorder: int = 2,
) -> FancyArrowPatch:
    """Draw a directed arrow between two points."""
    arrow = FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle=arrowstyle,
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_label(
    ax,
    x: float,
    y: float,
    text: str,
    fontsize: int = FONT_SIZE_LABEL,
    color: str = COLORS["black"],
    ha: str = "center",
    va: str = "center",
    zorder: int = 4,
) -> None:
    """Draw a standalone text label."""
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontfamily=FONT_FAMILY,
        color=color,
        zorder=zorder,
    )


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------

def draw_feature_map(
    ax,
    x: float,
    y: float,
    width: float = 0.4,
    height: float = 0.6,
    depth: float = 0.12,
    color: str = COLORS["light_blue"],
    edgecolor: str = COLORS["blue"],
    label: str = "",
    fontsize: int = FONT_SIZE_LABEL,
) -> None:
    """Draw a 3-D-style feature-map cuboid."""
    from matplotlib.patches import Polygon

    # Front face
    front = plt.Polygon(
        [
            [x,              y - height / 2],
            [x + width,      y - height / 2],
            [x + width,      y + height / 2],
            [x,              y + height / 2],
        ],
        closed=True,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(front)

    # Top face
    top = plt.Polygon(
        [
            [x,              y + height / 2],
            [x + width,      y + height / 2],
            [x + width + depth, y + height / 2 + depth],
            [x + depth,      y + height / 2 + depth],
        ],
        closed=True,
        facecolor=_darken(color, 0.85),
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(top)

    # Right face
    right = plt.Polygon(
        [
            [x + width,      y - height / 2],
            [x + width + depth, y - height / 2 + depth],
            [x + width + depth, y + height / 2 + depth],
            [x + width,      y + height / 2],
        ],
        closed=True,
        facecolor=_darken(color, 0.70),
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(right)

    if label:
        ax.text(
            x + width / 2,
            y - height / 2 - 0.08,
            label,
            ha="center",
            va="top",
            fontsize=fontsize,
            fontfamily=FONT_FAMILY,
            zorder=4,
        )


def _darken(hex_color: str, factor: float) -> str:
    """Return a darkened version of *hex_color*."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def save_figure(fig: plt.Figure, path: str) -> None:
    """Save *fig* to *path* with print-quality settings."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)
