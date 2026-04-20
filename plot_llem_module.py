"""
Generate fig_llem_module.pdf

Local Linear Enhancement Module (LLEM) of LUNA-Net.

The LLEM refines voxel-level LiDAR features by applying spatially-adaptive
local linear transformations, enabling fine-grained geometric details to be
preserved without increasing memory consumption.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plot_utils import (
    COLORS,
    draw_box,
    draw_arrow,
    draw_label,
    save_figure,
)

OUTPUT = "fig_llem_module.pdf"


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # ------------------------------------------------------------------ #
    # Input voxel feature                                                  #
    # ------------------------------------------------------------------ #
    draw_box(ax, 1.0, 2.25, 1.2, 0.65,
             label="Voxel\nFeature\n$F_{v}$",
             facecolor=COLORS["light_blue"],
             edgecolor=COLORS["blue"],
             fontsize=8)

    draw_arrow(ax, 1.6, 2.25, 2.3, 2.25)

    # ------------------------------------------------------------------ #
    # Local neighbourhood sampling                                        #
    # ------------------------------------------------------------------ #
    draw_box(ax, 3.0, 2.25, 1.2, 0.65,
             label="Local\nSampling",
             facecolor=COLORS["light_green"],
             edgecolor=COLORS["green"],
             fontsize=8)

    draw_arrow(ax, 3.6, 2.25, 4.3, 2.25)

    # ------------------------------------------------------------------ #
    # Linear coefficient predictor (upper branch)                         #
    # ------------------------------------------------------------------ #
    draw_box(ax, 5.1, 3.3, 1.3, 0.65,
             label="Coeff.\nPredictor\n$W, b$",
             facecolor=COLORS["yellow"],
             edgecolor="#9a7a00",
             fontsize=8)

    # ------------------------------------------------------------------ #
    # Local linear transform (lower branch)                               #
    # ------------------------------------------------------------------ #
    draw_box(ax, 5.1, 1.2, 1.3, 0.65,
             label="Local\nLinear\nTransform",
             facecolor=COLORS["light_blue"],
             edgecolor=COLORS["blue"],
             fontsize=8)

    # Arrows from sampling to both branches
    draw_arrow(ax, 4.6, 2.25, 4.6, 3.3)
    ax.annotate("", xy=(4.45, 3.3), xytext=(4.6, 3.3),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=1.2))
    draw_arrow(ax, 4.6, 2.25, 4.6, 1.2)
    ax.annotate("", xy=(4.45, 1.2), xytext=(4.6, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=1.2))

    # Arrows from branches to aggregation
    draw_arrow(ax, 5.75, 3.3, 6.6, 2.55)
    draw_arrow(ax, 5.75, 1.2, 6.6, 1.95)

    # ------------------------------------------------------------------ #
    # Aggregation (element-wise multiply + add)                          #
    # ------------------------------------------------------------------ #
    draw_box(ax, 7.1, 2.25, 0.8, 0.65,
             label="$\\odot + $",
             facecolor=COLORS["orange"],
             edgecolor="#8b1a1a",
             fontsize=11)

    draw_arrow(ax, 7.5, 2.25, 8.2, 2.25)

    # ------------------------------------------------------------------ #
    # Residual addition                                                    #
    # ------------------------------------------------------------------ #
    draw_box(ax, 8.7, 2.25, 0.7, 0.65,
             label="$+$",
             facecolor=COLORS["orange"],
             edgecolor="#8b1a1a",
             fontsize=11)

    # Skip connection from input
    ax.annotate("",
                xy=(8.35, 2.25), xytext=(1.0, 2.25),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=COLORS["gray"],
                    lw=1.0,
                    connectionstyle="arc3,rad=-0.35",
                ))

    draw_arrow(ax, 9.05, 2.25, 9.6, 2.25)

    # ------------------------------------------------------------------ #
    # Output                                                              #
    # ------------------------------------------------------------------ #
    draw_label(ax, 9.8, 2.25, "$\\hat{F}_{v}$",
               fontsize=12, color=COLORS["blue"])

    # ------------------------------------------------------------------ #
    # Labels & title                                                      #
    # ------------------------------------------------------------------ #
    ax.set_title("Local Linear Enhancement Module (LLEM)",
                 fontsize=11, fontweight="bold", pad=8)

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["light_blue"],  edgecolor=COLORS["blue"],  label="Feature Tensor"),
        mpatches.Patch(facecolor=COLORS["light_green"], edgecolor=COLORS["green"], label="Sampling Op"),
        mpatches.Patch(facecolor=COLORS["yellow"],      edgecolor="#9a7a00",        label="Coefficient Pred."),
        mpatches.Patch(facecolor=COLORS["orange"],      edgecolor="#8b1a1a",        label="Element-wise Op"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
              framealpha=0.9)

    save_figure(fig, OUTPUT)


if __name__ == "__main__":
    main()
