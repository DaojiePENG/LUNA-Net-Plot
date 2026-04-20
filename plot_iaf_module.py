"""
Generate fig_iaf_module.pdf

Instance-Aware Feature (IAF) module of LUNA-Net.

The IAF module fuses multi-scale image features with instance-level attention
to produce spatially calibrated feature representations for each detected
object proposal.
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

OUTPUT = "fig_iaf_module.pdf"


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # ------------------------------------------------------------------ #
    # Backbone feature maps  (left column)                                #
    # ------------------------------------------------------------------ #
    scales = [
        (0.9, 3.2, "P3\n(stride 8)"),
        (0.9, 2.2, "P4\n(stride 16)"),
        (0.9, 1.2, "P5\n(stride 32)"),
    ]
    for x, y, lbl in scales:
        draw_box(ax, x, y, 1.1, 0.55,
                 label=lbl,
                 facecolor=COLORS["light_blue"],
                 edgecolor=COLORS["blue"],
                 fontsize=8)
        draw_arrow(ax, x + 0.55, y, x + 1.35, y)

    draw_label(ax, 0.9, 3.75, "Backbone\nFeatures",
               fontsize=8, color=COLORS["gray"])

    # ------------------------------------------------------------------ #
    # RoI-Align                                                           #
    # ------------------------------------------------------------------ #
    roi_y_positions = [3.2, 2.2, 1.2]
    for y in roi_y_positions:
        draw_box(ax, 2.2, y, 1.2, 0.55,
                 label="RoI-Align",
                 facecolor=COLORS["light_green"],
                 edgecolor=COLORS["green"],
                 fontsize=8)
        draw_arrow(ax, 2.8, y, 3.5, y)

    # ------------------------------------------------------------------ #
    # Instance Attention Gate                                              #
    # ------------------------------------------------------------------ #
    draw_box(ax, 4.3, 2.2, 1.4, 1.9,
             label="Instance\nAttention\nGate",
             facecolor=COLORS["yellow"],
             edgecolor="#9a7a00",
             fontsize=8)
    draw_arrow(ax, 5.0, 2.2, 5.8, 2.2)

    # ------------------------------------------------------------------ #
    # Feature Aggregation                                                  #
    # ------------------------------------------------------------------ #
    draw_box(ax, 6.5, 2.2, 1.2, 0.7,
             label="Concat &\nConv 1×1",
             facecolor=COLORS["purple"],
             edgecolor="#5a3d99",
             fontsize=8)
    draw_arrow(ax, 7.1, 2.2, 7.65, 2.2)

    # ------------------------------------------------------------------ #
    # Output                                                              #
    # ------------------------------------------------------------------ #
    draw_label(ax, 7.85, 2.2, "IAF\nFeature",
               fontsize=9, color=COLORS["blue"])

    # ------------------------------------------------------------------ #
    # Title                                                               #
    # ------------------------------------------------------------------ #
    ax.set_title("Instance-Aware Feature (IAF) Module",
                 fontsize=11, fontweight="bold", pad=8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["light_blue"],  edgecolor=COLORS["blue"],   label="Backbone Feature"),
        mpatches.Patch(facecolor=COLORS["light_green"], edgecolor=COLORS["green"],  label="RoI-Align"),
        mpatches.Patch(facecolor=COLORS["yellow"],      edgecolor="#9a7a00",         label="Attention Gate"),
        mpatches.Patch(facecolor=COLORS["purple"],      edgecolor="#5a3d99",         label="Aggregation"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
              framealpha=0.9)

    save_figure(fig, OUTPUT)


if __name__ == "__main__":
    main()
