"""
Generate fig_rsne_module.pdf

Range-Space Normal Estimation (RSNE) module of LUNA-Net.

The RSNE module converts raw LiDAR range-image data into surface normals,
providing compact geometric cues for downstream detection heads.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from plot_utils import (
    COLORS,
    draw_box,
    draw_arrow,
    draw_label,
    save_figure,
)

OUTPUT = "fig_rsne_module.pdf"


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # ------------------------------------------------------------------ #
    # Input: Range Image                                                   #
    # ------------------------------------------------------------------ #
    draw_box(ax, 1.0, 2.0, 1.2, 0.7,
             label="Range\nImage\n$(H\\times W)$",
             facecolor=COLORS["light_blue"],
             edgecolor=COLORS["blue"],
             fontsize=8)
    draw_arrow(ax, 1.6, 2.0, 2.3, 2.0)

    # ------------------------------------------------------------------ #
    # Gradient computation                                                #
    # ------------------------------------------------------------------ #
    draw_box(ax, 3.0, 2.0, 1.2, 0.7,
             label="Sobel\nGradient\n$\\nabla r$",
             facecolor=COLORS["light_green"],
             edgecolor=COLORS["green"],
             fontsize=8)
    draw_arrow(ax, 3.6, 2.0, 4.3, 2.0)

    # ------------------------------------------------------------------ #
    # 3-D back-projection                                                 #
    # ------------------------------------------------------------------ #
    draw_box(ax, 5.0, 2.0, 1.2, 0.7,
             label="3D Back-\nprojection\n$\\mathbf{p}(u,v)$",
             facecolor=COLORS["yellow"],
             edgecolor="#9a7a00",
             fontsize=8)
    draw_arrow(ax, 5.6, 2.0, 6.3, 2.0)

    # ------------------------------------------------------------------ #
    # Cross-product → surface normal                                      #
    # ------------------------------------------------------------------ #
    draw_box(ax, 7.0, 2.0, 1.2, 0.7,
             label="Cross\nProduct\n$\\hat{n}$",
             facecolor=COLORS["orange"],
             edgecolor="#8b1a1a",
             fontsize=8)
    draw_arrow(ax, 7.6, 2.0, 8.3, 2.0)

    # ------------------------------------------------------------------ #
    # Output                                                              #
    # ------------------------------------------------------------------ #
    draw_label(ax, 8.65, 2.0, "Normal\nMap",
               fontsize=9, color=COLORS["blue"])

    # ------------------------------------------------------------------ #
    # Decorative normal arrows (visual hint)                              #
    # ------------------------------------------------------------------ #
    xs = np.linspace(0.4, 1.4, 5)
    for x in xs:
        ax.annotate("",
                    xy=(x + 0.08, 3.2),
                    xytext=(x, 2.8),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["blue"],
                                   lw=0.8, mutation_scale=7))

    # Curved surface hint
    theta = np.linspace(np.pi, 2 * np.pi, 60)
    xs_curve = 0.9 + 0.5 * np.cos(theta)
    ys_curve = 2.8 + 0.12 * np.sin(theta)
    ax.plot(xs_curve, ys_curve, color=COLORS["gray"], lw=1.0, zorder=1)

    ax.text(0.9, 2.6, "surface", ha="center", va="top", fontsize=7,
            color=COLORS["gray"])

    # ------------------------------------------------------------------ #
    # Title & legend                                                      #
    # ------------------------------------------------------------------ #
    ax.set_title("Range-Space Normal Estimation (RSNE) Module",
                 fontsize=11, fontweight="bold", pad=8)

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["light_blue"],  edgecolor=COLORS["blue"],  label="Input Range Image"),
        mpatches.Patch(facecolor=COLORS["light_green"], edgecolor=COLORS["green"], label="Gradient Estimation"),
        mpatches.Patch(facecolor=COLORS["yellow"],      edgecolor="#9a7a00",        label="Back-projection"),
        mpatches.Patch(facecolor=COLORS["orange"],      edgecolor="#8b1a1a",        label="Normal Computation"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
              framealpha=0.9)

    save_figure(fig, OUTPUT)


if __name__ == "__main__":
    main()
