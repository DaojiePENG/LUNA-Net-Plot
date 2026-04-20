"""
Generate fig_overall_architecture.pdf

Overall LUNA-Net architecture diagram.

LUNA-Net is a multi-modal 3D object detection network that fuses LiDAR
point clouds with camera images via three key modules:
  - RSNE  : Range-Space Normal Estimation
  - LLEM  : Local Linear Enhancement Module
  - IAF   : Instance-Aware Feature fusion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from plot_utils import (
    COLORS,
    draw_box,
    draw_arrow,
    draw_label,
    save_figure,
)

OUTPUT = "fig_overall_architecture.pdf"

# ---------------------------------------------------------------------- #
# Layout constants                                                        #
# ---------------------------------------------------------------------- #
W, H = 16, 8          # figure size (inches)
XL, XR = 0.2, 15.8   # left / right limits
YB, YT = 0.2, 7.8    # bottom / top limits


def draw_stream_label(ax, x, y, text, color=COLORS["gray"]):
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5,
            color=color, fontweight="bold", rotation=90)


def main() -> None:
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(XL, XR)
    ax.set_ylim(YB, YT)
    ax.axis("off")

    # ================================================================== #
    # Stream 1 – Camera branch (top row y ≈ 6)                           #
    # ================================================================== #
    y_cam = 6.0

    draw_box(ax, 1.2, y_cam, 1.4, 0.7,
             label="Camera\nImages", facecolor="#D6E4F0", edgecolor=COLORS["blue"], fontsize=8)
    draw_arrow(ax, 1.9, y_cam, 2.6, y_cam)

    draw_box(ax, 3.2, y_cam, 1.2, 0.7,
             label="Image\nBackbone", facecolor=COLORS["light_blue"], edgecolor=COLORS["blue"], fontsize=8)
    draw_arrow(ax, 3.8, y_cam, 4.5, y_cam)

    draw_box(ax, 5.2, y_cam, 1.3, 0.7,
             label="FPN", facecolor=COLORS["light_blue"], edgecolor=COLORS["blue"], fontsize=8)
    draw_arrow(ax, 5.85, y_cam, 6.6, y_cam)

    # IAF module receives FPN features + proposals
    draw_box(ax, 7.3, y_cam, 1.4, 0.7,
             label="IAF\nModule", facecolor=COLORS["yellow"], edgecolor="#9a7a00", fontsize=9)
    draw_arrow(ax, 8.0, y_cam, 8.8, y_cam)

    draw_box(ax, 9.4, y_cam, 1.4, 0.7,
             label="Image\nFeatures", facecolor=COLORS["light_blue"], edgecolor=COLORS["blue"], fontsize=8)

    # ================================================================== #
    # Stream 2 – LiDAR branch (bottom row y ≈ 2)                         #
    # ================================================================== #
    y_lid = 2.0

    draw_box(ax, 1.2, y_lid, 1.4, 0.7,
             label="LiDAR\nPoints", facecolor="#D5F0D6", edgecolor=COLORS["green"], fontsize=8)
    draw_arrow(ax, 1.9, y_lid, 2.6, y_lid)

    draw_box(ax, 3.2, y_lid, 1.2, 0.7,
             label="Voxel-\nization", facecolor=COLORS["light_green"], edgecolor=COLORS["green"], fontsize=8)
    draw_arrow(ax, 3.8, y_lid, 4.5, y_lid)

    draw_box(ax, 5.2, y_lid, 1.3, 0.7,
             label="RSNE\nModule", facecolor="#B5D9C2", edgecolor=COLORS["green"], fontsize=8)
    draw_arrow(ax, 5.85, y_lid, 6.6, y_lid)

    draw_box(ax, 7.3, y_lid, 1.4, 0.7,
             label="LLEM\nModule", facecolor=COLORS["light_green"], edgecolor=COLORS["green"], fontsize=9)
    draw_arrow(ax, 8.0, y_lid, 8.8, y_lid)

    draw_box(ax, 9.4, y_lid, 1.4, 0.7,
             label="Enhanced\nVoxel Feat.", facecolor=COLORS["light_green"], edgecolor=COLORS["green"], fontsize=8)

    # ================================================================== #
    # Fusion  (middle y ≈ 4)                                              #
    # ================================================================== #
    y_fus = 4.0

    # Arrows from both streams down/up to fusion
    draw_arrow(ax, 9.4, y_cam - 0.35, 9.4, y_fus + 0.45)
    draw_arrow(ax, 9.4, y_lid + 0.35, 9.4, y_fus - 0.45)

    draw_box(ax, 10.4, y_fus, 1.4, 0.8,
             label="Multi-Modal\nFusion", facecolor=COLORS["purple"], edgecolor="#5a3d99", fontsize=9)

    # Arrows from feature tensors to fusion box
    draw_arrow(ax, 10.1, y_cam, 10.1, y_fus + 0.4)
    draw_arrow(ax, 10.1, y_lid, 10.1, y_fus - 0.4)

    draw_arrow(ax, 11.1, y_fus, 11.8, y_fus)

    # ================================================================== #
    # 3-D Detection Head                                                  #
    # ================================================================== #
    draw_box(ax, 12.5, y_fus, 1.4, 0.8,
             label="3D\nDetection Head", facecolor=COLORS["orange"], edgecolor="#8b1a1a", fontsize=9)
    draw_arrow(ax, 13.2, y_fus, 13.9, y_fus)

    draw_box(ax, 14.6, y_fus, 1.5, 0.8,
             label="Predictions\n(BBox, Class,\nScore)", facecolor="#FDECEA", edgecolor=COLORS["orange"], fontsize=7.5)

    # ================================================================== #
    # Proposal arrow: camera → IAF                                        #
    # ================================================================== #
    draw_arrow(ax, 3.8, y_cam - 0.35, 3.8, 3.0)
    draw_box(ax, 3.8, 3.0, 1.2, 0.6,
             label="Region\nProposals", facecolor=COLORS["light_gray"], edgecolor=COLORS["gray"], fontsize=7.5)
    ax.annotate("",
                xy=(7.3, y_cam - 0.35), xytext=(4.4, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))

    # ================================================================== #
    # Title & legend                                                      #
    # ================================================================== #
    ax.set_title("LUNA-Net: Overall Architecture",
                 fontsize=14, fontweight="bold", pad=10)

    # Stream labels on the left
    draw_stream_label(ax, 0.45, y_cam, "Camera Stream", color=COLORS["blue"])
    draw_stream_label(ax, 0.45, y_lid, "LiDAR Stream", color=COLORS["green"])

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["light_blue"],  edgecolor=COLORS["blue"],  label="Camera Modules"),
        mpatches.Patch(facecolor=COLORS["light_green"], edgecolor=COLORS["green"], label="LiDAR Modules"),
        mpatches.Patch(facecolor=COLORS["yellow"],      edgecolor="#9a7a00",        label="IAF Module"),
        mpatches.Patch(facecolor=COLORS["purple"],      edgecolor="#5a3d99",        label="Fusion"),
        mpatches.Patch(facecolor=COLORS["orange"],      edgecolor="#8b1a1a",        label="Detection Head"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.9)

    save_figure(fig, OUTPUT)


if __name__ == "__main__":
    main()
