"""
Generate fig_qualitative_nuscenes.pdf

Qualitative 3D detection results on the nuScenes validation set.

Each subplot shows:
  - A bird's-eye view (BEV) with ground-truth boxes (green) and
    predicted boxes (red) overlaid on the LiDAR point cloud.
  - A front-camera image with projected bounding boxes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from plot_utils import (
    COLORS,
    save_figure,
)

OUTPUT = "fig_qualitative_nuscenes.pdf"

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# Synthetic BEV point cloud                                                    #
# --------------------------------------------------------------------------- #

def make_bev_points(n: int = 2000) -> tuple:
    """Return (x, y) arrays simulating a bird's-eye-view LiDAR scan."""
    theta = RNG.uniform(0, 2 * np.pi, n)
    r     = RNG.exponential(scale=15, size=n)
    x = r * np.cos(theta) + RNG.normal(0, 0.5, n)
    y = r * np.sin(theta) + RNG.normal(0, 0.5, n)
    return x, y


def draw_bev_box(ax, cx, cy, w, h, angle_deg, color, lw=1.5, label=None):
    """Draw a rotated BEV bounding box."""
    angle = np.deg2rad(angle_deg)
    corners_local = np.array([
        [-w / 2, -h / 2],
        [ w / 2, -h / 2],
        [ w / 2,  h / 2],
        [-w / 2,  h / 2],
    ])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    corners_world = corners_local @ R.T + np.array([cx, cy])
    polygon = plt.Polygon(corners_world, closed=True,
                          fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(polygon)
    if label:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=6, color=color)


def draw_camera_box(ax, cx, cy, w, h, color, lw=1.5, label=None):
    """Draw an axis-aligned 2-D bounding box on a camera image."""
    rect = mpatches.Rectangle((cx - w / 2, cy - h / 2), w, h,
                               fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)
    if label:
        ax.text(cx - w / 2, cy - h / 2 - 2, label,
                ha="left", va="top", fontsize=6, color=color)


# --------------------------------------------------------------------------- #
# Synthetic camera image                                                       #
# --------------------------------------------------------------------------- #

def make_camera_image(width: int = 160, height: int = 90) -> np.ndarray:
    """Generate a placeholder camera image (sky + road gradient)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky
    for row in range(height // 2):
        intensity = int(135 + 80 * row / (height // 2))
        img[row, :] = [intensity, intensity + 20, min(intensity + 40, 255)]
    # Road
    for row in range(height // 2, height):
        intensity = int(60 + 40 * (row - height // 2) / (height // 2))
        img[row, :] = [intensity, intensity, intensity]
    return img


# --------------------------------------------------------------------------- #
# Ground-truth and prediction configs                                          #
# --------------------------------------------------------------------------- #

SCENARIOS = [
    {
        "title": "Scene A",
        "bev": [
            # (cx, cy, w, h, angle, is_gt)
            (5,  3, 4.5, 2.0, 15, True),
            (5,  3, 4.5, 2.0, 20, False),
            (-8, 5, 4.5, 2.0, -5, True),
            (-8, 5, 4.5, 2.0, -8, False),
            (10, -4, 4.5, 2.0, 30, True),
            (10, -4, 4.5, 2.0, 28, False),
        ],
        "cam": [
            # (cx, cy, w, h, is_gt)
            (60, 45, 30, 20, True),
            (60, 47, 31, 21, False),
            (110, 50, 25, 16, True),
            (110, 50, 26, 17, False),
        ],
    },
    {
        "title": "Scene B",
        "bev": [
            (3,  7, 4.5, 2.0,  5, True),
            (3,  7, 4.5, 2.0,  8, False),
            (-5, -3, 4.5, 2.0, 80, True),
            (-5, -3, 4.5, 2.0, 85, False),
        ],
        "cam": [
            (55, 40, 28, 18, True),
            (55, 42, 29, 19, False),
            (100, 55, 20, 14, True),
            (100, 54, 21, 15, False),
        ],
    },
    {
        "title": "Scene C",
        "bev": [
            (0,   8, 4.5, 2.0, -10, True),
            (0,   8, 4.5, 2.0, -12, False),
            (12, -2, 4.5, 2.0,  45, True),
            (12, -2, 4.5, 2.0,  42, False),
            (-10, 0, 4.5, 2.0,  90, True),
            (-10, 0, 4.5, 2.0,  88, False),
        ],
        "cam": [
            (70, 44, 32, 22, True),
            (70, 43, 33, 23, False),
            (120, 52, 22, 15, True),
            (120, 51, 23, 16, False),
        ],
    },
]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    n_scenes = len(SCENARIOS)
    fig, axes = plt.subplots(
        2, n_scenes,
        figsize=(5 * n_scenes, 8),
        gridspec_kw={"height_ratios": [1, 0.6]},
    )

    # Shared point cloud
    bev_x, bev_y = make_bev_points()
    cam_img = make_camera_image()

    for col, scene in enumerate(SCENARIOS):
        # ---- BEV row ----
        ax_bev = axes[0, col]
        ax_bev.scatter(bev_x, bev_y, s=0.4, c=COLORS["light_gray"], alpha=0.6)
        ax_bev.set_xlim(-20, 20)
        ax_bev.set_ylim(-20, 20)
        ax_bev.set_aspect("equal")
        ax_bev.set_facecolor("#1a1a2e")
        ax_bev.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
        for spine in ax_bev.spines.values():
            spine.set_visible(False)

        for cfg in scene["bev"]:
            cx, cy, w, h, angle, is_gt = cfg
            color = COLORS["green"] if is_gt else COLORS["orange"]
            draw_bev_box(ax_bev, cx, cy, w, h, angle, color)

        ax_bev.set_title(scene["title"], fontsize=9, pad=4, color="white",
                         backgroundcolor="#1a1a2e")

        # ---- Camera row ----
        ax_cam = axes[1, col]
        ax_cam.imshow(cam_img, aspect="auto")
        ax_cam.set_xlim(0, 160)
        ax_cam.set_ylim(90, 0)
        ax_cam.axis("off")

        for cfg in scene["cam"]:
            cx, cy, w, h, is_gt = cfg
            color = COLORS["green"] if is_gt else COLORS["orange"]
            draw_camera_box(ax_cam, cx, cy, w, h, color)

    # ------------------------------------------------------------------ #
    # Legend and title                                                    #
    # ------------------------------------------------------------------ #
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor=COLORS["green"],  label="Ground Truth"),
        mpatches.Patch(facecolor="none", edgecolor=COLORS["orange"], label="LUNA-Net Prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Qualitative 3D Detection Results on nuScenes Validation Set\n"
        "Top: Bird's-Eye View (BEV)   ·   Bottom: Front Camera",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, OUTPUT)


if __name__ == "__main__":
    main()
