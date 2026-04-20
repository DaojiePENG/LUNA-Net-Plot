"""
Convenience script – regenerate all LUNA-Net figures.

Usage:
    python generate_all.py
"""

import importlib
import sys


SCRIPTS = [
    "plot_iaf_module",
    "plot_llem_module",
    "plot_rsne_module",
    "plot_overall_architecture",
    "plot_qualitative_nuscenes",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n{'─' * 50}")
        print(f"Running {script}.py …")
        module = importlib.import_module(script)
        module.main()

    print(f"\n{'─' * 50}")
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
