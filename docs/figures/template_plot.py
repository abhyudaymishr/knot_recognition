"""
Publication-ready plot template.

Usage:
    python docs/figures/template_plot.py

This generates a sample figure in docs/figures/output/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def set_scientific_style():
    plt.rcParams.update({
        "figure.figsize": (4.8, 3.2),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 3,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.alpha": 0.2,
    })


def main():
    set_scientific_style()

    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="sin(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Example Scientific Plot")
    ax.grid(True)
    ax.legend(loc="best")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "example_plot.pdf"))
    fig.savefig(os.path.join(out_dir, "example_plot.png"))


if __name__ == "__main__":
    main()
