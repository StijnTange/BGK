import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 8,
        "legend.handlelength": 1.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.5,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "patch.linewidth": 0.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",
    }
)


def get_figsize(fraction=1.0, ratio=0.618):
    fig_width = 6.0 * fraction
    fig_height = fig_width * ratio
    return (fig_width, fig_height)


def save_plot(filename):
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    plt.savefig(filename)
    plt.close()
    print(f"Exported: {filename}")
