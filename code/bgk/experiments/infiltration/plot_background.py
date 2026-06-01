import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import bgk.thesis_plots as tp


class PenetrationProblem:
    def __init__(self, xL, xR):
        self.x_bounds = (xL, xR)

    def rho_bg_func(self, x):
        x_norm = x / self.x_bounds[1]

        n0 = 1.0
        alpha = 2.0
        base_density = n0 * np.exp(alpha * (1 - x_norm))

        peak_height = 25.0
        x_center = 0.015
        peak_width = 0.015

        ionization_peak = peak_height * np.exp(
            -0.5 * ((x_norm - x_center) / peak_width) ** 2
        )

        return base_density + ionization_peak

    def u_bg_func(self, x):
        u_left = -1.0
        u_right = 1.0
        x_norm = x / self.x_bounds[1]

        k = 1.0
        linear_part = u_left + (u_right - u_left) * x_norm
        concave_bow = k * x_norm * (1.0 - x_norm)

        return linear_part + concave_bow

    def T_bg_func(self, x):
        T_left = 0.1
        T_right = 10.0
        x_norm = x / self.x_bounds[1]

        return T_left + (T_right - T_left) * (x_norm**2)


def main():
    # =========================================================================
    # PLOTTING CONFIGURATION
    # =========================================================================

    # Choose which fields to plot from: ["density", "velocity", "temperature", "tau"]
    PLOTS_TO_SHOW = ["density", "velocity", "temperature", "tau"]

    # Choose the grid layout: (rows, columns)
    # Examples: (2, 2) for a square, (1, 4) for a wide row, (4, 1) for a tall column
    LAYOUT = (2, 2)

    # Configure thesis dimensions
    FIG_FRACTION = 1.0  # 1.0 = full text width, 0.5 = half width
    FIG_RATIO = (
        0.618  # Standard golden ratio (or adjust for specific grids, e.g., 0.3 for 1x4)
    )
    FILENAME = "latex/thesis/figures/infiltration/background_profiles"

    # =========================================================================

    print("Generating background profiles...")

    # 1. Define the domain
    xL, xR = 0.0, 1.0
    problem = PenetrationProblem(xL, xR)

    # 2. Create a high-resolution spatial grid
    x_plot = np.linspace(xL, xR, 1000)

    # 3. Evaluate the background functions
    rho_plot = problem.rho_bg_func(x_plot)
    u_plot = problem.u_bg_func(x_plot)
    T_plot = problem.T_bg_func(x_plot)

    # 4. Calculate spatially varying Relaxation Time (tau)
    Kn = 5e-1
    omega = 0.5
    tau_ref = Kn * np.sqrt(2.0 / np.pi)
    tau_plot = (tau_ref * (T_plot ** (omega - 1.0))) / rho_plot

    # 5. Map the string names to their respective data and formatting
    plot_data_map = {
        "density": {
            "y": rho_plot,
            "ylabel": r"$\rho^{\text{bg}}(x)$",
            "title": r"Background Density ($\rho^{\text{bg}}$)",
            "color": "#4daf4a",  # Green
            "yscale": "linear",
        },
        "velocity": {
            "y": u_plot,
            "ylabel": r"$u^{\text{bg}}(x)$",
            "title": r"Background Velocity ($u^{\text{bg}}$)",
            "color": "#377eb8",  # Blue
            "yscale": "linear",
        },
        "temperature": {
            "y": T_plot,
            "ylabel": r"$T^{\text{bg}}(x)$",
            "title": r"Background Temperature ($T^{\text{bg}}$)",
            "color": "#e41a1c",  # Red
            "yscale": "linear",
        },
        "tau": {
            "y": tau_plot,
            "ylabel": r"$\tau^{\text{bg}}(x)$",
            "title": rf"Relaxation Time ($\tau^{{\text{{bg}}}}$) [Kn = {Kn}]",
            "color": "#984ea3",  # Purple
            "yscale": "log",
        },
    }

    # 6. Validate Layout
    nrows, ncols = LAYOUT
    if len(PLOTS_TO_SHOW) > nrows * ncols:
        raise ValueError(
            f"Layout {LAYOUT} is too small for {len(PLOTS_TO_SHOW)} plots."
        )

    # 7. Create Figure using thesis_plots sizing
    fig, axs = plt.subplots(
        nrows, ncols, figsize=tp.get_figsize(fraction=FIG_FRACTION, ratio=FIG_RATIO)
    )

    # Flatten the axes array to easily iterate over it regardless of grid shape
    axs_flat = np.atleast_1d(axs).flatten()

    # 8. Plot the requested fields
    for i, field_name in enumerate(PLOTS_TO_SHOW):
        ax = axs_flat[i]
        data = plot_data_map[field_name]

        # Plot data (using slightly thicker line than axis defaults for visibility)
        ax.plot(x_plot, data["y"], color=data["color"], lw=1.0)

        # Formatting
        # ax.set_title(data["title"])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(data["ylabel"])
        ax.set_yscale(data["yscale"])

        # Add zero-line for velocity
        if field_name == "velocity":
            ax.axhline(0, color="black", lw=0.5, linestyle=":")

        ax.grid(True, linestyle="--", alpha=0.3, which="both")

    # 9. Clean up empty subplots (if layout is larger than plots requested)
    for j in range(len(PLOTS_TO_SHOW), len(axs_flat)):
        fig.delaxes(axs_flat[j])

    # 10. Save via thesis_plots
    plt.tight_layout()
    tp.save_plot(FILENAME)


if __name__ == "__main__":
    main()
