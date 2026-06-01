import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import bgk.io.io_hdf5 as io_hdf5
import bgk.thesis_plots as tp

RESULTS_DIR = "simulation_results"


def load_sim(method_name, N_res, Kn, cfl, Nv=40, t_final=0.2):
    res = io_hdf5.load_result(
        method_name, N_res, Nv, None, Kn, t_final, "gaussian", RESULTS_DIR, cfl
    )
    if res is None:
        raise FileNotFoundError(
            f"Missing simulation: {method_name}, N={N_res}, Kn={Kn}, CFL={cfl}"
        )
    return res["x"], res["rho"]


def main():
    SAVE_PLOTS = True
    Kns = [1e-6, 1e-2, 1e2]
    methods = ["Strang", "SL", "Hybrid"]
    CFL_LIST = [5.0, 10.0, 20.0, 50.0, 100.0]
    N_test = 200
    N_ref = 12800

    color_maps = {
        "Strang": plt.cm.Blues(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "SL": plt.cm.Reds(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "Hybrid": plt.cm.RdPu(np.linspace(1.0, 0.35, len(CFL_LIST))),
    }

    markers = ["o", "^", "s", "D", "v"]
    line_styles = ["-", "--", ":", "-.", "--"]

    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(3, 3, figsize=(width, width * 0.8), constrained_layout=True)

    for col, Kn in enumerate(Kns):
        print(f"Loading Truth for Kn = {Kn}")
        truth_x, truth_rho = load_sim("FVM", N_ref, Kn, cfl=0.9)

        for row, method_name in enumerate(methods):
            ax = axs[row, col]

            ax.plot(
                truth_x,
                truth_rho,
                color="black",
                linestyle="-",
                linewidth=1.5,
                label="Reference",
                zorder=1,
            )

            for idx, cfl in enumerate(CFL_LIST):
                coarse_x, coarse_rho = load_sim(method_name, N_test, Kn, cfl)

                color = color_maps[method_name][idx]
                ax.plot(
                    coarse_x,
                    coarse_rho,
                    color=color,
                    marker=markers[idx % len(markers)],
                    linestyle=line_styles[idx % len(line_styles)],
                    label=f"CFL={cfl if cfl < 1 else int(cfl)}",
                    markersize=3,
                    linewidth=1,
                    markevery=15,
                )

            ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)

            if col == 0:
                ax.set_ylabel(f"{method_name}\nDensity $\\rho$")
            else:
                ax.set_yticklabels([])

            if row == 0:
                kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn != 1 else "1"
                ax.set_title(rf"Kn = ${kn_formatted}$", fontsize=10)

            if row == 2:
                ax.set_xlabel("Position $x$")
            else:
                ax.set_xticklabels([])

            if col == 2:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=8,
                    borderaxespad=0,
                )

    filename = "latex/thesis/figures/ch4/efficiency/density_profile_grid_comparison.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"\n✅ Saved 3x3 density profile grid to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
