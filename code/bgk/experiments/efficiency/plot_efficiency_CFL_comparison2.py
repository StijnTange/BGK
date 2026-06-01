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
    return res


def main():
    SAVE_PLOTS = True
    Kns = [1e-6, 1e-1, 1e2]
    methods = ["Strang", "SL", "Hybrid"]
    CFL_LIST = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    N_grids = [100, 200, 400, 800]

    color_maps = {
        "Strang": plt.cm.Blues(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "SL": plt.cm.Reds(np.linspace(1.0, 0.35, len(CFL_LIST))),
        "Hybrid": plt.cm.RdPu(np.linspace(1.0, 0.35, len(CFL_LIST))),
    }

    markers = ["o", "^", "s", "D", "v", "p", "*", "X"]
    line_styles = ["-", "--", ":", "-."]

    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(3, 3, figsize=(width, width * 0.8), constrained_layout=True)

    N_res_ref = N_grids[-1] * 16

    for col, Kn in enumerate(Kns):
        print(f"Loading Truth for Kn = {Kn}")
        truth_res = load_sim("FVM", N_res_ref, Kn, cfl=0.9)
        truth_x = truth_res["x"]
        truth_rho = truth_res["rho"]

        for row, method_name in enumerate(methods):
            ax = axs[row, col]

            for idx, cfl in enumerate(CFL_LIST):
                times = []
                errors = []

                for N_res in N_grids:
                    res = load_sim(method_name, N_res, Kn, cfl)
                    coarse_x, coarse_rho = res["x"], res["rho"]
                    exec_time, dx = res["execution_time"], res["dx"]

                    truth_projected = np.interp(coarse_x, truth_x, truth_rho)

                    if method_name in ["Strang", "SL", "Hybrid"]:
                        abs_error = (
                            np.sum(np.abs(coarse_rho[:-1] - truth_projected[:-1])) * dx
                        )
                        truth_norm = np.sum(np.abs(truth_projected[:-1])) * dx
                    else:
                        abs_error = np.sum(np.abs(coarse_rho - truth_projected)) * dx
                        truth_norm = np.sum(np.abs(truth_projected)) * dx

                    errors.append(abs_error / truth_norm)
                    times.append(exec_time)

                color = color_maps[method_name][idx]
                ax.loglog(
                    times,
                    errors,
                    color=color,
                    marker=markers[idx % len(markers)],
                    linestyle=line_styles[idx % len(line_styles)],
                    label=f"CFL={cfl if cfl == 0.5 else int(cfl)}",
                    markersize=4,
                    linewidth=1.2,
                )

            ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=0.5)
            ax.minorticks_on()
            ax.tick_params(which="minor", length=3, width=0.6)
            ax.tick_params(which="major", length=5, width=0.8)

            if col == 0:
                ax.set_ylabel(f"{method_name}\nRelative $L_1$ error")
            else:
                ax.set_yticklabels([])

            if row == 0:
                kn_formatted = f"10^{{{int(np.log10(Kn))}}}" if Kn != 1 else "1"
                ax.set_title(rf"Kn = ${kn_formatted}$", fontsize=10)

            if row == 2:
                ax.set_xlabel("Execution time (s)")
            else:
                ax.set_xticklabels([])

            if col == 2:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=8,
                    borderaxespad=0,
                )

    filename = "latex/thesis/figures/ch4/efficiency/work_precision_grid_comparison.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"\n✅ Saved 3x3 grid plot to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
