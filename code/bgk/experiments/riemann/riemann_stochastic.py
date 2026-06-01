import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig

# Import Particle Classes
from bgk.core.particle_runner import ParticleRunner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.problems.problems import get_problem
from bgk.solvers.rtsm import RTSMSolver
from bgk.solvers.ugkp import UGKPSolver


def main():
    # 1. Setup Base Configuration
    problem_name = "riemann"
    problem = get_problem(problem_name)

    SAVE_PLOTS = True

    vmax = 20.0
    t_final = 0.07
    R = 1.0
    L = problem.x_bounds[1] - problem.x_bounds[0]

    # 2. Per-solver configuration: CFL, number of cells, number of particles
    particle_solvers = {
        "RTSM": {
            "solver": RTSMSolver,
            "runner": ParticleRunner,
            "CFL": 0.9,
            "Nc": 100,
            "Np": 10**6,
        },
        "UGKP": {
            "solver": UGKPSolver,
            "runner": UGKPRunner,
            "CFL": 0.9,
            "Nc": 100,
            "Np": 10**6,
        },
    }

    # Lock to a single Knudsen number for direct comparison
    Kn = 1e-8
    Kns = [Kn]

    results = {
        "rho": {Kn: {} for Kn in Kns},
        "u": {Kn: {} for Kn in Kns},
        "T": {Kn: {} for Kn in Kns},
        "q": {Kn: {} for Kn in Kns},
        "x": {Kn: {} for Kn in Kns},
    }

    # 3. Run Experiments
    print("\n========================================")
    print(f" Running experiments for Kn = {Kn}")
    print("========================================")

    physics_conf = PhysicsConfig(
        Kn=Kn, problem_name=problem_name, R=R, constant_tau=True, tau=Kn
    )

    cpu_times = {}
    for name, solver_conf in particle_solvers.items():
        SolverClass = solver_conf["solver"]
        RunnerClass = solver_conf["runner"]
        cfl = solver_conf["CFL"]
        Nc = solver_conf["Nc"]
        Np = solver_conf["Np"]
        dx = L / Nc
        dt = cfl * dx / vmax
        grid_conf = GridConfig(
            xL=problem.x_bounds[0],
            xR=problem.x_bounds[1],
            Nx=None,
            Nc=Nc,
            Nv=[0],
            dim_v=1,
            vmax=vmax,
            vmin=-vmax,
            bc_type=problem.bc_type,
        )
        time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=cfl)
        config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

        print(f"--- Solving with {name} (CFL={cfl}, Nc={Nc}, Np={Np}) ---")
        solver = SolverClass(config)
        runner = RunnerClass(config, solver, problem, Np)

        # time
        start = time.time()
        sim = runner.run()
        end = time.time()
        cpu_time = end - start
        cpu_times[name] = cpu_time
        print(f"{name} completed in {cpu_time:.2f} seconds.")

        results["rho"][Kn][name] = sim.rho[-1]
        results["u"][Kn][name] = sim.u[-1]
        results["T"][Kn][name] = sim.T[-1]
        results["q"][Kn][name] = sim.q[-1]
        results["x"][Kn][name] = sim.x

    # 4. Load Reference Data
    reference_path = "code/bgk/reference_solutions/riemann.mat"
    reference_data = {}
    if os.path.exists(reference_path):
        mat_data = sio.loadmat(reference_path)
        reference_data = {
            "x": mat_data["x_refined"].flatten(),
            "rho": mat_data["rho_xx"].flatten(),
            "u": mat_data["u_xx"].flatten(),
            "T": mat_data["T_xx"].flatten(),
        }

    # 5. Plotting Style Dictionary (No lines, just markers)
    # Using alpha=0.6 makes overlapping points look like a denser cloud
    plot_styles = {
        "RTSM": {"marker": "o", "color": "blue", "alpha": 0.6, "markersize": 3},
        "UGKP": {"marker": "^", "color": "red", "alpha": 0.6, "markersize": 3},
    }

    macro_configs = [
        (
            "rho",
            r"$\rho$",
            0,
            0,
            {"pos": [0.05, 0.05, 0.4, 0.4], "xlim": (0.52, 0.68)},
        ),
        (
            "u",
            r"$u$",
            0,
            1,
            {"pos": [0.35, 0.05, 0.4, 0.4], "xlim": (0.82, 0.89)},
        ),
        (
            "T",
            r"$T$",
            1,
            0,
            {"pos": [0.05, 0.55, 0.4, 0.4], "xlim": (0.52, 0.68)},
        ),
        ("q", r"$q$", 1, 1, None),
    ]

    # 6. Generate the Plot
    plt.close("all")
    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(2, 2, figsize=(width, width * 0.7), constrained_layout=True)

    for macro_key, title, row, col, inset_conf in macro_configs:
        ax = axs[row, col]
        axins = ax.inset_axes(inset_conf["pos"]) if inset_conf else None

        # A. Plot Reference first (so it sits cleanly behind the noisy particles)
        if reference_data and macro_key in reference_data:
            ref_x = reference_data["x"]
            ref_data = reference_data[macro_key]
            ax.plot(ref_x, ref_data, "k--", alpha=0.8, linewidth=1.5, label="Reference")
            if axins:
                axins.plot(ref_x, ref_data, "k--", alpha=0.8, linewidth=1.5)

        # B. Plot the Scatter Data for both solvers
        for solver_name in particle_solvers.keys():
            x_grid = results["x"][Kn][solver_name]
            data = results[macro_key][Kn][solver_name]
            style = plot_styles[solver_name]

            # Note: linestyle="none" forces it to draw only the markers
            ax.plot(
                x_grid,
                data,
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                alpha=style["alpha"],
                markersize=style["markersize"],
                label=solver_name + f" ({cpu_times[solver_name]:.2f}s)",
            )

            if axins:
                axins.plot(
                    x_grid,
                    data,
                    linestyle="none",
                    marker=style["marker"],
                    color=style["color"],
                    alpha=style["alpha"],
                    markersize=style["markersize"],
                )

        # C. Format the inset box — compute y limits per solver using its own x_grid
        if axins:
            zoom_x_min, zoom_x_max = inset_conf["xlim"]
            axins.set_xlim(zoom_x_min, zoom_x_max)

            y_min_list, y_max_list = [], []
            for s in particle_solvers.keys():
                xg = results["x"][Kn][s]
                mask = (xg >= zoom_x_min) & (xg <= zoom_x_max)
                if np.any(mask):
                    y_min_list.append(np.min(results[macro_key][Kn][s][mask]))
                    y_max_list.append(np.max(results[macro_key][Kn][s][mask]))
                else:
                    y_min_list.append(np.min(results[macro_key][Kn][s]))
                    y_max_list.append(np.max(results[macro_key][Kn][s]))
            y_min = min(y_min_list)
            y_max = max(y_max_list)

            pad = (y_max - y_min) * 0.1
            if pad == 0:
                pad = 0.1

            axins.set_ylim(y_min - pad, y_max + pad)
            axins.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
            axins.set_xticks([])
            axins.set_yticks([])

            rect, connectors = ax.indicate_inset_zoom(
                axins, edgecolor="black", alpha=0.4, linewidth=1.0
            )
            for connector in connectors:
                connector.set_visible(True)

        if macro_key == "q":
            ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

        ax.set_ylabel(title)
        ax.set_xlabel(r"$x$")
        ax.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)

    axs[0, 0].legend(
        frameon=True, facecolor="white", edgecolor="black", markerscale=1.5
    )

    nc_str = "_".join(f"{n}:{c['Nc']}" for n, c in particle_solvers.items())
    np_str = "_".join(f"{n}:{c['Np']}" for n, c in particle_solvers.items())
    filename = (
        f"latex/thesis/figures/stochastic/riemann/"
        f"comparison_RTSM_UGKP_Nc_{nc_str}_Np_{np_str}.pdf"
    )
    if SAVE_PLOTS:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        tp.save_plot(filename)
    else:
        plt.show()


if __name__ == "__main__":
    main()
