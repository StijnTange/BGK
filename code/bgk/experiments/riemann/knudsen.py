import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp  # Assuming you put this at the top of your file
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver


def main():
    SAVE_PLOTS = True

    # 1. Setup Base Configuration
    problem_name = "riemann"  # Riemann problem is best for observing shock resolution
    problem = get_problem(problem_name)

    # Simulation parameters
    CFL = 0.9  # Keep CFL < 1 to ensure stability across all explicit transport steps
    vmax = 20.0
    t_final = 0.07
    Nv = 100
    R = 1.0

    # Define a base resolution number to use for Nx or Nc
    N_res = 100
    L = problem.x_bounds[1] - problem.x_bounds[0]

    methods = [
        "Strang",
        "SL",
        "Hybrid",
        "FVM",
        "UGKS",
    ]  # You can add more solvers here as you implement them
    solvers = {}
    for method in methods:
        if method == "FVM":
            solvers[method] = FVMSolver
        elif method == "SL":
            solvers[method] = SLSolver
        elif method == "Strang":
            solvers[method] = StrangSolver
        elif method == "UGKS":
            solvers[method] = UGKSSolver
        elif method == "Hybrid":
            solvers[method] = HybridSolver
        else:
            raise ValueError(f"Unknown method: {method}")

    # Test three regimes: Rarefied, Transition, Continuum
    Kns = [1e-8]

    # Dictionary to store the final simulations:
    # Notice we now store 'x' as well since grid points differ between methods
    results = {
        "rho": {Kn: {} for Kn in Kns},
        "u": {Kn: {} for Kn in Kns},
        "T": {Kn: {} for Kn in Kns},
        "q": {Kn: {} for Kn in Kns},
        "x": {Kn: {} for Kn in Kns},
    }

    # 3. Run Experiments
    for Kn in Kns:
        print("\n========================================")
        print(f" Running experiments for Kn = {Kn}")
        print("========================================")

        physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem_name, R=R)

        for name, SolverClass in solvers.items():
            print(f"--- Solving with {name} ---")

            # --- Grid switching logic ---
            if name in ["Strang", "SL", "Hybrid"]:
                # Grid-based methods
                Nx = N_res
                Nc = None
                dx = L / (Nx - 1)
            elif name in ["FVM", "UGKS"]:
                # Cell-based methods
                Nx = None
                Nc = N_res
                dx = L / Nc
            else:
                raise ValueError(f"Unknown method: {name}")

            # Ensure dt is consistent with the calculated dx
            dt = CFL * dx / vmax

            print(f"Setting up grid with Nx={Nx}, Nc={Nc}, dx={dx:.4f}")

            grid_conf = GridConfig(
                xL=problem.x_bounds[0],
                xR=problem.x_bounds[1],
                Nx=Nx,
                Nc=Nc,
                Nv=[Nv],
                dim_v=1,
                vmax=vmax,
                vmin=-vmax,
                bc_type=problem.bc_type,
            )
            time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)

            # Combine into main config for this specific solver
            config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

            solver = SolverClass(config)
            runner = Runner(config=config, solver=solver, problem=problem)

            # Run the simulation and capture the returned Simulation object
            sim = runner.run()

            # Store the final density profile (last element in the history array)
            results["rho"][Kn][name] = sim.rho[-1].flatten()
            results["u"][Kn][name] = sim.u[-1].flatten()
            results["T"][Kn][name] = sim.T[-1].flatten()
            results["q"][Kn][name] = sim.q[-1].flatten()

            # Store the x-coordinates for this specific method
            results["x"][Kn][name] = sim.x

    # reference from .mat file
    reference_path = "code/bgk/reference_solutions/riemann.mat"
    if reference_path and os.path.exists(reference_path):
        # Load the reference solution from riemann.mat
        mat_data = sio.loadmat(reference_path)
        reference_data = {
            "x": mat_data["x_refined"].flatten(),
            "rho": mat_data["rho_xx"].flatten(),
            "u": mat_data["u_xx"].flatten(),
            "T": mat_data["T_xx"].flatten(),
        }
    else:
        reference_data = None

    # 4. Plot Comparisons

    # 4. Plot Comparisons
    macro_configs = [
        (
            "rho",
            r"$\rho$",
            0,  # Index 0 (Top plot)
            {"pos": [0.05, 0.05, 0.4, 0.4], "xlim": (0.52, 0.68)},
        ),
        (
            "u",
            r"$u$",
            1,  # Index 1 (Middle plot)
            {"pos": [0.35, 0.05, 0.4, 0.4], "xlim": (0.84, 0.88)},
        ),
        (
            "T",
            r"$T$",
            2,  # Index 2 (Bottom plot)
            {"pos": [0.05, 0.55, 0.4, 0.4], "xlim": (0.52, 0.68)},
        ),
    ]

    # Define custom colors for each solver
    solver_colors = {
        "Strang": "tab:blue",
        "SL": "tab:red",
        "FVM": "tab:green",
        "UGKS": "tab:orange",
        "Hybrid": "tab:purple",
    }

    solver_markers = {
        "Strang": "o",
        "SL": "^",
        "FVM": "s",
        "UGKS": "D",
        "Hybrid": "v",
    }

    # Stagger marker start offsets so markers don't overlap between solvers
    _mark_step = 10
    solver_markevery = {
        "Strang": (0, _mark_step),
        "SL": (2, _mark_step),
        "FVM": (4, _mark_step),
        "UGKS": (6, _mark_step),
        "Hybrid": (8, _mark_step),
    }

    # Drawing order (bottom → top): Strang, FVM, Hybrid, SL, UGKS
    solver_zorder = {
        "Strang": 2,
        "FVM": 3,
        "Hybrid": 4,
        "SL": 5,
        "UGKS": 6,
    }

    for Kn in Kns:
        width, _ = tp.get_figsize(fraction=1.0)

        fig, axs = plt.subplots(
            3, 1, figsize=(width, width * 1.5), constrained_layout=True
        )

        for macro_key, title, row, inset_conf in macro_configs:
            ax = axs[row]

            axins = None
            if inset_conf:
                axins = ax.inset_axes(inset_conf["pos"])

            # 1. Plot all solvers
            for name in solvers.keys():
                data = results[macro_key][Kn][name]
                x_grid = results["x"][Kn][name]  # Use specific x_grid

                # Fetch the color and marker from the dictionaries
                line_color = solver_colors.get(name, "black")
                marker = solver_markers.get(name)

                ax.plot(
                    x_grid,
                    data,
                    label=name,
                    color=line_color,
                    marker=marker,
                    markevery=solver_markevery[name],
                    markersize=5,
                    zorder=solver_zorder[name],
                )
                if axins:
                    axins.plot(
                        x_grid,
                        data,
                        color=line_color,
                        marker=marker,
                        markevery=solver_markevery[name],
                        markersize=5,
                        zorder=solver_zorder[name],
                    )

            # 2. Plot reference
            if reference_data and macro_key in reference_data:
                ref_x = reference_data["x"]
                ref_data = reference_data[macro_key]
                ax.plot(ref_x, ref_data, "k--", alpha=0.6, label="Euler")
                if axins:
                    axins.plot(ref_x, ref_data, "k--", alpha=0.6)

            # 3. Format the inset box dynamically
            if axins:
                zoom_x_min, zoom_x_max = inset_conf["xlim"]
                axins.set_xlim(zoom_x_min, zoom_x_max)

                y_min_list = []
                y_max_list = []

                for n in solvers.keys():
                    x_g = results["x"][Kn][n]
                    mask = (x_g >= zoom_x_min) & (x_g <= zoom_x_max)

                    if np.any(mask):
                        y_min_list.append(np.min(results[macro_key][Kn][n][mask]))
                        y_max_list.append(np.max(results[macro_key][Kn][n][mask]))
                    else:
                        print(
                            f"Warning: Zoom window {inset_conf['xlim']} for {macro_key}"
                            f"({n}) is outside the x_grid domain!"
                        )

                # SAFETY NET: Check if the mask actually found any points
                if not y_min_list:
                    # Fallback to the whole domain so it doesn't crash
                    y_min = min(
                        [np.min(results[macro_key][Kn][n]) for n in solvers.keys()]
                    )
                    y_max = max(
                        [np.max(results[macro_key][Kn][n]) for n in solvers.keys()]
                    )
                else:
                    y_min = min(y_min_list)
                    y_max = max(y_max_list)

                pad = (y_max - y_min) * 0.03
                if pad == 0:
                    pad = 0.1

                # --- Format the Inset Box (Clean, No Numbers) ---
                axins.set_ylim(y_min - pad, y_max + pad)
                axins.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)

                # Remove numbers from the inset to keep it clean
                axins.set_xticks([])
                axins.set_yticks([])

                # --- Format the Bounding Box on the Main Plot ---
                rect, connectors = ax.indicate_inset_zoom(
                    axins, edgecolor="black", alpha=0.4, linewidth=0.5
                )

                # Force every single connecting line to be drawn
                for connector in connectors:
                    connector.set_visible(True)

            # # 4. Standard formatting for the main axis
            # if macro_key == "q":
            #     ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

            ax.set_ylabel(title)
            ax.set_xlabel(r"$x$")
            ax.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)

        # Single legend in the top right
        axs[0].legend(frameon=True, facecolor="white", edgecolor="black")

        # set y limits for heat flux
        # axs[1, 1].set_ylim(-0.005, 0.005)

        kn_str = f"{Kn:.0e}".replace("-0", "-")
        if SAVE_PLOTS:
            tp.save_plot(
                f"latex/thesis/figures/ch4/riemann/comparison_Kn_{kn_str}_Nx_{N_res}_CFL_{CFL}.pdf"
            )
        else:
            plt.show()


if __name__ == "__main__":
    main()
