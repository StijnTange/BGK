import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import your custom thesis plotting module
import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.problems.problems import get_problem
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver


class ConservationTrackerHook:
    """
    Custom hook to calculate and store macroscopic variables at each step.
    """

    def __init__(self, dx, R):
        self.dx = dx
        self.R = R
        self.times = []
        self.mass = []
        self.momentum = []
        self.energy = []

    def __call__(self, runner):
        # Extract macroscopics directly from the runner's distribution function
        rho, p, E = runner.df.compute_moments()

        # Integrate over the domain
        mass_tot = np.sum(rho) * self.dx
        mom_tot = np.sum(p) * self.dx
        ener_tot = np.sum(E) * self.dx

        self.times.append(runner.t)
        self.mass.append(mass_tot)
        self.momentum.append(mom_tot)
        self.energy.append(ener_tot)


def run_simulations(run_name):
    """Runs Strang and SL solvers and dumps the data to a temporary pickle file."""
    # 1. Setup Configuration
    problem_name = "gaussian"
    problem = get_problem(problem_name)

    CFL = 0.9
    dt = 1e-3
    vmax = 10.0
    t_final = 5.0
    Kn = 1e-3
    Nv = 100
    R = 1.0

    # Calculate Grid constraints based on CFL
    dx = dt * vmax / CFL
    Nx = int(np.ceil((problem.x_bounds[1] - problem.x_bounds[0]) / dx))

    print(f"[{run_name.upper()}] Setting up grid with Nx={Nx}, dx={dx:.4f}")

    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nx=Nx,
        Nv=Nv,
        vmax=vmax,
        vmin=-vmax,
        bc_type="periodic",
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem_name, R=R)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    # 2. Define Solvers to Test (Only Strang and SL)
    solvers = {
        "Strang": StrangSolver,
        "SL": SLSolver,
    }

    save_data = {}

    # 3. Run Experiments
    for name, SolverClass in solvers.items():
        print(f"\n--- Running {name} solver ({run_name} interpolation) ---")
        solver = SolverClass(config)
        runner = Runner(config=config, solver=solver, problem=problem)

        tracker = ConservationTrackerHook(dx=dx, R=R)
        runner.add_hook(tracker)

        runner.run()

        # Store data in a basic dictionary to avoid pickle issues with custom objects
        save_data[name] = {
            "times": np.array(tracker.times),
            "mass": np.array(tracker.mass),
            "momentum": np.array(tracker.momentum),
            "energy": np.array(tracker.energy),
        }

    # Save to disk
    filename = f"temp_results_{run_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\n✅ Saved {run_name} results to {filename}")


def plot_combined_results():
    """Loads the temporary pickle files and plots them using thesis_plots."""
    SAVE_PLOTS = False

    # Check if files exist
    if not os.path.exists("temp_results_weno.pkl") or not os.path.exists(
        "temp_results_lagrange.pkl"
    ):
        print(
            "❌ Error: Could not find both 'temp_results_weno.pkl' and 'temp_results_lagrange.pkl'."
        )
        print("Please run the script with '--run weno' and '--run lagrange' first.")
        sys.exit(1)

    with open("temp_results_weno.pkl", "rb") as f:
        data_weno = pickle.load(f)

    with open("temp_results_lagrange.pkl", "rb") as f:
        data_lagrange = pickle.load(f)

    # 4. Plot Relative Errors using thesis_plots dimensions
    # Assuming get_figsize returns (width, height). We manually override height to keep it proportional for 3 subplots.
    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(1, 3, figsize=(width, 2.8), constrained_layout=True)

    variables = ["mass", "momentum", "energy"]

    # Styles for distinguishing the methods and interpolations
    styles = {
        "Strang (WENO)": {"data": data_weno["Strang"], "marker": "o", "ls": "-"},
        "SL (WENO)": {"data": data_weno["SL"], "marker": "^", "ls": "-"},
        "Strang (Lagrange)": {
            "data": data_lagrange["Strang"],
            "marker": "s",
            "ls": "--",
        },
        "SL (Lagrange)": {"data": data_lagrange["SL"], "marker": "D", "ls": "--"},
    }

    for idx, var in enumerate(variables):
        ax = axs[idx]
        for label, props in styles.items():
            dataset = props["data"]
            data_var = dataset[var]
            times = dataset["times"]

            # Calculate relative error
            initial_val = data_var[0] if abs(data_var[0]) > 1e-14 else 1.0
            rel_error = np.abs(data_var - data_var[0]) / np.abs(initial_val)

            ax.plot(
                times,
                rel_error,
                label=label,
                marker=props["marker"],
                linestyle=props["ls"],
                markersize=3,
                markevery=500,
            )

        ax.set_ylabel(rf"Relative {var} error")
        ax.set_xlabel(r"Time $t$")
        ax.set_yscale("log")

        # Let the thesis_plots style sheet handle the grid if it does automatically,
        # otherwise force a light grid for readability on log scales:
        ax.grid(True, which="major", ls="-", alpha=0.5)
        ax.grid(True, which="minor", ls="--", alpha=0.2)

    # Place legend on the last subplot
    axs[0].legend()

    # 5. Save the plot using the thesis_plots save utility
    filename = "latex/thesis/figures/ch4/conservation/comparison_interpolation.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if SAVE_PLOTS:
        tp.save_plot(filename)
        print(f"✅ Saved combined plot using thesis_plots to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run and plot conservation tests for Strang/SL interpolations."
    )
    parser.add_argument(
        "--run",
        type=str,
        choices=["weno", "lagrange"],
        help="Run the simulation and save temporary data under the given interpolation name.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Load the saved WENO and Lagrange data and plot them.",
    )

    args = parser.parse_args()

    if args.run:
        run_simulations(args.run)
    elif args.plot:
        plot_combined_results()
    else:
        parser.print_help()
