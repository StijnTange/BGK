import os
import sys

import numpy as np

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.io.io_hdf5 import result_exists, save_result
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver


def run_and_save_simulation(
    SolverClass,
    solver_name,
    N_val,
    Kn,
    problem,
    is_cell_centered,
    results_dir,
    overwrite,
):
    CFL = 0.5
    vmax = 10.0
    t_final = 0.30
    Nv = 40
    R = 1.0

    # If the result exists and we aren't overwriting, skip
    if not overwrite and result_exists(
        solver=solver_name,
        Nc=N_val,
        Nvx=Nv,
        Nvy=None,
        Kn=Kn,
        t_final=t_final,
        problem=problem.name,
        results_dir=results_dir,
        CFL=CFL,
    ):
        print(f"  -> Skipping {solver_name} N={N_val} Kn={Kn:.2e} (already exists)")
        return

    dx = (problem.x_bounds[1] - problem.x_bounds[0]) / N_val
    dt = CFL * dx / vmax

    # Correctly assign Nx or Nc based on solver type
    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nx=None if is_cell_centered else N_val,
        Nc=N_val if is_cell_centered else None,
        Nv=[Nv],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type="periodic",
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name, R=R)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    solver = SolverClass(config)
    runner = Runner(config=config, solver=solver, problem=problem)
    sim = runner.run()

    # It is crucial to have the x-coordinates for interpolation later
    if not hasattr(sim, "x"):
        # Fallback in case the solver doesn't attach x to the sim object
        x_coords = np.linspace(problem.x_bounds[0], problem.x_bounds[1], N_val)
    else:
        x_coords = sim.x

    result_dict = {
        "problem": problem.name,
        "solver": solver_name,
        "Nc": N_val,
        "Nvx": Nv,
        "Nvy": None,
        "Kn": Kn,
        "t_final_requested": t_final,
        "CFL": CFL,
        "rho": sim.rho[-1],
        "dx": dx,
        "x": x_coords,
    }

    save_result(result_dict, results_dir=results_dir)


def main():
    # =========================================================================
    # SCRIPT SETTINGS
    # =========================================================================
    OVERWRITE_EXISTING = False
    RESULTS_DIR = (
        "code/bgk/experiments/convergence/sims"  # Make sure this matches plot_sims.py
    )

    # Choose completely arbitrary grid sizes for the convergence tests
    TEST_GRIDS = [50, 100, 200]
    N_REF = 800
    # =========================================================================

    problem = get_problem("gaussian")
    Kns = np.logspace(-8, 4, 6)

    solvers_to_run = ["Hybrid", "FVM", "SL", "UGKS", "Strang"]
    all_solvers = {
        "Hybrid": HybridSolver,
        "FVM": FVMSolver,
        "SL": SLSolver,
        "UGKS": UGKSSolver,
        "Strang": StrangSolver,
    }
    # select only solver names and classes for the solvers we want to run
    solvers = {name: all_solvers[name] for name in solvers_to_run}

    print(f"Starting simulation batch. Results will be saved to: {RESULTS_DIR}/")
    N_grids = TEST_GRIDS + [N_REF]

    for solver_name, SolverClass in solvers.items():
        print(f"\n=== Running {solver_name} ===")
        # Re-introduce the check for cell-centered vs grid points
        is_cell_centered = solver_name in ["FVM", "UGKS"]

        for Kn in Kns:
            print(f" Processing Kn: {Kn:.2e}")
            for N_val in N_grids:
                run_and_save_simulation(
                    SolverClass,
                    solver_name,
                    N_val,
                    Kn,
                    problem,
                    is_cell_centered,
                    RESULTS_DIR,
                    OVERWRITE_EXISTING,
                )


if __name__ == "__main__":
    main()
