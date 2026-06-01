import os
import sys
import time

import numpy as np

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.io.io_hdf5 as io_hdf5
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.rtsm import RTSMSolver
from bgk.solvers.ugkp import UGKPSolver

RESULTS_DIR = "simulation_results"


def run_and_save(method, Nc, Kn, cfl, Np=None, is_ref=False, overwrite=False):
    problem = get_problem("riemann")
    t_final = 0.07
    vmax = 20.0
    R = 1.0
    L = problem.x_bounds[1] - problem.x_bounds[0]
    dx = L / Nc
    dt = cfl * dx / vmax

    # Use Np for file naming since this is a periodic initial-value problem, not boundary-injected
    save_name = f"{method}_Np{int(Np)}" if not is_ref else f"{method}_REF"

    if (
        io_hdf5.result_exists(
            save_name, Nc, 40, None, Kn, t_final, problem.name, RESULTS_DIR, cfl
        )
        and not overwrite
    ):
        print(f"⏭️ Skipping (already exists): {save_name} (CFL={cfl})")
        return

    print(f"▶️ Running: {save_name} (CFL={cfl})")

    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nc=Nc,
        Nx=None,
        Nv=[100],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type="periodic",
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=cfl)

    # constant_tau=False triggers the full nonlinear collision mechanics
    physics_conf = PhysicsConfig(
        Kn=Kn, problem_name=problem.name, R=R, constant_tau=False
    )
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    # Initialize correct solver and runner
    if is_ref:
        solver = FVMSolver(config)
        runner = Runner(config=config, solver=solver, problem=problem)
    elif method == "rtsm":
        solver = RTSMSolver(config)
        runner = ParticleRunner(config=config, solver=solver, problem=problem, Np=Np)
    elif method == "ugkp":
        solver = UGKPSolver(config)
        runner = UGKPRunner(config=config, solver=solver, problem=problem, Np=Np)
    else:
        raise ValueError(f"Unknown method {method}")

    t0 = time.perf_counter()
    sim = runner.run()
    exec_time = time.perf_counter() - t0

    result_dict = {
        "problem": problem.name,
        "solver": save_name,
        "Nc": Nc,
        "Nvx": 100,
        "Nvy": None,
        "Kn": Kn,
        "t_final_requested": t_final,
        "CFL": cfl,
        "x": sim.x.flatten(),
        "rho": sim.rho[-1].flatten(),
        "execution_time": exec_time,
        "dx": dx,
    }

    io_hdf5.save_result(result_dict, results_dir=RESULTS_DIR)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- CONFIGURATION ---
    Kn = 1e-8  # Single Knudsen number
    Nc_coarse = 100  # Coarse spatial grid
    Nc_fine = 4000  # Fine spatial grid for reference

    NP_LIST = np.logspace(5, 6, 5)

    # Define solvers and the specific CFLs to run for each
    TASKS = [
        {"method": "rtsm", "cfls": []},
        {"method": "ugkp", "cfls": []},
    ]
    # ---------------------

    print("--- Generating Nonlinear Work-Precision Data ---")

    # 1. Generate Deterministic Reference (FVM)
    run_and_save("UGKS", Nc=Nc_fine, Kn=Kn, cfl=0.9, is_ref=True)

    # 2. Generate Stochastic Particle Data
    for task in TASKS:
        for cfl in task["cfls"]:
            for Np in NP_LIST:
                run_and_save(task["method"], Nc=Nc_coarse, Kn=Kn, cfl=cfl, Np=int(Np))

    print("\n✅ All simulations processed successfully.")


if __name__ == "__main__":
    main()
