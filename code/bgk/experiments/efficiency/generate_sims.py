import os
import sys
import time

# Adjust paths to ensure the bgk module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.io.io_hdf5 as io_hdf5
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver

RESULTS_DIR = "simulation_results"

# =========================================================
# ⚙️ SIMULATION TASK CONFIGURATION
# =========================================================
RUN_CFL_PROFILE = True  # Simulations for: plot_CFL_comparison.py
RUN_WP_GRID = False  # Simulations for: plot_efficiency_CFL_comparison2.py

RUN_WP_SINGLE = False  # Simulations for: plot_efficiency.py
KNS_WP_SINGLE = [1e-6, 1e-1, 1e2]  # <--- ADD YOUR Kn VALUES HERE

OVERWRITE_EXISTING = False  # Set to True to overwrite .h5 files. False skips them.
# =========================================================


def run_and_save(method, N_res, Kn, cfl, Nv, overwrite):
    problem = get_problem("gaussian")
    t_final = 0.2

    # --- Check for existing files ---
    exists = io_hdf5.result_exists(
        method, N_res, Nv, None, Kn, t_final, problem.name, RESULTS_DIR, cfl
    )

    if exists and not overwrite:
        print(
            f"⏭️  Skipping (already exists): {method}, N={N_res}, Kn={Kn}, CFL={cfl}, Nv={Nv}"
        )
        return
    elif exists and overwrite:
        print(f"⚠️  Overwriting: {method}, N={N_res}, Kn={Kn}, CFL={cfl}, Nv={Nv}")
    else:
        print(f"▶️  Running: {method}, N={N_res}, Kn={Kn}, CFL={cfl}, Nv={Nv}")

    vmax = 10.0
    R = 1.0
    L = problem.x_bounds[1] - problem.x_bounds[0]

    solver_classes = {
        "Strang": StrangSolver,
        "SL": SLSolver,
        "FVM": FVMSolver,
        "UGKS": UGKSSolver,
        "Hybrid": HybridSolver,
    }

    if method in ["Strang", "SL", "Hybrid"]:
        Nx, Nc = N_res, None
        dx = L / (Nx - 1)
    else:
        Nx, Nc = None, N_res
        dx = L / Nc

    dt = cfl * dx / vmax

    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nx=Nx,
        Nc=Nc,
        Nv=[Nv],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type="periodic",
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=cfl)
    physics_conf = PhysicsConfig(
        Kn=Kn, problem_name=problem.name, R=R, constant_tau=False
    )
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    solver = solver_classes[method](config)
    runner = Runner(config=config, solver=solver, problem=problem)

    start_time = time.perf_counter()
    sim = runner.run()
    execution_time = time.perf_counter() - start_time

    # Prepare data for HDF5
    result_dict = {
        "problem": problem.name,
        "solver": method,
        "Nc": N_res,
        "Nvx": Nv,
        "Nvy": None,
        "Kn": Kn,
        "t_final_requested": t_final,
        "CFL": cfl,
        "x": sim.x.flatten(),
        "rho": sim.rho[-1].flatten(),
        "execution_time": execution_time,
        "dx": dx,
    }

    io_hdf5.save_result(result_dict, results_dir=RESULTS_DIR)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sim_tasks = []

    print("--- Configuration ---")
    print(f"CFL Profile Tasks : {'ENABLED' if RUN_CFL_PROFILE else 'DISABLED'}")
    print(f"WP Grid Tasks     : {'ENABLED' if RUN_WP_GRID else 'DISABLED'}")
    print(
        f"WP Single Tasks   : {'ENABLED' if RUN_WP_SINGLE else 'DISABLED'} (Kn: {KNS_WP_SINGLE})"
    )
    print(f"Overwrite Existing: {'YES' if OVERWRITE_EXISTING else 'NO'}\n")

    # --- 1. Tasks for plot_CFL_comparison.py ---
    if RUN_CFL_PROFILE:
        for Kn in [1e-6, 5e-2, 1e2]:
            sim_tasks.append(
                {"method": "FVM", "N_res": 800, "Kn": Kn, "cfl": 0.9, "Nv": 40}
            )
            for method in ["Strang", "SL", "Hybrid"]:
                for cfl in [5.0, 10.0, 20.0, 50.0, 100.0]:
                    sim_tasks.append(
                        {"method": method, "N_res": 200, "Kn": Kn, "cfl": cfl, "Nv": 40}
                    )

    # --- 2. Tasks for plot_efficiency_CFL_comparison2.py ---
    if RUN_WP_GRID:
        for Kn in [1e-1]:
            sim_tasks.append(
                {"method": "FVM", "N_res": 12800, "Kn": Kn, "cfl": 0.9, "Nv": 40}
            )
            for method in ["Strang", "SL", "Hybrid"]:
                for cfl in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
                    for N_res in [100, 200, 400, 800]:
                        sim_tasks.append(
                            {
                                "method": method,
                                "N_res": N_res,
                                "Kn": Kn,
                                "cfl": cfl,
                                "Nv": 40,
                            }
                        )

    # --- 3. Tasks for plot_efficiency.py ---
    if RUN_WP_SINGLE:
        for Kn in KNS_WP_SINGLE:
            sim_tasks.append(
                {"method": "FVM", "N_res": 38400, "Kn": Kn, "cfl": 0.9, "Nv": 40}
            )  # Truth
            for method in ["Strang", "SL", "Hybrid", "FVM", "UGKS"]:
                for N_res in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
                    sim_tasks.append(
                        {
                            "method": method,
                            "N_res": N_res,
                            "Kn": Kn,
                            "cfl": 0.9,
                            "Nv": 40,
                        }
                    )

    # --- Remove Duplicates ---
    unique_tasks = []
    seen = set()
    for task in sim_tasks:
        key = (task["method"], task["N_res"], task["Kn"], task["cfl"], task["Nv"])
        if key not in seen:
            seen.add(key)
            unique_tasks.append(task)

    if not unique_tasks:
        print("No tasks selected. Exiting.")
        return

    print(f"Total unique simulations queued: {len(unique_tasks)}\n" + "=" * 40)

    # --- Execute ---
    for task in unique_tasks:
        run_and_save(**task, overwrite=OVERWRITE_EXISTING)

    print("\n✅ All requested simulations processed successfully.")


if __name__ == "__main__":
    main()
