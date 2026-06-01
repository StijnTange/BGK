import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import matplotlib.pyplot as plt
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.ugkp_runner import UGKPRunner
from bgk.problems.problems import get_problem
from bgk.solvers.ugkp import UGKPSolver


def run_spatial_convergence():
    # 1. Basic Setup
    problem_name = "gaussian"  # Smooth problems are better for order-of-accuracy tests
    problem = get_problem(problem_name)

    t_final = 0.05
    vmax = 10.0
    tau = 1e-5
    N_particles_total = 10**6

    # 2. Grid sizes to test
    nx_values = [40, 80, 160]
    errors = []
    dx_values = []

    # 3. Generate "Ground Truth" (Reference)
    # We use a very fine grid (e.g., 1280 cells) as the reference solution
    print("Generating high-resolution reference solution...")
    nx_ref = nx_values[-1] * 4
    dx_ref = (problem.x_bounds[1] - problem.x_bounds[0]) / nx_ref
    dt_ref = 0.5 * dx_ref / vmax

    conf_ref = Config(
        grid=GridConfig(
            xL=problem.x_bounds[0],
            xR=problem.x_bounds[1],
            Nx=nx_ref,
            vmin=-vmax,
            vmax=vmax,
            Nv=1,
            bc_type=problem.bc_type,
        ),
        time=TimeConfig(t_final=t_final, dt=dt_ref),
        physics=PhysicsConfig(tau=tau, problem_name=problem_name),
    )

    ref_runner = UGKPRunner(conf_ref, UGKPSolver(conf_ref), problem, int(1e4 * nx_ref))
    ref_sim = ref_runner.run()
    rho_ref = ref_sim.rho[-1]
    x_ref = ref_sim.x

    # 4. Loop through grid sizes
    for Nx in nx_values:
        dx = (problem.x_bounds[1] - problem.x_bounds[0]) / Nx
        dx_values.append(dx)
        dt = 0.5 * dx / vmax
        N_particles_total = int(1e5 * Nx)

        print(f"Testing Nx = {Nx} (dx = {dx:.5f})...")

        conf = Config(
            grid=GridConfig(
                xL=problem.x_bounds[0],
                xR=problem.x_bounds[1],
                Nx=Nx,
                vmin=-vmax,
                vmax=vmax,
                Nv=1,
                bc_type=problem.bc_type,
            ),
            time=TimeConfig(t_final=t_final, dt=dt),
            physics=PhysicsConfig(tau=tau, problem_name=problem_name),
        )

        solver = UGKPSolver(conf)
        runner = UGKPRunner(conf, solver, problem, N_particles_total)
        sim_result = runner.run()

        # Interpolate reference solution to the current coarser grid for comparison
        rho_ref_interp = np.interp(sim_result.x, x_ref, rho_ref)

        # Compute L2 error
        l2_err = np.sqrt(np.mean((sim_result.rho[-1] - rho_ref_interp) ** 2))
        errors.append(l2_err)

    # 5. Calculate Slope (Numerical Order of Accuracy)
    slopes = np.log(np.array(errors)[1:] / np.array(errors)[:-1]) / np.log(
        np.array(dx_values)[1:] / np.array(dx_values)[:-1]
    )
    print(f"Calculated Convergence Orders: {slopes}")

    # 6. Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_values, errors, "o-", label="UGKP Error")

    # Reference lines for 1st and 2nd order
    plt.loglog(
        dx_values,
        [errors[0] * (d / dx_values[0]) for d in dx_values],
        "k--",
        alpha=0.3,
        label="1st Order",
    )
    plt.loglog(
        dx_values,
        [errors[0] * (d / dx_values[0]) ** 2 for d in dx_values],
        "k:",
        alpha=0.6,
        label="2nd Order",
    )

    plt.xlabel("Grid Spacing ($\Delta x$)")
    plt.ylabel("$L_2$ Error (Density)")
    plt.title("Spatial Convergence of UGKP")
    plt.gca().invert_xaxis()  # Smaller dx to the right
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


if __name__ == "__main__":
    run_spatial_convergence()
