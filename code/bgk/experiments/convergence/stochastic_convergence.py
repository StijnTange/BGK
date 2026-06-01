import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
import matplotlib.pyplot as plt
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.ugkp_runner import UGKPRunner
from bgk.problems.problems import get_problem


def run_convergence_test(solver_name, problem_name="gaussian", save_plot=True):
    plt.close("all")  # Clears any ghost figures from memory
    # 1. Setup Base Configuration
    problem = get_problem(problem_name)
    Nx = 50  # Keep grid coarse so noise is easier to isolate
    vmax = 20.0
    dx = (problem.x_bounds[1] - problem.x_bounds[0]) / Nx
    dt = 0.5 * dx / vmax

    grid_conf = GridConfig(
        xL=problem.x_bounds[0],
        xR=problem.x_bounds[1],
        Nx=Nx,
        vmin=-vmax,
        vmax=vmax,
        Nv=0,
        bc_type=problem.bc_type,
    )
    time_conf = TimeConfig(t_final=0.2, dt=dt)
    physics_conf = PhysicsConfig(Kn=1e-3, problem_name=problem_name)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    # 2. Define Particle Counts to Test (Log scale)
    particle_counts = np.logspace(3, 5, num=4)
    particle_counts = [int(n) for n in particle_counts]
    errors_rho = []
    errors_u = []
    errors_T = []

    # 3. Get "Ground Truth" Solution
    print("Generating reference solution with high particle count...")
    ref_count = particle_counts[-1] * 10
    if solver_name == "ugkp":
        from bgk.solvers.ugkp import UGKPSolver

        ref_runner = UGKPRunner(config, UGKPSolver(config), problem, ref_count)
    else:
        from bgk.solvers.rtsm import RTSMSolver

        ref_runner = ParticleRunner(config, RTSMSolver(config), problem, ref_count)

    ref_sim = ref_runner.run()
    ref_rho = ref_sim.rho[-1]  # Final state density
    ref_u = ref_sim.u[-1]  # Final state velocity
    ref_T = ref_sim.T[-1]  # Final state temperature

    # 4. Loop through particle counts
    for Np in particle_counts:
        print(f"Testing Np = {Np}...")

        if solver_name == "ugkp":
            solver = UGKPSolver(config)
            runner = UGKPRunner(config, solver, problem, Np)
        else:
            solver = RTSMSolver(config)
            runner = ParticleRunner(config, solver, problem, Np)

        sim_result = runner.run()
        current_rho = sim_result.rho[-1]
        current_u = sim_result.u[-1]
        current_T = sim_result.T[-1]

        # Calculate Relative L1 Error (matches your deterministic scripts)
        # Note: dx cancels out in the numerator and denominator
        rel_l1_err_rho = np.sum(np.abs(current_rho - ref_rho)) / np.sum(np.abs(ref_rho))
        rel_l1_err_u = np.sum(np.abs(current_u - ref_u)) / np.sum(np.abs(ref_u))
        rel_l1_err_T = np.sum(np.abs(current_T - ref_T)) / np.sum(np.abs(ref_T))

        # You can choose to plot any of these or their combination
        errors_rho.append(rel_l1_err_rho)
        errors_u.append(rel_l1_err_u)
        errors_T.append(rel_l1_err_T)

    # =========================================================================
    # 5. Plot Results (Thesis Format)
    # =========================================================================
    width, height = tp.get_figsize(fraction=0.60)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    ax.loglog(particle_counts, errors_rho, "bo-", label="Density")
    ax.loglog(particle_counts, errors_u, "rs-", label="Velocity")
    ax.loglog(particle_counts, errors_T, "g^-", label="Temperature")

    # Plot theoretical 1/sqrt(N) slope for comparison
    theoretical_slope = [
        errors_rho[0] * np.sqrt(particle_counts[0] / n) for n in particle_counts
    ]

    # Use proper LaTeX math formatting for the theoretical order
    ax.loglog(
        particle_counts,
        theoretical_slope,
        "k--",
        alpha=0.6,
        label=r"$\mathcal{O}(1/\sqrt{N_p})$",
    )

    ax.set_xlabel(r"$N_p$")
    ax.set_ylabel(r"Relative $L_1$ error density")
    ax.grid(True, which="both", linestyle="--", alpha=0.4, linewidth=0.5)

    # Uses your global white background / black border settings
    ax.legend()

    # Save as a PDF vector graphic without the title
    filename = f"latex/thesis/figures/ch4/stochastic/convergence/stochastic_convergence_{solver_name.lower()}.pdf"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if save_plot:
        tp.save_plot(filename)
        print(f"Plot saved to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    run_convergence_test(solver_name="vj", save_plot=True)
