import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from bgk.problems.problems import get_problem
from bgk.solvers.fvm import FVMSolver
from bgk.solvers.hybrid import HybridSolver
from bgk.solvers.sl import SLSolver
from bgk.solvers.splitting import StrangSolver
from bgk.solvers.ugks import UGKSSolver


# ─── Solver factory ───────────────────────────────────────────────────────────
def build_solver(solver_name, config):
    mapping = {
        "sl": SLSolver,
        "strang": StrangSolver,
        "fvm": FVMSolver,
        "ugks": UGKSSolver,
        "hybrid": HybridSolver,
    }
    if solver_name.lower() not in mapping:
        raise ValueError(f"Unknown solver: {solver_name}")
    return mapping[solver_name.lower()](config)


# ─── Run one simulation ───────────────────────────────────────────────────────
def run_sim(
    problem_name, solver_name, N_val, Nv, t_final, vmax, Kn, CFL, omega=0.5, R=1.0
):
    """
    Build config, run simulation, return Simulation object and dx.
    N_val : number of spatial cells (Nc) OR grid points (Nx) depending on solver
    """
    is_cell_centered = solver_name.lower() in ["fvm", "ugks"]
    L = 2.0  # Domain [-1, 1] for gaussian problem

    # Calculate dx and set Nc/Nx appropriately
    if is_cell_centered:
        dx = L / N_val
        conf_Nx = None
        conf_Nc = N_val
    else:
        dx = L / (N_val - 1)
        conf_Nx = N_val
        conf_Nc = None

    # Couple dt to dx using CFL
    dt = CFL * dx / vmax

    grid_conf = GridConfig(
        xL=-1.0,
        xR=1.0,
        Nx=conf_Nx,
        Nc=conf_Nc,
        Nv=[Nv],
        dim_v=1,
        vmin=-vmax,
        vmax=vmax,
        bc_type="periodic",
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
    physics_conf = PhysicsConfig(
        problem_name=problem_name, Kn=Kn, omega=omega, R=R, constant_tau=True
    )
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)
    problem = get_problem(problem_name, config)
    solver = build_solver(solver_name, config)
    runner = Runner(config=config, solver=solver, problem=problem)

    return runner.run(), dx


# ─── L1 error ────────────────────────────────────────────────────────────────
def compute_macro_errors(sim, ref_sim, N_c, N_ref, is_cell_centered):
    """
    Compute relative L1 errors using perfect grid slicing matching knudsen2.py.
    """
    if is_cell_centered:
        factor = N_ref // N_c
        center_offset = factor // 2
    else:
        factor = (N_ref - 1) // (N_c - 1)
        center_offset = 0

    # Extract 1D arrays
    rho_num = np.squeeze(sim.rho[-1])
    u_num = np.squeeze(sim.u[-1])
    T_num = np.squeeze(sim.T[-1])

    rho_ref = np.squeeze(ref_sim.rho[-1])
    u_ref = np.squeeze(ref_sim.u[-1])
    T_ref = np.squeeze(ref_sim.T[-1])

    # Dynamic slicing natively projects the high-res reference onto the coarse points
    rho_ref_proj = rho_ref[center_offset::factor]
    u_ref_proj = u_ref[center_offset::factor]
    T_ref_proj = T_ref[center_offset::factor]

    def rel_l1(num, ref_proj):
        abs_diff = np.sum(np.abs(num - ref_proj))
        norm_ref = np.sum(np.abs(ref_proj))
        return abs_diff / norm_ref if norm_ref > 1e-14 else abs_diff

    return (
        rel_l1(rho_num, rho_ref_proj),
        rel_l1(u_num, u_ref_proj),
        rel_l1(T_num, T_ref_proj),
    )


# ─── Plot (thesis style) ──────────────────────────────────────────────────────
def plot_spatial_convergence(
    results, method_name, grid_sym, save_plots=False, filename=None
):
    dxs = np.array(results["dx"])
    rho = np.array(results["rho"])
    u = np.array(results["u"])
    T = np.array(results["T"])

    width, height = tp.get_figsize(fraction=0.48)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    ax.loglog(dxs, rho, marker="o", color="blue", label="Density")
    ax.loglog(dxs, u, marker="s", color="red", label="Velocity")
    ax.loglog(dxs, T, marker="^", color="green", label="Temperature")

    # Reference order lines anchored to the density curve at the coarsest grid (largest dx)
    anchor_val = rho[0]
    ref_order1 = anchor_val * (dxs / dxs[0])
    ref_order2 = anchor_val * (dxs / dxs[0]) ** 2
    ref_order3 = anchor_val * (dxs / dxs[0]) ** 3

    ax.loglog(dxs, ref_order1, "-.", color="grey", label="Order 1")
    ax.loglog(dxs, ref_order2, "--", color="gray", label="Order 2")
    ax.loglog(dxs, ref_order3, ":", color="grey", label="Order 3")

    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"Relative $L_1$ error")
    ax.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
    ax.legend()

    if save_plots:
        tp.save_plot(filename or f"spatial_convergence_{method_name}.pdf")
    else:
        plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(solver_name="sl"):
    problem_name = "gaussian"

    CFL = 10.0
    t_final = 0.3
    vmax = 10.0
    Kn = 1e-4
    Nv = 40
    omega = 0.5
    R = 1.0

    SAVE_PLOTS = False
    filename = f"latex/thesis/figures/ch4/convergence/{solver_name}_spatial_Kn_{Kn}"

    # Grid logic adapted from knudsen2.py
    is_cell_centered = solver_name.lower() in ["fvm", "ugks"]
    grid_sym = "N_c" if is_cell_centered else "N_x"

    base_Nx = 100
    num_test_grids = 9
    test_grids = []

    if is_cell_centered:
        refinement_factor = 3
        test_grids = [base_Nx * (refinement_factor**i) for i in range(num_test_grids)]
        ref_factor = 3
        N_ref = test_grids[-1] * ref_factor
    else:
        refinement_factor = 2
        current_N = base_Nx
        for _ in range(num_test_grids):
            test_grids.append(current_N)
            current_N = current_N * 2 - 1
        ref_factor = 4
        N_ref = (test_grids[-1] - 1) * ref_factor + 1

    print(
        f"[{solver_name}] Grid type: {'Cell Centers (Nc)' if is_cell_centered else 'Grid Points (Nx)'}"
    )
    print(f"[{solver_name}] Test grids ({grid_sym}): {test_grids}")
    print(f"[{solver_name}] Reference grid ({grid_sym}): {N_ref}")

    # Run reference simulation first
    print(f"\nRunning reference simulation with resolution={N_ref}...")
    ref_sim, dx_ref = run_sim(
        problem_name=problem_name,
        solver_name=solver_name,
        N_val=N_ref,
        Nv=Nv,
        t_final=t_final,
        vmax=vmax,
        Kn=Kn,
        CFL=CFL,
        omega=omega,
        R=R,
    )
    print(f"Reference done (dx={dx_ref:.2e})")

    results = {"dx": [], "rho": [], "u": [], "T": []}

    # Run test grids
    for N_c in test_grids:
        dt = CFL * (2.0 / N_c) * vmax  # dt from CFL relation for domain [-1, 1]
        print(f"Running test grid resolution={N_c} and dt={dt:.4e}...")
        sim, dx = run_sim(
            problem_name=problem_name,
            solver_name=solver_name,
            N_val=N_c,
            Nv=Nv,
            t_final=t_final,
            vmax=vmax,
            Kn=Kn,
            CFL=CFL,
            omega=omega,
            R=R,
        )

        err_rho, err_u, err_T = compute_macro_errors(
            sim, ref_sim, N_c, N_ref, is_cell_centered
        )
        print(f"  L1 Errors | rho={err_rho:.3e} | u={err_u:.3e} | T={err_T:.3e}")

        results["dx"].append(dx)
        results["rho"].append(err_rho)
        results["u"].append(err_u)
        results["T"].append(err_T)

    plot_spatial_convergence(results, solver_name, grid_sym, SAVE_PLOTS, filename)


if __name__ == "__main__":
    # Test your chosen solvers here
    for method in ["hybrid"]:
        print(f"\n=== Running spatial convergence test for {method} ===")
        main(solver_name=method)
