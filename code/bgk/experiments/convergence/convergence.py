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
    if solver_name not in mapping:
        raise ValueError(f"Unknown solver: {solver_name}")
    return mapping[solver_name](config)


# ─── Run one simulation ───────────────────────────────────────────────────────
def run_sim(problem_name, solver_name, Nx, Nv, dt, t_final, vmax, Kn, omega=0.5, R=1.0):
    """
    Build config, run simulation, return Simulation object.
    problem_name : str   (e.g. "gaussian")
    Nx           : number of spatial cells (Nc) OR grid points (Nx) depending on solver
    Nv           : number of velocity points (1D)
    """
    # Route to cells (Nc) or grid points (Nx) based on the solver
    if solver_name.lower() in ["fvm", "ugks"]:
        conf_Nx = None
        conf_Nc = Nx
    else:
        conf_Nx = Nx
        conf_Nc = None

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
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=None)
    physics_conf = PhysicsConfig(
        problem_name=problem_name, Kn=Kn, omega=omega, R=R, constant_tau=True
    )
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)
    problem = get_problem(problem_name, config)
    solver = build_solver(solver_name, config)
    runner = Runner(config=config, solver=solver, problem=problem)
    return runner.run()


# ─── L1 error ────────────────────────────────────────────────────────────────
def compute_macro_errors(sim, ref_sim):
    """
    Interpolate reference onto coarse grid and compute relative L1 errors.
    Flattens arrays to handle the (1, Nx) storage shape from _record_state.
    """
    x_c = sim.x
    dx = x_c[1] - x_c[0]
    x_r = ref_sim.x

    rho_ref = np.interp(x_c, x_r, np.array(ref_sim.rho[-1]).flatten())
    u_ref = np.interp(x_c, x_r, np.array(ref_sim.u[-1]).flatten())
    T_ref = np.interp(x_c, x_r, np.array(ref_sim.T[-1]).flatten())

    rho_num = np.array(sim.rho[-1]).flatten()
    u_num = np.array(sim.u[-1]).flatten()
    T_num = np.array(sim.T[-1]).flatten()

    def rel_l1(num, ref):
        denom = np.sum(np.abs(ref)) * dx
        return (
            np.sum(np.abs(num - ref)) * dx / denom
            if denom > 1e-14
            else np.sum(np.abs(num - ref)) * dx
        )

    return rel_l1(rho_num, rho_ref), rel_l1(u_num, u_ref), rel_l1(T_num, T_ref)


# ─── Plot (thesis style) ──────────────────────────────────────────────────────
def plot_convergence(results, method_name, save_plots=False, filename=None):
    dts = np.array(results["dt"])
    rho = np.array(results["rho"])
    u = np.array(results["u"])
    T = np.array(results["T"])

    width, height = tp.get_figsize(fraction=0.48)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    ax.loglog(dts, rho, marker="o", color="blue", label="Density")
    ax.loglog(dts, u, marker="s", color="red", label="Velocity")
    ax.loglog(dts, T, marker="^", color="green", label="Temperature")

    # Reference order lines anchored to the density curve at the largest dt
    anchor_val = rho[-1]
    ref_order1 = anchor_val * (dts / dts[-1])
    ref_order2 = anchor_val * (dts / dts[-1]) ** 2
    ref_order3 = anchor_val * (dts / dts[-1]) ** 3

    ax.loglog(dts, ref_order1, "-.", color="grey", label="Order 1")
    ax.loglog(dts, ref_order2, "--", color="gray", label="Order 2")
    ax.loglog(dts, ref_order3, ":", color="grey", label="Order 3")

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"Relative $L_1$ error")
    ax.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
    ax.legend()

    if save_plots:
        tp.save_plot(filename or f"convergence_{method_name}.pdf")
    else:
        plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(solver_name="sl"):
    problem_name = "gaussian"

    CFL = 0.9
    t_final = 0.1
    vmax = 10.0
    Kn = 1e-1
    Nv = 40
    omega = 0.5
    R = 1.0

    SAVE_PLOTS = False
    filename = f"latex/thesis/figures/ch4/convergence/{solver_name}_Kn_{Kn}"

    # dt values to test
    dt_values = np.logspace(-4, -3, num=6)

    # Reference: dt half the smallest test dt, Nx from CFL relation
    dt_ref = dt_values[0] / 4.0
    dx_ref = dt_ref * vmax / CFL
    L = 2.0  # gaussian domain [-1, 1]
    Nx_ref = int(np.ceil(L / dx_ref))

    print(f"Running reference simulation with dt={dt_ref:.2e}, resolution={Nx_ref}")
    ref_sim = run_sim(
        problem_name=problem_name,
        solver_name=solver_name,
        Nx=Nx_ref,
        Nv=Nv,
        dt=dt_ref,
        t_final=t_final,
        vmax=vmax,
        Kn=Kn,
        omega=omega,
        R=R,
    )
    print(f"Reference done (resolution={Nx_ref}, dt={dt_ref})")

    results = {"dt": [], "rho": [], "u": [], "T": []}

    for dt in dt_values:
        dx = dt * vmax / CFL
        Nx = int(np.ceil(L / dx))
        print(f"Running dt={dt:.2e}, resolution={Nx}")

        sim = run_sim(
            problem_name=problem_name,
            solver_name=solver_name,
            Nx=Nx,
            Nv=Nv,
            dt=dt,
            t_final=t_final,
            vmax=vmax,
            Kn=Kn,
            omega=omega,
            R=R,
        )

        err_rho, err_u, err_T = compute_macro_errors(sim, ref_sim)
        print(f"L1 Errors | rho={err_rho:.3e} | u={err_u:.3e} | T={err_T:.3e}")

        results["dt"].append(dt)
        results["rho"].append(err_rho)
        results["u"].append(err_u)
        results["T"].append(err_T)

    plot_convergence(results, solver_name, SAVE_PLOTS, filename)


if __name__ == "__main__":
    for method in ["hybrid"]:
        print(f"\n=== Running convergence test for {method} ===")
        main(solver_name=method)
