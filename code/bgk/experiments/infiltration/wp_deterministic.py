import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.runner import Runner
from run_infiltration import (
    SIMULATIONS_DIR,
    PenetrationProblem,
    make_filename,
    save_simulation,
)

# --- Settings ---
SKIP_EXISTING_FILES = True  # Flag for the coarse sweep runs
SKIP_EXISTING_REFERENCE = True  # Flag exclusively for the reference run
BOUNDARY_ONLY_EVAL = False  # If True, calculates L2 error only for x < 0.05

REFERENCE = ("ugks", 1e-3, 4000, 60, 0.9, 10.0, 4.0)

SOLVER_CFL_CONFIGS = {
    "sl": [0.9, 5.0],
    "fvm": [0.9],
    "strang": [0.9, 5.0],
    "hybrid": [0.9],
    "ugks": [0.9],
}

NX_SWEEP = [50, 100, 200, 400, 800]


def run_deterministic_sim(run_tuple, is_ref=False):
    solver_name, Kn, Nx, Nv, CFL, vmax, t_final = run_tuple
    filename = make_filename(run_tuple)
    if is_ref and solver_name == "ugks":
        filename = filename.replace("ugks_", "ugks_ref_")

    out_path = os.path.join(SIMULATIONS_DIR, filename)

    # Independent skip logic based on whether it is the reference
    if os.path.exists(out_path):
        if is_ref and SKIP_EXISTING_REFERENCE:
            return out_path
        elif not is_ref and SKIP_EXISTING_FILES:
            return out_path

    print(f"--- Solving: {filename} ---")
    xL, xR = 0.0, 1.0
    rho_in, u_in, T_in = 1.0, 2.0, 0.1
    R_gas = 1.0

    problem = PenetrationProblem(xL, xR)
    dx = (xR - xL) / Nx
    dt = CFL * dx / vmax

    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name)
    physics_conf.reflectance_left = 1.0

    # Grid logic for cell centers vs nodes
    if solver_name in ["fvm", "ugks"]:
        grid_Nx, grid_Nc = None, Nx
    else:
        grid_Nx, grid_Nc = Nx, None

    grid_conf = GridConfig(
        xL=xL,
        xR=xR,
        Nx=grid_Nx,
        Nc=grid_Nc,
        Nv=[Nv],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type=problem.bc_type,
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    x_centers = 0.5 * (
        np.linspace(xL, xR, Nx + 1)[:-1] + np.linspace(xL, xR, Nx + 1)[1:]
    )
    u_bg_array = problem.u_bg_func(x_centers)
    T_bg_array = problem.T_bg_func(x_centers)

    if solver_name == "ugks":
        from bgk.solvers.ugks import LinearUGKSSolver

        solver = LinearUGKSSolver(config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0)
    elif solver_name == "fvm":
        from bgk.solvers.fvm import LinearFVMSolver

        solver = LinearFVMSolver(config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0)
    elif solver_name == "sl":
        from bgk.solvers.sl import LinearSLSolver

        solver = LinearSLSolver(config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0)
    elif solver_name == "strang":
        from bgk.solvers.splitting import LinearStrangSolver

        solver = LinearStrangSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
        )
    elif solver_name == "hybrid":
        from bgk.solvers.hybrid import LinearHybridSolver

        solver = LinearHybridSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
        )

    runner = Runner(config=config, solver=solver, problem=problem)
    v = runner.grid.v
    pref = rho_in / np.sqrt(2.0 * np.pi * R_gas * T_in)
    f_inflow = pref * np.exp(-((v - u_in) ** 2) / (2.0 * R_gas * T_in))
    runner.df.f_flow_left = f_inflow
    runner.solver.f_flow_left = f_inflow

    t0 = time.time()
    sim = runner.run()
    cpu_time = time.time() - t0

    meta = {
        "solver": solver_name,
        "Kn": Kn,
        "Nx": Nx,
        "Nv": Nv,
        "CFL": CFL,
        "vmax": vmax,
        "t_final": t_final,
        "type": "deterministic",
    }
    save_simulation(
        filename,
        x=sim.x,
        rho=np.mean(sim.rho[-1:], axis=0).flatten(),
        u=np.mean(sim.u[-1:], axis=0).flatten(),
        T=np.mean(sim.T[-1:], axis=0).flatten(),
        q=np.mean(sim.q[-1:], axis=0).flatten(),
        cpu_time=cpu_time,
        meta=meta,
    )
    return out_path


def get_l2_error_and_data(coarse_file, ref_file, macro="rho"):
    with h5py.File(coarse_file, "r") as fc, h5py.File(ref_file, "r") as fr:
        xc, val_c, tc = fc["x"][:], fc[macro][:], float(fc["cpu_time"][0])
        xref, val_ref = fr["x"][:], fr[macro][:]

        if BOUNDARY_ONLY_EVAL:
            mask = xc <= 0.05
            xc, val_c = xc[mask], val_c[mask]

        val_ref_interp = np.interp(xc, xref, val_ref)
        err = np.linalg.norm(val_c - val_ref_interp) / np.linalg.norm(val_ref_interp)
        return err, tc, xc, val_c


def main():
    ref_path = run_deterministic_sim(REFERENCE, is_ref=True)

    unique_cfls = sorted(
        list(set(cfl for cfls in SOLVER_CFL_CONFIGS.values() for cfl in cfls)),
        reverse=True,
    )
    _COLORS = {
        "sl": "#e41a1c",
        "fvm": "#377eb8",
        "strang": "#4daf4a",
        "hybrid": "#ff7f00",
        "ugks": "#984ea3",
    }
    _MARKERS = ["o", "s", "^", "D", "v"]

    fig_wp, axs_wp = plt.subplots(
        1, len(unique_cfls), figsize=(5 * len(unique_cfls), 5), constrained_layout=True
    )
    if len(unique_cfls) == 1:
        axs_wp = [axs_wp]

    fig_prof, ax_prof = plt.subplots(figsize=(8, 5))
    with h5py.File(ref_path, "r") as fr:
        ax_prof.plot(fr["x"][:], fr["rho"][:], "k--", label="Reference", zorder=10)

    for idx, cfl in enumerate(unique_cfls):
        ax = axs_wp[idx]
        ax.set_title(f"CFL = {cfl}")

        for s_idx, (solver, cfl_list) in enumerate(SOLVER_CFL_CONFIGS.items()):
            if cfl not in cfl_list:
                continue

            times, errors = [], []
            for nx in NX_SWEEP:
                path = run_deterministic_sim((solver, 1e-3, nx, 60, cfl, 10.0, 4.0))
                err, tc, xc, rhoc = get_l2_error_and_data(path, ref_path)
                times.append(tc)
                errors.append(err)
                ax.annotate(
                    f"{nx}",
                    (tc, err),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    alpha=0.7,
                )

                # Plot finest grid on the diagnostic profile plot
                if nx == NX_SWEEP[-1]:
                    ax_prof.plot(
                        xc,
                        rhoc,
                        label=f"{solver.upper()} (CFL={cfl})",
                        color=_COLORS[solver],
                        linestyle=["-", "--", ":"][idx % 3],
                    )

            ax.loglog(
                times,
                errors,
                marker=_MARKERS[s_idx % len(_MARKERS)],
                color=_COLORS[solver],
                label=solver.upper(),
                linewidth=1.5,
                markersize=7,
            )

        ax.set_xlabel("CPU Time (s)")
        if idx == 0:
            ax.set_ylabel(r"Relative $L_2$ Error ($\rho$)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

    # Save WP Plot
    fig_wp.savefig("wp_deterministic.pdf")
    print("\nSaved Work-Precision diagram to wp_deterministic.pdf")

    # Save Diagnostic Density Plot
    if BOUNDARY_ONLY_EVAL:
        ax_prof.set_xlim(0.0, 0.05)
    ax_prof.set_xlabel("Position x")
    ax_prof.set_ylabel(r"Density $\rho$")
    ax_prof.set_title("Diagnostic Density Profiles (Finest Grid)")
    # set x limit
    ax_prof.set_xlim(0.0, 0.1)
    ax_prof.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig_prof.tight_layout()
    fig_prof.savefig("wp_deterministic_profiles.pdf")
    print("Saved boundary profiles to wp_deterministic_profiles.pdf")


if __name__ == "__main__":
    main()
