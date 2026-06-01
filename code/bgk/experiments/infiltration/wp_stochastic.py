import math
import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner
from run_infiltration import (
    SIMULATIONS_DIR,
    PenetrationProblem,
    make_filename,
    save_simulation,
)

# --- Settings ---
SKIP_EXISTING_FILES = False  # Flag for the coarse sweep runs
SKIP_EXISTING_REFERENCE = True  # Flag exclusively for the reference run
BOUNDARY_ONLY_EVAL = False  # If True, calculates L2 error only for x < 0.05

REFERENCE_DET = ("ugks", 1e-3, 4000, 60, 0.9, 10.0, 4.0)
FIXED_NX = 500

SOLVER_CFL_CONFIGS = {
    "rtsm": [0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    "ugkp": [],
    "vj": [0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
}

NINJ_SWEEP = [1e4, 5e4, 1e5, 5e5]


def _fmt(val):
    return f"{float(val):.1e}".replace("-0", "-").replace("+0", "").replace("+", "")


def run_reference_sim():
    solver_name, Kn, Nx, Nv, CFL, vmax, t_final = REFERENCE_DET
    filename = make_filename(REFERENCE_DET).replace("ugks_", "ugks_ref_")
    out_path = os.path.join(SIMULATIONS_DIR, filename)

    # Dedicated skip flag for the reference
    if os.path.exists(out_path) and SKIP_EXISTING_REFERENCE:
        return out_path

    print(f"--- Solving Reference Case: {filename} ---")
    xL, xR = 0.0, 1.0
    rho_in, u_in, T_in = 1.0, 2.0, 0.1
    R_gas = 1.0

    problem = PenetrationProblem(xL, xR)
    dx = (xR - xL) / Nx
    dt = CFL * dx / vmax

    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name)
    physics_conf.reflectance_left = 1.0
    grid_conf = GridConfig(
        xL=xL,
        xR=xR,
        Nx=None,
        Nc=Nx,
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

    from bgk.solvers.ugks import LinearUGKSSolver

    solver = LinearUGKSSolver(config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0)
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
        "is_ref": True,
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


def run_stochastic_sim(run_tuple):
    solver_name, Kn, Nx, Np, N_INJ_TOTAL, CFL, vmax, t_final = run_tuple
    filename = make_filename(run_tuple)
    out_path = os.path.join(SIMULATIONS_DIR, filename)

    # Regular skip flag for the sweeps
    if os.path.exists(out_path) and SKIP_EXISTING_FILES:
        return out_path

    print(f"--- Solving: {filename} ---")
    xL, xR = 0.0, 1.0
    rho_in, u_in, T_in = 1.0, 2.0, 0.1
    R_gas = 1.0

    problem = PenetrationProblem(xL, xR)
    dx = (xR - xL) / Nx
    dt = CFL * dx / vmax

    v_th = np.sqrt(2.0 * R_gas * T_in)
    U = u_in / v_th
    exact_flux = (
        rho_in
        * np.sqrt(R_gas * T_in / (2.0 * np.pi))
        * (np.exp(-(U**2)) + np.sqrt(np.pi) * U * (1.0 + math.erf(U)))
    )
    N_inj = float(N_INJ_TOTAL * (dt / t_final))
    M_REF = (exact_flux * dt) / N_inj

    physics_conf = PhysicsConfig(Kn=Kn, problem_name=problem.name)
    physics_conf.reflectance_left = 1.0
    grid_conf = GridConfig(
        xL=xL,
        xR=xR,
        Nx=None,
        Nc=Nx,
        Nv=[40],
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

    if solver_name == "rtsm":
        from bgk.solvers.rtsm import LinearRTSMSolver

        solver = LinearRTSMSolver(
            config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0, target_N_total=Np
        )
    elif solver_name == "ugkp":
        from bgk.solvers.ugkp import LinearUGKPSolver

        solver = LinearUGKPSolver(
            config,
            u_bg=u_bg_array,
            T_bg=T_bg_array,
            rho_bg=1.0,
            rho_in=rho_in,
            u_in=u_in,
            T_in=T_in,
        )
    elif solver_name == "vj":
        from bgk.solvers.vj import LinearVelocityJumpSolver

        solver = LinearVelocityJumpSolver(
            config,
            u_bg=u_bg_array,
            T_bg=T_bg_array,
            rho_bg=1.0,
            target_N_total=Np,
            use_null_collision=False,
        )

    RunnerClass = UGKPRunner if solver_name == "ugkp" else ParticleRunner
    runner = RunnerClass(
        config=config, solver=solver, problem=problem, Np=Np, m_ref=M_REF, N_inj=N_inj
    )
    runner.particles.flow_left = {"rho": rho_in, "u": u_in, "T": T_in, "vmax": vmax}

    t0 = time.time()
    sim = runner.run()
    cpu_time = time.time() - t0

    meta = {
        "solver": solver_name,
        "Kn": Kn,
        "Nx": Nx,
        "Np": Np,
        "N_INJ_TOTAL": float(N_INJ_TOTAL),
        "CFL": CFL,
        "vmax": vmax,
        "t_final": t_final,
        "type": "stochastic",
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
    ref_path = run_reference_sim()

    unique_cfls = sorted(
        list(set(cfl for cfls in SOLVER_CFL_CONFIGS.values() for cfl in cfls)),
        reverse=True,
    )
    _COLORS = {"rtsm": "#e41a1c", "ugkp": "#377eb8", "vj": "#4daf4a"}
    _MARKERS = ["o", "s", "^"]

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
        ax.set_title(f"CFL = {cfl} (Nx={FIXED_NX})")

        for s_idx, (solver, cfl_list) in enumerate(SOLVER_CFL_CONFIGS.items()):
            if cfl not in cfl_list:
                continue

            times, errors = [], []
            for ninj in NINJ_SWEEP:
                path = run_stochastic_sim(
                    (solver, 1e-3, FIXED_NX, None, ninj, cfl, 10.0, 4.0)
                )
                err, tc, xc, rhoc = get_l2_error_and_data(path, ref_path)
                times.append(tc)
                errors.append(err)
                ax.annotate(
                    _fmt(ninj),
                    (tc, err),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    alpha=0.7,
                )

                if ninj == NINJ_SWEEP[-1]:
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
    fig_wp.savefig("wp_stochastic.pdf")
    print("\nSaved Work-Precision diagram to wp_stochastic.pdf")

    # Save Diagnostic Plot
    if BOUNDARY_ONLY_EVAL:
        ax_prof.set_xlim(0.0, 0.05)
    ax_prof.set_xlabel("Position x")
    ax_prof.set_ylabel(r"Density $\rho$")
    ax_prof.set_title(f"Diagnostic Density Profiles (Highest Particles, Nx={FIXED_NX})")
    ax_prof.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig_prof.tight_layout()
    fig_prof.savefig("wp_stochastic_profiles.pdf")
    print("Saved boundary profiles to wp_stochastic_profiles.pdf")


if __name__ == "__main__":
    main()
