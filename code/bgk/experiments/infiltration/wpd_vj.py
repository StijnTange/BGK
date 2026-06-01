import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Ensure paths are set up correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner

# =============================================================================
# PROBLEM DEFINITION
# =============================================================================


class PenetrationProblem:
    def __init__(self, xL, xR):
        self.name = "neutral_penetration"
        self.x_bounds = (xL, xR)
        self.bc_type = "inflow/outflow"

    def u_bg_func(self, x):
        return -1.0 * (x / self.x_bounds[1])

    def T_bg_func(self, x):
        return 0.1 + 9.9 * (x / self.x_bounds[1])

    def f0_func(self, x_mesh, v_mesh):
        return np.zeros_like(x_mesh)


# =============================================================================
# MAIN WPD SCRIPT (VARYING PARTICLES)
# =============================================================================


def main():
    print("--- Starting VJ Work-Precision Diagram (Varying Particles) ---")

    # Domain & Plasma conditions
    xL, xR = 0.0, 1.0
    rho_in, u_in, T_in = 1.0, 2.0, 0.1
    R_gas = 1.0
    vmax = 10.0
    t_final = 4.0
    Kn = 1e-3

    Nx_stoch = 500
    REFLECTANCE_LEFT = 1.0

    # FIX dt to isolate the statistical noise from the structural error
    dt_val = 0.01

    N_VALUES = [1e4, 5e4, 1e5, 2.5e5]

    # Data collection dictionaries
    results = {
        "vj_ray": {"cpu": [], "err": [], "val": [], "x": [], "rho": []},
        "vj_null": {"cpu": [], "err": [], "val": [], "x": [], "rho": []},
    }

    problem = PenetrationProblem(xL, xR)
    physics_conf = PhysicsConfig(
        Kn=Kn, problem_name=problem.name, R=R_gas, reflectance_left=REFLECTANCE_LEFT
    )

    # -------------------------------------------------------------------------
    # 1. GENERATE FINE REFERENCE SOLUTION (Deterministic UGKS)
    # -------------------------------------------------------------------------
    print("\n[1/3] Generating Fine Reference Solution (UGKS)...")
    Nx_ref, Nv_ref, CFL_ref = 500, 60, 0.9
    dx_ref = (xR - xL) / Nx_ref
    dt_ref = CFL_ref * dx_ref / vmax

    grid_conf_ref = GridConfig(
        xL=xL,
        xR=xR,
        Nx=None,
        Nc=Nx_ref,
        Nv=[Nv_ref],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type=problem.bc_type,
    )
    time_conf_ref = TimeConfig(t_final=t_final, dt=dt_ref, CFL=CFL_ref)
    config_ref = Config(grid=grid_conf_ref, time=time_conf_ref, physics=physics_conf)

    x_centers_ref = 0.5 * (
        np.linspace(xL, xR, Nx_ref + 1)[:-1] + np.linspace(xL, xR, Nx_ref + 1)[1:]
    )
    from bgk.solvers.ugks import LinearUGKSSolver

    solver_ref = LinearUGKSSolver(
        config_ref,
        u_bg=problem.u_bg_func(x_centers_ref),
        T_bg=problem.T_bg_func(x_centers_ref),
        rho_bg=1.0,
    )

    runner_ref = Runner(config=config_ref, solver=solver_ref, problem=problem)
    v_ref = runner_ref.grid.v
    pref_ref = rho_in / np.sqrt(2.0 * np.pi * R_gas * T_in)
    runner_ref.df.f_flow_left = pref_ref * np.exp(
        -((v_ref - u_in) ** 2) / (2.0 * R_gas * T_in)
    )

    sim_ref = runner_ref.run()

    # Create interpolation function for the reference density
    ref_x = sim_ref.x
    ref_rho = np.mean(sim_ref.rho[-1:], axis=0).flatten()
    interp_func = interp1d(ref_x, ref_rho, kind="cubic", fill_value="extrapolate")

    # -------------------------------------------------------------------------
    # 2. RUN VELOCITY JUMP SOLVERS
    # -------------------------------------------------------------------------
    print("\n[2/3] Running Velocity Jump Simulations...")

    x_centers = 0.5 * (
        np.linspace(xL, xR, Nx_stoch + 1)[:-1] + np.linspace(xL, xR, Nx_stoch + 1)[1:]
    )
    u_bg_array = problem.u_bg_func(x_centers)
    T_bg_array = problem.T_bg_func(x_centers)

    grid_conf = GridConfig(
        xL=xL,
        xR=xR,
        Nx=None,
        Nc=Nx_stoch,
        Nv=[40],
        dim_v=1,
        vmax=vmax,
        vmin=-vmax,
        bc_type=problem.bc_type,
    )
    time_conf = TimeConfig(t_final=t_final, dt=dt_val, CFL=0.9)
    config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

    # Pre-calculate flux to avoid recomputing in the loop
    v_th = np.sqrt(2.0 * R_gas * T_in)
    U = u_in / v_th
    exact_flux = (
        rho_in
        * np.sqrt(R_gas * T_in / (2.0 * np.pi))
        * (np.exp(-(U**2)) + np.sqrt(np.pi) * U * (1.0 + math.erf(U)))
    )

    for solver_name in ["vj_ray", "vj_null"]:
        use_null = solver_name == "vj_null"

        for n_inj_tot in N_VALUES:
            print(f"  -> {solver_name.upper()} | N = {n_inj_tot:.1e} ", end="")
            sys.stdout.flush()

            from bgk.solvers.vj import LinearVelocityJumpSolver

            solver = LinearVelocityJumpSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=1.0,
                use_null_collision=use_null,
            )

            # Exact fractional boundary injection math
            N_inj_per_step = float(n_inj_tot * (dt_val / t_final))
            M_REF = (exact_flux * dt_val) / N_inj_per_step

            runner = ParticleRunner(
                config=config,
                solver=solver,
                problem=problem,
                Np=None,
                m_ref=M_REF,
                N_inj=N_inj_per_step,
            )
            runner.particles.flow_left = {
                "rho": rho_in,
                "u": u_in,
                "T": T_in,
                "vmax": vmax,
            }

            t0 = time.time()
            sim_stoch = runner.run()
            cpu_time = time.time() - t0

            # Extract final state and calculate error
            sim_x = sim_stoch.x
            sim_rho = np.mean(sim_stoch.rho[-1:], axis=0).flatten()

            ref_mapped = interp_func(sim_x)
            err_l2 = np.linalg.norm(sim_rho - ref_mapped) / np.linalg.norm(ref_mapped)

            results[solver_name]["cpu"].append(cpu_time)
            results[solver_name]["err"].append(err_l2)
            results[solver_name]["val"].append(n_inj_tot)
            results[solver_name]["x"].append(sim_x)
            results[solver_name]["rho"].append(sim_rho)

            print(f"... done in {cpu_time:.2f}s (Error: {err_l2:.4f})")

    # -------------------------------------------------------------------------
    # 3. PLOT RESULTS
    # -------------------------------------------------------------------------
    print("\n[3/3] Generating Plot...")

    width, _ = tp.get_figsize(fraction=1.0)
    fig, (ax_wpd, ax_prof) = plt.subplots(
        1, 2, figsize=(width, width * 0.45), constrained_layout=True
    )

    _COLORS = {"vj_ray": "#377eb8", "vj_null": "#984ea3"}
    _MARKERS = {"vj_ray": "s", "vj_null": "D"}
    _LABELS = {"vj_ray": "VJ (Ray Tracing)", "vj_null": "VJ (Null-Collision)"}

    # Plot Reference Spatial Profile
    ax_prof.plot(
        ref_x, ref_rho, "k-", linewidth=2.0, zorder=5, label="Reference (UGKS)"
    )

    for solver, metrics in results.items():
        color = _COLORS[solver]
        marker = _MARKERS[solver]
        lbl = _LABELS[solver]

        # Sort by CPU time for connecting lines in WPD
        sort_idx = np.argsort(metrics["cpu"])
        cpu_sorted = np.array(metrics["cpu"])[sort_idx]
        err_sorted = np.array(metrics["err"])[sort_idx]
        val_sorted = np.array(metrics["val"])[sort_idx]

        # WPD Plot
        ax_wpd.plot(
            cpu_sorted,
            err_sorted,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            label=lbl,
        )

        for i in range(len(cpu_sorted)):
            n_val = val_sorted[i]
            label_text = f"N={n_val:.0e}".replace("+0", "").replace("+", "")
            ax_wpd.text(
                cpu_sorted[i] * 1.05,
                err_sorted[i],
                label_text,
                fontsize=8,
                color=color,
                verticalalignment="center",
            )

        # Spatial Profile Plot
        for i in range(len(metrics["val"])):
            n_val = metrics["val"][i]
            x_arr = metrics["x"][i]
            m_arr = metrics["rho"][i]

            # Line styles based on particle count
            if n_val <= 5e4:
                ls = ":"
            elif n_val <= 1e5:
                ls = "-."
            elif n_val <= 2.5e5:
                ls = "--"
            else:
                ls = "-"

            me = max(1, len(x_arr) // 10)

            # Use concise labels in the legend to save space
            prof_lbl = f"{lbl} (N={n_val:.0e})".replace("+0", "").replace("+", "")
            ax_prof.plot(
                x_arr,
                m_arr,
                color=color,
                linestyle=ls,
                linewidth=1.2,
                marker=marker,
                ms=3,
                markevery=me,
                alpha=0.8,
                label=prof_lbl,
                zorder=3,
            )

    # Format WPD
    ax_wpd.set_xscale("log")
    ax_wpd.set_yscale("log")
    ax_wpd.set_xlabel("CPU Time (seconds)")
    ax_wpd.set_ylabel(r"Relative $L_2$ Error ($\rho$)")
    ax_wpd.set_title("Work-Precision Diagram (Varying $N$)")
    ax_wpd.grid(True, which="major", linestyle="-", alpha=0.5)
    ax_wpd.grid(True, which="minor", linestyle="--", alpha=0.2)
    ax_wpd.legend(frameon=True, facecolor="white", edgecolor="black")

    # Format Profile
    ax_prof.set_xlim(0.0, 0.5)  # Zoomed in to see the noise differences
    ax_prof.set_xlabel(r"Position $x$")
    ax_prof.set_ylabel(r"Density $\rho$")
    ax_prof.set_title(f"Spatial Distribution at t={t_final}")
    ax_prof.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_prof.legend(
        fontsize="x-small", ncol=1, frameon=True, facecolor="white", edgecolor="black"
    )

    # Show plot without saving
    plt.show()


if __name__ == "__main__":
    main()
