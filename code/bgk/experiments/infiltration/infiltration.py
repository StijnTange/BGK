import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import bgk.thesis_plots as tp
from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner


class PenetrationProblem:
    def __init__(self, xL, xR):
        self.name = "neutral_penetration"
        self.source_func = None
        self.x_bounds = (xL, xR)
        self.bc_type = "inflow/outflow"

    def u_bg_func(self, x):
        return -1.0 * (x / self.x_bounds[1])

    def T_bg_func(self, x):
        return 0.1 + 9.9 * (x / self.x_bounds[1])

    def f0_func(self, x_mesh, v_mesh):
        return np.zeros_like(x_mesh)


def main():
    # =========================================================================
    # 0. Control Panel
    # =========================================================================
    SOLVERS_TO_TEST = ["ugks"]
    REFLECTANCE_LEFT = 1.0

    Kn = 1e-1

    # --- Reference Solution Toggle ---
    GENERATE_FINE_REFERENCE = False
    Nx_ref = 500  # High resolution for the reference
    CFL_ref = 0.9

    print(f"--- Starting Neutral Penetration Comparison (Kn={Kn}) ---")

    # --- EXPERIMENT CONFIGURATION ---
    xL, xR = 0.0, 1.0
    CFL = 0.9
    vmax = 10.0
    t_final = 10.0
    Nv = 60
    Nx = 200
    N_particles = int(1e6)

    problem = PenetrationProblem(xL, xR)
    physics_conf = PhysicsConfig(
        Kn=Kn, problem_name=problem.name, reflectance_left=REFLECTANCE_LEFT
    )

    # Pre-calculate spatial background arrays for the test grid
    dx = (problem.x_bounds[1] - problem.x_bounds[0]) / Nx
    dt = CFL * dx / vmax
    grid_points = np.linspace(xL, xR, Nx + 1)
    x_centers = 0.5 * (grid_points[:-1] + grid_points[1:])
    u_bg_array = problem.u_bg_func(x_centers)
    T_bg_array = problem.T_bg_func(x_centers)

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

    # Dictionary to hold the final results
    results = {
        name: {"rho": None, "u": None, "T": None, "q": None, "time": 0.0}
        for name in SOLVERS_TO_TEST
    }

    # =========================================================================
    # 1. Generate Fine Reference Solution (UGKS)
    # =========================================================================
    reference_data = None
    if GENERATE_FINE_REFERENCE:
        print(f"\n--- Generating Fine Reference Solution (UGKS, Nx={Nx_ref}) ---")
        dx_ref = (xR - xL) / Nx_ref
        dt_ref = CFL_ref * dx_ref / vmax

        grid_points_ref = np.linspace(xL, xR, Nx_ref + 1)
        x_centers_ref = 0.5 * (grid_points_ref[:-1] + grid_points_ref[1:])
        u_bg_array_ref = problem.u_bg_func(x_centers_ref)
        T_bg_array_ref = problem.T_bg_func(x_centers_ref)

        grid_conf_ref = GridConfig(
            xL=xL,
            xR=xR,
            Nx=None,
            Nc=Nx_ref,
            Nv=[Nv],
            dim_v=1,
            vmax=vmax,
            vmin=-vmax,
            bc_type=problem.bc_type,
        )
        time_conf_ref = TimeConfig(t_final=t_final, dt=dt_ref, CFL=CFL_ref)
        config_ref = Config(
            grid=grid_conf_ref, time=time_conf_ref, physics=physics_conf
        )

        from bgk.solvers.ugks import LinearUGKSSolver

        solver_ref = LinearUGKSSolver(
            config_ref, u_bg=u_bg_array_ref, T_bg=T_bg_array_ref, rho_bg=1.0
        )
        runner_ref = Runner(config=config_ref, solver=solver_ref, problem=problem)

        rho_in, u_in, T_in = 1.0, 2.0, 0.1
        v_ref = runner_ref.grid.v
        pref_ref = rho_in / np.sqrt(2.0 * np.pi * config_ref.physics.R * T_in)
        f_inflow_ref = pref_ref * np.exp(
            -((v_ref - u_in) ** 2) / (2.0 * config_ref.physics.R * T_in)
        )

        if hasattr(runner_ref, "df"):
            runner_ref.df.f_flow_left = f_inflow_ref
        if hasattr(runner_ref, "solver"):
            runner_ref.solver.f_flow_left = f_inflow_ref

        t_start_ref = time.time()
        sim_ref = runner_ref.run()
        print(
            f"Reference generation completed in "
            f"{time.time() - t_start_ref:.2f} seconds."
        )

        steady_window_ref = min(10, sim_ref.rho.shape[0])
        reference_data = {
            "x": sim_ref.x,
            "rho": np.mean(sim_ref.rho[-steady_window_ref:, :], axis=0),
            "u": np.mean(sim_ref.u[-steady_window_ref:, :], axis=0),
            "T": np.mean(sim_ref.T[-steady_window_ref:, :], axis=0),
            "q": np.mean(sim_ref.q[-steady_window_ref:, :], axis=0),
        }

    # =========================================================================
    # 2. Run Test Simulations
    # =========================================================================
    for solver_name in SOLVERS_TO_TEST:
        print(f"\n--- Solving with {solver_name.upper()} ---")

        is_particle_method = False
        is_ugkp = False

        if solver_name == "strang":
            from bgk.solvers.splitting import LinearStrangSolver

            solver = LinearStrangSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )
        elif solver_name == "sl":
            from bgk.solvers.sl import LinearSLSolver

            solver = LinearSLSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )
        elif solver_name == "fvm":
            from bgk.solvers.fvm import LinearFVMSolver

            solver = LinearFVMSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )
        elif solver_name == "ugks":
            from bgk.solvers.ugks import LinearUGKSSolver

            solver = LinearUGKSSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )
        elif solver_name == "hybrid":
            from bgk.solvers.hybrid import LinearHybridSolver

            solver = LinearHybridSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=1.0,
            )
        elif solver_name == "rtsm":
            from bgk.solvers.rtsm import LinearRTSMSolver

            solver = LinearRTSMSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=1.0,
                target_N_total=N_particles,
            )
            is_particle_method = True
        elif solver_name == "vj":
            from bgk.solvers.vj import LinearVelocityJumpSolver

            solver = LinearVelocityJumpSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=1.0,
                target_N_total=N_particles,
            )
            is_particle_method = True
        elif solver_name == "ugkp":
            from bgk.solvers.ugkp import LinearUGKPSolver

            solver = LinearUGKPSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )
            is_ugkp = True
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        if is_particle_method:
            runner = ParticleRunner(
                config=config,
                solver=solver,
                problem=problem,
                N_particles_total=N_particles,
            )
        elif is_ugkp:
            runner = UGKPRunner(
                config=config,
                solver=solver,
                problem=problem,
                N_particles_total=N_particles,
            )
        else:
            runner = Runner(config=config, solver=solver, problem=problem)

        rho_in, u_in, T_in = 1.0, 2.0, 0.1

        if is_particle_method or is_ugkp:
            runner.particles.flow_left = {
                "rho": rho_in,
                "u": u_in,
                "T": T_in,
                "vmax": config.grid.vmax,
            }
        else:
            v = runner.grid.v
            pref = rho_in / np.sqrt(2.0 * np.pi * config.physics.R * T_in)
            f_inflow = pref * np.exp(
                -((v - u_in) ** 2) / (2.0 * config.physics.R * T_in)
            )

            if hasattr(runner, "df"):
                runner.df.f_flow_left = f_inflow
            if hasattr(runner, "solver"):
                runner.solver.f_flow_left = f_inflow

        t_start = time.time()
        sim = runner.run()
        t_end = time.time()

        cpu_time = t_end - t_start
        print(f"[{solver_name.upper()}] completed in {cpu_time:.2f} seconds.")

        steady_state_window = min(10, sim.rho.shape[0])
        # After — flatten immediately so shape is always (Nx,)
        results[solver_name]["rho"] = np.mean(
            sim.rho[-steady_state_window:, :], axis=0
        ).flatten()
        results[solver_name]["u"] = np.mean(
            sim.u[-steady_state_window:, :], axis=0
        ).flatten()
        results[solver_name]["T"] = np.mean(
            sim.T[-steady_state_window:, :], axis=0
        ).flatten()
        results[solver_name]["q"] = np.mean(
            sim.q[-steady_state_window:, :], axis=0
        ).flatten()
        results[solver_name]["time"] = cpu_time

        x_grid = sim.x

    # =========================================================================
    # 3. Plot Comparisons (Thesis Format - 2x2 Grid)
    # =========================================================================
    print("\nGenerating 2x2 comparison plot...")
    plt.close("all")

    width, _ = tp.get_figsize(fraction=1.0)
    fig, axs = plt.subplots(2, 2, figsize=(width, width * 0.7), constrained_layout=True)

    plot_configs = [
        (
            "rho",
            r"Neutral Density $\rho$",
            0,
            0,
            None,
            {"pos": [0.55, 0.1, 0.4, 0.8], "xlim": (0.0, 0.08)},
        ),
        ("u", r"Velocity $u$", 0, 1, u_bg_array, None),
        ("T", r"Temperature $T$", 1, 0, T_bg_array, None),
        (
            "q",
            r"Heat Flux $q$",
            1,
            1,
            None,
            {"pos": [0.55, 0.1, 0.4, 0.8], "xlim": (0.0, 0.05)},
        ),
    ]

    for data_key, ylabel, row, col, bg_data, inset_conf in plot_configs:
        ax = axs[row, col]
        axins = ax.inset_axes(inset_conf["pos"]) if inset_conf else None

        # 1. Plot Fine Reference Solution first (so it sits behind test lines)
        if reference_data is not None:
            ax.plot(
                reference_data["x"],
                reference_data[data_key].flatten(),
                "k--",
                linewidth=1.5,
                zorder=2,
                label="Reference (UGKS Fine)",
            )
            if axins:
                axins.plot(
                    reference_data["x"],
                    reference_data[data_key].flatten(),
                    "k--",
                    linewidth=1.5,
                    zorder=2,
                )

        # 2. Plot the plasma background if applicable
        if bg_data is not None:
            # Switched to dotted gray to distinguish from the reference line
            ax.plot(
                x_centers,
                bg_data,
                color="gray",
                linestyle=":",
                alpha=0.7,
                linewidth=1.5,
                zorder=1,
                label="Plasma Background",
            )
            if axins:
                axins.plot(
                    x_centers,
                    bg_data,
                    color="gray",
                    linestyle=":",
                    alpha=0.7,
                    linewidth=1.5,
                    zorder=1,
                )

        # 3. Plot each solver's results
        for name in SOLVERS_TO_TEST:
            data = results[name][data_key]
            elapsed = results[name]["time"]

            label_str = f"{name.upper()} ({elapsed:.2f}s)"
            ax.plot(x_grid, data.flatten(), label=label_str, zorder=3)

            if axins:
                axins.plot(x_grid, data.flatten(), zorder=3)

        # 4. Handle the Inset Box logic
        if axins:
            zoom_x_min, zoom_x_max = inset_conf["xlim"]
            axins.set_xlim(zoom_x_min, zoom_x_max)

            mask = (x_grid >= zoom_x_min) & (x_grid <= zoom_x_max)

            if not np.any(mask):
                y_min = min([np.min(results[n][data_key]) for n in SOLVERS_TO_TEST])
                y_max = max([np.max(results[n][data_key]) for n in SOLVERS_TO_TEST])
            else:
                y_min = min(
                    [np.min(results[n][data_key][mask]) for n in SOLVERS_TO_TEST]
                )
                y_max = max(
                    [np.max(results[n][data_key][mask]) for n in SOLVERS_TO_TEST]
                )

            pad = (y_max - y_min) * 0.05
            if pad == 0:
                pad = 0.05

            axins.set_ylim(y_min - pad, y_max + pad)
            axins.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)
            axins.set_xticks([])
            axins.set_yticks([])

            rect, connectors = ax.indicate_inset_zoom(
                axins, edgecolor="black", alpha=0.4, linewidth=1.0
            )
            for connector in connectors:
                connector.set_visible(True)

        if data_key == "q":
            ax.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.5)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Position $x$")
        ax.grid(True, which="both", alpha=0.4, linestyle="--", linewidth=0.5)

    # Place a single legend in the top-RIGHT plot (Velocity)
    axs[0, 1].legend(frameon=True, facecolor="white", edgecolor="black")

    # Generate a dynamic filename based on solvers and Kn
    solver_str = "_".join(SOLVERS_TO_TEST)
    kn_str = f"{Kn:.0e}".replace("-0", "-")
    filename = f"latex/thesis/figures/linear/infiltration/comparison_{solver_str}_Kn_{kn_str}_small_density.pdf"

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    tp.save_plot(filename)
    print(f"\nSaved plot to: {filename}")


if __name__ == "__main__":
    main()
