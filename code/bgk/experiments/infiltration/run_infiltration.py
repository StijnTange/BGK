import math
import os
import sys
import time

import h5py
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from bgk.config import Config, GridConfig, PhysicsConfig, TimeConfig
from bgk.core.particle_runner import ParticleRunner
from bgk.core.runner import Runner
from bgk.core.ugkp_runner import UGKPRunner

SIMULATIONS_DIR = os.path.join(os.path.dirname(__file__), "simulations")
STOCHASTIC_SOLVERS = {"rtsm", "ugkp", "vj"}


class PenetrationProblem:
    def __init__(self, xL, xR):
        self.name = "neutral_penetration"
        self.source_func = None
        self.x_bounds = (xL, xR)
        self.bc_type = "inflow/outflow"

    def rho_bg_func(self, x):
        """Exponential rise followed by a massive Gaussian ionization spike."""
        x_norm = x / self.x_bounds[1]

        n0 = 1.0
        alpha = 2.0
        base_density = n0 * np.exp(alpha * (1 - x_norm))

        peak_height = 25.0
        x_center = 0.015
        peak_width = 0.015

        ionization_peak = peak_height * np.exp(
            -0.5 * ((x_norm - x_center) / peak_width) ** 2
        )

        return base_density + ionization_peak

    def u_bg_func(self, x):
        """Concave velocity profile."""
        u_left = -1.0
        u_right = 1.0
        x_norm = x / self.x_bounds[1]

        k = 1.0
        linear_part = u_left + (u_right - u_left) * x_norm
        concave_bow = k * x_norm * (1.0 - x_norm)

        return linear_part + concave_bow

    def T_bg_func(self, x):
        """Convex temperature profile."""
        T_left = 0.1
        T_right = 10.0
        x_norm = x / self.x_bounds[1]

        return T_left + (T_right - T_left) * (x_norm**2)

    def f0_func(self, x_mesh, v_mesh):
        return np.zeros_like(x_mesh)


# --- Filename & String Helpers ---
def _fmt_sci(val):
    if val is None or str(val) == "None":
        return "None"
    return f"{float(val):.1e}".replace("-0", "-").replace("+0", "").replace("+", "")


def make_filename(run_tuple):
    solver = run_tuple[0]
    Kn = run_tuple[1]
    N_res = run_tuple[2]

    if solver in STOCHASTIC_SOLVERS:
        # Tuple: (solver, Kn, N_res, Np, N_INJ_TOTAL, CFL, vmax, t_final)
        _, _, _, Np, N_INJ_TOTAL, CFL, vmax, t_final = run_tuple
        return f"{solver}_Kn{_fmt_sci(Kn)}_Nx{N_res}_Np{_fmt_sci(Np)}_NinjTot{_fmt_sci(N_INJ_TOTAL)}_CFL{CFL}_vmax{vmax}_T{t_final}.h5"
    else:
        # Tuple: (solver, Kn, N_res, Nv, CFL, vmax, t_final)
        _, _, _, Nv, CFL, vmax, t_final = run_tuple
        return f"{solver}_Kn{_fmt_sci(Kn)}_Nx{N_res}_Nv{Nv}_CFL{CFL}_vmax{vmax}_T{t_final}.h5"


def save_simulation(filename, x, rho, u, T, q, cpu_time, meta):
    os.makedirs(SIMULATIONS_DIR, exist_ok=True)
    path = os.path.join(SIMULATIONS_DIR, filename)
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x, compression="gzip")
        f.create_dataset("rho", data=rho, compression="gzip")
        f.create_dataset("u", data=u, compression="gzip")
        f.create_dataset("T", data=T, compression="gzip")
        f.create_dataset("q", data=q, compression="gzip")
        f.create_dataset("cpu_time", data=np.array([cpu_time]))
        for k, v in meta.items():
            f.attrs[k] = v if v is not None else "None"
    print(f"  Saved -> {path}")


def main():
    # =========================================================================
    # 0.  Control Panel (Queue your runs here)
    # =========================================================================

    # Global Problem Settings
    xL, xR = 0.0, 1.0
    R_gas = 1.0
    REFLECTANCE_LEFT = 0.9

    # --- Bi-modal Inflow Parameters ---
    R_global = 0.95  # <--- NEW: 95% recycling, 5% absorption at the wall

    # These dictate the SHAPE of the returning gas (Should always sum to 1.0)
    R_R = 0.6  # Fast back-scattered fraction
    R_T = 0.4  # Slow thermal desorption fraction

    u_fast = 2.0
    T_fast = 1.0
    u_slow = 0.2
    T_slow = 0.1
    # ----------------------------------

    # Stochastic Tuple:   ("solver", Kn, N_res, Np, N_INJ_TOTAL, CFL, vmax, t_final)
    # Deterministic Tuple: ("solver", Kn, N_res, Nv, CFL, vmax, t_final)
    RUNS = [
        ("hybrid", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
        ("ugks", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
        ("strang", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
        ("sl", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
        ("fvm", 5e-1, 1000, 80, 0.9, 20.0, 10.0),
        # ("hybrid", 5e-1, 21, 80, 0.9, 20.0, 10.0),
        # ("ugkp", 5e-1, 30, None, int(1e5), 0.9, 20.0, 10.0),
        # ("rtsm", 5e-1, 30, None, int(1e5), 0.9, 20.0, 10.0),
        # ("vj", 5e-1, 30, None, int(1e5), 0.9, 20.0, 10.0),
    ]

    GENERATE_FINE_REFERENCE = False
    SKIP_EXISTING_FILES = False
    REFERENCE = ("fvm", 1e-2, 1000, 80, 0.9, 20.0, 5.0)

    # =========================================================================
    print("--- Starting Neutral Penetration Batch Runner ---")

    queue = []
    if GENERATE_FINE_REFERENCE and REFERENCE is not None:
        queue.append(REFERENCE)
    queue.extend(RUNS)

    for run in queue:
        solver_name = run[0]
        Kn = run[1]
        N_res = run[2]
        is_stoch = solver_name in STOCHASTIC_SOLVERS

        # Unpack tuple based on solver type
        if is_stoch:
            _, _, _, Np, N_INJ_TOTAL, CFL, vmax, t_final = run
            Nv = [40]  # Dummy value
        else:
            _, _, _, Nv, CFL, vmax, t_final = run
            Np, N_INJ_TOTAL = None, None
            Nv = [Nv]

        filename = make_filename(run)
        if solver_name == "ugks" and run == REFERENCE:
            filename = filename.replace("ugks_", "ugks_ref_")

        out_path = os.path.join(SIMULATIONS_DIR, filename)
        if os.path.exists(out_path) and SKIP_EXISTING_FILES:
            print(f"\nAlready exists, skipping: {filename}")
            continue

        print(f"\n--- Solving {filename} ---")

        L_domain = xR - xL
        if solver_name in []:
            Nx = N_res
            Nc = None
            dx = L_domain / (Nx - 1)
        elif solver_name in [
            "strang",
            "sl",
            "hybrid",
            "fvm",
            "ugks",
            "ugkp",
            "rtsm",
            "vj",
        ]:
            Nx = None
            Nc = N_res
            dx = L_domain / Nc
        else:
            raise ValueError(f"Unknown solver type: {solver_name}")

        dt = CFL * dx / vmax

        # Setup Configs
        problem = PenetrationProblem(xL, xR)
        physics_conf = PhysicsConfig(
            Kn=Kn,
            problem_name=problem.name,
        )
        physics_conf.reflectance_left = REFLECTANCE_LEFT
        grid_conf = GridConfig(
            xL=xL,
            xR=xR,
            Nx=Nx,
            Nc=Nc,
            Nv=Nv,
            dim_v=1,
            vmax=vmax,
            vmin=-vmax,
            bc_type=problem.bc_type,
        )
        time_conf = TimeConfig(t_final=t_final, dt=dt, CFL=CFL)
        config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

        if Nc is not None:
            x_centers = np.linspace(xL + dx / 2, xR - dx / 2, Nc)
        else:
            x_centers = np.linspace(xL, xR, Nx)

        u_bg_array = problem.u_bg_func(x_centers)
        T_bg_array = problem.T_bg_func(x_centers)
        rho_bg_array = problem.rho_bg_func(x_centers)

        # --- Calculate Dynamic Recycling Source (Horsten et al. method) ---

        # 1. How much plasma is crashing into the left wall (x=0)?
        # plasma_flux_at_wall = rho_bg_array[0] * abs(u_bg_array[0])
        # --- Calculate Dynamic Recycling Source (Horsten et al. method) ---

        # 1. How much plasma is crashing into the left wall (x=0)?
        # Evaluated exactly at the boundary using the analytical functions
        rho_bound = problem.rho_bg_func(0.0)
        u_bound = problem.u_bg_func(0.0)
        plasma_flux_at_wall = rho_bound * abs(u_bound)

        # 2. Apply absorption! How much mass is actually returning?
        target_neutral_flux = R_global * plasma_flux_at_wall

        # 3. What would the neutral flux be if rho_in was 1.0? (Unit Flux)
        U_fast = u_fast / np.sqrt(2.0 * R_gas * T_fast)
        unit_flux_fast = (
            (R_R)
            * np.sqrt(R_gas * T_fast / (2.0 * np.pi))
            * (
                np.exp(-(U_fast**2))
                + np.sqrt(np.pi) * U_fast * (1.0 + math.erf(U_fast))
            )
        )

        U_slow = u_slow / np.sqrt(2.0 * R_gas * T_slow)
        unit_flux_slow = (
            (R_T)
            * np.sqrt(R_gas * T_slow / (2.0 * np.pi))
            * (
                np.exp(-(U_slow**2))
                + np.sqrt(np.pi) * U_slow * (1.0 + math.erf(U_slow))
            )
        )

        total_unit_flux = unit_flux_fast + unit_flux_slow

        # 4. Calculate the EXACT required physical density parameter!
        rho_in = target_neutral_flux / total_unit_flux

        # 5. Now calculate your actual exact fluxes for the particle solvers
        exact_flux = rho_in * total_unit_flux
        # -----------------------------------------------------------------

        # Calculate N_inj and M_REF for Stochastic solvers as normal
        M_REF, N_inj = None, None
        if is_stoch:
            total_inflow_mass = exact_flux * t_final
            N_inj = float(N_INJ_TOTAL * (dt / t_final))
            M_REF = (exact_flux * dt) / N_inj

            print(f"    Plasma Flux at Wall: {plasma_flux_at_wall:.4f}")
            print(f"    Calculated rho_in  : {rho_in:.4f}")
            print(f"    Total Expected Mass: {total_inflow_mass:.4f}")

        # Initialize Solver
        if solver_name == "ugks":
            from bgk.solvers.ugks import LinearUGKSSolver

            solver = LinearUGKSSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
            )
        elif solver_name == "fvm":
            from bgk.solvers.fvm import LinearFVMSolver

            solver = LinearFVMSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
            )
        elif solver_name == "sl":
            from bgk.solvers.sl import LinearSLSolver

            solver = LinearSLSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
            )
        elif solver_name == "hybrid":
            from bgk.solvers.hybrid import LinearHybridSolver

            solver = LinearHybridSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
            )
        elif solver_name == "strang":
            from bgk.solvers.splitting import LinearStrangSolver

            solver = LinearStrangSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=rho_bg_array
            )
        elif solver_name == "rtsm":
            from bgk.solvers.rtsm import LinearRTSMSolver

            solver = LinearRTSMSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=rho_bg_array,
                target_N_total=Np,
            )
        elif solver_name == "ugkp":
            from bgk.solvers.ugkp import LinearUGKPSolver

            # We don't need to pass the detailed inflow stats to the linear UGKP solver anymore,
            # it safely ignores them for the macroscopic ghost and relies on particle injection.
            solver = LinearUGKPSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=rho_bg_array,
                rho_in=rho_in,
            )
        elif solver_name == "vj":
            from bgk.solvers.vj import LinearVelocityJumpSolver

            solver = LinearVelocityJumpSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=rho_bg_array,
                target_N_total=Np,
                use_null_collision=False,
            )
        else:
            raise ValueError(f"Solver {solver_name} not configured in runner.")

        # Initialize Runner
        if is_stoch:
            RunnerClass = UGKPRunner if solver_name == "ugkp" else ParticleRunner
            runner = RunnerClass(
                config=config,
                solver=solver,
                problem=problem,
                Np=Np,
                m_ref=M_REF,
                N_inj=N_inj,
            )

            runner.particles.flow_left = [
                {"rho": rho_in * R_R, "u": u_fast, "T": T_fast, "vmax": vmax},
                {
                    "rho": rho_in * R_T,
                    "u": u_slow,
                    "T": T_slow,
                    "vmax": vmax,
                },  # <--- UPDATED
            ]
        else:
            runner = Runner(config=config, solver=solver, problem=problem)
            v = runner.grid.v

            pref_fast = (rho_in * R_R) / np.sqrt(2.0 * np.pi * R_gas * T_fast)
            f_fast = pref_fast * np.exp(-((v - u_fast) ** 2) / (2.0 * R_gas * T_fast))

            pref_slow = (rho_in * R_T) / np.sqrt(2.0 * np.pi * R_gas * T_slow)
            f_slow = pref_slow * np.exp(
                -((v - u_slow) ** 2) / (2.0 * R_gas * T_slow)
            )  # <--- UPDATED

            f_inflow = f_fast + f_slow

            runner.df.f_flow_left = f_inflow
            runner.solver.f_flow_left = f_inflow

        # Execute
        t0 = time.time()
        sim = runner.run()
        cpu_time = time.time() - t0
        print(f"[{solver_name.upper()}] completed in {cpu_time:.2f}s")

        # Compile Metadata
        meta = {
            "solver": solver_name,
            "Kn": float(Kn),
            "N_res": int(N_res),
            "CFL": float(CFL),
            "vmax": float(vmax),
            "t_final": float(t_final),
            "type": "stochastic" if is_stoch else "deterministic",
        }

        if Nx is not None:
            meta["Nx"] = int(Nx)
        if Nc is not None:
            meta["Nc"] = int(Nc)

        if is_stoch:
            meta["Np"] = Np
            meta["N_INJ_TOTAL"] = float(N_INJ_TOTAL)
            meta["N_inj_per_step"] = int(N_inj)
            meta["m_ref"] = float(M_REF)
        else:
            meta["Nv"] = int(Nv[0])

        w = 1
        save_simulation(
            filename,
            x=sim.x,
            rho=np.mean(sim.rho[-w:], axis=0).flatten(),
            u=np.mean(sim.u[-w:], axis=0).flatten(),
            T=np.mean(sim.T[-w:], axis=0).flatten(),
            q=np.mean(sim.q[-w:], axis=0).flatten(),
            cpu_time=cpu_time,
            meta=meta,
        )

    print("\nDone. All simulations saved.")


if __name__ == "__main__":
    main()
