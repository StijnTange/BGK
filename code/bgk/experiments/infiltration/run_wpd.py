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

SIMULATIONS_DIR = os.path.join(os.path.dirname(__file__), "simulations_wpd")

# Define solver categories for distinct tuple parsing
DET_SOLVERS = {"ugks", "fvm", "sl", "strang", "hybrid"}
STOCH_DT_SOLVERS = {"rtsm", "vj"}
STOCH_CFL_SOLVERS = {"ugkp"}


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


def _fmt_sci(val):
    if val is None or str(val) == "None":
        return "None"
    return f"{float(val):.1e}".replace("-0", "-", 1).replace("+0", "").replace("+", "")


def make_filename(run_tuple, is_ref=False):
    solver = run_tuple[0]
    Kn = run_tuple[1]
    Nx = run_tuple[2]

    if solver in STOCH_DT_SOLVERS:
        _, _, _, N_INJ_TOTAL, dt, vmax, t_final = run_tuple
        fname = f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_NinjTot{_fmt_sci(N_INJ_TOTAL)}_dt{_fmt_sci(dt)}_T{t_final}.h5"
    elif solver in STOCH_CFL_SOLVERS:
        _, _, _, N_INJ_TOTAL, CFL, vmax, t_final = run_tuple
        fname = f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_NinjTot{_fmt_sci(N_INJ_TOTAL)}_CFL{CFL}_T{t_final}.h5"
    else:
        _, _, _, Nv, CFL, vmax, t_final = run_tuple
        fname = f"{solver}_Kn{_fmt_sci(Kn)}_Nx{Nx}_Nv{Nv}_CFL{CFL}_T{t_final}.h5"

    if is_ref:
        fname = fname.replace(f"{solver}_", f"{solver}_ref_")
    return fname


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
    xL, xR = 0.0, 1.0
    rho_in, u_in, T_in = 1.0, 2.0, 0.1
    R_gas = 1.0
    REFLECTANCE_LEFT = 1.0

    # =========================================================================
    # WPD QUEUE: Hybrid Definitions
    # =========================================================================

    # Deterministic: Defined by CFL -> ("solver", Kn, Nx, Nv, CFL, vmax, t_final)
    # UGKP: Defined by CFL -> ("ugkp", Kn, Nx, N_INJ_TOTAL, CFL, vmax, t_final)
    # Stochastic: Defined by dt -> ("solver", Kn, Nx, N_INJ_TOTAL, dt, vmax, t_final)

    RUNS = [
        # dt = 0.1
        ("rtsm", 1e-3, 100, 1e5, 0.1, 10.0, 4.0),
        ("vj", 1e-3, 100, 1e5, 0.1, 10.0, 4.0),
        # dt. = 0.05
        ("rtsm", 1e-3, 100, 1e5, 0.05, 10.0, 4.0),
        ("vj", 1e-3, 100, 1e5, 0.05, 10.0, 4.0),
        # dt = 0.01
        ("rtsm", 1e-3, 100, 1e5, 0.01, 10.0, 4.0),
        ("vj", 1e-3, 100, 1e5, 0.01, 10.0, 4.0),
        # dt = 0.0025
        ("rtsm", 1e-3, 100, 1e5, 0.0025, 10.0, 4.0),
        ("vj", 1e-3, 100, 1e5, 0.0025, 10.0, 4.0),
        # dt = 0.0005
        ("rtsm", 1e-3, 100, 1e5, 0.0005, 10.0, 4.0),
        ("vj", 1e-3, 100, 1e5, 0.0005, 10.0, 4.0),
        # dt = 0.0001
        ("rtsm", 1e-3, 100, 1e5, 0.0001, 10.0, 4.0),
        ("vj", 1e-3, 100, 1e5, 0.0001, 10.0, 4.0),
    ]

    GENERATE_FINE_REFERENCE = False
    SKIP_EXISTING_FILES = True

    Nx_ref = 4000
    Nv_ref = 60
    dx_ref = (xR - xL) / Nx_ref
    v_max = 10.0
    CFL_ref = 0.9
    dt_ref = CFL_ref * dx_ref / v_max
    t_final = 4.0
    REFERENCE = ("ugks", 1e-3, Nx_ref, Nv_ref, dt_ref, v_max, t_final)

    # =========================================================================

    queue = []
    if GENERATE_FINE_REFERENCE and REFERENCE is not None:
        queue.append(REFERENCE)
    queue.extend(RUNS)

    for run in queue:
        solver_name = run[0]
        Kn = run[1]
        Nx = run[2]

        dx = (xR - xL) / Nx

        # Dynamic Tuple Parsing
        if solver_name in STOCH_DT_SOLVERS:
            _, _, _, N_INJ_TOTAL, dt_val, vmax, t_final = run
            CFL_val = dt_val * vmax / dx
            Nv = [40]
        elif solver_name in STOCH_CFL_SOLVERS:
            _, _, _, N_INJ_TOTAL, CFL_val, vmax, t_final = run
            dt_val = CFL_val * dx / vmax
            Nv = [40]
        else:  # Deterministic
            _, _, _, Nv_val, CFL_val, vmax, t_final = run
            dt_val = CFL_val * dx / vmax
            N_INJ_TOTAL = None
            Nv = [Nv_val]

        filename = make_filename(run, is_ref=(run == REFERENCE))
        out_path = os.path.join(SIMULATIONS_DIR, filename)

        if os.path.exists(out_path) and SKIP_EXISTING_FILES:
            print(f"\nAlready exists, skipping: {filename}")
            continue

        print(f"\n--- Solving {filename} ---")

        is_stoch = solver_name in STOCH_DT_SOLVERS or solver_name in STOCH_CFL_SOLVERS

        M_REF = None
        N_inj = None
        if is_stoch:
            v_th = np.sqrt(2.0 * R_gas * T_in)
            U = u_in / v_th
            exact_flux = (
                rho_in
                * np.sqrt(R_gas * T_in / (2.0 * np.pi))
                * (np.exp(-(U**2)) + np.sqrt(np.pi) * U * (1.0 + math.erf(U)))
            )
            N_inj = float(N_INJ_TOTAL * (dt_val / t_final))
            M_REF = (exact_flux * dt_val) / N_inj

        problem = PenetrationProblem(xL, xR)
        physics_conf = PhysicsConfig(
            Kn=Kn, problem_name=problem.name, reflectance_left=REFLECTANCE_LEFT
        )
        grid_conf = GridConfig(
            xL=xL,
            xR=xR,
            Nx=None,
            Nc=Nx,
            Nv=Nv,
            dim_v=1,
            vmax=vmax,
            vmin=-vmax,
            bc_type=problem.bc_type,
        )
        time_conf = TimeConfig(t_final=t_final, dt=dt_val, CFL=CFL_val)
        config = Config(grid=grid_conf, time=time_conf, physics=physics_conf)

        x_centers = 0.5 * (
            np.linspace(xL, xR, Nx + 1)[:-1] + np.linspace(xL, xR, Nx + 1)[1:]
        )
        u_bg_array = problem.u_bg_func(x_centers)
        T_bg_array = problem.T_bg_func(x_centers)

        if solver_name == "ugks":
            from bgk.solvers.ugks import LinearUGKSSolver

            solver = LinearUGKSSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )
        elif solver_name == "fvm":
            from bgk.solvers.fvm import FVMSolver

            solver = FVMSolver(config)
        elif solver_name == "sl":
            from bgk.solvers.sl import SLSolver

            solver = SLSolver(config)
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
        elif solver_name == "rtsm":
            from bgk.solvers.rtsm import LinearRTSMSolver

            solver = LinearRTSMSolver(
                config,
                u_bg=u_bg_array,
                T_bg=T_bg_array,
                rho_bg=1.0,
                target_N_total=None,
            )
        elif solver_name == "ugkp":
            from bgk.solvers.ugkp import LinearUGKPSolver

            solver = LinearUGKPSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )
        elif solver_name == "vj":
            from bgk.solvers.vj import LinearVelocityJumpSolver

            solver = LinearVelocityJumpSolver(
                config, u_bg=u_bg_array, T_bg=T_bg_array, rho_bg=1.0
            )

        if is_stoch:
            RunnerClass = UGKPRunner if solver_name == "ugkp" else ParticleRunner
            runner = RunnerClass(
                config=config,
                solver=solver,
                problem=problem,
                Np=None,
                m_ref=M_REF,
                N_inj=N_inj,
            )
            runner.particles.flow_left = {
                "rho": rho_in,
                "u": u_in,
                "T": T_in,
                "vmax": vmax,
            }
        else:
            runner = Runner(config=config, solver=solver, problem=problem)
            v = runner.grid.v
            pref = rho_in / np.sqrt(2.0 * np.pi * R_gas * T_in)
            f_inflow = pref * np.exp(-((v - u_in) ** 2) / (2.0 * R_gas * T_in))
            if hasattr(runner, "df"):
                runner.df.f_flow_left = f_inflow
            if hasattr(runner, "solver"):
                runner.solver.f_flow_left = f_inflow

        t0 = time.time()
        sim = runner.run()
        cpu_time = time.time() - t0
        print(f"[{solver_name.upper()}] completed in {cpu_time:.2f}s")

        # Save both dt and CFL to the metadata so plot script has access to them
        meta = {
            "solver": solver_name,
            "Kn": float(Kn),
            "Nx": int(Nx),
            "CFL": float(CFL_val),
            "dt": float(dt_val),
            "vmax": float(vmax),
            "t_final": float(t_final),
            "type": "stochastic" if is_stoch else "deterministic",
        }
        if is_stoch:
            meta["N_INJ_TOTAL"] = float(N_INJ_TOTAL)
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


if __name__ == "__main__":
    main()
