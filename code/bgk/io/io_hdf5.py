import os
import re

import numpy as np

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("WARNING: h5py not available. Install with: pip install h5py")


def _nv_str(Nvx, Nvy=None):
    """'40' for 1D, '28x28' for 2D."""
    return str(Nvx) if Nvy is None else f"{Nvx}x{Nvy}"


def _filename(problem, solver, Nc, Nvx, Nvy, Kn, t_final, results_dir, CFL=None):
    nv = _nv_str(Nvx, Nvy)
    fname = f"{problem}_{solver}_Nc{Nc}_Nv{nv}_Kn{Kn:.4f}_T{t_final:.1f}"

    # Safely append CFL to the filename if it is provided
    if CFL is not None:
        fname += f"_CFL{CFL}"

    fname += ".h5"
    return os.path.join(results_dir, fname)


def save_result(result: dict, results_dir: str = "results") -> str:
    """
    Save a simulation result dict to HDF5.

    Required keys in result:
      problem, solver, Nc, Kn, t_final_requested, x, T
      Nvx            (int)
      Nvy            (int or None for dim_v=1)
      Couette: uy, tau_xy
      Fourier: qx

    t_final_requested is the hardcoded T_FINAL set by the user.
    t_final (actual) is stored as an attribute.
    """
    if not HDF5_AVAILABLE:
        raise RuntimeError("h5py required for HDF5 output")

    os.makedirs(results_dir, exist_ok=True)

    Nvx = result["Nvx"]
    Nvy = result.get("Nvy", None)
    CFL = result.get("CFL", None)

    path = _filename(
        result["problem"],
        result["solver"],
        result["Nc"],
        Nvx,
        Nvy,
        result["Kn"],
        result["t_final_requested"],
        results_dir,
        CFL,
    )

    with h5py.File(path, "w") as f:
        # Add "ux" to the allowed list of arrays!
        for key in ["x", "T", "ux", "uy", "tau_xy", "qx", "rho"]:
            if key in result and result[key] is not None:
                data = np.asarray(result[key], dtype=np.float64).flatten()
                f.create_dataset(key, data=data, compression="gzip", compression_opts=6)

        attr_keys = [
            "problem",
            "solver",
            "Nc",
            "Kn",
            "tau_mode",
            "omega",
            "R_nd",
            "Nvx",
            "Nvy",
            "dim_v",
            "t_final_requested",
            "t_final_actual",
            "CFL",
            "T_w_nd",
            "u_w_nd",
            "u_w",
            "T_w",
            "T_L_nd",
            "T_R_nd",
            "T_L",
            "T_R",
            "T_ref",
            "U0",
            "mu_0",
            "Pr_Ar",
            "L",
            "rho_0",
            "execution_time",
            "dx",
        ]
        for key in attr_keys:
            val = result.get(key)
            if val is not None:
                f.attrs[key] = val

    print(f"  Saved {path}  ({os.path.getsize(path) // 1024} KB)")
    return path


def load_result(
    solver: str,
    Nc: int,
    Nvx: int,
    Nvy,
    Kn: float,
    t_final: float,
    problem: str,
    results_dir: str = "results",
    CFL: float = None,
) -> dict:
    """
    Load a simulation result from HDF5.
    Nvy: int for dim_v=2, None for dim_v=1.
    Returns dict with all datasets and attributes, or None if not found.
    """
    if not HDF5_AVAILABLE:
        raise RuntimeError("h5py required for HDF5 input")

    path = _filename(problem, solver, Nc, Nvx, Nvy, Kn, t_final, results_dir, CFL)
    if not os.path.exists(path):
        print(f"  WARNING: not found — {path}")
        return None

    result = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            result[key] = f[key][:]
        for key in f.attrs:
            val = f.attrs[key]
            # h5py returns fixed-length strings as bytes — decode them
            if isinstance(val, bytes):
                val = val.decode()
            result[key] = val
    return result


def result_exists(
    solver: str,
    Nc: int,
    Nvx: int,
    Nvy,
    Kn: float,
    t_final: float,
    problem: str,
    results_dir: str = "results",
    CFL: float = None,
) -> bool:
    return os.path.exists(
        _filename(problem, solver, Nc, Nvx, Nvy, Kn, t_final, results_dir, CFL)
    )


def list_results(problem: str, results_dir: str = "results") -> list:
    if not os.path.exists(results_dir):
        return []

    # Updated regex to optionally capture the _CFL block
    pattern = re.compile(
        rf"^{problem}_(\w+)_Nc(\d+)_Nv([\dx]+)_Kn([\d.]+)_T([\d.]+)(?:_CFL([\d.]+))?\.h5$"
    )

    found = []
    for fname in sorted(os.listdir(results_dir)):
        m = pattern.match(fname)
        if m:
            # If CFL was in the filename, m.group(6) will have the value, otherwise None
            cfl_val = float(m.group(6)) if m.group(6) else None
            found.append(
                (
                    m.group(1),
                    int(m.group(2)),
                    m.group(3),
                    float(m.group(4)),
                    float(m.group(5)),
                    cfl_val,
                )
            )
    return found
