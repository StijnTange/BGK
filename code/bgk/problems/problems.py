from dataclasses import dataclass
from typing import Callable, Tuple

import bgk.problems.couette
import bgk.problems.fourier
import bgk.problems.gaussian
import bgk.problems.riemann


@dataclass
class Problem:
    name: str
    f0_func: Callable
    bc_type: str
    x_bounds: Tuple[float, float]


def get_problem(name: str, config=None) -> Problem:
    if name == "gaussian":
        return Problem(
            name="gaussian",
            f0_func=bgk.problems.gaussian.f0_func,
            bc_type="periodic",
            x_bounds=(-1.0, 1.0),
        )

    elif name == "riemann":
        return Problem(
            name="riemann",
            f0_func=bgk.problems.riemann.f0_func,
            bc_type="specular",
            x_bounds=(0.0, 1.0),
        )

    elif name == "couette":
        return Problem(
            name="couette",
            f0_func=bgk.problems.couette.f0_func,
            bc_type="diffusive",
            x_bounds=(0.0, 1.0),
        )

    elif name == "fourier":
        return Problem(
            name="fourier",
            f0_func=bgk.problems.fourier.get_fourier_f0_func(config),
            bc_type="diffusive",
            x_bounds=(0.0, 1.0),
        )
    else:
        raise ValueError(
            f"Unknown problem: {name}. Available: 'gaussian', 'riemann', 'couette'"
        )
