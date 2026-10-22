# used by: cells\schroedinger.py
"""Numerical solver for the 1D time-independent Schroedinger equation.

Units: hbar = m = 1, so H = -1/2 d^2/dx^2 + V(x).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal

from utils.calculator import compile_expression

_FloatArray = NDArray[np.float64]


def parse_potential(expr: str) -> Callable[[_FloatArray], _FloatArray]:
    """Compile a V(x) string into a function of a grid array.

    Uses utils.calculator; delta(x) is resolved to the grid spacing on evaluation.
    """
    formula = compile_expression(expr, variables=("x",))

    def potential(x: _FloatArray) -> _FloatArray:
        x = np.asarray(x, dtype=float)
        dx = float(x[1] - x[0]) if x.size > 1 else 1.0
        return formula(x, delta_width=dx)

    return potential


def solve_schroedinger(
    potential: Callable[[_FloatArray], _FloatArray],
    x_min: float,
    x_max: float,
    n_states: int,
    n_points: int = 800,
) -> tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray]:
    """Lowest eigenpairs of H with Dirichlet walls at x_min and x_max.

    Returns ``(x, V, energies, psi)`` where ``psi[n]`` is the n-th real
    eigenfunction on the interior grid ``x`` with integral of psi^2 equal to 1.
    """
    if not x_min < x_max:
        raise ValueError("x_min muss kleiner als x_max sein")
    if n_points < 10:
        raise ValueError("Zu wenige Gitterpunkte")
    n_states = int(min(max(n_states, 1), n_points))
    if n_points % 2 == 0 and math.isclose(x_min, -x_max):
        n_points += 1  # puts a grid node at x = 0 (exact for delta(x))

    x = np.linspace(x_min, x_max, n_points + 2)[1:-1]
    dx = float(x[1] - x[0])
    v = potential(x)

    diag = 1.0 / dx**2 + v
    off = np.full(n_points - 1, -0.5 / dx**2)
    energies, vecs = eigh_tridiagonal(
        diag, off, select="i", select_range=(0, n_states - 1)
    )

    psi = vecs.T / math.sqrt(dx)
    for i, state in enumerate(psi):
        significant = np.flatnonzero(np.abs(state) > 0.1 * np.abs(state).max())
        if state[significant[0]] < 0:
            psi[i] = -state
    return x, v, energies, psi


def oscillation_period(energies: _FloatArray) -> float:
    """Period 2*pi/dE of the slowest beat between neighbouring levels.

    Falls back to the ground-state period 2*pi/E_0 for a single level.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.size > 1:
        gap = float(np.min(np.diff(np.sort(energies))))
        if gap > 1e-12:
            return 2.0 * math.pi / gap
    return 2.0 * math.pi / max(abs(float(energies[0])), 1e-12)


def time_dependent(psi: _FloatArray, energy: float, t: float) -> NDArray[np.complex128]:
    """psi(x, t) = psi(x) * exp(-i E t)."""
    return psi * np.exp(-1j * energy * t)
