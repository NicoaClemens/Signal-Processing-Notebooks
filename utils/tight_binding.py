# used by: cells\lcao.py
"""Analytic tight-binding helpers for finite, open lattices."""

from __future__ import annotations

from itertools import product
from math import cos, isfinite, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

LatticeShape = tuple[int, ...]


def _validated_shape(shape: LatticeShape) -> LatticeShape:
    if not shape or any(
        not isinstance(size, (int, np.integer))
        or isinstance(size, (bool, np.bool_))
        or size < 1
        for size in shape
    ):
        raise ValueError("shape must contain one or more positive integers")
    return tuple(int(size) for size in shape)


def _validated_parameters(hopping: float, onsite: float) -> tuple[float, float]:
    hopping = float(hopping)
    onsite = float(onsite)
    if not isfinite(hopping) or not isfinite(onsite):
        raise ValueError("hopping and onsite energy must be finite")
    return hopping, onsite


def build_hamiltonian(
    shape: LatticeShape, hopping: float, onsite: float
) -> NDArray[np.float64]:
    """Build the nearest-neighbor Hamiltonian for a finite open lattice.

    ``shape`` follows NumPy axis order; for a 2D ``(Ny, Nx)`` lattice,
    the first coordinate is the row and the second is the column.
    """
    shape = _validated_shape(shape)
    hopping, onsite = _validated_parameters(hopping, onsite)
    site_count = int(np.prod(shape))
    hamiltonian = np.eye(site_count, dtype=float) * onsite

    for site in np.ndindex(shape):
        site_index = np.ravel_multi_index(site, shape)
        for axis, size in enumerate(shape):
            if site[axis] + 1 >= size:
                continue
            neighbor = list(site)
            neighbor[axis] += 1
            neighbor_index = np.ravel_multi_index(tuple(neighbor), shape)
            hamiltonian[site_index, neighbor_index] = -hopping
            hamiltonian[neighbor_index, site_index] = -hopping

    return hamiltonian


def analytic_energy(
    shape: LatticeShape, modes: tuple[int, ...], hopping: float, onsite: float
) -> float:
    """Return an open-lattice eigenenergy for 1-based mode numbers."""
    shape = _validated_shape(shape)
    hopping, onsite = _validated_parameters(hopping, onsite)
    if len(modes) != len(shape) or any(
        not isinstance(mode, (int, np.integer))
        or isinstance(mode, (bool, np.bool_))
        or not 1 <= mode <= size
        for mode, size in zip(modes, shape)
    ):
        raise ValueError("modes must be 1-based and match the lattice shape")

    return onsite - 2 * hopping * sum(
        cos(mode * pi / (size + 1)) for mode, size in zip(modes, shape)
    )


def analytic_eigenstate(
    shape: LatticeShape, modes: tuple[int, ...]
) -> NDArray[np.float64]:
    """Return a normalized product-of-sines eigenvector in row-major order."""
    shape = _validated_shape(shape)
    if len(modes) != len(shape) or any(
        not isinstance(mode, (int, np.integer))
        or isinstance(mode, (bool, np.bool_))
        or not 1 <= mode <= size
        for mode, size in zip(modes, shape)
    ):
        raise ValueError("modes must be 1-based and match the lattice shape")

    state = np.empty(int(np.prod(shape)), dtype=float)
    for flat_index, site in enumerate(np.ndindex(shape)):
        coefficient = 1.0
        for coordinate, mode, size in zip(site, modes, shape):
            coefficient *= sqrt(2 / (size + 1)) * sin(
                (coordinate + 1) * mode * pi / (size + 1)
            )
        state[flat_index] = coefficient
    return state


def analytic_eigenstates(
    shape: LatticeShape, hopping: float, onsite: float
) -> list[tuple[tuple[int, ...], float, NDArray[np.float64]]]:
    """Return every ``(mode, energy, state)`` triple for an open lattice."""
    shape = _validated_shape(shape)
    return [
        (
            modes,
            analytic_energy(shape, modes, hopping, onsite),
            analytic_eigenstate(shape, modes),
        )
        for modes in product(*(range(1, size + 1) for size in shape))
    ]
