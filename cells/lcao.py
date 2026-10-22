# used by:
"""Interactive finite-lattice LCAO and tight-binding demonstrations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib.patches import Rectangle

from utils.STYLES import COLORS, SLIDER_LAYOUT
from utils.tight_binding import (
    analytic_energy,
    analytic_eigenstate,
    analytic_eigenstates,
    build_hamiltonian,
)
from utils.ui import dark_ax, section


def _format_energy(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}".replace("-", "−")


def _create_tight_binding_ui(two_dimensional: bool) -> widgets.VBox:
    if two_dimensional:
        nx = widgets.IntSlider(
            value=3,
            min=2,
            max=5,
            description="Nₓ",
            continuous_update=False,
            layout=SLIDER_LAYOUT,
        )
        ny = widgets.IntSlider(
            value=3,
            min=2,
            max=5,
            description="Nᵧ",
            continuous_update=False,
            layout=SLIDER_LAYOUT,
        )
    else:
        nx = widgets.IntSlider(
            value=4,
            min=2,
            max=10,
            description="Anzahl Atome N",
            continuous_update=False,
            layout=SLIDER_LAYOUT,
        )
        ny = None

    hopping = widgets.FloatSlider(
        value=1.0,
        min=0.2,
        max=2.0,
        step=0.05,
        description="Vₛₛ (eV)",
        readout_format=".2f",
        continuous_update=False,
        layout=SLIDER_LAYOUT,
    )
    onsite = widgets.FloatSlider(
        value=0.0,
        min=-6.0,
        max=6.0,
        step=0.5,
        description="ε₀ (eV)",
        readout_format=".1f",
        continuous_update=False,
        layout=SLIDER_LAYOUT,
    )
    mode_x = widgets.IntSlider(
        value=1,
        min=1,
        max=nx.value,
        description="lₓ" if two_dimensional else "Eigenzustand l",
        continuous_update=False,
        layout=SLIDER_LAYOUT,
    )
    mode_y = (
        widgets.IntSlider(
            value=1,
            min=1,
            max=ny.value,
            description="lᵧ",
            continuous_update=False,
            layout=SLIDER_LAYOUT,
        )
        if two_dimensional and ny is not None
        else None
    )
    matrix_symbols = widgets.Checkbox(
        value=True, description="Symbole statt Zahlen anzeigen"
    )
    row_index = widgets.Dropdown(
        options=[(f"m = {index}", index) for index in range(1, 10)],
        value=1,
        description="Zeile",
    )
    column_index = widgets.Dropdown(
        options=[(f"n = {index}", index) for index in range(1, 10)],
        value=2,
        description="Spalte",
    )
    matrix_info = widgets.HTML()
    state_table = widgets.HTML()
    plot_output = widgets.Output()

    def _shape() -> tuple[int, ...]:
        if two_dimensional and ny is not None:
            return (ny.value, nx.value)
        return (nx.value,)

    def _selected_modes() -> tuple[int, ...]:
        if two_dimensional and mode_y is not None:
            return (mode_y.value, mode_x.value)
        return (mode_x.value,)

    def _update_index_options() -> None:
        site_count = int(np.prod(_shape()))
        options = [(f"{index}", index) for index in range(1, site_count + 1)]
        row_index.options = options
        column_index.options = options
        row_index.value = min(row_index.value, site_count)
        column_index.value = min(column_index.value, site_count)

    def _render(*_changed: object) -> None:
        shape = _shape()
        modes = _selected_modes()
        hamiltonian = build_hamiltonian(shape, hopping.value, onsite.value)
        state = analytic_eigenstate(shape, modes)
        energy = analytic_energy(shape, modes, hopping.value, onsite.value)
        residual = hamiltonian @ state - energy * state
        site_count = len(state)
        row = row_index.value - 1
        column = column_index.value - 1
        matrix_value = hamiltonian[row, column]
        hamiltonian_state = hamiltonian @ state

        if row == column:
            explanation = (
                f"Eigenatom {row + 1}: onsite energy ε₀ = "
                f"{_format_energy(onsite.value)} eV."
            )
        else:
            row_coordinates = np.unravel_index(row, shape)
            column_coordinates = np.unravel_index(column, shape)
            distance = sum(
                abs(first - second)
                for first, second in zip(row_coordinates, column_coordinates)
            )
            if distance == 1:
                explanation = (
                    f"Atome {row + 1} und {column + 1} sind nächste Nachbarn: "
                    f"⟨{row + 1}|ℋ|{column + 1}⟩ = −Vₛₛ = "
                    f"{_format_energy(matrix_value)} eV."
                )
            else:
                explanation = (
                    f"Atome {row + 1} und {column + 1} sind keine direkten "
                    "Nachbarn; in der Nächste-Nachbarn-Näherung ist das "
                    "Matrixelement null."
                )
        matrix_info.value = (
            f"<b>H<sub>{row + 1},{column + 1}</sub> = "
            f"{_format_energy(matrix_value)} eV.</b> {explanation}"
        )

        rows = []
        for index, coefficient in enumerate(state):
            coordinates = np.unravel_index(index, shape)
            coordinate_label = (
                str(index + 1)
                if len(shape) == 1
                else f"({coordinates[1] + 1}, {coordinates[0] + 1})"
            )
            rows.append(
                "<tr>"
                f"<td>{index + 1}</td><td>{coordinate_label}</td>"
                f"<td>{_format_energy(coefficient)}</td>"
                f"<td>{_format_energy(hamiltonian_state[index])}</td>"
                f"<td>{_format_energy(energy * coefficient)}</td>"
                "</tr>"
            )
        state_table.value = (
            "<table><thead><tr><th>i</th><th>Atom</th><th>cᵢ</th>"
            "<th>(H·c)ᵢ</th><th>E·cᵢ</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + f"<p>Probe: max |H·c − E·c| = "
            f"{np.max(np.abs(residual)):.2e}; norm(c) = "
            f"{np.linalg.norm(state):.6f}.</p>"
        )

        with plot_output:
            plot_output.clear_output(wait=True)
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.patch.set_facecolor(COLORS.bg)
            for axis in axes.flat:
                dark_ax(axis)

            basis_ax, matrix_ax, state_ax = axes[0]
            band_ax, levels_ax = axes[1, 0], axes[1, 2]
            axes[1, 1].set_visible(False)

            coordinates = np.array(
                [np.unravel_index(index, shape) for index in range(site_count)]
            )
            if len(shape) == 1:
                x_positions = np.arange(1, site_count + 1)
                y_positions = np.zeros(site_count)
                basis_ax.scatter(
                    x_positions,
                    y_positions,
                    s=550,
                    color=COLORS.primary,
                    alpha=0.55,
                )
                basis_ax.set_yticks([])
                basis_ax.set_xlabel("Atom n")
                basis_ax.set_title("Basis: ein Index = ein Atom")
                for index, (x_pos, y_pos) in enumerate(
                    zip(x_positions, y_positions), start=1
                ):
                    basis_ax.text(
                        x_pos,
                        y_pos,
                        str(index),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )
                basis_ax.set_xlim(0.3, site_count + 0.7)
                basis_ax.set_ylim(-0.8, 0.8)
            else:
                x_positions = coordinates[:, 1] + 1
                y_positions = coordinates[:, 0] + 1
                basis_ax.scatter(
                    x_positions,
                    y_positions,
                    s=500,
                    color=COLORS.primary,
                    alpha=0.55,
                )
                basis_ax.set_xticks(range(1, shape[1] + 1))
                basis_ax.set_yticks(range(1, shape[0] + 1))
                basis_ax.set_xlabel("nₓ")
                basis_ax.set_ylabel("nᵧ")
                basis_ax.set_title("Quadratgitter ohne periodische Ränder")
                for index, (x_pos, y_pos) in enumerate(
                    zip(x_positions, y_positions), start=1
                ):
                    basis_ax.text(
                        x_pos,
                        y_pos,
                        str(index),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )
                basis_ax.set_aspect("equal")
            for selected_index in (row, column):
                basis_ax.scatter(
                    [x_positions[selected_index]],
                    [y_positions[selected_index]],
                    s=700,
                    facecolors="none",
                    edgecolors=COLORS.orange,
                    linewidths=2,
                )

            matrix_ax.imshow(hamiltonian, cmap="coolwarm")
            matrix_ax.set_title("Hamiltonmatrix Hₘₙ")
            matrix_ax.set_xlabel("Spalte n")
            matrix_ax.set_ylabel("Zeile m")
            matrix_ax.set_xticks(range(site_count))
            matrix_ax.set_yticks(range(site_count))
            matrix_ax.tick_params(labelsize=7)
            if site_count > 12:
                matrix_ax.set_xticklabels([])
                matrix_ax.set_yticklabels([])
            for matrix_row in range(site_count):
                for matrix_column in range(site_count):
                    value = hamiltonian[matrix_row, matrix_column]
                    if matrix_symbols.value:
                        label = (
                            "ε₀"
                            if matrix_row == matrix_column
                            else "−V" if value != 0 else "0"
                        )
                    else:
                        label = _format_energy(value)
                    matrix_ax.text(
                        matrix_column,
                        matrix_row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6 if site_count > 12 else 8,
                        color=COLORS.bg_deep,
                    )
            matrix_ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=COLORS.orange,
                    linewidth=2,
                )
            )

            amplitudes = np.abs(state)
            marker_sizes = 40 + 500 * amplitudes / amplitudes.max()
            marker_colors = [
                COLORS.blue if coefficient >= 0 else COLORS.red for coefficient in state
            ]
            state_ax.scatter(
                x_positions,
                y_positions,
                s=marker_sizes,
                c=marker_colors,
                alpha=0.75,
            )
            for index, (x_pos, y_pos, coefficient) in enumerate(
                zip(x_positions, y_positions, state), start=1
            ):
                state_ax.text(
                    x_pos,
                    y_pos,
                    _format_energy(coefficient),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
            state_ax.set_title(
                f"Eigenzustand {modes[::-1] if two_dimensional else modes}"
            )
            if len(shape) == 2:
                state_ax.set_xticks(range(1, shape[1] + 1))
                state_ax.set_yticks(range(1, shape[0] + 1))
                state_ax.set_xlabel("nₓ")
                state_ax.set_ylabel("nᵧ")
                state_ax.set_aspect("equal")
            else:
                state_ax.set_yticks([])
                state_ax.set_xlabel("Atom n")
                state_ax.set_xlim(0.3, site_count + 0.7)

            eigenstates = analytic_eigenstates(shape, hopping.value, onsite.value)
            state_energies = [
                (mode, state_energy) for mode, state_energy, _ in eigenstates
            ]
            if len(shape) == 1:
                wave_numbers = np.linspace(0, np.pi, 300)
                dispersion = onsite.value - 2 * hopping.value * np.cos(wave_numbers)
                band_ax.plot(
                    wave_numbers,
                    dispersion,
                    color=COLORS.text_label,
                    linewidth=1.5,
                )
                for mode, state_energy in state_energies:
                    wave_number = mode[0] * np.pi / (shape[0] + 1)
                    band_ax.scatter(
                        [wave_number],
                        [state_energy],
                        color=COLORS.red if mode == modes else COLORS.blue,
                        s=45 if mode == modes else 24,
                    )
                band_ax.set_xlabel("ka")
                band_ax.set_ylabel("E (eV)")
                band_ax.set_title(f"Band E(k) = ε₀ − 2Vₛₛ cos(ka), Breite 4Vₛₛ")
            else:
                kx = np.linspace(0, np.pi, 100)
                ky = np.linspace(0, np.pi, 100)
                kx_grid, ky_grid = np.meshgrid(kx, ky)
                dispersion = onsite.value - 2 * hopping.value * (
                    np.cos(kx_grid) + np.cos(ky_grid)
                )
                band_ax.pcolormesh(
                    kx,
                    ky,
                    dispersion,
                    shading="auto",
                    cmap="coolwarm",
                )
                for mode, state_energy in state_energies:
                    mode_y, mode_x = mode
                    band_ax.scatter(
                        [mode_x * np.pi / (shape[1] + 1)],
                        [mode_y * np.pi / (shape[0] + 1)],
                        color=COLORS.text if mode == modes else COLORS.blue,
                        edgecolors=COLORS.bg,
                        s=65 if mode == modes else 30,
                    )
                band_ax.set_xlabel("kₓa")
                band_ax.set_ylabel("kᵧa")
                band_ax.set_title("E(kₓ,kᵧ) = ε₀ − 2Vₛₛ(cos kₓa + cos kᵧa)")
                band_ax.set_xlim(0, np.pi)
                band_ax.set_ylim(0, np.pi)
                band_ax.set_aspect("equal")

            ordered_energies = sorted(
                state_energies,
                key=lambda item: (item[1], item[0]),
            )
            for level, (mode, state_energy) in enumerate(ordered_energies):
                selected = mode == modes
                levels_ax.hlines(
                    state_energy,
                    level,
                    level + 0.8,
                    color=COLORS.red if selected else COLORS.primary,
                    linewidth=3 if selected else 1.5,
                )
            levels_ax.set_xlabel("Zustände, nach Energie sortiert")
            levels_ax.set_ylabel("E (eV)")
            levels_ax.set_title(
                f"Energieniveaus ({site_count} Zustände), "
                f"Bandbreite {4 * len(shape) * hopping.value:.2f} eV"
            )
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    def _on_dimensions_change(*_changed: object) -> None:
        if mode_x.value > nx.value:
            mode_x.value = nx.value
        mode_x.max = nx.value
        if ny is not None and mode_y is not None:
            if mode_y.value > ny.value:
                mode_y.value = ny.value
            mode_y.max = ny.value
        _update_index_options()
        _render()

    nx.observe(_on_dimensions_change, names="value")
    if ny is not None:
        ny.observe(_on_dimensions_change, names="value")
    for control in (hopping, onsite, mode_x, matrix_symbols, row_index, column_index):
        control.observe(_render, names="value")
    if mode_y is not None:
        mode_y.observe(_render, names="value")

    _update_index_options()
    controls = [section("Gitterparameter"), nx]
    if ny is not None:
        controls.append(ny)
    controls.extend([hopping, onsite])
    if mode_y is not None:
        controls.extend([section("Eigenzustand auswählen"), mode_x, mode_y])
    else:
        controls.extend([section("Eigenzustand auswählen"), mode_x])
    controls.extend(
        [
            section("Matrixelement untersuchen"),
            widgets.HBox([row_index, column_index, matrix_symbols]),
        ]
    )
    _render()
    return widgets.VBox([widgets.VBox(controls), matrix_info, plot_output, state_table])


def create_chain_ui() -> widgets.VBox:
    return _create_tight_binding_ui(two_dimensional=False)


def create_square_lattice_ui() -> widgets.VBox:
    return _create_tight_binding_ui(two_dimensional=True)
