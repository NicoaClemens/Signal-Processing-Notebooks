# used by:
"""Interactive explorer for the 1D time-independent Schroedinger equation."""

from __future__ import annotations

import io
from collections.abc import Callable

import ipywidgets as widgets
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from utils.STYLES import COLORS, SLIDER_LAYOUT
from utils.schroedinger import (
    oscillation_period,
    parse_potential,
    solve_schroedinger,
    time_dependent,
)
from utils.ui import dark_ax, section

PRESETS = {
    "Harmonischer Oszillator": ("x^2/2", -8.0, 8.0),
    "Unendlicher Potentialtopf": ("0", 0.0, 1.0),
    "Endlicher Potentialtopf": ("-10*rect(x/4)", -6.0, 6.0),
    "Delta-Potential": ("-5*delta(x)", -10.0, 10.0),
    "Doppelmulde": ("(x^2-4)^2/8", -5.0, 5.0),
    "Doppel-Delta": ("-4*delta(x-1) - 4*delta(x+1)", -8.0, 8.0),
    "Stufe + Wand": ("2*rect((x-3)/2) + x^2/20", -6.0, 8.0),
}


def _build_plot(sol: dict, k: int) -> Callable[[float], bytes]:
    """Build the figure once; the returned function updates time and renders a PNG."""
    x, v, energies, psi = sol["x"], sol["v"], sol["energies"], sol["psi"]
    n = len(energies)
    span = max(float(energies[-1] - energies[0]), 1.0)
    gap = span / max(n - 1, 1) if n > 1 else span
    lo = max(min(float(v.min()), float(energies[0])), float(energies[0]) - 0.5 * span)
    lo -= 0.1 * span
    hi = float(energies[-1]) + 0.5 * gap + 0.05 * span

    fig = Figure(figsize=(8, 6.4))
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(COLORS.bg)
    ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 2]})
    for ax in (ax1, ax2):
        dark_ax(ax)

    ax1.plot(x, v, color=COLORS.orange, lw=2.2, label="V(x)", zorder=5)
    amps = [0.42 * gap / max(float(np.abs(p).max()), 1e-12) for p in psi]
    level_lines = []
    for i in range(n):
        selected = i == k - 1
        colour = COLORS.green if selected else COLORS.blue
        ax1.axhline(energies[i], color=colour, lw=0.8, ls="--", alpha=0.6)
        label = None
        if selected:
            label = f"Re ψₖ + Eₖ (k = {k})"
        elif i == (0 if k != 1 else 1):
            label = "Re ψₙ + Eₙ (übrige Zustände)"
        (line,) = ax1.plot(
            x,
            energies[i] + amps[i] * psi[i],
            color=colour,
            lw=2.2 if selected else 1.1,
            alpha=1.0 if selected else 0.8,
            zorder=4 if selected else 3,
            label=label,
        )
        level_lines.append(line)
    ax1.set_ylim(lo, hi)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_xlabel("x", color=COLORS.text_title)
    ax1.set_ylabel("Energie E  /  Re ψₙ(x,t)", color=COLORS.text_title)
    ax1.legend(
        loc="upper right",
        facecolor=COLORS.bg,
        edgecolor="none",
        labelcolor=COLORS.text_title,
    )

    peak = max(float(np.abs(psi[k - 1]).max()), 1e-12)
    limit = 1.15 * peak
    (re_line,) = ax2.plot(x, psi[k - 1], color=COLORS.green, lw=1.8, label="Re ψ")
    (im_line,) = ax2.plot(x, 0 * x, color=COLORS.pink, lw=1.8, label="Im ψ")
    ax2.plot(
        x,
        psi[k - 1] ** 2 / peak,
        color="white",
        lw=1.2,
        ls=":",
        label="|ψ|² (skaliert)",
    )
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(-limit, limit)
    ax2.set_xlabel("x", color=COLORS.text_title)
    ax2.set_title(
        f"k = {k},  Eₖ = {energies[k - 1]:.4f}", color=COLORS.text_title, fontsize=11
    )
    ax2.legend(
        loc="upper right",
        facecolor=COLORS.bg,
        edgecolor="none",
        labelcolor=COLORS.text_title,
        ncol=3,
    )
    fig.tight_layout()

    def render(t: float) -> bytes:
        for i, line in enumerate(level_lines):
            wave = time_dependent(psi[i], float(energies[i]), t).real
            line.set_ydata(energies[i] + amps[i] * wave)
        state = time_dependent(psi[k - 1], float(energies[k - 1]), t)
        re_line.set_ydata(state.real)
        im_line.set_ydata(state.imag)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=125, facecolor=fig.get_facecolor())
        return buf.getvalue()

    return render


def create_schroedinger_ui() -> widgets.VBox:
    preset = widgets.Dropdown(
        options=["(eigene Eingabe)", *PRESETS],
        description="Beispiel",
        layout=SLIDER_LAYOUT,
    )
    expr = widgets.Text(
        value="step(x-1)",
        description="V(x) =",
        continuous_update=False,
        layout=widgets.Layout(width="520px"),
    )
    x_min = widgets.FloatText(
        value=-8.0, description="x min", layout=widgets.Layout(width="160px")
    )
    x_max = widgets.FloatText(
        value=8.0, description="x max", layout=widgets.Layout(width="160px")
    )
    points = widgets.IntSlider(
        value=800,
        min=100,
        max=3000,
        step=50,
        description="Gitterpunkte",
        continuous_update=False,
        layout=SLIDER_LAYOUT,
    )
    n_states = widgets.IntSlider(
        value=6,
        min=1,
        max=20,
        description="Zustände N",
        continuous_update=False,
        layout=SLIDER_LAYOUT,
    )
    k_slider = widgets.IntSlider(
        value=1, min=1, max=6, description="Zustand k", layout=SLIDER_LAYOUT
    )
    t_slider = widgets.IntSlider(
        value=0, min=0, max=100, description="t / T [%]", layout=SLIDER_LAYOUT
    )
    run = widgets.ToggleButton(
        value=False,
        description="Zeit läuft",
        icon="play",
        button_style="",
        tooltip="Zeit durchläuft eine Periode T = 2π/ΔE_min",
    )
    # Frontend-side timer: the Play widget loops the time slider without kernel round trips.
    play = widgets.Play(value=0, min=0, max=100, step=1, interval=120, loop=True)
    play.layout.display = "none"
    widgets.jslink((run, "value"), (play, "playing"))
    widgets.jslink((play, "value"), (t_slider, "value"))

    info = widgets.HTML()
    # Scale to the container instead of the PNG's pixel size (keeps it sharp when zooming)
    image = widgets.Image(
        format="png",
        layout=widgets.Layout(width="100%", max_width="800px", height="auto"),
    )
    sol: dict = {}

    def message(text: str, error: bool = False) -> None:
        colour = COLORS.red if error else COLORS.text_title
        info.value = f'<span style="color:{colour};font-family:monospace">{text}</span>'

    def redraw(*_) -> None:
        if not sol:
            return
        t = t_slider.value / 100 * sol["period"]
        if sol.get("render_k") != k_slider.value:
            sol["render"] = _build_plot(sol, k_slider.value)
            sol["render_k"] = k_slider.value
        image.value = sol["render"](t)
        message(
            f"T = 2π/ΔE_min = {sol['period']:.4f}   t = {t:.4f}   "
            f"(ħ = m = 1)   E₁…E_N = "
            + ", ".join(f"{e:.3f}" for e in sol["energies"][:8])
            + (" …" if len(sol["energies"]) > 8 else "")
        )

    def solve(*_) -> None:
        try:
            potential = parse_potential(expr.value)
            x, v, energies, psi = solve_schroedinger(
                potential, x_min.value, x_max.value, n_states.value, points.value
            )
        except Exception as exc:  # shown to the user instead of a traceback
            message(f"Fehler: {exc}", error=True)
            return
        sol.clear()
        sol.update(
            x=x, v=v, energies=energies, psi=psi, period=oscillation_period(energies)
        )
        k_slider.max = len(energies)
        k_slider.value = min(k_slider.value, len(energies))
        redraw()

    def apply_preset(change) -> None:
        if change["new"] in PRESETS:
            text, lo, hi = PRESETS[change["new"]]
            x_min.value, x_max.value = lo, hi
            expr.value = text  # triggers solve

    preset.observe(apply_preset, "value")
    for w in (expr, x_min, x_max, points, n_states):
        w.observe(solve, "value")
    k_slider.observe(redraw, "value")
    t_slider.observe(redraw, "value")

    solve()
    return widgets.VBox(
        [
            section("Potential V(x)"),
            preset,
            expr,
            widgets.HTML(
                "<small>Erlaubt: + − * / ^ exp sin cos sqrt abs rect(x) delta(x) "
                "step(x) pi</small>"
            ),
            widgets.HBox([x_min, x_max, points]),
            section("Eigenzustände"),
            n_states,
            k_slider,
            section("Zeitentwicklung"),
            widgets.HBox([t_slider, run, play]),
            info,
            image,
        ]
    )
