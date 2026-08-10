from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# --- Surfaces and ink -------------------------------------------------------------------------

BACKGROUND = "#0f172a"
SURFACE = "#1e293b"
TEXT_PRIMARY = "#f8fafc"
TEXT_SECONDARY = "#cbd5e1"
TEXT_MUTED = "#94a3b8"
TICK = "#64748b"
GRID = "#334155"

# --- Data colours -----------------------------------------------------------------------------

SERIES: tuple[str, ...] = (
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#9085e9",  # violet
)

SUCCESS = "#10b981"
FAILURE = "#f87171"

SEQUENTIAL_CMAP = "viridis"


def diverging_cmap() -> LinearSegmentedColormap:
    """Blue-to-red ramp for values that are signed around zero, e.g. network weights."""
    return LinearSegmentedColormap.from_list("vandal_diverging", [SERIES[0], GRID, FAILURE], N=256)


DARK_RCPARAMS: dict[str, object] = {
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TICK,
    "ytick.color": TICK,
    "grid.color": GRID,
    "grid.alpha": 0.5,
    "axes.edgecolor": GRID,
    "figure.titlesize": 14,
    "legend.labelcolor": TEXT_SECONDARY,
}


def apply_theme() -> None:
    """Applies the dark theme globally."""
    mpl.rcParams.update(DARK_RCPARAMS)


def style_axes(ax: plt.Axes) -> None:
    """Applies the grid/spine treatment every plot in these scripts uses."""
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# --- Titles -----------------------------------------------------------------------------------


def format_fields(fields: list[tuple[str, str]]) -> str:
    """Renders provenance fields as `key=value · key=value`, for figure subtitles."""
    return "  ·  ".join(f"{key}={value}" for key, value in fields)


def set_title(fig: plt.Figure, title: str, fields: list[tuple[str, str]] | None = None) -> None:
    """Gives a figure a bold title saying what it shows, plus a muted subtitle of the parameters it
    was produced under."""
    fig.suptitle(title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY, y=0.995)
    if fields:
        fig.text(
            0.5,
            0.957,
            format_fields(fields),
            ha="center",
            va="top",
            fontsize=8.5,
            color=TEXT_MUTED,
        )


def outcome_legend_handles() -> list[Line2D]:
    """Legend entries for success/failure episode colouring."""
    return [
        Line2D([], [], color=SUCCESS, marker="o", linestyle="none", label="success"),
        Line2D([], [], color=FAILURE, marker="X", linestyle="none", label="failure"),
    ]
