from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_ROOT = REPO_ROOT / "scripts" / "figures"

DEFAULT_DPI = 180


def slugify(text: str) -> str:
    """Makes a string safe to embed in a filename, collapsing runs of separators."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return re.sub(r"-{2,}", "-", cleaned)


def figure_path(
    script_name: str,
    plot_name: str,
    *,
    env_id: str | None = None,
    run_slug: str | None = None,
    out_dir: str | Path | None = None,
    ext: str = "png",
) -> Path:
    """Builds the output path for one plot.

    `script_name` is the calling script's stem, `plot_name` identifies the individual figure, and
    `run_slug` identifies the checkpoint/config it came from.
    """
    root = Path(out_dir).expanduser() if out_dir is not None else DEFAULT_FIGURE_ROOT
    directory = root / script_name
    if env_id:
        directory = directory / env_id

    stem = "_".join(part for part in (run_slug, slugify(plot_name)) if part)
    return directory / f"{stem}.{ext}"


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    show: bool = False,
    dpi: int = DEFAULT_DPI,
    reserve_title_space: bool = True,
    apply_tight_layout: bool = True,
) -> Path:
    """Writes a figure to `path`, creating parent directories, and closes it."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if apply_tight_layout:
        if reserve_title_space:
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        else:
            fig.tight_layout()

    fig.savefig(
        path,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )

    if show:
        plt.show()
    plt.close(fig)

    return path
