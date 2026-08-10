"""Shared helpers for the standalone analysis scripts.

Grouped by concern: `theme`/`figures` cover how a plot looks and where it lands, `report` covers
what gets written down, `checkpoints`/`episodes`/`rollouts` cover where the data comes from, and
`cli` keeps the scripts' command-line vocabulary consistent.
"""

from scripts.utils import (
    checkpoints,
    cli,
    episodes,
    figures,
    report,
    rollouts,
    taps,
    theme,
)
from scripts.utils.checkpoints import (
    build_external_cond,
    checkpoint_slug,
    describe_model_config,
    load_goal_conditioned_diffusion_policy,
    require_run_config,
    resolve_run_config,
    run_slug,
)
from scripts.utils.episodes import ensure_local_dataset
from scripts.utils.figures import REPO_ROOT, figure_path, save_figure
from scripts.utils.report import Report
from scripts.utils.rollouts import (
    DEFAULT_LOCKED_ROTATION_ENV_IDS,
    EpisodeRollout,
    RolloutStep,
    build_obs_transform,
    build_rollout_env,
    iter_env_kwargs,
    resolve_max_episode_steps,
    run_episode,
)
from scripts.utils.taps import (
    capture_pre_norm,
    embedder_output_norm,
    magnitude_band,
    relative_spread,
)
from scripts.utils.theme import apply_theme, set_title, style_axes

__all__ = [
    "DEFAULT_LOCKED_ROTATION_ENV_IDS",
    "EpisodeRollout",
    "REPO_ROOT",
    "Report",
    "RolloutStep",
    "apply_theme",
    "build_external_cond",
    "build_obs_transform",
    "build_rollout_env",
    "capture_pre_norm",
    "checkpoint_slug",
    "checkpoints",
    "cli",
    "describe_model_config",
    "embedder_output_norm",
    "ensure_local_dataset",
    "episodes",
    "figure_path",
    "figures",
    "iter_env_kwargs",
    "load_goal_conditioned_diffusion_policy",
    "magnitude_band",
    "relative_spread",
    "report",
    "require_run_config",
    "resolve_max_episode_steps",
    "resolve_run_config",
    "rollouts",
    "run_episode",
    "run_slug",
    "save_figure",
    "set_title",
    "style_axes",
    "taps",
    "theme",
]
