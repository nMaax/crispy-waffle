"""Lists every run in the HF Hub checkpoint repo and flags the ones that look abandoned."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from policy.algorithms.callbacks.hf_sync_model_checkpoint import RUN_STATUS_FILENAMES
from scripts.utils.checkpoints import split_run_path
from scripts.utils.report import Report, styled

# Folder/file names that sit directly inside a run directory, and so mark where one ends.
RUN_CHILDREN = ("checkpoints", ".hydra", *RUN_STATUS_FILENAMES.values())

# Marker filename -> status. A completed run wins if both are somehow present.
STATUS_BY_FILENAME = {name: status for status, name in RUN_STATUS_FILENAMES.items()}
STATUS_PRECEDENCE = ("completed", "interrupted", "unknown")


@dataclass
class RunEntry:
    prefix: str
    checkpoints: list[str] = field(default_factory=list)
    total_bytes: int = 0
    has_hydra_config: bool = False
    marker: str | None = None
    details: dict[str, object] | None = None
    config: dict[str, object] | None = None

    @property
    def experiment(self) -> str:
        return split_run_path(PurePosixPath(self.prefix).parts)[0] or self.prefix

    @property
    def when(self) -> str:
        return split_run_path(PurePosixPath(self.prefix).parts)[1] or self.prefix

    @property
    def her_ratio(self) -> str:
        return _shown(self.config.get("her_ratio") if self.config else None)

    @property
    def seed(self) -> str:
        return _shown(self.config.get("seed") if self.config else None)

    @property
    def sweep_identity(self) -> tuple[str, object, object]:
        """What makes two runs actual copies of each other, as opposed to different points in a
        hyperparameter sweep (her_ratio, seed) that happen to share an experiment name."""
        config = self.config or {}
        return (self.experiment, config.get("her_ratio"), config.get("seed"))

    @property
    def completeness(self) -> tuple[int, int]:
        return (1 if self.state == "completed" else 0, len(self.checkpoints))

    @property
    def state(self) -> str:
        return STATUS_BY_FILENAME.get(self.marker or "", "unknown")

    @property
    def global_step(self) -> int | None:
        if self.details is None:
            return None
        step = self.details.get("global_step")
        return int(step) if isinstance(step, int | float) else None

    @property
    def received_sigterm(self) -> bool:
        return bool(self.details.get("received_sigterm")) if self.details else False


def run_prefix_of(repo_path: str) -> str | None:
    """The run directory a repo file belongs to, or None if it isn't inside one."""
    parts = PurePosixPath(repo_path).parts
    for index, part in enumerate(parts):
        if part in RUN_CHILDREN:
            return "/".join(parts[:index])
    return None


def collect_runs(repo_id: str) -> dict[str, RunEntry]:
    """Walks the repo tree once, grouping every file under the run directory that owns it."""
    from huggingface_hub import HfApi

    api = HfApi()
    runs: dict[str, RunEntry] = {}
    # list_repo_tree, not list_repo_files: only the former reports sizes.
    for item in api.list_repo_tree(repo_id, repo_type="model", recursive=True):
        path = str(item.path)
        prefix = run_prefix_of(path)
        if prefix is None or not hasattr(item, "size"):
            continue

        run = runs.setdefault(prefix, RunEntry(prefix=prefix))
        # LFS-tracked files report their real size on `.lfs`; `.size` is the pointer's.
        lfs = getattr(item, "lfs", None)
        run.total_bytes += int(getattr(lfs, "size", None) or getattr(item, "size", 0) or 0)

        name = PurePosixPath(path).name
        if name.endswith(".ckpt"):
            run.checkpoints.append(name)
        elif path == f"{prefix}/.hydra/config.yaml":
            run.has_hydra_config = True
        elif name in STATUS_BY_FILENAME:
            # Keep the strongest verdict, in case a run dir somehow ended up with both.
            if run.marker is None or STATUS_PRECEDENCE.index(
                STATUS_BY_FILENAME[name]
            ) < STATUS_PRECEDENCE.index(run.state):
                run.marker = name
    return runs


def load_details(repo_id: str, runs: dict[str, RunEntry]) -> None:
    """Fetches each run's status marker and Hydra config into the HF cache."""
    from huggingface_hub import hf_hub_download

    def fetch(relative: str) -> Path | None:
        try:
            return Path(hf_hub_download(repo_id=repo_id, repo_type="model", filename=relative))
        except Exception as error:
            print(f"  could not read {relative}: {error}")
            return None

    for run in runs.values():
        if run.marker is not None and (local := fetch(f"{run.prefix}/{run.marker}")):
            run.details = json.loads(local.read_text(encoding="utf-8"))
        if run.has_hydra_config and (local := fetch(f"{run.prefix}/.hydra/config.yaml")):
            run.config = _read_config_fields(local)


def _read_config_fields(path: Path) -> dict[str, object]:
    """The handful of config values worth showing per run."""
    from omegaconf import DictConfig, OmegaConf

    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        return {}
    return {
        "seed": config.get("seed", None),
        # Same key `describe_model_config` reads for the figure subtitles.
        "her_ratio": config.get("datamodule", {}).get("her_ratio", None),
    }


def observations(run: RunEntry) -> list[str]:
    """What is notable about a run."""
    notes = []
    if run.marker is None:
        notes.append("no-marker")
    if not run.has_hydra_config:
        notes.append("no-config")
    if run.checkpoints == ["last.ckpt"]:
        # The top-k sync happens at train end, so its absence means the process never got there.
        notes.append("last-only")
    if run.received_sigterm:
        notes.append("sigterm")
    return notes


def superseded_by(run: RunEntry, runs: dict[str, RunEntry]) -> RunEntry | None:
    """A later run of the same experiment and sweep point (her_ratio, seed) that is at least as
    complete, if there is one."""
    for other in runs.values():
        if (
            other.sweep_identity == run.sweep_identity
            and other.when > run.when
            and other.completeness >= run.completeness
        ):
            return other
    return None


def prune_reasons(run: RunEntry, runs: dict[str, RunEntry]) -> list[str]:
    """Why this run could be deleted."""
    newer = superseded_by(run, runs)
    if newer is None:
        return []
    return [f"superseded by {newer.when}", *observations(run)]


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G"):
        if value < 1024 or unit == "G":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}G"


def _shown(value: object) -> str:
    """A value for the table, or `-` when there is nothing to show."""
    return "-" if value is None else str(value)


STATE_STYLES = {"completed": "green", "interrupted": "yellow", "unknown": "dim"}


def build_report(repo_id: str, runs: dict[str, RunEntry], sort: str) -> Report:
    keys = {
        "date": lambda run: run.when,
        "size": lambda run: -run.total_bytes,
        "name": lambda run: (run.experiment, run.when),
    }
    ordered = sorted(runs.values(), key=keys[sort])
    reasons = {run.prefix: prune_reasons(run, runs) for run in ordered}

    report = Report(
        "HF Hub checkpoint inventory",
        [("repo", repo_id), ("runs", str(len(ordered)))],
    )

    if sort == "name":
        for experiment in dict.fromkeys(run.experiment for run in ordered):
            group = [run for run in ordered if run.experiment == experiment]
            report.section(experiment)
            report.table(*_rows(group, reasons, with_experiment=False))
    else:
        report.section("Runs")
        report.table(*_rows(ordered, reasons, with_experiment=True))

    report.kv("total size", human_bytes(sum(run.total_bytes for run in ordered)))

    candidates = [run for run in ordered if reasons[run.prefix]]
    report.section("Prune candidates")
    if not candidates:
        report.note("None: no run is superseded by a later one.")
    else:
        report.note(
            f"{len(candidates)} of {len(ordered)} runs are superseded by a later run of "
            "the same experiment. Paste the ones you agree with; this script "
            "deletes nothing."
        )
        report.blank()
        for run in candidates:
            report.note(styled(f"# {', '.join(reasons[run.prefix])}", "red"))
            report.raw(
                f'HfApi().delete_folder(path_in_repo="{run.prefix}", '
                f'repo_id="{repo_id}", repo_type="model")'
            )
    report.blank()
    report.note(
        "Deleting files does not shrink the repo: every revision of an overwritten last.ckpt stays "
        "in LFS history. Squash it when the repo size stops matching what the tree above accounts "
        "for (irreversible; HF's storage quota takes up to 36h to reflect it):"
    )
    report.raw(
        'uv run python -c "from huggingface_hub import HfApi; '
        f"HfApi().super_squash_history(repo_id='{repo_id}', repo_type='model')\""
    )
    return report


def _rows(
    runs: list[RunEntry], reasons: dict[str, list[str]], *, with_experiment: bool
) -> tuple[list[str], list[list]]:
    """`(headers, rows)` for a group of runs."""
    show_state = len({run.state for run in runs}) > 1
    show_step = any(run.global_step is not None for run in runs)

    headers = (["experiment"] if with_experiment else []) + ["when"]
    if show_state:
        headers.append("state")
    headers += ["her", "seed"]
    if show_step:
        headers.append("step")
    headers += ["#ckpt", "size", "notes"]

    rows = [
        [
            *([run.experiment] if with_experiment else []),
            run.when,
            *([styled(run.state, STATE_STYLES.get(run.state, ""))] if show_state else []),
            run.her_ratio,
            run.seed,
            *([_shown(run.global_step)] if show_step else []),
            len(run.checkpoints),
            human_bytes(run.total_bytes),
            _notes_cell(observations(run), bool(reasons[run.prefix])),
        ]
        for run in runs
    ]
    return headers, rows


def _notes_cell(notes: list[str], is_prune_candidate: bool):
    """The `notes` column: dimmed when merely informational, red once it justifies a deletion."""
    if not notes:
        return "-"
    return styled(",".join(notes), "red" if is_prune_candidate else "dim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Checkpoint repo to inspect. (default: the HF_CHECKPOINT_REPO_ID env var)",
    )
    parser.add_argument(
        "--sort",
        choices=["date", "size", "name"],
        default="name",
        help="Row ordering. (default: name, grouping a run with its siblings)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Also write the report to this path, as a .txt.",
    )
    parser.add_argument(
        "--no-details",
        dest="details",
        action="store_false",
        help="Skip reading each run's status marker and config. Statuses still come from the file "
        "listing; the her/seed/step columns go blank, in exchange for one HTTP call total.",
    )
    return parser.parse_args()


def main() -> None:
    from policy.utils.hf_hub_utils import default_checkpoint_repo_id

    args = parse_args()
    repo_id = args.repo_id or default_checkpoint_repo_id()
    if not repo_id:
        raise SystemExit(
            "No checkpoint repo to inspect. Pass --repo-id, or export HF_CHECKPOINT_REPO_ID "
            "(and run `uv run hf auth login` if the repo is private)."
        )

    print(f"Listing {repo_id}...")
    runs = collect_runs(repo_id)
    if not runs:
        raise SystemExit(f"No run directories found in '{repo_id}'.")
    if args.details:
        load_details(repo_id, runs)

    build_report(repo_id, runs, args.sort).emit(args.out, save=args.out is not None)


if __name__ == "__main__":
    main()
