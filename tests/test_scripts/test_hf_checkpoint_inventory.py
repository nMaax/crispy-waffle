"""Tests for how the checkpoint inventory compares runs.

All three cases here are bugs found by running the script against the real repo: multirun paths lost
their date, `superseded` could never fire, and every legacy run was proposed for deletion.
"""

import pytest

from scripts.demos.hf_checkpoint_inventory import (
    RunEntry,
    build_report,
    collect_runs,
    observations,
    prune_reasons,
    run_prefix_of,
    superseded_by,
)

EXPERIMENT = "GoalConditionedDiffusionPolicyAttentionMLPPoolDeltaInput__X-v1__default__train"


def run(tail: str, *, checkpoints: int = 4, marker: str | None = None, config=True) -> RunEntry:
    """A run entry for `logs/<EXPERIMENT>/<tail>`, e.g. `runs/2026-08-05/21-19-46`."""
    return RunEntry(
        prefix=f"logs/{EXPERIMENT}/{tail}",
        checkpoints=["last.ckpt"] + [f"step_{i}.ckpt" for i in range(checkpoints - 1)],
        has_hydra_config=config,
        marker=marker,
    )


def as_dict(*runs: RunEntry) -> dict[str, RunEntry]:
    return {entry.prefix: entry for entry in runs}


class TestWhen:
    def test_single_run_keeps_date_and_time(self):
        assert run("runs/2026-08-05/21-13-04").when == "2026-08-05/21-13-04"

    def test_multirun_keeps_the_date_too(self):
        """Taking a fixed number of trailing segments would yield `12-03-25/1`, dropping the
        date."""
        assert run("multiruns/2026-08-08/12-03-25/1").when == "2026-08-08/12-03-25/1"

    def test_orders_chronologically_across_both_layouts(self):
        later = run("multiruns/2026-08-08/12-03-25/1")
        earlier = run("runs/2026-08-05/21-13-04")
        assert later.when > earlier.when

    def test_falls_back_to_the_prefix_for_an_unconventional_path(self):
        entry = RunEntry(prefix="logs/loose/checkpointless")
        assert entry.when == entry.prefix


class TestSuperseded:
    def test_fires_without_any_status_markers(self):
        """Every run predating the marker reads as `unknown`; redundancy is still detectable."""
        old = run("runs/2026-08-05/18-58-03", checkpoints=1)
        new = run("runs/2026-08-05/21-19-46", checkpoints=4)
        runs = as_dict(old, new)

        assert superseded_by(old, runs) is new
        assert superseded_by(new, runs) is None

    def test_a_later_equally_complete_run_wins(self):
        old = run("runs/2026-08-05/21-19-46")
        new = run("multiruns/2026-08-08/12-03-25/0")
        assert superseded_by(old, as_dict(old, new)) is new

    def test_a_later_but_less_complete_run_does_not_supersede(self):
        good = run("runs/2026-08-05/21-19-46", checkpoints=4)
        crashed = run("runs/2026-08-06/09-00-00", checkpoints=1)
        assert superseded_by(good, as_dict(good, crashed)) is None

    def test_a_confirmed_completion_beats_an_unconfirmed_one(self):
        unconfirmed = run("runs/2026-08-05/21-19-46", checkpoints=4)
        confirmed = run("runs/2026-08-06/09-00-00", checkpoints=1, marker="RUN_COMPLETED.json")
        assert confirmed.completeness > unconfirmed.completeness
        assert superseded_by(unconfirmed, as_dict(unconfirmed, confirmed)) is confirmed

    def test_a_different_experiment_never_supersedes(self):
        mine = run("runs/2026-08-05/21-19-46")
        other = RunEntry(
            prefix="logs/SomethingElse__X-v1__default__train/runs/2026-09-09/09-09-09"
        )
        assert superseded_by(mine, as_dict(mine, other)) is None

    def test_a_different_sweep_point_never_supersedes(self):
        """her_ratio=0.8 and her_ratio=0.0 are two intentional runs, not one redundant with the
        other, even under the same experiment name and even if one ran on a later day."""
        early = run("runs/2026-08-05/21-19-46")
        early.config = {"her_ratio": 0.8, "seed": 2}
        later = run("multiruns/2026-08-08/12-03-25/0")
        later.config = {"her_ratio": 0.0, "seed": 2}
        assert superseded_by(early, as_dict(early, later)) is None
        assert superseded_by(later, as_dict(early, later)) is None


class TestPruneReasons:
    def test_a_sole_copy_is_never_a_prune_candidate(self):
        """The signature of a hard-killed run, but it is the only copy -- keep it."""
        only = run("runs/2026-08-05/19-15-44", checkpoints=1, marker=None)
        assert observations(only) == ["no-marker", "last-only"]
        assert prune_reasons(only, as_dict(only)) == []

    def test_missing_marker_alone_is_not_a_reason(self):
        newest = run("runs/2026-08-08/13-38-21", marker=None)
        assert prune_reasons(newest, as_dict(newest)) == []

    def test_a_superseded_run_lists_the_newer_one_and_its_observations(self):
        old = run("runs/2026-08-05/18-58-03", checkpoints=1)
        new = run("runs/2026-08-05/21-19-46", checkpoints=4)
        reasons = prune_reasons(old, as_dict(old, new))
        assert reasons[0] == "superseded by 2026-08-05/21-19-46"
        assert "last-only" in reasons


class TestObservations:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"marker": "RUN_COMPLETED.json"}, []),
            ({}, ["no-marker"]),
            ({"config": False, "marker": "RUN_COMPLETED.json"}, ["no-config"]),
            ({"checkpoints": 1, "marker": "RUN_COMPLETED.json"}, ["last-only"]),
        ],
    )
    def test_flags(self, kwargs, expected):
        assert observations(run("runs/2026-08-05/21-19-46", **kwargs)) == expected

    def test_sigterm_comes_from_the_marker_body(self):
        entry = run("runs/2026-08-05/21-19-46", marker="RUN_COMPLETED.json")
        entry.details = {"received_sigterm": True}
        assert observations(entry) == ["sigterm"]


def header_of(text: str) -> str:
    """The first table's header row, out of a rendered report."""
    return next(line for line in text.splitlines() if line.startswith("  | "))


class TestBuildReport:
    def test_groups_runs_under_their_experiment(self):
        """The experiment name is too wide to repeat per row, so it becomes a section heading.

        The heading keeps the full name (including the `GoalConditioned` prefix `compact_name`
        strips for filenames) -- the report has room for it and truncating it here hid real
        experiment identity, not just repetition.
        """
        text = build_report("org/repo", as_dict(run("runs/2026-08-05/21-19-46")), "name").render()
        assert f"## {EXPERIMENT}" in text
        assert "experiment" not in header_of(text)

    def test_flat_sort_keeps_the_experiment_column(self):
        text = build_report("org/repo", as_dict(run("runs/2026-08-05/21-19-46")), "size").render()
        assert "experiment" in header_of(text)

    def test_step_column_is_dropped_when_no_run_has_one(self):
        runs = as_dict(run("runs/2026-08-05/21-19-46"))
        assert "step" not in header_of(build_report("org/repo", runs, "name").render())

    def test_state_column_is_dropped_when_every_run_agrees(self):
        """All legacy runs read `unknown`; a column of one repeated value is noise."""
        runs = as_dict(run("runs/2026-08-05/21-19-46"), run("runs/2026-08-06/09-00-00"))
        assert "state" not in header_of(build_report("org/repo", runs, "name").render())

    def test_state_column_appears_once_runs_differ(self):
        runs = as_dict(
            run("runs/2026-08-05/21-19-46", marker="RUN_COMPLETED.json"),
            run("runs/2026-08-06/09-00-00"),
        )
        assert "state" in header_of(build_report("org/repo", runs, "name").render())

    def test_step_column_appears_once_a_run_reports_one(self):
        entry = run("runs/2026-08-05/21-19-46", marker="RUN_COMPLETED.json")
        entry.details = {"global_step": 195000}
        text = build_report("org/repo", as_dict(entry), "name").render()
        assert "195000" in text

    def test_her_and_seed_come_from_the_fetched_config(self):
        entry = run("runs/2026-08-05/21-19-46")
        entry.config = {"her_ratio": 0.8, "seed": 42}
        text = build_report("org/repo", as_dict(entry), "name").render()
        assert "0.8" in text
        assert "42" in text


class TestRunPrefixOf:
    def test_normal_run_prefix(self):
        assert (
            run_prefix_of("logs/exp/runs/2026-08-05/21-19-46/checkpoints/last.ckpt")
            == "logs/exp/runs/2026-08-05/21-19-46"
        )

    def test_legacy_path_is_ignored(self):
        assert (
            run_prefix_of(".legacy/logs/exp/runs/2026-08-05/21-19-46/checkpoints/last.ckpt")
            is None
        )
        assert run_prefix_of(".legacy/checkpoints/last.ckpt") is None


class TestCollectRuns:
    def test_ignores_legacy_folder_and_flags_it(self, monkeypatch):
        class FakeItem:
            def __init__(self, path: str, size: int = 100):
                self.path = path
                self.size = size

        items = [
            FakeItem("logs/exp/runs/2026-08-05/21-19-46/checkpoints/last.ckpt"),
            FakeItem("logs/exp/runs/2026-08-05/21-19-46/.hydra/config.yaml"),
            FakeItem(".legacy/logs/old_exp/runs/2026-08-01/10-00-00/checkpoints/last.ckpt"),
            FakeItem(".legacy/some_file.txt"),
        ]

        class FakeHfApi:
            def list_repo_tree(self, repo_id, repo_type="model", recursive=True):
                return items

        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        runs, has_legacy = collect_runs("org/repo")
        assert has_legacy is True
        assert "logs/exp/runs/2026-08-05/21-19-46" in runs
        assert len(runs) == 1
        assert not any(".legacy" in prefix for prefix in runs)
