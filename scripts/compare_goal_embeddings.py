"""Compare goal embeddings across tasks via cosine similarity.

Question this answers: does the state embedder map the *goal* of one task to a different
representation than the goal of another task? The reference task is StackCube-v1, compared
against StackCubeSwapped-v1, PlaceCubeLeft-v1 and PlaceCubeRight-v1.

The yardstick is a within-task baseline: the cosine similarity between two StackCube goals
drawn from *different scenes*. Any cross-task similarity that sits at or above that baseline
means the embedder does not separate the tasks -- it is only reacting to the scene layout.

Goals come from ``env.generate_heuristic_goal()``, the same source the rollout evaluation uses
(``policy/algorithms/callbacks/rollout_evaluation.py``), so no demo dataset is required for
tasks that only ever get evaluated by rollout (e.g. PlaceCubeLeft-v1).

All environments are reset with the same seed, so scene layouts line up one-to-one across
tasks; the script verifies this and reports it. That makes the *same-scene* cross-task
comparison a clean isolation of the goal rule itself.

Example:
    uv run python scripts/compare_goal_embeddings.py --num-scenes 128
"""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import policy.environments  # noqa: F401  (registers StackCubeSwapped-v1, PlaceCubeLeft-v1, ...)
from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.transforms import observation_pipeline
from policy.utils import to_tensor
from policy.utils.typing_utils import GoalConditionedEnvProtocol, get_tensor

REFERENCE_ENV = "StackCube-v1"
COMPARISON_ENVS = ["StackCubeSwapped-v1", "PlaceCubeLeft-v1", "PlaceCubeRight-v1"]

# The three checkpoints under study: the fully-trained (200k step) GCDP-MLP, then the two
# frozen-embedder runs whose embedder stopped learning at 30k and 70k steps.
DEFAULT_CKPTS: dict[str, str] = {
    "GCDP-MLP 200k (never frozen)": (
        "logs/GoalConditionedDiffusionPolicyMLP__StackCube-v1__default__train"
        "/runs/2026-07-25/20-34-11/checkpoints/last.ckpt"
    ),
    "freeze @ 30k steps": "logs/gcdp_freeze30/continue/checkpoints/last.ckpt",
    "freeze @ 70k steps": "logs/gcdp_freeze70/continue/checkpoints/last.ckpt",
}

PAIR_COLORS = {
    "StackCube-v1": "#64748b",
    "StackCubeSwapped-v1": "#f97316",
    "PlaceCubeLeft-v1": "#a3e635",
    "PlaceCubeRight-v1": "#e879f9",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        default=None,
        help="Explicit checkpoint paths (labelled by path); defaults to the three studied runs.",
    )
    p.add_argument("--reference-env", type=str, default=REFERENCE_ENV)
    p.add_argument("--envs", type=str, nargs="+", default=COMPARISON_ENVS)
    p.add_argument("--num-scenes", type=int, default=128, help="Parallel envs = distinct scenes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-mode", type=str, default="state")
    p.add_argument("--control-mode", type=str, default="pd_ee_delta_pos")
    p.add_argument("--out-dir", type=str, default="scripts/figures/ablation")
    p.add_argument("--tag", type=str, default=None, help="Filename suffix for outputs.")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------------------


def load_policy(ckpt_path: Path, device: torch.device) -> GoalConditionedDiffusionPolicy:
    """Loads a GoalConditionedDiffusionPolicy, reconstructing the lazily-built submodules."""
    ckpt: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    network_config = dict(hparams.get("network", {}))
    if hparams.get("act_dim") is not None:
        network_config["act_dim"] = hparams["act_dim"]

    embedder_config = hparams.get("embedder")
    if embedder_config is None and "state_embedding_dim" in hparams:
        # Predates the configurable embedder (hard-coded MLP in the old ...MLP class).
        embedder_config = {
            "_target_": "policy.algorithms.networks.mlp.MLP",
            "input_dim": hparams.get("task_dim"),
            "output_dim": hparams["state_embedding_dim"],
            "hidden_dims": hparams.get("hidden_dims", [128, 128, 128]),
        }

    model = GoalConditionedDiffusionPolicy.load_from_checkpoint(
        ckpt_path, network=network_config, embedder=embedder_config, map_location="cpu"
    )
    if model.embedder is None:
        raise RuntimeError(f"{ckpt_path}: embedder was not built by load_from_checkpoint().")
    model.eval()
    model.to(device)
    return model


def checkpoint_step(ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt.get("global_step", -1))


def embedder_fingerprint(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Raw embedder tensors straight out of the checkpoint, for freeze verification."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        k[len("embedder.") :]: v.clone()
        for k, v in ckpt["state_dict"].items()
        if k.startswith("embedder.")
    }


# --------------------------------------------------------------------------------------
# goal collection
# --------------------------------------------------------------------------------------


def collect_goals(
    env_id: str, num_scenes: int, seed: int, obs_mode: str, control_mode: str
) -> dict[str, torch.Tensor]:
    """Resets ``num_scenes`` parallel scenes and returns the canonicalized heuristic goal.

    Also returns the canonicalized *initial observation* under the ``_obs0`` key so callers can
    check that the scene layout is identical across environments seeded the same way.
    """
    env = gym.make(id=env_id, obs_mode=obs_mode, control_mode=control_mode, num_envs=num_scenes)
    try:
        inner = cast(GoalConditionedEnvProtocol, env.unwrapped)
        obs, _ = env.reset(seed=seed)

        canon_flat = observation_pipeline(
            env_id=env_id, is_flat=True, canonicalize=True, as_dict=True
        )
        canon_dict = observation_pipeline(
            env_id=env_id, is_flat=False, canonicalize=True, as_dict=True
        )

        obs0 = canon_flat(to_tensor(obs, dtype=torch.float32))
        goal = canon_dict(to_tensor(inner.generate_heuristic_goal(), dtype=torch.float32))
        assert isinstance(goal, dict) and isinstance(obs0, dict)

        out = {k: get_tensor(goal, k).detach().cpu() for k in goal}
        out["_obs0"] = torch.cat(
            [get_tensor(obs0, k) for k in ("proprio", "tcp_pose", "a_pose", "b_pose")], dim=-1
        ).cpu()
        return out
    finally:
        env.close()


def embed_goals(
    model: GoalConditionedDiffusionPolicy, goal: dict[str, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (goal_embedding [N, D], raw goal task vector [N, task_dim])."""
    goal = {k: v for k, v in goal.items() if not k.startswith("_")}
    with torch.no_grad():
        # extract_embeddings needs an obs as well; feeding the goal into both slots is harmless
        # because only the goal branch is read back.
        emb = model.extract_embeddings(obs=goal, goal=goal)["goal_embedding"].float().numpy()
        _, task = model._split_proprio_task({k: v.to(model.device) for k, v in goal.items()})
    return emb, task.detach().cpu().float().numpy()


# --------------------------------------------------------------------------------------
# cosine analysis
# --------------------------------------------------------------------------------------


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def within_task_cosines(A: np.ndarray) -> np.ndarray:
    """Cosine between every pair of *distinct* scenes of the same task (the baseline)."""
    S = _unit(A) @ _unit(A).T
    iu = np.triu_indices(len(A), k=1)
    return S[iu]


def cross_task_cosines(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (same-scene cosines [N], different-scene cosines [N*(N-1)])."""
    S = _unit(A) @ _unit(B).T
    same = np.diag(S).copy()
    mask = ~np.eye(len(A), dtype=bool)
    return same, S[mask]


def describe(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "n": int(x.size),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (np.var(a) + np.var(b)))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("inf")


def analyze(
    ref: np.ndarray, others: dict[str, np.ndarray], center: bool
) -> dict[str, Any]:
    """Cosine analysis of one embedding set against the reference task's set.

    ``center`` subtracts the mean over *all* tasks first. Without centering, a large shared
    offset in the embeddings pushes every cosine towards 1 and hides real differences; with
    centering the cosine measures how the goals differ around that common component.
    """
    if center:
        pool = np.concatenate([ref, *others.values()], axis=0)
        mu = pool.mean(axis=0, keepdims=True)
        ref = ref - mu
        others = {k: v - mu for k, v in others.items()}

    baseline = within_task_cosines(ref)
    out: dict[str, Any] = {"baseline_within_reference": describe(baseline), "pairs": {}}

    for name, B in others.items():
        same, diff = cross_task_cosines(ref, B)
        out["pairs"][name] = {
            "same_scene": describe(same),
            "diff_scene": describe(diff),
            "delta_vs_baseline": float(np.mean(same) - np.mean(baseline)),
            "cohens_d_vs_baseline": cohens_d(same, baseline),
            "frac_below_baseline_p05": float(np.mean(same < np.percentile(baseline, 5))),
            "within_task_baseline_of_other": describe(within_task_cosines(B)),
        }
    return out


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


def print_report(label: str, res: dict[str, Any], ref_env: str) -> None:
    print(f"\n{'=' * 96}")
    print(f"  {label}")
    print("=" * 96)

    for mode in ("raw", "centered"):
        r = res[mode]
        b = r["baseline_within_reference"]
        print(f"\n  -- {mode} cosine " + "-" * 74)
        print(
            f"    BASELINE  {ref_env} vs {ref_env} (different scenes):"
            f"   {b['mean']:+.4f} ± {b['std']:.4f}   [p05 {b['p05']:+.3f}, p95 {b['p95']:+.3f}]"
        )
        print(
            f"\n    {'vs ' + ref_env:<28} {'same scene':>18} {'diff scene':>18} "
            f"{'Δ vs base':>10} {'d':>7}"
        )
        for name, m in r["pairs"].items():
            s, d = m["same_scene"], m["diff_scene"]
            print(
                f"    {name:<28} {s['mean']:+.4f} ± {s['std']:.4f} "
                f"{d['mean']:+.4f} ± {d['std']:.4f} "
                f"{m['delta_vs_baseline']:+10.4f} {m['cohens_d_vs_baseline']:+7.2f}"
            )


def plot_distributions(
    per_ckpt: dict[str, dict[str, Any]], ref_env: str, out_path: Path
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#1e293b",
            "text.color": "#f8fafc",
            "axes.labelcolor": "#94a3b8",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#64748b",
            "grid.color": "#334155",
            "grid.alpha": 0.5,
        }
    )

    labels = list(per_ckpt)
    modes = ["raw", "centered"]
    fig, axes = plt.subplots(
        len(modes),
        len(labels),
        figsize=(5.0 * len(labels), 4.4 * len(modes)),
        squeeze=False,
        sharey="row",
    )

    for row, mode in enumerate(modes):
        for col, label in enumerate(labels):
            ax = axes[row][col]
            samples = per_ckpt[label][f"{mode}_samples"]
            names = list(samples)
            data = [samples[n] for n in names]

            parts = ax.violinplot(data, showextrema=False, widths=0.85)
            for body, n in zip(parts["bodies"], names, strict=True):  # type: ignore[arg-type]
                key = ref_env if n.startswith("BASELINE") else n
                body.set_facecolor(PAIR_COLORS.get(key, "#cbd5e1"))
                body.set_alpha(0.75)
                body.set_edgecolor("#0f172a")

            for i, d in enumerate(data, start=1):
                ax.scatter(i, np.mean(d), color="#f8fafc", s=18, zorder=5)

            base_mean = float(np.mean(samples[names[0]]))
            ax.axhline(base_mean, color="#94a3b8", linestyle="--", linewidth=1.0, zorder=1)

            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(
                ["baseline\n(same task,\ndiff. scene)"]
                + [n.replace("-v1", "").replace("StackCube", "SC").replace("PlaceCube", "Place")
                   for n in names[1:]],
                fontsize=9,
            )
            ax.set_ylabel(f"{mode} cosine to {ref_env} goal", fontsize=10)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold", pad=12)

    fig.suptitle(
        f"Goal-embedding cosine similarity against {ref_env} (same scene, different task)\n"
        "dashed line = within-task baseline mean",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nFigure saved to {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = {c: Path(c) for c in args.ckpt} if args.ckpt else {
        k: Path(v) for k, v in DEFAULT_CKPTS.items()
    }
    missing = [str(p) for p in ckpts.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing checkpoints:\n  " + "\n  ".join(missing))

    all_envs = [args.reference_env, *args.envs]

    print(f"Collecting heuristic goals for {args.num_scenes} scenes per task (seed {args.seed})…")
    goals = {
        env_id: collect_goals(
            env_id, args.num_scenes, args.seed, args.obs_mode, args.control_mode
        )
        for env_id in all_envs
    }

    print("\nScene-layout alignment (max |Δ| of the initial canonical observation vs reference):")
    aligned = True
    for env_id in args.envs:
        delta = float((goals[env_id]["_obs0"] - goals[args.reference_env]["_obs0"]).abs().max())
        aligned &= delta < 1e-4
        print(f"  {args.reference_env} vs {env_id:<24} {delta:.3e}")
    print(
        "  -> scenes are paired; same-scene cosines isolate the goal rule."
        if aligned
        else "  -> scenes DIFFER across tasks; read same-scene numbers as distributional only."
    )

    per_ckpt: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "config": vars(args),
        "scenes_aligned": aligned,
        "checkpoints": {},
    }

    # Raw goal space, for reference: how different are the goals *before* the embedder?
    raw_by_task: dict[str, np.ndarray] | None = None

    for label, path in ckpts.items():
        model = load_policy(path, device)
        emb_by_task: dict[str, np.ndarray] = {}
        raw_now: dict[str, np.ndarray] = {}
        for env_id in all_envs:
            emb, raw = embed_goals(model, goals[env_id])
            emb_by_task[env_id] = emb
            raw_now[env_id] = raw
        raw_by_task = raw_by_task or raw_now

        ref = emb_by_task[args.reference_env]
        others = {e: emb_by_task[e] for e in args.envs}

        res: dict[str, Any] = {
            "checkpoint": str(path),
            "global_step": checkpoint_step(path),
            "raw": analyze(ref, others, center=False),
            "centered": analyze(ref, others, center=True),
        }

        # Keep the raw sample arrays around for plotting.
        for mode, center in (("raw", False), ("centered", True)):
            r, o = ref, others
            if center:
                mu = np.concatenate([ref, *others.values()], axis=0).mean(axis=0, keepdims=True)
                r, o = ref - mu, {k: v - mu for k, v in others.items()}
            samples = {f"BASELINE {args.reference_env}": within_task_cosines(r)}
            for name, B in o.items():
                samples[name] = cross_task_cosines(r, B)[0]
            res[f"{mode}_samples"] = samples

        per_ckpt[label] = res
        report["checkpoints"][label] = {
            k: v for k, v in res.items() if not k.endswith("_samples")
        }
        print_report(f"{label}   (step {res['global_step']}, {path})", res, args.reference_env)

        del model
        torch.cuda.empty_cache()

    assert raw_by_task is not None
    report["raw_goal_space"] = {
        "raw": analyze(
            raw_by_task[args.reference_env],
            {e: raw_by_task[e] for e in args.envs},
            center=False,
        ),
        "centered": analyze(
            raw_by_task[args.reference_env],
            {e: raw_by_task[e] for e in args.envs},
            center=True,
        ),
    }
    print_report(
        "RAW GOAL VECTORS (before the embedder) -- reference point",
        report["raw_goal_space"],
        args.reference_env,
    )

    print("\nFreeze check (max |Δ| between pretrain and continue embedder weights):")
    for n in (30, 70):
        pre = Path(f"logs/gcdp_freeze{n}/pretrain/checkpoints/last.ckpt")
        cont = Path(f"logs/gcdp_freeze{n}/continue/checkpoints/last.ckpt")
        if not (pre.exists() and cont.exists()):
            continue
        a, b = embedder_fingerprint(pre), embedder_fingerprint(cont)
        delta = max(float((a[k] - b[k]).abs().max()) for k in a)
        print(f"  freeze{n:<3} {delta:.3e}  {'(frozen)' if delta == 0.0 else '(CHANGED)'}")

    tag = args.tag or f"seed{args.seed}_n{args.num_scenes}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"goal_cosine_comparison_{tag}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nMetrics written to {json_path.resolve()}")

    if not args.no_plot:
        plot_distributions(
            per_ckpt, args.reference_env, out_dir / f"goal_cosine_by_task_{tag}.png"
        )


if __name__ == "__main__":
    main()
