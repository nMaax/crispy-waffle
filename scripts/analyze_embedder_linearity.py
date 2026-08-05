"""Check whether a GoalConditionedDiffusionPolicy's state embedder behaves like an (overly complex)
linear or identity function, using real held-out data.

Captures the embedder's actual input/output pairs as they flow during training/inference --
whichever `goal_delta` mode the checkpoint was trained with (absolute, delta-before-embed
"input", or delta-after-embed "embedding") -- then reports how well a single affine map explains
the mapping (R^2), how linearly related the two representations are overall (linear CKA), and how
directionally aligned the true output is with what that affine map would predict (cosine
similarity). Strictly scoped to the embedder/encoder MLP; the downstream UNet/FiLM network is
never touched.
"""

import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import r2_score

from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.utils import map_leaves
from policy.utils.checkpoint_utils import load_goal_conditioned_diffusion_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_glob",
        type=str,
        default="logs/GoalConditionedDiffusionPolicyMLPDeltaInput__*/**/checkpoints/last.ckpt",
        help="Recursive glob (relative to cwd) selecting checkpoints to analyze.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        action="append",
        default=[],
        help="Explicit checkpoint path to analyze in addition to --ckpt_glob matches. Repeatable.",
    )
    parser.add_argument(
        "--include_intermediate",
        action="store_true",
        help="Also match intermediate step_*.ckpt files alongside last.ckpt when resolving "
        "--ckpt_glob.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which dataset split to draw real embedder inputs from.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Cap on the number of batches to draw from the dataloader. Default: the whole split.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of captured samples held out to score the best-fit linear map's R^2 / "
        "cosine similarity out-of-sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the train/test split.")
    return parser.parse_args()


def resolve_checkpoints(
    ckpt_glob: str, extra_paths: list[str], include_intermediate: bool
) -> list[Path]:
    """Resolves --ckpt_glob (optionally widened to intermediate steps) plus explicit --ckpt_path
    entries into a deduplicated, order-preserving list of checkpoint paths."""
    root = Path(".")
    matches = list(root.glob(ckpt_glob))
    if include_intermediate:
        matches += list(root.glob(ckpt_glob.replace("last.ckpt", "step_*.ckpt")))
    matches += [Path(p) for p in extra_paths]

    seen: set[Path] = set()
    resolved: list[Path] = []
    for path in matches:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return sorted(resolved)


def build_datamodule(ckpt_path: Path):
    """Instantiates the datamodule this checkpoint was actually trained with, from its saved Hydra
    run config next to `checkpoints/` -- so the embedder sees the same real, canonicalized, dict-
    structured observations it saw at train time, not hand-rolled/raw ones."""
    config_file = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No saved Hydra run config found at {config_file}; cannot reconstruct the exact "
            "datamodule this checkpoint was trained with."
        )
    cfg = OmegaConf.load(config_file)
    return hydra.utils.instantiate(cfg.datamodule, num_workers=0)


def capture_embedder_io(
    model: GoalConditionedDiffusionPolicy,
    dataloader,
    num_batches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hooks `model.embedder` and drives the model's own conditioning-building logic over real
    batches, capturing every (input, output) pair the embedder actually sees.

    Calls
    `_build_external_cond` directly rather than `get_action`, so the UNet/FiLM network and its
    multi-step denoising loop are never invoked -- only the embedder runs.
    """
    captured_inputs: list[torch.Tensor] = []
    captured_outputs: list[torch.Tensor] = []

    def hook(_module, inputs, output):
        captured_inputs.append(inputs[0].detach().cpu())
        captured_outputs.append(output.detach().cpu())

    assert model.embedder is not None, "configure_model() must run before capturing embedder I/O."
    handle = model.embedder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                obs_seq = map_leaves(lambda t: t.to(model.device), batch["obs_seq"])
                goal = batch.get("goal")
                if goal is not None:
                    goal = map_leaves(lambda t: t.to(model.device), goal)
                if model.obs_normalizer is not None:
                    obs_seq = model.obs_normalizer.normalize(obs_seq)
                    if goal is not None:
                        goal = model.obs_normalizer.normalize(goal)
                model._build_external_cond(obs_seq, goal)
    finally:
        handle.remove()

    if not captured_inputs:
        raise RuntimeError("No batches were available to capture embedder input/output from.")

    x = torch.cat(captured_inputs, dim=0).double().numpy()
    y = torch.cat(captured_outputs, dim=0).double().numpy()
    return x, y


def fit_affine_map(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine map: y ~= x @ w + b."""
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    coeffs, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
    return coeffs[:-1], coeffs[-1]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019): bounded in [0, 1], invariant to rotation, and defined
    for representations of different width -- 1.0 iff `x` and `y` are related by an orthogonal
    transform (of which identity is a special case)."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(y.T @ x, ord="fro") ** 2  # codespell:ignore fro
    norm_x = np.linalg.norm(x.T @ x, ord="fro")  # codespell:ignore fro
    norm_y = np.linalg.norm(y.T @ y, ord="fro")  # codespell:ignore fro
    return float(hsic / (norm_x * norm_y))


def mean_cosine_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Per-row cosine similarity between two equal-shaped matrices; returns (mean, std)."""
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 0
    cos = np.full(a.shape[0], np.nan)
    cos[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return float(np.nanmean(cos)), float(np.nanstd(cos))


def verdict_from_metrics(r2: float, cka: float) -> str:
    if r2 > 0.95 and cka > 0.95:
        return "near-linear -- a single Linear layer would likely explain this embedder about as well."
    if r2 > 0.8 and cka > 0.8:
        return "mostly linear -- some nonlinear contribution, but the bulk is explained by an affine map."
    return "meaningfully nonlinear -- the MLP's nonlinear capacity is doing real work here."


def analyze_checkpoint(ckpt_path: Path, args: argparse.Namespace) -> dict:
    print(f"\n{'=' * 88}\nCheckpoint: {ckpt_path}")

    model = load_goal_conditioned_diffusion_policy(ckpt_path)
    datamodule = build_datamodule(ckpt_path)
    datamodule.setup(stage="fit")
    dataloader = (
        datamodule.train_dataloader() if args.split == "train" else datamodule.val_dataloader()
    )

    # HER goal relabeling (GoalConditionedTrajectoryDataset) draws from the global torch RNG on
    # every __getitem__, so re-seed here to make the captured (input, output) pairs -- and thus
    # the metrics below -- reproducible across runs and comparable across checkpoints.
    torch.manual_seed(args.seed)
    x, y = capture_embedder_io(model, dataloader, args.num_batches)
    n, task_dim = x.shape
    output_dim = y.shape[1]
    print(
        f"  goal_delta={model.goal_delta!r}  task_dim={task_dim}  output_dim={output_dim}  "
        f"samples={n}  (split={args.split!r})"
    )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * args.test_split)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    w, b = fit_affine_map(x[train_idx], y[train_idx])
    y_pred_test = x[test_idx] @ w + b
    r2 = float(r2_score(y[test_idx], y_pred_test))
    cos_mean, cos_std = mean_cosine_similarity(y[test_idx], y_pred_test)
    cka = linear_cka(x, y)

    print(f"  Linear-fit R^2 (held-out, {len(test_idx)} samples):  {r2:.4f}")
    print(f"  Linear CKA(input, output):                         {cka:.4f}")
    print(f"  Cosine sim(output, best-affine-fit prediction):    {cos_mean:.4f} +/- {cos_std:.4f}")

    if task_dim == output_dim:
        cos_id_mean, cos_id_std = mean_cosine_similarity(x, y)
        rel_l2 = float(np.linalg.norm(y - x) / np.linalg.norm(x))
        print(
            f"  [same-dim] Cosine sim(input, output):              "
            f"{cos_id_mean:.4f} +/- {cos_id_std:.4f}"
        )
        print(f"  [same-dim] Relative L2 error ||out-in|| / ||in||:  {rel_l2:.4f}")

    verdict = verdict_from_metrics(r2, cka)
    print(f"  Verdict (heuristic): {verdict}")

    return {
        "ckpt_path": str(ckpt_path),
        "goal_delta": model.goal_delta,
        "task_dim": task_dim,
        "output_dim": output_dim,
        "num_samples": n,
        "r2": r2,
        "cka": cka,
        "cosine_to_linear_fit": cos_mean,
    }


def main() -> None:
    args = parse_args()
    ckpt_paths = resolve_checkpoints(args.ckpt_glob, args.ckpt_path, args.include_intermediate)
    if not ckpt_paths:
        print(f"No checkpoints matched --ckpt_glob={args.ckpt_glob!r} (and no --ckpt_path given).")
        sys.exit(1)

    print(f"Found {len(ckpt_paths)} checkpoint(s) to analyze.")
    results = []
    for ckpt_path in ckpt_paths:
        try:
            results.append(analyze_checkpoint(ckpt_path, args))
        except Exception as e:
            print(f"  Skipping {ckpt_path}: {e}")

    if len(results) > 1:
        print(f"\n{'=' * 88}\nSummary across checkpoints")
        print(f"{'checkpoint':<70} {'R^2':>8} {'CKA':>8} {'cos(lin)':>10}")
        for r in results:
            name = str(r["ckpt_path"])[-70:]
            print(f"{name:<70} {r['r2']:>8.4f} {r['cka']:>8.4f} {r['cosine_to_linear_fit']:>10.4f}")


if __name__ == "__main__":
    main()
