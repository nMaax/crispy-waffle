"""Utilities for reconstructing algorithm instances directly from Lightning checkpoints, outside
the Hydra `experiment=...` entry points (`policy/main.py` / `policy/eval.py`) — used by the
standalone analysis scripts under `scripts/`."""

from pathlib import Path
from typing import Any

import torch

from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy


def load_goal_conditioned_diffusion_policy(ckpt_path: Path) -> GoalConditionedDiffusionPolicy:
    """Reconstructs a `GoalConditionedDiffusionPolicy` from a checkpoint's own hyperparameters.

    Handles checkpoints that predate the configurable `embedder` (trained as the now-deleted
    `GoalConditionedDiffusionPolicyMLP`, which hard-coded an MLP embedder).
    """
    checkpoint_data: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint_data.get("hyper_parameters", {})

    act_dim = hparams.get("act_dim")
    network_config = dict(hparams.get("network", {}))
    if act_dim is not None:
        network_config["act_dim"] = act_dim

    embedder_config = hparams.get("embedder")
    if embedder_config is None and "state_embedding_dim" in hparams:
        print("Checkpoint predates the embedder config; reconstructing its MLP embedder.")
        embedder_config = {
            "_target_": "policy.algorithms.networks.mlp.MLP",
            "input_dim": hparams.get("task_dim"),
            "output_dim": hparams["state_embedding_dim"],
            "hidden_dims": hparams.get("hidden_dims", [128, 128, 128]),
        }

    model = GoalConditionedDiffusionPolicy.load_from_checkpoint(
        ckpt_path,
        network=network_config,
        embedder=embedder_config,
    )
    model.eval()
    return model
