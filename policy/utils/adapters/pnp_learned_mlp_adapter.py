from collections.abc import Sequence

import torch

from policy.algorithms.state_translator import StateTranslator
from policy.utils.adapters.canonical_pnp_adapter import CanonicalPnPAdapter
from policy.utils.adapters.learned_mlp_adapter import LearnedMLPAdapter


class PnPLearnedMLPAdapter(LearnedMLPAdapter):
    def __init__(
        self,
        env_id: str,
        ckpt_path: str,
        task_mapping: dict[str, int],
        passthrough_mapping: Sequence[tuple[str, int]] | None = None,
    ):
        self.model = StateTranslator.load_from_checkpoint(ckpt_path, strict=False)
        self.model.eval()
        self.model.freeze()
        self.passthrough_mapping = passthrough_mapping

        self.env_id = env_id
        self.pnp_adapter = CanonicalPnPAdapter(self.env_id)
        self.task_mapping = task_mapping

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.pnp_adapter.apply(obs)
        task_idx_val = self.task_mapping[self.env_id]

        # If obs is unbatched [39], task_idx must be a scalar []
        # If obs is batched [B, 39], task_idx must be [B]
        if obs.dim() == 1:
            task_idx = torch.tensor(task_idx_val, dtype=torch.long, device=obs.device)
        else:
            task_idx = torch.full(
                obs.shape[:-1], task_idx_val, dtype=torch.long, device=obs.device
            )

        with torch.no_grad():
            model_predicted_swaps = self.model(obs, task_idx)

        swapped = torch.zeros(
            (*obs.shape[:-1], model_predicted_swaps.shape[-1]), dtype=obs.dtype, device=obs.device
        )

        if self.passthrough_mapping:
            for in_slice, out_slice in self.passthrough_mapping:
                swapped[..., out_slice] = model_predicted_swaps[..., in_slice]
        else:
            swapped = model_predicted_swaps

        return swapped
