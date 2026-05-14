from collections.abc import Sequence

import torch

from policy.algorithms.mlp_adapter import MLPAdapter
from policy.utils.hydra_utils import parse_slice


class LearnedMLPAdapter:
    def __init__(
        self,
        ckpt_path: str,
        passthrough_mapping: Sequence[tuple[str, int]] | None = None,
    ):
        self.model = MLPAdapter.load_from_checkpoint(ckpt_path, strict=False)

        self.model.eval()
        self.model.freeze()

        self.passthrough_mapping = []
        if passthrough_mapping is not None:
            for in_s, out_s in passthrough_mapping:
                self.passthrough_mapping.append((parse_slice(in_s), parse_slice(out_s)))

    def apply(
        self, obs: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            model_predicted_swaps = self.model(obs)

        swapped = torch.zeros(
            (*obs.shape[:-1], model_predicted_swaps.shape[-1]), dtype=obs.dtype, device=obs.device
        )

        if self.passthrough_mapping:
            for in_slice, out_slice in self.passthrough_mapping:
                swapped[..., out_slice] = model_predicted_swaps[..., in_slice]
        else:
            swapped = model_predicted_swaps

        return swapped

    def _apply_to_dict(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("LearnedMLPAdapter does not support dict observations yet.")
