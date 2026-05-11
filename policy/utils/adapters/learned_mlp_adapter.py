from collections.abc import Sequence

import torch

from policy.algorithms.mlp_adapter import MLPAdapter
from policy.utils.adapters.stack_cube_permuter import CubesPermuter, IndexSelector
from policy.utils.hydra_utils import parse_slice


class LearnedMLPAdapter(CubesPermuter):
    def __init__(
        self,
        ckpt_path: str,
        selector: IndexSelector | list[int] | torch.Tensor | None = None,
        passthrough_mapping: Sequence[tuple[str, int]] | None = None,
    ):
        super().__init__(selector=selector)
        self.model = MLPAdapter.load_from_checkpoint(ckpt_path, strict=False)

        self.model.eval()
        self.model.freeze()

        # TODO: should be moved to CUDA, but in a more graceful way
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        self.passthrough_mapping = []
        if passthrough_mapping is not None:
            for in_s, out_s in passthrough_mapping:
                self.passthrough_mapping.append((parse_slice(in_s), parse_slice(out_s)))

    def _apply_to_tensor(self, obs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        swapped = obs.clone()
        with torch.no_grad():
            model_predicted_swaps = self.model(obs)

        swapped[indices] = model_predicted_swaps[indices]

        if self.passthrough_mapping:
            for in_slice, out_slice in self.passthrough_mapping:
                swapped[..., out_slice] = obs[..., in_slice]

        return swapped

    def _apply_to_dict(
        self, obs: dict[str, torch.Tensor], indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("LearnedMLPAdapter does not support dict observations yet.")
