import torch

from policy.algorithms.mlp_adapter import MLPAdapter
from policy.utils.adapters.stack_cube_permuter import CubesPermuter, IndexSelector


class LearnedMLPAdapter(CubesPermuter):
    def __init__(
        self,
        ckpt_path: str,
        selector: IndexSelector | list[int] | torch.Tensor | None = None,
    ):
        super().__init__(selector=selector)
        self.model = MLPAdapter.load_from_checkpoint(ckpt_path)

        self.model.eval()
        self.model.freeze()

        # TODO: should be moved to CUDA, but in a more graceful way
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    def _apply_to_tensor(self, obs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        swapped = obs.clone()
        with torch.no_grad():
            swapped[indices] = self.model(swapped[indices])
        return swapped

    def _apply_to_dict(
        self, obs: dict[str, torch.Tensor], indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("LearnedMLPAdapter does not support dict observations yet.")
