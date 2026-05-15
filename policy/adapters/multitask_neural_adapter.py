from collections.abc import Sequence

import torch

from policy.algorithms.state_translator import StateTranslator
from policy.transforms.pnp_canonicalizer import PnPCanonicalizer
from policy.utils.typing_utils import AdapterProtocol


class MultitaskNeuralAdapter(AdapterProtocol):
    def __init__(
        self,
        ckpt_path: str,
        env_id: str,
        env_idx: int,
        task_to_idx: dict[str, int],
        passthrough_mapping: Sequence[tuple[str, int]] | None = None,
    ):
        self.model = StateTranslator.load_from_checkpoint(ckpt_path, strict=False)
        self.model.eval()
        self.model.freeze()

        self.env_id = env_id
        self.env_idx = env_idx
        self.pnp_canonicalizer = PnPCanonicalizer(env_id)
        self.task_mapping = task_to_idx

        self.passthrough_mapping = passthrough_mapping

    def apply(
        self, obs: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.pnp_canonicalizer.apply(obs)

        # If obs is unbatched [39], task_idx must be a scalar []
        # If obs is batched [B, 39], task_idx must be [B]
        if obs.dim() == 1:
            task_idx = torch.tensor(self.env_idx, dtype=torch.long, device=obs.device)
        else:
            task_idx = torch.full(
                obs.shape[:-1], self.env_idx, dtype=torch.long, device=obs.device
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

    def _apply_to_dict(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("LearnedMLPAdapter does not support dict observations yet.")
