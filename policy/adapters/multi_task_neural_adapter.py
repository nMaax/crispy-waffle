from collections.abc import Sequence

import torch
from typing_extensions import deprecated

from policy.algorithms.multi_task_state_aligner import MultiTaskStateAligner
from policy.transforms import observation_pipeline
from policy.utils.typing_utils import AdapterProtocol, TensorTree


@deprecated("Adapters are deprecated and no longer maintained.")
class MultiTaskNeuralAdapter(AdapterProtocol):
    def __init__(
        self,
        ckpt_path: str,
        task_name: str,
        passthrough_mapping: Sequence[tuple[str, int]] | None = None,
    ):
        self.model = MultiTaskStateAligner.load_from_checkpoint(ckpt_path, strict=False)
        self.model.eval()
        self.model.freeze()

        self.task_name = task_name

        if hasattr(self.model, "task_mapping"):
            task_mapping = self.model.task_mapping
        else:
            task_mapping = getattr(self.model.hparams, "task_mapping", None)
            if task_mapping is None:
                raise ValueError("The loaded model does not have a task_mapping attribute.")

        self.task_idx = task_mapping[task_name]
        self.passthrough_mapping = passthrough_mapping

    def apply(self, obs: TensorTree) -> TensorTree:
        is_flat = isinstance(obs, torch.Tensor)
        canonicalize = observation_pipeline(
            self.task_name, is_flat=is_flat, canonicalize=True, as_dict=False
        )
        with torch.no_grad():
            canonical_obs = canonicalize(obs)
        assert isinstance(canonical_obs, torch.Tensor)
        return self._run_model(canonical_obs)

    def _run_model(self, obs: torch.Tensor) -> torch.Tensor:
        # If obs is unbatched [39], task_idx must be a scalar []
        # If obs is batched [B, 39], task_idx must be [B]
        if obs.dim() == 1:
            task_idx = torch.tensor(self.task_idx, dtype=torch.long, device=obs.device)
        else:
            task_idx = torch.full(
                obs.shape[:-1], self.task_idx, dtype=torch.long, device=obs.device
            )

        if self.model.device != obs.device:
            self.model.to(obs.device)

        with torch.no_grad():
            prediction = self.model(obs, task_idx)

        result = torch.zeros(
            (*obs.shape[:-1], prediction.shape[-1]), dtype=obs.dtype, device=obs.device
        )

        if self.passthrough_mapping:
            for in_slice, out_slice in self.passthrough_mapping:
                result[..., out_slice] = prediction[..., in_slice]
        else:
            result = prediction

        return result
