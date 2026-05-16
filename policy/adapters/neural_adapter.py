from collections.abc import Sequence

import torch

from policy.algorithms.state_aligner import StateAligner
from policy.utils.hydra_utils import parse_slice, slice_size
from policy.utils.typing_utils import AdapterProtocol


class NeuralAdapter(AdapterProtocol):
    def __init__(
        self,
        ckpt_path: str,
        passthrough_mapping: Sequence[tuple[str, int]] | None = None,
    ):
        self.model = StateAligner.load_from_checkpoint(ckpt_path, strict=False)
        self.model.eval()
        self.model.freeze()

        self.passthrough_mapping = []
        if passthrough_mapping is not None:
            for in_s, out_s in passthrough_mapping:
                in_s, out_s = parse_slice(in_s), parse_slice(out_s)
                if slice_size(in_s) != slice_size(out_s):
                    raise ValueError(
                        f"Passthrough mapping slices must have the same size: {in_s} vs {out_s}"
                    )
                self.passthrough_mapping.append((in_s, out_s))

    def apply(
        self, obs: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if self.model.network is None:
            raise ValueError(
                "Model does not have a network attribute. Cannot apply NeuralAdapter."
            )

        if self.model.device != obs.device:
            self.model.to(obs.device)

        # Snap weights to nearest integer (0, 1, or -1) to induce a pure permutation matrix if the model is a single linear layer without bias.
        #
        # NOTE: In StackCubeSwapped doing this allow us to go from 66% to 89% as expected
        #
        # layer0 = self.model.network.net[0]
        # if (
        #     isinstance(self.model.network.net, torch.nn.Sequential)
        #     and len(self.model.network.net) == 1
        #     and isinstance(layer0, torch.nn.Linear)
        #     and layer0.bias is None
        # ):
        #     with torch.no_grad():
        #         layer0.weight.copy_(torch.round(layer0.weight))

        with torch.no_grad():
            result = self.model(obs)

        if self.passthrough_mapping:
            for in_slice, out_slice in self.passthrough_mapping:
                result[..., out_slice] = obs[..., in_slice]

        return result

    def _apply_to_dict(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("LearnedMLPAdapter does not support dict observations yet.")
