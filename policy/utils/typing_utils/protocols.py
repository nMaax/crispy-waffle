from __future__ import annotations

import typing
from collections.abc import Mapping
from typing import Any, Literal, ParamSpec, Protocol, TypeVar, runtime_checkable

import torch

if typing.TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

P = ParamSpec("P")
OutT = TypeVar("OutT", covariant=True)


@runtime_checkable
class Module(Protocol[P, OutT]):
    """Small protocol that can be used to annotate the input/output types of `torch.nn.Module`s."""

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> OutT:
        raise NotImplementedError

    if typing.TYPE_CHECKING:
        # note: Only define this for typing purposes so that we don't actually override anything.
        def __call__(self, *args: P.args, **kwagrs: P.kwargs) -> OutT: ...

        modules = nn.Module.modules
        named_modules = nn.Module.named_modules
        state_dict = nn.Module.state_dict
        zero_grad = nn.Module.zero_grad
        parameters = nn.Module.parameters
        named_parameters = nn.Module.named_parameters
        cuda = nn.Module.cuda
        cpu = nn.Module.cpu
        # note: the overloads on nn.Module.to cause a bug with missing `self`.
        # This shouldn't be a problem.
        to = nn.Module().to


BatchType = TypeVar("BatchType", covariant=True)


@runtime_checkable
class DataModule(Protocol[BatchType]):
    """Protocol that shows the minimal attributes / methods of the `LightningDataModule` class.

    This is used to type hint the batches that are yielded by the DataLoaders.
    """

    def prepare_data(self) -> None: ...

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None: ...

    def train_dataloader(self) -> DataLoader[BatchType]: ...


@runtime_checkable
class PolicyProtocol(Protocol):
    """Protocol for imitation-learning policies that can be used during rollout evaluation.

    Any LightningModule that satisfies this interface can be used by the
    :class:`RolloutEvaluationCallback` without depending on a specific implementation.
    """

    cond_horizon: int
    """Number of past observations used to build the conditioning window."""

    device: torch.device
    """Device on which the policy parameters live."""

    def get_action(self, cond_seq: torch.Tensor | Mapping[str, Any] | None) -> torch.Tensor:
        """Return a sequence of actions given a (batched) conditioning window.

        Args:
            cond_seq: Either a float tensor of shape ``(B, cond_horizon, cond_dim)`` or a
                nested dict of such tensors, depending on the conditioning source.

        Returns:
            Action tensor of shape ``(B, act_horizon, act_dim)``.
        """
        ...


@runtime_checkable
class DiffusionSchedulerProtocol(Protocol):
    """Protocol defining the expected interface for diffusion noise schedulers."""

    config: dict[str, Any]

    @property
    def timesteps(self) -> torch.Tensor: ...

    def set_timesteps(
        self, num_inference_steps: int, device: str | torch.device | None = None
    ) -> None: ...

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | torch.Tensor
    ) -> torch.Tensor: ...

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor
    ) -> torch.Tensor: ...

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor
    ) -> torch.Tensor: ...

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any | tuple: ...


@runtime_checkable
class AdapterProtocol(Protocol):
    """Protocol for adapters used during rollouts."""

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        """Adapts the given observation to be compatible with the policy's expected input."""
        ...
