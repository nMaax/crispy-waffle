from __future__ import annotations

import typing
from collections.abc import Mapping
from typing import Any, Literal, ParamSpec, Protocol, TypeVar, runtime_checkable

import torch

if typing.TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

    from policy.utils.typing_utils import TensorTree

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

    obs_horizon: int
    """Number of past observations used to build the observations window."""

    device: torch.device
    """Device on which the policy parameters live."""

    def get_action(
        self,
        obs_seq: torch.Tensor | Mapping[str, Any] | None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Return a sequence of actions given a (batched) observations window.

        Args:
            obs_seq: Either a float tensor of shape ``(B, obs_horizon, obs_dim)`` or a
                nested dict of such tensors.

        Returns:
            Action tensor of shape ``(B, act_horizon, act_dim)``.
        """
        ...


@runtime_checkable
class GoalConditionedPolicyProtocol(Protocol):
    """Protocol for goal-conditioned imitation-learning policies that can be used during rollout
    evaluation."""

    obs_horizon: int
    """Number of past observations used to build the observations window."""

    device: torch.device
    """Device on which the policy parameters live."""

    def get_action(
        self,
        obs_seq: torch.Tensor | Mapping[str, Any] | None,
        goal: torch.Tensor | Mapping[str, Any] | None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Return a sequence of actions given a (batched) observations window and a goal.

        Args:
            obs_seq: Either a float tensor of shape ``(B, obs_horizon, obs_dim)`` or a
                nested dict of such tensors.
            goal: Either a float tensor of shape ``(B, obs_dim)`` or a nested dict of such
                tensors.

        Returns:
            Action tensor of shape ``(B, act_horizon, act_dim)``.
        """
        ...


@runtime_checkable
class DiffusionNetworkProtocol(Protocol):
    """Protocol defining the expected interface for diffusion policy networks (e.g. UNet, GPT)."""

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        external_cond: Mapping[str, TensorTree],
    ) -> torch.Tensor:
        """Predicts the noise or target action sequence.

        Args:
            sample: Tensor of shape (B, pred_horizon, act_dim) or (B, seq_len, act_dim)
            timestep: Tensor of shape (B,) or scalar representing the timestep/noise level
            external_cond: Conditioning tensor tree (e.g. ``{"obs": ...}`` or
                ``{"obs": ..., "goal": ...}``).

        Returns:
            Tensor of same shape as sample (predicted noise or target action sequence)
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
class EnvProtocol(Protocol):
    """Protocol representing a standard environment (e.g., gym.Env)."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...


@runtime_checkable
class GoalConditionedEnvProtocol(EnvProtocol, Protocol):
    """Protocol for goal-conditioned environments that can generate heuristic goals."""

    def generate_heuristic_goal(self) -> torch.Tensor | dict[str, Any]:
        """Generate a heuristic goal state for the environment."""
        ...
