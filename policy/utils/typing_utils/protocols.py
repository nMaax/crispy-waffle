from __future__ import annotations

import typing
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

import torch

if typing.TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from policy.utils.typing_utils import DimSpec, TensorTree

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

    act_horizon: int
    """Number of actions executed per :meth:`get_action` call."""

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

    act_horizon: int
    """Number of actions executed per :meth:`get_action` call."""

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
class TokenizerProtocol(Protocol):
    """Protocol for turning a canonicalized, proprio-already-split-off obs/goal task tree into raw
    (pre-embedder) tokens.

    Named with a `Protocol` suffix to avoid colliding with the concrete
    `StateTokenizer` class (`policy/algorithms/tokenizers/state.py`), one of
    several implementations of this protocol.
    """

    @property
    def output_dim(self) -> int:
        """Width ``D`` of one raw token; becomes the downstream embedder's ``input_dim``."""
        ...

    @property
    def tokens_per_step(self) -> int | None:
        """Number of tokens ``K`` produced per observed timestep (None if dynamic/variable)."""
        ...

    @property
    def token_spec(self) -> DimSpec:
        """Dim spec of what :meth:`tokenize` emits, i.e. the space normalization is fit in."""
        ...

    @property
    def normalization_mask(self) -> TensorTree:
        """``[output_dim]`` bool mask, False on channels an affine rescale would destroy.

        Consumed by the algorithm's obs normalizer so that e.g. one-hot role indicators survive
        normalization instead of being z-scored to zero.
        """
        ...

    supports_single_side: ClassVar[bool]
    """Whether :meth:`tokenize` can be called with exactly one of ``obs_task`` and
    ``goal_task)``."""

    def tokenize(self, obs_task: TensorTree | None, goal_task: TensorTree | None) -> TensorTree:
        """Builds the raw (pre-embedder) token tensor for an observation window and/or a goal.

        Args:
            obs_task: Leaves of shape ``[B, T, *]`` (task-only, proprio already popped), or
                ``None``.
            goal_task: Leaves of shape ``[B, *]`` (single-frame, task-only), or ``None``.
                Exactly one of ``obs_task``/``goal_task`` may be ``None`` iff
                ``supports_single_side``; both being ``None`` is always invalid.

        Returns:
            ``[B, T, output_dim]`` if ``tokens_per_step == 1``,
            else ``[B, T, K, output_dim]``.

            A single-side call without a ``T`` axis drops the leading
            ``T`` too (``[B, output_dim]`` / ``[B, K, output_dim]``).

            A tokenizer may also emit a subtree matching :attr:`token_spec`, when tokens alone
            do not carry everything the embedder needs (e.g. ``GraphTokenizer`` adds per-node validity
            and pairwise edge features).
        """
        ...


@runtime_checkable
class PoolingProtocol(Protocol):
    """Protocol for pooling a token sequence into a fixed-size representation."""

    mode: Literal["all", "objects", "time"]

    @property
    def pools_time(self) -> bool: ...

    @property
    def pools_objects(self) -> bool: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class NormalizerProtocol(Protocol):
    """Protocol for the normalizers configurable on a diffusion agent (e.g. ``ZScoreNormalizer``,
    ``MinMaxNormalizer``)."""

    @property
    def is_fit(self) -> torch.Tensor: ...

    def fit(self, data: TensorTree) -> None: ...

    def fit_incremental(self, data_iterator: typing.Iterable[TensorTree]) -> None: ...

    @typing.overload
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def normalize(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...
    def normalize(self, x: TensorTree) -> TensorTree: ...

    @typing.overload
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor: ...
    @typing.overload
    def unnormalize(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...
    def unnormalize(self, x: TensorTree) -> TensorTree: ...


@runtime_checkable
class EnvProtocol(Protocol):
    """Protocol representing a standard environment (e.g., gym.Env)."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...


@runtime_checkable
class GoalConditionedEnvProtocol(EnvProtocol, Protocol):
    """Protocol for goal-conditioned environments that can generate heuristic goals."""

    def generate_heuristic_goal(self) -> torch.Tensor | dict[str, Any]:
        """Generate a heuristic goal state for the environment."""
        ...
