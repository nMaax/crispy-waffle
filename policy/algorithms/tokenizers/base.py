from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch

from policy.utils.typing_utils import TensorTree


class BaseTokenizer(ABC):
    """Base class for all task state tokenizers."""

    supports_single_side: ClassVar[bool] = True
    output_dim: int
    tokens_per_step: int | None

    def __init__(self, relative_goal: bool = True):
        self.relative_goal = relative_goal

    @property
    @abstractmethod
    def categorical_mask(self) -> torch.Tensor:
        """``[output_dim]`` bool mask, False on channels an affine rescale would destroy."""
        ...

    def tokenize(
        self,
        obs_task: TensorTree | None,
        goal_task: TensorTree | None = None,
    ) -> torch.Tensor:
        if obs_task is not None and goal_task is not None:
            return self._tokenize_relative(obs_task, goal_task)
        state = obs_task if obs_task is not None else goal_task
        if state is None:
            raise ValueError("tokenize() requires at least one of obs_task or goal_task.")
        return self._tokenize_absolute(state)

    @abstractmethod
    def _tokenize_relative(self, obs_task: Any, goal_task: Any) -> torch.Tensor: ...

    @abstractmethod
    def _tokenize_absolute(self, task: Any) -> torch.Tensor: ...
