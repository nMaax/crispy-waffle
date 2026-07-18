from collections.abc import Mapping
from typing import Any, overload

import numpy as np
import torch

import policy.environments  # noqa: F401
from policy.utils import slice_by_schema
from policy.utils.typing_utils import RawTree, TensorTree


class ManiSkillStateDeFlattener:
    """Reconstructs native ManiSkill state_dict structure from flat observation states.

    Reads the environment class's declarative STATE_SCHEMA attribute.
    """

    def __init__(self, env_id: str):
        self.env_id = env_id
        self.schema = self._get_env_schema(env_id)

    def _get_env_schema(self, env_id: str) -> dict[str, Any]:
        from mani_skill.utils.registration import REGISTERED_ENVS

        schema = None

        if env_id in REGISTERED_ENVS:
            env_cls = REGISTERED_ENVS[env_id].cls
            if hasattr(env_cls, "STATE_SCHEMA"):
                schema = getattr(env_cls, "STATE_SCHEMA")

        if schema is None:
            raise ValueError(
                f"No STATE_SCHEMA found for environment '{env_id}'. Ensure you defined a STATE_SCHEMA in your custom environment class."
            )

        return schema

    @overload
    def __call__(self, obs: torch.Tensor) -> dict[str, TensorTree]: ...

    @overload
    def __call__(self, obs: np.ndarray) -> dict[str, RawTree]: ...

    @overload
    def __call__(self, obs: Mapping[str, TensorTree]) -> Mapping[str, TensorTree]: ...

    @overload
    def __call__(self, obs: Mapping[str, RawTree]) -> Mapping[str, RawTree]: ...

    def __call__(
        self, obs: np.ndarray | torch.Tensor | Mapping[str, Any]
    ) -> dict[str, Any] | Mapping[str, Any]:
        if isinstance(obs, Mapping):
            return obs
        else:
            return slice_by_schema(obs, self.schema)
