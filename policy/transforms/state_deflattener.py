from collections.abc import Mapping

import numpy as np
import torch

import policy.environments  # noqa: F401
from policy.utils import slice_by_schema


class ManiSkillStateDeFlattener:
    """Reconstructs native ManiSkill state_dict structure from flat observation states.

    Reads the environment class's declarative STATE_SCHEMA attribute.
    """

    def __init__(self, env_id: str):
        self.env_id = env_id
        self.schema = self._get_env_schema(env_id)

    def _get_env_schema(self, env_id: str) -> dict:
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

    def __call__(self, obs: np.ndarray | torch.Tensor | dict) -> dict:
        if isinstance(obs, Mapping):
            return obs
        else:
            return slice_by_schema(obs, self.schema)
