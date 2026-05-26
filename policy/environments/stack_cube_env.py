import numpy as np
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv as ManiSkillStackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("StackCube-v1", max_episode_steps=50, override=True)
class StackCubeEnv(ManiSkillStackCubeEnv):
    def get_goal_state(self) -> np.ndarray:
        """Returns the raw observation vector for the goal state (Cube A on top of Cube B)."""
        obs = self.get_obs()

        if isinstance(obs, dict):
            # In case ManiSkill returns a dict, we need to handle it or ensure it's flattened
            # But with obs_mode="state", it's usually flattened by the time it reaches here
            # if it's been wrapped. However, get_obs() on the base class might return the dict.
            # For now, let's assume it's flat as per our experience with the indices.
            pass

        # Heuristic from RolloutEvaluationCallback:
        # Indices: Proprio (0:18), TCP (18:25), Cube A (25:32), Cube B (32:39)
        # Goal: A on top of B

        # Cube B pose (p, q) is at 32:39
        cube_B_pose = obs[32:39].copy()

        # Target for A: B_pos + [0, 0, 0.04], B_quat
        target_pos_A = cube_B_pose[:3].copy()
        target_pos_A[2] += 0.04 # Standard 4cm offset for stacking

        # Set Cube A pose
        obs[25:28] = target_pos_A
        obs[28:32] = cube_B_pose[3:7]

        # TCP goal: same as Cube A (just ungrasped)
        obs[18:25] = obs[25:32]

        return obs
