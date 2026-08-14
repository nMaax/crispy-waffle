import gymnasium as gym
import pytest

import policy.environments  # noqa: F401


@pytest.mark.parametrize(
    "env_id",
    [
        "StackCubeClutter-v1",
        "StackCubeClutterLockedRotation-v1",
        "StackCubeClutterRandomPick-v1",
        "StackCubeClutterRandomPickLockedRotation-v1",
    ],
)
def test_clutter_environment_initialization_and_goal(env_id: str):
    env = gym.make(env_id, obs_mode="state_dict", sim_backend="physx_cpu")
    obs, info = env.reset(seed=42)

    assert isinstance(obs, dict)
    assert "agent" in obs
    assert "extra" in obs
    assert "tcp_pose" in obs["extra"]

    assert hasattr(env.unwrapped, "generate_heuristic_goal")
    goal = env.unwrapped.generate_heuristic_goal()

    assert isinstance(goal, dict)
    assert "agent" in goal
    assert "extra" in goal
    assert "tcp_pose" in goal["extra"]

    env.close()
