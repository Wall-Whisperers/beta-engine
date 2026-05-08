"""MuJoCo Gymnasium environments for MoonBoard RL training (Week 1+)."""

import gymnasium as gym

from src.envs.moonboard_env import MoonBoardEnv


def make_env(route, humanoid_xml_path: str) -> MoonBoardEnv:
    """Return a configured MoonBoardEnv for the given route.

    Args:
        route: Route object from any MoonBoard parser.
        humanoid_xml_path: Absolute path to humanoid.xml on disk.

    Returns:
        MoonBoardEnv instance with default sim_substeps and max_episode_steps.
    """
    return MoonBoardEnv(route=route, humanoid_xml_path=humanoid_xml_path)


gym.register(
    id="MoonBoard-v0",
    entry_point="src.envs.moonboard_env:MoonBoardEnv",
    max_episode_steps=2000,
)
