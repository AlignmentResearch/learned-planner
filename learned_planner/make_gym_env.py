import gym_sokoban  # noqa: F401 # pyright: ignore
import gymnasium as gym


def make_env(name, **kwargs):
    out = gym.make(name, **kwargs)
    return out
