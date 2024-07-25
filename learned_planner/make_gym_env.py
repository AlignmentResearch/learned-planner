import gym_sokoban  # noqa
import gymnasium as gym


def make_env(name, **kwargs):
    out = gym.make(name, **kwargs)
    return out
