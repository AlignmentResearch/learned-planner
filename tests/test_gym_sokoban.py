from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.environments import BoxobanConfig, FixedBoxobanConfig


def test_boxoban_rendering(BOXOBAN_CACHE: Path):
    env = BoxobanConfig(cache_path=BOXOBAN_CACHE).make()
    obs, _ = env.reset()
    assert obs.shape == (80, 80, 3)

    obs2, *_ = env.step(0)
    assert obs2.shape == (80, 80, 3)

    img = check_cast(np.ndarray, env.render())
    assert img.shape == (80, 80, 3)

    tiny_env = BoxobanConfig(cache_path=BOXOBAN_CACHE, tinyworld_obs=True).make()
    tiny_obs, _ = tiny_env.reset()
    assert tiny_obs.shape == (10, 10, 3)

    tiny_obs2, *_ = tiny_env.step(0)
    assert tiny_obs2.shape == (10, 10, 3)

    tiny_img = check_cast(np.ndarray, tiny_env.render())
    assert tiny_img.shape == (80, 80, 3)


@pytest.mark.parametrize("reward_finished", [20.0, 40.0])
def test_reward_fn(BOXOBAN_CACHE: Path, reward_finished: float):
    """
    Load this map and check that the reward obtained for each step is correct

    ##########
    # ########
    #     ####
    # ..$  ###
    #. $. ####
    #  #######
    #   ######
    #$$#######
    #@ #######
    ##########
    """
    env = BoxobanConfig(cache_path=BOXOBAN_CACHE, reward_finished=reward_finished).make()
    env.reset(seed=0)

    # Actions: up down left right (with push if available)
    U = 0
    D = 1
    L = 2
    R = 3

    for a in [U, U]:
        _, r, _, _, _ = env.step(a)
        assert r == -0.1

    # Push the box onto the dot
    _, r, _, _, _ = env.step(U)
    assert r == 0.9

    # Now remove the box
    for a in [R, U, U, L]:
        _, r, _, _, _ = env.step(a)
        assert r == -0.1

    _, r, _, _, _ = env.step(D)
    # NOTE: perhaps this penalty will be bad, because we don't care if the agent actually removes boxes in the process
    # of solving the whole thing, and removing boxes is often necessary for harder levels. Something to look at later.
    assert r == -1.1, "The agent should be penalized for removing boxes"

    # Now put the box on target again
    for a in [R, D, D, L]:
        _, r, _, _, _ = env.step(a)
        assert r == -0.1

    _, r, _, _, _ = env.step(U)
    assert r == 0.9

    assert env.unwrapped.reward_finished == reward_finished  # type: ignore


def test_terminate_on_first_box(BOXOBAN_CACHE: Path):
    """
    same map as test_reward_fn
    """
    env = BoxobanConfig(cache_path=BOXOBAN_CACHE, terminate_on_first_box=True).make()
    env.reset(seed=0)

    # Actions: up down left right (with push if available)
    U = 0

    for a in [U, U]:
        _, r, terminated, truncated, _ = env.step(a)
        assert r == -0.1
        assert not (terminated or truncated)

    # Push the box onto the dot
    _, r, terminated, truncated, _ = env.step(U)
    assert r == 0.9
    assert terminated or truncated


def test_reset_randomness(BOXOBAN_CACHE: Path):
    random_env = BoxobanConfig(cache_path=BOXOBAN_CACHE, tinyworld_obs=True, seed=1234).make()
    fixed_env = FixedBoxobanConfig(cache_path=BOXOBAN_CACHE, tinyworld_obs=True, seed=4231).make()

    assert not np.array_equal(random_env.reset(seed=1)[0], random_env.reset(seed=2)[0])
    fixed_obs, _ = fixed_env.reset(seed=4)
    assert np.array_equal(fixed_obs, fixed_env.reset(seed=5)[0])

    new_obs, *_ = fixed_env.step(2)
    assert not np.array_equal(new_obs, fixed_obs)
    assert np.array_equal(fixed_obs, fixed_env.reset()[0])
