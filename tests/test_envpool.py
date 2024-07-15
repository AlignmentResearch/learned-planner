from pathlib import Path

import pytest
import torch as th
from PIL import Image
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from learned_planner.environments import BoxobanConfig, EnvpoolSokobanVecEnvConfig


def test_envpool_vecenv(BOXOBAN_CACHE: Path):
    NUM_ENVS = 10
    NUM_ENVS_TO_RENDER = 4
    cfg = EnvpoolSokobanVecEnvConfig(
        cache_path=BOXOBAN_CACHE,
        max_episode_steps=10,
        min_episode_steps=3,
        n_envs=NUM_ENVS,
        n_envs_to_render=NUM_ENVS_TO_RENDER,
    )

    assert int(NUM_ENVS_TO_RENDER**0.5) ** 2 == NUM_ENVS_TO_RENDER, "not a perfect square"

    venv = cfg.make(device=th.device("cpu"))

    obs = venv.reset()
    assert isinstance(obs, th.Tensor)
    assert obs.shape == (NUM_ENVS, 3, venv.cfg.dim_room, venv.cfg.dim_room)

    img = venv.render()
    assert img is not None
    # Each environment has 4x4 tiles
    SIDE_N_ENVS = int(NUM_ENVS_TO_RENDER**0.5)
    total_size = SIDE_N_ENVS * venv.cfg.dim_room * venv.cfg.px_scale
    assert img.shape == (total_size, total_size, 3)


@pytest.mark.parametrize("min_episode_steps", [0, 10, 20])
def test_random_episode_steps(min_episode_steps: int, BOXOBAN_CACHE: Path):
    max_episode_steps = 20
    NUM_ENVS = 10
    cfg = EnvpoolSokobanVecEnvConfig(
        cache_path=BOXOBAN_CACHE,
        min_episode_steps=min_episode_steps,
        max_episode_steps=20,
        seed=42,
        n_envs=NUM_ENVS,
        n_envs_to_render=NUM_ENVS,
    )

    venv = cfg.make(device=th.device("cpu"))

    for step in range(0, min_episode_steps - 1):
        obs, reward, done, info = venv.step(th.zeros(NUM_ENVS, dtype=th.int32))
        assert not th.any(done), "At least one env has reset before its min_episode_steps"

    dones = []
    for step in range(min_episode_steps - 1, max_episode_steps - 1):
        obs, reward, done, info = venv.step(th.zeros(NUM_ENVS, dtype=th.int32))
        dones.append(done)

    done_per_env = th.any(th.stack(dones), dim=0) if dones else th.zeros(NUM_ENVS, dtype=th.bool)

    assert len(dones) == 0 or th.any(done_per_env), "None of the envs has reset *before* their max_episode_steps"

    obs, reward, done, info = venv.step(th.zeros(NUM_ENVS, dtype=th.int32))
    done_per_env = done | done_per_env

    assert th.all(done_per_env), "One of the envs didn't reset at exactly max_episode_steps"


@pytest.fixture
def SINGLE_LEVEL_BOXOBAN_CACHE(tmp_path: Path):
    levels_path = tmp_path / "boxoban-levels-master" / "unfiltered" / "train"
    levels_path.mkdir(parents=True)
    with (levels_path / "000.txt").open("w") as f:
        f.write(
            """
##########
# ########
#  #######
#  #######
#  ## ####
#.#@$  . #
# # $   .#
#$#  #$  #
#   .#   #
##########
    """
        )
    return tmp_path


@pytest.mark.parametrize("reward_finished", [20.0, 40.0])
@pytest.mark.parametrize("reward_step", [-0.1, -0.01])
@pytest.mark.parametrize("reward_box", [1.0, 2.0])
def test_reward_fn(
    tmp_path: Path, SINGLE_LEVEL_BOXOBAN_CACHE: Path, reward_finished: float, reward_step: float, reward_box: float
):
    """
    Load the map above and check that the reward obtained for each step is correct
    """
    venv = EnvpoolSokobanVecEnvConfig(
        cache_path=SINGLE_LEVEL_BOXOBAN_CACHE,
        reward_finished=reward_finished,
        reward_step=reward_step,
        reward_box=reward_box,
        seed=1234,
        min_episode_steps=70,
        max_episode_steps=70,
        n_envs=1,
        n_envs_to_render=1,
    ).make(device=th.device("cpu"))

    gym_cfg = BoxobanConfig(
        cache_path=SINGLE_LEVEL_BOXOBAN_CACHE,
        reward_finished=reward_finished,
        reward_step=reward_step,
        reward_box=reward_box,
        tinyworld_obs=True,
        min_episode_steps=70,
        max_episode_steps=70,
        n_envs=1,
        n_envs_to_render=0,
    )
    gym_sokoban_env = VecTransposeImage(DummyVecEnv([gym_cfg.make] * gym_cfg.n_envs))

    elapsed_step = 0

    def reset_envs():
        nonlocal elapsed_step
        elapsed_step = 0

        obs_envpool = venv.reset()
        assert isinstance(obs_envpool, th.Tensor)
        obs_gym = gym_sokoban_env.reset()
        assert isinstance(obs_gym, th.Tensor)
        assert th.equal(obs_envpool, obs_gym)

    # Actions: up down left right (with push if available)
    U = th.as_tensor(0).unsqueeze(0)
    D = th.as_tensor(1).unsqueeze(0)
    L = th.as_tensor(2).unsqueeze(0)
    R = th.as_tensor(3).unsqueeze(0)

    def check_env_behavior(action: th.Tensor, envpool_reward: float, done: bool = False):
        nonlocal elapsed_step

        obs_envpool, r, venv_done, venv_info = venv.step(action)
        elapsed_step += 1
        assert elapsed_step == venv_info[0]["real_info"]["real_info"]["elapsed_step"].item()

        assert isinstance(obs_envpool, th.Tensor)
        obs_gym, *_ = gym_sokoban_env.step(action)
        assert isinstance(obs_gym, th.Tensor)

        # Save the current state in the temporary dir, to help diagnose what went wrong.
        im = Image.fromarray(obs_gym.squeeze(0).moveaxis(0, -1).numpy())
        im.save(tmp_path / f"gym_{elapsed_step:02d}.png")
        im = Image.fromarray(obs_envpool.squeeze(0).moveaxis(0, -1).numpy())
        im.save(tmp_path / f"envpool_{elapsed_step:02d}.png")

        # The reward gets converted to float32 just before being returned, so we have to convert envpool_reward to
        # float32 too.
        assert th.equal(r, th.as_tensor(envpool_reward, dtype=th.float32).unsqueeze(0))
        assert venv_done.item() == done

        assert th.equal(obs_envpool, obs_gym)

    # Check rewards of an unsolved level
    reset_envs()

    for a in [D, R, R, R]:
        check_env_behavior(a, reward_step)

    # Push the box onto the target
    check_env_behavior(R, reward_box + reward_step)

    # Now remove the box
    for a in [U, R]:
        check_env_behavior(a, reward_step)

    check_env_behavior(D, -reward_box + reward_step)

    # Now put the box on target again
    for a in [L, D, D, R]:
        check_env_behavior(a, reward_step)

    check_env_behavior(U, reward_box + reward_step)

    # Run down the timer and check that the reward is correct every time
    while elapsed_step < venv.env.spec.config.max_episode_steps - 1:
        check_env_behavior(R, reward_step, done=False)
    check_env_behavior(R, reward_step, done=True)

    ## Now solve the level and check that the rewards are all correct
    reset_envs()

    for a in [R, D]:
        check_env_behavior(a, reward_step)
    check_env_behavior(D, reward_step + reward_box)
    for a in [U, R, R, R, D, D, L, U, R, U, U, L, L, D, R]:
        check_env_behavior(a, reward_step)
    check_env_behavior(R, reward_step + reward_box)
    for a in [L, L, L, L, U, R, R]:
        check_env_behavior(a, reward_step)
    check_env_behavior(R, reward_step + reward_box)
    for a in [L, L, L, D, D, D, L, L, U]:
        check_env_behavior(a, reward_step)
    check_env_behavior(U, reward_step + reward_box + reward_finished, done=True)


def test_reset_on_max_steps(tmp_path: Path, SINGLE_LEVEL_BOXOBAN_CACHE: Path):
    venv = EnvpoolSokobanVecEnvConfig(
        cache_path=SINGLE_LEVEL_BOXOBAN_CACHE,
        reward_finished=10,
        reward_step=-0.1,
        reward_box=1,
        seed=1234,
        min_episode_steps=2,
        max_episode_steps=2,
        n_envs=1,
        n_envs_to_render=0,
    ).make(device=th.device("cpu"))

    gym_cfg = BoxobanConfig(
        cache_path=SINGLE_LEVEL_BOXOBAN_CACHE,
        tinyworld_obs=True,
        min_episode_steps=2,
        max_episode_steps=2,
        n_envs=1,
        n_envs_to_render=0,
    )
    gym_sokoban_env = VecTransposeImage(DummyVecEnv([gym_cfg.make]))

    R = th.as_tensor(3).unsqueeze(0)

    init_obs = venv.reset()
    obs_1, r1, d1, i1 = venv.step(R)
    obs_2, r2, d2, i2 = venv.step(R)
    obs_3, r3, d3, i3 = venv.step(R)
    assert isinstance(init_obs, th.Tensor)
    assert isinstance(obs_1, th.Tensor)
    assert isinstance(obs_2, th.Tensor)
    assert isinstance(obs_3, th.Tensor)

    assert not (th.equal(init_obs, obs_1) or d1), "First step should not reset the environment"
    assert th.equal(init_obs, obs_2) and d2, "Environment should reset on max_episode_steps"
    assert not (th.equal(init_obs, obs_3) or d3), "Environment should not reset after being done"

    init_obs = gym_sokoban_env.reset()
    obs_1, r1, d1, i1 = gym_sokoban_env.step(R)
    obs_2, r2, d2, i2 = gym_sokoban_env.step(R)
    obs_3, r3, d3, i3 = gym_sokoban_env.step(R)
    assert isinstance(init_obs, th.Tensor)
    assert isinstance(obs_1, th.Tensor)
    assert isinstance(obs_2, th.Tensor)
    assert isinstance(obs_3, th.Tensor)
    assert not (th.equal(init_obs, obs_1) or d1), "First step should not reset the environment"
    assert th.equal(init_obs, obs_2) and d2, "Environment should reset on max_episode_steps"
    assert not (th.equal(init_obs, obs_3) or d3), "Environment should not reset after being done"
