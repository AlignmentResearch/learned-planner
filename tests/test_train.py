from pathlib import Path

import torch as th
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.environments import EnvpoolSokobanVecEnvConfig
from learned_planner.policies import MlpPolicyConfig
from learned_planner.train import (
    RecurrentPPOConfig,
    TrainConfig,
    init_learning_staggered_environment,
)


def test_init_learning_staggered_environment(BOXOBAN_CACHE: Path):
    device = th.device("cpu")
    N_ENVS = 2
    N_STEPS = 4

    args = TrainConfig(
        policy=MlpPolicyConfig(),
        env=EnvpoolSokobanVecEnvConfig(cache_path=BOXOBAN_CACHE, n_envs=N_ENVS, n_envs_to_render=N_ENVS),
        eval_env=EnvpoolSokobanVecEnvConfig(
            cache_path=BOXOBAN_CACHE,
            split="valid",
            load_sequentially=True,
            n_envs=N_ENVS,
            n_envs_to_render=N_ENVS,
        ),
        alg=RecurrentPPOConfig(batch_time=N_STEPS, batch_envs=N_ENVS),
    )
    assert isinstance(args.eval_env, EnvpoolSokobanVecEnvConfig), "eval_env should be the same as env"
    assert args.eval_env.load_sequentially, "eval_env should load eval set sequentially during training"
    assert args.eval_env.split == "valid", "eval_env should use the validation set during training"

    # TODO: make this check_cast unnecessary (see #48)
    env = check_cast(EnvpoolSokobanVecEnvConfig, args.env).make(device=device)
    policy, policy_kwargs = args.policy.policy_and_kwargs(env)
    model = args.alg.make(
        policy=check_cast(str, policy), env=env, n_steps=4, seed=1234, device=device, policy_kwargs=policy_kwargs
    )

    init_obs = check_cast(th.Tensor, env.reset())
    assert model._last_obs is None, "_last_obs should start out empty"

    init_learning_staggered_environment(args, model, max_episode_step_multiple=0)
    last_obs = check_cast(th.Tensor, model._last_obs)
    assert not th.equal(init_obs, last_obs), "init_learning_staggered_environment should init _last_obs"

    # Save the obs that we'll use to check whether environments reset later on
    prev_last_obs = last_obs.clone().detach()

    def fake_reset(*args, **kwargs):
        raise RuntimeError(f"env.reset() should not be called, but it was called with {args=}, {kwargs=}")

    # Raise an error if the environment is reset again
    env.reset = fake_reset

    init_learning_staggered_environment(args, model, max_episode_step_multiple=0)
    last_obs = check_cast(th.Tensor, model._last_obs)
    assert th.equal(
        prev_last_obs, last_obs
    ), "init_learning_staggered_environment called again with 0 steps should not reset the environment"

    init_learning_staggered_environment(args, model, max_episode_step_multiple=2)
    last_obs = check_cast(th.Tensor, model._last_obs)
    assert not th.equal(
        prev_last_obs, last_obs
    ), "init_learning_staggered_environment with nonzero steps should advance the environment"
