import logging
import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence
from unittest.mock import Mock, patch

import numpy as np
import pytest
import wandb
from farconf import update_fns_to_cli
from stable_baselines3.common.type_aliases import check_cast

import learned_planner.configs.train_drc as train_drc
from learned_planner.__main__ import main
from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.train_drc import DeviceLiteral
from learned_planner.convlstm import ConvConfig, ConvLSTMOptions
from learned_planner.environments import BoxobanConfig, EnvpoolSokobanVecEnvConfig
from learned_planner.optimizers import AdamOptimizerConfig
from learned_planner.policies import NetArchConfig
from learned_planner.train import RecurrentPPOConfig, TrainConfig

N_ENVS = 4


def _update_recurrent_ppo_sokoban_to_small_values(
    cfg: WandbCommandConfig, device: DeviceLiteral, training_mount: Path, boxoban_cache: Path, *, assert_is_envpool: bool
) -> WandbCommandConfig:
    cfg.base_save_prefix = training_mount
    assert isinstance(cfg.cmd, TrainConfig)
    cfg.cmd.device = device
    cfg.cmd.test_that_eval_split_is_validation = True  # Default configs should check the validation set

    features_extractor = check_cast(ConvLSTMOptions, cfg.cmd.policy.features_extractor)
    features_extractor.embed = [check_cast(ConvConfig, features_extractor.embed[0])]
    features_extractor.embed[0].features = 1
    features_extractor.n_recurrent = 1
    features_extractor.recurrent.conv.features = 1

    cfg.cmd.policy.net_arch = NetArchConfig([1], [1])

    cfg.cmd.n_steps = 20
    cfg.cmd.env.n_envs = 2
    cfg.cmd.total_timesteps = 2 * 20

    cfg.cmd.n_eval_steps = 10
    cfg.cmd.n_eval_episodes = cfg.cmd.env.n_envs
    cfg.cmd.env.max_episode_steps = 4

    # Make Boxoban load envs from the cache in `boxoban_cache`
    assert isinstance(cfg.cmd.env, (EnvpoolSokobanVecEnvConfig, BoxobanConfig))
    cfg.cmd.env.cache_path = boxoban_cache
    cfg.cmd.env.n_envs = 2
    cfg.cmd.env.n_envs_to_render = 1
    cfg.cmd.env.min_episode_steps = 4

    assert isinstance(cfg.cmd.eval_env, (EnvpoolSokobanVecEnvConfig, BoxobanConfig))
    cfg.cmd.eval_env.cache_path = boxoban_cache
    cfg.cmd.eval_env.n_envs = 2
    cfg.cmd.eval_env.n_envs_to_render = 2

    if assert_is_envpool:
        assert isinstance(cfg.cmd.env, EnvpoolSokobanVecEnvConfig)

    check_cast(TrainConfig, cfg.cmd).checkpoint_freq = 10

    alg = check_cast(RecurrentPPOConfig, cfg.cmd.alg)
    alg.batch_envs = 2
    alg.batch_time = 2
    alg.n_epochs = 1
    check_cast(AdamOptimizerConfig, alg.optimizer).fused = device == "cuda"
    return cfg


@pytest.mark.parametrize(
    "train_fn_orig, assert_is_envpool",
    [
        (train_drc.train_cluster, True),
        (train_drc.train_cluster_103, True),
        (train_drc.train_cluster_114, True),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@patch("wandb.log")
def test_train_drc(
    _wandb_log: Mock,
    tmpdir: Path,
    BOXOBAN_CACHE: Path,
    train_fn_orig: Callable[[], WandbCommandConfig],
    assert_is_envpool: bool,
    device: DeviceLiteral,
):
    cli, _ = update_fns_to_cli(
        train_fn_orig,
        partial(
            _update_recurrent_ppo_sokoban_to_small_values,
            device=device,
            training_mount=tmpdir,
            boxoban_cache=BOXOBAN_CACHE,
            assert_is_envpool=assert_is_envpool,
        ),
    )
    main(cli, run_dir=tmpdir)
