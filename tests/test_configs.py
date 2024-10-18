#!/usr/bin/env python3
import importlib

from farconf import config_diff, parse_cli, parse_cli_into_dict, to_dict
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.train_drc import train_local_114
from learned_planner.train import TrainConfig


def test_command_114():
    runs = importlib.import_module("experiments.114_adam_params").get_runs()
    all_clis = sum((r.commands for r in runs), start=[])

    all_configs = [parse_cli(cli, WandbCommandConfig) for cli in all_clis]
    target_config = train_cluster_114()
    cmd = check_cast(TrainConfig, target_config.cmd)
    all_configs[0].cmd = check_cast(TrainConfig, all_configs[0].cmd)
    cmd.checkpoint_freq = 300_000
    cmd.total_timesteps = int(3e10)
    cmd.eval_env = all_configs[0].cmd.eval_env

    # The seeds are repeated for each configuration so we can pick an arbitrary pair. We pick the one from the first
    # command
    cmd.seed = all_configs[0].cmd.seed
    cmd.env.seed = all_configs[0].cmd.env.seed  # type: ignore

    for cli in all_clis:
        diff = config_diff(parse_cli_into_dict(cli), to_dict(target_config, WandbCommandConfig))
        print(diff)

    assert any(cfg == target_config for cfg in all_configs)
