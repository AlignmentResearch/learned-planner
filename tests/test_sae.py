import sys
from functools import partial
from pathlib import Path
from typing import Callable, Literal
from unittest.mock import Mock, patch

import pytest
import wandb  # noqa: F401  # pyright: ignore
from farconf import update_fns_to_cli
from stable_baselines3.common.type_aliases import check_cast

from learned_planner import LP_DIR
from learned_planner.__main__ import main
from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.train_drc import DeviceLiteral
from learned_planner.configs.train_sae import train_sae as train_sae_cfg_fn
from learned_planner.interp import train_sae
from learned_planner.interp.collect_dataset import DatasetStore

# when running the test using pytest, the DatasetStore class is not available in the main module
# and pickle.load searches for it in the main module. So we need to set it manually
setattr(sys.modules["__main__"], "DatasetStore", DatasetStore)


@pytest.mark.parametrize("train_fn", [train_sae_cfg_fn])
@pytest.mark.parametrize("device", ["cpu"])
@patch("wandb.log")
def test_train_sae(
    _wandb_log: Mock,
    tmpdir: Path,
    train_fn: Callable[[], WandbCommandConfig],
    device: Literal["cpu", "cuda"],
):
    wandb.init(mode="disabled")

    def _update_train_fn(cfg: WandbCommandConfig, device: DeviceLiteral, training_mount: Path) -> WandbCommandConfig:
        cfg.base_save_prefix = training_mount
        cfg.cmd = check_cast(train_sae.TrainSAEConfig, cfg.cmd)
        cfg.cmd.training_tokens = 10
        cfg.cmd.train_batch_size_tokens = 5
        cfg.cmd.epochs = 1
        cfg.cmd.cached_activations_path = str(LP_DIR / "tests/probes_dataset/")
        cfg.cmd.num_envs = 1
        cfg.cmd.wandb_log_frequency = 1
        cfg.cmd.eval_every_n_wandb_logs = 2
        cfg.cmd.expansion_factor = 2
        cfg.cmd.envpool = False
        cfg.cmd.device = device

        return cfg

    cli, _ = update_fns_to_cli(
        train_fn,
        partial(
            _update_train_fn,
            device=device,
            training_mount=tmpdir,
        ),
    )
    main(cli, run_dir=tmpdir)  # type: ignore
