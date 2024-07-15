import abc
import dataclasses
import json
from pathlib import Path
from typing import Literal

from farconf import to_dict
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.configs.misc import DEFAULT_TRAINING
from learned_planner.train import ABCCommandConfig


@dataclasses.dataclass
class WandbCommandConfig(abc.ABC):
    cmd: ABCCommandConfig
    base_save_prefix: Path = DEFAULT_TRAINING  # Where to store the experiment's results
    log_level: Literal["ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"  # Log level for the experiments

    def to_cli(self) -> list[str]:
        self_dict = check_cast(dict, to_dict(self))
        cli = [f"--set-json={k}={json.dumps(v)}" for k, v in self_dict.items()]
        return cli
