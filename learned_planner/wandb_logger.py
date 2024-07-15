import html
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import stable_baselines3.common.logger
import torch as th
import wandb
from stable_baselines3.common.logger import (
    Figure,
    FormatUnsupportedError,
    HParam,
    Image,
    KVWriter,
    Logger,
    Video,
)


class WandBOutputFormat(KVWriter):
    """
    Dumps key/value pairs into Weights & Biases' format.

    :param folder: the Path prefix to write to.
    """

    def __init__(self, folder: str):
        super().__init__()
        self.prefix = folder
        self.current_logs = {"step": 0}

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        if step != self.current_logs["step"]:
            wandb.log(self.current_logs)
            self.current_logs = log_dict = {"step": step}
        else:
            log_dict = self.current_logs

        def _add_to_log_dict(key: str, value: Any) -> None:
            if key in log_dict:
                raise ValueError(f"Duplicate key {key}")
            log_dict[key] = value

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "wandb" in excluded:
                continue

            key = self.prefix + "/" + key

            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    _add_to_log_dict(key, wandb.Html(html.escape(value)))
                else:
                    _add_to_log_dict(key, value)

            if isinstance(value, th.Tensor):
                _add_to_log_dict(key, wandb.Histogram(value.cpu().numpy()))

            if isinstance(value, Video):
                _add_to_log_dict(key, wandb.Video(value.frames.cpu().numpy(), fps=int(value.fps)))

            if isinstance(value, Figure):
                _add_to_log_dict(key, wandb.Image(value.figure))

            if isinstance(value, Image):
                if value.dataformats != "HWC":
                    raise FormatUnsupportedError(["wandb"], f"{value.dataformats=}")
                _add_to_log_dict(key, wandb.Image(value.image))

            if isinstance(value, HParam):
                raise FormatUnsupportedError(["wandb"], str(value))

    def close(self) -> None:
        if len(self.current_logs) > 1:
            wandb.log(self.current_logs)
            self.current_logs = {"step": 0}


def configure_logger(run_dir: Path, prefix: str, verbose: int) -> Logger:
    format_strings = ["stdout"] if verbose > 0 else []
    logger = stable_baselines3.common.logger.configure(folder=str(run_dir / prefix), format_strings=format_strings)
    logger.output_formats.append(WandBOutputFormat(prefix))
    return logger
