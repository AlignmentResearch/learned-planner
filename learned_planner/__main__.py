import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from farconf import parse_cli, to_dict
from names_generator import generate_name
from stable_baselines3.common.type_aliases import check_cast

# Make sure the command configs are registered
import learned_planner.cmd  # type: ignore
import learned_planner.evaluate  # noqa: F401
from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.interp.collect_dataset import DatasetStore  # noqa: F401


def setup_run(cfg: WandbCommandConfig) -> Path:
    import wandb
    import wandb.util

    wandb_kwargs: dict[str, Any]
    try:
        wandb_kwargs = dict(
            entity=os.environ["WANDB_ENTITY"],
            name=os.environ.get("WANDB_JOB_NAME", generate_name(style="hyphen")),
            project=os.environ["WANDB_PROJECT"],
            group=os.environ["WANDB_RUN_GROUP"],
            mode=os.environ.get("WANDB_MODE", "online"),  # Default to online here
        )
    except KeyError:
        # If any of the essential WANDB environment variables are missing,
        # simply don't upload this run.
        # It's fine to do this without giving any indication because Wandb already prints that the run is offline.

        wandb_kwargs = dict(mode=os.environ.get("WANDB_MODE", "offline"), group="default")

    command = cfg.cmd.__class__.__name__
    run_dir = cfg.base_save_prefix / command / wandb_kwargs["group"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # We don't want to use tensorboard so comment out this line
    # wandb.tensorboard.patch(root_logdir=str(run_dir))
    wandb.init(
        **wandb_kwargs,
        config=check_cast(dict, to_dict(cfg)),
        save_code=True,  # Make sure git diff is saved
        job_type=command,
        dir=run_dir,
        monitor_gym=False,  # Must manually log videos to wandb
        sync_tensorboard=False,  # Manually log tensorboard
        settings=wandb.Settings(code_dir=str(Path(__file__).parent.parent)),
    )
    assert wandb.run is not None

    # Avoid syncing saved files (e.g. checkpoints) to weights and biases. Wandb syncs the `run.dir` -- which is equal to
    # `run_dir / "wandb" / <timestamp-runid> / "files"`.
    #
    # We make a new directory with a different name ("local-files") and save our files there, so they don't get synced
    # to wandb.

    wandb_run_dir = Path(wandb.run.dir)
    if wandb_kwargs["mode"] == "disabled":
        files_dir = wandb_run_dir / "local-files"
    else:
        assert wandb_run_dir.name == "files"
        files_dir = wandb_run_dir.parent / "local-files"
    files_dir.mkdir(exist_ok=True)
    return files_dir


def main(args: Optional[Sequence[str]] = None, run_dir: Optional[Path] = None) -> None:
    if args is None:
        args = sys.argv[1:]

    cfg = parse_cli(list(args), WandbCommandConfig)
    logging.basicConfig(level=cfg.log_level)

    if run_dir is None:
        run_dir = setup_run(cfg)

    return cfg.cmd.run(run_dir)


if __name__ == "__main__":
    main()
