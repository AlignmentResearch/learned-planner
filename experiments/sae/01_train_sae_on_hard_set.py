from pathlib import Path

from farconf import update_fns_to_cli
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.train_sae import train_local
from learned_planner.interp.train_sae import TrainSAEConfig
from learned_planner.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
for lr in [5e-5]:
    for k in [4, 6, 8, 10, 12]:
        # for expansion_factor in [16, 32]:
        # for normalize_activations in ["none", "layer_norm"]:
        for l1_coefficient in [1e-10, 1e-4]:

            def update(config: WandbCommandConfig) -> WandbCommandConfig:
                config.cmd = check_cast(TrainSAEConfig, config.cmd)
                config.cmd.lr = lr
                config.cmd.device = "cpu"
                config.cmd.activation_fn = "topk"
                config.cmd.topk = k
                config.cmd.expansion_factor = 16
                config.cmd.normalize_activations = "layer_norm"
                config.cmd.l1_coefficient = l1_coefficient
                config.cmd.n_checkpoints = 5
                config.cmd.cached_activations_path = [
                    "/training/activations_dataset/train_unfiltered/0_think_step/",
                    "/training/activations_dataset/train_medium/0_think_step/",
                    "/training/activations_dataset/hard/0_think_step/",
                ]
                config.cmd.training_tokens = int(6e8)
                config.cmd.eval_every_n_wandb_logs = 500
                config.cmd.hook_layer = 1
                config.cmd.hook_name = f"features_extractor.cell_list.{config.cmd.hook_layer}.hook_h"

                return config

            cli, _ = update_fns_to_cli(train_local, update)
            clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "learned_planner", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="b12433f-main",
            CPU=4,
            MEMORY="40G",
            GPU=1,
            PRIORITY="normal-batch",
            parallel=False,
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="lp_sae",
    )
